"""Contact-driven variable-angle CPU rollouts, with arm-assisted angle feedback.

This is a scripted privileged-state controller, NOT a trained/ICL policy or the
fixed-arm Isaac environment. The rotor has only the passive shaft brake.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from cpu_rollouts import ROOT, PreviewSim, KnobGeometry, command, prepare_urdfs, p, CONTROL_DT, DT
from dexx.deploy_config import ARM_BASE_POS
from dexx.tasks.sharpa_contact_adapt.geometry_tools import origin_transform, axis_rotation


class GoalController:
    def __init__(self, sim, pose, g, mode="baseline", goal_deg=180):
        self.sim = sim
        self.arm = [i for i, n in enumerate(sim.names) if n.startswith("fr3_")]
        self.arm_names = [sim.names[i] for i in self.arm]
        joints = ET.parse(ROOT/"assets/generated/fr3_with_right_sharpa_wave.urdf").getroot().findall("joint")
        by_child = {j.find("child").get("link"): j for j in joints}
        chain, link = [], "right_hand_C_MC"
        while link in by_child:
            joint = by_child[link]
            axis = joint.find("axis")
            chain.append((joint.get("name"), origin_transform(joint.find("origin")),
                          np.fromstring(axis.get("xyz"), sep=" ") if axis is not None else None))
            link = joint.find("parent").get("link")
        self.chain = chain[::-1]
        self.previous = sim.initial[self.arm].copy()
        self.initial_frame = self.fk(self.previous)
        self.center = np.array(pose["fixture_position_m"])+[0, 0, g.rotor_center_z_m]
        self.phase = 0.0
        self.goal = np.deg2rad(goal_deg)
        self.ik_position_error = 0.0
        self.mode = mode
        self.filtered_speed = 0.0
        self.orbit_speed = 0.0
        self.grip_scale = 1.0 if mode == "smooth" else 3.0
        self.grip_engaged = False

    def fk(self, q):
        frame = np.eye(4)
        frame[:3, 3] = ARM_BASE_POS
        for name, origin, axis in self.chain:
            frame = frame @ origin
            if name in self.arm_names:
                rot = np.eye(4)
                rot[:3, :3] = axis_rotation(axis, q[self.arm_names.index(name)])
                frame = frame @ rot
        return frame

    def action(self, t, measured_angle, measured_speed=0.0):
        # Close during first two seconds, then keep those finger targets. Only
        # the arm orbit is adapted by measured angle error; no condition label.
        if self.mode == "smooth":
            self.grip_engaged |= measured_angle > np.deg2rad(10)
            if self.grip_engaged:
                self.grip_scale = min(2.0, self.grip_scale+.35*CONTROL_DT)
        target = command(self.sim, min(t, 2.0), self.grip_scale)
        if t <= 2.0:
            return target
        if self.mode == "smooth":
            self.filtered_speed += .15*(measured_speed-self.filtered_speed)
            requested = np.clip(.5*(self.goal-measured_angle)-.8*self.filtered_speed, -.15, .15)
            self.orbit_speed += np.clip(requested-self.orbit_speed, -.12*CONTROL_DT, .12*CONTROL_DT)
            self.phase += self.orbit_speed*CONTROL_DT
        else:
            self.phase += np.clip(1.5*(self.goal-measured_angle), -.25, .25)*CONTROL_DT
        rot = axis_rotation([0, 0, 1], self.phase)
        position = self.center + rot @ (self.initial_frame[:3, 3]-self.center)
        orientation = rot @ self.initial_frame[:3, :3]

        def residual(q):
            frame = self.fk(q)
            return np.r_[frame[:3, 3]-position,
                         .1*Rotation.from_matrix(orientation.T @ frame[:3, :3]).as_rotvec(),
                         .0001*(q-self.previous)]

        fit = least_squares(residual, self.previous,
                            bounds=(self.sim.limits[self.arm, 0]+.005,
                                    self.sim.limits[self.arm, 1]-.005), max_nfev=50)
        self.previous = fit.x
        self.ik_position_error = float(np.linalg.norm(self.fk(fit.x)[:3, 3]-position))
        target[self.arm] = fit.x
        return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--goal-degrees", type=float, default=180)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--controller", choices=["baseline", "smooth"], default="baseline")
    parser.add_argument("--hold-seconds", type=float, default=1)
    parser.add_argument("--brakes", type=float, nargs="+", default=[0, .005, .03, .10])
    args = parser.parse_args()
    if args.seconds < 15:
        parser.error("Use at least 15 seconds to include settling")
    if not np.isfinite(args.seconds) or not np.isfinite(args.hold_seconds) or not 0 < args.hold_seconds < args.seconds:
        parser.error("Require finite duration and 0 < hold-seconds < seconds")
    if not all(np.isfinite(b) and b >= 0 for b in args.brakes):
        parser.error("Brakes must be finite and nonnegative")
    folder = args.out.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    (folder/"assets").mkdir()
    source = ROOT/"assets/tasks/knob"
    g = KnobGeometry(**json.loads((source/"geometry.json").read_text()))
    (folder/"geometry.json").write_text((source/"geometry.json").read_text())
    pose = json.loads((source/"pregrasp.json").read_text())
    assets = prepare_urdfs(folder/"assets", g)
    p.connect(p.DIRECT)
    summaries = []
    try:
        sim = PreviewSim(assets, pose, g, .002, "solver_friction")
        pose["joint_positions_rad"] = sim.fit_pregrasp()
        pose["status"] = "pybullet_only_collision_fit_arm_assisted_goal_control"
        (folder/"cpu_pregrasp.json").write_text(json.dumps(pose, indent=2)+"\n")
        for index, brake in enumerate(args.brakes, 1):
            sim = PreviewSim(assets, pose, g, brake, "solver_friction")
            controller = GoalController(sim, pose, g, args.controller, args.goal_degrees)
            header = {"type": "header", "schema": "cpu_arm_assisted_goal_v1",
                      "engine": "PyBullet 3.2.7", "label": f"{index:02} Brake {brake:.3f} N m",
                      "joint_names": sim.names, "brake_nm": brake, "brake_model": "solver_friction",
                      "goal_angle_deg": args.goal_degrees, "angle_tolerance_deg": .25,
                      "speed_tolerance_rad_s": .05, "required_hold_s": args.hold_seconds,
                      "physics_dt_s": DT, "control_dt_s": CONTROL_DT,
                      "contact_friction": .8, "damping_nms": .0003,
                      "controller": "identical angle-feedback wrist-orbit controller for all conditions",
                      "arm_control_enabled": True, "rotor_position_drive": False,
                      "uses_physical_condition_as_input": False, "trained_policy": False,
                      "condition_id": index-1, "trial_id": 0,
                      "episode_termination": "first continuous required_hold_s within tolerance and speed bound",
                      "controller_variant": args.controller,
                      "controller_gains": ({"kp": .5, "kd": .8, "speed_filter_alpha": .15,
                                            "max_orbit_speed_rad_s": .15, "max_orbit_accel_rad_s2": .12,
                                            "initial_grip_scale": 1.0, "final_grip_scale": 2.0,
                                            "grip_ramp_per_s": .35, "grip_ramp_trigger_deg": 10} if args.controller == "smooth" else
                                           {"kp": 1.5, "max_orbit_speed_rad_s": .25, "grip_scale": 3}),
                      "motor_force_limits_nm": sim.efforts,
                      "motor_position_gain": .04, "motor_velocity_gain": 1,
                      "limitations": ["not Isaac", "not ICL", "uncalibrated contact forces",
                                      "no robot self collision", "no Isaac armature"]}
            rows, hold, first_hold = [], 0, None
            before = sim.snapshot()
            for step in range(round(args.seconds/CONTROL_DT)):
                t = (step+1)*CONTROL_DT
                after = sim.step(controller.action(t, before["angle_rad"], before["angular_velocity_rad_s"]))
                good = abs(after["angle_rad"]-controller.goal) <= np.deg2rad(.25) and abs(after["angular_velocity_rad_s"]) <= .05
                hold = hold+1 if good else 0
                if hold >= round(args.hold_seconds/CONTROL_DT) and first_hold is None:
                    first_hold = t
                rows.append({"time_s": t, "duration_s": CONTROL_DT, "before": before,
                             "issued_target_rad": sim.target.tolist(), "after": after,
                             "controller_wrist_orbit_rad": controller.phase,
                             "controller_grip_scale": controller.grip_scale,
                             "ik_position_error_m": controller.ik_position_error})
                before = after
                if step % 150 == 149:
                    print(f"{index:02}: t={t:.1f}, angle={np.rad2deg(after['angle_rad']):.3f}, stable_steps={hold}", flush=True)
                if hold >= round(args.hold_seconds/CONTROL_DT):
                    break
            with (folder/f"{index:02}.jsonl").open("x") as file:
                file.write(json.dumps(header)+"\n")
                for row in rows:
                    file.write(json.dumps(row, allow_nan=False)+"\n")
            q = np.array([r["after"]["q_rad"] for r in rows])
            angles = np.rad2deg([r["after"]["angle_rad"] for r in rows])
            summary = {"condition": index, "brake_nm": brake, "success": hold >= round(args.hold_seconds/CONTROL_DT),
                       "final_angle_deg": float(np.rad2deg(after["angle_rad"])),
                       "final_speed_rad_s": after["angular_velocity_rad_s"],
                       "first_stable_goal_time_s": first_hold,
                       "final_continuous_hold_s": hold*CONTROL_DT,
                       "max_ik_position_error_m": max(r["ik_position_error_m"] for r in rows),
                       "max_joint_limit_violation_rad": float(max(0, (sim.limits[:, 0]-q).max(), (q-sim.limits[:, 1]).max())),
                       "max_tip_force_n": max(float(np.linalg.norm(r["after"]["tip_normal_force_world_n"], axis=1).max()) for r in rows)}
            summary.update({"max_overshoot_deg": float(max(0, angles.max()-args.goal_degrees)),
                            "initial_reverse_excursion_deg": float(max(0, -angles.min())),
                            "angular_travel_deg": float(np.abs(np.diff(np.r_[0, angles])).sum()),
                            "controller_variant": args.controller})
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
        (folder/"summary.json").write_text(json.dumps(summaries, indent=2)+"\n")
        if not all(s["success"] for s in summaries):
            raise SystemExit("Not all conditions reached and held the goal; failures retained.")
    finally:
        p.disconnect()


if __name__ == "__main__":
    main()
