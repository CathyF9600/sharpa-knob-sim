"""Generate real CPU/PyBullet diagnostic rollouts and replay-render their states.

NOT the Isaac task, an ICL evaluation, or a calibrated digital twin. Uses Bullet
position servos/convex collision hulls, not Isaac actuators/armature. Never drives
the rotor except for a passive brake. All plotted forces come from contacts.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pybullet as p
import trimesh
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from dexx.deploy_config import ARM_BASE_POS, TABLE_SURFACE_Z
from dexx.tasks.sharpa_contact_adapt.knob_spec import FINGERS, HAND_JOINTS, KnobGeometry

DT = 1 / 960
CONTROL_DT = 1 / 30


def prepare_urdfs(folder, g):
    source = ROOT / "assets/generated/fr3_with_right_sharpa_wave.urdf"
    tree = ET.parse(source)
    for link in tree.getroot().findall("link"):
        for visual in link.findall("visual"):
            link.remove(visual)
        for i, collision in enumerate(link.findall("collision")):
            geom = collision.find("geometry/mesh")
            if geom is not None:
                mesh = trimesh.load((source.parent / geom.get("filename")).resolve(), force="mesh")
                dest = folder / f'{link.get("name")}_{i}.obj'
                mesh.export(dest)
                geom.set("filename", str(dest))
            visual = copy.deepcopy(collision)
            visual.tag = "visual"
            material = ET.SubElement(visual, "material", name=f'preview_{link.get("name")}_{i}')
            rgba = "0.94 0.48 0.16 1" if "elastomer" in link.get("name") else "0.68 0.73 0.79 1"
            ET.SubElement(material, "color", rgba=rgba)
            link.append(visual)
    robot_path = folder / "preview_robot.urdf"
    tree.write(robot_path)
    z = g.rotor_center_z_m
    ix = g.rotor_mass_kg*(3*g.rotor_radius_m**2+g.rotor_height_m**2)/12
    knob_path = folder / "preview_knob.urdf"
    knob_path.write_text(f'''<robot name="passive_knob">
<link name="base"><inertial><mass value="0.2"/><inertia ixx="0.001" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/></inertial>
<visual><origin xyz="0 0 {g.flange_height_m/2}"/><geometry><cylinder radius="{g.flange_radius_m}" length="{g.flange_height_m}"/></geometry><material name="base"><color rgba="0.23 0.29 0.38 1"/></material></visual>
<visual><origin xyz="0 0 {g.flange_height_m+g.pedestal_height_m/2}"/><geometry><cylinder radius="{g.pedestal_radius_m}" length="{g.pedestal_height_m}"/></geometry><material name="base"/></visual>
<collision><origin xyz="0 0 {g.flange_height_m/2}"/><geometry><cylinder radius="{g.flange_radius_m}" length="{g.flange_height_m}"/></geometry></collision>
<collision><origin xyz="0 0 {g.flange_height_m+g.pedestal_height_m/2}"/><geometry><cylinder radius="{g.pedestal_radius_m}" length="{g.pedestal_height_m}"/></geometry></collision></link>
<link name="rotor"><inertial><mass value="{g.rotor_mass_kg}"/><inertia ixx="{ix}" iyy="{ix}" izz="{g.rotor_inertia_z}" ixy="0" ixz="0" iyz="0"/></inertial>
<visual><geometry><cylinder radius="{g.rotor_radius_m}" length="{g.rotor_height_m}"/></geometry><material name="blue"><color rgba="0.05 0.58 0.84 1"/></material></visual>
<visual><origin xyz="0.012 0 {g.rotor_height_m/2+.0003}"/><geometry><box size="0.020 0.002 0.0006"/></geometry><material name="mark"><color rgba="1 0.85 0.1 1"/></material></visual>
<collision><geometry><cylinder radius="{g.rotor_radius_m}" length="{g.rotor_height_m}"/></geometry></collision></link>
<joint name="shaft" type="continuous"><parent link="base"/><child link="rotor"/><origin xyz="0 0 {z}"/><axis xyz="0 0 1"/><limit effort="1" velocity="100"/><dynamics damping="0" friction="0"/></joint></robot>''')
    return robot_path, knob_path


class PreviewSim:
    def __init__(self, assets, pose, g, brake, brake_model="capped_explicit", self_collision=False, physics_dt=DT, solver_iterations=150):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        self.physics_dt = physics_dt
        p.setTimeStep(physics_dt)
        p.setPhysicsEngineParameter(numSolverIterations=solver_iterations, deterministicOverlappingPairs=1)
        self.robot = p.loadURDF(str(assets[0]), ARM_BASE_POS, useFixedBase=True,
                                flags=p.URDF_USE_INERTIA_FROM_FILE | (p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT if self_collision else 0))
        self.knob = p.loadURDF(str(assets[1]), pose["fixture_position_m"], useFixedBase=True,
                               flags=p.URDF_USE_INERTIA_FROM_FILE)
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[.75, 1.2, .015])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[.75, 1.2, .015], rgbaColor=[.86, .84, .79, 1])
        self.table = p.createMultiBody(0, col, vis, basePosition=[.1, 0, TABLE_SURFACE_Z-.015])
        self.ids, self.names, self.limits, self.efforts, self.links = [], [], [], [], {}
        for i in range(p.getNumJoints(self.robot)):
            info = p.getJointInfo(self.robot, i)
            self.links[info[12].decode()] = i
            p.changeDynamics(self.robot, i, lateralFriction=.8, restitution=0)
            if info[2] != p.JOINT_FIXED:
                name = info[1].decode()
                self.ids.append(i)
                self.names.append(name)
                self.limits.append([info[8], info[9]])
                self.efforts.append(min(info[10], .25) if name.startswith("right_") else info[10])
                p.resetJointState(self.robot, i, pose["joint_positions_rad"][name])
        self.self_collision = self_collision
        if self_collision:
            # Match the repository's named mechanical-adjacency exclusions.
            import ast
            cfg = ast.parse((ROOT/"src/dexx/tasks/franka_sharpa/franka_sharpa_env_cfg.py").read_text())
            pairs = next(ast.literal_eval(n.value) for n in cfg.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SELF_COLLISION_FILTER_PAIRS" for t in n.targets))
            for a, b in pairs:
                p.setCollisionFilterPair(self.robot, self.robot, self.links['right_'+a], self.links['right_'+b], 0)
            # Fixed links belong to the same body; exclude body-adjacent pairs.
            infos = {i:p.getJointInfo(self.robot,i) for i in range(p.getNumJoints(self.robot))}
            for i in infos:
                j, movable = i, 0
                while j >= 0:
                    movable += int(infos[j][2] != p.JOINT_FIXED)
                    parent = infos[j][16]
                    if movable <= 1:p.setCollisionFilterPair(self.robot,self.robot,i,parent,0)
                    j = parent
                    if movable > 1:break
        self.limits = np.array(self.limits)
        self.initial = np.array([pose["joint_positions_rad"][n] for n in self.names])
        self.hand_idx = [self.names.index(n) for n in HAND_JOINTS]
        self.tip_ids = [self.links[f"right_{f}_elastomer"] for f in FINGERS]
        self.brake, self.g, self.brake_model = brake, g, brake_model
        p.changeDynamics(self.knob, 0, lateralFriction=.8, restitution=0,
                         linearDamping=0, angularDamping=0)
        p.setJointMotorControl2(self.knob, 0, p.VELOCITY_CONTROL, force=0)
        self.target = self.initial.copy()

    def fit_pregrasp(self):
        """CPU-specific collision-aware fit; does not alter the Isaac pregrasp."""
        from scipy.optimize import least_squares
        active = [i for i, name in enumerate(self.names) if name.startswith("fr3_") or name in HAND_JOINTS[:13]]
        lo, hi = self.limits[active].T
        lo, hi = lo+.005, hi-.005
        x0 = np.clip(self.initial[active], lo+1e-6, hi-1e-6)
        def residual(x):
            q = self.initial.copy()
            q[active] = x
            for j, value in zip(self.ids, q):
                p.resetJointState(self.robot, j, value)
            distances = np.full(p.getNumJoints(self.robot), .04)
            for c in p.getClosestPoints(self.robot, self.knob, .04, linkIndexB=0):
                if c[3] >= 0:
                    distances[c[3]] = min(distances[c[3]], c[8])
            tip_errors = distances[self.tip_ids[:3]]-.003
            # Collision clearance on non-tip links and table; true convex queries.
            collision = 20*np.maximum(.002-distances, 0)
            return np.r_[5*tip_errors, collision, .003*(x-x0)]
        fit = least_squares(residual, x0, bounds=(lo, hi), max_nfev=200, diff_step=1e-4)
        residual(fit.x)
        q = self.initial.copy()
        q[active] = fit.x
        result = dict(zip(self.names, q.tolist()))
        print("CPU pregrasp fit residual:", float(np.linalg.norm(residual(fit.x))), flush=True)
        return result

    def snapshot(self):
        state = np.array([s[:2] for s in p.getJointStates(self.robot, self.ids)])
        angle, speed = p.getJointState(self.knob, 0)[:2]
        forces = np.zeros((5, 3))
        all_force = 0.0
        contacts = []
        for c in p.getContactPoints(self.robot, self.knob):
            if c[4] != 0:
                continue
            all_force += c[9]
            if c[3] in self.tip_ids:
                forces[self.tip_ids.index(c[3])] += np.array(c[7])*c[9]
            contacts.append({"robot_link": c[3], "position_w_m": c[5], "normal_force_n": c[9]})
        return {"q_rad": state[:, 0].tolist(), "dq_rad_s": state[:, 1].tolist(),
                "angle_rad": angle, "angular_velocity_rad_s": speed,
                "tip_normal_force_world_n": forces.tolist(),
                "all_robot_rotor_normal_force_n": all_force, "contacts": contacts,
                **({"max_self_penetration_m": max([0.0]+[-c[8] for c in p.getContactPoints(self.robot,self.robot)]),
                    "max_moving_link_table_penetration_m": max([0.0]+[-c[8] for c in p.getContactPoints(self.robot,self.table) if c[3]>0])} if self.self_collision else {})}

    def step(self, target):
        self.target = np.clip(target, self.limits[:, 0]+.005, self.limits[:, 1]-.005)
        p.setJointMotorControlArray(self.robot, self.ids, p.POSITION_CONTROL,
                                    targetPositions=self.target, forces=self.efforts,
                                    positionGains=[.04]*len(self.ids), velocityGains=[1]*len(self.ids))
        for _ in range(round(CONTROL_DT/self.physics_dt)):
            velocity = p.getJointState(self.knob, 0)[1]
            if self.brake_model == "solver_friction":
                # A zero-speed, torque-limited constraint resists motion, including
                # at rest. It never commands the goal angle or a nonzero speed.
                p.setJointMotorControl2(self.knob, 0, p.VELOCITY_CONTROL,
                                       targetVelocity=0, force=self.brake)
                p.applyExternalTorque(self.knob, 0, [0, 0, -.0003*velocity], p.WORLD_FRAME)
            else:
                torque = -np.sign(velocity)*min(self.brake+.0003*abs(velocity), self.g.rotor_inertia_z*abs(velocity)/self.physics_dt)
                p.setJointMotorControl2(self.knob, 0, p.TORQUE_CONTROL, force=torque)
            p.stepSimulation()
        result = self.snapshot()
        if not np.isfinite(np.array(result["q_rad"])).all():
            raise RuntimeError("Non-finite robot state")
        return result


def command(sim, t, moving):
    target = sim.initial.copy()
    if not moving:
        return target
    close = np.clip((t-.5)/1.5, 0, 1)
    sweep = np.clip((t-2.5)/2, 0, 1)
    close *= moving
    sweep *= moving
    offsets = {"right_thumb_MCP_FE": .08*close, "right_thumb_IP": .08*close,
               "right_index_MCP_FE": .06*close, "right_index_PIP": .08*close,
               "right_middle_PIP": -.08*close,
               "right_thumb_CMC_AA": -.12*sweep,
               "right_index_MCP_AA": -.20*sweep,
               "right_middle_MCP_AA": -.15*sweep}
    for name, delta in offsets.items():
        target[sim.names.index(name)] += delta
    return target


def font(size):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def save_gif(path, frames):
    # One shared palette without dithering keeps the plots/text legible.
    palette = frames[len(frames)//2].quantize(colors=256)
    indexed = [f.quantize(palette=palette, dither=0) for f in frames]
    indexed[0].save(path, save_all=True, append_images=indexed[1:], duration=100,
                    loop=0, disposal=2, optimize=False)


def render(sim, records, label, goal_angle_deg=60, plot_limits=None):
    view = p.computeViewMatrixFromYawPitchRoll([.48, -.15, .49], .31, 35, -30, 0, 2)
    proj = p.computeProjectionMatrixFOV(45, 1.0, .015, 3)
    frames = []
    angles = np.rad2deg([r["after"]["angle_rad"] for r in records])
    forces = [sum(np.linalg.norm(r["after"]["tip_normal_force_world_n"], axis=1)) for r in records]
    for step in range(2, len(records), 3):
        state = records[step]["after"]
        for i, q in zip(sim.ids, state["q_rad"]):
            p.resetJointState(sim.robot, i, q)
        p.resetJointState(sim.knob, 0, state["angle_rad"])
        rgba = np.asarray(p.getCameraImage(480, 480, view, proj, renderer=p.ER_TINY_RENDERER,
                                          shadow=1, lightDirection=[-1, -1, 3])[2], dtype=np.uint8)
        frame = Image.new("RGB", (480, 730), "#111827")
        frame.paste(Image.fromarray(rgba[:, :, :3]), (0, 80))
        d = ImageDraw.Draw(frame)
        d.text((16, 9), label, font=font(19), fill="white")
        d.text((16, 37), "CPU physics / PyBullet diagnostic", font=font(14), fill="#94a3b8")
        d.text((16, 58), f't={records[step]["time_s"]:.2f}s   angle={angles[step]:+.2f} deg', font=font(14), fill="#67e8f9")
        d.ellipse((390, 102, 466, 178), fill="#1e293b", outline="#94a3b8", width=2)
        theta = state["angle_rad"]
        d.line((428, 140, 428+32*np.cos(theta), 140-32*np.sin(theta)), fill="#facc15", width=4)
        d.text((364, 182), "angle readout", font=font(12), fill="#111827")
        q0 = np.array(records[0]["before"]["q_rad"])[sim.hand_idx]
        actual_delta = np.array(state["q_rad"])[sim.hand_idx]-q0
        target_delta = np.array(records[step]["issued_target_rad"])[sim.hand_idx]-q0
        d.rectangle((0, 520, 480, 560), fill="#1e293b")
        d.text((16, 523), f'Hand RMS delta: target {np.linalg.norm(target_delta)/np.sqrt(22):.3f} / actual {np.linalg.norm(actual_delta)/np.sqrt(22):.3f} rad', font=font(12), fill="white")
        d.text((16, 542), f'Scripted probe; goal {goal_angle_deg:+g} deg (not a goal policy)', font=font(12), fill="#fbbf24")
        for row, values, title, color in [(0, angles, "Measured rotor angle (deg)", "#67e8f9"),
                                          (1, forces, "Measured fingertip normal force (N)", "#fb923c")]:
            y = 570+row*75
            lo, hi = (min(-30, min(values)), max(30, max(values))) if row == 0 else (0, max(30, max(values)))
            if plot_limits is not None:
                lo, hi = plot_limits[row]
            span = max(hi-lo, .01)
            d.text((16, y), f"{title}: {values[step]:.2f}", font=font(13), fill=color)
            d.text((340, y+17), f"[{lo:.0f}, {hi:.0f}]", font=font(10), fill="#94a3b8")
            points = [(16+i/(len(values)-1)*446, y+58-(v-lo)/span*34) for i, v in enumerate(values)]
            d.line(points, fill="#475569", width=1)
            if step:
                d.line(points[:step+1], fill=color, width=2)
            d.ellipse((points[step][0]-3, points[step][1]-3, points[step][0]+3, points[step][1]+3), fill=color)
        frames.append(frame)
    return frames


def save_comparison(folder, movies):
    columns = 2 if len(movies) == 4 else len(movies)
    height = 730*((len(movies)+columns-1)//columns)
    comparison = []
    for frames in zip(*movies):
        canvas = Image.new("RGB", (480*columns, height), "#111827")
        for i, frame in enumerate(frames):
            canvas.paste(frame, (480*(i % columns), 730*(i//columns)))
        comparison.append(canvas)
    save_gif(folder/"comparison.gif", comparison)
    comparison[-1].save(folder/"comparison.png")


def render_saved(folder):
    """Replay recorded joint states only: no new simulation or force synthesis."""
    pose = json.loads((folder/"cpu_pregrasp.json").read_text())
    g = KnobGeometry(**json.loads((folder/"geometry.json").read_text()))
    assets = (folder/"assets/preview_robot.urdf", folder/"assets/preview_knob.urdf")
    movies = []
    p.connect(p.DIRECT)
    try:
        paths = sorted(folder.glob("[0-9][0-9].jsonl"))
        all_rows = [[json.loads(line) for line in path.open()] for path in paths]
        angle_bound = max(30, max(abs(np.rad2deg(r["after"]["angle_rad"])) for rows in all_rows for r in rows[1:]))
        force_bound = max(30, max(np.linalg.norm(r["after"]["tip_normal_force_world_n"], axis=1).sum() for rows in all_rows for r in rows[1:]))
        plot_limits = [(-angle_bound, angle_bound), (0, force_bound)]
        for path, rows in zip(paths, all_rows):
            name = path.stem
            header, rows = rows[0], rows[1:]
            sim = PreviewSim(assets, pose, g, header["brake_nm"], header.get("brake_model", "capped_explicit"))
            if sim.names != header["joint_names"]:
                raise RuntimeError("Replay joint order differs from saved data")
            frames = render(sim, rows, header["label"], header.get("goal_angle_deg", 60), plot_limits)
            save_gif(folder/f"{name}.gif", frames)
            frames[-1].save(folder/f"{name}.png")
            movies.append(frames)
        save_comparison(folder, movies)
    finally:
        p.disconnect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--replay", action="store_true", help="Only re-render existing recorded data")
    parser.add_argument("--suite", choices=["actions", "resistance"], default="actions")
    parser.add_argument("--brake-model", choices=["capped_explicit", "solver_friction"], default="capped_explicit")
    parser.add_argument("--goal-angle-deg", type=float, default=120)
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.replay:
        render_saved(args.out)
        return
    args.out.mkdir(parents=True, exist_ok=False)
    assets_dir = args.out / "assets"
    assets_dir.mkdir()
    g = KnobGeometry(**json.loads((ROOT/"assets/tasks/knob/geometry.json").read_text()))
    (args.out/"geometry.json").write_text((ROOT/"assets/tasks/knob/geometry.json").read_text())
    pose = json.loads((ROOT/"assets/tasks/knob/pregrasp.json").read_text())
    assets = prepare_urdfs(assets_dir, g)
    p.connect(p.DIRECT)
    summaries = []
    try:
        fitting_sim = PreviewSim(assets, pose, g, .002)
        pose["joint_positions_rad"] = fitting_sim.fit_pregrasp()
        pose["status"] = "pybullet_only_collision_fit_not_isaac_pose"
        (args.out/"cpu_pregrasp.json").write_text(json.dumps(pose, indent=2)+"\n")
        cases = [("01 Hold / zero action", 0, .002),
                                     ("02 Gentle probe", 1, .002),
                                     ("03 Stronger close + sweep", 3, .002)]
        if args.suite == "resistance":
            cases = [(f"{i+1:02} Brake {brake:.3f} N m", 3, brake)
                     for i, brake in enumerate([0, .005, .03, .10])]
        for label, moving, brake in cases:
            sim = PreviewSim(assets, pose, g, brake, args.brake_model)
            rows = []
            before = sim.snapshot()
            for step in range(180):
                target = command(sim, (step+1)*CONTROL_DT, moving)
                after = sim.step(target)
                rows.append({"time_s": (step+1)*CONTROL_DT, "duration_s": CONTROL_DT,
                             "before": before, "issued_target_rad": sim.target.tolist(), "after": after})
                before = after
            name = label[:2]
            header = {"type": "header", "engine": "PyBullet 3.2.7", "purpose": "diagnostic_not_training",
                      "label": label, "joint_names": sim.names, "hand_joint_names": list(HAND_JOINTS),
                      "brake_nm": brake, "damping_nms": .0003, "contact_friction": .8,
                      "physics_dt_s": DT, "control_dt_s": CONTROL_DT,
                      "differences_from_isaac": ["Bullet position servos; no Isaac armature",
                          "convex mesh hulls", "robot self collisions disabled", "normal forces in world frame"],
                      "scripted_policy": True, "parameter_updates": False}
            header["script_action_scale"] = moving
            header["brake_model"] = args.brake_model
            header["goal_angle_deg"] = args.goal_angle_deg
            header["suite"] = args.suite
            header["motor_force_limits_nm"] = sim.efforts
            header["motor_position_gain"] = .04
            header["motor_velocity_gain"] = 1.0
            header["condition_id"] = int(name)-1 if args.suite == "resistance" else 0
            header["trial_id"] = 0 if args.suite == "resistance" else int(name)-1
            with (args.out/f"{name}.jsonl").open("x") as file:
                file.write(json.dumps(header)+"\n")
                for row in rows:
                    file.write(json.dumps(row, allow_nan=False)+"\n")
            q = np.array([r["after"]["q_rad"] for r in rows])
            summary = {"label": label, "frames": len(rows), "brake_nm": brake,
                       "final_angle_deg": float(np.rad2deg(rows[-1]["after"]["angle_rad"])),
                       "max_tip_force_n": max(float(np.linalg.norm(r["after"]["tip_normal_force_world_n"], axis=1).max()) for r in rows),
                       "max_all_rotor_force_n": max(r["after"]["all_robot_rotor_normal_force_n"] for r in rows),
                       "max_joint_limit_violation_rad": float(max(0, (sim.limits[:, 0]-q).max(), (q-sim.limits[:, 1]).max())),
                       "zero_action_max_angle_deg": max(abs(np.rad2deg(r["after"]["angle_rad"])) for r in rows) if not moving else None}
            good = [abs(np.rad2deg(r["after"]["angle_rad"])-args.goal_angle_deg) <= 5 and abs(r["after"]["angular_velocity_rad_s"]) <= .1 for r in rows]
            summary["goal_angle_deg"] = args.goal_angle_deg
            summary["success"] = any(all(good[k:k+15]) for k in range(len(good)-14))
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
        (args.out/"summary.json").write_text(json.dumps(summaries, indent=2)+"\n")
    finally:
        p.disconnect()
    if args.render:
        render_saved(args.out)


if __name__ == "__main__":
    main()
