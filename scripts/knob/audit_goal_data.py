"""Audit measured trajectories for demonstration quality, not just terminal success.

Passing these engineering thresholds does NOT prove optimality, calibrated force,
or Isaac/real-robot validity. Raw records are never changed by this audit.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def audit(path):
    data = [json.loads(line) for line in path.open()]
    header, rows = data[0], data[1:]
    if not rows:
        return {"file":path.name,"quality_gate_pass":False,"rejection_reasons":["empty"],"training_status":"diagnostic_not_expert"}
    goal = header["goal_angle_deg"]
    origin = header.get("reference_angle_rad", 0.0)
    angle = np.rad2deg([rows[0]["before"]["angle_rad"]]+[r["after"]["angle_rad"] for r in rows])-np.rad2deg(origin)
    speed = np.array([r["after"]["angular_velocity_rad_s"] for r in rows])
    goals = (np.abs(angle[1:]-goal) <= .25) & (np.abs(speed) <= .05)
    count, first_one_second = 0, None
    dt = header["control_dt_s"]
    for row, good in zip(rows, goals):
        count = count+1 if good else 0
        if count*dt >= 1-1e-8 and first_one_second is None:
            first_one_second = row["time_s"]
    force = np.array([r["after"]["tip_normal_force_world_n"] for r in rows])
    overshoot = float(max(0, angle.max()-goal))
    reverse = float(max(0, -angle.min()))
    travel = float(np.abs(np.diff(angle)).sum())
    reasons = []
    numeric = np.r_[angle, speed, force.ravel(), np.array([r["after"]["q_rad"] for r in rows]).ravel()]
    if not np.isfinite(numeric).all():
        reasons.append("non_finite_state")
    max_ik = max(r.get("ik_position_error_m", 0) for r in rows)
    if max_ik > .002:
        reasons.append("ik_position_error_above_2_mm")
    max_self_penetration = max(r["after"].get("max_self_penetration_m", 0) for r in rows)
    max_table_penetration = max(r["after"].get("max_moving_link_table_penetration_m", 0) for r in rows)
    if max_self_penetration > .002:
        reasons.append("self_penetration_above_2_mm")
    if max_table_penetration > .002:
        reasons.append("moving_link_table_penetration_above_2_mm")
    if header.get("schema") == "cpu_regrasp_goal_v2" and not header.get("self_collision_enabled", False):
        reasons.append("self_collision_not_enabled")
    violation = None
    if "joint_limits_rad" in header:
        limits = np.asarray(header["joint_limits_rad"])
        q = np.asarray([r["after"]["q_rad"] for r in rows])
        violation = float(max(0, (limits[:, 0]-q).max(), (q-limits[:, 1]).max()))
        if violation > .01:
            reasons.append("joint_limit_violation_above_0_01_rad")
    if np.linalg.norm(force, axis=-1).max() > 50:
        reasons.append("tip_force_above_50_N_uncalibrated_screen")
    if overshoot > 1:
        reasons.append("overshoot_above_1_deg")
    if reverse > 1:
        reasons.append("initial_reverse_above_1_deg")
    if travel > goal+5:
        reasons.append("total_angular_travel_above_goal_plus_5_deg")
    if count*dt < 3-1e-8:
        reasons.append("final_stable_hold_below_3_s")
    for k, row in enumerate(rows):
        if k and row["before"] != rows[k-1]["after"]:
            reasons.append("discontinuous_record")
            break
    return {"file": path.name, "brake_nm": header["brake_nm"],
            "goal_angle_deg": goal, "max_ik_position_error_m": max_ik,
            "max_joint_limit_violation_rad": violation,
            "max_self_penetration_m": max_self_penetration, "max_table_penetration_m": max_table_penetration,
            "final_angle_deg": float(angle[-1]), "max_overshoot_deg": overshoot,
            "initial_reverse_excursion_deg": reverse, "angular_travel_deg": travel,
            "max_tip_force_n": float(np.linalg.norm(force, axis=-1).max()),
            "max_all_robot_rotor_normal_force_n": max(r["after"].get("all_robot_rotor_normal_force_n",0) for r in rows),
            "first_1s_completion_s": first_one_second, "duration_s": rows[-1]["time_s"],
            "final_stable_hold_s": count*dt, "quality_gate_pass": not reasons,
            "rejection_reasons": reasons,
            "training_status": "cpu_candidate_requires_further_validation" if not reasons else "diagnostic_not_expert"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", type=Path, nargs="+")
    args = parser.parse_args()
    for folder in args.folders:
        source = folder/"accepted" if (folder/"accepted").is_dir() else folder
        paths = sorted(set(source.glob("[0-9][0-9].jsonl")) | set(source.glob("c[0-3]_*.jsonl")))
        result = {"thresholds": {"max_overshoot_deg": 1, "max_initial_reverse_deg": 1,
                                  "max_excess_angular_travel_deg": 5, "final_hold_s": 3,
                                  "final_error_deg": .25, "final_speed_rad_s": .05},
                  "optimality_claim": False,
                  "episodes": [audit(path) for path in paths]}
        (folder/"quality_audit.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps({"folder": str(folder), **result}, indent=2))


if __name__ == "__main__":
    main()
