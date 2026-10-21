"""One figure with all four measured +120 degree goal outcomes and histories."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cpu_rollouts import PreviewSim, KnobGeometry, p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    folder = args.out.resolve()
    summary = json.loads((folder/"summary.json").read_text())
    if len(summary) != 4 or not all(s["success"] for s in summary):
        raise SystemExit("Refusing an all-success figure: not all four trials passed.")
    pose = json.loads((folder/"cpu_pregrasp.json").read_text())
    g = KnobGeometry(**json.loads((folder/"geometry.json").read_text()))
    assets = (folder/"assets/preview_robot.urdf", folder/"assets/preview_knob.urdf")
    fig = plt.figure(figsize=(16, 10), facecolor="white")
    grid = fig.add_gridspec(2, 4, height_ratios=[1.2, 1], hspace=.24, wspace=.27)
    angle_ax = fig.add_subplot(grid[1, :2])
    force_ax = fig.add_subplot(grid[1, 2:])
    colors = ["#64748b", "#0284c7", "#d97706", "#be123c"]
    view = p.computeViewMatrixFromYawPitchRoll([.48, -.15, .49], .32, 155, -30, 0, 2)
    projection = p.computeProjectionMatrixFOV(45, 1, .015, 3)
    p.connect(p.DIRECT)
    try:
        with (folder/"measurements.csv").open("w") as file:
            writer = csv.writer(file)
            writer.writerow(["condition", "brake_nm", "time_s", "angle_deg", "speed_rad_s", "tip_normal_force_sum_n"])
            for i, (result, color) in enumerate(zip(summary, colors), 1):
                data = [json.loads(line) for line in (folder/f"{i:02}.jsonl").open()]
                header, rows = data[0], data[1:]
                sim = PreviewSim(assets, pose, g, result["brake_nm"], "solver_friction")
                if sim.names != header["joint_names"]:
                    raise RuntimeError("Joint ordering mismatch")
                final = rows[-1]["after"]
                # Rendering only: restore recorded states, never integrate or
                # manufacture a new target-angle trajectory here.
                for joint, q in zip(sim.ids, final["q_rad"]):
                    p.resetJointState(sim.robot, joint, q)
                p.resetJointState(sim.knob, 0, final["angle_rad"])
                rgba = np.asarray(p.getCameraImage(600, 600, view, projection,
                                                   renderer=p.ER_TINY_RENDERER, shadow=1,
                                                   lightDirection=[-1, -1, 3])[2], dtype=np.uint8)
                ax = fig.add_subplot(grid[0, i-1])
                ax.imshow(rgba[:, :, :3]);ax.axis("off")
                ax.set_title(f"Brake {result['brake_nm']:.3f} N m\nMeasured {result['final_angle_deg']:.2f} deg", color=color, fontsize=14, fontweight="bold")
                ax.text(.5, -.035, f"Finished: {rows[-1]['time_s']:.2f} s | Hold: {result['final_continuous_hold_s']:.1f} s", ha="center", transform=ax.transAxes, fontsize=11)
                times = [r["time_s"] for r in rows]
                angles = np.rad2deg([r["after"]["angle_rad"] for r in rows])
                forces = [np.linalg.norm(r["after"]["tip_normal_force_world_n"], axis=1).sum() for r in rows]
                angle_ax.plot(times, angles, color=color, lw=1.8, label=f"{result['brake_nm']:.3f} N m")
                force_ax.plot(times, forces, color=color, lw=1.0, alpha=.85)
                for row, angle, force in zip(rows, angles, forces):
                    writer.writerow([i, result["brake_nm"], row["time_s"], angle,
                                     row["after"]["angular_velocity_rad_s"], force])
        angle_ax.axhline(120, color="#111827", ls="--", lw=1, label="Target +120 deg")
        angle_ax.axhspan(119.75, 120.25, alpha=.15, color="green")
        angle_ax.set_ylabel("Measured rotor angle (deg)")
        angle_ax.set_title("All four reach and hold the same target")
        angle_ax.legend(ncol=2, fontsize=10, loc="lower right")
        force_ax.set_title("Different contact-force histories")
        force_ax.set_ylabel("Sum of fingertip normal-force norms (N)")
        for ax in (angle_ax, force_ax):
            ax.set_xlabel("Time (s)");ax.grid(alpha=.2)
            ax.axvline(2, color="gray", lw=.7, ls=":")
        fig.suptitle("Different shaft resistance, same +120 deg target", fontsize=21, y=.985)
        fig.text(.5, .943, "Measured episode-end states | Identical feedback controller | Arm-assisted Sharpa Wave grasp", ha="center", fontsize=12)
        fig.text(.5, .02, "PyBullet CPU simulation, not an ICL policy. Contact-driven rotor. End episode: error <= 0.25 deg, speed <= 0.05 rad/s, hold >= 1 s.", ha="center", fontsize=10)
        fig.subplots_adjust(top=.9, bottom=.09, left=.055, right=.98)
        fig.savefig(folder/"all_120_degrees.png", dpi=150)
        fig.savefig(folder/"all_120_degrees.pdf")
        plt.close(fig)
    finally:
        p.disconnect()


if __name__ == "__main__":
    main()
