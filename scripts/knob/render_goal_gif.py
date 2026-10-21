"""Synchronized 2x2 GIF from recorded, contact-driven goal episodes only."""
import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from cpu_rollouts import PreviewSim, KnobGeometry, p, font


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--playback-speed", type=float, default=1)
    parser.add_argument("--marked", action="store_true")
    parser.add_argument("--show-arm", action="store_true")
    parser.add_argument("--yaw", type=float, default=-25)
    parser.add_argument("--elevation", type=float, default=30)
    parser.add_argument("--distance", type=float, default=.37)
    parser.add_argument("--fov", type=float, default=45)
    args = parser.parse_args()
    if not np.isfinite(args.playback_speed) or args.playback_speed <= 0:parser.error("Playback speed must be finite and positive")
    folder = args.out.resolve()
    summary = json.loads((folder/"summary.json").read_text())
    assert len(summary) == 4 and all(s["success"] for s in summary)
    datasets = [[json.loads(line) for line in (folder/f"{i:02}.jsonl").open()] for i in range(1, 5)]
    goal = datasets[0][0]["goal_angle_deg"]
    assert all(data[0]["goal_angle_deg"] == goal for data in datasets)
    force_plot_max = max(10, 1.1*max(np.linalg.norm(r["after"]["tip_normal_force_world_n"], axis=1).sum() for data in datasets for r in data[1:]))
    end_times = [rows[-1]["time_s"] for rows in datasets]
    timeline = np.arange(int(np.ceil(max(end_times)*10/args.playback_speed))+1)*args.playback_speed/10
    pose = json.loads((folder/"cpu_pregrasp.json").read_text())
    geometry = KnobGeometry(**json.loads((folder/"geometry.json").read_text()))
    asset_folder = folder/"assets" if (folder/"assets").exists() else folder.parent/"assets"
    # Cull off-camera arm visuals for this hand close-up; retain all joints and
    # collision geometry. Recorded states are replayed without simulation steps.
    tree = ET.parse(asset_folder/"preview_robot.urdf")
    for link in tree.getroot().findall("link"):
        if not args.show_arm and not link.get("name").startswith("right_"):
            for visual in link.findall("visual"):link.remove(visual)
    for mesh in tree.getroot().iter("mesh"):
        source = Path(mesh.get("filename"))
        if not source.is_absolute():source = asset_folder/source
        mesh.set("filename",os.path.relpath(source.resolve(),folder))
    robot = folder/"render_robot.urdf"
    tree.write(robot)
    assets = (robot, asset_folder/"preview_knob.urdf")
    colors = ["#94a3b8", "#38bdf8", "#fbbf24", "#fb7185"]
    width, height = 400, 590
    composites = [Image.new("RGB", (800, 1240), "#0f172a") for _ in timeline]
    target = np.array(pose["fixture_position_m"])+[0,0,geometry.rotor_center_z_m+.025]
    view = p.computeViewMatrixFromYawPitchRoll(target, args.distance, args.yaw, -args.elevation, 0, 2)
    camera_eye = np.linalg.inv(np.array(view).reshape(4,4,order="F"))[:3,3].tolist()
    projection = p.computeProjectionMatrixFOV(args.fov, 1, .015, 3)
    p.connect(p.DIRECT)
    try:
        for n, (data, color) in enumerate(zip(datasets, colors)):
            header, rows = data[0], data[1:]
            sim = PreviewSim(assets, pose, geometry, header["brake_nm"], "solver_friction")
            for name, link in sim.links.items():
                if not args.show_arm and not name.startswith("right_"):p.changeVisualShape(sim.robot,link,rgbaColor=[0,0,0,0])
            assert sim.names == header["joint_names"]
            times = np.array([0]+[r["time_s"] for r in rows])
            states = [rows[0]["before"]]+[r["after"] for r in rows]
            origin = header.get("reference_angle_rad", 0.0)
            angles = np.rad2deg([s["angle_rad"]-origin for s in states])
            # Fixed reference ticks show start (white) and target (green).
            center = np.array(pose["fixture_position_m"])+[0, 0, geometry.rotor_center_z_m]
            markings = None
            if args.marked:
                from visual_markings import build, Markings
                markfolder = build(folder/"markings")
                p.changeVisualShape(sim.knob,0,rgbaColor=[0,0,0,0])
                markings = Markings(markfolder,center)
            for offset, rgba in ([] if args.marked else [(0, [1,1,1,1]), (np.deg2rad(goal), [.2,1,.35,1])]):
                a = origin+offset
                visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[.006,.0015,.001], rgbaColor=rgba)
                position = center+[.034*np.cos(a), .034*np.sin(a), geometry.rotor_height_m/2+.001]
                p.createMultiBody(0, -1, visual, position, p.getQuaternionFromEuler([0,0,a]))
            forces = [np.linalg.norm(s["tip_normal_force_world_n"], axis=1).sum() for s in states]
            cached_index, cached_rgb = None, None
            for frame_index, t in enumerate(timeline):
                index = min(np.searchsorted(times, t+1e-8, side="right")-1, len(states)-1)
                state = states[index]
                finished = t+1e-8 >= times[-1]
                if index != cached_index:
                    for joint, q in zip(sim.ids, state["q_rad"]):
                        p.resetJointState(sim.robot, joint, q)
                    p.resetJointState(sim.knob, 0, state["angle_rad"])
                    if markings:markings.update(state["angle_rad"])
                    rgba = np.asarray(p.getCameraImage(width, width, view, projection,
                                                       renderer=p.ER_TINY_RENDERER, shadow=0,
                                                       lightDirection=[-1, -1, 3])[2], dtype=np.uint8)
                    cached_rgb = Image.fromarray(rgba[:, :, :3])
                    cached_index = index
                tile = Image.new("RGB", (width, height), "#111827")
                tile.paste(cached_rgb, (0, 76))
                d = ImageDraw.Draw(tile)
                d.text((12, 8), f"Brake {header['brake_nm']:.3f} N m", font=font(20), fill=color)
                d.text((12, 34), f"Angle {angles[index]:.2f} / target {goal:g} deg", font=font(17), fill="white")
                status = f"DONE at {times[-1]:.2f}s | replay frozen" if finished else f"Recorded t={times[index]:.2f}s | feedback control"
                d.text((12, 58), status, font=font(12), fill="#4ade80" if finished else "#cbd5e1")
                # A measured-angle dial supplements the mesh view; never drives it.
                cx, cy, radius = 350, 125, 28
                d.ellipse((cx-radius-7,cy-radius-7,cx+radius+7,cy+radius+7),fill="#111827",outline="#94a3b8",width=1)
                d.line((cx,cy-radius,cx,cy-radius+7),fill="white",width=3)
                d.ellipse((cx-3,cy+radius-1,cx+3,cy+radius+5),fill="#4ade80")
                theta = np.deg2rad(angles[index])-np.pi/2
                d.line((cx,cy,cx+(radius-5)*np.cos(theta),cy+(radius-5)*np.sin(theta)),fill="#facc15",width=3)
                d.text((310,164), "relative angle",font=font(10),fill="#111827")
                for j, (values, low, high, label, line_color) in enumerate([
                    (angles, -5, goal+10, "Angle (deg)", color),
                    (forces, 0, force_plot_max, "Tip normal-force sum (N)", "#fb923c"),
                ]):
                    top = 480+j*54
                    d.text((12, top), f"{label}: {values[index]:.2f}", font=font(12), fill=line_color)
                    points = [(12+times[k]/timeline[-1]*374, top+46-(values[k]-low)/(high-low)*29) for k in range(index+1)]
                    if j == 0:
                        target_y = top+46-(goal-low)/(high-low)*29
                        for x in range(12, 386, 10):
                            d.line((x, target_y, x+5, target_y), fill="#64748b")
                    if len(points) > 1:
                        d.line(points, fill=line_color, width=2)
                    x, y = points[-1]
                    d.ellipse((x-2, y-2, x+2, y+2), fill=line_color)
                composites[frame_index].paste(tile, (width*(n % 2), 60+height*(n//2)))
            print(f"Rendered condition {n+1}/4 ({len(timeline)} frames)", flush=True)
        for frame, t in zip(composites, timeline):
            d = ImageDraw.Draw(frame)
            d.text((14, 7), f"Four resistance levels - all reach +{goal:g} deg", font=font(22), fill="white")
            d.text((14, 36), f"PyBullet {args.playback_speed:g}x | {'marked rotor + full robot' if args.marked else 'arm-assisted close-up'} | t={t:.1f}s", font=font(14), fill="#cbd5e1")
        palette = composites[len(composites)//2].quantize(colors=256)
        indexed = [frame.quantize(palette=palette, dither=0) for frame in composites]
        durations = [100]*len(indexed)
        durations[-1] += 2000  # Presentation pause; explicitly labeled frozen.
        suffix = "" if args.playback_speed == 1 else f"_{args.playback_speed:g}x"
        path = folder/f"all_{goal:g}_degrees{suffix}.gif"
        indexed[0].save(path, save_all=True, append_images=indexed[1:], duration=durations,
                        loop=0, disposal=2, optimize=False)
        composites[-1].save(folder/"gif_final_frame.png")
        (folder/"gif_replay_metadata.json").write_text(json.dumps({
            "source": "01.jsonl through 04.jsonl", "physics_resimulated": False,
            "goal_angle_deg": goal, "angle_reference": "measured shaft angle after preparation",
            "reference_angles_rad": [d[0].get("reference_angle_rad",0) for d in datasets],
            "playback_speed": args.playback_speed, "arm_visuals_hidden_for_hand_closeup": not args.show_arm,
            "visual_markings": args.marked, "camera_eye_world_m": camera_eye, "camera_target_world_m": target.tolist(),
            "camera_fov_y_deg": args.fov, "camera_yaw_deg": args.yaw, "camera_elevation_deg": args.elevation, "camera_distance_m": args.distance,
            "fps": 10, "timeline_end_s": float(timeline[-1]), "frames": len(indexed),
            "episode_end_times_s": end_times, "final_presentation_pause_s": 2,
            "finished_tiles": "last measured state frozen, explicitly labeled",
        }, indent=2)+"\n")
        print(path, flush=True)
    finally:
        p.disconnect()


if __name__ == "__main__":
    main()
