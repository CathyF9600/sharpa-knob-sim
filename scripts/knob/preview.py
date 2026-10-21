"""Preview the actual collision meshes + printable fixture, without Isaac Sim.

--export-only writes a Viser recording, a static PNG, and geometric diagnostics.
Otherwise also serves an interactive Viser scene (no robot connection).
Diagnostics flag sampled penetration; they do not certify collision freedom.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from dexx.deploy_config import ARM_BASE_POS, TABLE_SURFACE_Z
from dexx.tasks.sharpa_contact_adapt.geometry_tools import URDFFK, origin_transform
from dexx.tasks.sharpa_contact_adapt.knob_spec import KnobGeometry


def load_scene(pose_path=None):
    import trimesh

    folder = ROOT / "assets/tasks/knob"
    pose = json.loads((pose_path or folder / "pregrasp.json").read_text())
    g = KnobGeometry(**json.loads((folder / "geometry.json").read_text()))
    urdf = ROOT / "assets/generated/fr3_with_right_sharpa_wave.urdf"
    fk = URDFFK(urdf)
    frames = fk.frames(pose["joint_positions_rad"], ARM_BASE_POS)
    meshes = []
    for link in fk.root.findall("link"):
        name = link.get("name")
        for i, collision in enumerate(link.findall("collision")):
            geom = collision.find("geometry/mesh")
            if geom is None:
                continue
            file = (urdf.parent / geom.get("filename")).resolve()
            mesh = trimesh.load(file, force="mesh")
            mesh.apply_scale(np.fromstring(geom.get("scale", "1 1 1"), sep=" "))
            mesh.apply_transform(frames[name] @ origin_transform(collision.find("origin")))
            meshes.append((f"robot/{name}_{i}", mesh))
    for part, dz in [("base", 0), ("rotor_round", g.rotor_center_z_m-g.rotor_height_m/2)]:
        mesh = trimesh.load(folder / f"{part}.stl", force="mesh")
        mesh.apply_scale(.001)  # STL millimetres -> scene metres
        mesh.apply_translation(np.array(pose["fixture_position_m"]) + [0, 0, dz])
        meshes.append((part, mesh))
    return meshes, frames, pose, g


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "logs/knob_preview")
    parser.add_argument("--pose", type=Path)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    import trimesh
    import viser
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    meshes, frames, pose, g = load_scene(args.pose)
    args.out.mkdir(parents=True, exist_ok=True)
    server = viser.ViserServer(host="127.0.0.1", port=args.port)
    try:
        server.scene.set_up_direction("+z")
        fig = plt.figure(figsize=(12, 6))
        axes = [fig.add_subplot(1, 2, i+1, projection="3d") for i in range(2)]
        fixture = np.array(pose["fixture_position_m"])
        center = fixture + [0, 0, g.rotor_center_z_m]
        suspicious = {}
        for name, mesh in meshes:
            color = (50, 155, 205) if name == "rotor_round" else (130, 140, 150)
            if "elastomer" in name:
                color = (225, 120, 60)
            server.scene.add_mesh_simple("/"+name, vertices=mesh.vertices, faces=mesh.faces, color=color)
            if name.startswith("robot/right_"):
                sample = np.vstack([mesh.vertices, mesh.triangles_center])
                rho = np.linalg.norm(sample[:, :2]-center[:2], axis=1)
                inside = (rho < g.rotor_radius_m-.0005) & (abs(sample[:, 2]-center[2]) < g.rotor_height_m/2-.0005)
                below = sample[:, 2] < TABLE_SURFACE_Z-.0005
                if inside.any() or below.any():
                    suspicious[name] = {"samples_inside_rotor": int(inside.sum()), "samples_below_table": int(below.sum())}
            # Static closeups from the same mesh transforms as Viser.
            if name.startswith("robot/right_") or name in ("base", "rotor_round"):
                triangles = mesh.triangles
                stride = max(1, len(triangles)//2200)
                for ax in axes:
                    ax.add_collection3d(Poly3DCollection(triangles[::stride], facecolor=np.array(color)/255,
                                                       edgecolor="none", alpha=.7))
        for finger in ("thumb", "index", "middle"):
            p = frames[f"right_{finger}_elastomer"][:3, 3]
            server.scene.add_frame(f"/pads/{finger}", position=p, axes_length=.015, axes_radius=.0005)
        table = trimesh.creation.box(extents=(.45, .45, .01))
        table.apply_translation([center[0], center[1], TABLE_SURFACE_Z-.005])
        server.scene.add_mesh_simple("/table", vertices=table.vertices, faces=table.faces, color=(185, 175, 155))
        for ax, elev, azim in zip(axes, [80, 15], [-90, -45]):
            ax.set(xlim=(center[0]-.13, center[0]+.13), ylim=(center[1]-.13, center[1]+.13),
                   zlim=(TABLE_SURFACE_Z-.01, center[2]+.16), xlabel="x (m)", ylabel="y (m)", zlabel="z (m)")
            ax.view_init(elev=elev, azim=azim)
            ax.set_box_aspect([1, 1, .7])
        fig.suptitle("Sharpa knob pregrasp — kinematic preview, contact not yet validated")
        fig.tight_layout()
        fig.savefig(args.out / "pregrasp.png", dpi=160)
        plt.close(fig)
        (args.out / "scene.viser").write_bytes(server.get_scene_serializer().serialize())
        diagnostics = {"sampled_penetration_flags": suspicious,
                       "warning": "Sampled geometry only; full collision/contact and physics unverified."}
        (args.out / "geometry_check.json").write_text(json.dumps(diagnostics, indent=2)+"\n")
        print(json.dumps(diagnostics, indent=2))
        print(f"Preview files: {args.out}")
        if not args.export_only:
            print(f"Interactive view: http://127.0.0.1:{args.port}; Ctrl-C to stop")
            while True:
                time.sleep(.5)
    finally:
        server.stop()


if __name__ == "__main__":
    main()
