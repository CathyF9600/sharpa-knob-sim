"""Solve a collision-UNVERIFIED pregrasp from the unchanged robot URDF.

Geometric contact targets are elastomer link origins, not calibrated skin points.
Review in the supplied viewer and run a simulator hold/close check before use.
"""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from dexx.deploy_config import ARM_BASE_POS, TABLE_SURFACE_Z
from dexx.tasks.sharpa_contact_adapt.geometry_tools import URDFFK, axis_rotation, origin_transform
from dexx.tasks.sharpa_contact_adapt.knob_spec import ARM_JOINTS, HAND_JOINTS, KnobGeometry


def main():
    g = KnobGeometry(**json.loads((ROOT / "assets/tasks/knob/geometry.json").read_text()))
    fk = URDFFK(ROOT / "assets/generated/fr3_with_right_sharpa_wave.urdf")
    q0 = dict(zip(ARM_JOINTS, [.24435, .17453, -.13963, -2.14675, -1.78024, 1.83260, -.05236]))
    q0.update({name: 0.0 for name in HAND_JOINTS})
    q0["right_thumb_CMC_FE"] = 1.0
    # Stock demo pose is palm-up. Roll about the finger direction so fingers
    # approach from ABOVE the table, rather than solving an under-table grasp.
    ref_rotation = fk.frames(q0, ARM_BASE_POS)["right_hand_C_MC"][:3, :3] @ axis_rotation([0, 0, 1], np.pi)
    center = np.array([.48, -.15, TABLE_SURFACE_Z + g.rotor_center_z_m])
    names = list(ARM_JOINTS) + list(HAND_JOINTS[:13])
    fingers = ["thumb", "index", "middle"]
    angles = np.deg2rad([0, -130, 135])
    radius = g.rotor_radius_m + .015
    targets = np.array([center + [radius*np.cos(a), radius*np.sin(a), .005] for a in angles])
    lo, hi = np.array([fk.limits[n] for n in names]).T
    x0 = np.clip([q0[n] for n in names], lo+1e-5, hi-1e-5)
    # Use actual collision-mesh samples to discourage intermediate phalanges
    # from cutting through the rotor while the tip origins meet their targets.
    import trimesh
    samples = []
    dense_samples = []
    for link in fk.root.findall("link"):
        if not link.get("name").startswith("right_"):
            continue
        for collision in link.findall("collision"):
            mesh_xml = collision.find("geometry/mesh")
            if mesh_xml is None:
                continue
            mesh = trimesh.load((ROOT / "assets/generated" / mesh_xml.get("filename")).resolve(), force="mesh")
            mesh.apply_scale(np.fromstring(mesh_xml.get("scale", "1 1 1"), sep=" "))
            mesh.apply_transform(origin_transform(collision.find("origin")))
            points = np.vstack([mesh.vertices, mesh.triangles_center])
            dense_samples.append((link.get("name"), points))
            points = points[::max(1, len(points)//300)]
            samples.append((link.get("name"), points))

    def residual(x):
        q = {**q0, **dict(zip(names, x))}
        frames = fk.frames(q, ARM_BASE_POS)
        tips = np.array([frames[f"right_{f}_elastomer"][:3, 3] for f in fingers])
        wrist = frames["right_hand_C_MC"]
        rot = Rotation.from_matrix(ref_rotation.T @ wrist[:3, :3]).as_rotvec()
        penalties = []
        for name, points in samples:
            frame = frames[name]
            world = points @ frame[:3, :3].T + frame[:3, 3]
            radial = np.linalg.norm(world[:, :2]-center[:2], axis=1)-g.rotor_radius_m
            vertical = abs(world[:, 2]-center[2])-g.rotor_height_m/2
            penalties.append(5*np.maximum(.002-np.maximum(radial, vertical), 0))
            penalties.append(5*np.maximum(TABLE_SURFACE_Z+.003-world[:, 2], 0))
        return np.concatenate([(tips-targets).ravel(), .05*rot, .0005*(x-x0), *penalties])

    candidates = []
    existing = ROOT / "assets/tasks/knob/pregrasp.json"
    if existing.exists():
        prior = json.loads(existing.read_text())["joint_positions_rad"]
        seed = np.clip([prior[n] for n in names], lo+1e-5, hi-1e-5)
        candidates.append(least_squares(residual, seed, bounds=(lo+1e-5, hi-1e-5), max_nfev=300))
    for wrist_seed in (() if candidates else (2.7, -2.7)):
        seed = x0.copy()
        seed[6] = np.clip(wrist_seed, lo[6]+1e-5, hi[6]-1e-5)
        candidates.append(least_squares(residual, seed, bounds=(lo+1e-5, hi-1e-5), max_nfev=500,
                                        ftol=1e-10, xtol=1e-10, gtol=1e-10))
    fit = min(candidates, key=lambda result: np.linalg.norm(residual(result.x)))
    # Refine using any dense samples missed by the coarse optimisation.
    for _ in range(3):
        frames = fk.frames({**q0, **dict(zip(names, fit.x))}, ARM_BASE_POS)
        added = False
        for name, points in dense_samples:
            frame = frames[name]
            world = points @ frame[:3, :3].T + frame[:3, 3]
            radial = np.linalg.norm(world[:, :2]-center[:2], axis=1)-g.rotor_radius_m
            vertical = abs(world[:, 2]-center[2])-g.rotor_height_m/2
            near = np.maximum(radial, vertical) < .001
            if near.any():
                samples.append((name, points[near]))
                added = True
        if not added:
            break
        fit = least_squares(residual, fit.x, bounds=(lo+1e-5, hi-1e-5), max_nfev=100)
    q = {**q0, **dict(zip(names, fit.x))}
    poses = fk.frames(q, ARM_BASE_POS)
    errors = [float(np.linalg.norm(poses[f"right_{f}_elastomer"][:3, 3]-target)) for f, target in zip(fingers, targets)]
    if max(errors) > .01:
        raise RuntimeError(f"Pregrasp fit failed (m): {errors}")
    result = {
        "status": "kinematic_only_collision_and_contact_unverified",
        "joint_positions_rad": q,
        "fixture_position_m": [float(center[0]), float(center[1]), TABLE_SURFACE_Z],
        "target_pad_origins_m": {f: t.tolist() for f, t in zip(fingers, targets)},
        "pad_origin_errors_m": errors,
        "note": "15 mm radial clearance is to link origins, not measured skin. Review before physics/printing.",
    }
    path = ROOT / "assets/tasks/knob/pregrasp.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {path}; pad-origin errors (mm): {np.array(errors)*1000}")


if __name__ == "__main__":
    main()
