"""Build printable STL (millimetres), editable OpenSCAD, and simulator USDA (metres).

No mesh/Isaac dependencies. The rotor has no printed thread. Use commercial M8
hardware, washers and a locknut to adjust axial preload; calibrate actual drag.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from dexx.tasks.sharpa_contact_adapt.knob_spec import KnobGeometry


def ring_mesh(profile, segments=96):
    """Closed surface from (radius, z, circle|hex) cross-section. SI coordinates."""
    vertices = []
    for radius, z, shape in profile:
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            r = radius
            if shape == "hex":  # radius is apothem, NOT circumscribed radius
                r /= math.cos((angle + math.pi / 6) % (math.pi / 3) - math.pi / 6)
            vertices.append((r * math.cos(angle), r * math.sin(angle), z))
    faces = []
    for k in range(len(profile)):
        nxt = (k + 1) % len(profile)
        for i in range(segments):
            j = (i + 1) % segments
            a, b, c, d = k * segments + i, k * segments + j, nxt * segments + j, nxt * segments + i
            faces.extend([(a, b, c), (a, c, d)])
    volume = signed_volume(vertices, faces)
    if volume < 0:
        faces = [(a, c, b) for a, b, c in faces]
    return vertices, faces


def signed_volume(vertices, faces):
    def determinant(a, b, c):
        return (a[0] * (b[1]*c[2] - b[2]*c[1]) +
                a[1] * (b[2]*c[0] - b[0]*c[2]) + a[2] * (b[0]*c[1] - b[1]*c[0]))
    return sum(determinant(vertices[a], vertices[b], vertices[c]) for a, b, c in faces) / 6


def printable_meshes(g):
    base = [(g.flange_radius_m, 0, "circle"), (g.flange_radius_m, g.flange_height_m, "circle"),
            (g.pedestal_radius_m, g.flange_height_m, "circle"),
            (g.pedestal_radius_m, g.flange_height_m + g.pedestal_height_m, "circle"),
            (g.bore_radius_m, g.flange_height_m + g.pedestal_height_m, "circle"),
            (g.bore_radius_m, g.bolt_head_depth_m, "circle"),
            (g.bolt_head_af_m / 2, g.bolt_head_depth_m, "hex"),
            (g.bolt_head_af_m / 2, 0, "hex")]
    result = {"base": ring_mesh(base)}
    for shape in ("round", "hex"):
        kind = "circle" if shape == "round" else "hex"
        # Both variants fit inside a 50 mm envelope; hex across flats is 43.3 mm.
        r = g.rotor_radius_m if shape == "round" else g.rotor_radius_m * math.cos(math.pi/6)
        result[f"rotor_{shape}"] = ring_mesh([
            (r, 0, kind), (r, g.rotor_height_m, kind),
            (g.bore_radius_m, g.rotor_height_m, "circle"), (g.bore_radius_m, 0, "circle")])
    return result


def write_stl(path, vertices, faces):
    with path.open("wb") as f:
        f.write(b"Dex-X knob, millimetres, no printed thread".ljust(80, b"\0"))
        f.write(struct.pack("<I", len(faces)))
        for face in faces:
            points = [vertices[i] for i in face]
            a, b, c = points
            u, v = [b[i]-a[i] for i in range(3)], [c[i]-a[i] for i in range(3)]
            n = [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]
            length = math.sqrt(sum(x*x for x in n))
            if length < 1e-15:
                raise ValueError("Degenerate STL triangle")
            values = [x/length for x in n] + [x*1000 for p in points for x in p]
            f.write(struct.pack("<12fH", *values, 0))


def usda(g):
    """Passive one-joint articulation. No drive and no target servo on rotor."""
    z = g.rotor_center_z_m
    iz = g.rotor_inertia_z
    ix = g.rotor_mass_kg * (3*g.rotor_radius_m**2 + g.rotor_height_m**2) / 12
    return f'''#usda 1.0
(
    defaultPrim = "Knob"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "Knob" (prepend apiSchemas = ["PhysicsArticulationRootAPI", "PhysxArticulationAPI"])
{{
    bool physxArticulation:enabledSelfCollisions = false
    def Xform "base" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"])
    {{
        float physics:mass = 0.2
        def Cylinder "flange" (prepend apiSchemas = ["PhysicsCollisionAPI"])
        {{
            double radius = {g.flange_radius_m}
            double height = {g.flange_height_m}
            token axis = "Z"
            double3 xformOp:translate = (0, 0, {g.flange_height_m/2})
            uniform token[] xformOpOrder = ["xformOp:translate"]
            color3f[] primvars:displayColor = [(0.23, 0.26, 0.30)]
        }}
        def Cylinder "pedestal" (prepend apiSchemas = ["PhysicsCollisionAPI"])
        {{
            double radius = {g.pedestal_radius_m}
            double height = {g.pedestal_height_m}
            token axis = "Z"
            double3 xformOp:translate = (0, 0, {g.flange_height_m+g.pedestal_height_m/2})
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }}
    }}
    def Xform "rotor" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"])
    {{
        double3 xformOp:translate = (0, 0, {z})
        uniform token[] xformOpOrder = ["xformOp:translate"]
        float physics:mass = {g.rotor_mass_kg}
        float3 physics:diagonalInertia = ({ix}, {ix}, {iz})
        point3f physics:centerOfMass = (0, 0, 0)
        def Cylinder "body" (prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI"])
        {{
            double radius = {g.rotor_radius_m}
            double height = {g.rotor_height_m}
            token axis = "Z"
            float physxCollision:contactOffset = 0.0005
            float physxCollision:restOffset = 0
            color3f[] primvars:displayColor = [(0.12, 0.55, 0.80)]
        }}
        def Cube "angle_mark"
        {{
            double size = 1
            double3 xformOp:scale = ({g.rotor_radius_m*0.65}, 0.002, 0.0004)
            double3 xformOp:translate = ({g.rotor_radius_m*0.55}, 0, {g.rotor_height_m/2+0.0002})
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
            color3f[] primvars:displayColor = [(1, 0.85, 0.1)]
        }}
    }}
    def PhysicsFixedJoint "mount"
    {{
        rel physics:body1 = </Knob/base>
    }}
    def PhysicsRevoluteJoint "shaft"
    {{
        rel physics:body0 = </Knob/base>
        rel physics:body1 = </Knob/rotor>
        token physics:axis = "Z"
        point3f physics:localPos0 = (0, 0, {z})
        point3f physics:localPos1 = (0, 0, 0)
        bool physics:collisionEnabled = false
    }}
}}
'''


def scad(g):
    return f'''// Units: mm. No printed threads. Edit source JSON and rebuild to keep sim aligned.
// openscad -o base.stl -D 'part="base"' fixture.scad
part = "assembly"; // base, rotor_round, rotor_hex, assembly
$fn = 96;
r = {1000*g.rotor_radius_m}; h = {1000*g.rotor_height_m}; bore = {1000*g.bore_radius_m};
flange_r = {1000*g.flange_radius_m}; flange_h = {1000*g.flange_height_m};
post_r = {1000*g.pedestal_radius_m}; post_h = {1000*g.pedestal_height_m};
head_af = {1000*g.bolt_head_af_m}; head_h = {1000*g.bolt_head_depth_m};
gap = {1000*g.washer_gap_m};
module base() {{ difference() {{
    union() {{ cylinder(r=flange_r,h=flange_h); translate([0,0,flange_h]) cylinder(r=post_r,h=post_h); }}
    translate([0,0,-0.1]) cylinder(r=bore,h=flange_h+post_h+0.2);
    translate([0,0,-0.1]) rotate([0,0,30]) cylinder(r=head_af/sqrt(3),h=head_h+0.1,$fn=6);
}} }}
module rotor(hex=false) {{ difference() {{
    if (hex) rotate([0,0,30]) cylinder(r=r,h=h,$fn=6); else cylinder(r=r,h=h);
    translate([0,0,-0.1]) cylinder(r=bore,h=h+0.2);
}} }}
if (part=="base") base();
else if (part=="rotor_round") rotor();
else if (part=="rotor_hex") rotor(true);
else {{ base(); translate([0,0,flange_h+post_h+gap]) rotor(); }}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, help="Geometry JSON in SI units")
    parser.add_argument("--out", type=Path, default=ROOT / "assets/tasks/knob")
    args = parser.parse_args()
    g = KnobGeometry(**json.loads(args.spec.read_text())) if args.spec else KnobGeometry()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, (vertices, faces) in printable_meshes(g).items():
        write_stl(args.out / f"{name}.stl", vertices, faces)
    (args.out / "geometry.json").write_text(json.dumps(asdict(g), indent=2) + "\n")
    (args.out / "fixture.scad").write_text(scad(g))
    (args.out / "knob.usda").write_text(usda(g))
    print(f"Built printable millimetre STLs and SI simulation asset in {args.out}")


if __name__ == "__main__":
    main()
