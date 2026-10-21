# Printable passive knob

See [task, assembly and validation guide](../../../docs/KNOB_ADAPTATION.md).

STL and SCAD use **millimetres**. JSON and USD use **metres**.
These meshes and the USD were generated locally by `scripts/knob/build_assets.py`.
No external CAD model is included. `geometry.json` is the dimension source.

Print one base and either the round or hexagonal rotor. Use metal axle hardware;
there are no printed threads. The simulation currently uses the round rotor only.
Mesh watertightness has been checked; printing tolerances and physical assembly
have not. `pregrasp.json` is a kinematic candidate, not a verified robot command.

## Visual angle scales

`markings/` contains 50 mm top and 157.080 × 20 mm wraparound side SVG templates, PNG textures, and visual-only OBJ surfaces. Generate with `scripts/knob/visual_markings.py` via its `build()` helper; enable replay rendering with `render_goal_gif.py --marked`. Existing STL/collision geometry and Isaac USD are unchanged. Use flat paint/printing rather than raised grip features, and recalibrate friction after applying coatings. Camera evaluation and previews: `logs/knob180_marked/README.md`.
