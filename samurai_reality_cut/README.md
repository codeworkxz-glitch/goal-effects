# Samurai "Reality Cut" (iaido draw → diagonal slash → sheathe)

- `output/reality_cut.glb` – rigged samurai + katana + saya + baked animation (2.4 MB)
- `output/reality_cut.blend` – same, editable in Blender
- `build_reality_cut.py` – regenerates everything: `python build_reality_cut.py wushi_export2.fbx out_dir`
- `preview.py` – contact-sheet renders for checking poses

Timeline (60 fps, 2.9 s): calm 0–0.35 · grip 0.35–0.7 · draw 0.7–1.15 · raise/coil 1.15–1.52 ·
slash 1.52–1.69 (~0.1 s) · hold 1.76–2.0 · sheathe 2.0–2.7 · click 2.69–2.75 · settle to 2.9.
Hips lead the cut by 35 ms, spine 18 ms, then chest/arms/wrists; right foot steps forward, rear heel lifts.

Notes: the FBX had no katana (only two placeholder cylinders, removed) and no embedded textures,
so the katana is modelled procedurally and materials are untextured. Original character meshes,
rig and weights are untouched.
