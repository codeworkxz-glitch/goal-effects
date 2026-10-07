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

## v2 – clipping fix
- Sword now sits at the belly front (tsuba ~10 cm left of centre) so the right hand reaches it without the
  forearm passing through the armor; after the draw the scabbard settles at the left hip, and returns for the sheathe.
- Per-frame search (`optimize_swivel`) over elbow swivel, wrist twist, hand-path lift and clavicle follow, scored on the
  *deformed* meshes (`clips.py`, BVH triangle overlap), then Viterbi-smoothed in time.
- Checks: `clipcheck.py` (triangle-overlap table), `breakdown.py` (which meshes), `depthcheck.py` (penetration depth, cm).
  Result: arm-arm, hand-hand, hand-armor overlaps are 0 except a short graze near t=0.9 s; remaining arm-vs-inner-cloth
  overlap is <= 1.7 cm (the bind pose itself has 1.3 cm there), hidden under the armor.
- Higher-res katana/saya, 60 fps baked on every frame, FBX 4.3 MB (limit 20 MB).

## v3 – arm anatomy + textures
- Arm solver is now anatomical: the elbow hinges in the shoulder-elbow-wrist plane (humerus roll solved to match),
  forearm takes pronation up to 85 deg, remaining wrist twist/deviation is limited, fingers wrap the handle radius.
  Per frame, swivel / wrist twist / hand-flip / hand path / clavicle follow are searched against the deformed meshes
  and joint-limit costs (4 parallel workers), then Viterbi-smoothed in time. `report_anatomy` prints the joint angles.
- The sword is raised upright in front of the chest between draw and guard (no head/back-armor clipping); the
  katana/scabbard never intersect the armor, head or sleeves (only the gripping fingers touch the handle).
- Procedural PBR textures (`texturing.py`, 2K albedo / ORM / normal) painted through the original UV atlas.
- Checks: `depthcheck.py` (worst arm-into-body depth 1.3 cm; bind pose itself is 1.1-1.3 cm), `swordclip.py`-style
  sword overlaps are 0, `multiview.py` renders front/side/back/top.
- FBX 18.9 MB (embedded textures).
