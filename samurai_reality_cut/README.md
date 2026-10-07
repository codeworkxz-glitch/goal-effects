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

## Viewer note
Embedded-texture FBX showed grey in some viewers, so the delivery is now `Samurai_RealityCut_textured.zip` (FBX + ./textures, relative paths) and `Samurai_RealityCut_colored.fbx` (flat colours + baked vertex colours, no texture files). Built with `export_variants.py`.

## v4 – 5 s, flat colours (Roblox-friendly)
- Timeline stretched to 5.0 s (301 frames @ 60 fps) with a smooth time-warp (`KNOTS`): slow breathing calm, slow grip + draw,
  a long coil before the cut, the slash itself still ~0.17 s, a longer hold, slow sheathe + click.
- Extra detail: lobed tsuba + gold seppa/fuchi/menuki and ito bands on the katana, gold bands + kurikata on the saya,
  spring-driven secondary motion on skirt panels and belt cords (`secondary_motion`), `post_skirts.py` swings the panels
  clear of the thighs/greaves.
- Colours only, no image textures: `color_only.py` paints red-lacquer armour with gold trim, slate-blue lattice cloth,
  tan leather greaves/gloves, gold crest, skin face; flat class materials + face-flat vertex colours.
- Final file: `Samurai_RealityCut_colored.fbx` (5.1 MB).
