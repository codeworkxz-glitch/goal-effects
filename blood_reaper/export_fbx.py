"""Export the Roblox FBX from a built rig:  python export_fbx.py <BloodReaper.blend> <out.fbx>
(same settings as build_reaper.py; lets the reviewed .blend be exported without re-solving)"""
import bpy, os, sys
argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
bpy.ops.wm.open_mainfile(filepath=os.path.abspath(argv[0]))
out = os.path.abspath(argv[1])
ao = bpy.data.objects['BloodReaper']
bpy.ops.object.select_all(action='DESELECT')
for o in bpy.data.objects:
    if o.type in ('ARMATURE', 'MESH'): o.select_set(True)
bpy.context.view_layer.objects.active = ao
bpy.context.scene.frame_set(0)
bpy.ops.export_scene.fbx(
    filepath=out, use_selection=True, object_types={'ARMATURE', 'MESH'}, add_leaf_bones=False,
    apply_scale_options='FBX_SCALE_NONE', apply_unit_scale=True, mesh_smooth_type='FACE',
    use_mesh_modifiers=False, use_armature_deform_only=False, primary_bone_axis='Y', secondary_bone_axis='X',
    bake_anim=True, bake_anim_use_all_bones=True, bake_anim_use_nla_strips=False, bake_anim_use_all_actions=False,
    bake_anim_force_startend_keying=True, bake_anim_step=1.0, bake_anim_simplify_factor=0.0,
    path_mode='COPY', embed_textures=True)
print('fbx', out, '%.2f MB' % (os.path.getsize(out) / 1e6))
