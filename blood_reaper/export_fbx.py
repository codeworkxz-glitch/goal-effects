"""Export the Roblox FBX from a built rig:  python export_fbx.py <BloodReaper.blend> <out.fbx>

Meshes: body parts merged into one skinned mesh per material; the file's default bone transforms are the bind pose
(the clip runs on frames 1..N).

Roblox-safe export: Blender's default FBX export converts Z-up metres to Y-up centimetres by putting a -90 deg X
rotation and a x100 scale on every root node (armature and meshes).  Roblox's importer mishandles rotated / scaled
root nodes on skinned, animated rigs (the rig comes in flipped and parts riding on their own bones stop following
the animation).  Here the conversion is baked into the data instead -- mesh vertices, bone rest poses and the
bone-local translation keys -- so every node in the file has identity rotation and unit scale, the scene is in
centimetres (unit scale 0.01, FBX UnitScaleFactor 1) and Y is up.  The character faces +Z.
"""
import bpy, os, sys
from mathutils import Matrix
argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
bpy.ops.wm.open_mainfile(filepath=os.path.abspath(argv[0]))
out = os.path.abspath(argv[1])
sc = bpy.context.scene
ao = bpy.data.objects['BloodReaper']
meshes = [o for o in bpy.data.objects if o.type == 'MESH']
assert ao.matrix_world == Matrix.Identity(4) and all(o.matrix_world == Matrix.Identity(4) for o in meshes)

# one skinned mesh per material (Roblox makes one MeshPart per material anyway).  Meshes bound to a single bone
# are turned into welded rigid parts by Roblox's importer (they then ignore the animation), so the scythe and the
# eyes get a token second influence on the bone they ride with (0.1 %: a few millimetres at most).
bpy.context.view_layer.objects.active = None
body = [o for o in meshes if o.data.materials and o.data.materials[0].name == 'BloodReaper_Body']
bpy.ops.object.select_all(action='DESELECT')
for o in body: o.select_set(True)
main = bpy.data.objects['cadnav']
bpy.context.view_layer.objects.active = main
bpy.ops.object.join()
main.name = main.data.name = 'BloodReaper_Body'
for name, bone2 in (('Scythe', 'Hand_L'), ('BloodReaper_Eyes', 'Neck')):
    o = bpy.data.objects[name]
    g1 = o.vertex_groups[0]
    g2 = o.vertex_groups.new(name=bone2)
    idx = list(range(len(o.data.vertices)))
    g1.add(idx, 0.999, 'REPLACE'); g2.add(idx, 0.001, 'REPLACE')
meshes = [o for o in bpy.data.objects if o.type == 'MESH']

CM = 100.0
CONV = Matrix.Scale(CM, 4) @ Matrix.Rotation(-1.5707963267948966, 4, 'X')     # Z-up metres -> Y-up centimetres

# meshes: bake into the vertices
for me in {o.data for o in meshes}:
    me.transform(CONV)
    me.update()

# bone rest poses
sc.frame_set(0)
bpy.context.view_layer.objects.active = ao
for o in bpy.context.view_layer.objects: o.select_set(o == ao)
bpy.ops.object.mode_set(mode='EDIT')
for eb in ao.data.edit_bones:
    eb.transform(CONV, scale=True, roll=True)
bpy.ops.object.mode_set(mode='OBJECT')

# animation: rotations are bone-local (unchanged by a uniform rotate + scale of the rest pose); translations are
# bone-local lengths -> centimetres
act = ao.animation_data.action
for fc in (list(act.fcurves) if hasattr(act, 'fcurves') else [fc for ly in act.layers for st in ly.strips for cb in st.channelbags for fc in cb.fcurves]):
    if fc.data_path.endswith('.location'):
        for kp in fc.keyframe_points:
            kp.co[1] *= CM; kp.handle_left[1] *= CM; kp.handle_right[1] *= CM
        fc.update()

# the exporter writes the bones' default (node) transforms from the current frame, and Roblox builds the imported
# model from them: they must be the bind pose, or the skinned body appears posed while rigid parts sit at the
# bind pose.  Shift the clip to frames 1..N, key the rest pose on frame 0 (outside the exported range) and export
# from frame 0.
fcs = list(act.fcurves) if hasattr(act, 'fcurves') else [fc for ly in act.layers for st in ly.strips for cb in st.channelbags for fc in cb.fcurves]
for fc in fcs:
    for kp in fc.keyframe_points:
        kp.co[0] += 1; kp.handle_left[0] += 1; kp.handle_right[0] += 1
    rest_v = 1.0 if (fc.data_path.endswith('rotation_quaternion') and fc.array_index == 0) else 0.0
    if fc.data_path.endswith('scale'): rest_v = 1.0
    fc.keyframe_points.insert(0.0, rest_v, options={'FAST'}).interpolation = 'CONSTANT'
    fc.update()
n_frames = sc.frame_end - sc.frame_start + 1
sc.frame_start, sc.frame_end = 1, n_frames

sc.unit_settings.system = 'METRIC'
sc.unit_settings.scale_length = 0.01
sc.unit_settings.length_unit = 'CENTIMETERS'

bpy.ops.object.select_all(action='DESELECT')
for o in [ao] + meshes: o.select_set(True)
bpy.context.view_layer.objects.active = ao
sc.frame_set(0)
assert all(max(abs(a - b) for ra, rb in zip(pb.matrix_basis, Matrix.Identity(4)) for a, b in zip(ra, rb)) < 1e-5
           for pb in ao.pose.bones), 'not at the rest pose'
bpy.ops.export_scene.fbx(
    filepath=out, use_selection=True, object_types={'ARMATURE', 'MESH'}, add_leaf_bones=False,
    apply_scale_options='FBX_SCALE_UNITS', apply_unit_scale=True, global_scale=1.0,
    use_space_transform=False, axis_forward='-Z', axis_up='Y', bake_space_transform=False,
    mesh_smooth_type='FACE', use_mesh_modifiers=False, use_armature_deform_only=False,
    primary_bone_axis='Y', secondary_bone_axis='X', armature_nodetype='NULL',
    bake_anim=True, bake_anim_use_all_bones=True, bake_anim_use_nla_strips=False, bake_anim_use_all_actions=False,
    bake_anim_force_startend_keying=True, bake_anim_step=1.0, bake_anim_simplify_factor=0.0,
    path_mode='COPY', embed_textures=True)
print('fbx', out, '%.2f MB' % (os.path.getsize(out) / 1e6))
