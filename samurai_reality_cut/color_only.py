"""
Flat-colour pass for the samurai (no image textures - e.g. for Roblox, which can't take the texture maps).

    python color_only.py built.blend out_dir

Paints classic samurai colours straight onto the mesh:
  * red-lacquer armour (plate rows alternate two reds), gold fittings on every plate rim / the helmet crest
  * slate-blue hemp cloth with a diamond lattice, tan leather greaves and gloves, iron kote plates
  * skin face with painted eyes, katana in steel / black silk wrap / gold fittings
Every face gets (a) a class material (flat colour, ~14 materials) and (b) a face-flat vertex colour carrying the shading
variation, so the colours survive an FBX export with no texture files at all.
Writes Samurai_RealityCut_colored.fbx and Samurai_RealityCut_colored.blend.
"""
import bpy, bmesh, sys, os, math
import numpy as np
from mathutils import Vector

blend, out = sys.argv[-2], sys.argv[-1]
os.makedirs(out, exist_ok=True)
bpy.ops.wm.open_mainfile(filepath=blend)
arm = bpy.data.objects['wushi']

C = {   # sRGB colours (0-1)
    'red': (0.68, 0.16, 0.18), 'red_dark': (0.40, 0.06, 0.08), 'gold': (0.92, 0.72, 0.22), 'slate': (0.25, 0.32, 0.41),
    'tan': (0.78, 0.60, 0.44), 'iron': (0.17, 0.17, 0.20), 'skin': (0.76, 0.56, 0.44), 'cream': (0.80, 0.72, 0.56),
    'leather': (0.42, 0.26, 0.15), 'black': (0.035, 0.035, 0.045), 'steel': (0.80, 0.82, 0.86), 'white': (0.92, 0.90, 0.84),
    'mask_red': (0.50, 0.07, 0.08), 'brown_dark': (0.16, 0.08, 0.05),
}
METAL = {'gold': 0.9, 'steel': 0.95, 'iron': 0.6}
ROUGH = {'gold': 0.3, 'steel': 0.2, 'iron': 0.4, 'red': 0.35, 'black': 0.3, 'mask_red': 0.35}


def lin(c): return tuple(float(((x / 12.92) if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4)) for x in c)


def make_material(cls):
    m = bpy.data.materials.new('Samurai_' + cls); m.use_nodes = True
    nt = m.node_tree; b = nt.nodes['Principled BSDF']
    b.inputs['Base Color'].default_value = (*lin(C[cls]), 1.0)
    b.inputs['Roughness'].default_value = ROUGH.get(cls, 0.55)
    b.inputs['Metallic'].default_value = METAL.get(cls, 0.0)
    vc = nt.nodes.new('ShaderNodeVertexColor'); vc.layer_name = 'Col'
    nt.links.new(vc.outputs['Color'], b.inputs['Base Color'])
    return m


mats = {}
def mat_index(me, cls):
    if cls not in mats: mats[cls] = make_material(cls)
    m = mats[cls]
    if m.name not in [x.name for x in me.materials if x]:
        me.materials.append(m)
    return [x.name for x in me.materials].index(m.name)


# ---- per-bone frames for bone-local plate rows / lattice -------------------------------------
names = [b.name for b in arm.data.bones]
head = {}; axis = {}; e1 = {}; e2 = {}
for b in arm.data.bones:
    h = np.array(b.head_local); d = np.array(b.tail_local) - h
    ax = d / np.linalg.norm(d) if np.linalg.norm(d) > 1e-4 else np.array([0, 1, 0.0])
    head[b.name] = h; axis[b.name] = ax
for n in ('HIP', 'spine', 'chest', 'neck'):
    head[n] = np.array([-5.8, 0.0, 3.0]); axis[n] = np.array([0, 1.0, 0])
for n in names:
    ref = np.array([0, 0, 1.0]) if abs(axis[n][2]) < 0.9 else np.array([1.0, 0, 0])
    a = np.cross(axis[n], ref); a /= np.linalg.norm(a); e1[n] = a; e2[n] = np.cross(axis[n], a)


def local_coords(bone, p):
    v = p - head[bone]; s = float(v @ axis[bone]); rv = v - s * axis[bone]
    r = max(float(np.linalg.norm(rv)), 3.0)
    return s, math.atan2(float(rv @ e2[bone]), float(rv @ e1[bone])) * r


def dominant_bone(me, vgs, poly):
    w = {}
    for vi in poly.vertices:
        for g in me.vertices[vi].groups:
            n = vgs.get(g.group)
            if n: w[n] = w.get(n, 0.0) + g.weight
    return max(w, key=w.get) if w else None


def paint(obj):
    me = obj.data
    vgs = {i: g.name for i, g in enumerate(obj.vertex_groups)}
    mw = arm.matrix_world.inverted() @ obj.matrix_world
    bm = bmesh.new(); bm.from_mesh(me); bm.faces.ensure_lookup_table()
    boundary = [any(len(e.link_faces) == 1 for e in f.edges) for f in bm.faces]
    bm.free()
    uvl = me.uv_layers.active.data if me.uv_layers else None
    rng = np.random.default_rng(7)
    name = obj.name
    ypos = [ (mw @ p.center).y for p in me.polygons ]
    if name in ('helmet',):
        y95 = np.percentile(ypos, 97); y30 = np.percentile(ypos, 30)
    cols = []; classes = []
    for p in me.polygons:
        c3 = np.array(mw @ p.center)
        bone = dominant_bone(me, vgs, p)
        uv = None
        if uvl:
            uv = np.mean([list(uvl[l].uv) for l in p.loop_indices], axis=0)
        cls = 'red'; shade = 1.0
        if name in ('armo', 'Protective_L', 'helmet', 'mask'):
            s, arc = local_coords(bone, c3) if bone else (0.0, 0.0)
            row = int(math.floor(s / 3.0)); cord = int(math.floor(arc / 1.6))
            shade = 1.0 if row % 2 == 0 else 0.90
            if cord % 3 == 0: shade *= 0.94
            cls = 'red'
        if name == 'armo':
            if bone and bone.startswith('rope'):
                cls = 'red_dark'; shade = 1.0 if int(c3[1] // 1.6) % 2 == 0 else 0.85
                if bone.endswith('3') or bone.endswith('4'): cls = 'gold'; shade = 1.0
            elif boundary[p.index]:
                cls = 'gold'; shade = 1.0
        elif name == 'neiyi_':
            cls = 'slate'
            s, arc = local_coords(bone, c3) if bone else (0.0, 0.0)
            u_, v_ = (s + arc) / 4.2, (s - arc) / 4.2
            lat = (int(math.floor(u_)) + int(math.floor(v_))) % 2
            shade = 1.0 if lat == 0 else 1.28
        elif name == 'Protective_L':
            if bone and any(bone.endswith(k) for k in ('ankle', 'foot', 'knee', 'leg', 'foottop')):
                cls = 'tan'; s, arc = local_coords(bone, c3)
                shade = 1.0 if int(math.floor(arc / 2.2)) % 2 == 0 else 0.84
                if boundary[p.index]: cls = 'red'; shade = 1.0
            else:
                cls = 'red'
                if boundary[p.index]: cls = 'gold'; shade = 1.0
        elif name == 'glovesL':
            u, v = uv if uv is not None else (0, 0)
            if u < 0.40 and v > 0.60:
                cls = 'red'; shade = 1.0
                if boundary[p.index]: cls = 'gold'
            elif u > 0.42 and v > 0.70:
                cls = 'iron'
            else:
                cls = 'leather'; shade = 1.0 + (0.12 if (int(c3[0] * 2) + int(c3[2] * 2)) % 2 else 0.0)
        elif name == 'head_2':
            cls = 'skin'; u, v = uv if uv is not None else (0, 0)
            for cu in (0.41, 0.535):
                d = math.hypot(u - cu, v - 0.688)
                if d < 0.0105: cls = 'black'
                elif d < 0.0185: cls = 'white'
                if abs(u - cu) < 0.03 and abs(v - 0.722) < 0.009: cls = 'black'
            if math.hypot(u - 0.4725, v - 0.578) < 0.026 and abs(v - 0.578) < 0.012: cls = 'red_dark'
            if cls == 'skin': shade = 0.97 + 0.06 * float(rng.random())
        elif name == 'helmet':
            y = (mw @ p.center).y
            cls = 'red'
            if y > y95: cls = 'gold'; shade = 1.0
            elif boundary[p.index]: cls = 'gold'; shade = 1.0
            elif y < y30: shade *= 0.92
        elif name == 'mask':
            cls = 'mask_red'; shade = 0.9 + 0.2 * float(rng.random())
            if boundary[p.index]: cls = 'gold'; shade = 1.0
        elif name in ('Katana', 'Saya'):
            cls = {0: 'steel', 1: 'iron', 2: 'gold', 3: 'black', 4: 'brown_dark'}[p.material_index]
            if name == 'Saya' and cls == 'brown_dark': shade = 1.0
            if cls == 'black': shade = 1.0 + (0.7 if int(c3[2] // 1.4) % 2 else 0.0)
        classes.append(cls); cols.append(tuple(min(1.0, x * shade) for x in C[cls]))
    # materials per class, polygon material indices
    me.materials.clear(); mats_local = {}
    for p, cls in zip(me.polygons, classes):
        p.material_index = mat_index(me, cls)
    # face-flat vertex colours (linear floats; the FBX exporter converts back to sRGB)
    for a in list(me.color_attributes): me.color_attributes.remove(a)
    ca = me.color_attributes.new('Col', 'FLOAT_COLOR', 'CORNER')
    data = np.zeros((len(me.loops), 4), np.float32)
    for p, c in zip(me.polygons, cols):
        l = lin(c)
        for li in p.loop_indices: data[li] = (*l, 1.0)
    ca.data.foreach_set('color', data.ravel())
    me.color_attributes.active_color = ca
    from collections import Counter
    print(name, dict(Counter(classes)))


for o in bpy.data.objects:
    if o.type == 'MESH' and o.name in ('armo', 'neiyi_', 'Protective_L', 'glovesL', 'head_2', 'helmet', 'mask', 'Katana', 'Saya'):
        paint(o)
for img in list(bpy.data.images): bpy.data.images.remove(img)
for m in list(bpy.data.materials):
    if m.users == 0: bpy.data.materials.remove(m)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out, 'Samurai_RealityCut_colored.blend'))
bpy.ops.object.select_all(action='SELECT')
bpy.ops.export_scene.fbx(filepath=os.path.join(out, 'Samurai_RealityCut_colored.fbx'), use_selection=True,
                         object_types={'ARMATURE', 'MESH'}, add_leaf_bones=False, bake_anim=True, bake_anim_use_all_bones=True,
                         bake_anim_use_nla_strips=False, bake_anim_use_all_actions=False, bake_anim_force_startend_keying=True,
                         bake_anim_step=1.0, bake_anim_simplify_factor=0.0, apply_scale_options='FBX_SCALE_NONE',
                         mesh_smooth_type='FACE', use_armature_deform_only=False, path_mode='AUTO', embed_textures=False,
                         colors_type='SRGB')
print('FBX MB', os.path.getsize(os.path.join(out, 'Samurai_RealityCut_colored.fbx')) / 1e6)
