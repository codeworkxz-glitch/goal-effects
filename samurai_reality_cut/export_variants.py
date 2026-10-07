"""Make viewer-friendly exports from the built blend.
    python export_variants.py reality_cut.blend textures_dir out_dir

  Samurai_RealityCut.fbx          textured: albedo + normal maps as *external* files in ./textures (relative paths)
  Samurai_RealityCut_colored.fbx  no texture files at all: flat material colours + baked vertex colours (shows in any viewer)
"""
import bpy, sys, os, shutil
import numpy as np
from PIL import Image

blend, tex_dir, out = sys.argv[-3], sys.argv[-2], sys.argv[-1]
os.makedirs(out, exist_ok=True)
bpy.ops.wm.open_mainfile(filepath=blend)
arm = bpy.data.objects['wushi']


def lin(c): return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def albedo_path(mat_name):          # tex_armo -> armo_albedo.jpg
    return os.path.join(tex_dir, mat_name[4:] + '_albedo.jpg')


def export(path, **kw):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.export_scene.fbx(filepath=path, use_selection=True, object_types={'ARMATURE', 'MESH'}, add_leaf_bones=False,
                             bake_anim=True, bake_anim_use_all_bones=True, bake_anim_use_nla_strips=False,
                             bake_anim_use_all_actions=False, bake_anim_force_startend_keying=True, bake_anim_step=1.0,
                             bake_anim_simplify_factor=0.0, apply_scale_options='FBX_SCALE_NONE', mesh_smooth_type='FACE',
                             use_armature_deform_only=False, **kw)
    print('wrote', path, round(os.path.getsize(path) / 1e6, 2), 'MB')


# ---------------- 1. colored fallback (flat colours + vertex colours) ----------------
avg = {}
for m in bpy.data.materials:
    if not m.name.startswith('tex_'): continue
    im = np.asarray(Image.open(albedo_path(m.name)).convert('RGB').resize((128, 128)), np.float32) / 255.0
    avg[m.name] = lin(im.reshape(-1, 3).mean(0))
for o in bpy.data.objects:
    if o.type != 'MESH' or not o.data.materials or not o.data.uv_layers: continue
    mat = o.data.materials[0]
    if mat.name not in avg: continue
    img = np.asarray(Image.open(albedo_path(mat.name)).convert('RGB'), np.float32) / 255.0
    H, W = img.shape[:2]
    me = o.data
    uv = np.empty(len(me.loops) * 2, np.float32); me.uv_layers.active.data.foreach_get('uv', uv); uv = uv.reshape(-1, 2)
    px = np.clip((uv[:, 0] * W).astype(int), 0, W - 1); py = np.clip(((1 - uv[:, 1]) * H).astype(int), 0, H - 1)
    rgb = lin(img[py, px])
    for a in list(me.color_attributes): me.color_attributes.remove(a)
    ca = me.color_attributes.new('Col', 'FLOAT_COLOR', 'CORNER')
    ca.data.foreach_set('color', np.concatenate([rgb, np.ones((len(rgb), 1), np.float32)], 1).ravel())
    me.color_attributes.active_color = ca
saved = {}
for m in bpy.data.materials:
    if m.name in avg and m.use_nodes:
        b = m.node_tree.nodes.get('Principled BSDF')
        saved[m.name] = [(l.from_socket, l.to_socket) for l in m.node_tree.links if l.to_node == b]
        # rebuild as plain colour material: remove texture links, set flat colour
        for l in list(m.node_tree.links):
            if l.to_node == b and l.to_socket.name in ('Base Color', 'Normal', 'Roughness', 'Metallic'):
                m.node_tree.links.remove(l)
        b.inputs['Base Color'].default_value = (*avg[m.name], 1.0)
        b.inputs['Roughness'].default_value = 0.5
        b.inputs['Metallic'].default_value = 0.2
        # also feed the vertex colours so viewers that honour them show the painted detail
        vc = m.node_tree.nodes.new('ShaderNodeVertexColor'); vc.layer_name = 'Col'
        m.node_tree.links.new(vc.outputs['Color'], b.inputs['Base Color'])
export(os.path.join(out, 'Samurai_RealityCut_colored.fbx'), path_mode='AUTO', embed_textures=False, colors_type='SRGB')

# ---------------- 2. textured with external relative textures ----------------
tex_out = os.path.join(out, 'textures'); os.makedirs(tex_out, exist_ok=True)
for m in bpy.data.materials:
    if m.name not in avg: continue
    nt = m.node_tree; b = nt.nodes.get('Principled BSDF')
    for l in list(nt.links):
        if l.to_node == b and l.to_socket.name == 'Base Color': nt.links.remove(l)
    for n in list(nt.nodes):
        if n.type == 'VERTEX_COLOR': nt.nodes.remove(n)
    b.inputs['Base Color'].default_value = (1, 1, 1, 1)
    b.inputs['Roughness'].default_value = 0.5; b.inputs['Metallic'].default_value = 0.0
    for n in nt.nodes:
        if n.type != 'TEX_IMAGE': continue
        nm = n.image.name                                  # e.g. armo_albedo / armo_normal
        ext = 'jpg'
        src = os.path.join(tex_dir, f'{nm}.{ext}')
        dst = os.path.join(tex_out, f'{nm}.{ext}')
        if os.path.exists(src): shutil.copy(src, dst)
        if n.label == 'BaseColor':
            nt.links.new(n.outputs['Color'], b.inputs['Base Color'])
        img = bpy.data.images.load(dst, check_existing=False)
        img.colorspace_settings.name = 'sRGB' if n.label == 'BaseColor' else 'Non-Color'
        n.image = img
export(os.path.join(out, 'Samurai_RealityCut.fbx'), path_mode='RELATIVE', embed_textures=False)
