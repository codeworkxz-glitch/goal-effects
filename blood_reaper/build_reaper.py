"""Blood Reaper – build the rigged, animated Roblox goal effect.

    python build_reaper.py <out_dir> [replay|anim]

replay : drives the new armature with the original Alembic bone samples (frames 30-96) and reports the skinning
         error against the cached meshes (validates the recovered weights)
anim   : the Mythic "BLOOD REAPER" goal-effect animation from choreo_reaper.py, then FBX export
"""
import bpy, bmesh, sys, os, math
import numpy as np
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import reaper_rig as RR

ABC = os.path.join(HERE, 'source', 'reaper.abc')
TEX = os.path.join(HERE, 'source', 'textures')
K = float(os.environ.get('REAPER_SCALE', '0.042'))      # model units -> metres (see README: 15 studs tall)
FPS = 60


def log(*a): print('[reaper]', *a, flush=True)


# --------------------------------------------------------------------------------------------------
def import_static(rig):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.alembic_import(filepath=ABC)
    sc = bpy.context.scene
    sc.frame_set(RR.BIND)
    dg = bpy.context.evaluated_depsgraph_get()
    out = {}
    for o in [o for o in bpy.data.objects if o.type == 'MESH']:
        e = o.evaluated_get(dg)
        me = bpy.data.meshes.new_from_object(e, preserve_all_data_layers=True, depsgraph=dg)
        W = np.array(e.matrix_world)
        M = Matrix(W.tolist())
        me.transform(M)
        me.transform(Matrix.Translation(Vector((-rig.off).tolist())))
        me.transform(Matrix.Scale(K, 4))
        uv = me.uv_layers[0].data
        arr = np.zeros(len(uv) * 2); uv.foreach_get('uv', arr); arr = arr.reshape(-1, 2)
        arr[:, 1] -= 1.0                      # the Alembic UVs sit one tile up
        uv.foreach_set('uv', arr.reshape(-1))
        out[o.name] = me
    for o in list(bpy.data.objects): bpy.data.objects.remove(o)
    return out


# --------------------------------------------------------------------------------------------------
def make_textures(out_tex):
    from PIL import Image, ImageFilter
    os.makedirs(out_tex, exist_ok=True)
    alb = Image.open(os.path.join(TEX, 'body_albedo.jpg')).convert('RGB')
    a = Image.open(os.path.join(TEX, 'body_alpha.jpg')).convert('L').resize(alb.size)
    rgba = alb.copy(); rgba.putalpha(a)
    p_body = os.path.join(out_tex, 'BloodReaper_Body.png'); rgba.save(p_body, optimize=True)
    # body normal map: high-passed height from the painted albedo
    lum = alb.convert('L')
    h = np.asarray(lum.filter(ImageFilter.GaussianBlur(1.2))).astype(np.float32) / 255
    h -= np.asarray(lum.filter(ImageFilter.GaussianBlur(10))).astype(np.float32) / 255
    gy, gx = np.gradient(h); st = 6.0
    n = np.stack([-gx * st, gy * st, np.ones_like(h)], 2); n /= np.linalg.norm(n, axis=2, keepdims=True)
    p_bn = os.path.join(out_tex, 'BloodReaper_Body_Normal.png')
    Image.fromarray(((n * 0.5 + 0.5) * 255).astype(np.uint8)).save(p_bn, optimize=True)
    p_sc = os.path.join(out_tex, 'BloodReaper_Scythe.png'); Image.open(os.path.join(TEX, 'scythe_albedo.jpg')).convert('RGB').save(p_sc, optimize=True)
    p_sn = os.path.join(out_tex, 'BloodReaper_Scythe_Normal.png'); Image.open(os.path.join(TEX, 'scythe_normal.jpg')).convert('RGB').save(p_sn, optimize=True)
    p_em = os.path.join(out_tex, 'BloodReaper_Body_Emission.png'); Image.open(os.path.join(TEX, 'body_emission.jpg')).convert('RGB').save(p_em, optimize=True)
    return dict(body=p_body, body_n=p_bn, scythe=p_sc, scythe_n=p_sn, emit=p_em)


def make_material(name, color, normal=None, alpha=False, emission=None, emit_strength=0.0, rough=0.6, metal=0.0):
    m = bpy.data.materials.new(name)
    try: m.use_nodes = True
    except Exception: pass
    nt = m.node_tree; nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial'); p = nt.nodes.new('ShaderNodeBsdfPrincipled')
    nt.links.new(p.outputs[0], out.inputs[0])
    p.inputs['Roughness'].default_value = rough; p.inputs['Metallic'].default_value = metal
    if isinstance(color, str):
        t = nt.nodes.new('ShaderNodeTexImage'); t.image = bpy.data.images.load(color)
        nt.links.new(t.outputs['Color'], p.inputs['Base Color'])
        if alpha:
            nt.links.new(t.outputs['Alpha'], p.inputs['Alpha'])
            try: m.surface_render_method = 'DITHERED'
            except Exception: pass
    else:
        p.inputs['Base Color'].default_value = color
    if normal:
        tn = nt.nodes.new('ShaderNodeTexImage'); tn.image = bpy.data.images.load(normal); tn.image.colorspace_settings.name = 'Non-Color'
        nm = nt.nodes.new('ShaderNodeNormalMap'); nt.links.new(tn.outputs[0], nm.inputs['Color']); nt.links.new(nm.outputs[0], p.inputs['Normal'])
    if emission is not None:
        if isinstance(emission, str):
            te = nt.nodes.new('ShaderNodeTexImage'); te.image = bpy.data.images.load(emission)
            nt.links.new(te.outputs[0], p.inputs['Emission Color'])
        else:
            p.inputs['Emission Color'].default_value = emission
        p.inputs['Emission Strength'].default_value = emit_strength
    return m


def assign_material(o, m):
    o.data.materials.clear(); o.data.materials.append(m)
    for p in o.data.polygons: p.material_index = 0


def make_eyes(ao):
    """two slanted, glowing crimson eyes floating just in front of the hood's inner surface, skinned to Head"""
    import bmesh
    from mathutils import Euler
    bm = bmesh.new()
    for side in (-1, 1):
        g = bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=10, radius=1.0)
        vs = g['verts']
        M = (Matrix.Translation(Vector((side * 0.052, -0.318, 3.805))) @
             Euler((0.0, side * 0.0, side * -0.30)).to_matrix().to_4x4() @
             Matrix.Diagonal(Vector((0.026, 0.008, 0.011, 1.0))))
        bmesh.ops.transform(bm, matrix=M, verts=vs)
    me = bpy.data.meshes.new('BloodReaper_Eyes'); bm.to_mesh(me); bm.free()
    if not me.uv_layers: me.uv_layers.new(name='UVMap')
    for p in me.polygons: p.use_smooth = True
    o = bpy.data.objects.new('BloodReaper_Eyes', me); bpy.context.scene.collection.objects.link(o); o.parent = ao
    md = o.modifiers.new('Armature', 'ARMATURE'); md.object = ao
    g = o.vertex_groups.new(name='Head'); g.add(list(range(len(me.vertices))), 1.0, 'REPLACE')
    return o


def build_armature(rig, extra):
    """extra: list of (name, parent, E-frame 4x4 in model units, length) for non-biped bones"""
    ad = bpy.data.armatures.new('BloodReaper_Rig')
    ao = bpy.data.objects.new('BloodReaper', ad)
    bpy.context.scene.collection.objects.link(ao)
    bpy.context.view_layer.objects.active = ao
    bpy.ops.object.mode_set(mode='EDIT')
    eb = ad.edit_bones
    root = eb.new('Root'); root.head = (0, 0, 0); root.tail = (0, 0.3, 0)
    E = rig.E
    lengths = {}
    for i, n in enumerate(rig.names):
        ch = [j for j, p in enumerate(rig.parents) if p == n]
        if ch:
            L = np.mean([np.linalg.norm(E[j][:3, 3] - E[i][:3, 3]) for j in ch])
        else:
            p = rig.parents[i]
            L = 0.6 * np.linalg.norm(E[i][:3, 3] - E[rig.idx[p]][:3, 3]) if p else 5.0
        lengths[n] = max(L, 0.6)
    allb = [(n, rig.parents[i] or 'Root', E[i], lengths[n]) for i, n in enumerate(rig.names)] + list(extra)
    for n, p, Em, L in allb:
        b = eb.new(n)
        b.head = (0, 0, 0); b.tail = (0, L * K, 0)      # non-zero length first, or .matrix loses the orientation
        R = Em @ RR.Q
        M = R.copy(); M[:3, 3] = M[:3, 3] * K
        b.matrix = Matrix(M.tolist())
        b.length = L * K
    for n, p, Em, L in allb:
        eb[n].parent = eb[p]
    bpy.ops.object.mode_set(mode='OBJECT')
    for pb in ao.pose.bones: pb.rotation_mode = 'QUATERNION'
    return ao


def skin(ao, me, name, idx, w, bone_names, parent_obj):
    o = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(o)
    o.parent = ao
    md = o.modifiers.new('Armature', 'ARMATURE'); md.object = ao
    groups = {}
    for v in range(len(me.vertices)):
        for k in range(idx.shape[1]):
            if w[v, k] <= 1e-4: continue
            bn = bone_names[idx[v, k]]
            if bn not in groups: groups[bn] = o.vertex_groups.new(name=bn)
            groups[bn].add([v], float(w[v, k]), 'ADD')
    return o


# --------------------------------------------------------------------------------------------------
def bake(ao, rig, frames_W, extra_W=None):
    """frames_W: list of (F) arrays of biped world E-matrices (model units); extra_W: list of {name: 4x4 E-frame}"""
    sc = bpy.context.scene
    sc.render.fps = FPS; sc.frame_start = 0; sc.frame_end = len(frames_W) - 1
    rest = {b.name: np.array(b.matrix_local) for b in ao.data.bones}
    par = {b.name: (b.parent.name if b.parent else None) for b in ao.data.bones}
    order = [b.name for b in ao.data.bones]
    keys = {n: [] for n in order}
    for fi, W in enumerate(frames_W):
        P = {'Root': rest['Root']}
        for i, n in enumerate(rig.names):
            M = W[i] @ RR.Q; M = M.copy(); M[:3, 3] *= K; P[n] = M
        if extra_W:
            for n, Wm in extra_W[fi].items():
                M = Wm @ RR.Q; M = M.copy(); M[:3, 3] *= K; P[n] = M
        for n in order:
            p = par[n]
            if p is None:
                basis = np.linalg.inv(rest[n]) @ P[n]
            else:
                basis = np.linalg.inv(np.linalg.inv(rest[p]) @ rest[n]) @ np.linalg.inv(P[p]) @ P[n]
            keys[n].append(basis)
    ao.animation_data_create()
    act = bpy.data.actions.new('BloodReaper_GoalEffect'); ao.animation_data.action = act
    fr = np.arange(len(frames_W), dtype=float)
    for n in order:
        locs = np.array([M[:3, 3] for M in keys[n]])
        qs = []; prev = None
        for M in keys[n]:
            q = Matrix(M[:3, :3].tolist()).to_quaternion().normalized()
            if prev is not None and prev.dot(q) < 0: q.negate()
            qs.append([q.w, q.x, q.y, q.z]); prev = q
        qs = np.array(qs)
        for path, arr in ((f'pose.bones["{n}"].location', locs), (f'pose.bones["{n}"].rotation_quaternion', qs)):
            for c in range(arr.shape[1]):
                fc = act.fcurve_ensure_for_datablock(ao, path, index=c, group_name=n)
                fc.keyframe_points.add(len(fr))
                co = np.empty(len(fr) * 2); co[0::2] = fr; co[1::2] = arr[:, c]
                fc.keyframe_points.foreach_set('co', co)
                fc.keyframe_points.foreach_set('interpolation', [1] * len(fr))
                fc.update()
    return act


# --------------------------------------------------------------------------------------------------
def main():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    out = os.path.abspath(argv[0]); mode = argv[1] if len(argv) > 1 else 'anim'
    os.makedirs(out, exist_ok=True)
    rig = RR.Rig()
    meshes = import_static(rig)
    Wt = np.load(os.path.join(RR.WORK, 'weights.npz'))
    ao = build_armature(rig, [])
    objs = {}
    for mname, me in meshes.items():
        if mname.startswith('Object001'): continue
        objs[mname] = skin(ao, me, mname.replace('_trans_offset', ''), Wt['idx_' + mname], Wt['w_' + mname], rig.names, ao)
    if mode == 'replay':
        F = list(range(30, 97))
        bake(ao, rig, [rig.BM[f] for f in F])
        D = rig.D
        sc = bpy.context.scene
        for fi, f in enumerate(F[::6]):
            sc.frame_set(F.index(f)); dg = bpy.context.evaluated_depsgraph_get()
            errs = []
            for mname, o in objs.items():
                e = o.evaluated_get(dg); me = e.to_mesh()
                V = np.array([v.co[:] for v in me.vertices]) / K; e.to_mesh_clear()
                ref = D['V_' + mname][f] - rig.off
                errs.append((mname[:10], float(np.median(np.linalg.norm(V - ref, axis=1))), float(np.linalg.norm(V - ref, axis=1).max())))
            log('frame', f, ' '.join(f'{a}:{b:.2f}/{c:.1f}' for a, b, c in errs))
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out, 'replay.blend'))
        return
    # ---- the goal-effect animation ----
    import choreo_reaper as C
    solver = C.Solver()
    frames_W, extra_W, diags = [], [], []
    for f in range(C.NFRAMES):
        W, ex, dg = solver.pose(f / C.FPS, use_prev=True)
        frames_W.append(W); extra_W.append(ex); diags.append(dg)
    np.save(os.path.join(out, 'diag.npy'), np.array(diags, dtype=object), allow_pickle=True)
    # scythe: own bone, rigid skin
    import cloth as CL
    cl = CL.Cloth(rig)
    cloth_W = cl.simulate(frames_W, fps=C.FPS)
    for fi in range(len(extra_W)): extra_W[fi].update(cloth_W[fi])
    S0 = solver.S0
    ao_extra = [('Scythe', 'Root', S0, 60.0)] + cl.bones()
    bpy.data.objects.remove(ao)
    for o in list(objs.values()): bpy.data.objects.remove(o)
    ao = build_armature(rig, ao_extra)
    objs = {}
    for mname, me in meshes.items():
        if mname.startswith('Object001'):
            o = bpy.data.objects.new('Scythe', me); bpy.context.scene.collection.objects.link(o); o.parent = ao
            md = o.modifiers.new('Armature', 'ARMATURE'); md.object = ao
            g = o.vertex_groups.new(name='Scythe'); g.add(list(range(len(me.vertices))), 1.0, 'REPLACE')
            objs[mname] = o
        elif mname == 'Object002_trans_offset':
            objs[mname] = skin(ao, me, 'RobeSkirt', cl.robe_idx, cl.robe_w, cl.robe_names, ao)
        elif mname in CL.TRINKETS:
            bn = CL.TRINKETS[mname]
            o = bpy.data.objects.new(bn, me); bpy.context.scene.collection.objects.link(o); o.parent = ao
            md = o.modifiers.new('Armature', 'ARMATURE'); md.object = ao
            g = o.vertex_groups.new(name=bn); g.add(list(range(len(me.vertices))), 1.0, 'REPLACE')
            objs[mname] = o
        else:
            objs[mname] = skin(ao, me, mname.replace('_trans_offset', ''), Wt['idx_' + mname], Wt['w_' + mname], rig.names, ao)
    eyes = make_eyes(ao)
    tex = make_textures(os.path.join(out, 'textures'))
    m_body = make_material('BloodReaper_Body', tex['body'], tex['body_n'], alpha=True, emission=tex['emit'], emit_strength=4.0, rough=0.65)
    m_scy = make_material('BloodReaper_Scythe', tex['scythe'], tex['scythe_n'], rough=0.45, metal=0.35)
    for mname, o in objs.items():
        assign_material(o, m_scy if mname.startswith('Object001') else m_body)
    from PIL import Image
    p_eye = os.path.join(out, 'textures', 'BloodReaper_EyeGlow.png')
    Image.new('RGB', (64, 64), (255, 18, 8)).save(p_eye)
    m_eye = make_material("BloodReaper_EyeGlow", p_eye, emission=(1.0, 0.03, 0.01, 1), emit_strength=18.0, rough=0.2)
    assign_material(eyes, m_eye)
    bake(ao, rig, frames_W, extra_W)
    bpy.context.scene.frame_set(0)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out, 'BloodReaper.blend'))
    log('saved', os.path.join(out, 'BloodReaper.blend'))
    if os.environ.get('NO_FBX'): return
    bpy.ops.object.select_all(action='DESELECT')
    ao.select_set(True)
    for o in list(objs.values()) + [eyes]: o.select_set(True)
    bpy.context.view_layer.objects.active = ao
    path = os.path.join(out, 'BloodReaper_MythicGoal.fbx')
    bpy.ops.export_scene.fbx(
        filepath=path, use_selection=True, object_types={'ARMATURE', 'MESH'}, add_leaf_bones=False,
        apply_scale_options='FBX_SCALE_NONE', apply_unit_scale=True, mesh_smooth_type='FACE',
        use_mesh_modifiers=False, use_armature_deform_only=False, primary_bone_axis='Y', secondary_bone_axis='X',
        bake_anim=True, bake_anim_use_all_bones=True, bake_anim_use_nla_strips=False, bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True, bake_anim_step=1.0, bake_anim_simplify_factor=0.0,
        path_mode='COPY', embed_textures=True)
    log('fbx', path, '%.2f MB' % (os.path.getsize(path) / 1e6))


if __name__ == '__main__':
    main()
