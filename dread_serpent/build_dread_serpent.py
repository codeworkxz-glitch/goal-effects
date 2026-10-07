"""
Dread Serpent – Legendary basketball goal effect.

    python build_dread_serpent.py <source.fbx> <textures_dir> <out_dir>

* imports the source serpent (mesh, UVs, textures), drops the water plane and the old 18-bone rig
* re-rigs it with a dense spine (61 x 2-unit bones) + head + jaw + the original arm/finger bones;
  body weights are recomputed from the rest-pose arclength (smooth two-bone blending), every spike /
  mane / tail-fin island is skinned rigidly to its root so it never shears, head/jaw/arm weights are kept
* drives the spine follow-the-leader along the per-frame centre line from choreo.py, adds head look,
  jaw, arm/claw animation and bakes everything at 60 fps
* exports DreadSerpent_LegendaryGoal.fbx (embedded textures) and a .blend
"""
import bpy, bmesh, sys, os, math
import numpy as np
from mathutils import Matrix, Vector, Quaternion

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import choreo as C

K = C.METRES_PER_UNIT
L_BONE = 2.0
N_CHAIN = int(round(C.BODY_LEN / L_BONE))          # 61
U_J = C.U_HEAD + L_BONE * np.arange(N_CHAIN + 1)   # joint rest positions (unit Y)

ARM_MAP = {}
for side in ('R', 'L'):
    for b in ('Shoulder', 'Upperarm', 'Forearm', 'Middlefinger 1', 'Middlefinger 2', 'Middlefinger 3',
              'Pinky 1', 'Pinky 2', 'Pinky 3', 'Pointer 1', 'Pointer 2', 'Pointer 3', 'Thumb 1', 'Thumb 2', 'Thumb 3'):
        ARM_MAP[f'{b} {side}'] = f'{b.replace(" ", "")}_{side}'
HEAD_GROUPS = ('Head', 'Jawbone')
JAW_GROUPS = ('Jaw',)


def log(*a):
    print('[serpent]', *a, flush=True)


# --------------------------------------------------------------------------------------------------
# import + cleanup
# --------------------------------------------------------------------------------------------------
def import_source(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=path)
    for nm in ('Plane.001',):
        if nm in bpy.data.objects: bpy.data.objects.remove(bpy.data.objects[nm])
    for a in list(bpy.data.actions): bpy.data.actions.remove(a)
    return bpy.data.objects['Armature']


def mesh_islands(me):
    bm = bmesh.new(); bm.from_mesh(me); bm.verts.ensure_lookup_table()
    lab = -np.ones(len(bm.verts), int); n = 0
    for v in bm.verts:
        if lab[v.index] >= 0: continue
        st = [v]; lab[v.index] = n
        while st:
            x = st.pop()
            for e in x.link_edges:
                w = e.other_vert(x)
                if lab[w.index] < 0: lab[w.index] = n; st.append(w)
        n += 1
    bm.free()
    return lab, n


def chain_weights(u):
    """Tent weights between bone centres. Returns (k0, w0, k1, w1)."""
    c = (u - C.U_HEAD) / L_BONE - 0.5
    c = np.clip(c, 0.0, N_CHAIN - 1.0)
    k0 = np.floor(c).astype(int); f = c - k0
    k1 = np.minimum(k0 + 1, N_CHAIN - 1)
    return k0, 1 - f, k1, f


def compute_weights(o):
    """Returns world coords (units) and a list of {group: weight} per vertex."""
    me = o.data
    mw = o.matrix_world.copy()
    me.transform(mw)
    if mw.determinant() < 0: me.flip_normals()
    o.matrix_world = Matrix.Identity(4)
    nv = len(me.vertices)
    co = np.zeros(nv * 3); me.vertices.foreach_get('co', co); co = co.reshape(-1, 3)
    names = {g.index: g.name for g in o.vertex_groups}
    jaw = np.zeros(nv); arm = [dict() for _ in range(nv)]; armsum = np.zeros(nv); tot = np.zeros(nv)
    head0 = np.zeros(nv)
    for v in me.vertices:
        for g in v.groups:
            n = names[g.group]; w = g.weight
            if w <= 0: continue
            tot[v.index] += w
            if n in JAW_GROUPS: jaw[v.index] += w
            elif n in ARM_MAP: arm[v.index][ARM_MAP[n]] = arm[v.index].get(ARM_MAP[n], 0) + w; armsum[v.index] += w
            elif n in HEAD_GROUPS: head0[v.index] += w
    tot = np.where(tot > 0, tot, 1.0)
    jaw /= tot; armsum /= tot; head0 /= tot
    rest = np.clip(1 - jaw - armsum, 0, 1)
    y, z = co[:, 1], co[:, 2]
    u_eff = y - 0.6 * np.clip(z - 2.0, 0, 6.0) * (y < 12)
    headness = 1 - C.smoothstep(2.5, 6.5, u_eff)
    headness = np.maximum(headness, np.where(y < 8, head0 / np.maximum(rest, 1e-6), 0) * 0.0)
    k0, w0, k1, w1 = chain_weights(np.clip(y, C.U_HEAD, C.U_TAIL))
    W = []
    for i in range(nv):
        d = {}
        if jaw[i] > 0: d['Jaw'] = jaw[i]
        for a, w in arm[i].items(): d[a] = w / tot[i]
        h = rest[i] * headness[i]
        if h > 0: d['Head'] = d.get('Head', 0) + h
        b = rest[i] - h
        if b > 0:
            for k, w in ((k0[i], w0[i]), (k1[i], w1[i])):
                if w * b > 1e-5:
                    nm = f'Spine_{k:02d}'; d[nm] = d.get(nm, 0) + w * b
        W.append(d)
    return co, W


def rigidify_islands(o, co, W):
    lab, n = mesh_islands(o.data)
    for isl in range(n):
        idx = np.where(lab == isl)[0]
        if len(idx) < 2: continue
        P = co[idx]
        r = np.sqrt(P[:, 0] ** 2 + P[:, 2] ** 2)
        # root of a spike = verts nearest the body axis
        base = idx[r <= np.quantile(r, 0.2) + 1e-6]
        acc = {}
        for i in base:
            for g, w in W[i].items(): acc[g] = acc.get(g, 0) + w
        s = sum(acc.values())
        acc = {g: w / s for g, w in acc.items()}
        for i in idx: W[i] = dict(acc)


def limit4(W):
    out = []
    for d in W:
        it = sorted(d.items(), key=lambda kv: -kv[1])[:4]
        s = sum(w for _, w in it) or 1.0
        out.append({g: w / s for g, w in it if w / s > 1e-4})
    return out


# --------------------------------------------------------------------------------------------------
# new armature
# --------------------------------------------------------------------------------------------------
def build_armature(arm0):
    arm_rest = {}
    for b in arm0.data.bones:
        if b.name in ARM_MAP:
            arm_rest[ARM_MAP[b.name]] = (b.matrix_local.copy(), b.length, ARM_MAP.get(b.parent.name) if b.parent else None)
    ad = bpy.data.armatures.new('DreadSerpent_Rig')
    ao = bpy.data.objects.new('DreadSerpent', ad)
    bpy.context.scene.collection.objects.link(ao)
    bpy.context.view_layer.objects.active = ao
    bpy.ops.object.mode_set(mode='EDIT')
    eb = ad.edit_bones
    root = eb.new('Root'); root.head = (0, 0, 0); root.tail = (0, 0, 0.3); root.roll = 0
    prev = None
    for k in range(N_CHAIN):
        b = eb.new(f'Spine_{k:02d}')
        b.head = (0, U_J[k] * K, 0); b.tail = (0, U_J[k + 1] * K, 0)
        b.align_roll(Vector((0, 0, 1)))
        if prev is None: b.parent = root
        else: b.parent = prev; b.use_connect = True
        prev = b
    h = eb.new('Head'); h.head = (0, C.U_HEAD * K, 0); h.tail = (0, -6.0 * K, 0); h.align_roll(Vector((0, 0, 1)))
    h.parent = eb['Spine_00']
    j = eb.new('Jaw'); j.head = Vector((0.0, 1.65, -0.26)) * K; j.tail = Vector((0.0, -2.86, -2.58)) * K
    j.align_roll(Vector((0, 0, 1))); j.parent = h
    shoulder_k = int(np.clip((17.0 - C.U_HEAD) // L_BONE, 0, N_CHAIN - 1))
    for nm in sorted(arm_rest, key=lambda n: ('Shoulder' not in n, 'Upperarm' not in n, 'Forearm' not in n, n)):
        M, ln, par = arm_rest[nm]
        b = eb.new(nm)
        M = M.copy(); M.translation = M.translation * K
        b.length = ln * K
        b.matrix = M
        b.length = ln * K
    for nm, (M, ln, par) in arm_rest.items():
        eb[nm].parent = eb[par] if par else eb[f'Spine_{shoulder_k:02d}']
    bpy.ops.object.mode_set(mode='OBJECT')
    for pb in ao.pose.bones: pb.rotation_mode = 'QUATERNION'
    return ao


# --------------------------------------------------------------------------------------------------
# materials
# --------------------------------------------------------------------------------------------------
def tex_node(mat, img, loc=(-500, 300)):
    nt = mat.node_tree
    t = nt.nodes.new('ShaderNodeTexImage'); t.image = img; t.location = loc
    return t


def setup_materials(tex_dir, out_tex):
    from PIL import Image
    os.makedirs(out_tex, exist_ok=True)

    def conv(src, dst, size=None, fmt='PNG', q=93):
        im = Image.open(os.path.join(tex_dir, src)).convert('RGB')
        if size: im = im.resize((size, size), Image.LANCZOS)
        p = os.path.join(out_tex, dst); im.save(p, fmt, quality=q) if fmt == 'JPEG' else im.save(p, fmt, optimize=True)
        return bpy.data.images.load(p)

    body = conv('body_atlas.webp', 'DreadSerpent_Body.png')
    # scale relief: tangent-space normal map from the painted atlas (bright scale highlights = raised)
    from PIL import ImageFilter
    lum = Image.open(os.path.join(tex_dir, 'body_atlas.webp')).convert('L')
    h = np.asarray(lum.filter(ImageFilter.GaussianBlur(1.6))).astype(np.float32) / 255.0
    h = h - np.asarray(lum.filter(ImageFilter.GaussianBlur(14))).astype(np.float32) / 255.0   # high-pass
    gy, gx = np.gradient(h)
    st = 9.0
    n = np.stack([-gx * st, gy * st, np.ones_like(h)], 2)
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    p = os.path.join(out_tex, 'DreadSerpent_Body_Normal.png')
    Image.fromarray(((n * 0.5 + 0.5) * 255).astype(np.uint8)).save(p, optimize=True)
    body_n = bpy.data.images.load(p); body_n.colorspace_settings.name = 'Non-Color'
    teeth = conv('teeth.webp', 'DreadSerpent_Fangs.png', 1024)
    # eyes: the painted eye pushed to a hot ember colour so it reads as glowing even without emission
    # support (dark pupils kept); the same map drives emission where the engine supports it
    im = Image.open(os.path.join(tex_dir, 'eye.webp')).convert('RGB').resize((1024, 1024), Image.LANCZOS)
    a = np.asarray(im).astype(np.float32) / 255.0
    lum = a.mean(axis=2, keepdims=True)
    hot = np.clip(np.concatenate([lum * 1.55 + 0.12, lum * 0.62, lum * 0.10], 2), 0, 1)
    hot = hot ** np.array([0.8, 1.15, 1.4])
    p = os.path.join(out_tex, 'DreadSerpent_EyeGlow.png')
    Image.fromarray((np.clip(hot, 0, 1) * 255).astype(np.uint8)).save(p, optimize=True)
    eye = bpy.data.images.load(p)

    def principled(name, base=None, img=None, rough=0.5, metal=0.0, emit=None, emit_img=None, strength=0.0, spec=0.5, nmap=None):
        m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
        m.use_nodes = True if hasattr(m, 'use_nodes') else None
        nt = m.node_tree; nt.nodes.clear()
        out = nt.nodes.new('ShaderNodeOutputMaterial'); out.location = (300, 0)
        p = nt.nodes.new('ShaderNodeBsdfPrincipled'); p.location = (0, 0)
        nt.links.new(p.outputs[0], out.inputs[0])
        if base is not None: p.inputs['Base Color'].default_value = base
        if img is not None:
            t = tex_node(m, img); nt.links.new(t.outputs[0], p.inputs['Base Color'])
        p.inputs['Roughness'].default_value = rough; p.inputs['Metallic'].default_value = metal
        if emit is not None:
            p.inputs['Emission Color'].default_value = emit; p.inputs['Emission Strength'].default_value = strength
        if emit_img is not None:
            t2 = tex_node(m, emit_img, (-500, -100)); nt.links.new(t2.outputs[0], p.inputs['Emission Color'])
            p.inputs['Emission Strength'].default_value = strength
        if nmap is not None:
            tn = tex_node(m, nmap, (-700, -350))
            nm = nt.nodes.new('ShaderNodeNormalMap'); nm.location = (-300, -350); nm.inputs['Strength'].default_value = 1.0
            nt.links.new(tn.outputs[0], nm.inputs['Color']); nt.links.new(nm.outputs[0], p.inputs['Normal'])
        m.diffuse_color = base if base is not None else (0.5, 0.5, 0.5, 1)
        return m
    mats = dict(
        skin=principled('DreadSerpent_Scales', img=body, rough=0.42, metal=0.15, base=(0.1, 0.12, 0.12, 1), nmap=body_n),
        mouth=principled('DreadSerpent_Maw', base=(0.035, 0.004, 0.004, 1), rough=0.35),
        spikes=principled('DreadSerpent_Spikes', base=(0.018, 0.02, 0.024, 1), rough=0.3, metal=0.35),
        fangs=principled('DreadSerpent_Fangs', img=teeth, rough=0.3, base=(0.9, 0.85, 0.7, 1)),
        eyes=principled('DreadSerpent_EyeGlow', img=eye, emit_img=eye, strength=3.5, rough=0.15, base=(1, 0.35, 0.05, 1)),
    )
    return mats


# --------------------------------------------------------------------------------------------------
# geometry assembly
# --------------------------------------------------------------------------------------------------
GROUPS = {  # source object -> output part
    'Plane': 'DreadSerpent_Body', 'NurbsPath': 'DreadSerpent_Spikes', 'NurbsPath.015': 'DreadSerpent_Spikes',
    'NurbsPath.030': 'DreadSerpent_Spikes', 'NurbsPath.027': 'DreadSerpent_Mane',
    'Cube.001': 'DreadSerpent_Fangs', 'Cube.003': 'DreadSerpent_Fangs', 'Sphere': 'DreadSerpent_Eyes'}


def assemble(arm0, ao, mats):
    parts = {}
    for src, dst in GROUPS.items():
        o = bpy.data.objects[src]
        o.parent = None
        for m in list(o.modifiers): o.modifiers.remove(m)
        co, W = compute_weights(o)
        if src.startswith('NurbsPath') or src.startswith('Cube') or src == 'Sphere':
            rigidify_islands(o, co, W)
        W = limit4(W)
        for g in list(o.vertex_groups): o.vertex_groups.remove(g)
        vg = {}
        for i, d in enumerate(W):
            for g, w in d.items():
                if g not in vg: vg[g] = o.vertex_groups.new(name=g)
                vg[g].add([i], w, 'REPLACE')
        o.data.transform(Matrix.Scale(K, 4))
        # materials
        newm = []
        for s in o.material_slots:
            nm = s.material.name if s.material else ''
            newm.append({'Material': mats['skin'], 'Material.006': mats['mouth'], 'Material.001': mats['spikes'],
                         'Material.004': mats['fangs'], 'Material.003': mats['eyes']}.get(nm, mats['skin']))
        for i, m in enumerate(newm): o.material_slots[i].material = m
        parts.setdefault(dst, []).append(o)
    bpy.data.objects.remove(arm0)
    out = []
    for dst, objs in parts.items():
        bpy.ops.object.select_all(action='DESELECT')
        for o in objs: o.select_set(True)
        bpy.context.view_layer.objects.active = objs[0]
        if len(objs) > 1: bpy.ops.object.join()
        o = bpy.context.view_layer.objects.active
        o.name = dst; o.data.name = dst
        out.append(o)
    return out


def split_large(objs, max_tris=19000):
    """Roblox-style importers cap triangles per mesh: cut big parts into slabs along the body."""
    res = []
    for o in objs:
        me = o.data
        ntri = sum(len(p.vertices) - 2 for p in me.polygons)
        if ntri <= max_tris: res.append(o); continue
        nparts = int(math.ceil(ntri / max_tris))
        # polygons sorted by centre Y (rest pose); islands kept whole
        lab, nisl = mesh_islands(me)
        py = np.array([p.center.y for p in me.polygons]); pt = np.array([len(p.vertices) - 2 for p in me.polygons])
        pisl = np.array([lab[p.vertices[0]] for p in me.polygons])
        if nisl > 1:
            iy = np.zeros(nisl); np.add.at(iy, pisl, py); cnt = np.bincount(pisl, minlength=nisl); iy /= np.maximum(cnt, 1)
            key = iy[pisl]
        else:
            key = py
        order = np.argsort(key, kind='stable'); cum = np.cumsum(pt[order])
        cuts = [key[order][np.searchsorted(cum, ntri * j / nparts)] for j in range(1, nparts)]
        part_of = np.searchsorted(np.array(cuts), key, side='right')
        base = o.name
        for j in range(nparts - 1, 0, -1):
            bpy.ops.object.select_all(action='DESELECT'); o.select_set(True); bpy.context.view_layer.objects.active = o
            bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.select_all(action='DESELECT')
            bpy.ops.object.mode_set(mode='OBJECT')
            cur = np.array([p.center.y for p in o.data.polygons])
            if nisl > 1:
                lab2, n2 = mesh_islands(o.data)
                pis = np.array([lab2[p.vertices[0]] for p in o.data.polygons])
                iy2 = np.zeros(n2); np.add.at(iy2, pis, cur); c2 = np.bincount(pis, minlength=n2); iy2 /= np.maximum(c2, 1)
                kk = iy2[pis]
            else:
                kk = cur
            sel = kk >= cuts[j - 1]
            o.data.polygons.foreach_set('select', sel)
            bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.separate(type='SELECTED'); bpy.ops.object.mode_set(mode='OBJECT')
        parts = [ob for ob in bpy.data.objects if ob.type == 'MESH' and (ob.name == base or ob.name.startswith(base + '.'))]
        parts.sort(key=lambda ob: np.mean([v.co.y for v in ob.data.vertices]))
        for i, ob in enumerate(parts):
            ob.name = f'{base}_{i + 1}'; ob.data.name = ob.name
            # drop empty vertex groups
            used = set(g.group for v in ob.data.vertices for g in v.groups if g.weight > 0)
            unused = [g.name for g in ob.vertex_groups if g.index not in used]
            for nm in unused: ob.vertex_groups.remove(ob.vertex_groups[nm])
            res.append(ob)
    return res


# --------------------------------------------------------------------------------------------------
# animation
# --------------------------------------------------------------------------------------------------
def frame_matrix(Y, Z, pos):
    Y = Y / np.linalg.norm(Y); Z = Z - Y * (Z @ Y); Z /= np.linalg.norm(Z); X = np.cross(Y, Z)
    M = np.eye(4); M[:3, 0] = X; M[:3, 1] = Y; M[:3, 2] = Z; M[:3, 3] = pos
    return M


def rot_about(axis, ang):
    axis = np.asarray(axis, float); axis = axis / np.linalg.norm(axis)
    return np.array(Matrix.Rotation(ang, 3, Vector(axis)))


def jaw_open(t):
    """degrees"""
    k = [(0.00, 6), (0.12, 30), (0.45, 34), (0.80, 14), (1.30, 8), (1.75, 6), (2.10, 10), (2.55, 16),
         (2.85, 17), (3.10, 44), (3.20, 46), (3.45, 40), (3.70, 18), (4.20, 10)]
    ts, vs = zip(*k)
    return float(C.pchip(ts, vs)(min(max(t, 0), 4.2)))


def arm_pose(t):
    """(tuck, spread, reach) weights, each 0..1: tucked while travelling and striking, opened out to the
    sides while staring."""
    ss = C.smootherstep
    spread = ss(C.T_REAR_END - 0.1, C.T_REAR_END + 0.35, t) * (1 - ss(C.T_STRIKE - 0.06, C.T_STRIKE + 0.05, t))
    return float(1 - spread), float(spread), 0.0


def look_weight(t):
    return float(C.smootherstep(C.T_REAR_END - 0.40, C.T_REAR_END + 0.05, t) * (1 - C.smootherstep(C.T_STRIKE, C.T_STRIKE_HIT - 0.04, t)))


def ripple(t, u):
    """lateral travelling wave (head -> tail) used while the body is otherwise still"""
    a = C.smootherstep(C.T_REAR_END - 0.3, C.T_REAR_END + 0.2, t) * (1 - C.smootherstep(C.T_STRIKE - 0.1, C.T_STRIKE + 0.05, t))
    env = C.smoothstep(14.0, 30.0, u) * (0.55 + 1.6 * C.smoothstep(95.0, C.U_TAIL, u))
    return a * env * np.sin(2 * np.pi * (1.15 * t) - 2 * np.pi * u / 38.0)


def solve_frame(t):
    fp = C.frame_path(t)
    P, s, sh = fp['P'], fp['s'], fp['s_head']
    T, D = C.path_frames(P, s, fp['keys'])
    uj = U_J - C.U_HEAD
    X = C.sample_on_path(P, s, sh - uj)
    Tj = C.sample_vec(T, s, sh - uj)
    Dj = C.sample_vec(D, s, sh - uj)
    # ripple: sideways in the body frame, only where the body is free (neck/rise and the hanging tail)
    sj = sh - uj
    free = ((sj > fp['sN0'] + 2.0) | (sj < (s[fp['nG'] - 1] if fp['nG'] > 0 else -1)))
    Bj = np.cross(Tj, Dj)
    amp = ripple(t, U_J) * np.where(free, 1.0, 0.25)
    X = X + amp[:, None] * Bj
    # chain with exact bone lengths
    J = np.zeros_like(X); J[0] = X[0]
    for k in range(1, len(X)):
        d = X[k] - J[k - 1]; J[k] = J[k - 1] + L_BONE * d / max(np.linalg.norm(d), 1e-9)
    mats = {}
    for k in range(N_CHAIN):
        Y = J[k + 1] - J[k]
        Z = Dj[k] + Dj[k + 1]
        mats[f'Spine_{k:02d}'] = frame_matrix(Y, Z, J[k] * K)
    # head: path frame, blended toward the "stare" look while paused
    Th = C.sample_vec(T, s, np.array([sh]))[0]; Dh = C.sample_vec(D, s, np.array([sh]))[0]
    Yh = -(J[1] - J[0]); Yh = 0.5 * nrm(Yh) + 0.5 * Th
    Mh = frame_matrix(Yh, Dh, J[0] * K)
    w = look_weight(t)
    if w > 0:
        # forward, slightly down; during the recoil the head lifts into a roar
        rec = C.smootherstep(C.T_RECOIL, C.T_STRIKE - 0.02, t)
        sway = np.sin(2 * np.pi * 0.9 * (t - C.T_REAR_END)) * (1 - rec)
        F = nrm(np.array([0.06 * sway, -1.0, -0.20 + 0.42 * rec]))
        U = np.array([0.04 * sway, 0.15, 1.0])
        Ml = frame_matrix(F, U, J[0] * K)
        qa = Matrix(Mh[:3, :3].tolist()).to_quaternion(); qb = Matrix(Ml[:3, :3].tolist()).to_quaternion()
        if qa.dot(qb) < 0: qb.negate()
        q = qa.slerp(qb, w)
        Mh[:3, :3] = np.array(q.to_matrix())
    mats['Head'] = Mh
    return mats, fp, J


def nrm(v):
    return v / max(np.linalg.norm(v), 1e-12)


def bake(ao):
    sc = bpy.context.scene
    sc.render.fps = C.FPS
    sc.frame_start = 0; sc.frame_end = C.NFRAMES - 1
    pbs = ao.pose.bones
    rest = {b.name: np.array(b.matrix_local) for b in ao.data.bones}
    par = {b.name: (b.parent.name if b.parent else None) for b in ao.data.bones}
    order = [b.name for b in ao.data.bones]  # parents before children
    jaw_axis_local = np.array([1.0, 0.0, 0.0])
    arm_names = [n for n in order if n.endswith(('_R', '_L'))]
    data = {n: [] for n in order}
    for fi in range(C.NFRAMES):
        t = fi / C.FPS
        mats, fp, J = solve_frame(t)
        arm_world = {}
        tuck, spread, reach = arm_pose(t)
        for n in order:
            p = par[n]
            if n in mats:
                M = mats[n]
                Mp = arm_world[p] if p else np.eye(4)
                Rp = rest[p] if p else np.eye(4)
                basis = np.linalg.inv(np.linalg.inv(Rp) @ rest[n]) @ np.linalg.inv(Mp) @ M
            else:
                basis = np.eye(4)
                if n == 'Jaw':
                    a = math.radians(jaw_open(t))
                    basis[:3, :3] = np.array(Matrix.Rotation(a, 3, 'X'))
                elif n in arm_names:
                    Rb = rest[n][:3, :3]
                    side = 1 if n.endswith('_R') else -1
                    if n.startswith('Upperarm'):
                        # tucked back while travelling, opened to the sides while staring, claws thrown
                        # forward with the strike
                        R = rot_about([1, 0, 0], math.radians(55 * tuck - 8 * spread + 10 * reach)) @ \
                            rot_about([0, 1, 0], math.radians(-side * 24 * spread - side * 26 * reach))
                        basis[:3, :3] = Rb.T @ R @ Rb
                    elif n.startswith('Forearm'):
                        ang = math.radians(30 * tuck - 10 * spread)
                        basis[:3, :3] = Rb.T @ rot_about([1, 0, 0], ang) @ Rb
                    elif n.startswith('Shoulder'):
                        basis[:3, :3] = Rb.T @ rot_about([0, 0, 1], math.radians(side * 6 * (spread + reach))) @ Rb
                    else:
                        # claws: curl when tucked, splay when menacing / striking
                        curl = 14 * tuck - 10 * spread - 8 * reach + 4 * math.sin(2 * math.pi * 1.7 * t) * spread
                        basis[:3, :3] = np.array(Matrix.Rotation(math.radians(curl), 3, 'X'))
                Mp = arm_world[p] if p else np.eye(4)
                Rp = rest[p] if p else np.eye(4)
                M = Mp @ (np.linalg.inv(Rp) @ rest[n]) @ basis
                aim_w = tuck * C.smootherstep(C.T_STRIKE - 0.05, C.T_STRIKE + 0.02, t) * (1 - C.smootherstep(C.T_STRIKE_HIT + 0.08, C.T_STRIKE_HIT + 0.2, t))
                if aim_w > 0 and n.startswith(('Upperarm', 'Forearm')):
                    # during the whip the spine bends hard behind the shoulders: aim the tucked arm segment
                    # at a point under the belly further down the (bent) spine so it follows the curve
                    side = 1 if n.endswith('_R') else -1
                    k, ven, lat = (9, 4.4, 3.8) if n.startswith('Upperarm') else (12, 4.8, 3.6)
                    S_ = mats[f'Spine_{k:02d}']
                    target = S_[:3, 3] - S_[:3, 2] * ven * K + S_[:3, 0] * side * lat * K
                    cur = M[:3, 1] / np.linalg.norm(M[:3, 1])
                    want = target - M[:3, 3]; want /= np.linalg.norm(want)
                    q = Vector(cur).rotation_difference(Vector(want))
                    q = Quaternion().slerp(q, aim_w)
                    M = M.copy(); M[:3, :3] = np.array(q.to_matrix()) @ M[:3, :3]
                    basis = np.linalg.inv(np.linalg.inv(Rp) @ rest[n]) @ np.linalg.inv(Mp) @ M
            arm_world[n] = M
            data[n].append(basis)
        if fi % 30 == 0: log('frame', fi, 'phase t=%.2f' % t)
    # write keys
    ao.animation_data_create()
    act = bpy.data.actions.new('DreadSerpent_LegendaryGoal')
    ao.animation_data.action = act
    frames = np.arange(C.NFRAMES, dtype=float)
    for n in order:
        Ms = data[n]
        locs = np.array([M[:3, 3] for M in Ms])
        quats = []
        prev = None
        for M in Ms:
            q = Matrix(M[:3, :3].tolist()).to_quaternion().normalized()
            if prev is not None and prev.dot(q) < 0: q.negate()
            quats.append(q); prev = q
        quats = np.array([[q.w, q.x, q.y, q.z] for q in quats])
        for path, arr in ((f'pose.bones["{n}"].location', locs), (f'pose.bones["{n}"].rotation_quaternion', quats)):
            for idx in range(arr.shape[1]):
                fc = act.fcurve_ensure_for_datablock(ao, path, index=idx, group_name=n)
                fc.keyframe_points.add(len(frames))
                co = np.empty(len(frames) * 2); co[0::2] = frames; co[1::2] = arr[:, idx]
                fc.keyframe_points.foreach_set('co', co)
                fc.keyframe_points.foreach_set('interpolation', [1] * len(frames))  # LINEAR
                fc.update()
    return act


# --------------------------------------------------------------------------------------------------
def main():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    src, tex_dir, out = argv[:3]
    os.makedirs(out, exist_ok=True)
    arm0 = import_source(src)
    mats = setup_materials(tex_dir, os.path.join(out, 'textures'))
    ao = build_armature(arm0)
    objs = assemble(arm0, ao, mats)
    objs = split_large(objs)
    for o in objs:
        o.parent = ao
        md = o.modifiers.new('Armature', 'ARMATURE'); md.object = ao
    for m in list(bpy.data.materials):
        if m.users == 0: bpy.data.materials.remove(m)
    for im in list(bpy.data.images):
        if im.users == 0: bpy.data.images.remove(im)
    log('parts', [(o.name, len(o.data.vertices), sum(len(p.vertices) - 2 for p in o.data.polygons)) for o in objs])
    bake(ao)
    bpy.context.scene.frame_set(0)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out, 'DreadSerpent_LegendaryGoal.blend'))
    log('saved blend')
    export_fbx(ao, objs, os.path.join(out, 'DreadSerpent_LegendaryGoal.fbx'))


def export_fbx(ao, objs, path):
    bpy.ops.object.select_all(action='DESELECT')
    ao.select_set(True)
    for o in objs: o.select_set(True)
    bpy.context.view_layer.objects.active = ao
    bpy.ops.export_scene.fbx(
        filepath=path, use_selection=True, object_types={'ARMATURE', 'MESH'}, add_leaf_bones=False,
        apply_scale_options='FBX_SCALE_NONE', apply_unit_scale=True, mesh_smooth_type='FACE',
        use_mesh_modifiers=False, use_armature_deform_only=False, primary_bone_axis='Y', secondary_bone_axis='X',
        bake_anim=True, bake_anim_use_all_bones=True, bake_anim_use_nla_strips=False, bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True, bake_anim_step=1.0, bake_anim_simplify_factor=float(os.environ.get('SIMPLIFY', '0.0')),
        path_mode='COPY', embed_textures=True)
    log('fbx', path, '%.2f MB' % (os.path.getsize(path) / 1e6))


if __name__ == '__main__':
    main()
