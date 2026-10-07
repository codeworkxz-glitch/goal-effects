"""Collision-avoid pass for the hanging skirt panels / belt cords (run after build_reality_cut.py).

    python post_skirts.py built.blend out.blend

For every skirt chain root a per-frame outward swing is searched so the thighs / greaves never poke through the panels
(scored on the deformed meshes, Viterbi-smoothed in time) and then baked into the root bones' rotation curves."""
import bpy, sys, os, math, bisect
import numpy as np
from mathutils import Matrix, Quaternion
from mathutils.bvhtree import BVHTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clips

src, dst = sys.argv[-2], sys.argv[-1]
bpy.ops.wm.open_mainfile(filepath=src)
arm = bpy.data.objects['wushi']; sc = bpy.context.scene
N = sc.frame_end
bones = arm.data.bones
chain = [b.name for b in bones if 'hujia' in b.name or b.name.startswith('rope')]
roots = [n for n in chain if bones[n].parent is None or bones[n].parent.name not in chain]
members = {r: [n for n in chain if n == r or any(a.name == r for a in bones[n].parent_recursive)] for r in roots}
print('roots', roots)


def outward(r):
    if r.startswith('L_'): return ('z', +1)
    if r.startswith('R_'): return ('z', -1)
    if 'Back' in r: return ('x', +1)
    return ('x', -1)


RR = {r: bones[r].matrix_local.to_3x3() for r in roots}


def q_extra(r, a, b=0.0):
    ax, sg = outward(r)
    R = Matrix.Rotation(sg * a, 3, 'X' if ax == 'x' else 'Z') @ Matrix.Rotation(b, 3, 'Z' if ax == 'x' else 'X')
    return (RR[r].inverted() @ R @ RR[r]).to_quaternion()


for pb in arm.pose.bones: pb.rotation_mode = 'QUATERNION'
armo = bpy.data.objects['armo']
vgs = {i: g.name for i, g in enumerate(armo.vertex_groups)}
chain_polys = {r: [] for r in roots}
for p in armo.data.polygons:
    w = {}
    for vi in p.vertices:
        for g in armo.data.vertices[vi].groups:
            n_ = vgs.get(g.group)
            if n_: w[n_] = w.get(n_, 0) + g.weight
    if not w: continue
    top = max(w, key=w.get)
    for r in roots:
        if top in members[r]: chain_polys[r].append(p.index)
scorer = clips.Scorer()
ANG = [0.0, 0.1, 0.2, 0.32, 0.45, 0.6]
SEC = [-0.35, 0.0, 0.35]
STATES = [(a, b) for a in ANG for b in SEC]
step = 4
idx = list(range(0, N, step))
if idx[-1] != N - 1: idx.append(N - 1)
qbase = {}
cost = {r: [] for r in roots}
for fi in idx:
    sc.frame_set(fi + 1)
    qb = {r: arm.pose.bones[r].rotation_quaternion.copy() for r in roots}
    row = {r: [] for r in roots}
    for (a, b_) in STATES:
        for r in roots: arm.pose.bones[r].rotation_quaternion = qb[r] @ q_extra(r, a, b_)
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        T = scorer.trees(dg)
        ob = armo.evaluated_get(dg); me = ob.to_mesh(); mw = ob.matrix_world
        verts = [mw @ v.co for v in me.vertices]; polys = [tuple(p.vertices) for p in me.polygons]
        for r in roots:
            if not chain_polys[r]: row[r].append(0.0); continue
            tr = BVHTree.FromPolygons(verts, [polys[i] for i in chain_polys[r]], epsilon=0.0)
            row[r].append(sum(len(tr.overlap(t_)) for (_, t_) in T.get('leg', [])) + 60.0 * a + 40.0 * abs(b_))
        ob.to_mesh_clear()
    for r in roots: cost[r].append(row[r])
    print('frame', fi, flush=True)
K = len(STATES)
trans = np.array([[8.0 * abs(STATES[j][0] - STATES[jp][0]) / 0.1 + 8.0 * abs(STATES[j][1] - STATES[jp][1]) / 0.35 for jp in range(K)] for j in range(K)])
extra = {}
for r in roots:
    C = np.array(cost[r]); best = C[0].copy(); back = []
    for f in range(1, len(idx)):
        tot = best[None, :] + trans; m = tot.argmin(1); best = C[f] + tot[np.arange(K), m]; back.append(m)
    j = int(best.argmin()); path = [j]
    for bk in reversed(back): j = int(bk[j]); path.append(j)
    path.reverse()
    a_s = [STATES[j] for j in path]
    fa, fb = [], []
    for i in range(N):
        f = min(max(bisect.bisect_right(idx, i) - 1, 0), len(idx) - 2)
        u = (i - idx[f]) / float(idx[f + 1] - idx[f])
        fa.append(a_s[f][0] + (a_s[f + 1][0] - a_s[f][0]) * u); fb.append(a_s[f][1] + (a_s[f + 1][1] - a_s[f][1]) * u)
    sm = lambda v: [sum(v[min(max(i + d, 0), N - 1)] for d in range(-3, 4)) / 7.0 for i in range(N)]
    extra[r] = list(zip(sm(fa), sm(fb)))
    print('skirt', r, 'max swing', round(max(e[0] for e in extra[r]), 2), 'residual', float(sum(C[f][path[f]] for f in range(len(idx)))))
# bake into the rotation curves of the chain roots
act = arm.animation_data.action
fcs = []
for layer in act.layers:
    for strip in layer.strips:
        for cb in strip.channelbags: fcs += list(cb.fcurves)
done = 0
for r in roots:
    chans = [fc for fc in fcs if fc.data_path == f'pose.bones["{r}"].rotation_quaternion']
    chans.sort(key=lambda f: f.array_index)
    if len(chans) != 4: print('no curves for', r); continue
    for i in range(N):
        q = Quaternion([chans[c].evaluate(i + 1) for c in range(4)]) @ q_extra(r, extra[r][i][0], extra[r][i][1])
        for c in range(4): chans[c].keyframe_points[i].co[1] = q[c]
    for c in chans: c.update()
    done += 1
print('baked', done, 'chains')
bpy.ops.wm.save_as_mainfile(filepath=dst)
