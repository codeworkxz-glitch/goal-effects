"""Deformation / clipping checks on the baked animation.

    python checks.py <file.blend> [step]

Per sampled frame (deformed meshes):
  * body self-intersection: triangles of body sections more than 14 units apart along the spine that overlap
    (the coil loops, neck vs coil, strike arch, descending column vs coil ...)
  * arm/claw vs body triangles
  * rim clearance: closest deformed vertex to the rim tube (units and cm), count of vertices inside the tube
  * backboard: max Y of any vertex in the board's height band (must stay < board face)
  * stretch: max edge-length ratio deformed/rest on the body skin (detects pinching / stretching)
"""
import bpy, sys, os, math
import numpy as np
from mathutils.bvhtree import BVHTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
C = importlib.import_module(os.environ.get('SERPENT_CHOREO', 'choreo'))

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
bpy.ops.wm.open_mainfile(filepath=argv[0])
step = int(argv[1]) if len(argv) > 1 else 3
K = C.METRES_PER_UNIT
SEC = 7.0  # section length (units) along the rest spine
ARM = ('Shoulder', 'Upperarm', 'Forearm', 'Middle', 'Pinky', 'Pointer', 'Thumb')

objs = [o for o in bpy.data.objects if o.type == 'MESH']
info = {}
for o in objs:
    me = o.data
    vy = np.array([v.co.y / K for v in me.vertices])
    gname = {g.index: g.name for g in o.vertex_groups}
    armw = np.zeros(len(me.vertices))
    for v in me.vertices:
        for g in v.groups:
            if gname[g.group].startswith(ARM): armw[v.index] += g.weight
    polys = [tuple(p.vertices) for p in me.polygons]
    py = np.array([vy[list(p)].mean() for p in polys])
    pa = np.array([armw[list(p)].mean() for p in polys])
    sec = np.floor(py / SEC).astype(int)
    edges = np.array([tuple(e.vertices) for e in me.edges])
    rest = np.array([v.co[:] for v in me.vertices])
    rl = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)
    info[o.name] = dict(polys=polys, sec=sec, arm=pa > 0.5, edges=edges, rl=rl, body=o.name.startswith('DreadSerpent_Body'))

HAS_RIM = hasattr(C, 'RIM_R')
rim_R = (C.RIM_R + 0.45) if HAS_RIM else 1e6; tube = 0.45
prevV = None; jump_max = 0.0
rows = []
sc = bpy.context.scene
worst = dict(self=0, arm=0, rim_in=0, rim_min=1e9, bb=-1e9, stretch=0, squash=1e9, jump=0)
for f in range(0, C.NFRAMES, step):
    sc.frame_set(f)
    dg = bpy.context.evaluated_depsgraph_get()
    secs = {}; armT = []; allv = []
    st_max = 0; st_min = 1e9
    for o in objs:
        ob = o.evaluated_get(dg); me = ob.to_mesh()
        V = np.array([v.co[:] for v in me.vertices]) / K
        ob.to_mesh_clear()
        I = info[o.name]
        allv.append(V)
        if I['body']:
            dl = np.linalg.norm(V[I['edges'][:, 0]] - V[I['edges'][:, 1]], axis=1) * K
            ok = I['rl'] > 1e-5
            r = dl[ok] / I['rl'][ok]
            st_max = max(st_max, np.percentile(r, 99.9)); st_min = min(st_min, np.percentile(r, 0.1))
        Vl = [tuple(v) for v in V]
        by = {}
        for pi, p in enumerate(I['polys']):
            if I['arm'][pi]: armT.append([Vl[i] for i in p]); continue
            by.setdefault(int(I['sec'][pi]), []).append([Vl[i] for i in p])
        for s_, lst in by.items(): secs.setdefault(s_, []).extend(lst)

    def tree(tris):
        verts = []; faces = []
        for t in tris:
            b = len(verts); verts.extend(t); faces.append(tuple(range(b, b + len(t))))
        return BVHTree.FromPolygons(verts, faces, epsilon=0.0)
    T = {s_: tree(l) for s_, l in secs.items()}
    keys = sorted(T)
    n_self = 0; pairs = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if b - a < 3: continue   # sections closer than ~14 units along the spine
            n = len(T[a].overlap(T[b]))
            if n: n_self += n; pairs.append((a, b, n))
    n_arm = 0
    if armT:
        TA = tree(armT)
        for s_ in keys:
            if s_ * SEC > 26: continue
            n_arm += len(TA.overlap(T[s_]))
    V = np.concatenate(allv)
    # largest per-vertex move since the previous checked frame (units per frame): pops / teleports show here
    jump = float(np.linalg.norm(V - prevV, axis=1).max() / step) if prevV is not None else 0.0
    prevV = V
    worst['jump'] = max(worst['jump'], jump)
    rho = np.sqrt(V[:, 0] ** 2 + V[:, 1] ** 2)
    dtube = np.sqrt((rho - rim_R) ** 2 + V[:, 2] ** 2) - tube
    band = ((V[:, 2] > -7) & (V[:, 2] < 41) & (np.abs(V[:, 0]) < 41)) if HAS_RIM else (V[:, 2] > 0)
    bb = V[band, 1].max() if band.any() else -1e9
    rows.append((f, f / C.FPS, n_self, n_arm, int((dtube < 0).sum()), float(dtube.min()), float(bb), st_max, st_min, pairs[:4], jump))
    worst['self'] = max(worst['self'], n_self); worst['arm'] = max(worst['arm'], n_arm)
    worst['rim_in'] = max(worst['rim_in'], int((dtube < 0).sum())); worst['rim_min'] = min(worst['rim_min'], float(dtube.min()))
    worst['bb'] = max(worst['bb'], float(bb)); worst['stretch'] = max(worst['stretch'], st_max); worst['squash'] = min(worst['squash'], st_min)

print(' frame    t   self  arm  rimIn  rimGap(u)  maxY(u)  stretch99.9  squash0.1  jump(u/f)  pairs')
for r in rows:
    print(f'{r[0]:5d} {r[1]:5.2f} {r[2]:6d} {r[3]:4d} {r[4]:6d} {r[5]:9.2f} {r[6]:8.2f} {r[7]:8.2f} {r[8]:8.2f} {r[10]:8.2f}  {r[9]}')
print('WORST', worst, (' backboard face at Y=%.1f units,' % C.BACKBOARD_Y) if HAS_RIM else '', '1 unit = %.2f cm' % (K * 100))
