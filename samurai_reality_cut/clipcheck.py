"""Mesh-intersection detector.  python clipcheck.py file.blend t0,t1,... | range:0.3:1.5:0.05 [verbose]
Counts overlapping triangle pairs between body parts (evaluated, i.e. deformed) per frame."""
import bpy, sys, math
from mathutils.bvhtree import BVHTree
blend, spec = sys.argv[-2], sys.argv[-1]
if spec.startswith('range'):
    _, a, b, c = spec.split(':'); a, b, c = float(a), float(b), float(c)
    times = [a + i * c for i in range(int((b - a) / c) + 1)]
else:
    times = [float(x) for x in spec.split(',')]
bpy.ops.wm.open_mainfile(filepath=blend)
sc = bpy.context.scene

FING = ('m', 'f', 'r', 'l', 't')
def classify(g):
    for s in 'LR':
        if g in (f'{s}_arm', f'{s}_forarm'): return f'arm{s}'
        if g == f'{s}_hand' or (g.startswith(s + '_') and g[2] in FING and g[3:].isdigit() or g in [f'{s}_{x}' for x in FING]): return f'hand{s}'
        if g.startswith(f'{s}_armojian'): return f'plate{s}'
        if g == f'{s}_shoulder': return f'shoulder{s}'
    if g in ('HIP', 'spine', 'chest', 'neck') or 'hujia' in g or g.startswith('rope'): return 'body'
    if g.endswith(('leg', 'knee', 'ankle', 'foot', 'foottop')): return 'leg'
    return 'other'

parts = {}   # obj name -> {category: [poly indices]}
for o in bpy.data.objects:
    if o.type != 'MESH' or o.name in ('Katana', 'Saya') or o.name.startswith(('head', 'helmet', 'mask')): continue
    vg = {i: g.name for i, g in enumerate(o.vertex_groups)}
    d = {}
    for p in o.data.polygons:
        w = {}
        for vi in p.vertices:
            for g in o.data.vertices[vi].groups:
                c = classify(vg[g.group]); w[c] = w.get(c, 0) + g.weight
        if not w: continue
        c = max(w, key=w.get)
        d.setdefault(c, []).append(p.index)
    parts[o.name] = d

PAIRS = [('armL', 'body'), ('armR', 'body'), ('handL', 'body'), ('handR', 'body'), ('armL', 'armR'), ('handL', 'handR'),
         ('handL', 'armR'), ('handR', 'armL'), ('armL', 'leg'), ('armR', 'leg'), ('handL', 'leg'), ('handR', 'leg'),
         ('plateL', 'body'), ('plateR', 'body'), ('plateL', 'plateR')]
SWORD = ['hand', 'arm', 'body', 'leg']

def trees(dg):
    T = {}   # category -> list of BVH (one per object)
    for on, d in parts.items():
        ob = bpy.data.objects[on].evaluated_get(dg)
        me = ob.to_mesh()
        mw = ob.matrix_world
        verts = [mw @ v.co for v in me.vertices]
        polys_all = [tuple(p.vertices) for p in me.polygons]
        for c, idx in d.items():
            polys = [polys_all[i] for i in idx]
            T.setdefault(c, []).append((on, BVHTree.FromPolygons(verts, polys, epsilon=0.0), idx))
        ob.to_mesh_clear()
    for nm in ('Katana', 'Saya'):
        ob = bpy.data.objects[nm].evaluated_get(dg); me = ob.to_mesh(); mw = ob.matrix_world
        verts = [mw @ v.co for v in me.vertices]; polys = [tuple(p.vertices) for p in me.polygons]
        T[nm] = [(nm, BVHTree.FromPolygons(verts, polys, epsilon=0.0), list(range(len(polys))))]
        ob.to_mesh_clear()
    return T

def count(T, a, b):
    n = 0
    for (oa, ta, _) in T.get(a, []):
        for (ob_, tb, _) in T.get(b, []):
            n += len(ta.overlap(tb)) if oa != ob_ or a != b else 0
    return n

verbose = True
base = None
print('t      ' + ' '.join(f'{a[:5]}-{b[:5]}' for a, b in PAIRS) + '  | katana-vs: hand arm body leg')
tot = 0
for t in times:
    sc.frame_set(int(round(t * 60)) + 1)
    dg = bpy.context.evaluated_depsgraph_get()
    T = trees(dg)
    row = [count(T, a, b) for a, b in PAIRS]
    ks = []
    for nm in ('Katana',):
        for cat in ('hand', 'arm', 'body', 'leg'):
            ks.append(sum(count(T, nm, cat + s) if cat in ('hand', 'arm') else 0 for s in 'LR') if cat in ('hand', 'arm') else count(T, nm, cat))
    sy = [sum(count(T, 'Saya', c + s) for s in 'LR') if c in ('hand', 'arm') else count(T, 'Saya', c) for c in ('hand', 'arm', 'body', 'leg')]
    print(f'{t:5.2f} ' + ' '.join(f'{x:11d}' for x in row) + '  | ' + ' '.join(map(str, ks)) + ' / saya ' + ' '.join(map(str, sy)))
