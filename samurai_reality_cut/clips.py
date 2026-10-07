"""Shared mesh-intersection scoring (deformed meshes, BVH triangle overlap)."""
import bpy
from mathutils.bvhtree import BVHTree

FING = ('m', 'f', 'r', 'l', 't')


def classify(g):
    for s in 'LR':
        if g in (f'{s}_arm', f'{s}_forarm'): return 'arm' + s
        if g == f'{s}_hand': return 'hand' + s
        if g.startswith(s + '_') and g[2:3] in FING and (len(g) == 3 or g[3:].isdigit()): return 'hand' + s
        if g.startswith(f'{s}_armojian'): return 'plate' + s
        if g == f'{s}_shoulder': return 'shoulder' + s
    if g in ('HIP', 'spine', 'chest', 'neck') or 'hujia' in g or g.startswith('rope'): return 'body'
    if g.endswith(('leg', 'knee', 'ankle', 'foot', 'foottop')): return 'leg'
    return 'other'


class Scorer:
    SKIP = ('head', 'helmet', 'mask')

    def __init__(self):
        self.parts = {}
        for o in bpy.data.objects:
            if o.type != 'MESH' or o.name in ('Katana', 'Saya') or o.name.startswith(self.SKIP): continue
            vg = {i: g.name for i, g in enumerate(o.vertex_groups)}
            d = {}
            for p in o.data.polygons:
                w = {}
                for vi in p.vertices:
                    for g in o.data.vertices[vi].groups:
                        c = classify(vg[g.group]); w[c] = w.get(c, 0) + g.weight
                if w: d.setdefault(max(w, key=w.get), []).append(p.index)
            self.parts[o.name] = d

    def trees(self, dg):
        T = {}
        for on, d in self.parts.items():
            ob = bpy.data.objects[on].evaluated_get(dg)
            me = ob.to_mesh(); mw = ob.matrix_world
            verts = [mw @ v.co for v in me.vertices]
            polys_all = [tuple(p.vertices) for p in me.polygons]
            for c, idx in d.items():
                T.setdefault(c, []).append((on, BVHTree.FromPolygons(verts, [polys_all[i] for i in idx], epsilon=0.0)))
            ob.to_mesh_clear()
        for nm in ('Katana', 'Saya'):
            ob = bpy.data.objects[nm].evaluated_get(dg); me = ob.to_mesh(); mw = ob.matrix_world
            verts = [mw @ v.co for v in me.vertices]
            T[nm] = [(nm, BVHTree.FromPolygons(verts, [tuple(p.vertices) for p in me.polygons], epsilon=0.0))]
            ob.to_mesh_clear()
        return T

    @staticmethod
    def count(T, a, b):
        n = 0
        for (oa, ta) in T.get(a, []):
            for (ob_, tb) in T.get(b, []):
                if oa == ob_ and a == b: continue
                n += len(ta.overlap(tb))
        return n

    def side_cost(self, T, s):
        """clip cost attributable to one arm (s = 'L'/'R')."""
        o = 'R' if s == 'L' else 'L'
        c = self.count
        cost = 1.0 * (c(T, 'arm' + s, 'body') + c(T, 'hand' + s, 'body') + c(T, 'arm' + s, 'leg') + c(T, 'hand' + s, 'leg'))
        cost += 3.0 * (c(T, 'arm' + s, 'arm' + o) + c(T, 'hand' + s, 'hand' + o) + c(T, 'hand' + s, 'arm' + o) + c(T, 'arm' + s, 'hand' + o))
        cost += 2.0 * (c(T, 'Katana', 'arm' + s) + c(T, 'Saya', 'arm' + s))
        return 4.0 * cost

    def sword_cost(self, T):
        c = self.count
        return 1.5 * (c(T, 'Katana', 'body') + c(T, 'Saya', 'body') + c(T, 'Katana', 'leg') + c(T, 'Saya', 'leg'))


def _inside(tree, p, axis=(0.0, 0.0, 1.0)):
    from mathutils import Vector
    d = Vector(axis); o = p.copy(); n = 0
    for _ in range(16):
        loc, nor, idx, dist = tree.ray_cast(o, d)
        if loc is None: break
        n += 1; o = loc + d * 1e-4
    return n % 2 == 1


def penetration(scorer, dg, s):
    """(inside vertex count, max depth in cm) of arm/hand vertices of side s sunk into body / opposite arm parts."""
    from mathutils import Vector
    o = 'R' if s == 'L' else 'L'
    meshes = {}
    for on in scorer.parts:
        ob = bpy.data.objects[on].evaluated_get(dg)
        me = ob.to_mesh(); mw = ob.matrix_world
        meshes[on] = ([mw @ v.co for v in me.vertices], [tuple(p.vertices) for p in me.polygons])
        ob.to_mesh_clear()
    def tree_of(cats):
        verts = []; polys = []; base = {}
        for on, d in scorer.parts.items():
            v, p = meshes[on]
            off = len(verts); verts += v
            for c in cats:
                for i in d.get(c, []): polys.append(tuple(off + k for k in p[i]))
        return BVHTree.FromPolygons(verts, polys, epsilon=0.0)
    targets = {'body': tree_of(('body',)), 'otherarm': tree_of(('arm' + o, 'hand' + o)), 'leg': tree_of(('leg',))}
    worst = (0, 0.0, '')
    cnt = 0
    for on, d in scorer.parts.items():
        v, p = meshes[on]
        vs = set()
        for c in ('arm' + s, 'hand' + s):
            for i in d.get(c, []): vs.update(p[i])
        for vi in vs:
            for nm, tr in targets.items():
                if _inside(tr, v[vi]) and _inside(tr, v[vi], (1.0, 0.0, 0.0)):
                    loc, nor, idx, dist = tr.find_nearest(v[vi])
                    cnt += 1
                    if dist is not None and dist > worst[1]: worst = (cnt, dist * 100.0, nm + ':' + on)
    return cnt, worst[1], worst[2]
