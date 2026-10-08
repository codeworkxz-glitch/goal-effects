"""Per-frame checks on the baked Blood Reaper (deformed meshes):  python checks_reaper.py <blend> [step]

columns: scythe-vs-body overlaps (hands/fingers excluded: they hold it), scythe-vs-wings, arm-vs-torso, wing-vs-body,
grip: max distance of finger joints from the shaft surface (both hands; cm), lowest blade point (cm, <0 = in the
ground), lowest robe point while standing (cm), largest per-frame vertex move (cm/frame)
"""
import bpy, sys, os, numpy as np
from mathutils.bvhtree import BVHTree
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import choreo_reaper as C, reaper_rig as RR

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
bpy.ops.wm.open_mainfile(filepath=argv[0]); step = int(argv[1]) if len(argv) > 1 else 2
K = float(os.environ.get('REAPER_SCALE', '0.042'))
ao = bpy.data.objects['BloodReaper']
HAND = ('Hand_', 'Finger')
ARM = ('UpperArm_', 'Forearm_')
TORSO = ('Pelvis', 'Spine', 'Ribcage', 'Neck', 'Head', 'Clavicle', 'Robe', 'Trinket')
WING = ('Wing',)


def classify(name):
    for tag, keys in (('hand', HAND), ('arm', ARM), ('wing', WING), ('torso', TORSO)):
        if name.startswith(keys): return tag
    return 'torso'


meshes = [o for o in bpy.data.objects if o.type == 'MESH']
parts = {}
for o in meshes:
    gname = {g.index: g.name for g in o.vertex_groups}
    cls = {}
    for p in o.data.polygons:
        acc = {}
        for vi in p.vertices:
            for g in o.data.vertices[vi].groups:
                c = classify(gname[g.group]); acc[c] = acc.get(c, 0) + g.weight
        c = 'scythe' if o.name == 'Scythe' else (max(acc, key=acc.get) if acc else 'torso')
        cls.setdefault(c, []).append(p.index)
    parts[o.name] = cls

solver = C.Solver()
sc = bpy.context.scene
prevV = None
rows = []
fingers = [n for n in solver.r.names if n.startswith('Finger') and n.endswith(('_2_L', '_3_L', '_2_R', '_3_R'))]
for f in range(0, C.NFRAMES, step):
    sc.frame_set(f); dg = bpy.context.evaluated_depsgraph_get()
    T = {}; allV = []
    lowest_blade = 1e9; lowest_robe = 1e9
    for o in meshes:
        e = o.evaluated_get(dg); me = e.to_mesh()
        V = np.array([v.co[:] for v in me.vertices]); allV.append(V)
        polys = [tuple(p.vertices) for p in me.polygons]
        for c, idx in parts[o.name].items():
            T.setdefault(c, []).append(BVHTree.FromPolygons([tuple(v) for v in V], [polys[i] for i in idx], epsilon=0.0))
        if o.name == 'Scythe': lowest_blade = V[:, 2].min()
        if o.name in ('RobeSkirt', 'cadnav'): lowest_robe = min(lowest_robe, V[:, 2].min())
        e.to_mesh_clear()
    def ov(a, b):
        return sum(len(x.overlap(y)) for x in T.get(a, []) for y in T.get(b, []))
    n_sb = ov('scythe', 'torso') + ov('scythe', 'arm')
    n_sw = ov('scythe', 'wing')
    n_at = ov('arm', 'torso')
    n_wt = ov('wing', 'torso')
    # grip: finger joints vs shaft axis
    t = f / C.FPS
    S = C.scythe_frame(t, solver.r)
    gR = C.scalar_keys(C.GRIP_R_KEYS, t)
    grip = []
    for n in fingers:
        if n.endswith('R') and gR < 1: continue
        p = np.array(ao.pose.bones[n].head) / K
        q = p - S[:3, 3]; d = S[:3, 0]; r = np.linalg.norm(q - d * (q @ d))
        grip.append(r)
    Vall = np.concatenate(allV)
    jump = np.linalg.norm(Vall - prevV, axis=1).max() / step * 100 if prevV is not None else 0.0
    prevV = Vall
    rows.append((f, t, n_sb, n_sw, n_at, n_wt, max(grip) * K * 100 if grip else 0, lowest_blade * 100, lowest_robe * 100, jump))
print('frame    t   scy-body scy-wing arm-torso wing-body  grip(cm)  bladeZ(cm) robeZ(cm) jump(cm/f)')
for r in rows:
    print(f'{r[0]:5d} {r[1]:5.2f} {r[2]:8d} {r[3]:8d} {r[4]:9d} {r[5]:9d} {r[6]:9.1f} {r[7]:10.1f} {r[8]:9.1f} {r[9]:9.1f}')
