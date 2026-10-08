"""Check an exported FBX against the built scene:  python verify_fbx.py <BloodReaper.blend> <out.fbx>

1. raw file: global axes / units, every Model node's Lcl Rotation / Scaling (must be identity for Roblox)
2. bone influences per vertex (<= 4), bone count, frame range
3. re-import (Blender converts the file's Y-up centimetres back to Z-up metres) and compare the deformed
   vertices with the source scene at several frames
"""
import bpy, addon_utils, sys, os, importlib, numpy as np
argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
blend, fbx = os.path.abspath(argv[0]), os.path.abspath(argv[1])
FRAMES = (0, 60, 120, 150, 175, 186, 240)

mod = [m for m in addon_utils.modules() if m.__name__.endswith('io_scene_fbx')][0]
pf = importlib.import_module(mod.__name__ + '.parse_fbx')
root, _ = pf.parse(fbx)
def kids(e, name): return [c for c in e.elems if c.id == name]
gs = {p.props[0].decode(): p.props[-1] for p in kids(kids(root, b'GlobalSettings')[0], b'Properties70')[0].elems}
print('file: UpAxis', gs['UpAxis'], 'FrontAxis', gs['FrontAxis'], 'UnitScaleFactor', gs['UnitScaleFactor'])
bad = []
for m in kids(kids(root, b'Objects')[0], b'Model'):
    name = m.props[1].split(b'\x00')[0].decode()
    for p in kids(m, b'Properties70')[0].elems:
        k = p.props[0]
        v = np.array(p.props[4:7], float)
        if m.props[2] in (b'Mesh', b'Null') and k == b'Lcl Rotation' and np.abs(v).max() > 1e-3: bad.append((name, 'rot', v))
        if k == b'Lcl Scaling' and np.abs(v - 1).max() > 1e-4: bad.append((name, 'scale', v))
print('non-identity root rotations / any scaling:', bad or 'none')
for m in kids(kids(root, b'Objects')[0], b'Model'):
    name = m.props[1].split(b'\x00')[0].decode()
    if name in ('Pelvis', 'Scythe', 'Hand_L'):
        pr = {p.props[0].decode(): [round(x, 1) for x in p.props[4:7]] for p in kids(m, b'Properties70')[0].elems if p.props[0] in (b'Lcl Translation', b'Lcl Rotation')}
        print('  default node', name, pr)


def deformed(objs):
    dg = bpy.context.evaluated_depsgraph_get()
    out = {}
    for o in objs:
        e = o.evaluated_get(dg); me = e.to_mesh()
        V = np.array([v.co[:] for v in me.vertices]); M = np.array(o.matrix_world)
        out[o.name] = V @ M[:3, :3].T + M[:3, 3]
        e.to_mesh_clear()
    return out


bpy.ops.wm.open_mainfile(filepath=blend)
src_meshes = [o for o in bpy.data.objects if o.type == 'MESH']
ref = {}
for f in FRAMES:
    bpy.context.scene.frame_set(f); ref[f] = deformed(src_meshes)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=fbx)
arm = [o for o in bpy.data.objects if o.type == 'ARMATURE'][0]
meshes = [o for o in bpy.data.objects if o.type == 'MESH']
print('bones', len(arm.data.bones), 'frames', tuple(arm.animation_data.action.frame_range), 'fps', bpy.context.scene.render.fps)
for o in meshes:
    infl = max(sum(1 for g in v.groups if g.weight > 1e-4) for v in o.data.vertices)
    tris = sum(len(p.vertices) - 2 for p in o.data.polygons)
    print(f'  {o.name:18s} tris {tris:5d} max influences {infl}')
from mathutils.kdtree import KDTree
def nn_err(A, B):
    kd = KDTree(len(B))
    for i, p in enumerate(B): kd.insert(p, i)
    kd.balance()
    return max(kd.find(p)[2] for p in A[::3])
off = int(round(arm.animation_data.action.frame_range[0]))       # first exported frame = source frame 0
worst = 0.0
for f in FRAMES:
    bpy.context.scene.frame_set(f + off); got = deformed(meshes)
    A = np.concatenate(list(ref[f].values())); B = np.concatenate(list(got.values()))
    err = max(nn_err(A, B), nn_err(B, A)); worst = max(worst, err)
    print(f'frame {f:3d}: max vertex error {err * 100:.3f} cm   height {B[:, 2].min():6.2f} .. {B[:, 2].max():5.2f} m')
# the file's default bone transforms must be the bind pose: imported rest == pose at no animation
print('OK' if worst < 0.01 and not bad else 'MISMATCH', f'(worst {worst * 100:.3f} cm)')
