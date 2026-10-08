"""Dump the Alembic bake to numpy:  python extract_cache.py source/reaper.abc work/cache.npz

bones   : biped / wing empties (name, parent) and their world matrices on every frame (F x 4 x 4)
meshes  : every mesh's world-space vertex positions on every frame (F x N x 3) + parent empty
"""
import bpy, sys, numpy as np

src, out = sys.argv[-2], sys.argv[-1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.alembic_import(filepath=src)
sc = bpy.context.scene
F = list(range(sc.frame_start, sc.frame_end + 1))

empties = [o for o in bpy.data.objects if o.type == 'EMPTY']
bones = [o for o in empties if o.name.startswith('Base Human')]
meshes = [o for o in bpy.data.objects if o.type == 'MESH']
bnames = [b.name for b in bones]
bpar = [b.parent.name if (b.parent and b.parent.name in bnames) else '' for b in bones]

BM = np.zeros((len(F), len(bones), 4, 4), np.float64)
MV = {m.name: [] for m in meshes}
OM = {m.name: [] for m in meshes}
for fi, f in enumerate(F):
    sc.frame_set(f)
    dg = bpy.context.evaluated_depsgraph_get()
    for j, b in enumerate(bones):
        BM[fi, j] = np.array(b.matrix_world)
    for m in meshes:
        e = m.evaluated_get(dg); me = e.to_mesh()
        co = np.zeros(len(me.vertices) * 3); me.vertices.foreach_get('co', co); co = co.reshape(-1, 3)
        W = np.array(e.matrix_world)
        MV[m.name].append(co @ W[:3, :3].T + W[:3, 3])
        OM[m.name].append(W)
        e.to_mesh_clear()
data = dict(frames=np.array(F), bone_names=np.array(bnames), bone_parents=np.array(bpar), bone_mats=BM,
            mesh_names=np.array([m.name for m in meshes]),
            mesh_parents=np.array([m.parent.parent.name if (m.parent and m.parent.parent) else (m.parent.name if m.parent else '') for m in meshes]))
for m in meshes:
    data['V_' + m.name] = np.array(MV[m.name], np.float32)
    data['M_' + m.name] = np.array(OM[m.name])
np.savez_compressed(out, **data)
print('frames', len(F), 'bones', len(bones), 'meshes', [(m.name, np.array(MV[m.name]).shape) for m in meshes])
print('mesh parents', list(zip([m.name for m in meshes], data['mesh_parents'])))
