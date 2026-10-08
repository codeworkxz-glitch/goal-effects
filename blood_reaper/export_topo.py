"""Topology + per-triangle body-part class for the collision-aware optimiser:  python export_topo.py -> work/topo.npz"""
import bpy, sys, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import reaper_rig as RR, build_reaper as B
rig = RR.Rig(); meshes = B.import_static(rig)
Wt = np.load(os.path.join(RR.WORK, 'weights.npz'))
out = {}
def cls(n):
    if n.startswith(('Hand_', 'Finger')): return 3
    if n.startswith(('UpperArm_', 'Forearm_')): return 2
    if n.startswith('Wing'): return 4
    return 1
for mname, me in meshes.items():
    me.calc_loop_triangles()
    tris = np.array([tuple(t.vertices) for t in me.loop_triangles], int)
    out['tri_' + mname] = tris
    if mname.startswith('Object001'): continue
    if 'idx_' + mname in Wt:
        idx, w = Wt['idx_' + mname], Wt['w_' + mname]
        vc = np.zeros((len(idx), 5))
        for k in range(4):
            c = np.array([cls(rig.names[j]) for j in idx[:, k]])
            np.add.at(vc, (np.arange(len(idx)), c), w[:, k])
        tc = vc[tris].sum(1).argmax(1)
        out['tcls_' + mname] = tc
np.savez(os.path.join(RR.WORK, 'topo.npz'), **out)
print({k: v.shape for k, v in out.items()})
