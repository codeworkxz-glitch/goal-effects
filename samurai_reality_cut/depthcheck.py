"""python depthcheck.py file.blend range:t0:t1:dt  – max penetration depth (cm) of each arm into body/other arm"""
import bpy, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clips
blend, spec = sys.argv[-2], sys.argv[-1]
_, a, b, c = spec.split(':'); a, b, c = float(a), float(b), float(c)
times = [a + i * c for i in range(int(round((b - a) / c)) + 1)]
bpy.ops.wm.open_mainfile(filepath=blend)
sc = bpy.context.scene; S = clips.Scorer()
worst = 0
for t in times:
    sc.frame_set(int(round(t * 60)) + 1)
    dg = bpy.context.evaluated_depsgraph_get()
    L = clips.penetration(S, dg, 'L'); R = clips.penetration(S, dg, 'R')
    worst = max(worst, L[1], R[1])
    flag = '' if max(L[1], R[1]) < 0.3 else '  <<<'
    print(f't={t:.2f}  L: {L[0]:4d} verts depth {L[1]:.2f}cm {L[2]:18s} R: {R[0]:4d} verts depth {R[1]:.2f}cm {R[2]}{flag}', flush=True)
print('WORST DEPTH cm', round(worst, 2))
