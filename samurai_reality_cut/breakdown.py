"""python breakdown.py file.blend t1,t2,...  – which object/part pairs intersect at each time"""
import bpy, sys
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import clips
bpy.ops.wm.open_mainfile(filepath=sys.argv[-2])
sc = bpy.context.scene; S = clips.Scorer()
for t in [float(x) for x in sys.argv[-1].split(',')]:
    sc.frame_set(int(round(t * 60)) + 1)
    T = S.trees(bpy.context.evaluated_depsgraph_get())
    out = []
    keys = [k for k in T]
    for i, a in enumerate(keys):
        for b in keys[i:]:
            if {a, b} <= {'body', 'leg', 'other', 'plateL', 'plateR', 'shoulderL', 'shoulderR'}: continue
            if a in ('Katana', 'Saya') and b in ('Katana', 'Saya') and a == b: continue
            for (oa, ta) in T[a]:
                for (ob, tb) in T[b]:
                    if a == b and oa == ob: continue
                    n = len(ta.overlap(tb))
                    if n and not ({a, b} & {'plateL', 'plateR', 'shoulderL', 'shoulderR'} and {a, b} - {'plateL', 'plateR', 'shoulderL', 'shoulderR'} <= {'body', 'leg', 'other'}):
                        out.append(f'{a}[{oa}]x{b}[{ob}]={n}')
    print(f't={t}: ' + ('; '.join(out) if out else 'CLEAN'))
