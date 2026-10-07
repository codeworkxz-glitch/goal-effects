import bpy, sys, os
sys.path.insert(0,'/home/user/goal-effects/samurai_reality_cut')
import clips
bpy.ops.wm.open_mainfile(filepath=sys.argv[-1])
sc=bpy.context.scene; S=clips.Scorer()
print('t   skirt/body x leg (armo[body]xneiyi[leg] , armo[body]xProt[leg])')
for i in range(0,302,20):
    sc.frame_set(i+1); T=S.trees(bpy.context.evaluated_depsgraph_get())
    n=S.count(T,'body','leg')
    print(f'{i/60:4.2f} {n}')
