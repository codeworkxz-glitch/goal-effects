import bpy, sys
from mathutils.bvhtree import BVHTree
bpy.ops.wm.open_mainfile(filepath=sys.argv[-1])
sc=bpy.context.scene
names=['Katana','Saya','head_2','helmet','mask','armo','neiyi_','Protective_L','glovesL']
def tree(dg,n):
    ob=bpy.data.objects[n].evaluated_get(dg); me=ob.to_mesh(); mw=ob.matrix_world
    t=BVHTree.FromPolygons([mw@v.co for v in me.vertices],[tuple(p.vertices) for p in me.polygons]); ob.to_mesh_clear(); return t
print('t     Katana x [head helmet mask armo neiyi prot gloves]   Saya x [same]')
for i in range(0,302,10):
    sc.frame_set(i+1); dg=bpy.context.evaluated_depsgraph_get()
    T={n:tree(dg,n) for n in names}
    k=[len(T['Katana'].overlap(T[n])) for n in names[2:]]
    s=[len(T['Saya'].overlap(T[n])) for n in names[2:]]
    print(f'{i/60:4.2f}  {k}   {s}')
