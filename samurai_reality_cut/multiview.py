"""python multiview.py file.blend out.png t1,t2,.. [zoom]  -> grid: rows = times, cols = front/side/back/top"""
import bpy, sys, math, os
from mathutils import Vector
from PIL import Image
blend, out, times = sys.argv[-3], sys.argv[-2], [float(x) for x in sys.argv[-1].split(',')]
bpy.ops.wm.open_mainfile(filepath=blend)
sc = bpy.context.scene
sc.render.engine = 'CYCLES'; sc.cycles.samples = int(os.environ.get('SAMP', '10')); sc.cycles.device = 'CPU'
w = bpy.data.worlds.new('w'); w.use_nodes = True
w.node_tree.nodes['Background'].inputs[0].default_value = (0.6, 0.62, 0.66, 1); w.node_tree.nodes['Background'].inputs[1].default_value = float(os.environ.get('WORLD', '1.0'))
sc.world = w
sun = bpy.data.lights.new('s', 'SUN'); sun.energy = float(os.environ.get('SUN', '2.5')); so = bpy.data.objects.new('s', sun); sc.collection.objects.link(so)
so.rotation_euler = (math.radians(50), 0, math.radians(30))
cam = bpy.data.cameras.new('c'); cam.lens = 50
co = bpy.data.objects.new('c', cam); sc.collection.objects.link(co); sc.camera = co
D = float(os.environ.get('CAMD', '2.6')); Z = float(os.environ.get('LOOKZ', '1.05'))
views = {'front': (0, -1, 0.15), 'side': (1, 0, 0.15), 'back': (0, 1, 0.15), 'top': (0.001, -0.35, 1.9)}
sc.render.resolution_x = 420; sc.render.resolution_y = 520
rows = []
for t in times:
    sc.frame_set(int(round(t * 60)) + 1)
    row = []
    for name, v in views.items():
        loc = Vector((v[0] * D, v[1] * D, Z + v[2] * D * (1.0 if name != 'top' else 1.0) * (0.0 if name != 'top' else 1.0)))
        if name == 'top': loc = Vector((0.0, -0.35, Z + 2.2))
        else: loc = Vector((v[0] * D, v[1] * D, Z + 0.1))
        co.location = loc
        co.rotation_euler = (Vector((0, 0, Z)) - loc).to_track_quat('-Z', 'Y').to_euler()
        sc.render.filepath = '/tmp/_mv.png'; bpy.ops.render.render(write_still=True)
        row.append(Image.open('/tmp/_mv.png').convert('RGB'))
    rows.append(row)
sheet = Image.new('RGB', (420 * 4, 520 * len(rows)))
for r, row in enumerate(rows):
    for c, im in enumerate(row): sheet.paste(im, (c * 420, r * 520))
sheet.save(out); print(out, sheet.size)
