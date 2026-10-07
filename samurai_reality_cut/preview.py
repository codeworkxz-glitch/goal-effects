"""Render a contact sheet:  python preview.py reality_cut.blend out.png t1,t2,...  [front|side|three]"""
import bpy, sys, math
from mathutils import Vector
from PIL import Image
blend, out, times = sys.argv[-4], sys.argv[-3], [float(x) for x in sys.argv[-2].split(',')]
view = sys.argv[-1]
bpy.ops.wm.open_mainfile(filepath=blend)
sc = bpy.context.scene
sc.render.engine = 'CYCLES'; sc.cycles.samples = 6; sc.cycles.device = 'CPU'
try: sc.cycles.use_denoising = False
except Exception: pass
w = bpy.data.worlds.new('w'); w.use_nodes = True
w.node_tree.nodes['Background'].inputs[0].default_value = (0.55, 0.58, 0.62, 1); w.node_tree.nodes['Background'].inputs[1].default_value = 0.6
sc.world = w
sun = bpy.data.lights.new('s', 'SUN'); sun.energy = 1.5; so = bpy.data.objects.new('s', sun); sc.collection.objects.link(so)
so.rotation_euler = (math.radians(50), 0, math.radians(30))
cam = bpy.data.cameras.new('c'); cam.lens = 45
co = bpy.data.objects.new('c', cam); sc.collection.objects.link(co); sc.camera = co
dirs = {'front': (0, -1), 'side': (1, 0), 'three': (0.8, -0.6)}[view]
import os; CD=float(os.environ.get('CAMD','3.6')); CZ=float(os.environ.get('CAMZ','1.1')); co.location = (dirs[0] * CD, dirs[1] * CD, CZ)
co.rotation_euler = (Vector((0, 0, float(os.environ.get('LOOKZ','0.85')))) - co.location).to_track_quat('-Z', 'Y').to_euler()
sc.render.resolution_x = 480; sc.render.resolution_y = 600
ims = []
for t in times:
    sc.frame_set(int(round(t * 60)) + 1)
    sc.render.filepath = '/tmp/_pv.png'
    bpy.ops.render.render(write_still=True)
    ims.append(Image.open('/tmp/_pv.png').convert('RGB'))
cols = min(len(ims), 4); rows = (len(ims) + cols - 1) // cols
sheet = Image.new('RGB', (480 * cols, 600 * rows))
for i, im in enumerate(ims): sheet.paste(im, ((i % cols) * 480, (i // cols) * 600))
sheet.save(out)
