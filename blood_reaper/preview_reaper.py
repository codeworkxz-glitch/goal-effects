"""Contact sheets for the Blood Reaper:  python preview_reaper.py <blend> <out.png> <times> <view[,view2...]> [res]
views: front side back three top low close hands slam  (one row per time, one column per view)"""
import bpy, sys, os, math
from mathutils import Vector
from PIL import Image, ImageDraw
argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
blend, out, times, views = argv[0], argv[1], [float(x) for x in argv[2].split(',')], argv[3].split(',')
res = int(argv[4]) if len(argv) > 4 else 300
bpy.ops.wm.open_mainfile(filepath=blend)
sc = bpy.context.scene
sc.render.engine = 'CYCLES'; sc.cycles.samples = int(os.environ.get('SAMP', '8')); sc.cycles.device = 'CPU'
sc.cycles.use_denoising = False
w = bpy.data.worlds.new('w'); sc.world = w; w.use_nodes = True
w.node_tree.nodes['Background'].inputs[0].default_value = (0.55, 0.57, 0.62, 1); w.node_tree.nodes['Background'].inputs[1].default_value = 0.9
sun = bpy.data.lights.new('s', 'SUN'); sun.energy = 3.2; so = bpy.data.objects.new('s', sun); sc.collection.objects.link(so)
so.rotation_euler = (math.radians(45), 0, math.radians(-35))
m = bpy.data.materials.new('g'); m.use_nodes = True; m.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.32, 0.26, 0.2, 1)
bpy.ops.mesh.primitive_plane_add(size=60, location=(0, 0, 0)); bpy.context.active_object.data.materials.append(m)
cam = bpy.data.cameras.new('c'); co = bpy.data.objects.new('c', cam); sc.collection.objects.link(co); sc.camera = co
H = 4.4   # approx reaper height (m)
V = {'front': ((0, -11, 3.0), (0, 0, 2.2), 35), 'side': ((11, 0, 3.0), (0, 0, 2.2), 35), 'back': ((0, 11, 3.0), (0, 0, 2.2), 35),
     'three': ((7.5, -8.5, 4.5), (0, 0, 2.0), 35), 'top': ((0.01, -1.0, 14), (0, 0, 1.5), 35), 'low': ((-3.5, -8, 0.6), (0, 0, 2.6), 30),
     'close': ((2.2, -4.5, 3.6), (0, -0.3, 2.9), 40), 'hands': ((-2.6, -4.2, 3.2), (0, -0.6, 2.6), 45),
     'slam': ((5.5, -6.5, 1.6), (0, -2.0, 1.2), 32), 'wide': ((10, -14, 5), (0, -1, 1.8), 32), 'face': ((0.0, -2.2, 3.9), (0, 0, 3.75), 50), 'faceside': ((1.6, -1.6, 3.9), (0, 0, 3.75), 50)}
ims = []
for t in times:
    sc.frame_set(int(round(t * 60)))
    row = []
    for v in views:
        loc, tgt, lens = V[v]; cam.lens = lens
        co.location = loc; co.rotation_euler = (Vector(tgt) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
        sc.render.resolution_x = res; sc.render.resolution_y = res
        sc.render.filepath = os.path.join(os.path.dirname(os.path.abspath(out)), '_rp.png')
        bpy.ops.render.render(write_still=True)
        im = Image.open(sc.render.filepath).convert('RGB'); ImageDraw.Draw(im).text((4, 4), f't={t:.2f} {v}', fill=(255, 255, 0))
        row.append(im)
    ims.append(row)
sheet = Image.new('RGB', (res * len(views), res * len(times)))
for r, row in enumerate(ims):
    for c, im in enumerate(row): sheet.paste(im, (c * res, r * res))
sheet.save(out)
