"""Contact sheet renders with a reference hoop (rim, net-free, backboard, floor):
    python preview.py <file.blend> <out.png> <t1,t2,...> [front|side|three|top|close] [res]
"""
import bpy, sys, math, os
from mathutils import Vector
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
C = importlib.import_module(os.environ.get('SERPENT_CHOREO', 'choreo'))

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
blend, out, times = argv[0], argv[1], [float(x) for x in argv[2].split(',')]
view = argv[3] if len(argv) > 3 else 'three'
res = int(argv[4]) if len(argv) > 4 else 420
bpy.ops.wm.open_mainfile(filepath=blend)
K = C.METRES_PER_UNIT
sc = bpy.context.scene
sc.render.engine = 'CYCLES'; sc.cycles.samples = int(os.environ.get('SAMP', '10')); sc.cycles.device = 'CPU'
sc.cycles.use_denoising = False
w = bpy.data.worlds.new('w'); sc.world = w
w.use_nodes = True
w.node_tree.nodes['Background'].inputs[0].default_value = (0.42, 0.45, 0.5, 1)
w.node_tree.nodes['Background'].inputs[1].default_value = 0.9
sun = bpy.data.lights.new('s', 'SUN'); sun.energy = 3.0
so = bpy.data.objects.new('s', sun); sc.collection.objects.link(so); so.rotation_euler = (math.radians(40), 0, math.radians(-30))


def mat(name, col):
    m = bpy.data.materials.new(name); m.use_nodes = True
    m.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = col
    return m


if hasattr(C, 'RIM_R'):
    rim_r = (C.RIM_R + 0.45) * K
    bpy.ops.mesh.primitive_torus_add(major_radius=rim_r, minor_radius=0.009, major_segments=64, minor_segments=8)
    rim = bpy.context.active_object; rim.data.materials.append(mat('rim', (0.9, 0.25, 0.02, 1)))
    if os.environ.get('BOARD', '1') == '1':
        bpy.ops.mesh.primitive_cube_add(size=1)
        bb = bpy.context.active_object
        bb.scale = (1.8, 0.03, 1.05); bb.location = (0, C.BACKBOARD_Y * K + 0.015, -0.15 + 0.525)
        bb.data.materials.append(mat('bb', (0.85, 0.88, 0.9, 1)))
    bpy.ops.mesh.primitive_plane_add(size=12, location=(0, 0, C.FLOOR_Z * K))
    bpy.context.active_object.data.materials.append(mat('floor', (0.55, 0.38, 0.22, 1)))
else:
    # opaque stand-in for the water surface / terrain the effect will sit on
    bpy.ops.mesh.primitive_plane_add(size=400, location=(0, 0, C.GROUND_Z * K))
    bpy.context.active_object.data.materials.append(mat('surface', (0.05, 0.16, 0.2, 1)))

cam = bpy.data.cameras.new('c'); cam.lens = float(os.environ.get('LENS', '30'))
co = bpy.data.objects.new('c', cam); sc.collection.objects.link(co); sc.camera = co
views = {'front': ((0, -3.6, 0.6), (0, 0, 0.0)), 'side': ((3.6, 0, 0.5), (0, 0, 0.0)),
         'three': ((2.3, -2.9, 1.0), (0, 0, -0.05)), 'top': ((0.01, -0.3, 4.0), (0, 0, 0)),
         'close': ((0.9, -1.5, 0.75), (0, 0.0, 0.25)), 'low': ((0.4, -3.6, -1.6), (0, 0, -0.4)),
         'wide': ((3.2, -5.5, 0.4), (0, 0, -1.2)), 'swimside': ((0, -38.0, 6.0), (0, 0, 1.5)), 'swimthree': ((-26.0, -32.0, 16.0), (0, 0, 0.0)), 'swimlow': ((4.0, -24.0, 0.8), (0, -12.0, 2.5)), 'swimtop': ((0.0, -6.0, 45.0), (0, 0, 0)), 'court': ((0, -1.9, 0.55), (0, 0, 0.2)), 'face': ((0.25, -1.1, 0.75), (0, 0.05, 0.62)), 'armr': ((0.95, -0.25, 0.5), (0, 0.1, 0.5)), 'arml': ((-0.95, -0.35, 0.5), (0, 0.1, 0.45)), 'under': ((0.5, -0.9, -0.2), (0.15, 0.0, 0.3)), 'courtwide': ((0.6, -6.0, 0.9), (0, 0, -1.0)), 'eyes': ((0.12, -0.75, 0.62), (0, 0.06, 0.6))}
loc, tgt = views[view]
co.location = loc; co.rotation_euler = (Vector(tgt) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
sc.render.resolution_x = res; sc.render.resolution_y = res
ims = []
for t in times:
    sc.frame_set(int(round(t * C.FPS)))
    sc.render.filepath = os.path.join(os.path.dirname(os.path.abspath(out)), '_pv.png')
    bpy.ops.render.render(write_still=True)
    im = Image.open(sc.render.filepath).convert('RGB')
    ImageDraw.Draw(im).text((6, 6), f't={t:.2f}s', fill=(255, 255, 0))
    ims.append(im)
cols = min(len(ims), 4); rows = (len(ims) + cols - 1) // cols
sheet = Image.new('RGB', (res * cols, res * rows))
for i, im in enumerate(ims): sheet.paste(im, ((i % cols) * res, (i // cols) * res))
sheet.save(out)
