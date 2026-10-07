"""Animated preview: python preview_gif.py <file.blend> <out.gif> [step] [res]"""
import bpy, sys, os, math
from mathutils import Vector
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import choreo as C
argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
blend, gif_out = argv[0], argv[1]
step = int(argv[2]) if len(argv) > 2 else 2
res = int(argv[3]) if len(argv) > 3 else 400
# reuse the reference scene (hoop, board, floor, light) from preview.py
sys.argv = ['preview.py', blend, '/dev/null', '0', 'three', str(res)]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'preview.py')).read()
src = src[:src.index('ims = []')]
exec(compile(src, 'preview.py', 'exec'))
sc = bpy.context.scene
co = sc.camera; co.data.lens = 30
co.location = (1.1, -4.4, 0.35)
co.rotation_euler = (Vector((0, 0, -0.85)) - co.location).to_track_quat('-Z', 'Y').to_euler()
sc.render.resolution_x = int(res * 1.0); sc.render.resolution_y = int(res * 1.15)
frames = []
tmp = os.path.join(os.path.dirname(os.path.abspath(gif_out)), "_gif.png")
for f in range(0, C.NFRAMES, step):
    sc.frame_set(f); sc.render.filepath = tmp
    bpy.ops.render.render(write_still=True)
    frames.append(Image.open(tmp).convert('RGB').quantize(colors=128, method=Image.Quantize.MEDIANCUT))
frames[0].save(gif_out, save_all=True, append_images=frames[1:], duration=int(1000 * step / C.FPS), loop=0, optimize=True)
