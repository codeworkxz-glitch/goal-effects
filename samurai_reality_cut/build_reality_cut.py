"""
Samurai "Reality Cut" – procedural iaido animation for wushi_export2.fbx

Run (headless, `pip install bpy` or Blender's own python):
    python build_reality_cut.py <wushi_export2.fbx> <out_dir> [--textures DIR]

What it does
  * imports the rigged FBX, drops the placeholder cylinders + floor plane
  * builds a katana (blade / tsuba / tsuka) and a saya (scabbard) and skins them
    to two extra root bones so they export cleanly to glTF
  * solves the whole motion procedurally (FK torso, analytic 2-bone IK for arms and
    legs, grip alignment of the hands to the sword) and bakes it at 60 fps
  * saves reality_cut.blend and an optimised reality_cut.glb (Draco, <20 MB)

All pose maths happens in armature space of the FBX rig:
    +Y up, +Z forward (character faces +Z), +X = character's LEFT   (units: cm)
"""
import bpy, bmesh, sys, os, math, argparse, bisect
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mathutils import Matrix, Vector, Quaternion, Euler

FPS = 60
DUR = 5.0
N = int(round(DUR * FPS)) + 1
BLADE = 66.0          # nagasa (cm)
TSUKA = 27.0
R_GRIP_U = 6.5        # right fist centre, cm behind the tsuba
L_GRIP_U = 22.5       # left fist centre, cm behind the tsuba

# ----------------------------------------------------------------------------------
# small maths helpers
# ----------------------------------------------------------------------------------
I3 = Matrix.Identity(3)


def Ry(a): return Matrix.Rotation(math.radians(a), 3, 'Y')
def Rx(a): return Matrix.Rotation(math.radians(a), 3, 'X')
def Rz(a): return Matrix.Rotation(math.radians(a), 3, 'Z')
def rot_axis(axis, ang): return Matrix.Rotation(math.radians(ang), 3, Vector(axis))


def rot_min(a, b):
    a = a.normalized(); b = b.normalized()
    if (a + b).length < 1e-6:
        perp = a.cross(Vector((0, 0, 1)))
        if perp.length < 1e-6: perp = a.cross(Vector((1, 0, 0)))
        return Matrix.Rotation(math.pi, 3, perp.normalized())
    return a.rotation_difference(b).to_matrix()


def slerp3(A, B, f):
    return A.to_quaternion().slerp(B.to_quaternion(), f).to_matrix()


def lerp(a, b, f): return a + (b - a) * f


def smooth(x):
    x = max(0.0, min(1.0, x)); return x * x * (3 - 2 * x)


# ---- time warp: the choreography below is authored on a 2.9 s clock; KNOTS stretch it to the final 5 s clip
# (slow breathing calm, slow grip + draw, a long coil before the cut, a longer hold and sheathe) while the slash itself
# keeps real-time speed.  (authored_time, final_time)
KNOTS = [(0, 0.0), (0.35, 0.9), (0.70, 1.55), (1.15, 2.45), (1.45, 3.05), (1.52, 3.35), (1.69, 3.52), (1.76, 3.7),
         (2.0, 4.0), (2.3, 4.4), (2.7, 4.8), (2.9, 5.0)]


def fwd(t):
    return pchip([(a, v) for a, v in KNOTS], t)


def inv(tn):
    lo, hi = 0.0, 2.9
    if tn <= 0: return 0.0
    if tn >= 5.0: return 2.9
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if fwd(mid) < tn: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)


def pchip(keys, t):
    """Monotone cubic Hermite through (t, v) keys; v may be float or Vector."""
    ts = [k[0] for k in keys]
    if t <= ts[0]: return keys[0][1]
    if t >= ts[-1]: return keys[-1][1]
    i = 0
    while ts[i + 1] < t: i += 1
    isv = not isinstance(keys[0][1], (int, float))
    comps = len(keys[0][1]) if isv else 1

    def val(k, c): return keys[k][1][c] if isv else keys[k][1]

    out = []
    for c in range(comps):
        vs = [val(k, c) for k in range(len(keys))]
        n = len(vs)
        h = [ts[k + 1] - ts[k] for k in range(n - 1)]
        d = [(vs[k + 1] - vs[k]) / h[k] for k in range(n - 1)]
        m = [0.0] * n
        for k in range(1, n - 1):
            if d[k - 1] * d[k] > 0:
                w1 = 2 * h[k] + h[k - 1]; w2 = h[k] + 2 * h[k - 1]
                m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])
        # end tangents: zero (ease in / out) unless explicitly continuing
        u = (t - ts[i]) / h[i]
        h00 = 2 * u ** 3 - 3 * u ** 2 + 1; h10 = u ** 3 - 2 * u ** 2 + u
        h01 = -2 * u ** 3 + 3 * u ** 2; h11 = u ** 3 - u ** 2
        out.append(h00 * vs[i] + h10 * h[i] * m[i] + h01 * vs[i + 1] + h11 * h[i] * m[i + 1])
    return Vector(out) if isv else out[0]


# ----------------------------------------------------------------------------------
# animation data (times in seconds)
# ----------------------------------------------------------------------------------
# torso -----------------------------------------------------------------------------
# yaw: + = turn to character's left (CCW from above). pitch: + = lean forward. roll: + = lean right
TORSO = {
    #            t     v
    'hipYaw':   [(0, 0), (0.35, 0), (0.70, 10), (1.15, 22), (1.30, 8), (1.45, -30), (1.52, -36),
                 (1.66, 42), (1.74, 46), (1.85, 44), (2.0, 44), (2.3, 18), (2.65, 6), (2.75, 4), (2.9, 4)],
    'hipPitch': [(0, 0), (0.7, 3), (1.15, 5), (1.45, 6), (1.52, 8), (1.66, 2), (1.74, 0), (2.0, 0), (2.4, 2), (2.75, 0), (2.9, 0)],
    'hipRoll':  [(0, 0), (1.15, 2), (1.45, -5), (1.52, -6), (1.66, 3), (1.80, 2), (2.0, 2), (2.4, 0), (2.9, 0)],
    'hipX':     [(0, 0), (0.7, 1), (1.15, 2), (1.45, -5), (1.52, -6), (1.66, 2), (1.8, 3), (2.0, 3), (2.5, 0), (2.9, 0)],
    'hipY':     [(0, 0), (0.35, 0), (0.7, -4), (1.15, -8), (1.45, -14), (1.52, -17), (1.66, -20), (1.74, -21),
                 (2.0, -20), (2.3, -14), (2.65, -9), (2.72, -10), (2.9, -9)],
    'hipZ':     [(0, 0), (0.7, -2), (1.15, -4), (1.45, -9), (1.52, -11), (1.66, 12), (1.74, 14), (2.0, 13), (2.4, 8), (2.9, 5)],
    'spYaw':    [(0, 0), (0.7, 4), (1.15, 8), (1.45, -12), (1.52, -16), (1.66, 18), (1.74, 20), (2.0, 20), (2.3, 6), (2.65, 2), (2.9, 2)],
    'spPitch':  [(0, 0), (1.45, -3), (1.52, -5), (1.66, 8), (2.0, 8), (2.5, 3), (2.9, 1)],
    'spRoll':   [(0, 0), (1.45, -3), (1.52, -4), (1.66, 5), (2.0, 5), (2.5, 1), (2.9, 0)],
    'chYaw':    [(0, 0), (0.7, 4), (1.15, 8), (1.45, -14), (1.52, -18), (1.66, 16), (1.74, 12), (2.0, 12), (2.3, 6), (2.65, 2), (2.9, 2)],
    'chPitch':  [(0, 0), (1.15, 2), (1.45, -5), (1.52, -7), (1.66, 12), (1.74, 13), (2.0, 12), (2.5, 4), (2.9, 2)],
    'chRoll':   [(0, 0), (1.45, -4), (1.52, -5), (1.66, 4), (2.0, 4), (2.5, 1), (2.9, 0)],
    # head: absolute (world) yaw / pitch
    'headYaw':  [(0, 0), (0.7, 6), (1.15, 8), (1.45, 0), (1.52, 0), (1.66, 22), (2.0, 26), (2.3, 10), (2.65, 6), (2.9, 0)],
    'headPitch': [(0, 0), (1.45, 4), (1.66, 6), (2.0, 4), (2.5, 3), (2.9, 0)],
}
TORSO_LEAD = {'hip': 0.035, 'sp': 0.018, 'ch': 0.0, 'head': -0.02}   # hips lead the cut, head lags

# feet: offsets (cm) of the toe point, yaw (deg), heel lift (deg)  -------------------
FEET = {
    'R': {
        'x': [(0, 0), (1.18, 0), (1.50, -3), (2.4, -3), (2.9, -1)],
        'y': [(0, 0), (1.18, 0), (1.30, 8), (1.42, 6), (1.50, 0), (2.9, 0)],
        'z': [(0, 0), (1.18, 0), (1.50, 30), (2.4, 30), (2.9, 14)],
        'yaw': [(0, 0), (1.18, 0), (1.5, -8), (2.4, -8), (2.9, -2)],
        'pitch': [(0, 0), (2.9, 0)],
    },
    'L': {
        'x': [(0, 0), (1.45, 0), (1.62, 4), (2.4, 4), (2.9, 1)],
        'y': [(0, 0), (2.9, 0)],
        'z': [(0, 0), (1.45, 0), (1.62, -10), (2.4, -10), (2.9, -4)],
        'yaw': [(0, 0), (1.45, 0), (1.66, 34), (2.4, 34), (2.9, 14)],
        'pitch': [(0, 0), (1.45, 0), (1.7, 28), (2.4, 28), (2.9, 4)],
    },
}

# saya (scabbard). Calm: worn on the left hip. Draw: the left hand swings it back/out.
SAYA_CALM_M = Vector((4, 36, 38))                       # mouth, hip-local (armature rest coords rel. to world)
SAYA_CALM_D = Vector((0.50, -0.22, -0.85)).normalized()
SAYA_HIP_M = Vector((30, 26, 16))                        # where the scabbard hangs once the left hand lets go
SAYA_HIP_D = Vector((0.18, -0.30, -0.93)).normalized()
SAYA_H = [(0, 0), (1.2, 0), (1.42, 1), (2.12, 1), (2.45, 0), (2.9, 0)]
SAYA_DRAW_M_C = Vector((30, -26, -6))                    # chest-local, mouth at end of draw
SAYA_DRAW_D_C = None                                     # computed from grip exit point
RIGHT_EXIT_C = Vector((-8, -8, 32))                      # chest-local right-hand grip at full draw
SAYA_S = [(0, 0), (0.70, 0), (0.92, 0.75), (1.15, 1), (1.34, 0), (2.10, 0), (2.30, 1), (2.50, 0.95), (2.69, 0), (2.9, 0)]   # 0 = on hip, 1 = held out
# withdrawn length of blade (cm): 0 sheathed, BLADE+2 = tip just leaving the mouth
WITHDRAW = [(0, 0), (0.74, 0), (0.95, 30), (1.15, BLADE + 2)]
INSERT = [(2.30, BLADE + 2), (2.40, BLADE + 1), (2.64, 3.0), (2.685, 0.0), (2.71, 1.4), (2.745, 0.0), (2.9, 0.0)]
T_EXIT = 1.15
T_ENTER = 2.30

# free-flight sword keys, chest-local (t, tsuba pos, blade dir, edge hint)
V = Vector
SWORD_FREE = [
    (1.15, 'saya'),
    (1.26, V((-12, 20, 30)), V((0.20, 0.70, 0.10)), V((0, 1, 0))),
    (1.34, V((-16, 36, 24)), V((-0.10, 0.90, -0.30)), V((0.5, 0.0, 1))),
    (1.45, V((-18, 46, 26)), V((-0.20, 0.88, -0.42)), V((0.5, 0.0, 1))),
    (1.52, V((-22, 50, 18)), V((-0.32, 0.86, -0.40)), V((0.5, 0.0, 1))),
    (1.585, V((-14, 38, 26)), V((-0.10, 0.55, 0.83)), V((0.6, -0.4, 0.4))),
    (1.62, V((-4, 18, 38)), V((0.15, 0.05, 0.99)), V((0.8, -0.5, 0.1))),
    (1.655, V((4, -2, 38)), V((0.45, -0.45, 0.77)), V((0.8, -0.6, 0.0))),
    (1.69, V((8, -6, 38)), V((0.40, -0.62, 0.67)), V((0.8, -0.6, 0.0))),
    (1.76, V((6, -2, 38)), V((0.35, -0.70, 0.60)), V((0.8, -0.6, 0.0))),
    (2.0, V((6, -2, 38)), V((0.35, -0.70, 0.60)), V((0.8, -0.6, 0.0))),
    (2.15, V((-6, 14, 30)), V((0.90, -0.30, 0.20)), V((0.2, 0.0, 1))),
    (2.30, 'saya'),
]

# hand blends: grip weight on the sword (right), left hand mode 0=saya 1=tsuka
GRIP_R = [(0, 0), (0.35, 0), (0.70, 1), (2.9, 1)]
LEFT_ON_TSUKA = [(0, 0), (1.15, 0), (1.40, 1), (1.74, 1), (1.98, 0), (2.9, 0)]
CURL_R = [(0, 0.15), (0.35, 0.15), (0.70, 1), (2.9, 1)]
CURL_L = [(0, 0.8), (2.9, 0.8)]
SH_FOLLOW = 0.22                             # how much the clavicle chases the hand
L_SAYA_U = 5.0                            # left fist position on the saya, cm in front of the tsuba
R_FREE_WRIST = Vector((-41, 24, 3))        # relaxed right hand next to the thigh


# ----------------------------------------------------------------------------------
# scene / rig helpers
# ----------------------------------------------------------------------------------
class Rig:
    def __init__(self, arm_obj):
        self.obj = arm_obj
        self.order = [b.name for b in arm_obj.data.bones]
        self.parent = {b.name: (b.parent.name if b.parent else None) for b in arm_obj.data.bones}
        self.rest = {b.name: b.matrix_local.copy() for b in arm_obj.data.bones}
        self.rh = {k: m.translation.copy() for k, m in self.rest.items()}
        self.rr = {k: m.to_3x3() for k, m in self.rest.items()}


def length(rig, a, b): return (rig.rh[a] - rig.rh[b]).length


def two_bone(root, mid_len, end_len, target, pole):
    """returns elbow/knee position for a 2-bone chain rooted at `root` reaching `target`."""
    to = target - root
    d = to.length
    clamped = False
    dmax = (mid_len + end_len) * 0.998
    dmin = abs(mid_len - end_len) * 1.02 + 0.01
    if d > dmax: d = dmax; clamped = True
    if d < dmin: d = dmin
    dirv = to.normalized()
    a = (mid_len ** 2 - end_len ** 2 + d ** 2) / (2 * d)
    h = math.sqrt(max(mid_len ** 2 - a ** 2, 0.0))
    perp = pole - dirv * pole.dot(dirv)
    if perp.length < 1e-5: perp = Vector((0, -1, 0)) - dirv * (-dirv.y)
    perp.normalize()
    return root + dirv * a + perp * h, root + dirv * d, clamped


# ----------------------------------------------------------------------------------
# katana modelling (armature-space cm, sword frame: x = edge, y = flat normal, z = blade direction)
# ----------------------------------------------------------------------------------
def make_materials():
    def mat(name, col, metal=0.0, rough=0.5):
        m = bpy.data.materials.new(name); m.use_nodes = True
        b = m.node_tree.nodes['Principled BSDF']
        b.inputs['Base Color'].default_value = (*col, 1)
        b.inputs['Metallic'].default_value = metal
        b.inputs['Roughness'].default_value = rough
        return m
    return {
        'steel': mat('kat_steel', (0.78, 0.80, 0.84), 1.0, 0.22),
        'tsuba': mat('kat_tsuba', (0.05, 0.045, 0.04), 0.9, 0.35),
        'gold': mat('kat_gold', (0.72, 0.55, 0.18), 1.0, 0.3),
        'wrap': mat('kat_wrap', (0.06, 0.07, 0.12), 0.0, 0.7),
        'lacquer': mat('kat_saya', (0.025, 0.02, 0.02), 0.0, 0.18),
    }


def lobed_ring(z, r, lobes=4, depth=0.5, n=40):
    return [((r + depth * math.cos(lobes * 2 * math.pi * i / n)) * math.cos(2 * math.pi * i / n),
             (r + depth * math.cos(lobes * 2 * math.pi * i / n)) * 0.93 * math.sin(2 * math.pi * i / n), z) for i in range(n)]


def blob(bm, c, size, mat_idx):
    res = bmesh.ops.create_cube(bm, size=1.0)
    vs = res['verts']
    for v in vs:
        v.co.x = v.co.x * size[0] + c[0]; v.co.y = v.co.y * size[1] + c[1]; v.co.z = v.co.z * size[2] + c[2]
    for f in {f for v in vs for f in v.link_faces}: f.material_index = mat_idx


def loft(bm, rings, mat_idx, cap0=True, cap1=True):
    vr = [[bm.verts.new(p) for p in ring] for ring in rings]
    n = len(vr[0])
    for a, b in zip(vr[:-1], vr[1:]):
        for i in range(n):
            j = (i + 1) % n
            f = bm.faces.new((a[i], a[j], b[j], b[i])); f.material_index = mat_idx
    if cap0:
        f = bm.faces.new(vr[0][::-1]); f.material_index = mat_idx
    if cap1:
        f = bm.faces.new(vr[-1]); f.material_index = mat_idx


def ellipse_ring(z, rx, ry, n=14, ox=0.0, oy=0.0):
    return [(ox + rx * math.cos(2 * math.pi * i / n), oy + ry * math.sin(2 * math.pi * i / n), z) for i in range(n)]


def build_katana(rig):
    mats = make_materials()
    slots = ['steel', 'tsuba', 'gold', 'wrap', 'lacquer']
    # ---- sword (blade + habaki + tsuba + tsuka)
    bm = bmesh.new()
    w, t = 3.0, 0.75
    segs = 40
    rings = []
    for i in range(segs + 1):
        s = i / segs
        z = 2.0 + (BLADE - 2.0) * s
        sc = 1.0 - 0.32 * s            # taper
        bulge = 1.5 * 4 * s * (1 - s)  # sori: edge side convex
        if i == segs:
            sc = 0.12
        ring = [(w / 2 * sc + bulge, 0, z), (-w / 4 * sc + bulge, t / 2 * sc, z), (-w / 2 * sc + bulge, t / 2 * sc, z),
                (-w / 2 * sc + bulge, -t / 2 * sc, z), (-w / 4 * sc + bulge, -t / 2 * sc, z)]
        rings.append(ring)
    loft(bm, rings, 0)
    loft(bm, [ellipse_ring(0.0, 2.4, 1.0, 10), ellipse_ring(2.6, 2.2, 0.9, 10)], 2)                 # habaki
    loft(bm, [lobed_ring(-0.6, 3.8), lobed_ring(0.0, 3.9), lobed_ring(0.5, 3.8)], 1)                                      # tsuba (mokko lobes)
    loft(bm, [ellipse_ring(0.5, 3.0, 2.7, 24), ellipse_ring(0.75, 2.9, 2.6, 24)], 2)                                       # gold rim ring
    loft(bm, [ellipse_ring(-0.6, 2.7, 2.3, 20), ellipse_ring(-0.95, 2.6, 2.2, 20)], 2)                                     # seppa
    loft(bm, [ellipse_ring(-0.95, 2.35, 1.85, 20), ellipse_ring(-4.2, 2.25, 1.75, 20)], 2)                                 # fuchi collar
    rings = []
    L = TSUKA
    for i in range(0, 81):
        z = -L * i / 80.0
        bump = 0.06 if (i % 2) else 0.0
        bell = 1.0 + 0.10 * math.sin(math.pi * i / 80.0)
        rings.append(ellipse_ring(z - 0.6, (1.75 + bump) * bell, (1.35 + bump) * bell, 20))
    loft(bm, rings, 3)                                                                              # tsuka (ito wrap)
    loft(bm, [ellipse_ring(-L - 0.6, 1.9, 1.5, 12), ellipse_ring(-L - 1.6, 1.9, 1.5, 12), ellipse_ring(-L - 2.1, 1.0, 0.8, 12)], 2)  # kashira
    blob(bm, (1.9, 0.0, -11.0), (0.5, 0.9, 2.6), 2); blob(bm, (-1.9, 0.0, -17.0), (0.5, 0.9, 2.6), 2)           # menuki
    for zz in (-8.5, -14.0, -19.5, -25.0): loft(bm, [ellipse_ring(zz, 1.95, 1.5, 20), ellipse_ring(zz - 0.5, 1.95, 1.5, 20)], 2)  # ito bands
    sword = bpy.data.meshes.new('Katana'); bm.to_mesh(sword); bm.free()
    # ---- saya
    bm = bmesh.new()
    rings = []
    SL = BLADE + 6
    for i in range(0, 49):
        s = i / 48.0
        z = 1.0 + SL * s
        rx = 2.15 - 0.55 * s; ry = 1.55 - 0.3 * s
        if s > 0.97: rx *= 0.5; ry *= 0.5
        rings.append(ellipse_ring(z, rx, ry, 24, ox=1.5 * 4 * s * (1 - s) * 0.95))
    loft(bm, rings, 4)
    loft(bm, [ellipse_ring(0.8, 2.5, 1.9, 14), ellipse_ring(2.6, 2.5, 1.9, 14)], 2)                   # koiguchi collar
    loft(bm, [ellipse_ring(SL - 0.5, 0.65, 0.5, 8), ellipse_ring(SL + 0.9, 0.5, 0.4, 8)], 2)            # kojiri cap
    for zz in (14.0, 30.0, 46.0): loft(bm, [ellipse_ring(zz, 2.3 - 0.0005 * zz, 1.75, 24), ellipse_ring(zz + 1.1, 2.3 - 0.0005 * zz, 1.75, 24)], 2)   # gold bands
    blob(bm, (2.4, 0.0, 8.0), (1.6, 1.2, 2.4), 2)                                                                            # kurikata knob
    saya = bpy.data.meshes.new('Saya'); bm.to_mesh(saya); bm.free()

    # ---- armature bones
    arm = rig.obj
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='EDIT')
    eb = arm.data.edit_bones
    for nm in ('katana_root', 'saya_root'):
        b = eb.new(nm); b.head = (0, 0, 0); b.tail = (0, 0, 10); b.roll = 0
        b.use_deform = True
    bpy.ops.object.mode_set(mode='OBJECT')

    objs = {}
    for nm, me, bone in (('Katana', sword, 'katana_root'), ('Saya', saya, 'saya_root')):
        ob = bpy.data.objects.new(nm, me)
        bpy.context.scene.collection.objects.link(ob)
        for k in slots: me.materials.append(mats[k])
        ob.parent = arm
        ob.matrix_parent_inverse = Matrix.Identity(4)
        vg = ob.vertex_groups.new(name=bone)
        vg.add([v.index for v in me.vertices], 1.0, 'REPLACE')
        mod = ob.modifiers.new('Armature', 'ARMATURE'); mod.object = arm
        for p in me.polygons: p.use_smooth = True
        objs[nm] = ob
    return objs


# ----------------------------------------------------------------------------------
# the solver
# ----------------------------------------------------------------------------------
class Solver:
    def __init__(self, rig):
        self.rig = rig
        rh = rig.rh
        self.arm_len = {s: (length(rig, f'{s}_arm', f'{s}_forarm'), length(rig, f'{s}_forarm', f'{s}_hand')) for s in 'LR'}
        self.leg_len = {s: (length(rig, f'{s}_leg', f'{s}_knee'), length(rig, f'{s}_knee', f'{s}_ankle')) for s in 'LR'}
        self.warn = []
        # grip point in hand-rest coordinates (fist centre)
        self.pg = {}
        for s in 'LR':
            m = rh[f'{s}_m']; h = rh[f'{s}_hand']
            self.pg[s] = (m - h) + Vector((0, -2.6, 0))
        self.exit_D_c = (SAYA_DRAW_M_C - RIGHT_EXIT_C).normalized()
        self.h0 = {}
        self.hand0 = {}
        for s_ in 'LR':
            d0 = (rh[f'{s_}_forarm'] - rh[f'{s_}_arm']).normalized()
            d0b = (rh[f'{s_}_hand'] - rh[f'{s_}_forarm']).normalized()
            self.h0[s_] = d0.cross(d0b).normalized() if d0.cross(d0b).length > 0.05 else d0.cross(Vector((0, 0, 1))).normalized()
            self.hand0[s_] = (rh[f'{s_}_m'] - rh[f'{s_}_hand']).normalized()
        self.met = {}
        # sword key times for chest-local interpolation (computed per solve)

    # -- sword/saya frames in chest-local needs the torso; do it per-frame ------------
    def torso(self, t):
        g = lambda ch, lead: pchip(TORSO[ch], t + lead)
        L = TORSO_LEAD
        hip = dict(yaw=g('hipYaw', L['hip']), pitch=g('hipPitch', L['hip']), roll=g('hipRoll', L['hip']),
                   pos=Vector((g('hipX', L['hip']), g('hipY', L['hip']), g('hipZ', L['hip']))))
        sp = dict(yaw=g('spYaw', L['sp']), pitch=g('spPitch', L['sp']), roll=g('spRoll', L['sp']))
        ch = dict(yaw=g('chYaw', L['ch']), pitch=g('chPitch', L['ch']), roll=g('chRoll', L['ch']))
        hd = dict(yaw=g('headYaw', L['head']), pitch=g('headPitch', L['head']))
        # breathing / idle life
        br = math.sin(fwd(min(max(t, 0.0), 2.9)) * 2 * math.pi / 3.4)
        ch['pitch'] += 0.7 * br; hip['pos'] = hip['pos'] + Vector((0, 0.25 * br, 0))
        return hip, sp, ch, hd

    @staticmethod
    def eul(d): return Ry(d['yaw']) @ Rx(d['pitch']) @ Rz(d.get('roll', 0))

    def sword_frame(self, P, D, hint):
        D = D.normalized()
        E = hint - D * hint.dot(D)
        if E.length < 1e-4: E = Vector((0, 1, 0)) - D * D.y
        E.normalize()
        N = D.cross(E)
        M = Matrix(((E.x, N.x, D.x), (E.y, N.y, D.y), (E.z, N.z, D.z)))
        return P, M

    def solve(self, t, ctx_cache, opt=None):
        t = inv(t)
        opt = opt or {}
        sw = opt.get('swiv', (0.0, 0.0)); twv = opt.get('twist', (0.0, 0.0)); lfv = opt.get('lift', (0.0, 0.0))
        shf = opt.get('shf', (SH_FOLLOW, SH_FOLLOW))
        flp = opt.get('flip', (1, 1))
        rig = self.rig
        rh, rr = rig.rh, rig.rr
        hipP, spP, chP, hdP = self.torso(t)
        POS, DEL = {}, {}

        def place(b, Dabs=None, Dloc=None, pos=None):
            p = rig.parent[b]
            if pos is None:
                pos = POS[p] + DEL[p] @ (rh[b] - rh[p]) if p else rh[b].copy()
            POS[b] = pos
            if Dabs is not None: DEL[b] = Dabs
            elif Dloc is not None: DEL[b] = (DEL[p] if p else I3) @ Dloc
            else: DEL[b] = DEL[p] if p else I3.copy()

        # ---------- torso
        Dh = self.eul(hipP)
        place('HIP', Dabs=Dh, pos=rh['HIP'] + hipP['pos'])
        place('spine', Dloc=self.eul(spP))
        place('chest', Dloc=self.eul(chP))
        Dchest, Pchest = DEL['chest'], POS['chest']
        Dhead = Ry(hdP['yaw']) @ Rx(hdP['pitch'])
        place('neck', Dabs=slerp3(Dchest, Dhead, 0.45))
        place('head', Dabs=Dhead)

        # ---------- feet targets (needed early: they define ground contact)
        feet = {}
        for s in 'LR':
            f = FEET[s]
            off = Vector((pchip(f['x'], t), pchip(f['y'], t), pchip(f['z'], t)))
            yaw = pchip(f['yaw'], t); pitch = pchip(f['pitch'], t)
            feet[s] = (off, yaw, pitch)

        # ---------- chest-local helpers
        def c2w(v): return Pchest + Dchest @ v
        def cd2w(v): return Dchest @ v

        # ---------- saya
        sS = pchip(SAYA_S, t)
        sS = smooth(sS) if 0 < sS < 1 else sS
        hh = smooth(pchip(SAYA_H, t))
        Mh = POS['HIP'] + Dh @ (lerp(SAYA_CALM_M, SAYA_HIP_M, hh) - rh['HIP'])
        Dh_s = Dh @ lerp(SAYA_CALM_D, SAYA_HIP_D, hh).normalized()
        Mc = c2w(SAYA_DRAW_M_C - rh['chest']) if False else Pchest + Dchest @ SAYA_DRAW_M_C
        Dc_s = Dchest @ self.exit_D_c
        Msaya = lerp(Mh, Mc, sS)
        Dsaya = lerp(Dh_s, Dc_s, sS).normalized()
        Psaya, Fsaya = self.sword_frame(Msaya, Dsaya, Vector((0, 1, 0)) * (1 - sS) + Dchest @ Vector((0.2, 1, 0.1)) * sS)

        # ---------- sword
        def sword_from_saya(withdraw, Msy, Dsy, Fsy):
            P = Msy - Dsy * withdraw
            return P, Dsy, Fsy.col[0].to_3d()

        def sword_state(tt):
            """returns absolute (P, D, E-hint) for sword at time tt"""
            # which regime
            if tt <= T_EXIT: w = pchip(WITHDRAW, tt)
            elif tt >= T_ENTER: w = pchip(INSERT, tt)
            else: w = None
            if w is not None:
                return ('saya', w)
            return ('free', None)

        # free-flight interpolation needs keys in chest-local; saya keys converted using torso/saya at that key time
        free = ctx_cache.get('free')
        if free is None:
            free = []
            for k in SWORD_FREE:
                if k[1] == 'saya':
                    free.append((k[0], 'saya'))
                else:
                    free.append((k[0], k[1], k[2], k[3]))
            ctx_cache['free'] = free
        if 'saya_keys' not in ctx_cache:
            ctx_cache['saya_keys'] = {}
        if t <= T_EXIT or t >= T_ENTER:
            w = pchip(WITHDRAW, t) if t <= T_EXIT else pchip(INSERT, t)
            P = Msaya - Dsaya * w
            D = Dsaya
            Fm = Fsaya
        else:
            # build chest-local key list; the 'saya' ones evaluated at their own times
            keys_P, keys_D, keys_H = [], [], []
            for k in free:
                if k[1] == 'saya':
                    tk = k[0]
                    ck = ctx_cache['saya_keys'].get(tk)
                    if ck is None:
                        ck = self._saya_key_chest_local(tk)
                        ctx_cache['saya_keys'][tk] = ck
                    keys_P.append((tk, ck[0])); keys_D.append((tk, ck[1])); keys_H.append((tk, ck[2]))
                else:
                    keys_P.append((k[0], k[1])); keys_D.append((k[0], k[2].normalized())); keys_H.append((k[0], k[3].normalized()))
            Pc = pchip(keys_P, t)
            Dc = pchip(keys_D, t).normalized()
            Hc = pchip(keys_H, t)
            P = Pchest + Dchest @ Pc
            D = Dchest @ Dc
            Fm = self.sword_frame(P, D, Dchest @ Hc)[1]
        P, Fm = self.sword_frame(P, D, Fm.col[0].to_3d() if hasattr(Fm, 'col') else Fm)

        # fix: when in saya regime make the edge face up/outward via the saya frame
        place_katana = (P, Fm)

        # ---------- shoulders / arms
        hands = {}
        # right-hand target
        gR = smooth(pchip(GRIP_R, t))
        Gr_sword = P - D * R_GRIP_U
        Gl_tsuka = P - D * L_GRIP_U
        Gl_saya = Msaya + Dsaya * L_SAYA_U
        bL = smooth(pchip(LEFT_ON_TSUKA, t))
        sh_pos = {}

        def shoulder(s, wrist_t):
            sh = f'{s}_shoulder'
            place(sh, Dabs=Dchest)
            arm_head_rest = rh[f'{s}_arm']
            # reach direction compared to rest
            S0 = POS[sh] + DEL[sh] @ (rh[f'{s}_arm'] - rh[sh])
            wr = rh[f'{s}_hand']
            A0 = (Dchest @ (wr - rh[f'{s}_arm']))
            A1 = (wrist_t - S0)
            q = rot_min(A0, A1)
            Dsh = slerp3(I3, q, shf[0 if s == 'L' else 1]) @ Dchest
            place(sh, Dabs=Dsh)
            return Dsh

        def grip_pose(s, G, A, shoulder_pos, fhint, pole_):
            """hand delta (3x3) so the fist wraps handle axis A at grip point G; returns (Delta, wrist).
            The fingers are aimed along the forearm (elbow -> grip) so the wrist stays close to neutral."""
            pg = self.pg[s]
            z = Vector((0, 0, 1))
            p_perp = pg - z * pg.dot(z); p_perp.normalize()
            r = (G - shoulder_pos); r = r - A * r.dot(A)
            if r.length < 1e-3: r = fhint - A * fhint.dot(A)
            r.normalize()
            a_, b_ = self.arm_len[s]
            W0 = G - r * (pg - z * pg.dot(z)).length
            ax0 = W0 - shoulder_pos
            if ax0.length > 1e-3:
                E0, _, _ = two_bone(shoulder_pos, a_, b_, W0, rot_axis(ax0.normalized(), sw[0 if s == 'L' else 1]) @ pole_)
                r2 = G - E0; r2 = r2 - A * r2.dot(A)
                if r2.length > 1e-3: r = r2.normalized()
            r = rot_axis(A, twv[0 if s == 'L' else 1]) @ r
            fs = max(-1.0, min(1.0, flp[0 if s == 'L' else 1]))
            wflip = (1.0 - fs) * 0.5
            # rest frame (z, p_perp, z x p_perp) -> target frame (A, r, A x r)
            B0 = Matrix((z, p_perp, z.cross(p_perp))).transposed()
            Dl = (Matrix((A, r, A.cross(r))).transposed()) @ B0.inverted()
            if wflip > 1e-3:
                Dm = (Matrix((-A, r, (-A).cross(r))).transposed()) @ B0.inverted()
                Dl = slerp3(Dl, Dm, wflip)
            return Dl, G - Dl @ pg

        # right side
        sR = Vector((0, 0, 0))
        liftR = Dchest @ Vector((0, 8, 12)) * (lfv[1] * 4 * gR * (1 - gR))
        DshR_tmp = shoulder('R', lerp(R_FREE_WRIST, Gr_sword, gR))
        armR_head = POS['R_shoulder'] + DEL['R_shoulder'] @ (rh['R_arm'] - rh['R_shoulder'])
        polR = Dchest @ Vector((-0.5, -1.0, -0.7)); polL = Dchest @ Vector((0.5, -1.0, -0.7))
        DR_g, WR_g = grip_pose('R', Gr_sword, D, armR_head, Vector((0, -1, 0.3)), polR)
        DR_free = DEL['R_shoulder']
        Wr = lerp(R_FREE_WRIST, WR_g, gR) + liftR
        Dhand_R = slerp3(DR_free, DR_g, gR)
        # left side
        Gl_s = Gl_saya; Gl_t = Gl_tsuka
        G_l = lerp(Gl_s, Gl_t, bL) + Dchest @ Vector((0, 8, 12)) * (lfv[0] * 4 * bL * (1 - bL))
        A_l = (lerp(Dsaya, D, bL)).normalized()
        DshL_tmp = shoulder('L', G_l)
        armL_head = POS['L_shoulder'] + DEL['L_shoulder'] @ (rh['L_arm'] - rh['L_shoulder'])
        DL_g, WL_g = grip_pose('L', G_l, A_l, armL_head, Vector((0, -1, 0.3)), polL)

        def solve_arm(s, wrist, Dhand, pole, Ahand=None):
            a, b = self.arm_len[s]
            Sh = POS[f'{s}_shoulder'] + DEL[f'{s}_shoulder'] @ (rh[f'{s}_arm'] - rh[f'{s}_shoulder'])
            ax_ = (wrist - Sh)
            swv_ = sw[0 if s == 'L' else 1]
            if ax_.length > 1e-4:
                pole = rot_axis(ax_.normalized(), swv_) @ pole
            elbow, wr_pos, clamp = two_bone(Sh, a, b, wrist, pole)
            if clamp and (wrist - Sh).length - (a + b) > 2.0:
                self.warn.append((round(t, 3), s, round((wrist - Sh).length - (a + b), 1)))
            sh = f'{s}_shoulder'
            d0 = (rh[f'{s}_forarm'] - rh[f'{s}_arm']).normalized()
            d1 = (elbow - Sh).normalized()
            d0b = (rh[f'{s}_hand'] - rh[f'{s}_forarm']).normalized()
            d1b = (wr_pos - elbow).normalized()
            Dsw = rot_min(DEL[sh] @ d0, d1) @ DEL[sh]
            # humerus roll: make the elbow hinge axis perpendicular to the plane (shoulder, elbow, wrist)
            roll = 0.0
            n_ = d1.cross(d1b)
            if n_.length > 0.08:
                n_.normalize()
                hs = Dsw @ self.h0[s]
                hs = hs - d1 * hs.dot(d1); n2 = n_ - d1 * n_.dot(d1)
                if hs.length > 1e-4 and n2.length > 1e-4:
                    hs.normalize(); n2.normalize()
                    roll = math.atan2(d1.dot(hs.cross(n2)), hs.dot(n2))
            Da = Matrix.Rotation(roll, 3, d1) @ Dsw
            place(f'{s}_arm', Dabs=Da)
            Df = rot_min(Da @ d0b, d1b) @ Da
            # forearm pronation / supination: take most of the twist, wrist keeps the (small) remainder
            q = (Dhand @ Df.inverted()).to_quaternion()
            ang = 2 * math.atan2(Vector((q.x, q.y, q.z)).dot(d1b), q.w)
            if ang > math.pi: ang -= 2 * math.pi
            if ang < -math.pi: ang += 2 * math.pi
            lim = math.radians(85)
            applied = max(-lim, min(lim, ang * 0.75))
            Df = Matrix.Rotation(applied, 3, d1b) @ Df
            place(f'{s}_forarm', Dabs=Df)
            place(f'{s}_hand', Dabs=Dhand, pos=wr_pos)
            elbow_int = math.degrees(math.acos(max(-1, min(1, (Sh - elbow).normalized().dot((wr_pos - elbow).normalized())))))
            dev = math.degrees(math.acos(max(-1, min(1, (Dhand @ self.hand0[s]).dot(d1b)))))
            gang = 90.0 if Ahand is None else math.degrees(math.acos(max(0.0, min(1.0, abs(Ahand.normalized().dot(d1b))))))
            self.met[s] = (elbow_int, abs(math.degrees(applied)), abs(math.degrees(ang - applied)), dev, abs(math.degrees(roll)), gang)

        solve_arm('R', Wr, Dhand_R, polR, D if gR > 0.5 else None)
        solve_arm('L', WL_g, DL_g, polL, A_l)

        # fingers: wrap around the handle (fitted) or relaxed
        FB = {'m': (62, 78, 50), 'f': (62, 78, 50), 'r': (66, 80, 52), 'l': (70, 82, 55)}

        def finger_joints(s, fn, c, Dhand, wr):
            ax = Vector((0, 0, -1)) if s == 'L' else Vector((0, 0, 1))
            b0, b1, b2 = FB[fn]
            p0 = wr + Dhand @ (rh[f'{s}_{fn}'] - rh[f'{s}_hand'])
            D0 = Dhand @ rot_axis(ax, b0 * c)
            p1 = p0 + D0 @ (rh[f'{s}_{fn}1'] - rh[f'{s}_{fn}'])
            D1 = D0 @ rot_axis(ax, b1 * c)
            p2 = p1 + D1 @ (rh[f'{s}_{fn}2'] - rh[f'{s}_{fn}1'])
            D2 = D1 @ rot_axis(ax, b2 * c)
            p3 = p2 + D2 @ (rh[f'{s}_{fn}3'] - rh[f'{s}_{fn}2'])
            return (p1, p2, p3)

        def fit_curl(s, fn, Dhand, wr, G, A, rad):
            best, bc = 1e9, 0.8
            for k in range(14):
                c = 0.25 + 0.1 * k
                err = 0.0
                for p in finger_joints(s, fn, c, Dhand, wr):
                    v = p - G; d = (v - A * v.dot(A)).length
                    e = d - (rad + 0.65)
                    err += e * e * (3.0 if e < 0 else 1.0)
                if err < best: best, bc = err, c
            return bc

        def fingers(s, curl, Dhand, wr, wrapG=None, wrapA=None, rad=1.6, w=0.0):
            ax = Vector((0, 0, -1)) if s == 'L' else Vector((0, 0, 1))
            cs = []
            for fn, (b0, b1, b2) in FB.items():
                c = curl
                if wrapG is not None and w > 0:
                    c = lerp(curl, fit_curl(s, fn, Dhand, wr, wrapG, wrapA, rad), w)
                cs.append(c)
                D0 = Dhand @ rot_axis(ax, b0 * c)
                place(f'{s}_{fn}', Dabs=D0)
                D1 = D0 @ rot_axis(ax, b1 * c); place(f'{s}_{fn}1', Dabs=D1)
                D2 = D1 @ rot_axis(ax, b2 * c); place(f'{s}_{fn}2', Dabs=D2)
                place(f'{s}_{fn}3', Dabs=D2)
            cm = sum(cs) / len(cs)
            ta = Vector((0, 0.5, -1)).normalized() if s == 'L' else Vector((0, 0.5, 1)).normalized()
            T0 = Dhand @ rot_axis(ta, 25 * cm)
            place(f'{s}_t', Dabs=T0)
            T1 = T0 @ rot_axis(ax, 30 * cm); place(f'{s}_t1', Dabs=T1)
            T2 = T1 @ rot_axis(ax, 25 * cm); place(f'{s}_t2', Dabs=T2)

        wrR = POS['R_hand']; wrL = POS['L_hand']
        fingers('R', pchip(CURL_R, t), Dhand_R, wrR, Gr_sword, D, 1.6, gR)
        fingers('L', pchip(CURL_L, t), DL_g, wrL, G_l, A_l, 1.6 + 0.4 * (1 - bL), 1.0)

        # ---------- legs
        for s in 'LR':
            off, yaw, pitch = feet[s]
            toe0 = rh[f'{s}_foottop']
            foot0 = rh[f'{s}_foot']; ank0 = rh[f'{s}_ankle']
            Df = Ry(yaw) @ Rx(pitch)
            toe = toe0 + off
            ank_t = toe + Df @ (ank0 - toe0)
            # leg
            leg = f'{s}_leg'
            place(leg, Dloc=I3)
            Th = POS[leg]
            a, b = self.leg_len[s]
            pole = Df @ Vector((0.25 if s == 'L' else -0.25, 0.0, 1.0))
            knee, ank_pos, clamp = two_bone(Th, a, b, ank_t, pole)
            if clamp and (ank_t - Th).length - (a + b) > 2.0:
                self.warn.append((round(t, 3), s + 'leg', round((ank_t - Th).length - (a + b), 1)))
            d0 = (rh[f'{s}_knee'] - rh[leg]).normalized()
            Dl = rot_min(DEL['HIP'] @ d0, (knee - Th).normalized()) @ DEL['HIP']
            place(leg, Dabs=Dl)
            d0b = (rh[f'{s}_ankle'] - rh[f'{s}_knee']).normalized()
            Dk = rot_min(Dl @ d0b, (ank_pos - knee).normalized()) @ Dl
            place(f'{s}_knee', Dabs=Dk, pos=knee)
            place(f'{s}_ankle', Dabs=Df, pos=ank_pos)
            place(f'{s}_foot', Dabs=Df)
            place(f'{s}_foottop', Dabs=Df)

        # ---------- everything else follows its parent (with light secondary motion)
        for b in rig.order:
            if b not in POS:
                place(b)
        return POS, DEL, place_katana, (Msaya, Dsaya, Fsaya)

    def _saya_key_chest_local(self, tk):
        """Evaluate the sheathed-sword pose at time tk, return chest-local (P, D, hint)."""
        hipP, spP, chP, hdP = self.torso(tk)
        Dh = self.eul(hipP)
        rig = self.rig
        Phip = rig.rh['HIP'] + hipP['pos']
        Dspine = Dh @ self.eul(spP)
        Dchest = Dspine @ self.eul(chP)
        # chest position
        Pspine = Phip + Dh @ (rig.rh['spine'] - rig.rh['HIP'])
        Pchest = Pspine + Dspine @ (rig.rh['chest'] - rig.rh['spine'])
        sS = pchip(SAYA_S, tk)
        sS = smooth(sS) if 0 < sS < 1 else sS
        hh = smooth(pchip(SAYA_H, tk))
        Mh = Phip + Dh @ (lerp(SAYA_CALM_M, SAYA_HIP_M, hh) - rig.rh['HIP'])
        Mc = Pchest + Dchest @ SAYA_DRAW_M_C
        M = lerp(Mh, Mc, sS)
        D = lerp(Dh @ lerp(SAYA_CALM_D, SAYA_HIP_D, hh).normalized(), Dchest @ self.exit_D_c, sS).normalized()
        w = pchip(WITHDRAW, tk) if tk <= T_EXIT else pchip(INSERT, tk)
        P = M - D * w
        Hn = Vector((0, 1, 0)) * (1 - sS) + Dchest @ Vector((0.2, 1, 0.1)) * sS
        Pc = Dchest.inverted() @ (P - Pchest)
        Dc = Dchest.inverted() @ D
        Hc = Dchest.inverted() @ Hn
        return Pc, Dc, Hc.normalized()


# ----------------------------------------------------------------------------------
# baking
# ----------------------------------------------------------------------------------

def rig_tables(rig, arm):
    allb = rig.order + ['katana_root', 'saya_root']
    parent = dict(rig.parent); parent['katana_root'] = None; parent['saya_root'] = None
    rest = dict(rig.rest)
    for nm in ('katana_root', 'saya_root'):
        rest[nm] = arm.data.bones[nm].matrix_local.copy()
    rr = {k: m.to_3x3() for k, m in rest.items()}
    return allb, parent, rest, rr


def set_pose(arm, tabs, POS, DEL, Pk, Fk, Ms, Fs):
    allb, parent, rest, rr = tabs
    POS = dict(POS); DEL = dict(DEL)
    POS['katana_root'] = Pk; DEL['katana_root'] = Fk
    POS['saya_root'] = Ms; DEL['saya_root'] = Fs
    M = {}
    for b in allb:
        R4 = (DEL[b] @ rr[b]).to_4x4(); R4.translation = POS[b]
        M[b] = R4
        p = parent[b]
        base = (M[p] @ rest[p].inverted() @ rest[b]) if p else rest[b]
        arm.pose.bones[b].matrix_basis = base.inverted() @ R4
    bpy.context.view_layer.update()


def anat_pen(m, swiv):
    """soft joint-limit cost: elbow flexion, forearm pronation, wrist deviation/twist, humerus roll, elbow swivel."""
    elbow_int, applied, resid, dev, roll, gang = m
    return (0.012 * max(0.0, 50.0 - gang) ** 2 + 0.02 * max(0.0, 35.0 - elbow_int) ** 2 + 0.004 * max(0.0, applied - 70.0) ** 2 + 0.01 * max(0.0, resid - 20.0) ** 2
            + 0.006 * max(0.0, dev - 50.0) ** 2 + 0.004 * max(0.0, roll - 80.0) ** 2 + 0.02 * max(0.0, abs(swiv) - 30.0) ** 2)


def opt_setup(quick):
    SW = (-50, -30, -15, 0, 15, 30, 45, 60)
    TW = (-120, -90, -60, -30, 0, 30, 60, 90, 120)
    LF = (0.0, 1.0)
    SF = (0.15, 0.5)
    step = 4
    if quick:
        SW = (-45, -20, 0, 20, 45); TW = (-120, -80, -40, 0, 40, 80, 120); SF = (0.15, 0.55); step = 4
    import itertools
    cands = list(itertools.product(range(len(SW)), range(len(TW)), range(len(LF)), range(len(SF)), range(2)))
    idx = list(range(0, N, step))
    if idx[-1] != N - 1: idx.append(N - 1)
    return SW, TW, LF, SF, cands, idx


def opt_costs(rig, solver, arm, frames, setup):
    """clip + joint-limit cost of every (swivel, twist, lift, clavicle) candidate at the given frames."""
    import clips
    SW, TW, LF, SF, cands, idx = setup
    scorer = clips.Scorer()
    tabs = rig_tables(rig, arm)
    cache = {}
    out = {}
    for i in frames:
        t = i / FPS
        rowL, rowR = [], []
        for (a_, b_, c_, d_, e_) in cands:
            fl_ = 1 if e_ == 0 else -1
            o = {'swiv': (SW[a_],) * 2, 'twist': (TW[b_],) * 2, 'lift': (LF[c_],) * 2, 'shf': (SF[d_],) * 2, 'flip': (fl_, fl_)}
            POS, DEL, (Pk, Fk), (Ms, Ds, Fs) = solver.solve(t, cache, o)
            set_pose(arm, tabs, POS, DEL, Pk, Fk, Ms, Fs)
            T = scorer.trees(bpy.context.evaluated_depsgraph_get())
            pen = abs(SW[a_]) * 0.03 + abs(TW[b_]) * 0.03 + LF[c_] * 1.0 + abs(SF[d_] - 0.22) * 2
            rowL.append(scorer.side_cost(T, 'L') + pen + 7 * anat_pen(solver.met['L'], SW[a_]))
            rowR.append(scorer.side_cost(T, 'R') + pen + 7 * anat_pen(solver.met['R'], SW[a_]))
        out[i] = (rowL, rowR)
        print('  opt frame', i, 'minL', round(min(rowL), 1), 'minR', round(min(rowR), 1), flush=True)
    return out


def optimize_swivel(rig, solver, arm, fbx, outdir, quick=False, workers=4):
    """per-frame search over (elbow swivel, wrist twist, path lift, clavicle follow) against the real deformed meshes
    and joint limits, evaluated in parallel worker processes, then Viterbi-smoothed in time."""
    import numpy as np, json, subprocess
    setup = opt_setup(quick)
    SW, TW, LF, SF, cands, idx = setup
    costs = {}
    if workers > 1:
        procs = []
        for k in range(workers):
            cmd = [sys.executable, '-I', os.path.abspath(__file__), fbx, outdir, '--worker', f'{k}/{workers}']
            if quick: cmd.append('--quick')
            procs.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        for p in procs: p.wait()
        for k in range(workers):
            with open(os.path.join(outdir, f'_cost_{k}.json')) as f:
                for key, v in json.load(f).items(): costs[int(key)] = v
            os.remove(os.path.join(outdir, f'_cost_{k}.json'))
    else:
        costs = opt_costs(rig, solver, arm, idx, setup)
    out = {'swiv': {}, 'twist': {}, 'lift': {}, 'shf': {}, 'flip': {}}
    lam = (5.0, 3.0, 4.0)
    K = len(cands)
    ca = np.array(cands)
    trans = (lam[0] * np.abs(ca[:, None, 0] - ca[None, :, 0]) + lam[1] * np.abs(ca[:, None, 1] - ca[None, :, 1])
             + lam[2] * 2 * np.abs(ca[:, None, 2] - ca[None, :, 2]) + 6.0 * np.abs(ca[:, None, 3] - ca[None, :, 3]) + 60.0 * np.abs(ca[:, None, 4] - ca[None, :, 4])).astype(np.float64)
    for si, s in enumerate('LR'):
        C = np.array([costs[i][si] for i in idx], np.float64); n = len(idx)
        best = C[0].copy(); back = []
        for f in range(1, n):
            tot = best[None, :] + trans
            m = tot.argmin(axis=1)
            best = C[f] + tot[np.arange(K), m]
            back.append(m)
        j = int(best.argmin()); path = [j]
        for bk in reversed(back):
            j = int(bk[j]); path.append(j)
        path.reverse()
        print('opt', s, 'residual', float(sum(C[f][path[f]] for f in range(n))))
        for d, vals in enumerate((SW, TW, LF, SF, (1, -1))):
            ang = [vals[cands[j][d]] for j in path]
            full = []
            for i in range(N):
                f = min(max(bisect.bisect_right(idx, i) - 1, 0), n - 2)
                u = (i - idx[f]) / float(idx[f + 1] - idx[f])
                full.append(lerp(ang[f], ang[f + 1], u))
            sm = []
            for i in range(N):
                rr = 5 if d == 4 else 3
                w = [full[min(max(i + dd, 0), N - 1)] for dd in range(-rr, rr + 1)]
                sm.append(sum(w) / len(w))
            out[('swiv', 'twist', 'lift', 'shf', 'flip')[d]][s] = sm
    return out


def report_anatomy(solver, swt):
    cache = {}
    worst = {}
    print('t     | L: elbow  pron  wristTw  wristDev  humRoll gripAng | R: elbow  pron  wristTw  wristDev  humRoll gripAng')
    for i in range(0, N, 6):
        t = i / FPS
        solver.solve(t, cache, {k: (v['L'][i], v['R'][i]) for k, v in swt.items()} if swt else None)
        mL, mR = solver.met['L'], solver.met['R']
        print(f'{t:4.2f}  | ' + ' '.join(f'{x:6.0f}' for x in mL) + '   | ' + ' '.join(f'{x:6.0f}' for x in mR))
        for k, m in (('L', mL), ('R', mR)):
            w = worst.setdefault(k, [999, 0, 0, 0, 0, 999])
            w[0] = min(w[0], m[0]); w[5] = min(w[5], m[5])
            for j in range(1, 5): w[j] = max(w[j], m[j])
    print('WORST  (min elbow interior, max pron, max wrist twist, max wrist dev, max roll):', worst)


frames_orig = None


def secondary_motion(rig, frames, amp=0.22, clamp=0.30):
    """damped-pendulum follow-through for the hanging skirt panels (hujia*) and belt cords (rope*): driven by the
    acceleration of each chain root, so they lag the cut, swing on the stop and settle."""
    rh, rr = rig.rh, rig.rr
    chain = [b for b in rig.order if ('hujia' in b or b.startswith('rope'))]
    roots = [b for b in chain if rig.parent[b] not in chain]
    members = {}
    for r_ in roots:
        members[r_] = [b for b in chain if b == r_ or _is_desc(rig, b, r_)]
    dt = 1.0 / FPS
    n = len(frames)
    ang = {r_: [(0.0, 0.0)] * n for r_ in roots}
    for r_ in roots:
        w0 = 2 * math.pi * (2.3 if 'rope' not in r_ else 3.2); zeta = 0.30
        p = [Vector(f[0][r_]) for f in frames]
        fx = fz = vx = vz = 0.0
        out = []
        for i in range(n):
            a = (p[min(i + 1, n - 1)] - 2 * p[i] + p[max(i - 1, 0)]) / (dt * dt) if 0 < i < n - 1 else Vector((0, 0, 0))
            tz = max(-clamp, min(clamp, -a.x / 981.0 * amp * 4))      # rotation about Z -> swing in X
            tx = max(-clamp, min(clamp, a.z / 981.0 * amp * 4))       # rotation about X -> swing in Z
            for _ in range(2):                                         # 2 substeps
                h = dt / 2
                vz += (-w0 * w0 * (fz - tz) - 2 * zeta * w0 * vz) * h; fz += vz * h
                vx += (-w0 * w0 * (fx - tx) - 2 * zeta * w0 * vx) * h; fx += vx * h
            fz = max(-clamp, min(clamp, fz)); fx = max(-clamp, min(clamp, fx))
            out.append((fx, fz))
        ang[r_] = out
    for i, (POS, DEL) in enumerate(frames):
        for b in rig.order:
            if b not in chain: continue
            par = rig.parent[b]
            root = next(r_ for r_ in roots if b in members[r_])
            depth = 0 if b == root else 1
            fx, fz = ang[root][i]
            k = 1.0 if depth == 0 else 0.5
            R = Matrix.Rotation(fx * k, 3, 'X') @ Matrix.Rotation(fz * k, 3, 'Z')
            # local delta of b relative to parent under the original pose
            oldp = frames_orig[i][1][par]; oldb = frames_orig[i][1][b]
            L = oldp.inverted() @ oldb
            DEL[b] = DEL[par] @ L @ R
            POS[b] = POS[par] + DEL[par] @ (rh[b] - rh[par])


def _is_desc(rig, b, anc):
    p = rig.parent[b]
    while p:
        if p == anc: return True
        p = rig.parent[p]
    return False


def bake(rig, solver, katana_objs, swt=None):
    arm = rig.obj
    scene = bpy.context.scene
    scene.render.fps = FPS
    scene.frame_start = 1; scene.frame_end = N
    cache = {}
    frames = []
    for i in range(N):
        t = i / FPS
        POS, DEL, (Pk, Fk), (Ms, Ds, Fs) = solver.solve(t, cache, {k: (v['L'][i], v['R'][i]) for k, v in swt.items()} if swt else None)
        # extra bones: katana / saya
        POS['katana_root'] = Pk; DEL['katana_root'] = Fk
        POS['saya_root'] = Ms; DEL['saya_root'] = Fs
        # saya frame: edge-up basis already in Fs via sword_frame; placed at the mouth
        frames.append((POS, DEL))
    global frames_orig
    frames_orig = [(dict(P), dict(D)) for P, D in frames]
    secondary_motion(rig, frames)
    allb = rig.order + ['katana_root', 'saya_root']
    parent = dict(rig.parent); parent['katana_root'] = None; parent['saya_root'] = None
    rest = dict(rig.rest)
    for nm in ('katana_root', 'saya_root'):
        rest[nm] = arm.data.bones[nm].matrix_local.copy()
    rr = {k: m.to_3x3() for k, m in rest.items()}

    act = bpy.data.actions.new('RealityCut')
    arm.animation_data_create(); arm.animation_data.action = act
    for pb in arm.pose.bones: pb.rotation_mode = 'QUATERNION'
    data = {b: {'q': [], 'l': []} for b in allb}
    for POS, DEL in frames:
        M = {}
        for b in allb:
            R4 = (DEL[b] @ rr[b]).to_4x4(); R4.translation = POS[b]
            M[b] = R4
            p = parent[b]
            base = (M[p] @ rest[p].inverted() @ rest[b]) if p else rest[b]
            basis = base.inverted() @ R4
            q = basis.to_quaternion(); tr = basis.to_translation()
            data[b]['q'].append(q); data[b]['l'].append(tr)
    # fix quaternion sign continuity & write fcurves
    moving = []
    for b in allb:
        qs = data[b]['q']
        for i in range(1, len(qs)):
            if qs[i].dot(qs[i - 1]) < 0: qs[i] = -qs[i]
        ident = all(abs(q.w - qs[0].w) < 1e-5 and abs(q.x - qs[0].x) < 1e-5 and abs(q.y - qs[0].y) < 1e-5 and abs(q.z - qs[0].z) < 1e-5 for q in qs) \
            and all((tr - data[b]['l'][0]).length < 1e-4 for tr in data[b]['l'])
        if ident and abs(qs[0].w - 1) < 1e-5 and data[b]['l'][0].length < 1e-4:
            continue
        moving.append(b)
        for c in range(4):
            fc = act.fcurve_ensure_for_datablock(arm, f'pose.bones["{b}"].rotation_quaternion', index=c, group_name=b)
            fc.keyframe_points.add(N)
            co = []
            for i in range(N): co += [i + 1, qs[i][c]]
            fc.keyframe_points.foreach_set('co', co)
            for kp in fc.keyframe_points: kp.interpolation = 'LINEAR'
        ls = data[b]['l']
        if any(tr.length > 1e-4 for tr in ls):
            for c in range(3):
                fc = act.fcurve_ensure_for_datablock(arm, f'pose.bones["{b}"].location', index=c, group_name=b)
                fc.keyframe_points.add(N)
                co = []
                for i in range(N): co += [i + 1, ls[i][c]]
                fc.keyframe_points.foreach_set('co', co)
                for kp in fc.keyframe_points: kp.interpolation = 'LINEAR'
    return moving, frames


# ----------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------
def load(fbx):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=fbx)
    for nm in ('pCylinder21', 'pCylinder21_2', 'plane'):
        o = bpy.data.objects.get(nm)
        if o: bpy.data.objects.remove(o, do_unlink=True)
    return bpy.data.objects['wushi']


def export_glb(arm, path, tex_dir=None):
    bpy.ops.object.select_all(action='DESELECT')
    kw = dict(filepath=path, export_format='GLB', export_animations=True, export_animation_mode='ACTIVE_ACTIONS',
              export_frame_range=True, export_optimize_animation_size=True, export_def_bones=True,
              export_image_format='JPEG', export_image_quality=85, export_apply=False,
              export_draco_mesh_compression_enable=True, export_draco_mesh_compression_level=6)
    try:
        bpy.ops.export_scene.gltf(**kw)
    except TypeError as e:
        print('export arg issue', e)
        for k in ('export_image_quality', 'export_optimize_animation_size', 'export_frame_range'): kw.pop(k, None)
        bpy.ops.export_scene.gltf(**kw)
    print('GLB size MB', os.path.getsize(path) / 1e6)


def export_fbx(path):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.export_scene.fbx(filepath=path, use_selection=True, object_types={'ARMATURE', 'MESH'}, add_leaf_bones=False,
                             bake_anim=True, bake_anim_use_all_bones=True, bake_anim_use_nla_strips=False,
                             bake_anim_use_all_actions=False, bake_anim_force_startend_keying=True, bake_anim_step=1.0,
                             bake_anim_simplify_factor=0.0, apply_scale_options='FBX_SCALE_NONE', path_mode='COPY',
                             embed_textures=True, mesh_smooth_type='FACE', use_armature_deform_only=False)
    print('FBX size MB', os.path.getsize(path) / 1e6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('fbx'); ap.add_argument('out')
    ap.add_argument('--textures', default=None)
    ap.add_argument('--no-export', action='store_true')
    ap.add_argument('--no-optimize', action='store_true')
    ap.add_argument('--glb', action='store_true')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--worker', default=None)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--tex-scale', type=float, default=1.0)
    ap.add_argument('--no-textures', action='store_true')
    a = ap.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:])
    os.makedirs(a.out, exist_ok=True)
    arm = load(a.fbx)
    rig = Rig(arm)
    objs = build_katana(rig)
    rig = Rig(arm)       # refresh with new bones present (order list gets the extras, harmless)
    rig.order = [b for b in rig.order if b not in ('katana_root', 'saya_root')]
    solver = Solver(rig)
    if a.worker:
        import json
        k, n = [int(x) for x in a.worker.split('/')]
        setup = opt_setup(a.quick)
        res = opt_costs(rig, solver, arm, setup[5][k::n], setup)
        with open(os.path.join(a.out, f'_cost_{k}.json'), 'w') as f: json.dump(res, f)
        return
    swt = None
    if not a.no_optimize:
        swt = optimize_swivel(rig, solver, arm, a.fbx, a.out, quick=a.quick, workers=a.workers)
        arm.animation_data_clear() if arm.animation_data else None
    report_anatomy(solver, swt)
    moving, frames = bake(rig, solver, objs, swt)
    print('moving bones:', len(moving), ' frames:', N)
    if solver.warn:
        seen = {}
        for t, s, over in solver.warn: seen.setdefault(s, []).append((t, over))
        for s, v in seen.items(): print('REACH WARN', s, v[:6], '... n=', len(v))
    if not a.no_textures:
        import texturing
        texturing.texture_scene(arm, os.path.join(a.out, 'textures'), scale=a.tex_scale)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(a.out, 'reality_cut.blend'))
    if not a.no_export:
        export_fbx(os.path.join(a.out, 'Samurai_RealityCut.fbx'))
        if a.glb: export_glb(arm, os.path.join(a.out, 'reality_cut.glb'), a.textures)
    return arm


if __name__ == '__main__':
    main()
