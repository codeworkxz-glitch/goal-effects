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
DUR = 2.9
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
    'chYaw':    [(0, 0), (0.7, 4), (1.15, 8), (1.45, -14), (1.52, -18), (1.66, 22), (1.74, 24), (2.0, 24), (2.3, 6), (2.65, 2), (2.9, 2)],
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
    (1.30, V((-10, 16, 26)), V((0.15, 0.20, -0.97)), V((0, 1, 0))),
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
    loft(bm, [ellipse_ring(-0.6, 4.3, 3.9, 20), ellipse_ring(0.0, 4.3, 3.9, 20), ellipse_ring(0.5, 4.2, 3.8, 20)], 1)   # tsuba
    rings = []
    L = TSUKA
    for i in range(0, 81):
        z = -L * i / 80.0
        bump = 0.06 if (i % 2) else 0.0
        bell = 1.0 + 0.10 * math.sin(math.pi * i / 80.0)
        rings.append(ellipse_ring(z - 0.6, (1.75 + bump) * bell, (1.35 + bump) * bell, 20))
    loft(bm, rings, 3)                                                                              # tsuka (ito wrap)
    loft(bm, [ellipse_ring(-L - 0.6, 1.9, 1.5, 12), ellipse_ring(-L - 1.6, 1.9, 1.5, 12), ellipse_ring(-L - 2.1, 1.0, 0.8, 12)], 2)  # kashira
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
        br = math.sin(t * 2 * math.pi / 3.0)
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
        opt = opt or {}
        sw = opt.get('swiv', (0.0, 0.0)); twv = opt.get('twist', (0.0, 0.0)); lfv = opt.get('lift', (0.0, 0.0))
        shf = opt.get('shf', (SH_FOLLOW, SH_FOLLOW))
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

        def grip_pose(s, G, A, shoulder_pos, fhint):
            """hand delta (3x3) so the fist wraps handle axis A at grip point G; returns (Delta, wrist)"""
            pg = self.pg[s]
            z = Vector((0, 0, 1))
            p_perp = pg - z * pg.dot(z); p_perp.normalize()
            r = (G - shoulder_pos); r = r - A * r.dot(A)
            if r.length < 1e-3: r = fhint - A * fhint.dot(A)
            r.normalize()
            r = rot_axis(A, twv[0 if s == 'L' else 1]) @ r
            # rest frame (z, p_perp, z x p_perp) -> target frame (A, r, A x r)
            B0 = Matrix((z, p_perp, z.cross(p_perp))).transposed()
            B1 = Matrix((A, r, A.cross(r))).transposed()
            Dl = B1 @ B0.inverted()
            return Dl, G - Dl @ pg

        # right side
        sR = Vector((0, 0, 0))
        liftR = Dchest @ Vector((0, 8, 12)) * (lfv[1] * 4 * gR * (1 - gR))
        DshR_tmp = shoulder('R', lerp(R_FREE_WRIST, Gr_sword, gR))
        armR_head = POS['R_shoulder'] + DEL['R_shoulder'] @ (rh['R_arm'] - rh['R_shoulder'])
        DR_g, WR_g = grip_pose('R', Gr_sword, D, armR_head, Vector((0, -1, 0.3)))
        DR_free = DEL['R_shoulder']
        Wr = lerp(R_FREE_WRIST, WR_g, gR) + liftR
        Dhand_R = slerp3(DR_free, DR_g, gR)
        # left side
        Gl_s = Gl_saya; Gl_t = Gl_tsuka
        G_l = lerp(Gl_s, Gl_t, bL) + Dchest @ Vector((0, 8, 12)) * (lfv[0] * 4 * bL * (1 - bL))
        A_l = (lerp(Dsaya, D, bL)).normalized()
        DshL_tmp = shoulder('L', G_l)
        armL_head = POS['L_shoulder'] + DEL['L_shoulder'] @ (rh['L_arm'] - rh['L_shoulder'])
        DL_g, WL_g = grip_pose('L', G_l, A_l, armL_head, Vector((0, -1, 0.3)))

        def solve_arm(s, wrist, Dhand, pole):
            a, b = self.arm_len[s]
            Sh = POS[f'{s}_shoulder'] + DEL[f'{s}_shoulder'] @ (rh[f'{s}_arm'] - rh[f'{s}_shoulder'])
            ax_ = (wrist - Sh)
            if ax_.length > 1e-4:
                pole = rot_axis(ax_.normalized(), sw[0 if s == 'L' else 1]) @ pole
            elbow, wr_pos, clamp = two_bone(Sh, a, b, wrist, pole)
            if clamp and (wrist - Sh).length - (a + b) > 2.0:
                self.warn.append((round(t, 3), s, round((wrist - Sh).length - (a + b), 1)))
            sh = f'{s}_shoulder'
            d0 = (rh[f'{s}_forarm'] - rh[f'{s}_arm']).normalized()
            d1 = (elbow - Sh).normalized()
            Da = rot_min(DEL[sh] @ d0, d1) @ DEL[sh]
            place(f'{s}_arm', Dabs=Da)
            d0b = (rh[f'{s}_hand'] - rh[f'{s}_forarm']).normalized()
            d1b = (wr_pos - elbow).normalized()
            Df = rot_min(Da @ d0b, d1b) @ Da
            # twist forearm half-way towards hand orientation (swing-twist)
            q = (Dhand @ Df.inverted()).to_quaternion()
            axis = d1b
            ang = 2 * math.atan2(Vector((q.x, q.y, q.z)).dot(axis), q.w)
            if ang > math.pi: ang -= 2 * math.pi
            if ang < -math.pi: ang += 2 * math.pi
            Df = Matrix.Rotation(ang * 0.6, 3, axis) @ Df
            place(f'{s}_forarm', Dabs=Df)
            place(f'{s}_hand', Dabs=Dhand, pos=wr_pos)

        # elbow poles: down and out
        polR = Dchest @ Vector((-0.5, -1.0, -0.7))
        polL = Dchest @ Vector((0.5, -1.0, -0.7))
        solve_arm('R', Wr, Dhand_R, polR)
        solve_arm('L', WL_g, DL_g, polL)

        # fingers
        def fingers(s, curl, Dhand, grip_w):
            ax = Vector((0, 0, -1)) if s == 'L' else Vector((0, 0, 1))
            for fn, (b0, b1, b2) in {'m': (62, 78, 50), 'f': (62, 78, 50), 'r': (66, 80, 52), 'l': (70, 82, 55)}.items():
                c = curl
                D0 = Dhand @ rot_axis(ax, b0 * c)
                place(f'{s}_{fn}', Dabs=D0)
                D1 = D0 @ rot_axis(ax, b1 * c); place(f'{s}_{fn}1', Dabs=D1)
                D2 = D1 @ rot_axis(ax, b2 * c); place(f'{s}_{fn}2', Dabs=D2)
                place(f'{s}_{fn}3', Dabs=D2)
            ta = Vector((0, 0.5, -1)).normalized() if s == 'L' else Vector((0, 0.5, 1)).normalized()
            T0 = Dhand @ rot_axis(ta, 25 * curl)
            place(f'{s}_t', Dabs=T0)
            T1 = T0 @ rot_axis(ax, 30 * curl); place(f'{s}_t1', Dabs=T1)
            T2 = T1 @ rot_axis(ax, 25 * curl); place(f'{s}_t2', Dabs=T2)

        fingers('R', pchip(CURL_R, t), Dhand_R, gR)
        fingers('L', pchip(CURL_L, t), DL_g, 1.0)

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


def optimize_swivel(rig, solver, arm, step=3, quick=False):
    """per-frame search over (elbow swivel, wrist twist, path lift) against the real deformed meshes,
    then Viterbi smoothing across time."""
    import clips, itertools
    scorer = clips.Scorer()
    tabs = rig_tables(rig, arm)
    SW = (-90, -60, -30, 0, 30, 60, 90)
    TW = (-50, -25, 0, 25, 50)
    LF = (0.0, 1.0)
    if quick:
        SW = (-80, -40, 0, 40, 80); TW = (-45, 0, 45); LF = (0.0, 1.0); step = 4
    SF = (0.1, 0.4, 0.7)
    if quick: SF = (0.15, 0.55)
    cands = list(itertools.product(range(len(SW)), range(len(TW)), range(len(LF)), range(len(SF))))
    idx = list(range(0, N, step))
    if idx[-1] != N - 1: idx.append(N - 1)
    cache = {}
    cost = {'L': [], 'R': []}
    for i in idx:
        t = i / FPS
        rowL, rowR = [], []
        for (a_, b_, c_, d_) in cands:
            o = {'swiv': (SW[a_],) * 2, 'twist': (TW[b_],) * 2, 'lift': (LF[c_],) * 2, 'shf': (SF[d_],) * 2}
            POS, DEL, (Pk, Fk), (Ms, Ds, Fs) = solver.solve(t, cache, o)
            set_pose(arm, tabs, POS, DEL, Pk, Fk, Ms, Fs)
            T = scorer.trees(bpy.context.evaluated_depsgraph_get())
            pen = abs(SW[a_]) * 0.03 + abs(TW[b_]) * 0.03 + LF[c_] * 1.0 + abs(SF[d_] - 0.22) * 2
            rowL.append(scorer.side_cost(T, 'L') + pen)
            rowR.append(scorer.side_cost(T, 'R') + pen)
        cost['L'].append(rowL); cost['R'].append(rowR)
        print('  opt frame', i, 'minL', min(rowL), 'minR', min(rowR), flush=True)
    out = {'swiv': {}, 'twist': {}, 'lift': {}, 'shf': {}}
    lam = (5.0, 3.0, 4.0)
    K = len(cands)
    trans = [[lam[0] * abs(cands[j][0] - cands[jp][0]) + lam[1] * abs(cands[j][1] - cands[jp][1]) + lam[2] * 2 * abs(cands[j][2] - cands[jp][2]) + 6.0 * abs(cands[j][3] - cands[jp][3])
              for jp in range(K)] for j in range(K)]
    for s in 'LR':
        C = cost[s]; n = len(C)
        best = C[0][:]; back = []
        for f in range(1, n):
            nb = []; bk = []
            for j in range(K):
                tj = trans[j]
                m = min(range(K), key=lambda q: best[q] + tj[q])
                nb.append(C[f][j] + best[m] + tj[m]); bk.append(m)
            best = nb; back.append(bk)
        j = min(range(K), key=lambda q: best[q]); path = [j]
        for bk in reversed(back):
            j = bk[j]; path.append(j)
        path.reverse()
        print('opt', s, 'residual', sum(C[f][path[f]] for f in range(n)))
        for d, vals in enumerate((SW, TW, LF, SF)):
            ang = [vals[cands[j][d]] for j in path]
            full = []
            for i in range(N):
                f = min(max(bisect.bisect_right(idx, i) - 1, 0), n - 2)
                u = (i - idx[f]) / float(idx[f + 1] - idx[f])
                full.append(lerp(ang[f], ang[f + 1], u))
            sm = []
            for i in range(N):
                w = [full[min(max(i + dd, 0), N - 1)] for dd in range(-3, 4)]
                sm.append(sum(w) / len(w))
            out[('swiv', 'twist', 'lift', 'shf')[d]][s] = sm
    return out


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
                             bake_anim_simplify_factor=0.0, apply_scale_options='FBX_SCALE_NONE', path_mode='AUTO',
                             embed_textures=False, mesh_smooth_type='FACE', use_armature_deform_only=False)
    print('FBX size MB', os.path.getsize(path) / 1e6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('fbx'); ap.add_argument('out')
    ap.add_argument('--textures', default=None)
    ap.add_argument('--no-export', action='store_true')
    ap.add_argument('--no-optimize', action='store_true')
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:])
    os.makedirs(a.out, exist_ok=True)
    arm = load(a.fbx)
    rig = Rig(arm)
    objs = build_katana(rig)
    rig = Rig(arm)       # refresh with new bones present (order list gets the extras, harmless)
    rig.order = [b for b in rig.order if b not in ('katana_root', 'saya_root')]
    solver = Solver(rig)
    swt = None
    if not a.no_optimize:
        swt = optimize_swivel(rig, solver, arm, quick=a.quick)
        arm.animation_data_clear() if arm.animation_data else None
    moving, frames = bake(rig, solver, objs, swt)
    print('moving bones:', len(moving), ' frames:', N)
    if solver.warn:
        seen = {}
        for t, s, over in solver.warn: seen.setdefault(s, []).append((t, over))
        for s, v in seen.items(): print('REACH WARN', s, v[:6], '... n=', len(v))
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(a.out, 'reality_cut.blend'))
    if not a.no_export:
        export_fbx(os.path.join(a.out, 'Samurai_RealityCut.fbx'))
        export_glb(arm, os.path.join(a.out, 'reality_cut.glb'), a.textures)
    return arm


if __name__ == '__main__':
    main()
