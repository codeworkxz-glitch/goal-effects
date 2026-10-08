"""Blood Reaper – choreography + solver (pure numpy).

Everything is driven by the scythe: a keyed trajectory for the left-hand grip point and the shaft/blade
orientation (quaternion Catmull-Rom = smooth momentum through the keys).  Both hands are solved onto the shaft
(left: the original bind grip, right: the mirrored "bat" grip 26 units toward the pommel), arms with an
anatomical two-bone IK (elbow hinge plane from the bind pose, wrist twist shared with the forearm), the torso
twists/leans into the swing, the head looks down / up, the wings fold and unfurl, the robe panels and belt
trinkets are spring chains.

Model units (recentred, origin on the ground under the pelvis, Z up, Reaper faces -Y; its left is +X).
"""
import os, numpy as np
import reaper_rig as RR
from reaper_rig import rot, nrm, frame_from, mat2quat, quat2mat, slerp

FPS = 60
DURATION = 4.0
NFRAMES = int(round(DURATION * FPS)) + 1
REST_Z = None          # filled from the rig (pelvis height at rest)

T_LG = 10.9            # left grip: axial position of the left palm on the shaft (bind)
T_RG = T_LG - 26.0     # right grip, toward the pommel


# ------------------------------------------------------------------------------------------------
# interpolation helpers
# ------------------------------------------------------------------------------------------------
def smoothstep(e0, e1, x):
    x = np.clip((x - e0) / (e1 - e0), 0, 1); return x * x * (3 - 2 * x)


def smootherstep(e0, e1, x):
    x = np.clip((x - e0) / (e1 - e0), 0, 1); return x * x * x * (x * (x * 6 - 15) + 10)


def hermite_keys(keys, t):
    """keys: [(time, value(array))] Catmull-Rom tangents (non-uniform), clamped ends (zero slope)"""
    ts = np.array([k[0] for k in keys]); vs = [np.asarray(k[1], float) for k in keys]
    if t <= ts[0]: return vs[0]
    if t >= ts[-1]: return vs[-1]
    i = np.searchsorted(ts, t) - 1
    def tan(j):
        if j == 0 or j == len(ts) - 1: return vs[j] * 0
        return (vs[j + 1] - vs[j - 1]) / (ts[j + 1] - ts[j - 1])
    h = ts[i + 1] - ts[i]; x = (t - ts[i]) / h
    h00 = 2 * x ** 3 - 3 * x ** 2 + 1; h10 = x ** 3 - 2 * x ** 2 + x; h01 = -2 * x ** 3 + 3 * x ** 2; h11 = x ** 3 - x ** 2
    return h00 * vs[i] + h10 * h * tan(i) + h01 * vs[i + 1] + h11 * h * tan(i + 1)


def quat_keys(keys, t):
    """keys: [(time, quat)] – Catmull-Rom on the unit sphere (Barry-Goldman pyramid of slerps)"""
    ts = np.array([k[0] for k in keys]); qs = [np.asarray(k[1], float) for k in keys]
    for j in range(1, len(qs)):
        if qs[j] @ qs[j - 1] < 0: qs[j] = -qs[j]
    if t <= ts[0]: return qs[0]
    if t >= ts[-1]: return qs[-1]
    i = np.searchsorted(ts, t) - 1
    i0, i3 = max(i - 1, 0), min(i + 2, len(ts) - 1)
    t0, t1, t2, t3 = ts[i0], ts[i], ts[i + 1], ts[i3]
    if i0 == i: t0 = t1 - (t2 - t1)
    if i3 == i + 1: t3 = t2 + (t2 - t1)
    q0, q1, q2, q3 = qs[i0], qs[i], qs[i + 1], qs[i3]
    def L(qa, qb, ta, tb): return slerp(qa, qb, (t - ta) / (tb - ta))
    a1 = L(q0, q1, t0, t1); a2 = L(q1, q2, t1, t2); a3 = L(q2, q3, t2, t3)
    b1 = L(a1, a2, t0, t2); b2 = L(a2, a3, t1, t3)
    return L(b1, b2, t1, t2)


def scalar_keys(keys, t):
    return float(hermite_keys([(a, np.array([b])) for a, b in keys], t)[0])


def mono_keys(keys, t):
    return float(RR_pchip([k[0] for k in keys], [k[1] for k in keys])(t))


def RR_pchip(xk, yk):
    xk = np.asarray(xk, float); yk = np.asarray(yk, float)
    h = np.diff(xk); d = np.diff(yk) / h
    m = np.zeros_like(yk)
    for i in range(1, len(xk) - 1):
        if d[i - 1] * d[i] > 0:
            w1 = 2 * h[i] + h[i - 1]; w2 = h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
    def f(x):
        x = np.clip(x, xk[0], xk[-1]); i = np.clip(np.searchsorted(xk, x) - 1, 0, len(h) - 1)
        tt = (x - xk[i]) / h[i]
        return ((2 * tt ** 3 - 3 * tt ** 2 + 1) * yk[i] + (tt ** 3 - 2 * tt ** 2 + tt) * h[i] * m[i]
                + (-2 * tt ** 3 + 3 * tt ** 2) * yk[i + 1] + (tt ** 3 - tt ** 2) * h[i] * m[i + 1])
    return f


def scythe_rot(d, e):
    d = nrm(d); e = nrm(e - d * (e @ d)); n = np.cross(d, e)
    return np.stack([d, e, n], 1)


# ------------------------------------------------------------------------------------------------
# choreography keys
# ------------------------------------------------------------------------------------------------
T_RISE_END = 1.40
T_LOOK = 1.85
T_GRAB = 2.02
T_WIND = 2.25
T_SWEEP_MID = 2.40
T_FOLLOW = 2.56
T_RAISE = 2.92
T_IMPACT = 3.12
T_SINK0 = 3.36

DEG = np.pi / 180

# scythe keys: (time, left-grip point (root frame), shaft dir d, blade dir e)
SCYTHE_KEYS = [
    (0.00, (24.0, -10.0, 62.0), (0.12, -0.12, 0.98), (0.6, -0.8, 0.0)),
    (1.40, (24.0, -10.0, 62.0), (0.12, -0.12, 0.98), (0.6, -0.8, 0.0)),
    (1.85, (23.0, -13.0, 66.0), (0.18, -0.20, 0.96), (0.5, -0.86, 0.0)),
    # diagonal reaping arc (fitted for reach / torso clearance): wound up high over the left shoulder, low through
    # the front, out to the right (the left hand holds the blade end, so this direction never crosses the arms)
    (T_GRAB, (3.3, -25.2, 79.4), (0.352, -0.610, 0.710), (-0.866, -0.500, 0.000)),
    (2.14, (8.1, -22.6, 77.1), (0.852, -0.310, 0.423), (-0.342, -0.940, 0.000)),
    (T_WIND, (14.9, -18.6, 76.2), (0.941, 0.252, 0.226), (0.259, -0.966, 0.000)),        # wound up
    (2.33, (14.0, -25.1, 73.2), (0.903, -0.421, -0.087), (-0.423, -0.906, 0.000)),
    (T_SWEEP_MID, (3.3, -27.4, 69.8), (0.057, -0.653, -0.755), (-0.996, -0.087, 0.000)),  # low through the front
    (2.48, (-1.0, -27.5, 72.0), (-0.278, -0.397, -0.875), (-0.819, 0.574, 0.000)),
    (T_FOLLOW, (1.1, -26.0, 81.2), (-0.857, -0.495, 0.147), (-0.500, 0.866, 0.000)),     # follow-through
    (2.70, (11.7, -19.9, 84.7), (-0.541, -0.644, 0.541), (-0.766, 0.643, 0.000)),
    (T_RAISE, (1.0, -20.0, 104.0), (0.10, 0.05, 0.99), (0.0, -1.0, 0.05)),
    (3.02, (-2.0, -26.0, 92.0), (0.35, -0.65, 0.67), (0.0, -0.7, -0.7)),
    (T_IMPACT, (-5.0, -36.0, 74.0), (0.50, -0.85, 0.15), (0.0, -0.17, -0.99)),
    (3.24, (-5.0, -35.0, 75.5), (0.50, -0.85, 0.17), (0.0, -0.19, -0.98)),
    (T_SINK0, (-5.0, -36.0, 74.5), (0.50, -0.85, 0.15), (0.0, -0.17, -0.99)),
    (DURATION, (-5.0, -36.0, 74.5), (0.50, -0.85, 0.15), (0.0, -0.17, -0.99)),
]

# right-hand spacing down the shaft (units from the left grip)
SEP_KEYS = [(0, 28.0), (T_GRAB, 30.0), (DURATION, 30.0)]


def sep(t):
    return scalar_keys(SEP_KEYS, t)


# body channels (degrees / units)
TWIST_KEYS = [(0, 0), (1.85, 0), (T_GRAB, 10), (T_WIND, 46), (T_SWEEP_MID, 0), (T_FOLLOW, -40), (2.70, -16),
              (T_RAISE, 0), (T_IMPACT, -4), (T_SINK0, -2), (DURATION, 0)]
LEAN_KEYS = [(0, 14), (1.2, 12), (1.6, 4), (T_LOOK, -4), (T_GRAB, 0), (T_WIND, 4), (T_SWEEP_MID, 10),
             (T_FOLLOW, 8), (2.70, -2), (T_RAISE, -10), (T_IMPACT, 22), (3.24, 18), (T_SINK0, 20), (DURATION, 26)]
HEAD_PITCH_KEYS = [(0, 32), (1.35, 30), (1.55, 20), (T_LOOK, -6), (T_GRAB, -4), (T_WIND, 0), (T_FOLLOW, 0),
                   (T_RAISE, -10), (T_IMPACT, 6), (T_SINK0, 4), (DURATION, 26)]
HEAD_TURN_KEYS = [(0, 0), (T_GRAB, 0), (T_WIND, -22), (T_SWEEP_MID, 0), (T_FOLLOW, 22), (2.70, 6), (T_RAISE, 0),
                  (DURATION, 0)]
WING_KEYS = [(0, 0.0), (1.30, 0.0), (1.55, 0.55), (T_LOOK, 1.0), (T_WIND, 1.05), (T_FOLLOW, 0.85),
             (T_RAISE, 1.15), (T_IMPACT, 0.95), (T_SINK0, 0.8), (3.75, 0.1), (DURATION, 0.0)]
GRIP_R_KEYS = [(0, 0.0), (1.90, 0.0), (T_GRAB, 1.0), (DURATION, 1.0)]


def root_z(t):
    """pelvis height: summoned rise (eases out), sinks back (eases in)"""
    under = -105.0
    if t <= T_RISE_END:
        f = smootherstep(0.0, T_RISE_END, t) * 0.75 + 0.25 * (1 - (1 - np.clip(t / T_RISE_END, 0, 1)) ** 3)
        return under + (REST_Z - under) * f
    if t < T_SINK0:
        bob = 0.8 * np.sin(2 * np.pi * (t - T_RISE_END) / 1.6) * (1 - smoothstep(T_WIND - 0.2, T_WIND, t))
        dip = -2.5 * np.exp(-((t - T_IMPACT - 0.04) / 0.08) ** 2)
        return REST_Z + bob + dip
    f = smoothstep(T_SINK0, DURATION, t) ** 1.25
    return REST_Z + (under - REST_Z) * f


_OPT = None


def active_keys():
    """optimised keys (optimize_keys.py) when available, else the hand-authored intent"""
    global _OPT
    if _OPT is None:
        p = os.path.join(RR.WORK, 'scythe_keys.json')
        if os.path.exists(p) and not os.environ.get('REAPER_RAW_KEYS'):
            import json
            _OPT = [(k['t'], tuple(k['p']), tuple(k['d']), tuple(k['e'])) for k in json.load(open(p))]
        else:
            _OPT = SCYTHE_KEYS
    return _OPT


def twist_adj(t):
    KEYS = active_keys()
    if not hasattr(twist_adj, 'keys'):
        p = os.path.join(RR.WORK, 'scythe_keys.json')
        twist_adj.keys = None
        if os.path.exists(p) and not os.environ.get('REAPER_RAW_KEYS'):
            import json
            js = json.load(open(p))
            if all('twist' in k for k in js):
                twist_adj.keys = [(k['t'], k['twist']) for k in js]
    return scalar_keys(twist_adj.keys, t) if twist_adj.keys else 0.0


def scythe_frame(t, rig):
    """4x4 world frame of the scythe (columns d, e, n; origin = left grip point on the axis)"""
    KEYS = active_keys()
    pts = [(k[0], np.array(k[1], float)) for k in KEYS]
    qs = [(k[0], mat2quat(scythe_rot(np.array(k[2]), np.array(k[3])))) for k in KEYS]
    p = hermite_keys(pts, t)
    R = quat2mat(quat_keys(qs, t))
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = p
    M[2, 3] += root_z(t) - REST_Z            # keys are relative to the resting root height
    return M


# ------------------------------------------------------------------------------------------------
# solver
# ------------------------------------------------------------------------------------------------
class Solver:
    def __init__(self):
        global REST_Z
        self.r = rig = RR.Rig()
        E = rig.E; i = rig.idx
        REST_Z = E[i['Pelvis']][2, 3]
        S = np.load(os.path.join(RR.WORK, 'scythe_frame.npz'))
        c = S['c'] - rig.off; d = S['d']; e = S['e']
        R0 = scythe_rot(d, e)
        self.S0 = np.eye(4); self.S0[:3, :3] = R0; self.S0[:3, 3] = c + d * T_LG   # bind frame at the left grip
        self.S0_axis = (c, d)
        # left hand in scythe frame (constant grip)
        self.GL = np.linalg.inv(self.S0) @ E[i['Hand_L']]
        # right hand: mirror the left grip across the plane containing the shaft and the palm offset normal
        # (palms on opposite sides of the shaft, thumbs the same way = two-handed "bat" grip), slid to T_RG
        Lp = self.GL[:3, 3]
        o = Lp - np.array([Lp[0], 0, 0]); m = nrm(o)                 # offset from the axis, in scythe coords
        Mref = np.eye(4); Mref[:3, :3] = np.eye(3) - 2 * np.outer(m, m)
        Sfl = np.diag([1.0, -1.0, 1.0, 1.0])
        GR = Mref @ self.GL @ Sfl
        GR[0, 3] += T_RG - T_LG
        self.GR_template = GR.copy()
        self.GR = GR
        gp = os.path.join(RR.WORK, 'grip_r.json')
        if os.path.exists(gp) and not os.environ.get('REAPER_RAW_KEYS'):
            import json
            self.GR = np.array(json.load(open(gp))['GR'])
        self.finger_L = [n for n in rig.names if n.startswith('Finger') and n.endswith('_L')]
        self.finger_R = [n for n in rig.names if n.startswith('Finger') and n.endswith('_R')]
        # right-hand grip finger shape: mirrored left local rotations
        self.fR_grip = {}
        for n in self.finger_R:
            nl = n[:-1] + 'L'
            Ll = rig.L0[i[nl]][:3, :3]
            self.fR_grip[n] = Sfl[:3, :3] @ Ll @ Sfl[:3, :3]
        # arm bind data
        self.arm = {}
        for s in 'LR':
            U, F, H = E[i['UpperArm_' + s]], E[i['Forearm_' + s]], E[i['Hand_' + s]]
            a = np.linalg.norm(F[:3, 3] - U[:3, 3]); b = np.linalg.norm(H[:3, 3] - F[:3, 3])
            n_b = nrm(np.cross(F[:3, 3] - U[:3, 3], H[:3, 3] - F[:3, 3]))     # hinge normal at bind
            self.arm[s] = dict(a=a, b=b, hU=U[:3, :3].T @ n_b, hF=F[:3, :3].T @ n_b,
                               xU=U[:3, :3].T @ nrm(F[:3, 3] - U[:3, 3]), xF=F[:3, :3].T @ nrm(H[:3, 3] - F[:3, 3]))
        self.diag = []
        self.prev_phi = {}
        self.prev_roll = {}

    # ---- grip roll keys --------------------------------------------------------------------------
    def roll_keys(self):
        if hasattr(self, '_roll_keys'): return self._roll_keys
        import json
        p = os.path.join(RR.WORK, 'scythe_keys.json')
        js = json.load(open(p)) if (os.path.exists(p) and not os.environ.get('REAPER_RAW_KEYS')) else None
        if js is not None and all('rollL' in k for k in js):
            self._roll_keys = {s: [(k['t'], k['roll' + s]) for k in js if k.get('roll' + s) is not None] for s in 'LR'}
            return self._roll_keys
        keys = {'L': [], 'R': []}
        for k in active_keys():
            t = k[0]
            W, ex, dg = self.pose(t, roll_free=True)
            for s_ in 'LR':
                if s_ in dg and (s_ == 'L' or scalar_keys(GRIP_R_KEYS, t) >= 1.0):
                    keys[s_].append((t, float(np.radians(dg[s_]['roll']))))
        for s_ in 'LR':           # unwrap so interpolation takes the short way round
            if keys[s_]:
                un = np.unwrap([r for _, r in keys[s_]])
                keys[s_] = [(t, float(r)) for (t, _), r in zip(keys[s_], un)]
        if js is not None:
            for k in js:
                for s_ in 'LR':
                    m = [r for t, r in keys[s_] if abs(t - k['t']) < 1e-6]
                    k['roll' + s_] = m[0] if m else None
            json.dump(js, open(p, 'w'), indent=1)
        self._roll_keys = keys
        return keys

    # ---- two-bone arm IK with swivel search --------------------------------------------------------
    def arm_candidates(self, s, sh, Htarget, step=6):
        """all elbow-swivel candidates for one arm at once (vectorised)"""
        from scipy.spatial.transform import Rotation as Rot
        rig = self.r; i = rig.idx; A = self.arm[s]
        wr = Htarget[:3, 3]
        v = wr - sh; L = np.linalg.norm(v)
        Lc = np.clip(L, abs(A['a'] - A['b']) + 0.5, A['a'] + A['b'] - 0.15)
        clamp = max(0.0, L - Lc)
        ax = v / L
        cosA = (A['a'] ** 2 + Lc ** 2 - A['b'] ** 2) / (2 * A['a'] * Lc)
        sinA = np.sqrt(max(0, 1 - cosA ** 2))
        u = nrm(np.cross(ax, [0, 0, 1.0]) if abs(ax[2]) < 0.95 else np.cross(ax, [1.0, 0, 0])); w = np.cross(ax, u)
        wr_c = sh + ax * Lc
        RH = Htarget[:3, :3]
        rest_local = rig.E[i['Forearm_' + s]][:3, :3].T @ rig.E[i['Hand_' + s]][:3, :3]
        xF = nrm(A['xF'])
        phi = np.radians(np.arange(-180, 180, step))
        pd = np.cos(phi)[:, None] * u + np.sin(phi)[:, None] * w
        el = sh + A['a'] * (cosA * ax + sinA * pd)
        def bframe(x, h):
            x = x / np.linalg.norm(x, axis=-1, keepdims=True)
            z = h - x * (h * x).sum(-1, keepdims=True); z /= np.linalg.norm(z, axis=-1, keepdims=True)
            y = np.cross(z, x)
            return np.stack([x, y, z], -1)
        hinge = np.cross(el - sh, wr_c - el); hinge /= np.linalg.norm(hinge, axis=1, keepdims=True)
        RU = bframe(el - sh, hinge) @ frame_from(A['xU'], A['hU']).T
        RF = bframe(wr_c - el, hinge) @ frame_from(A['xF'], A['hF']).T
        a_loc = rest_local.T @ xF
        dl = rest_local.T[None] @ np.transpose(RF, (0, 2, 1)) @ RH[None]
        q = Rot.from_matrix(dl).as_quat()          # x y z w
        tw = 2 * np.arctan2(q[:, :3] @ a_loc, q[:, 3]); tw = np.angle(np.exp(1j * tw))
        share = 0.6 * tw
        RF2 = RF @ Rot.from_rotvec(np.outer(share, xF)).as_matrix()
        dl2 = rest_local.T[None] @ np.transpose(RF2, (0, 2, 1)) @ RH[None]
        q2 = Rot.from_matrix(dl2).as_quat()
        ang = np.degrees(2 * np.arccos(np.clip(np.abs(q2[:, 3]), 0, 1)))
        tw2 = np.degrees(np.angle(np.exp(1j * 2 * np.arctan2(q2[:, :3] @ a_loc, q2[:, 3]))))
        swing = np.maximum(0.0, ang - np.abs(tw2))
        ed = el - (sh + np.outer((el - sh) @ ax, ax)); ed /= np.linalg.norm(ed, axis=1, keepdims=True)
        elbow = np.degrees(np.arccos(np.clip(((el - sh) / A['a'] * (wr_c - el) / np.linalg.norm(wr_c - el, axis=1, keepdims=True)).sum(1), -1, 1)))
        return dict(phi=phi, RU=RU, RF=RF2, el=el, sh=sh, wr=wr_c, RH=RH, ed=ed, clamp=clamp, wrist=ang, swing=swing,
                    twist=np.abs(tw2), fore=np.abs(np.degrees(share)), elbow=elbow)

    def arm_cost(self, c, s, ribR, prev_phi=None):
        side = 1 if s == 'L' else -1
        pref = nrm(ribR @ np.array([side * 0.8, 0.35, -0.55]))      # elbows out, back, down
        cost = (c['swing'] / 55.0) ** 2 + (c['twist'] / 45.0) ** 2 + (c['fore'] / 85.0) ** 2
        cost = cost + 0.8 * (1 - c['ed'] @ pref)
        cost = cost + 40 * c['clamp'] ** 2
        if prev_phi is not None:
            dphi = np.angle(np.exp(1j * (c['phi'] - prev_phi)))
            cost = cost + (np.degrees(dphi) / 25.0) ** 2
        return cost

    def solve_grip(self, s, W, Sm, G, prev_phi=None, prev_roll=None, step=6, rstep=10, rcentre=None, cont_roll=None):
        """hand rolls around the round shaft (grip angle) + elbow swivel, jointly"""
        best = None
        if rcentre is None:
            centre, span, st = 0.0, 180, rstep            # free search (key optimisation)
        else:
            centre, span, st = np.degrees(rcentre), 25, 2.5  # baking: refine around the keyed grip roll
        for roll in np.radians(np.arange(centre - span, centre + span + 0.1, st)):
            Rr = np.eye(4); Rr[:3, :3] = rot([1.0, 0, 0], roll)
            b, c = self.solve_arm(s, W, Sm @ Rr @ G, prev_phi, step)
            c += 0.25 * (np.degrees(np.angle(np.exp(1j * roll))) / 90.0) ** 2
            if rcentre is not None:
                c += (np.degrees(roll - rcentre) / 20.0) ** 2
            if cont_roll is not None:          # key optimisation: stay close to the previous key's grip
                c += (np.degrees(np.angle(np.exp(1j * (roll - cont_roll)))) / 35.0) ** 2
            if prev_roll is not None and rcentre is not None:
                c += (np.degrees(roll - prev_roll) / 4.0) ** 2
            if best is None or c < best[1]:
                b['roll'] = roll; best = (b, c)
        return best

    def solve_arm(self, s, W, Htarget, prev_phi=None, step=6):
        rig = self.r; i = rig.idx
        sh = W[i['UpperArm_' + s]][:3, 3]
        ribR = W[i['Ribcage']][:3, :3] @ rig.E[i['Ribcage']][:3, :3].T
        c = self.arm_candidates(s, sh, Htarget, step)
        cost = self.arm_cost(c, s, ribR, prev_phi)
        k = int(np.argmin(cost))
        U = np.eye(4); U[:3, :3] = c['RU'][k]; U[:3, 3] = c['sh']
        F = np.eye(4); F[:3, :3] = c['RF'][k]; F[:3, 3] = c['el'][k]
        H = np.eye(4); H[:3, :3] = c['RH']; H[:3, 3] = c['wr']
        best = dict(phi=c['phi'][k], U=U, F=F, H=H, clamp=c['clamp'], wrist=c['wrist'][k], swing=c['swing'][k],
                    twist=c['twist'][k], fore=c['fore'][k], elbow=c['elbow'][k])
        nocost = self.arm_cost(c, s, ribR, None)[k]
        return best, nocost

    # ---- full pose -------------------------------------------------------------------------------
    def pose(self, t, S_override=None, use_prev=False, twist_extra=None, roll_free=False):
        rig = self.r; i = rig.idx; E = rig.E
        local = {}
        # torso: twist about world Z, lean about world X, distributed
        tw = (scalar_keys(TWIST_KEYS, t) + (twist_adj(t) if twist_extra is None else twist_extra)) * DEG
        ln = scalar_keys(LEAN_KEYS, t) * DEG
        dist = {'Pelvis': 0.18, 'Spine1': 0.2, 'Spine2': 0.22, 'Spine3': 0.2, 'Ribcage': 0.2}
        for n, f in dist.items():
            Rw = rot([0, 0, 1], tw * f) @ rot([1, 0, 0], ln * (f if n != 'Pelvis' else 0.25))
            local[n] = rig.world_rot_local(n, Rw)
        hp = scalar_keys(HEAD_PITCH_KEYS, t) * DEG
        ht = scalar_keys(HEAD_TURN_KEYS, t) * DEG
        # breathing / menace sway while staring
        sway = np.sin(2 * np.pi * 0.7 * t) * (smoothstep(T_RISE_END - 0.2, T_LOOK, t) - smoothstep(T_GRAB - 0.1, T_GRAB, t))
        local['Neck'] = rig.world_rot_local('Neck', rot([1, 0, 0], hp * 0.45) @ rot([0, 0, 1], ht * 0.4))
        local['Head'] = rig.world_rot_local('Head', rot([1, 0, 0], hp * 0.55) @ rot([0, 0, 1], ht * 0.6 + 2.5 * DEG * sway))
        # wings
        wo = scalar_keys(WING_KEYS, t)
        local.update(self.wing_pose(wo, t))
        # root
        root = E[i['Pelvis']].copy(); root[2, 3] = root_z(t)
        W = rig.fk(local, root=root)
        # scythe + hands
        S = scythe_frame(t, rig) if S_override is None else S_override
        HL = S @ self.GL
        GRt = self.GR.copy(); GRt[0, 3] = T_LG - sep(t)
        HR = S @ GRt
        gR = scalar_keys(GRIP_R_KEYS, t)
        out = {}
        diag = {}
        for s_, Ht in (('L', HL), ('R', HR)):
            if s_ == 'R' and gR <= 0.0:
                continue
            ribR = W[i['Ribcage']][:3, :3] @ E[i['Ribcage']][:3, :3].T
            sh0 = W[i['UpperArm_' + s_]][:3, 3]
            lift = np.clip((Ht[2, 3] - sh0[2]) / 30.0, -0.3, 1.0)
            cl = 'Clavicle_' + s_
            local[cl] = rig.world_rot_local(cl, rot(ribR @ np.array([0, 1.0, 0]), (1 if s_ == 'L' else -1) * 14 * DEG * lift))
            W = rig.fk(local, root=root, overrides=out)
            prev = self.prev_phi.get(s_) if use_prev else None
            G = self.GL if s_ == 'L' else self.GR
            if s_ == 'R':
                G = G.copy(); G[0, 3] = T_LG - sep(t)
            if roll_free or S_override is not None:
                if s_ == 'R': G = G.copy(); G[0, 3] = T_LG - sep(t)
                best, cost = self.solve_grip(s_, W, S, G, prev, None)
            else:
                rk = self.roll_keys()[s_]
                rc = scalar_keys(rk, t) if rk else 0.0
                prev_r = self.prev_roll.get(s_) if use_prev else None
                if s_ == 'R' and gR < 1.0:
                    # still reaching for the shaft: aim at the grip it will take
                    Rr = np.eye(4); Rr[:3, :3] = rot([1.0, 0, 0], rc)
                    Hg = S @ Rr @ G
                    Hfk = W[i['Hand_R']]
                    g = smootherstep(0, 1, gR)
                    q = slerp(mat2quat(Hfk[:3, :3]), mat2quat(Hg[:3, :3]), g)
                    Ht = np.eye(4); Ht[:3, :3] = quat2mat(q); Ht[:3, 3] = Hfk[:3, 3] * (1 - g) + Hg[:3, 3] * g
                    best, cost = self.solve_arm(s_, W, Ht, prev); best['roll'] = rc
                else:
                    best, cost = self.solve_grip(s_, W, S, G, prev, rc if prev_r is None else prev_r, rcentre=rc)
            if use_prev:
                self.prev_phi[s_] = best['phi']; self.prev_roll[s_] = best['roll']
            out['UpperArm_' + s_] = best['U']; out['Forearm_' + s_] = best['F']; out['Hand_' + s_] = best['H']
            diag[s_] = dict(clamp=best['clamp'], elbow=best['elbow'], wrist=best['wrist'], swing=best['swing'],
                            twist=best['twist'], fore=best['fore'], cost=cost, phi=np.degrees(best['phi']), roll=np.degrees(best['roll']))
        # fingers: left keeps its bind grip, right closes onto the shaft
        for n in self.finger_R:
            g = smoothstep(0.35, 1.0, gR)
            qa = mat2quat(np.eye(3)); qb = mat2quat(rig.L0[i[n]][:3, :3].T @ self.fR_grip[n])
            local[n] = quat2mat(slerp(qa, qb, g))
        W = rig.fk(local, root=root, overrides=out)
        # scythe bone frame (E-frame convention: columns d, e, n at the left grip point)
        extra = {'Scythe': S}
        return W, extra, diag

    # ---- wings -----------------------------------------------------------------------------------
    def wing_pose(self, wo, t):
        """wo: 0 = folded back and closed, 1 = spread wide, >1 = raised high."""
        rig = self.r; local = {}
        flutter = 0.06 * np.sin(2 * np.pi * 1.1 * t) * smoothstep(0.3, 0.8, wo)
        for n in rig.names:
            if not n.startswith('Wing'): continue
            side = 1 if n.endswith('_L') else -1
            depth = int(n[4])
            if depth == 0:     # root: swing the whole wing back/up
                Rw = rot([1, 0, 0], -(1 - wo) * 118 * DEG) @ rot([0, 1, 0], side * (1 - wo) * 26 * DEG)
            elif depth == 1:   # humerus: raise/lower
                Rw = rot([0, 1, 0], side * ((wo - 1) * 28 + flutter * 40) * DEG)
            elif depth == 2:   # forearm: fold in
                Rw = rot([0, 1, 0], side * (wo - 1) * 35 * DEG)
            elif depth == 3:   # finger roots: close the fan
                k = int(n.split('_')[1])
                Rw = rot([0, 1, 0], -side * (1 - min(wo, 1)) * (10 + 14 * k) * DEG)
            else:              # finger segments: curl
                Rw = rot([0, 1, 0], -side * (1 - min(wo, 1)) * 18 * DEG)
            local[n] = rig.world_rot_local(n, Rw)
        return local
