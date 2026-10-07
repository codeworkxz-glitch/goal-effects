"""
Dread Serpent – choreography (pure numpy, no Blender needed).

World frame: rim centre at the origin, Z up, the court/camera is at -Y, the backboard is behind (+Y).
Lengths are in *model units* (the source FBX units); METRES_PER_UNIT converts at export time.

The body is driven "follow-the-leader": every frame a centre-line path is built and the spine is laid
on it, the head at arclength s_head and each body point u units behind it at s_head - u.  Whatever the
head does therefore travels head -> spine -> tail on its own.  On top of that come time-varying parts:
  * the coil radius tightens at rim height once the thick head/chest has passed (constriction),
  * the neck curve N(t) is a cubic Bezier from the end of the coil to the head (rear-up, recoil, strike),
  * small travelling ripples (head -> tail) keep the body alive during the stare.
"""
import numpy as np

FPS = 60
RIM_R = 10.5                     # inner rim radius in model units (sized so head+chest pass through)
METRES_PER_UNIT = 0.2286 / RIM_R  # regulation rim: 45.72 cm inner diameter
FLOOR_Z = -3.048 / METRES_PER_UNIT  # floor 3.05 m below the rim
BACKBOARD_Y = (0.2286 + 0.1524) / METRES_PER_UNIT  # board face: 15 cm behind the rim's back inner edge

U_HEAD = 4.0          # rest-pose Y of the head pivot (neck joint)
U_TAIL = 126.0        # rest-pose Y of the tail tip
BODY_LEN = U_TAIL - U_HEAD

# ------------------------------------------------------------------------------------------------
# timing (seconds)
# ------------------------------------------------------------------------------------------------
T_COIL_END = 1.50     # head reaches the end of the coil
T_REAR_END = 2.08     # head arrives above the hoop (stare starts)
T_RECOIL = 2.88       # recoil / draw-back starts
T_STRIKE = 3.16       # strike starts
T_STRIKE_HIT = 3.30   # head pointing straight down above the rim centre
T_GONE = 4.05         # tail tip is below the floor
DURATION = 4.00
NFRAMES = int(round(DURATION * FPS)) + 1


def smoothstep(e0, e1, x):
    x = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return x * x * (3 - 2 * x)


def smootherstep(e0, e1, x):
    x = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return x * x * x * (x * (x * 6 - 15) + 10)


def nrm(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def pchip(xk, yk):
    """Monotone cubic (Fritsch-Carlson) interpolant; returns f(x)."""
    xk = np.asarray(xk, float); yk = np.asarray(yk, float)
    h = np.diff(xk); d = np.diff(yk) / h
    m = np.zeros_like(yk)
    for i in range(1, len(xk) - 1):
        if d[i - 1] * d[i] > 0:
            w1 = 2 * h[i] + h[i - 1]; w2 = h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
    m[0] = d[0]; m[-1] = d[-1]

    def f(x):
        x = np.asarray(x, float)
        i = np.clip(np.searchsorted(xk, x) - 1, 0, len(h) - 1)
        t = (x - xk[i]) / h[i]
        h00 = 2 * t ** 3 - 3 * t ** 2 + 1; h10 = t ** 3 - 2 * t ** 2 + t
        h01 = -2 * t ** 3 + 3 * t ** 2; h11 = t ** 3 - t ** 2
        return h00 * yk[i] + h10 * h[i] * m[i] + h01 * yk[i + 1] + h11 * h[i] * m[i + 1]
    return f


def hermite_s(t, t0, t1, s0, s1, v0, v1):
    """Cubic Hermite in time for an arclength with given end speeds."""
    h = t1 - t0; x = np.clip((t - t0) / h, 0, 1)
    h00 = 2 * x ** 3 - 3 * x ** 2 + 1; h10 = x ** 3 - 2 * x ** 2 + x
    h01 = -2 * x ** 3 + 3 * x ** 2; h11 = x ** 3 - x ** 2
    return h00 * s0 + h10 * h * v0 + h01 * s1 + h11 * h * v1


def bezier(p0, p1, p2, p3, n=120):
    t = np.linspace(0, 1, n)[:, None]
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * t * t * p2 + (t ** 3) * p3


def polylen(P):
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())


def cumlen(P):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])


def resample(P, step):
    s = cumlen(P); n = max(2, int(np.ceil(s[-1] / step)) + 1)
    ss = np.linspace(0, s[-1], n)
    return np.stack([np.interp(ss, s, P[:, i]) for i in range(3)], 1)


def gsmooth(P, sigma_pts, keep_ends=True):
    """Gaussian smoothing of a polyline (sampled ~uniformly), with reflected/pinned ends."""
    if sigma_pts <= 0: return P
    r = int(3 * sigma_pts); k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_pts) ** 2); k /= k.sum()
    # odd reflection keeps endpoints and end tangents
    a = 2 * P[0] - P[r:0:-1]; b = 2 * P[-1] - P[-2:-r - 2:-1]
    Q = np.concatenate([a, P, b])
    out = np.stack([np.convolve(Q[:, i], k, mode='valid') for i in range(3)], 1)
    return out


# ------------------------------------------------------------------------------------------------
# static pieces
# ------------------------------------------------------------------------------------------------
COIL_TH0 = np.radians(58.0)       # coil starts back-right, below the backboard
COIL_SPAN = np.radians(302.0)     # wraps counter-clockwise (seen from above) to the right side, the neck completes the turn
COIL_Z0, COIL_Z1 = -26.0, 16.0    # rises through the rim plane (crossing at the front)
COIL_R_TIGHT = 15.0               # body (half-width 2) hugs the rim:  14.2-2.1 = 12.1 > rim outer 10.9
COIL_R_WIDE = 19.6                # while the 6-unit-wide head/chest passes the rim plane
TRANS_R = 14.0                    # column -> coil turn radius
STEP = 0.5

S_PAUSE = np.array([0.0, 3.5, 28.0])               # head pivot during the stare
D_PAUSE = nrm([0.0, -0.62, 0.785])                 # chest direction during the stare (leans forward)
S_RECOIL = np.array([0.0, 5.6, 30.8])
D_RECOIL = nrm([0.0, -0.40, 0.917])
NECK_HANDLE = 12.0                                 # straight chest behind the head
Q_HIT_Z = 27.0                                     # head pivot when the head points straight down


CINCH_NEAR, CINCH_FAR = 16.0, 42.0                  # coil stays wide this far behind the head (units)


def coil_points(thv):
    """Helix around the rim. thv = head position along the coil (0..1, >1 once past the end, <0 not
    reached).  The coil is wide right around the head/chest (so the head, the hanging claws and the
    mane clear the rim) and cinches tight onto the rim behind it.  The first part (below the backboard)
    stays put so the erupting column doesn't move."""
    th = np.linspace(0, 1, 400)
    z = COIL_Z0 + (COIL_Z1 - COIL_Z0) * th
    ang = COIL_TH0 + COIL_SPAN * th
    lc = COIL_SPAN * COIL_R_WIDE
    behind = (thv - th) * lc
    w = 1.0 - smoothstep(CINCH_NEAR, CINCH_FAR, behind)
    w = np.maximum(w, 1.0 - smoothstep(0.06, 0.22, th))
    r = COIL_R_TIGHT + (COIL_R_WIDE - COIL_R_TIGHT) * w
    return np.stack([r * np.cos(ang), r * np.sin(ang), z], 1)


def column_points(C):
    """Erupting column (from deep below the floor) + turn into the coil start; returns points, lateral dir."""
    p0 = C[0]; t0 = nrm(C[3] - C[0])
    up = np.array([0.0, 0.0, 1.0])
    w = nrm(t0 - up * (t0 @ up))
    alpha = np.arccos(np.clip(t0 @ up, -1, 1))
    beta = np.linspace(0, alpha, 60)[:, None]
    arc_rel = TRANS_R * (np.sin(beta) * up + (1 - np.cos(beta)) * w)
    A = p0 - arc_rel[-1]
    arc = A + arc_rel
    lat = np.cross(up, w)
    h = np.arange(0, 470.0, 1.0)[::-1]                 # depth below A
    fade = smoothstep(8.0, 45.0, h)
    wig = (4.2 * np.sin(2 * np.pi * h / 46.0 + 0.7)) * fade
    wig2 = (1.6 * np.sin(2 * np.pi * h / 31.0 + 2.1)) * fade
    col = A[None] - h[:, None] * up + wig[:, None] * lat + wig2[:, None] * w
    return np.concatenate([col, arc[1:]]), w


def descent_points(q_end):
    """From the strike point straight down through the rim centre, then curving out toward the court and
    diving through the floor (serpentine wiggle once clear of the rim)."""
    z_top = q_end[2]
    seg1 = np.stack([np.zeros(80), np.zeros(80), np.linspace(z_top, -12.0, 80)], 1)
    R = 26.0; ang = np.radians(38.0)
    b = np.linspace(0, ang, 50)[:, None]
    # turn toward -Y
    arc = np.concatenate([np.zeros_like(b), -R * (1 - np.cos(b)), -12.0 - R * np.sin(b)], 1)
    d = np.array([0.0, -np.sin(ang), -np.cos(ang)])
    L = np.arange(1.0, 520.0, 1.0)[:, None]
    lat = np.array([1.0, 0.0, 0.0])
    fade = smoothstep(5.0, 40.0, L[:, 0])
    line = arc[-1] + L * d + (fade * 4.6 * np.sin(2 * np.pi * L[:, 0] / 48.0))[:, None] * lat
    P = np.concatenate([seg1, arc[1:], line])
    return P


def len_base(base):
    return polylen(base)


def static_tail_part(thv):
    C = coil_points(thv)
    G, w = column_points(C)
    P = np.concatenate([G, C[1:]])
    return P, w, len(G)


# ------------------------------------------------------------------------------------------------
# neck curve and head target over time
# ------------------------------------------------------------------------------------------------
def coil_end_frame(thv):
    C = coil_points(thv)
    E = C[-1]; TE = nrm(C[-1] - C[-4])
    return E, TE


NECK_OUT = 7.0   # the neck leaves the coil bulging outward (wider, rounder arch over the hoop)


def neck_bezier(E, TE, Q, d, a=9.0, b=NECK_HANDLE):
    radial = nrm(np.array([E[0], E[1], 0.0]))
    return bezier(E, E + TE * a + radial * NECK_OUT, Q - d * b, Q, 160)


def strike_state(t):
    """Head pivot Q and chest direction d (path tangent at the head) for t >= T_REAR_END."""
    if t <= T_RECOIL:
        return S_PAUSE.copy(), D_PAUSE.copy()
    if t <= T_STRIKE:
        f = smootherstep(T_RECOIL, T_STRIKE, t)
        Q = S_PAUSE + (S_RECOIL - S_PAUSE) * f
        d = nrm(D_PAUSE + (D_RECOIL - D_PAUSE) * f)
        return Q, d
    # strike: d swings from D_RECOIL to straight down; Q sweeps forward/down onto the rim axis
    f = smoothstep(T_STRIKE, T_STRIKE_HIT, t)
    f = f ** 1.35
    phi0 = np.arctan2(-D_RECOIL[1], D_RECOIL[2])
    phi = phi0 + (np.pi - phi0) * f
    d = np.array([0.0, -np.sin(phi), np.cos(phi)])
    Qe = np.array([0.0, 0.0, Q_HIT_Z])
    # blend of a swinging arc and a straight line so that Q lands exactly on Qe
    Q = S_RECOIL + (Qe - S_RECOIL) * smoothstep(0, 1, f)
    Q = Q + np.array([0.0, -2.5, 2.0]) * np.sin(np.pi * f)
    return Q, d


# ------------------------------------------------------------------------------------------------
# head arclength schedule
# ------------------------------------------------------------------------------------------------
_cache = {}


def _lengths():
    if 'L' in _cache: return _cache['L']
    P0, w, nG = static_tail_part(-1.0)
    sG = cumlen(P0)
    s_col = sG[nG - 1]
    floor_i = np.argmax(P0[:, 2] > FLOOR_Z)
    s_floor = sG[floor_i]
    s_coil_end = sG[-1]
    E, TE = coil_end_frame(-1.0)
    N = neck_bezier(E, TE, S_PAUSE, D_PAUSE)
    s_pause = s_coil_end + polylen(N)
    _cache['L'] = dict(s_floor=s_floor, s_coil_end=s_coil_end, s_pause=s_pause, lenN=polylen(N), s_col=s_col,
                       lc=s_coil_end - s_col)
    return _cache['L']


def head_thv(t):
    """Head position along the coil in coil units (see coil_points)."""
    L = _lengths()
    s_ref = head_s_pre(min(t, T_REAR_END))
    thv = (s_ref - L['s_col']) / L['lc']
    if s_ref < L['s_col']: thv = -1.0
    # after arriving above the hoop the coil keeps cinching onto the rim
    return thv + 0.6 * smootherstep(T_REAR_END - 0.1, T_REAR_END + 0.6, t)


def head_s_pre(t):
    """Head arclength on the static path (column + wide coil + pause neck) for t <= T_REAR_END."""
    L = _lengths()
    s0 = L['s_floor'] - 17.0           # snout just below the floor at t=0
    f = pchip([0.0, 0.10, 0.55, 1.05, T_COIL_END, T_REAR_END - 0.25, T_REAR_END],
              [s0, s0 + 22.0, s0 + 118.0, L['s_coil_end'] - 46.0, L['s_coil_end'],
               L['s_pause'] - 3.0, L['s_pause']])
    return float(f(t))


# ------------------------------------------------------------------------------------------------
# per-frame centre line
# ------------------------------------------------------------------------------------------------
def frame_path(t):
    """Returns dict: P (Mx3 polyline, start deep underground, ends at/after the head), s (cum length),
    s_head, roll keys (list of (s, dorsal)), head_dir (unit), phase."""
    sq = head_thv(t)
    base, w, nG = static_tail_part(sq)
    E, TE = coil_end_frame(sq)
    keys = []
    up = np.array([0.0, 0.0, 1.0])
    sb = cumlen(base)
    # dorsal keys: column faces away from the turn (belly inside), coil dorsal up
    keys.append((0.0, -w))
    keys.append((sb[nG - 60], -w))
    for i in range(nG + 20, len(base), 25):
        keys.append((sb[i], up))
    keys.append((sb[-1], up))
    if t <= T_REAR_END:
        N = neck_bezier(E, TE, S_PAUSE, D_PAUSE)
        P = np.concatenate([base, N[1:]])
        s = cumlen(P)
        s_pre = head_s_pre(t); Lr = _lengths()
        # schedule is in reference (all-wide) arclength; map it onto this frame's coil
        if s_pre <= Lr['s_col']:
            s_head = s_pre
        elif s_pre <= Lr['s_coil_end']:
            i = nG - 1 + int(round((s_pre - Lr['s_col']) / Lr['lc'] * 399))
            s_head = sb[min(i, len(sb) - 1)]
        else:
            s_head = sb[-1] + (s_pre - Lr['s_coil_end'])
        d_end = D_PAUSE
    elif t <= T_STRIKE_HIT:
        Q, d = strike_state(t)
        # the neck handle grows through the strike so the arch over the hoop stays round
        fb = smootherstep(T_STRIKE + 0.05, T_STRIKE_HIT, t)
        N = neck_bezier(E, TE, Q, d, a=9.0, b=NECK_HANDLE + 10.0 * fb)
        P = np.concatenate([base, N[1:]])
        s = cumlen(P)
        s_head = s[-1]
        d_end = d
    else:
        Qe, d = strike_state(T_STRIKE_HIT)
        N = neck_bezier(E, TE, Qe, d, a=9.0, b=NECK_HANDLE + 10.0)
        D = descent_points(Qe)
        P = np.concatenate([base, N[1:], D[1:]])
        s = cumlen(P)
        s_hit = len(base) + len(N) - 2
        s_hit = s[s_hit]
        L = _lengths()
        # fast, accelerating dive; tail must clear the floor by T_GONE
        v_hit = 260.0
        dist_total = 560.0
        s_head = s_hit + hermite_s(t, T_STRIKE_HIT, DURATION, 0.0, dist_total, v_hit, 420.0)
        d_end = np.array([0.0, 0.0, -1.0])
    sN0 = cumlen(base)[-1]
    lenN = polylen(N)
    # neck dorsal: perpendicular to the chest, pointing up/back (head upright when it looks forward)
    # neck dorsal: in the YZ plane, perpendicular to the chest and on its back/up side; this stays
    # continuous through the whole hammer swing of the strike (top of the head leads when it points down)
    dn = np.array([0.0, d_end[2], -d_end[1]])
    keys.append((sN0 + 0.55 * lenN, nrm(nrm(dn) + up)))
    keys.append((sN0 + lenN, nrm(dn)))
    if t > T_STRIKE_HIT:
        sD0 = sN0 + lenN
        # corkscrew half-roll while plunging through the rim, so the belly faces the turn toward the court
        keys.append((sD0 + 28.0, np.array([1.0, 0.0, 0.0])))
        keys.append((sD0 + 50.0, np.array([0.0, 1.0, 0.0])))
        keys.append((sD0 + 75.0, nrm([0.0, 0.788, -0.616])))
        keys.append((s[-1], nrm([0.0, 0.788, -0.616])))
    return dict(P=P, s=s, s_head=s_head, keys=keys, sq=sq, sN0=sN0, lenN=lenN, nG=nG)


# ------------------------------------------------------------------------------------------------
# frames along the path: parallel transport + roll interpolated toward the dorsal keys
# ------------------------------------------------------------------------------------------------
def path_frames(P, s, keys):
    T = np.gradient(P, axis=0); T = nrm(T)
    n = len(P)
    D = np.zeros_like(P)
    d0 = keys[0][1]; d0 = nrm(d0 - T[0] * (d0 @ T[0])); D[0] = d0
    for i in range(1, n):
        a, b = T[i - 1], T[i]
        v = np.cross(a, b); c = a @ b
        if np.linalg.norm(v) < 1e-9:
            D[i] = D[i - 1]
        else:
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R = np.eye(3) + vx + vx @ vx * (1 / (1 + c))
            D[i] = R @ D[i - 1]
        D[i] = nrm(D[i] - b * (D[i] @ b))
    B = np.cross(T, D)
    # roll (angle about T) needed at each key
    ks = np.array([k[0] for k in keys]); order = np.argsort(ks); ks = ks[order]
    ang = []
    for j in order:
        sk, dk = keys[j]
        i = int(np.clip(np.searchsorted(s, sk), 0, n - 1))
        dk = dk - T[i] * (dk @ T[i])
        if np.linalg.norm(dk) < 1e-6: dk = D[i]
        dk = nrm(dk)
        ang.append(np.arctan2(dk @ B[i], dk @ D[i]))
    ang = np.unwrap(np.array(ang))
    roll = np.interp(s, ks, ang)
    # smooth the roll a little more in arclength (keys are sparse)
    roll = gsmooth(np.stack([roll, roll, roll], 1), 12)[:, 0] if len(roll) > 80 else roll
    Dr = np.cos(roll)[:, None] * D + np.sin(roll)[:, None] * B
    return T, nrm(Dr)


def sample_on_path(P, s, x):
    x = np.clip(x, s[0], s[-1])
    return np.stack([np.interp(x, s, P[:, i]) for i in range(3)], -1)


def sample_vec(V, s, x):
    x = np.clip(x, s[0], s[-1])
    return nrm(np.stack([np.interp(x, s, V[:, i]) for i in range(3)], -1))
