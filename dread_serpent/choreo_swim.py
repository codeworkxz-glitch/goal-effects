"""
Dread Serpent – "Sea Serpent Swim" choreography (pure numpy).

The ground plane (Z = 0) is an invisible water surface.  The serpent swims along one long, smooth,
static centre line: a gentle weave in plan view plus three breach arcs above the surface and deep
troughs below it.  The spine is laid on the line follow-the-leader (every joint a fixed arclength
behind the head), so each breach rises from the head, rolls down the body and leaves through the tail,
exactly like a whale or sea serpent breaching and diving.  A small lateral swim wave travels head -> tail
on top of that.  The clip starts and ends with the whole body below the surface.

World frame: origin on the surface under the middle breach, Z up, the serpent swims along +X,
the viewer is at -Y.  Lengths in model units; METRES_PER_UNIT converts.
"""
import numpy as np
from choreo import (smoothstep, smootherstep, nrm, pchip, cumlen, path_frames, sample_on_path, sample_vec,
                    U_HEAD, U_TAIL, BODY_LEN)

OUT_NAME = 'DreadSerpent_SeaSerpentSwim'
FPS = 60
METRES_PER_UNIT = 0.10          # ~12.5 m serpent, ~1.3 m head
DURATION = 6.0
NFRAMES = int(round(DURATION * FPS)) + 1
GROUND_Z = 0.0

# breaches along the swim line (X = arclength-ish distance along the plan curve): centre, rise, width.
# Width >= sqrt(32 * rise) keeps the crest bend radius >= ~16 units (the head/chest never fold);
# spacing keeps the troughs between breaches deep enough that the dorsal spikes stay under the surface.
BASE_Z = -24.0
BREACHES = [(-148.0, 58.0, 43.5), (0.0, 66.0, 46.5), (148.0, 61.0, 44.5)]
PLAN_R = 135.0                  # the swim line circles the origin (counter-clockwise from above)
WEAVE_A, WEAVE_L = 6.0, 118.0   # sideways S weave on top of the circle
X_MIN, X_MAX, DX = -560.0, 520.0, 0.5


def _profile_z(X):
    z = np.full_like(X, BASE_Z)
    for xc, a, w in BREACHES:
        z = z + a * np.exp(-((X - xc) / w) ** 2)
    return z


def _build():
    X = np.arange(X_MIN, X_MAX + DX, DX)
    # middle breach at the front (-Y, facing the viewer), swimming toward +X there
    phi = -np.pi / 2 + X / PLAN_R
    r = PLAN_R + WEAVE_A * np.sin(2 * np.pi * X / WEAVE_L + 0.4)
    P = np.stack([r * np.cos(phi), r * np.sin(phi), _profile_z(X)], 1)
    return X, P


_X, _P = _build()
_S = cumlen(_P)
_UP = np.array([0.0, 0.0, 1.0])
_KEYS = [(float(_S[i]), _UP) for i in range(0, len(_S), 40)] + [(float(_S[-1]), _UP)]


def _s_at_X(x):
    return float(np.interp(x, _X, _S))


# head schedule: slows a little while climbing, surges while diving (momentum), starts and ends submerged
_s0 = _s_at_X(BREACHES[0][0] - 1.55 * BREACHES[0][2])
_last_xc, _, _last_w = BREACHES[-1]
_s1 = _s_at_X(_last_xc + 1.28 * _last_w) + BODY_LEN + 6.0   # tail fin fully under again


def _schedule():
    n = 4000
    s = np.linspace(_s0, _s1, n)
    z = np.interp(s, _S, _P[:, 2])
    slope = np.gradient(z, s)
    speed = 1.0 - 0.16 * np.clip(slope, -1.2, 1.2)      # climbing -> slower, diving -> faster
    dt = np.diff(s) / (0.5 * (speed[1:] + speed[:-1]))
    tt = np.concatenate([[0.0], np.cumsum(dt)])
    tt *= DURATION / tt[-1]
    return tt, s


_TT, _SS = _schedule()


def head_s(t):
    return float(np.interp(t, _TT, _SS))


def head_z(t):
    return float(np.interp(head_s(t), _S, _P[:, 2]))


def frame_path(t):
    return dict(P=_P, s=_S, s_head=head_s(t), keys=_KEYS, sq=0.0, sN0=-1e9, lenN=0.0, nG=0)


# ------------------------------------------------------------------------------------------------
# secondary motion hooks used by build_dread_serpent.py
# ------------------------------------------------------------------------------------------------
def ripple_amp(t, u, sj, fp):
    """lateral swim wave travelling head -> tail, growing toward the tail"""
    env = smoothstep(8.0, 30.0, u) * (0.75 + 1.25 * smoothstep(60.0, U_TAIL, u))
    return 1.25 * env * np.sin(2 * np.pi * 1.05 * t - 2 * np.pi * (u - U_HEAD) / 64.0)


def head_matrix(t, Mh, J, T, D, s, sh, K):
    """the head leads into each arc: it looks along the line a few units ahead of itself"""
    ahead = sample_vec(T, s, np.array([sh + 7.0]))[0]
    Y = Mh[:3, 1] * 0.45 + ahead * 0.55
    Y = Y / np.linalg.norm(Y)
    Z = Mh[:3, 2] - Y * (Mh[:3, 2] @ Y); Z /= np.linalg.norm(Z)
    M = Mh.copy(); M[:3, 1] = Y; M[:3, 2] = Z; M[:3, 0] = np.cross(Y, Z)
    return M


def jaw_open(t):
    """mostly closed while swimming; a slow breath/snarl when the head is high in the air"""
    h = smoothstep(4.0, 22.0, head_z(t))
    return float(3.0 + 13.0 * h)


def arm_pose(t):
    """arms tucked like fins, with a slow paddle"""
    paddle = 0.10 + 0.08 * np.sin(2 * np.pi * 1.05 * t - 0.8)
    return float(1.0 - paddle), float(paddle), 0.0


def look_weight(t):
    return 0.0


def arm_aim_weight(t):
    return 0.0
