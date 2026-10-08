"""Global search for the authored scythe keys of the sweep and slam (the intent keys optimize_keys.py refines).

Each key is described by what it should read as -- the direction the blade head points (yaw / pitch, soft), which
way the blade faces, how low the blade runs -- and the search picks the scythe pose and position that best satisfy
that while keeping both hands in reach, the wrists relaxed, and the whole scythe (shaft, pommel and blade) and the
arms clear of the body and wings (optimize_keys.key_cost).  Keys run in parallel.

    [ONLY=t,t..] [TWISTS=a,b,..] python intent_search.py   -> work/intent_search.json  (paste into choreo_reaper.SCYTHE_KEYS)
"""
import json, os, sys, numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
TWISTS = [float(v) for v in os.environ.get('TWISTS', '-20,0,20').split(',')]

# t, head yaw (deg, 0 = straight ahead, + = the Reaper's left), yaw sigma, pitch, pitch sigma, blade facing,
# blade height target (units, None = free), left-hand height target (None = free)
CW = 'cw'      # blade faces the direction of travel of a left -> front -> right sweep
PLAN = [
    (2.02, 30, 25, 45, 20, CW, None, None),
    (2.14, 70, 20, 25, 20, CW, None, None),
    (2.25, 105, 20, 15, 20, CW, None, None),       # wound up over the left shoulder
    (2.33, 65, 20, -5, 20, CW, None, None),
    (2.40, 0, 20, -10, 25, CW, None, None),         # through the front
    (2.48, -30, 20, -5, 25, CW, None, None),
    (2.56, -60, 20, 8, 20, CW, None, None),         # follow-through to the right
    (2.70, -40, 25, 40, 20, CW, None, None),
    (2.92, 180, 30, 50, 15, (0.0, -1.0, 0.0), None, 100.0),    # raised overhead, head back over the shoulders
    (3.02, 0, 30, 40, 20, (0.0, -0.7, -0.7), None, 88.0),       # coming down
    (3.12, 10, 30, 10, 25, (0.0, 0.0, -1.0), None, None),       # impact: tip bites into the ground (key_cost target)
]


def dirs(yaw, pitch, beta, face):
    y, pi = np.radians(yaw), np.radians(pitch)
    h = np.array([np.sin(y), -np.cos(y), 0.0])
    d = np.cos(pi) * h + np.array([0, 0, np.sin(pi)])
    e = np.array([h[1], -h[0], 0.0]) if face == CW else np.array(face, float)
    e = e - d * (d @ e); e /= np.linalg.norm(e)
    n = np.cross(d, e)
    b = np.radians(beta)
    return d, np.cos(b) * e + np.sin(b) * n


def search(plan):
    sys.path.insert(0, HERE)
    import optimize_keys as O, choreo_reaper as C
    t, yaw0, ysig, pit0, psig, face, zt, hz = plan
    GR = O.make_GR(np.radians(300), -28)
    both = C.scalar_keys(C.GRIP_R_KEYS, t) >= 1.0
    best = None
    def context(tw, x):
        # body (torso + wings) proxy for the arms solved for scythe pose x (some torso-classed costume
        # triangles follow the arms); arms themselves are re-solved inside key_cost
        d0, e0 = dirs(x[0], x[1], x[2], face)
        M = np.eye(4); M[:3, :3] = C.scythe_rot(d0, e0); M[:3, 3] = x[3:]; M[2, 3] += C.root_z(t) - C.REST_Z
        W = O.S.pose(t, S_override=M, twist_extra=tw)[0]
        return (W, O.skin_proxy(W))

    def cost(x, ctx, tw, report=False):
        yaw, pitch, beta = x[:3]; p = x[3:]
        d, e = dirs(yaw, pitch, beta, face)
        R = C.scythe_rot(d, e)
        c, dg, _, _, info = O.key_cost(np.zeros(6), (t, p, R, both, 1.8), GR, ctx, report=True)
        c += ((yaw - yaw0) / ysig) ** 2 + ((pitch - pit0) / psig) ** 2 + (beta / 25.0) ** 2
        c += max(0.0, abs(pitch) - 85) ** 2
        if zt is not None: c += ((info['zmin'] - zt) / 8.0) ** 2
        if hz is not None: c += ((p[2] - hz) / 6.0) ** 2
        c += (tw / 30.0) ** 2 * 0.6
        return (c, dg, info, d, e) if report else c

    for tw in TWISTS:
        for p0 in ((0, -25, 75), (12, -20, 80), (-12, -20, 80), (0, -32, 90) if hz is None else (0, -25, hz)):
            x = np.array([yaw0, pit0, 0.0, *p0], float)
            for it in range(3):          # re-pose the body around the current solution, then refine
                ctx = context(tw, x)
                r = minimize(cost, x, args=(ctx, tw), method='Nelder-Mead',
                             options=dict(maxiter=300 if it == 0 else 150, xatol=0.05, fatol=1e-3,
                                          initial_simplex=x + np.vstack([np.zeros(6), np.diag([15, 12, 15, 6, 6, 6]) / (1 + 2 * it)])))
                x = r.x
            ctx = context(tw, x)
            fx = cost(x, ctx, tw)
            if best is None or fx < best[0]:
                best = (fx, x, tw, cost(x, ctx, tw, report=True))
    fun, x, tw, (c, dg, info, d, e) = best
    line = (f't={t:.2f} cost {fun:6.2f} tw {tw:+.0f} yaw {x[0]:6.1f} pitch {x[1]:5.1f} beta {x[2]:5.1f} p {np.round(x[3:], 1).tolist()} '
            f'colS {info["col_s"]:.2f} colA {info["col_a"]:.2f} zmin {info["zmin"]:.1f} ' +
            ' | '.join(f'{s}: wr {v["wrist"]:.0f} elb {v["elbow"]:.0f} cl {v["clamp"]:.1f}' for s, v in dg.items()))
    print(line, flush=True)
    return dict(t=t, p=x[3:].tolist(), d=d.tolist(), e=e.tolist(), twist=tw, cost=fun, line=line)


if __name__ == '__main__':
    only = [float(v) for v in os.environ['ONLY'].split(',')] if os.environ.get('ONLY') else None
    plan = [p for p in PLAN if only is None or any(abs(p[0] - o) < 1e-6 for o in only)]
    with Pool(4) as pool:
        res = pool.map(search, plan, chunksize=1)
    out = os.path.join(HERE, 'work', 'intent_search.json')
    old = {round(r['t'], 3): r for r in json.load(open(out))} if os.path.exists(out) else {}
    old.update({round(r['t'], 3): r for r in res})
    json.dump(sorted(old.values(), key=lambda r: r['t']), open(out, 'w'), indent=1)
    for r in sorted(old.values(), key=lambda r: r['t']):
        print(f"    ({r['t']:.2f}, ({r['p'][0]:.1f}, {r['p'][1]:.1f}, {r['p'][2]:.1f}), "
              f"({r['d'][0]:.3f}, {r['d'][1]:.3f}, {r['d'][2]:.3f}), ({r['e'][0]:.3f}, {r['e'][1]:.3f}, {r['e'][2]:.3f})),")
