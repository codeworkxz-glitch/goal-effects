"""Make the scythe keys anatomically valid and collision-free.

For each scythe key (plus extra in-between keys through the fast swing) search the scythe orientation/position
near the intended pose that minimises
  * wrist / forearm strain, elbow orientation and reach for both arms (choreo_reaper.Solver.solve_arm)
  * the shaft / blade coming within a margin of the torso, robe, hood or wings (posed mesh, BVH)
  * the arms sinking into the torso
  * blade height: >= 1 unit above the ground, except the impact/hold keys where the blade tip must bite into it

    GRIP=psi,off python optimize_keys.py     -> work/scythe_keys.json, work/grip_r.json
"""
import json, os, sys, numpy as np
from scipy.optimize import minimize
import bpy  # noqa: F401  (mathutils)
from mathutils import Vector
from mathutils.bvhtree import BVHTree
import choreo_reaper as C
import reaper_rig as RR
from reaper_rig import rot, nrm

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ['REAPER_RAW_KEYS'] = '1'           # intent keys, mirrored grip template
S = C.Solver()
rig = S.r
TOPO = np.load(os.path.join(RR.WORK, 'topo.npz'))
WT = np.load(os.path.join(RR.WORK, 'weights.npz'))
D = rig.D
EINV = np.linalg.inv(rig.E)

# ---- body proxy: torso + robe + hood flaps (+ wings) triangles, skinned in numpy
PROXY = []
for mn in ['cadnav_trans_offset', 'Object002_trans_offset', 'Object003_trans_offset', 'Object004_trans_offset']:
    V0 = D['V_' + mn][RR.BIND].astype(float) - rig.off
    tris = TOPO['tri_' + mn]; tc = TOPO['tcls_' + mn]
    keep = np.isin(tc, [1, 4])                 # torso + wings (arms and hands are not obstacles here)
    PROXY.append((V0, WT['idx_' + mn], WT['w_' + mn], tris[keep]))

# ---- scythe samples in the scythe (S0) frame: shaft axis + blade vertices
c_ax, d_ax = S.S0_axis
S0inv = np.linalg.inv(S.S0)
SHAFT = np.array([(S0inv @ np.append(c_ax + d_ax * t, 1))[:3] for t in np.arange(-50, 38, 1.0)])
VS = D['V_Object001_trans_offset'][RR.BIND].astype(float) - rig.off
VS_loc = (S0inv @ np.c_[VS, np.ones(len(VS))].T).T[:, :3]
rng = np.random.default_rng(0)
BLADE = VS_loc[rng.choice(len(VS_loc), 700, replace=False)]


def skin_proxy(W):
    Dm = W @ EINV
    verts, tris = [], []
    off = 0
    for V0, idx, w, tr in PROXY:
        Vh = np.c_[V0, np.ones(len(V0))]
        V = np.zeros_like(V0)
        for k in range(4):
            V += w[:, k:k + 1] * np.einsum('nij,nj->ni', Dm[idx[:, k]], Vh)[:, :3]
        verts.append(V); tris.append(tr + off); off += len(V)
    V = np.concatenate(verts); T = np.concatenate(tris)
    return BVHTree.FromPolygons([Vector(p) for p in V], [tuple(t) for t in T], epsilon=0.0)


def near_pen(bvh, P, margin):
    pen = 0.0
    for p in P:
        r = bvh.find_nearest(Vector(p), margin)
        if r[0] is not None:
            pen += (margin - r[3]) ** 2
    return pen


def angle(a, b): return np.degrees(np.arccos(np.clip(nrm(a) @ nrm(b), -1, 1)))


def make_GR(psi, off):
    G = S.GR_template.copy()
    R = np.eye(4); R[:3, :3] = rot([1.0, 0, 0], psi)
    G = R @ G
    G[0, 3] = C.T_LG + off
    return G


IMPACT_KEYS = (C.T_IMPACT, 3.24, C.T_SINK0, C.DURATION)


def key_cost(x, k, GR, ctx, report=False):
    t, p_des, R_des, both, loose = k[:5]
    cont = k[5] if len(k) > 5 else {}
    om, dp = x[:3], x[3:]
    th = np.linalg.norm(om)
    R = rot(om / th, th) @ R_des if th > 1e-9 else R_des
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = p_des + dp
    M[2, 3] += C.root_z(t) - C.REST_Z
    W, bvh = ctx
    dg = {}; c = 0.0; segs = []
    GRt = GR.copy(); GRt[0, 3] = C.T_LG - C.sep(t)
    for s, G in (('L', S.GL), ('R', GRt)):
        if s == 'R' and not both: continue
        best, cost = S.solve_grip(s, W, M, G, None, None, step=12, rstep=30, cont_roll=cont.get(s))
        c += cost; dg[s] = best
        for A, B in ((best['U'][:3, 3], best['F'][:3, 3]), (best['F'][:3, 3], best['H'][:3, 3])):
            segs.extend([A + (B - A) * u for u in np.linspace(0.15, 0.85, 5)])
    c += (angle(R[:, 0], R_des[:, 0]) / (25.0 * loose)) ** 2 + 0.5 * (angle(R[:, 1], R_des[:, 1]) / (35.0 * loose * loose)) ** 2
    c += (np.linalg.norm(dp) / (8.0 * loose)) ** 2
    shaft = (M @ np.c_[SHAFT, np.ones(len(SHAFT))].T).T[:, :3]
    blade = (M @ np.c_[BLADE, np.ones(len(BLADE))].T).T[:, :3]
    col_s = near_pen(bvh, shaft, 3.2) + near_pen(bvh, blade, 1.2)
    col_a = near_pen(bvh, np.array(segs), 1.6) if segs else 0.0
    c += 6.0 * col_s + 3.0 * col_a
    zmin = min(blade[:, 2].min(), shaft[:, 2].min())
    standing = C.T_RISE_END - 0.05 < t < C.T_SINK0 + 0.01
    if any(abs(t - ti) < 1e-6 for ti in IMPACT_KEYS):
        target = -5.0 + (C.root_z(t) - C.REST_Z)
        c += ((zmin - target) / 2.0) ** 2
    elif standing and zmin < 1.0:
        c += ((1.0 - zmin) / 2.0) ** 2
    if report: return c, dg, R, p_des + dp, dict(col_s=col_s, col_a=col_a, zmin=zmin)
    return c


def build_keys():
    """intent keys + in-between keys (taken from the intent curve) through the fast part"""
    keys = [list(k) for k in C.SCYTHE_KEYS]
    extra_t = [2.63, 2.81, 3.07]
    for t in extra_t:
        M = C.scythe_frame(t, rig)
        M[2, 3] -= C.root_z(t) - C.REST_Z
        keys.append([t, tuple(M[:3, 3]), tuple(M[:3, 0]), tuple(M[:3, 1])])
    keys.sort(key=lambda k: k[0])
    return keys


def optimise_all(GR, keys, restarts=4, iters=700, verbose=False, prev=None):
    total = 0; res = []
    for k in keys:
        t, p, d, e = k
        R_des = C.scythe_rot(np.array(d), np.array(e)); p = np.array(p, float)
        both = C.scalar_keys(C.GRIP_R_KEYS, t) >= 1.0
        loose = 1.0 if (t < C.T_GRAB + 0.05 or t > C.T_RAISE + 0.01) else 1.8
        kk = (t, p, R_des, both, loose)
        M = np.eye(4); M[:3, :3] = R_des; M[:3, 3] = p; M[2, 3] += C.root_z(t) - C.REST_Z
        twists = [0.0] if not both else [-20.0, 0.0, 20.0]
        best = None; best_tw = 0.0; best_ctx = None
        for tw in twists:
            W = S.pose(t, S_override=M, twist_extra=tw)[0]
            ctx = (W, skin_proxy(W))
            starts = [np.zeros(6)]
            for seed in range(restarts):
                rr = np.random.default_rng(seed + 1)
                starts.append(np.concatenate([rr.normal(0, 0.35, 3), rr.normal(0, 4, 3)]))
            for x0 in starts:
                r = minimize(key_cost, x0, args=(kk, GR, ctx), method='Nelder-Mead', options=dict(maxiter=iters, xatol=1e-3, fatol=1e-4))
                fun = r.fun + (tw / 30.0) ** 2 * 0.6          # prefer the authored twist
                if best is None or fun < best[0]: best = (fun, r); best_tw = tw; best_ctx = ctx
        r = best[1]
        c, dg, R, pp, info = key_cost(r.x, kk, GR, best_ctx, report=True)
        total += c
        res.append(dict(t=t, p=pp.tolist(), d=R[:, 0].tolist(), e=R[:, 1].tolist(), cost=c, twist=best_tw))
        best = r
        if verbose:
            print(f't={t:4.2f} tw {best_tw:+4.0f} cost {c:6.2f} dDev {angle(R[:,0], R_des[:,0]):5.1f} eDev {angle(R[:,1], R_des[:,1]):5.1f} '
                  f'dp {np.linalg.norm(best.x[3:]):4.1f} colS {info["col_s"]:5.2f} colA {info["col_a"]:5.2f} zmin {info["zmin"]:6.1f} ' +
                  ' | '.join(f'{s}: roll {np.degrees(v["roll"]):+4.0f} cl {v["clamp"]:.1f} wr {v["wrist"]:.0f} sw {v["swing"]:.0f} tw {v["twist"]:.0f} fo {v["fore"]:.0f} elb {v["elbow"]:.0f}' for s, v in dg.items()), flush=True)
    return total, res


def refine(GR, keys_intent, current, iters=500):
    """sequential pass with continuity (twist, grip rolls) to the previous key; hold keys copy the impact"""
    out = []
    prev = None
    cur = {round(k['t'], 3): k for k in current}
    for k in keys_intent:
        t, p, d, e = k
        both = C.scalar_keys(C.GRIP_R_KEYS, t) >= 1.0
        if t > C.T_IMPACT + 1e-6 and out:
            # hold / sink: the impact pose (with the authored small rebound offset)
            imp = [o_ for o_ in out if abs(o_['t'] - C.T_IMPACT) < 1e-6][0]
            ip = np.array(imp['p']); dp = np.array(p) - np.array([x_ for x_ in C.SCYTHE_KEYS if abs(x_[0] - C.T_IMPACT) < 1e-6][0][1])
            o_ = dict(imp); o_['t'] = t; o_['p'] = (ip + dp).tolist(); out.append(o_); continue
        R_des = C.scythe_rot(np.array(d), np.array(e)); p = np.array(p, float)
        loose = 1.0 if (t < C.T_GRAB + 0.05 or t > C.T_RAISE + 0.01) else 1.8
        if abs(t - C.T_IMPACT) < 1e-6: loose = 2.2
        c0 = cur.get(round(t, 3))
        # warm start: current solution relative to the intent
        x0 = np.zeros(6)
        if c0 is not None:
            Rc = C.scythe_rot(np.array(c0['d']), np.array(c0['e']))
            Rrel = Rc @ R_des.T
            from scipy.spatial.transform import Rotation as Rot
            x0 = np.concatenate([Rot.from_matrix(Rrel).as_rotvec(), np.array(c0['p']) - p])
        cont = {}
        if prev is not None:
            cont = {s_: prev['roll' + s_] for s_ in 'LR' if prev.get('roll' + s_) is not None}
        tw_prev = prev['twist'] if prev is not None else 0.0
        twists = [0.0] if not both else sorted(set([tw_prev - 10, tw_prev, tw_prev + 10, (c0 or {}).get('twist', 0.0)]))
        twists = [max(-30.0, min(30.0, tw)) for tw in twists]
        kk = (t, p, R_des, both, loose, cont)
        M = np.eye(4); M[:3, :3] = R_des; M[:3, 3] = p; M[2, 3] += C.root_z(t) - C.REST_Z
        best = None
        for tw in twists:
            W = S.pose(t, S_override=M, twist_extra=tw)[0]
            ctx = (W, skin_proxy(W))
            for xs in (x0, np.zeros(6)):
                r = minimize(key_cost, xs, args=(kk, GR, ctx), method='Nelder-Mead', options=dict(maxiter=iters, xatol=1e-3, fatol=1e-4))
                fun = r.fun + ((tw - tw_prev) / 15.0) ** 2 + (tw / 30.0) ** 2 * 0.4
                if best is None or fun < best[0]: best = (fun, r, tw, ctx)
        fun, r, tw, ctx = best
        c, dg, R, pp, info = key_cost(r.x, kk, GR, ctx, report=True)
        o_ = dict(t=t, p=pp.tolist(), d=R[:, 0].tolist(), e=R[:, 1].tolist(), cost=c, twist=tw,
                  rollL=float(dg['L']['roll']) if 'L' in dg else None, rollR=float(dg['R']['roll']) if 'R' in dg else None)
        out.append(o_); prev = o_
        print(f't={t:4.2f} tw {tw:+4.0f} cost {c:6.2f} dDev {angle(R[:,0], R_des[:,0]):5.1f} eDev {angle(R[:,1], R_des[:,1]):5.1f} '
              f'colS {info["col_s"]:5.2f} colA {info["col_a"]:5.2f} zmin {info["zmin"]:6.1f} ' +
              ' | '.join(f'{s}: roll {np.degrees(v["roll"]):+4.0f} wr {v["wrist"]:.0f} sw {v["swing"]:.0f} tw {v["twist"]:.0f} fo {v["fore"]:.0f} elb {v["elbow"]:.0f}' for s, v in dg.items()), flush=True)
    # unwrap the rolls along time so the bake interpolates the short way round
    for s_ in 'LR':
        idx = [i for i, o_ in enumerate(out) if o_.get('roll' + s_) is not None]
        un = np.unwrap([out[i]['roll' + s_] for i in idx])
        for i, r_ in zip(idx, un): out[i]['roll' + s_] = float(r_)
    return out


if __name__ == '__main__':
    if os.environ.get('REFINE'):
        psi, off = [float(v) for v in os.environ.get('GRIP', '300,-28').split(',')]
        GR = make_GR(np.radians(psi), off)
        cur = json.load(open(os.path.join(HERE, 'work', 'scythe_keys.json')))
        out = refine(GR, build_keys(), cur)
        json.dump(out, open(os.path.join(HERE, 'work', 'scythe_keys.json'), 'w'), indent=1)
        sys.exit()
    psi, off = [float(v) for v in os.environ.get('GRIP', '300,-28').split(',')]
    GR = make_GR(np.radians(psi), off)
    keys = build_keys()
    tot, res = optimise_all(GR, keys, restarts=1, iters=500, verbose=True)
    print('total', tot)
    json.dump(res, open(os.path.join(HERE, 'work', 'scythe_keys.json'), 'w'), indent=1)
    json.dump(dict(psi=float(np.radians(psi)), off=off, GR=GR.tolist()), open(os.path.join(HERE, 'work', 'grip_r.json'), 'w'), indent=1)
