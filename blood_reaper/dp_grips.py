"""Global grip-roll / torso-twist selection over all keys (Viterbi).

Uses the scythe key poses in work/scythe_keys.json (pass 1), evaluates both arms for every (twist, rollL, rollR)
state at every key, and picks the sequence minimising strain + smoothness (twist and grip changes between keys).
Writes twist / rollL / rollR back into the json.  Hold/sink keys copy the impact.
"""
import json, os, numpy as np
os.environ.pop('REAPER_RAW_KEYS', None)
import choreo_reaper as C
from reaper_rig import rot

HERE = os.path.dirname(os.path.abspath(__file__))
S = C.Solver()
keys = json.load(open(os.path.join(HERE, 'work', 'scythe_keys.json')))
for k in keys:
    for f in ('twist', 'rollL', 'rollR'): k.pop(f, None)
TW = np.array([-20.0, -10.0, 0.0, 10.0, 20.0])
RO = np.radians(np.arange(-180, 180, 30))
nT, nR = len(TW), len(RO)
imp_i = [i for i, k in enumerate(keys) if abs(k['t'] - C.T_IMPACT) < 1e-6][0]
act = keys[:imp_i + 1]

costs = []
for k in act:
    t = k['t']
    M = np.eye(4); M[:3, :3] = C.scythe_rot(np.array(k['d']), np.array(k['e'])); M[:3, 3] = k['p']
    M[2, 3] += C.root_z(t) - C.REST_Z
    both = C.scalar_keys(C.GRIP_R_KEYS, t) >= 1.0
    A = np.zeros((nT, nR, nR))
    for a, tw in enumerate(TW):
        W = S.pose(t, S_override=M, twist_extra=tw)[0]
        cl = np.zeros(nR); cr = np.zeros(nR)
        for j, r in enumerate(RO):
            Rr = np.eye(4); Rr[:3, :3] = rot([1.0, 0, 0], r)
            cl[j] = S.solve_arm('L', W, M @ Rr @ S.GL, None, step=8)[1] + 0.25 * (np.degrees(r) / 90) ** 2
            if both:
                GRt = S.GR.copy(); GRt[0, 3] = C.T_LG - C.sep(t)
                cr[j] = S.solve_arm('R', W, M @ Rr @ GRt, None, step=8)[1] + 0.25 * (np.degrees(r) / 90) ** 2
        A[a] = cl[:, None] + cr[None, :] + (tw / 30.0) ** 2 * 0.4
    costs.append(A)
    print('key', t, 'best', round(float(A.min()), 2), flush=True)


def wrap(d): return np.degrees(np.angle(np.exp(1j * d)))


dR = (wrap(RO[:, None] - RO[None, :]) / 45.0) ** 2
dT = ((TW[:, None] - TW[None, :]) / 12.0) ** 2
V = costs[0].copy(); back = []
for i in range(1, len(act)):
    dt = act[i]['t'] - act[i - 1]['t']
    sc = max(dt / 0.15, 0.4)
    both_prev = C.scalar_keys(C.GRIP_R_KEYS, act[i - 1]['t']) >= 1.0
    # T[pt, pl, pr, nt, nl, nr]
    T = (V[:, :, :, None, None, None]
         + dT[:, None, None, :, None, None] / sc
         + dR[None, :, None, None, :, None] / sc
         + (dR[None, None, :, None, None, :] / sc if both_prev else 0.0 * dR[None, None, :, None, None, :]))
    T = T.reshape(nT * nR * nR, nT * nR * nR)
    arg = T.argmin(0)
    V = T.min(0).reshape(nT, nR, nR) + costs[i]
    back.append(arg)
state = np.unravel_index(int(V.argmin()), V.shape); path = [state]
for arg in reversed(back):
    flat = np.ravel_multi_index(path[-1], V.shape)
    path.append(np.unravel_index(int(arg[flat]), V.shape))
path = path[::-1]
rl = np.unwrap([RO[p[1]] for p in path]); rr = np.unwrap([RO[p[2]] for p in path])
for i, (k, p) in enumerate(zip(act, path)):
    both = C.scalar_keys(C.GRIP_R_KEYS, k['t']) >= 1.0
    k['twist'] = float(TW[p[0]]); k['rollL'] = float(rl[i]); k['rollR'] = float(rr[i]) if both else None
    print(f"t={k['t']:4.2f} twist {k['twist']:+4.0f} rollL {np.degrees(rl[i]):+5.0f} "
          f"rollR {np.degrees(rr[i]) if both else float('nan'):+5.0f} cost {float(costs[i][p]):.2f}")
imp = act[-1]
intent_imp = [x for x in C.SCYTHE_KEYS if abs(x[0] - C.T_IMPACT) < 1e-6][0]
for k in keys[imp_i + 1:]:
    intent = [x for x in C.SCYTHE_KEYS if abs(x[0] - k['t']) < 1e-6]
    dp = np.array(intent[0][1]) - np.array(intent_imp[1]) if intent else np.zeros(3)
    k.update({x: imp[x] for x in ('d', 'e', 'twist', 'rollL', 'rollR')}); k['p'] = (np.array(imp['p']) + dp).tolist()
json.dump(keys, open(os.path.join(HERE, 'work', 'scythe_keys.json'), 'w'), indent=1)
