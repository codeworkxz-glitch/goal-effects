"""Recover linear-blend-skinning weights from the Alembic bake.

    python fit_weights.py work/cache.npz work/weights.npz

For every vertex v:  v(f) ~= sum_j w_j * T_j(f) v(0),  T_j(f) = B_j(f) B_j(0)^-1 (bone world matrices from the
empties), w >= 0, sum w = 1, at most 4 bones.  Solved per vertex over all 97 frames with NNLS, plus a weak
prior toward distance-based weights so that bones that always move together don't produce noisy splits.
"""
import sys, numpy as np
from scipy.optimize import nnls

cache, out = sys.argv[-2], sys.argv[-1]
D = np.load(cache)
names = list(D['bone_names']); parents = list(D['bone_parents'])
BM = D['bone_mats']                       # F x B x 4 x 4
F, B = BM.shape[:2]
BIND = int(__import__('os').environ.get('BIND', '60'))
FR0 = int(__import__('os').environ.get('FR0', '30'))   # frames before this are a fast fly-in where the mesh
                                                      # cache and the bone samples are out of sync
T = BM @ np.linalg.inv(BM[BIND])[None]    # F x B x 4 x 4
pos0 = BM[BIND, :, :3, 3]
children = {j: [k for k in range(B) if parents[k] == names[j]] for j in range(B)}
tail0 = np.zeros_like(pos0)
for j in range(B):
    ch = children[j]
    if ch:
        tail0[j] = pos0[ch].mean(0)
    else:
        p = names.index(parents[j]) if parents[j] in names else None
        d = pos0[j] - pos0[p] if p is not None else np.array([0, 0, 5.0])
        tail0[j] = pos0[j] + 0.5 * d

# bones that should never deform skin (nub ends of fingers are fine to keep; they rarely win)
SKIP = set()


def seg_dist(P, a, b):
    ab = b - a; L = (ab * ab).sum() + 1e-9
    t = np.clip(((P - a) @ ab) / L, 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


res = {}
fstep = 1
for mname in D['mesh_names']:
    if mname.startswith('Object001'):      # the scythe: rigid, gets its own bone
        continue
    V = D['V_' + mname].astype(np.float64)  # F x N x 3
    N = V.shape[1]
    v0 = np.concatenate([V[BIND], np.ones((N, 1))], 1)
    dist = np.stack([seg_dist(V[BIND], pos0[j], tail0[j]) for j in range(B)], 1)  # N x B
    K = 10
    cand = np.argsort(dist, 1)[:, :K]
    idx = np.zeros((N, 4), int); wts = np.zeros((N, 4)); err = np.zeros(N)
    fr = np.arange(FR0, F, fstep)
    lam = 30.0 * np.sqrt(len(fr))
    mu = 0.6
    for i in range(N):
        c = cand[i]
        A = np.stack([(T[fr][:, j] @ v0[i])[:, :3].reshape(-1) for j in c], 1)   # 3F x K
        b = V[fr, i].reshape(-1)
        d = dist[i, c]
        prior = np.exp(-((d - d.min()) / (1.0 + 0.35 * d.min())) ** 2); prior /= prior.sum()
        A2 = np.vstack([A, lam * np.ones((1, K)), mu * np.eye(K)])
        b2 = np.concatenate([b, [lam], mu * prior])
        w, _ = nnls(A2, b2, maxiter=200)
        top = np.argsort(-w)[:4]
        c4 = c[top]
        A4 = A[:, top]
        A4b = np.vstack([A4, lam * np.ones((1, 4)), mu * np.eye(4)])
        b4b = np.concatenate([b, [lam], mu * prior[top] / max(prior[top].sum(), 1e-9)])
        w4, _ = nnls(A4b, b4b, maxiter=200)
        s = w4.sum()
        w4 = w4 / s if s > 1e-9 else np.eye(4)[0]
        idx[i] = c4; wts[i] = w4
        r = A4 @ w4 - b
        err[i] = np.sqrt((r.reshape(-1, 3) ** 2).sum(1).max())
    res['idx_' + mname] = idx; res['w_' + mname] = wts; res['err_' + mname] = err
    print(f'{mname:28s} verts {N:5d}  max-over-frames error: median {np.median(err):.3f}  p95 {np.percentile(err, 95):.3f}  max {err.max():.3f}  (units)')
np.savez(out, bone_names=np.array(names), tail0=tail0, bind=BIND, **res)
