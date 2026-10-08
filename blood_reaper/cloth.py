"""Spring-driven cloth bones for the parts that were cloth-simulated in the source (robe skirt panels, hip pouch,
belt charms).  Model units, E-frame convention (X along the bone).

Robe: 8 chains x 3 bones around the waist; every skirt vertex blends the two nearest chains (by angle) and the
two nearest joints (by height).  Trinkets: one pendulum bone each.  Joints follow damped springs toward their
rest shape on the pelvis (so the cloth keeps its shape but lags, swings and lifts), with fixed segment lengths.
"""
import numpy as np
import reaper_rig as RR
from reaper_rig import nrm

N_CHAIN = 8
N_SEG = 3
TRINKETS = {'Object004_trans_offset': 'Trinket_Pouch', 'Object006_trans_offset': 'Trinket_Charms',
            'Object007_trans_offset': 'Trinket_Chain'}


def rot_between(a, b):
    a = nrm(a); b = nrm(b); v = np.cross(a, b); c = a @ b
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else RR.rot(np.cross(a, [1, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1, 0]), np.pi)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


class Cloth:
    def __init__(self, rig):
        self.rig = rig
        D = rig.D; B = RR.BIND
        V = D['V_Object002_trans_offset'][B].astype(float) - rig.off
        self.Vrobe = V
        top = V[:, 2].max(); bot = V[:, 2].min()
        cen = V[V[:, 2] > top - 6][:, :2].mean(0)
        self.cen = cen
        ang = np.arctan2(V[:, 1] - cen[1], V[:, 0] - cen[0])
        self.chain_ang = np.linspace(-np.pi, np.pi, N_CHAIN, endpoint=False) + np.pi / N_CHAIN
        hz = np.linspace(top, bot, N_SEG + 1)
        self.hz = hz
        # chain joints: mean robe surface position in each angular sector / height
        J = np.zeros((N_CHAIN, N_SEG + 1, 3))
        for k, a in enumerate(self.chain_ang):
            da = np.abs(np.angle(np.exp(1j * (ang - a))))
            for j, z in enumerate(hz):
                w = np.exp(-(da / 0.5) ** 2) * np.exp(-((V[:, 2] - z) / 6.0) ** 2)
                if w.sum() < 1e-3:
                    w = np.exp(-(da / 1.0) ** 2) * np.exp(-((V[:, 2] - z) / 12.0) ** 2)
                J[k, j] = (V * w[:, None]).sum(0) / w.sum()
        self.J = J
        # weights for the robe vertices
        idx = np.zeros((len(V), 4), int); wts = np.zeros((len(V), 4))
        names = [f'Robe{k}_{j}' for k in range(N_CHAIN) for j in range(N_SEG)]
        self.robe_names = names
        for v in range(len(V)):
            da = np.angle(np.exp(1j * (ang[v] - self.chain_ang)))
            order = np.argsort(np.abs(da))[:2]
            a0, a1 = np.abs(da[order[0]]), np.abs(da[order[1]])
            wa = np.array([a1, a0]) / max(a0 + a1, 1e-9)
            # height parameter along the chain: tent over bone mid-points
            tpar = np.clip((top - V[v, 2]) / (top - bot) * N_SEG - 0.5, 0, N_SEG - 1)
            j0 = int(np.floor(tpar)); f = tpar - j0; j1 = min(j0 + 1, N_SEG - 1)
            c = 0
            for kk, wk in zip(order, wa):
                for jj, wj in ((j0, 1 - f), (j1, f)):
                    idx[v, c] = names.index(f'Robe{kk}_{jj}'); wts[v, c] = wk * wj; c += 1
        self.robe_idx, self.robe_w = idx, wts
        # trinkets: pivot at the top centre
        self.trink = {}
        for mn, bn in TRINKETS.items():
            P = D['V_' + mn][B].astype(float) - rig.off
            topz = P[:, 2].max()
            piv = P[P[:, 2] > topz - 1.5].mean(0)
            tip = P[P[:, 2] < P[:, 2].min() + 1.5].mean(0)
            self.trink[bn] = (piv, tip)
        # bone rest E-frames (X along the bone, Z ~ outward)
        self.rest = {}
        pel = rig.E[rig.idx['Pelvis']]
        for k in range(N_CHAIN):
            for j in range(N_SEG):
                a, b = J[k, j], J[k, j + 1]
                out = np.array([np.cos(self.chain_ang[k]), np.sin(self.chain_ang[k]), 0.0])
                M = np.eye(4); M[:3, :3] = RR.frame_from(b - a, out); M[:3, 3] = a
                self.rest[f'Robe{k}_{j}'] = M
        for bn, (piv, tip) in self.trink.items():
            M = np.eye(4); M[:3, :3] = RR.frame_from(tip - piv, np.array([0, -1.0, 0])); M[:3, 3] = piv
            self.rest[bn] = M
        self.pel_rest = pel

    def bones(self):
        """(name, parent, E-frame rest, length) for build_armature"""
        out = []
        for k in range(N_CHAIN):
            for j in range(N_SEG):
                n = f'Robe{k}_{j}'
                L = np.linalg.norm(self.J[k, j + 1] - self.J[k, j])
                out.append((n, 'Pelvis' if j == 0 else f'Robe{k}_{j - 1}', self.rest[n], L))
        for bn, (piv, tip) in self.trink.items():
            out.append((bn, 'Pelvis', self.rest[bn], np.linalg.norm(tip - piv)))
        return out

    def simulate(self, frames_W, fps=60, sub=4, stiff=55.0, damp=7.5, grav=60.0):
        """returns per frame {bone: E-frame world matrix}"""
        rig = self.rig; ip = rig.idx['Pelvis']
        P0inv = np.linalg.inv(self.pel_rest)
        # particles: robe joints 1..N_SEG per chain, trinket tips
        chains = [(k, j) for k in range(N_CHAIN) for j in range(1, N_SEG + 1)]
        def targets(Wp):
            T = Wp @ P0inv
            tj = {kj: (T @ np.append(self.J[kj[0], kj[1]], 1))[:3] for kj in chains}
            tt = {bn: (T @ np.append(tip, 1))[:3] for bn, (piv, tip) in self.trink.items()}
            pv = {k: (T @ np.append(self.J[k, 0], 1))[:3] for k in range(N_CHAIN)}
            pt = {bn: (T @ np.append(piv, 1))[:3] for bn, (piv, tip) in self.trink.items()}
            return T, tj, tt, pv, pt
        T, tj, tt, pv, pt = targets(frames_W[0][ip])
        pos = {**{kj: tj[kj].copy() for kj in chains}, **{bn: tt[bn].copy() for bn in self.trink}}
        vel = {key: np.zeros(3) for key in pos}
        seg = {kj: np.linalg.norm(self.J[kj[0], kj[1]] - self.J[kj[0], kj[1] - 1]) for kj in chains}
        tl = {bn: np.linalg.norm(tip - piv) for bn, (piv, tip) in self.trink.items()}
        dt = 1.0 / fps / sub
        out = []
        prevW = frames_W[0][ip]
        for fi, W in enumerate(frames_W):
            Wp = W[ip]
            for s in range(sub):
                a = (s + 1) / sub
                Wi = prevW.copy(); Wi[:3, 3] = prevW[:3, 3] * (1 - a) + Wp[:3, 3] * a
                Wi[:3, :3] = RR.quat2mat(RR.slerp(RR.mat2quat(prevW[:3, :3]), RR.mat2quat(Wp[:3, :3]), a))
                T, tj, tt, pv, pt = targets(Wi)
                for key in pos:
                    tgt = tj[key] if key in tj else tt[key]
                    acc = stiff * (tgt - pos[key]) - damp * vel[key] + np.array([0, 0, -grav]) * 0.08
                    vel[key] = vel[key] + acc * dt
                    pos[key] = pos[key] + vel[key] * dt
                # length constraints, root to tip
                for k in range(N_CHAIN):
                    prev = pv[k]
                    for j in range(1, N_SEG + 1):
                        d = pos[(k, j)] - prev; L = seg[(k, j)]
                        newp = prev + d / max(np.linalg.norm(d), 1e-9) * L
                        vel[(k, j)] += (newp - pos[(k, j)]) / dt * 0.5
                        pos[(k, j)] = newp; prev = newp
                for bn in self.trink:
                    d = pos[bn] - pt[bn]
                    newp = pt[bn] + d / max(np.linalg.norm(d), 1e-9) * tl[bn]
                    vel[bn] += (newp - pos[bn]) / dt * 0.5; pos[bn] = newp
            prevW = Wp
            # bone frames
            T = Wp @ P0inv
            fr = {}
            for k in range(N_CHAIN):
                prev = (T @ np.append(self.J[k, 0], 1))[:3]
                for j in range(N_SEG):
                    n = f'Robe{k}_{j}'
                    R0 = T[:3, :3] @ self.rest[n][:3, :3]
                    Rm = rot_between(R0[:, 0], pos[(k, j + 1)] - prev)
                    M = np.eye(4); M[:3, :3] = Rm @ R0; M[:3, 3] = prev
                    fr[n] = M; prev = pos[(k, j + 1)]
            for bn, (piv, tip) in self.trink.items():
                p = (T @ np.append(piv, 1))[:3]
                R0 = T[:3, :3] @ self.rest[bn][:3, :3]
                Rm = rot_between(R0[:, 0], pos[bn] - p)
                M = np.eye(4); M[:3, :3] = Rm @ R0; M[:3, 3] = p
                fr[bn] = M
            out.append(fr)
        return out
