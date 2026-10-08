"""Blood Reaper rig data + math (pure numpy).

Coordinates: model units (the Alembic's), recentred so the origin is on the ground under the pelvis at the bind
pose; Z up; the Reaper faces -Y (its left hand is +X).  Bone frames follow the original 3ds Max biped empties
("E" frames): X runs along the bone.  Blender bones use R = E @ Q (Y along the bone).
"""
import os, re, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, 'work')
BIND = 60
GROUND_GAP = 3.0           # the robe hem hovers this far above the ground at rest

Q = np.eye(4)
Q[:3, :3] = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], float)   # columns: bone X,Y,Z in E coordinates


def clean_name(raw):
    n = raw.replace('Base Human', '').replace('_trans', '')
    m = re.match(r'([LR])(Collarbone|Upperarm|Forearm|Palm)$', n)
    if m:
        return {'Collarbone': 'Clavicle', 'Upperarm': 'UpperArm', 'Forearm': 'Forearm', 'Palm': 'Hand'}[m.group(2)] + '_' + m.group(1)
    m = re.match(r'([LR])Finger(\d)(\d)$', n)
    if m:
        return f'Finger{m.group(2)}_{m.group(3)}_{m.group(1)}'
    if n == 'HeadBone001':
        return 'HeadTop'
    return n


def load():
    D = np.load(os.path.join(WORK, 'cache.npz'))
    raw = list(D['bone_names']); rawpar = list(D['bone_parents'])
    BM = D['bone_mats']
    # recentre: pelvis over the origin, ground under the hem
    pel = raw.index('Base HumanPelvis_trans')
    hem = min(float(D['V_' + m][BIND][:, 2].min()) for m in D['mesh_names'] if not m.startswith('Object001'))
    off = np.array([BM[BIND, pel, 0, 3], BM[BIND, pel, 1, 3], hem - GROUND_GAP])
    names = [clean_name(r) for r in raw]
    # wing chains: name by side + chain order
    for i, r in enumerate(raw):
        if r.startswith('Base HumanRibcageBone'):
            names[i] = None
    wing = [i for i, r in enumerate(raw) if r.startswith('Base HumanRibcageBone')]
    depth = {}
    def dep(i):
        p = rawpar[i]
        return 0 if not p.startswith('Base HumanRibcageBone') else 1 + dep(raw.index(p))
    counters = {}
    for i in sorted(wing, key=lambda i: (dep(i), BM[BIND, i, 1, 3])):
        side = 'L' if BM[BIND, i, 0, 3] > off[0] else 'R'
        key = (side, dep(i)); counters[key] = counters.get(key, 0) + 1
        names[i] = f'Wing{dep(i)}_{counters[key]}_{side}'
    parents = [names[raw.index(p)] if p in raw else '' for p in rawpar]
    E0 = BM.copy(); E0[..., :3, 3] -= off
    return dict(D=D, raw=raw, names=names, parents=parents, BM=E0, E=E0[BIND], off=off)


def rot(axis, ang):
    axis = np.asarray(axis, float); axis = axis / np.linalg.norm(axis)
    x, y, z = axis; c, s = np.cos(ang), np.sin(ang); C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def nrm(v):
    v = np.asarray(v, float); return v / max(np.linalg.norm(v), 1e-12)


def frame_from(x, h):
    """3x3 with columns (x, y, z) where x is exact and z ~ h (hinge)."""
    x = nrm(x); z = nrm(h - x * (h @ x)); y = np.cross(z, x)
    return np.stack([x, y, z], 1)


def swing_twist(Rl, axis):
    """split a local rotation into swing * twist about axis; returns twist angle (rad)."""
    from_q = mat2quat(Rl)
    p = from_q[1:] @ axis
    tw = np.array([from_q[0], *(p * axis)])
    n = np.linalg.norm(tw)
    if n < 1e-9: return 0.0
    tw /= n
    return 2 * np.arctan2(tw[1:] @ axis, tw[0])


def mat2quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1) * 2; w = 0.25 * s; x = (R[2, 1] - R[1, 2]) / s; y = (R[0, 2] - R[2, 0]) / s; z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2; w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s; y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2; w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s; y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2; w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s; y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    q = np.array([w, x, y, z]); return q / np.linalg.norm(q)


def quat2mat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def slerp(q0, q1, f):
    q0 = q0 / np.linalg.norm(q0); q1 = q1 / np.linalg.norm(q1)
    d = q0 @ q1
    if d < 0: q1 = -q1; d = -d
    if d > 0.9995:
        q = q0 + f * (q1 - q0); return q / np.linalg.norm(q)
    th = np.arccos(d); s = np.sin(th)
    return (np.sin((1 - f) * th) * q0 + np.sin(f * th) * q1) / s


class Rig:
    def __init__(self):
        L = load()
        self.__dict__.update(L)
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.order = []
        seen = set()
        def visit(i):
            if i in seen: return
            p = self.parents[i]
            if p: visit(self.idx[p])
            seen.add(i); self.order.append(i)
        for i in range(len(self.names)): visit(i)
        E = self.E
        self.L0 = np.zeros_like(E)
        for i, n in enumerate(self.names):
            p = self.parents[i]
            self.L0[i] = np.linalg.inv(E[self.idx[p]]) @ E[i] if p else E[i]

    def fk(self, local_rot=None, root=None, overrides=None):
        """local_rot: {name: 3x3 rotation in the bone's own E frame}, root: 4x4 pelvis world,
        overrides: {name: 4x4 world matrix} (used for IK'd bones; children follow)."""
        local_rot = local_rot or {}; overrides = overrides or {}
        W = np.zeros_like(self.E)
        for i in self.order:
            n = self.names[i]; p = self.parents[i]
            if n in overrides:
                W[i] = overrides[n]; continue
            M = self.L0[i].copy()
            if n in local_rot:
                M[:3, :3] = M[:3, :3] @ local_rot[n]
            if not p:
                W[i] = root if root is not None else M
                if root is not None and n in local_rot:
                    W[i] = W[i].copy(); W[i][:3, :3] = W[i][:3, :3] @ local_rot[n]
            else:
                W[i] = W[self.idx[p]] @ M
        return W

    def world_rot_local(self, name, Rw):
        """a rotation given in world axes (at the bind orientation of the bone) -> local E-frame rotation"""
        Eb = self.E[self.idx[name]][:3, :3]
        return Eb.T @ Rw @ Eb
