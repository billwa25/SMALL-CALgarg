"""Hand skeleton hierarchy + forward kinematics for the 42-bone Anomaly hud hands (parents inferred from the bone names)."""
import numpy as np


def parents(names):
    par = []
    for n in names:
        if n == 'bip01':
            p = None
        elif n in ('l_clavicle', 'r_clavicle', 'lead_gun'):
            p = 'bip01'
        elif n.endswith('_upperarm'):
            p = n[0] + '_clavicle'
        elif n.endswith('_forearm'):
            p = n[0] + '_upperarm'
        elif n.endswith('_forearm_twist'):
            p = n[0] + '_forearm'
        elif n.endswith('_hand'):
            p = n[0] + '_forearm'
        elif n.startswith('bip01_') and 'finger' in n:
            side = n[6]
            d = n.split('finger')[1]
            p = (side + '_hand') if len(d) == 1 else ('bip01_' + side + '_finger' + d[:-1])
        elif 'finger0' in n:
            side = n[0]
            d = n.split('finger0')[1]
            p = (side + '_hand') if d == '' else (side + '_finger0' + d[:-1])
        else:
            raise KeyError(n)
        par.append(names.index(p) if p is not None else -1)
    return par


def quat_to_mat(q):
    """(…,4) x,y,z,w -> (…,3,3)"""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    M = np.empty(q.shape[:-1] + (3, 3))
    M[..., 0, 0] = 1 - 2 * (y * y + z * z); M[..., 0, 1] = 2 * (x * y - z * w);     M[..., 0, 2] = 2 * (x * z + y * w)
    M[..., 1, 0] = 2 * (x * y + z * w);     M[..., 1, 1] = 1 - 2 * (x * x + z * z); M[..., 1, 2] = 2 * (y * z - x * w)
    M[..., 2, 0] = 2 * (x * z - y * w);     M[..., 2, 1] = 2 * (y * z + x * w);     M[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return M


CONJ = np.array([-1.0, -1.0, -1.0, 1.0])


def key_mat(R):
    """X-Ray key quaternion -> rotation matrix acting on column vectors (the stored quaternion is the inverse of the textbook one)"""
    return quat_to_mat(R * CONJ)


def fk(R, T, par):
    """R (nb,n,4) T (nb,n,3) local -> world rotation (nb,n,3,3) and position (nb,n,3).  child_world = parent_world * local."""
    nb, n = R.shape[:2]
    Rw = np.zeros((nb, n, 3, 3)); Pw = np.zeros((nb, n, 3))
    Rl = key_mat(R)
    for b in range(nb):
        p = par[b]
        if p < 0:
            Rw[b] = Rl[b]; Pw[b] = T[b]
        else:
            Rw[b] = Rw[p] @ Rl[b]
            Pw[b] = Pw[p] + np.einsum('nij,nj->ni', Rw[p], T[b])
    return Rw, Pw
