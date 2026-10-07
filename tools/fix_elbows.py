"""Left-elbow correction and positional pop easing for the hands clips.

The hand keeps its exact world position and orientation.  The shoulder is kept behind the camera plane (the clips slide the clavicle up to
the lens to extend reach, which fills the screen edge with sleeve), the arm is re-solved as two-bone IK, and the elbow swivels about the
shoulder-wrist axis towards a natural pole (down and slightly out) when it splays further outboard than X_MIN.  Hand jumps of more than
JUMP_CM in one frame are eased over a short window of the whole arm chain.

python tools/fix_elbows.py [--dry OUTDIR] [--keys cpp38,9galo22] [--xmin -25]"""
import sys, os, argparse, copy
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import omf, skel, render
from build_sprint import mat_to_quat, key_from_mat, slerp, smoothstep
from fix_pops import ease_window, reattach_gun, chains

X_MIN = -22.0           # cm, left elbow may not go further out than this (semi-auto reloads reach -28 at most, idle sits at -18)
POLE = np.array([-0.25, -1.0, 0.10])
NEAR = 0.06             # m, the hud near plane; an arm segment crossing it inside the view fills the screen edge
MARGIN = 0.05           # m, arm radius plus a little
VFOV = 49.8
JUMP_CM = 5.0
SKIP = ('sprint', 'running', 'walk2run', 'run2walk', 'walktosprint', 'sprinttowalk')


def rot_between(a, b):
    """rotation matrix taking unit vector a onto unit vector b"""
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1 / (1 + c))


def gauss_smooth(w, sigma):
    n = int(3 * sigma); k = np.exp(-0.5 * (np.arange(-n, n + 1) / sigma) ** 2); k /= k.sum()
    return np.convolve(np.pad(w, n, mode='edge'), k, 'valid')


def fix_elbow(R, T, names, par, side='l', x_min=X_MIN, pole=POLE, moved=None, W_target=None, Rw_hand_target=None):
    """R, T (nb, n, ...) local keys -> corrected copies.  Returns (R, T, frames changed, worst elbow x before/after)"""
    ua, fa, hd = names.index(f'{side}_upperarm'), names.index(f'{side}_forearm'), names.index(f'{side}_hand')
    cl = par[ua]
    Rw, Pw = skel.fk(R, T, par)
    S, E, W = Pw[ua], Pw[fa], Pw[hd]
    n = R.shape[1]
    a = np.linalg.norm(E - S, axis=1); b = np.linalg.norm(W - E, axis=1)
    W_old = W
    if W_target is not None:
        W = W_target
    Rw_hand = Rw[hd] if Rw_hand_target is None else Rw_hand_target
    L = np.linalg.norm(W - S, axis=1)
    u = (W - S) / L[:, None]
    d1 = (a ** 2 - b ** 2 + L ** 2) / (2 * L)
    r = np.sqrt(np.maximum(a ** 2 - d1 ** 2, 0))
    C = S + u * d1[:, None]
    # current swivel vector (the old elbow projected onto the new circle plane) and the pole's projection onto it
    cur = E - C; cur = cur - u * np.sum(cur * u, axis=1)[:, None]
    pv = pole - u * (u @ pole)[:, None]
    pv_n = np.linalg.norm(pv, axis=1); ok = (pv_n > 1e-6) & (r > 1e-4)
    pv = np.where(ok[:, None], pv / np.where(pv_n > 1e-6, pv_n, 1)[:, None], cur / np.maximum(np.linalg.norm(cur, axis=1), 1e-9)[:, None])
    E_pole = C + pv * r[:, None]
    # weight: how far the elbow is outboard of the limit (cm) -> 0..1 over 6 cm, smoothed in time
    excess = np.maximum(x_min - E[:, 0] * 100, 0)
    w = np.clip(excess / 6.0, 0, 1)
    if w.max() <= 0 and (moved is None or not moved.any()):
        return R, T, 0, (E[:, 0].min() * 100, E[:, 0].min() * 100)
    w = gauss_smooth(w, 2.0)
    w[:2] = 0; w[-2:] = 0
    rebuild = w > 0
    if moved is not None:
        rebuild = rebuild | moved
    # blend along the circle (slerp of the swivel direction)
    cu = cur / np.maximum(np.linalg.norm(cur, axis=1), 1e-9)[:, None]
    dots = np.clip(np.sum(cu * pv, axis=1), -1, 1); ang = np.arccos(dots)
    E_new = C + cu * r[:, None]                        # old swivel direction on the (possibly new) circle
    for f in range(n):
        if w[f] <= 0 or ang[f] < 1e-4:
            continue
        axis = np.cross(cu[f], pv[f]); an = np.linalg.norm(axis)
        if an < 1e-9:
            continue
        axis /= an; th = ang[f] * w[f]
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        Rr = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
        E_new[f] = C[f] + Rr @ (cu[f] * r[f])
    # rebuild the two bones' world rotations, keep the hand's world rotation
    Rw2 = Rw.copy()
    for f in range(n):
        if not rebuild[f]:
            continue
        Rw2[ua, f] = rot_between(E[f] - S[f], E_new[f] - S[f]) @ Rw[ua, f]
        Rw2[fa, f] = rot_between(W_old[f] - E[f], W[f] - E_new[f]) @ Rw[fa, f]
    R2 = R.copy()
    for f in range(n):
        if not rebuild[f]:
            continue
        Rc = Rw[cl, f]
        R2[ua, f] = key_from_mat(Rc.T @ Rw2[ua, f])
        R2[fa, f] = key_from_mat(Rw2[ua, f].T @ Rw2[fa, f])
        R2[hd, f] = key_from_mat(Rw2[fa, f].T @ Rw_hand[f])
    for b_ in (ua, fa, hd):
        R2[b_] = omf.quat_continuous(R2[b_])
    return R2, T, int(rebuild.sum()), (E[:, 0].min() * 100, E_new[:, 0].min() * 100)


Z_MAX = -0.03           # m, the shoulder joint stays at least this far behind the camera plane


def rein_shoulder(R, T, names, par, side='l'):
    """pull the clavicle back on frames where the animator slid the shoulder up to the camera, never further than the arm can still reach the
    hand; the arm is re-solved by fix_elbow afterwards so the hand does not move.  Returns (T, max pull cm, frames)"""
    ua, fa, hd, cl = names.index(f'{side}_upperarm'), names.index(f'{side}_forearm'), names.index(f'{side}_hand'), names.index(f'{side}_clavicle')
    Rw, Pw = skel.fk(R, T, par)
    S, E, W = Pw[ua], Pw[fa], Pw[hd]
    reach = 0.97 * (np.linalg.norm(E - S, axis=1) + np.linalg.norm(W - E, axis=1))
    dz = np.minimum(Z_MAX - S[:, 2], 0)
    if dz.min() >= 0:
        return T, 0.0, 0
    def clamp(dz):                                       # keep the hand reachable, frame by frame
        dz = dz.copy()
        for f in range(len(dz)):
            for _ in range(40):
                if dz[f] >= 0 or np.linalg.norm(W[f] - (S[f] + [0, 0, dz[f]])) <= reach[f]:
                    break
                dz[f] *= 0.9
        return dz
    dz = clamp(gauss_smooth(dz, 1.5))
    dz = clamp(np.minimum(gauss_smooth(dz, 1.0), 0))
    dz[:2] = 0; dz[-2:] = 0
    T2 = T.copy(); Rb = Rw[par[cl]]
    for f in range(len(dz)):
        if dz[f] >= 0:
            continue
        T2[cl, f] = T[cl, f] + Rb[f].T @ np.array([0.0, 0.0, dz[f]])
    return T2, float(-dz.min() * 100), int((dz < -0.001).sum())


PITCH_START, PITCH_FULL, LOWER_M = 30.0, 60.0, 0.08


def lower_gun(R, T, names, par, idle_R, idle_T):
    """when the barrel is pitched up far beyond its idle angle (shells being ejected muzzle-up), lower the whole viewmodel so the cylinder
    sits at screen centre instead of the top edge.  Returns (T, max lowering cm, frames)"""
    lg = names.index('lead_gun'); root = names.index('bip01')
    Rw, Pw = skel.fk(R, T, par); Rwi, _ = skel.fk(idle_R[:, :1], idle_T[:, :1], par)
    fwd = Rw[lg] @ np.array([0.0, 0.0, 1.0]); fwd_i = Rwi[lg, 0] @ np.array([0.0, 0.0, 1.0])
    pitch = np.degrees(np.arcsin(np.clip(fwd[:, 1], -1, 1))) - np.degrees(np.arcsin(np.clip(fwd_i[1], -1, 1)))
    s_ = np.clip((pitch - PITCH_START) / (PITCH_FULL - PITCH_START), 0, 1)
    if s_.max() <= 0:
        return T, 0.0, 0
    s_ = smoothstep(gauss_smooth(s_, 2.0)); s_[:3] = 0; s_[-3:] = 0
    T2 = T.copy()
    T2[root] = T[root] - np.outer(s_, np.array([0.0, LOWER_M, 0.0]))
    return T2, float(s_.max() * LOWER_M * 100), int((s_ > 0.01).sum())


def positional_pops(R, T, names, par):
    """-> {side: [frame]} where a hand moves more than JUMP_CM in one frame and 2.5x faster than both neighbours"""
    Rw, Pw = skel.fk(R, T, par); out = {}
    for s in 'lr':
        P = Pw[names.index(s + '_hand')] * 100; v = np.linalg.norm(np.diff(P, axis=0), axis=1)
        out[s] = [f for f in range(1, len(v) - 1) if v[f] > JUMP_CM and v[f] > 2.5 * max(v[f - 1], v[f + 1], 0.5)]
    return out


def fix_file(path, out_path, keys_filter=None, x_min=X_MIN, report=True):
    o = omf.load(path); names = o.bone_names; par = skel.parents(names); ch = chains(names)
    idle = next(m for m in o.motions if m.name.endswith('_idle')); idle_R, idle_T = omf.motion_arrays(idle)
    changed = 0
    for m in o.motions:
        if m.length < 8 or any(k in m.name for k in SKIP):
            continue
        R, T = omf.motion_arrays(m)
        Rw0, Pw0 = skel.fk(R, T, par)
        pops = positional_pops(R, T, names, par)
        for s, frames in pops.items():
            for f in frames:
                a, b = max(2, f - 3), min(m.length - 3, f + 4)
                arm, fingers = ch[s]
                ease_window(R, T, [names.index(s + '_clavicle')] + arm + fingers, a, b)
                if s == 'r':
                    reattach_gun(R, T, names, par, a, b)
        Rw_ref, Pw_ref = skel.fk(R, T, par)
        lh_ = names.index('l_hand')
        Tm, drop, ndrop = rein_shoulder(R, T, names, par, 'l')
        moved = np.any(Tm != T, axis=(0, 2)) if ndrop else None
        T = Tm
        R, T, nf, (x0, x1) = fix_elbow(R, T, names, par, 'l', x_min, moved=moved, W_target=Pw_ref[lh_], Rw_hand_target=Rw_ref[lh_])
        lowered, nlow = 0.0, 0
        if 'reload' in m.name or 'unjam' in m.name:
            T, lowered, nlow = lower_gun(R, T, names, par, idle_R, idle_T)
        if nf or nlow or any(pops.values()):
            Rw1, Pw1 = skel.fk(R, T, par)
            lh = names.index('l_hand'); ua = names.index('l_upperarm'); fa = names.index('l_forearm')
            drift = np.linalg.norm((Pw1[lh] - Pw1[names.index('bip01')]) - (Pw_ref[lh] - Pw_ref[names.index('bip01')]), axis=1).max() * 1000
            z0 = Pw0[ua][:, 2].max() * 100; z1 = Pw1[ua][:, 2].max() * 100
            omf.set_motion_arrays(m, R, T); changed += 1
            if report:
                print(f"  {m.name:32s} rebuilt {nf:3d}f  shoulder pulled {drop:4.1f} cm on {ndrop:3d}f (z max {z0:5.1f} -> {z1:5.1f})  elbow x {x0:6.1f} -> {x1:6.1f}  lowered {lowered:4.1f} cm on {nlow:3d}f  pops {pops}  hand drift {drift:.3f} mm")
    omf.save(o, out_path)
    return changed


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--dry', default=None); ap.add_argument('--keys', default='cpp38,9galo22'); ap.add_argument('--xmin', type=float, default=X_MIN)
    a = ap.parse_args()
    for key in a.keys.split(','):
        src = os.path.join(render.HANDS_DIR, render.KEYS[key][0])
        out = os.path.join(a.dry, os.path.basename(src)) if a.dry else src
        print(key)
        print('  ->', fix_file(src, out, x_min=a.xmin), 'clips changed')
