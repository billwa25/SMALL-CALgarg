"""Rebuild the pistol sprints from the pack's own calm running cycle (hand_cpp38_running): the right arm carries the gun lowered and canted,
the left arm hangs out of view.  Each weapon keeps its own grip: its idle finger poses, and the gun stays rigid to the right hand with the
weapon's own hand-to-gun offset (lead_gun is the engine's attach bone).  Also writes eased idle<->sprint transitions.

python tools/build_sprint.py [--dry OUTDIR]        (default: writes the hands OMFs in place)"""
import sys, os, argparse, copy
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import omf, skel, render

ARM = ['clavicle', 'upperarm', 'forearm', 'forearm_twist', 'hand']
DONOR_KEY = 'cpp38'
DONOR_LOOP = 'hand_cpp38_running'
START_LEN = 16     # idle -> sprint, frames at 30 fps
END_LEN = 18       # sprint -> idle

# per weapon: (loop motion name to replace, start transition name, end transition name)
PLAN = {
    'pt25': ('hand_pt25_sprint', 'hand_pt25_sprint_start', 'hand_pt25_sprint_end'),
    'sav1907': ('hand_sav1907_sprint', 'hand_sav1907_sprint_start', 'hand_sav1907_sprint_end'),
    'rem51': ('hand_rem51_sprint', 'hand_rem51_sprint_start', 'hand_rem51_sprint_end'),
    'trejo22': ('hand_trejo22_sprint', 'hand_trejo22_sprint_start', 'hand_trejo22_sprint_end'),
    'cvp1908': ('cvp1908_hand_sprint', 'cvp1908_hand_walktosprint', 'cvp1908_hand_sprinttowalk'),
    '9galo22': ('hand_9galo22_sprint', 'hand_9galo22_sprint_start', 'hand_9galo22_sprint_end'),
    'cpp38': (None, 'hand_cpp38_walk2run', 'hand_cpp38_run2walk'),
}


def mat_to_quat(M):
    """rotation matrix (…,3,3) -> textbook quaternion (x,y,z,w)"""
    M = np.asarray(M)
    sh = M.shape[:-2]
    q = np.zeros(sh + (4,))
    t = np.trace(M, axis1=-2, axis2=-1)
    for idx in np.ndindex(*sh) if sh else [()]:
        m = M[idx]; tr = t[idx]
        if tr > 0:
            s = np.sqrt(tr + 1) * 2
            q[idx] = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            q[idx] = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            q[idx] = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
        else:
            s = np.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            q[idx] = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    return q


def key_from_mat(M):
    return mat_to_quat(M) * skel.CONJ


def slerp(a, b, s):
    """a, b (…,4), s scalar or (…,1)"""
    a = a / np.linalg.norm(a, axis=-1, keepdims=True); b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    d = np.sum(a * b, axis=-1, keepdims=True)
    b = np.where(d < 0, -b, b); d = np.abs(d).clip(0, 1)
    th = np.arccos(d)
    sin = np.sin(th)
    w0 = np.where(sin > 1e-6, np.sin((1 - s) * th) / np.where(sin > 1e-6, sin, 1), 1 - s)
    w1 = np.where(sin > 1e-6, np.sin(s * th) / np.where(sin > 1e-6, sin, 1), s)
    return w0 * a + w1 * b


def smoothstep(t):
    return t * t * (3 - 2 * t)


def gun_offset(R, T, names, par):
    """hand-to-gun rigid offset of frame 0: lead_gun in r_hand's frame"""
    Rw, Pw = skel.fk(R[:, :1], T[:, :1], par)
    rh = names.index('r_hand'); lg = names.index('lead_gun')
    Rr, Pr = Rw[rh, 0], Pw[rh, 0]
    return Rr.T @ Rw[lg, 0], Rr.T @ (Pw[lg, 0] - Pr)


def attach_gun(R, T, names, par, off):
    """overwrite lead_gun's local keys so it rides rigidly on r_hand with offset `off` (R_off, t_off)"""
    Rw, Pw = skel.fk(R, T, par)
    rh = names.index('r_hand'); lg = names.index('lead_gun'); b0 = par[lg]
    R_off, t_off = off
    Lw_R = Rw[rh] @ R_off
    Lw_t = Pw[rh] + np.einsum('nij,j->ni', Rw[rh], t_off)
    if b0 < 0:
        loc_R, loc_t = Lw_R, Lw_t
    else:
        Rb, Pb = Rw[b0], Pw[b0]
        loc_R = np.einsum('nji,njk->nik', Rb, Lw_R)
        loc_t = np.einsum('nji,nj->ni', Rb, Lw_t - Pb)
    R = R.copy(); T = T.copy()
    R[lg] = omf.quat_continuous(key_from_mat(loc_R)); T[lg] = loc_t
    return R, T


def bone_sets(names):
    left_arm = [names.index(s + '_' + b) for s in 'l' for b in ARM]
    right_arm = [names.index(s + '_' + b) for s in 'r' for b in ARM]
    lf = [i for i, n in enumerate(names) if 'finger' in n and (n.startswith('l_') or n.startswith('bip01_l_'))]
    rf = [i for i, n in enumerate(names) if 'finger' in n and (n.startswith('r_') or n.startswith('bip01_r_'))]
    return left_arm, right_arm, lf, rf


def build_loop(Rd, Td, Ri, Ti, names, par):
    """donor running (nb,n) + weapon idle frame 0 (nb,1) -> new loop (nb,n)"""
    n = Rd.shape[1]
    la, ra, lf, rf = bone_sets(names)
    R = np.tile(Ri[:, :1], (1, n, 1)); T = np.tile(Ti[:, :1], (1, n, 1))
    for b in la + ra + lf + [names.index('bip01')]:
        R[b] = Rd[b]; T[b] = Td[b]
    off = gun_offset(Ri, Ti, names, par)
    return attach_gun(R, T, names, par, off)


def build_transition(Ra, Ta, Rb, Tb, n, names, par, off):
    """eased blend from pose a (nb,1) to pose b (nb,1) over n frames, gun rigid to the right hand"""
    s = smoothstep(np.linspace(0, 1, n))[None, :, None]
    R = slerp(np.repeat(Ra, n, axis=1), np.repeat(Rb, n, axis=1), s)
    T = Ta * (1 - s) + Tb * s
    return attach_gun(R, T, names, par, off)


def rebuild(key, donor, dry=None):
    a = render.assets(key)
    o = copy.deepcopy(a['hands']); names = o.bone_names; par = skel.parents(names)
    loop_name, start_name, end_name = PLAN[key]
    idle = o.motion(a['hpre'] + 'idle'); Ri, Ti = omf.motion_arrays(idle)
    Rd, Td = omf.motion_arrays(donor)
    off = gun_offset(Ri, Ti, names, par)
    if loop_name:
        Rl, Tl = build_loop(Rd, Td, Ri, Ti, names, par)
        m = o.motion(loop_name); omf.set_motion_arrays(m, Rl, Tl)
        d = next(d for d in o.defs if d.name == loop_name); d.flags = 0; d.accrue = d.falloff = 2.0
    else:
        Rl, Tl = omf.motion_arrays(o.motion(DONOR_LOOP))
    la, ra, lf, rf = bone_sets(names)
    # loop frame 0 with the weapon's own right-hand fingers (already so in the rebuilt loop)
    Rs, Ts = build_transition(Ri[:, :1], Ti[:, :1], Rl[:, :1], Tl[:, :1], START_LEN, names, par, off)
    Re, Te = build_transition(Rl[:, :1], Tl[:, :1], Ri[:, :1], Ti[:, :1], END_LEN, names, par, off)
    tmpl = next(d.name for d in o.defs if d.name.endswith('draw') or d.name.endswith('_draw'))
    for nm, (R, T) in ((start_name, (Rs, Ts)), (end_name, (Re, Te))):
        if any(m.name == nm for m in o.motions):
            m = o.motion(nm); omf.set_motion_arrays(m, R, T)
            d = next(d for d in o.defs if d.name == nm); d.flags = 2; d.accrue = d.falloff = 5.0
        else:
            omf.add_motion(o, nm, tmpl, R, T, flags=2)
    path = os.path.join(render.HANDS_DIR, render.KEYS[key][0])
    if dry:
        path = os.path.join(dry, os.path.basename(path))
    omf.save(o, path)
    return path


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--dry', default=None); ap.add_argument('--keys', default=','.join(PLAN))
    args = ap.parse_args()
    donor = render.assets(DONOR_KEY)['hands'].motion(DONOR_LOOP)
    for key in args.keys.split(','):
        print(key, '->', rebuild(key, donor, args.dry))
