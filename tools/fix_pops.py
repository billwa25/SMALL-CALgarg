"""Remove single-frame pops of the arms: wherever an arm bone jumps more than ARM_THR degrees (or a finger more than FINGER_THR) between two
frames, that side's whole chain (upper arm .. finger tips) is re-interpolated with an eased blend over a short window around the jump.
The first and last two frames of a clip are never touched (they must keep matching the idle).

python tools/fix_pops.py [--dry OUTDIR] [--report]"""
import sys, os, argparse, copy, glob
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import omf, skel
from anim_report import qang
from build_sprint import slerp, smoothstep

ARM_THR = 15.0
FINGER_THR = 25.0
HALF = 3                 # frames on each side of the jump
ARM = ['upperarm', 'forearm', 'forearm_twist', 'hand']
SKIP = ('sprint', 'running', 'walk2run', 'run2walk', 'walktosprint', 'sprinttowalk', 'holster', 'draw')
HANDS_DIR = os.path.join(os.path.dirname(__file__), '..', 'gamedata', 'meshes', 'anomaly_weapons', 'hud_hands_animation')


def chains(names):
    out = {}
    for s in 'lr':
        arm = [names.index(s + '_' + b) for b in ARM]
        fingers = [i for i, n in enumerate(names) if 'finger' in n and (n.startswith(s + '_') or n.startswith('bip01_' + s + '_'))]
        out[s] = (arm, fingers)
    return out


def find_pops(R, names):
    """-> {side: [jump frame f (between f and f+1)]}"""
    st = qang(R[:, 1:], R[:, :-1])
    res = {}
    for s, (arm, fingers) in chains(names).items():
        hit = (st[arm] > ARM_THR).any(axis=0) | (st[fingers] > FINGER_THR).any(axis=0)
        res[s] = list(np.nonzero(hit)[0])
    return res


def windows(frames, n, half=HALF):
    """merge jump frames into [a, b] windows (inclusive) clamped away from the clip ends"""
    out = []
    for f in frames:
        a, b = max(2, f - half), min(n - 3, f + 1 + half)
        if b - a < 2:
            continue
        if out and a <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def ease_window(R, T, bones, a, b):
    """replace frames a..b (inclusive) of `bones` by an eased blend between the poses at a and b"""
    n = b - a + 1
    s = smoothstep(np.linspace(0, 1, n))[None, :, None]
    Ra = R[bones, a:a + 1]; Rb = R[bones, b:b + 1]
    R[bones, a:b + 1] = slerp(np.repeat(Ra, n, axis=1), np.repeat(Rb, n, axis=1), s)
    T[bones, a:b + 1] = T[bones, a:a + 1] * (1 - s) + T[bones, b:b + 1] * s


def gun_rel(R, T, names, par, f):
    """lead_gun expressed in r_hand's frame at frame f -> (3,3), (3,)"""
    Rw, Pw = skel.fk(R[:, f:f + 1], T[:, f:f + 1], par)
    rh = names.index('r_hand'); lg = names.index('lead_gun')
    return Rw[rh, 0].T @ Rw[lg, 0], Rw[rh, 0].T @ (Pw[lg, 0] - Pw[rh, 0])


def reattach_gun(R, T, names, par, a, b):
    """after the right chain was eased over a..b: keep the gun on the hand, its hand-relative offset eased between the offsets at a and b"""
    from build_sprint import key_from_mat, mat_to_quat
    Ra_, ta = gun_rel(R, T, names, par, a); Rb_, tb = gun_rel(R, T, names, par, b)
    qa = mat_to_quat(Ra_); qb = mat_to_quat(Rb_)
    n = b - a + 1
    s = smoothstep(np.linspace(0, 1, n))
    Rw, Pw = skel.fk(R[:, a:b + 1], T[:, a:b + 1], par)
    rh = names.index('r_hand'); lg = names.index('lead_gun'); b0 = par[lg]
    for i in range(n):
        q = slerp(qa[None], qb[None], s[i])[0]; Roff = skel.quat_to_mat(q); toff = ta * (1 - s[i]) + tb * s[i]
        Lw_R = Rw[rh, i] @ Roff; Lw_t = Pw[rh, i] + Rw[rh, i] @ toff
        if b0 >= 0:
            Lw_R = Rw[b0, i].T @ Lw_R; Lw_t = Rw[b0, i].T @ (Lw_t - Pw[b0, i])
        R[lg, a + i] = key_from_mat(Lw_R); T[lg, a + i] = Lw_t
    R[lg] = omf.quat_continuous(R[lg])


def fix_file(path, out_path=None, report=True):
    o = omf.load(path); names = o.bone_names
    ch = chains(names)
    changed = 0
    for m in o.motions:
        if m.length < 8 or any(k in m.name for k in SKIP):
            continue
        R, T = omf.motion_arrays(m)
        pops = find_pops(R, names)
        todo = []
        for s, frames in pops.items():
            for a, b in windows(frames, m.length):
                todo.append((s, a, b))
        if not todo:
            continue
        before = qang(R[:, 1:], R[:, :-1]).max()
        par = skel.parents(names)
        for s, a, b in todo:
            arm, fingers = ch[s]
            ease_window(R, T, arm + fingers, a, b)
            if s == 'r':
                reattach_gun(R, T, names, par, a, b)
        after = qang(R[:, 1:], R[:, :-1]).max()
        omf.set_motion_arrays(m, R, T)
        changed += 1
        if report:
            print(f"  {m.name:34s} windows {[(s, a, b) for s, a, b in todo]}  max step {before:5.1f} -> {after:5.1f} deg")
    omf.save(o, out_path or path)
    return changed


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--dry', default=None); ap.add_argument('--src', default=HANDS_DIR)
    args = ap.parse_args()
    for p in sorted(glob.glob(os.path.join(args.src, '*.omf'))):
        print(os.path.basename(p))
        out = os.path.join(args.dry, os.path.basename(p)) if args.dry else p
        n = fix_file(p, out)
        print(f"  -> {n} clips changed")
