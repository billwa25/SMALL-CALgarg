"""Inventory + smoothness metrics of every hud clip.   python tools/anim_report.py [substring]"""
import sys, glob, os, re
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, omf

ROOT = os.path.join(os.path.dirname(__file__), '..', 'gamedata')


def qang(a, b):
    """angle (deg) between quaternions (…,4), stable for tiny angles: normalise, align hemispheres, 2*asin(|a-b|/2)"""
    a = a / np.linalg.norm(a, axis=-1, keepdims=True); b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    s = np.sign(np.sum(a * b, axis=-1, keepdims=True)); s[s == 0] = 1
    d = np.linalg.norm(a - b * s, axis=-1)
    return np.degrees(2 * np.arcsin(np.clip(d / 2, 0, 1)))


def report(path, filt=''):
    o = omf.load(path)
    names = o.bone_names
    print(f"\n=== {os.path.relpath(path, ROOT)}   bones={len(names)}")
    defs = {d.motion: d for d in o.defs}
    for mi, m in enumerate(o.motions):
        if filt and filt not in m.name:
            continue
        d = defs.get(mi)
        R, T = omf.motion_arrays(m)
        n = m.length
        kind = 'FX' if d and d.flags & 0x2 else 'cyc'
        stop = (d.flags & 0x1) if d else 0   # esmStopAtEnd
        # per frame rotation step per bone (deg) and translation step (cm)
        if n > 1:
            rs = qang(R[:, 1:], R[:, :-1])            # (nb, n-1)
            ts = np.linalg.norm(T[:, 1:] - T[:, :-1], axis=-1) * 100
            moving = (rs.max(axis=1) > 0.05) | (ts.max(axis=1) > 0.02)
            nb_mov = int(moving.sum())
            # held frames: all moving bones identical to previous frame
            same = (rs < 0.01) & (ts < 0.005)
            held = int(np.all(same[moving], axis=0).sum()) if nb_mov else 0
            # jitter: velocity sign flips of the rotation step magnitude's 2nd difference, use max 2nd diff / max 1st diff
            acc = np.abs(np.diff(rs, axis=1)) if n > 2 else np.zeros((len(names), 1))
            jit = float(acc.max()) if acc.size else 0.0
            step_max = float(rs.max()); step_mean = float(rs[moving].mean()) if nb_mov else 0.0
            tstep_max = float(ts.max())
            seam_r = float(qang(R[:, -1], R[:, 0]).max()); seam_t = float(np.linalg.norm(T[:, -1] - T[:, 0], axis=-1).max() * 100)
        else:
            nb_mov = 0; held = 0; jit = 0; step_max = step_mean = tstep_max = 0; seam_r = seam_t = 0
        spd = f"{d.speed:.2f}" if d else '-'
        print(f"  {m.name:34s} {n:4d}f {n/30:5.2f}s {kind} stop={stop} spd={spd} mov_bones={nb_mov:2d} held={held:3d} "
              f"rstep max={step_max:6.2f} mean={step_mean:5.2f} acc_max={jit:6.2f} tstep max={tstep_max:5.2f}cm "
              f"seam r={seam_r:6.2f}deg t={seam_t:5.2f}cm")
    return o


if __name__ == '__main__':
    filt = sys.argv[1] if len(sys.argv) > 1 else ''
    for p in sorted(glob.glob(os.path.join(ROOT, 'meshes/anomaly_weapons/hud_animation/*.omf'))) + sorted(glob.glob(os.path.join(ROOT, 'meshes/anomaly_weapons/hud_hands_animation/*.omf'))):
        report(p, filt)
