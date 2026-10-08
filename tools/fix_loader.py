"""Revolver reload: hide the speedloader grab.  In both revolver reloads the left hand only drops to the bottom edge of the
frame while the speedloader teleports into it, so the loader is seen appearing in the fingers.  This dips the left arm
clearly below the frame around the grab (clavicle translation, eased in and out) and re-authors the speedloader bone of the
item animation so it rides in the hand from the grab until the animator's own hand-off, emerging with the hand.

python tools/fix_loader.py [--keys cpp38,9galo22] [--dip 0.12] [--dry]"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import omf, skel, render
from build_sprint import key_from_mat, smoothstep

HELD_CM = 13.0            # loader this close to the wrist counts as held
PRE, HOLD, POST = 6, 6, 5 # frames: dip ramps in over PRE before the grab, holds HOLD, ramps out over POST

WEAPONS = {
    'cpp38': dict(hands='cpp38_hands.omf', item='wpn_cpp38_hud.omf', hclip='hand_cpp38_reload', iclip='cpp38_reload'),
    '9galo22': dict(hands='wpn_9galo22_hands_anims.omf', item='wpn_9galo22_hud_animation.omf', hclip='hand_9galo22_reload', iclip='9galo22_reload'),
}


def item_world(a, m_item, f, Rw, Pw, names):
    """item bones in hands-world for one frame (anchored to lead_gun), plus the item-space pose and the anchor"""
    g = a['ogf']; R, T = omf.motion_arrays(m_item); ipar = g.parent_index()
    Riw, Piw = skel.fk(R[:, f:f + 1], T[:, f:f + 1], ipar); Riw = Riw[:, 0]; Piw = Piw[:, 0]
    lg = names.index('lead_gun'); A_R, A_t = Rw[lg], Pw[lg]
    M = A_R @ Riw[0].T
    Ww = np.einsum('ij,njk->nik', M, Riw); Pwld = A_t[None] + np.einsum('ij,nj->ni', M, Piw - Piw[0][None])
    return Ww, Pwld, Riw, Piw, (A_R, A_t)


def screen(a, P):
    (x, y), z = render.project(P + a['hpos'])
    return x, y


def fix(key, w, dip_m, dry):
    a = render.assets(key)
    hands = omf.load(os.path.join(render.HANDS_DIR, w['hands'])); item = omf.load(os.path.join(render.ITEM_DIR, w['item']))
    names = hands.bone_names; par = skel.parents(names)
    hm = hands.motion(w['hclip']); im = item.motion(w['iclip'])
    R, T = omf.motion_arrays(hm); n = min(hm.length, im.length)
    g = a['ogf']; ibones = [b.name for b in g.bones]; sl = ibones.index('speedloader'); ipar = g.parent_index()
    lh = names.index('l_hand'); clav = names.index('l_clavicle'); cpar = par[clav]
    Rw, Pw = skel.fk(R, T, par)
    # where is the loader relative to the hand, frame by frame
    Wl = np.zeros((n, 3, 3)); Pl = np.zeros((n, 3))
    for f in range(n):
        Ww, Pwld, _, _, _ = item_world(a, im, f, Rw[:, f], Pw[:, f], names)
        Wl[f], Pl[f] = Ww[sl], Pwld[sl]
    d = np.linalg.norm(Pl - Pw[lh, :n], axis=1) * 100
    held = d < HELD_CM
    runs = []; f = 0                                   # the longest stretch in the hand is the carry to the cylinder
    while f < n:
        if held[f]:
            g0 = f
            while f + 1 < n and held[f + 1]:
                f += 1
            runs.append((g0, f))
        f += 1
    h0, h1 = max(runs, key=lambda r: r[1] - r[0])
    off = np.median([Rw[lh, f].T @ (Pl[f] - Pw[lh, f]) for f in range(h0 + 3, h1 - 2)], axis=0)
    fm = min(range(h0 + 3, h1 - 2), key=lambda f: np.linalg.norm(Rw[lh, f].T @ (Pl[f] - Pw[lh, f]) - off))
    rel = Rw[lh, fm].T @ Wl[fm]
    # dip schedule
    s = np.zeros(n)
    for f in range(n):
        if h0 - PRE <= f < h0:
            s[f] = smoothstep((f - (h0 - PRE)) / PRE)
        elif h0 <= f <= h0 + HOLD:
            s[f] = 1.0
        elif h0 + HOLD < f <= h0 + HOLD + POST:
            s[f] = 1 - smoothstep((f - (h0 + HOLD)) / POST)
    dip = np.array([0.0, -dip_m, 0.0])
    T2 = T.copy()
    for f in range(n):
        if s[f] > 0:
            T2[clav, f] += Rw[cpar, f].T @ (dip * s[f])
    Rw2, Pw2 = skel.fk(R, T2, par)
    # the loader rides in the hand from the first frame where the dipped hand would carry it out of frame (before that it
    # keeps the animator's parked path, which is off screen) to the animator's hand-off
    Ri, Ti = omf.motion_arrays(im); Ri2, Ti2 = Ri.copy(), Ti.copy()
    f_attach = h0
    for f in range(h0 - PRE, h0 + 1):
        _, ly = screen(a, Pw2[lh, f] + Rw2[lh, f] @ off)
        if ly > 560:
            f_attach = f
            break
    for f in range(f_attach, h1 + 1):
        W_R = Rw2[lh, f] @ rel; W_P = Pw2[lh, f] + Rw2[lh, f] @ off
        _, _, Riw, Piw, (A_R, A_t) = item_world(a, im, f, Rw2[:, f], Pw2[:, f], names)
        Minv = Riw[0] @ A_R.T                                   # hands-world -> item space
        Riw_des = Minv @ W_R; Piw_des = Piw[0] + Minv @ (W_P - A_t)
        p = ipar[sl]
        Rp, Pp = (Riw[p], Piw[p]) if p >= 0 else (np.eye(3), np.zeros(3))
        Ri2[sl, f] = key_from_mat(Rp.T @ Riw_des); Ti2[sl, f] = Rp.T @ (Piw_des - Pp)
    # report
    print(f"{key}: loader held frames {h0}..{h1} (of {n}); offset in hand {np.round(off * 100, 1)} cm; dip {dip_m * 100:.0f} cm over frames {h0 - PRE}..{h0 + HOLD + POST}; loader attached from frame {f_attach}")
    Rw3, Pw3 = skel.fk(R, T2, par)
    im2 = omf.load(os.path.join(render.ITEM_DIR, w['item'])).motion(w['iclip']); omf.set_motion_arrays(im2, Ri2, Ti2)
    print('   f: hand y before -> after | loader y before -> after (screen px, frame bottom 540; * = visible)')
    for f in range(max(0, h0 - PRE - 3), min(n, h0 + HOLD + POST + 4)):
        _, hy0 = screen(a, Pw[lh, f]); _, hy1 = screen(a, Pw3[lh, f])
        Ww1, Pw1l, _, _, _ = item_world(a, im2, f, Rw3[:, f], Pw3[:, f], names)
        _, ly0 = screen(a, Pl[f]); lx1, ly1 = screen(a, Pw1l[sl])
        vis0 = '*' if 0 <= ly0 <= 540 else ' '; vis1 = '*' if (0 <= ly1 <= 540 and 0 <= lx1 <= 960) else ' '
        print(f"   {f:3d}: hand {hy0:6.0f} -> {hy1:6.0f} | loader {ly0:7.0f}{vis0} -> {ly1:7.0f}{vis1}  d {np.linalg.norm(Pw1l[sl] - Pw3[lh, f]) * 100:5.1f} cm  dip {s[f]:.2f}")
    if not dry:
        omf.set_motion_arrays(hm, R, T2); omf.save(hands, os.path.join(render.HANDS_DIR, w['hands']))
        omf.set_motion_arrays(im, Ri2, Ti2); omf.save(item, os.path.join(render.ITEM_DIR, w['item']))
        print('   written')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--keys', default=','.join(WEAPONS)); ap.add_argument('--dip', type=float, default=0.12); ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    for key in a.keys.split(','):
        fix(key, WEAPONS[key], a.dip, a.dry)
