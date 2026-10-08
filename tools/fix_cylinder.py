"""Revolver cylinders: turn one chamber on every shot.

The pack's shot clips key the cylinder (`cilindro`) as a one-frame flick: the Colt steps back 10 deg and then jumps
60 deg inside a single frame, starting the clip 50 deg off its idle pose, and the 9 Galo twitches 20 deg on a 40 deg
pitch.  In game that reads as a shiver, not a turn.  This re-keys only that bone in the shot clips: the clip starts one
chamber behind the idle pose (the same picture, the chambers being symmetric), turns through one pitch with an ease
over the frames right after the hammer drops, and ends exactly on the idle pose so the blend back to idle has nothing
left to move.  Direction follows the real guns: the Colt Police Positive turns clockwise as the shooter sees it, the
S&W-pattern 9 Galo counter-clockwise.

python tools/fix_cylinder.py [--keys cpp38,9galo22] [--dry DIR] [--sheet DIR]"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import omf, skel, render
from build_sprint import key_from_mat, smoothstep

BONE = 'cilindro'

WEAPONS = {  # clips: (first frame of the turn, frame it finishes on)
    'cpp38': dict(item='wpn_cpp38_hud.omf', idle='cpp38_idle', chambers=6, clockwise=True,
                  clips={'cpp38_shoot': (1, 6), 'cpp38_shoot_empty': (1, 6)}),
    '9galo22': dict(item='wpn_9galo22_hud_animation.omf', idle='9galo22_idle', chambers=9, clockwise=False,
                    clips={'9galo22_shoot': (1, 5)}),
}


def rz(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def spin_about_z(M0, M1):
    """signed angle (deg) of M1 relative to M0 about M0's own z axis"""
    D = M0.T @ M1
    return np.degrees(np.arctan2(D[1, 0], D[0, 0]))


def screen_sign(a, clip):
    """+1 if a positive turn about the cylinder's z moves its top to the right on screen (clockwise for the shooter), else -1"""
    hands = a['hands']; names = hands.bone_names; par = skel.parents(names)
    hm = hands.motion(a['hpre'] + 'shoot'); R, T = omf.motion_arrays(hm)
    Rw, Pw = skel.fk(R[:, :1], T[:, :1], par)
    Riw, Piw = render.item_pose(a, clip, 0, Rw[:, 0], Pw[:, 0], names, 'lead_gun')
    ci = [b.name for b in a['ogf'].bones].index(BONE)
    c = Piw[ci] + a['hpos']; top = np.array([0.0, 1.0, 0.0])
    (x0, _), _ = render.project(c + 0.01 * (Riw[ci] @ top))
    (x1, _), _ = render.project(c + 0.01 * (Riw[ci] @ (rz(np.radians(10)) @ top)))
    return 1 if x1 > x0 else -1


def rekey(key, w, dry, sheet):
    a = render.assets(key)
    src = os.path.join(render.ITEM_DIR, w['item'])
    item = omf.load(src)
    g = a['ogf']; ib = [b.name for b in g.bones]; ci = ib.index(BONE)
    idle = item.motion(w['idle']); Ri, _ = omf.motion_arrays(idle)
    base = skel.key_mat(Ri[ci, 0])
    pitch = 360.0 / w['chambers']
    sgn = screen_sign(a, next(iter(w['clips'])))
    direction = sgn if w['clockwise'] else -sgn
    print(f'{key}: {w["chambers"]} chambers, pitch {pitch:.1f} deg, {"clockwise" if w["clockwise"] else "counter-clockwise"} for the shooter '
          f'(local z sign {direction:+d})')
    before = {m.name: omf.motion_arrays(m) for m in item.motions}
    for clip, (f0, f1) in w['clips'].items():
        m = item.motion(clip); n = m.length
        R, _ = omf.motion_arrays(m)
        old = [spin_about_z(base, skel.key_mat(R[ci, f])) for f in range(n)]
        q = np.zeros((n, 4))
        for f in range(n):
            s = smoothstep(float(np.clip((f - f0) / (f1 - f0), 0.0, 1.0)))
            q[f] = key_from_mat(base @ rz(np.radians(direction * pitch * (s - 1.0))))
        omf.set_rot_f(m.bones[ci], omf.quat_continuous(q))
        R2, _ = omf.motion_arrays(m)
        new = [spin_about_z(base, skel.key_mat(R2[ci, f])) for f in range(n)]
        print(f'  {clip} ({n} frames): turn over frames {f0}-{f1} ({(f1 - f0) / 30 * 1000:.0f} ms)')
        print('    before: ' + ' '.join(f'{v:6.1f}' for v in old))
        print('    after:  ' + ' '.join(f'{v:6.1f}' for v in new))
    # nothing else may change
    for m in item.motions:
        R2, T2 = omf.motion_arrays(m); R1, T1 = before[m.name]
        for bi in range(len(ib)):
            if bi == ci and m.name in w['clips']:
                continue
            assert np.array_equal(R1[bi], R2[bi]) and np.array_equal(T1[bi], T2[bi]), (m.name, ib[bi])
    out = src if dry is None else os.path.join(dry, w['item'])
    omf.save(item, out)
    chk = omf.load(out)
    for m in chk.motions:
        R2, T2 = omf.motion_arrays(m); R1, T1 = before[m.name]
        for bi in range(len(ib)):
            if bi == ci and m.name in w['clips']:
                continue
            assert np.array_equal(R1[bi], R2[bi]) and np.array_equal(T1[bi], T2[bi]), ('after save', m.name, ib[bi])
    print(f'  wrote {out}')
    if sheet:
        render._cache.pop(key, None)
        if dry is not None:
            render._cache[key] = dict(a, item=chk)
        contact(key, w, sheet)


def contact(key, w, sheet):
    """zoomed frames of the shot around the cylinder, cartridge bones marked, as the hud camera sees them"""
    from PIL import Image, ImageDraw
    a = render.assets(key); clip = next(iter(w['clips'])); f0, f1 = w['clips'][clip]
    hands = a['hands']; names = hands.bone_names; par = skel.parents(names)
    hclip = a['hpre'] + 'shoot'; hm = hands.motion(hclip)
    g = a['ogf']; ib = [b.name for b in g.bones]; ci = ib.index(BONE); ipar = g.parent_index()
    kids = [i for i, p in enumerate(ipar) if p == ci and ib[i] != 'estrela']
    W0, H0 = render.W, render.H; render.W, render.H = W0 * 2, H0 * 2
    frames = list(range(0, f1 + 3))
    tiles = []
    try:
        for f in frames:
            img = render.render_frame(key, hclip, f, iclip=clip, label=' ')
            R, T = omf.motion_arrays(hm); ff = min(f, hm.length - 1)
            Rw, Pw = skel.fk(R[:, ff:ff + 1], T[:, ff:ff + 1], par); Rw = Rw[:, 0]; Pw = Pw[:, 0]
            Riw, Piw = render.item_pose(a, clip, f, Rw, Pw, names, 'lead_gun')
            d = ImageDraw.Draw(img)
            for j, k in enumerate(kids):
                (x, y), _ = render.project(Piw[k] + a['hpos'])
                col = (255, 60, 60) if j == 0 else (255, 230, 120)
                d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=col, outline=(0, 0, 0))
            (cx, cy), _ = render.project(Piw[ci] + a['hpos'])
            box = (int(cx) - 230, int(cy) - 170, int(cx) + 230, int(cy) + 170)
            t = img.crop(box)
            ImageDraw.Draw(t).text((8, 6), f'{key} {clip} frame {f}', fill=(240, 240, 240))
            tiles.append(t)
    finally:
        render.W, render.H = W0, H0
    cols = 4; tw, th = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    out = Image.new('RGB', (cols * tw, rows * th), (0, 0, 0))
    for i, t in enumerate(tiles):
        out.paste(t, ((i % cols) * tw, (i // cols) * th))
    os.makedirs(sheet, exist_ok=True)
    p = os.path.join(sheet, f'{key}_cylinder.png'); out.save(p); print(f'  sheet {p}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--keys', default=','.join(WEAPONS))
    ap.add_argument('--dry', default=None, help='write the OMFs here instead of into gamedata')
    ap.add_argument('--sheet', default=None, help='write a zoomed contact sheet per weapon here')
    args = ap.parse_args()
    if args.dry:
        os.makedirs(args.dry, exist_ok=True)
    for k in args.keys.split(','):
        rekey(k, WEAPONS[k], args.dry, args.sheet)
