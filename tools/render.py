"""Offline first-person render of a hud frame: weapon mesh (OGF, skinned) + hand skeleton sticks, camera at the origin looking +Z.
python tools/render.py <key> <hands_clip>[:<frame>[,<frame>...]] [--item <clip>] [--out file.png] [--anchor identity|lead_gun]"""
import sys, os, glob, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import omf, ogf, skel

ROOT = os.path.join(os.path.dirname(__file__), '..', 'gamedata')
W, H = 960, 540
VFOV = 49.8
HANDS_DIR = os.path.join(ROOT, 'meshes/anomaly_weapons/hud_hands_animation')
ITEM_DIR = os.path.join(ROOT, 'meshes/anomaly_weapons/hud_animation')

KEYS = {  # key: (hands omf, item omf, hud ogf, hands_position m, hands prefix pattern, item prefix)
    'pt25': ('wpn_pt25_hand_animations.omf', 'wpn_pt25_hud_animation.omf', 'wpn_pt25/wpn_pt25_hud.ogf', (-0.022028, 0.03171, -0.072894), 'hand_pt25_', 'pt25_'),
    'pt22': ('wpn_pt25_hand_animations.omf', 'wpn_pt25_hud_animation.omf', 'wpn_pt22/wpn_pt22_hud.ogf', (-0.022028, 0.03171, -0.072894), 'hand_pt25_', 'pt25_'),
    'sav1907': ('wpn_hand_sav1907_hud_animation.omf', 'wpn_sav1907_hud_animation.omf', 'wpn_sav1907/wpn_sav1907_hud.ogf', (0, 0, -0.04), 'hand_sav1907_', 'sav1907_'),
    'rem51': ('wpn_hand_rem51_hud_animation.omf', 'wpn_rem51_hud_animation.omf', 'wpn_rem51/wpn_rem51_hud.ogf', (0, 0, -0.04), 'hand_rem51_', 'rem51_'),
    'trejo22': ('wpn_trejo22_hand_hud_animation.omf', 'wpn_trejo22_hud_animation.omf', 'wpn_trejo22/wpn_trejo22_hud.ogf', (0, 0, -0.025), 'hand_trejo22_', 'trejo22_'),
    'cvp1908': ('wpn_hand_cvp1908_hud_animation.omf', 'wpn_cvp1908_hud_animation.omf', 'wpn_cvp1908/wpn_cvp1908_hud.ogf', (0, 0, -0.05), 'cvp1908_hand_', 'cvp1908_'),
    'cpp38': ('cpp38_hands.omf', 'wpn_cpp38_hud.omf', 'wpn_cpp38/wpn_cpp38_hud.ogf', (0.025754, -0.007962, -0.035943), 'hand_cpp38_', 'cpp38_'),
    '9galo22': ('wpn_9galo22_hands_anims.omf', 'wpn_9galo22_hud_animation.omf', 'wpn_9galo22/wpn_9galo22_hud.ogf', (-0.004377, 0.005548, 0.006617), 'hand_9galo22_', '9galo22_'),
}

_cache = {}
HANDS_OVERRIDE = {}      # key -> alternative hands omf path (to look at a build before it is installed)


def assets(key):
    if key not in _cache:
        hf, itf, gf, hp, hpre, ipre = KEYS[key]
        hpath = HANDS_OVERRIDE.get(key, os.path.join(HANDS_DIR, hf))
        _cache[key] = dict(hands=omf.load(hpath), item=omf.load(os.path.join(ITEM_DIR, itf)), ogf=ogf.load(os.path.join(ROOT, 'meshes/anomaly_weapons', gf)), hpos=np.array(hp), hpre=hpre, ipre=ipre)
    return _cache[key]


def project(P):
    f = (H / 2) / np.tan(np.radians(VFOV / 2))
    z = np.maximum(P[..., 2], 1e-3)
    return np.stack([W / 2 + f * P[..., 0] / z, H / 2 - f * P[..., 1] / z], -1), z


def item_pose(a, item_motion, frame, hands_Rw, hands_Pw, hnames, anchor):
    """world matrices of the item bones for one frame"""
    g = a['ogf']; it = a['item']
    m = it.motion(item_motion); R, T = omf.motion_arrays(m)
    f = min(frame, m.length - 1)
    ipar = g.parent_index()
    Riw, Piw = skel.fk(R[:, f:f + 1], T[:, f:f + 1], ipar)
    Riw = Riw[:, 0]; Piw = Piw[:, 0]
    if anchor == 'lead_gun':
        lg = hnames.index('lead_gun')
        A_R, A_t = hands_Rw[lg], hands_Pw[lg]
        root_R, root_t = Riw[0], Piw[0]
        # put the item root at the anchor: world = anchor * root^-1 * bone
        Rinv = root_R.T
        Rw = np.einsum('ij,jk,nkl->nil', A_R, Rinv, Riw)
        Pw = A_t[None] + np.einsum('ij,nj->ni', A_R @ Rinv, Piw - root_t[None])
        return Rw, Pw
    return Riw, Piw


def skin(g, Rw, Pw):
    """skinned world positions + normals of every child mesh (list of (pos, nrm, tris))"""
    Rb, Pb = ogf.bind_world(g)
    nb = len(g.bones)
    # per-bone skin matrices: world * inv(bind)
    M = np.zeros((nb, 3, 3)); t = np.zeros((nb, 3))
    for i in range(nb):
        M[i] = Rw[i] @ Rb[i].T
        t[i] = Pw[i] - M[i] @ Pb[i]
    out = []
    for c in g.children:
        pos = np.zeros_like(c.pos); nrm = np.zeros_like(c.nrm)
        for k in range(4):
            w = c.weights[:, k]
            if not np.any(w): continue
            b = c.bones[:, k]
            pos += w[:, None] * (np.einsum('nij,nj->ni', M[b], c.pos) + t[b])
            nrm += w[:, None] * np.einsum('nij,nj->ni', M[b], c.nrm)
        out.append((pos, nrm, c.tris))
    return out


def draw_mesh(draw, meshes, offset, color=(180, 170, 150)):
    tris_all = []
    for pos, nrm, tris in meshes:
        P = pos + offset
        xy, z = project(P)
        n = nrm[tris].mean(axis=1); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
        light = np.array([0.3, 0.8, -0.5]); light /= np.linalg.norm(light)
        shade = 0.35 + 0.65 * np.clip(n @ light, 0, 1)
        depth = z[tris].mean(axis=1)
        tris_all.append((xy[tris], depth, shade))
    xy = np.concatenate([t[0] for t in tris_all]); depth = np.concatenate([t[1] for t in tris_all]); shade = np.concatenate([t[2] for t in tris_all])
    order = np.argsort(-depth)
    col = np.array(color)
    for i in order:
        c = tuple(int(v) for v in col * shade[i])
        draw.polygon([tuple(p) for p in xy[i]], fill=c)


ARM_W = {'clavicle': 0.06, 'upperarm': 0.075, 'forearm': 0.065, 'hand': 0.04}


def draw_hands(img, Rw, Pw, names, par, offset, side_colors=False):
    ov = Image.new('RGBA', img.size, (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    P = Pw + offset
    xy, z = project(P)
    f = (H / 2) / np.tan(np.radians(VFOV / 2))
    segs = []
    for b, p in enumerate(par):
        if p < 0 or names[b] == 'lead_gun' or 'clavicle' in names[b]: continue
        if 'twist' in names[b]: continue
        if names[p] in ('bip01',): continue
        kind = 'finger' if 'finger' in names[b] else next((k for k in ARM_W if names[b].endswith(k)), None)
        if kind is None: continue
        width_m = 0.018 if kind == 'finger' else ARM_W[kind]
        if kind == 'hand': width_m = 0.045
        depth = (z[b] + z[p]) / 2
        side = names[b][0] if not names[b].startswith('bip01') else names[b][6]
        segs.append((depth, b, p, width_m, side))
    segs.sort(key=lambda s: -s[0])
    NEAR = 0.06
    for depth, b, p, width_m, side in segs:
        A = P[p].copy(); B = P[b].copy()
        if A[2] < NEAR and B[2] < NEAR:
            continue
        if A[2] < NEAR:
            A = A + (B - A) * (NEAR - A[2]) / (B[2] - A[2])
        if B[2] < NEAR:
            B = B + (A - B) * (NEAR - B[2]) / (A[2] - B[2])
        xa, _ = project(A[None]); xb, _ = project(B[None]); xa = xa[0]; xb = xb[0]
        depth = (A[2] + B[2]) / 2
        w = max(1, int(f * width_m / depth))
        if side_colors:
            col = (220, 80, 80, 170) if side == 'l' else (80, 120, 230, 170)
        else:
            col = (205, 160, 130, 190)
        d.line([tuple(xa), tuple(xb)], fill=col, width=w)
        d.ellipse([xb[0] - w / 2, xb[1] - w / 2, xb[0] + w / 2, xb[1] + w / 2], fill=col)
    img.paste(Image.alpha_composite(img.convert('RGBA'), ov).convert('RGB'))


def render_frame(key, hclip, frame, iclip=None, anchor='lead_gun', side_colors=True, label=None):
    a = assets(key); h = a['hands']; names = h.bone_names; par = skel.parents(names)
    m = h.motion(hclip); R, T = omf.motion_arrays(m); f = min(frame, m.length - 1)
    Rw, Pw = skel.fk(R[:, f:f + 1], T[:, f:f + 1], par); Rw = Rw[:, 0]; Pw = Pw[:, 0]
    img = Image.new('RGB', (W, H), (40, 44, 50)); d = ImageDraw.Draw(img)
    # crosshair
    d.line([(W / 2 - 10, H / 2), (W / 2 + 10, H / 2)], fill=(90, 95, 100)); d.line([(W / 2, H / 2 - 10), (W / 2, H / 2 + 10)], fill=(90, 95, 100))
    if iclip is None:
        iclip = a['ipre'] + 'idle'
    Riw, Piw = item_pose(a, iclip, f, Rw, Pw, names, anchor)
    meshes = skin(a['ogf'], Riw, Piw)
    draw_mesh(d, meshes, a['hpos'])
    draw_hands(img, Rw, Pw, names, par, a['hpos'], side_colors)
    d = ImageDraw.Draw(img)
    d.text((8, 6), label or f"{key} {hclip} f{f}/{m.length}  item={iclip} anchor={anchor}", fill=(230, 230, 230))
    return img


def sheet(frames, cols=4, scale=0.5):
    w = int(W * scale); h = int(H * scale)
    rows = (len(frames) + cols - 1) // cols
    out = Image.new('RGB', (cols * w, rows * h), (0, 0, 0))
    for i, im in enumerate(frames):
        out.paste(im.resize((w, h)), ((i % cols) * w, (i // cols) * h))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('key'); ap.add_argument('clip'); ap.add_argument('--item', default=None); ap.add_argument('--out', default='frame.png'); ap.add_argument('--anchor', default='lead_gun'); ap.add_argument('--cols', type=int, default=4); ap.add_argument('--scale', type=float, default=0.5)
    args = ap.parse_args()
    clip, _, fr = args.clip.partition(':')
    frames = [int(x) for x in fr.split(',')] if fr else [0]
    ims = [render_frame(args.key, clip, f, args.item, args.anchor) for f in frames]
    (sheet(ims, args.cols, args.scale) if len(ims) > 1 else ims[0]).save(args.out)
    print('wrote', args.out)
