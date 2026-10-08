"""Gunshots mixed from the Dark Signal pistol sounds (sources/darksignal) and the pack's own cleaned recordings
(sources/pocketpops via tools/pocketpops.py): a Dark Signal shot is the body (the boom, the action, the room), the cleaned
recording's crack is laid on top at the onset for the pop and the gun's own identity.  No compression, no limiting: levels
only, a soft clip against overs.  Distance layers and echoes are Dark Signal's own files, copied with their headers.

Per weapon:   very_close (npc, mono, one per take) + close (npc, mono, one per take) + close_distance (3) + medium_distance (3)
              + echo (6 of the 18 shared echoes)                                              -> [smallcal_<w>_snd_shoot]
              1p (actor, stereo, one per take) + echo                                          -> [smallcal_<w>_snd_shoot_actor]

python tools/mix_ds.py [--out DIR] [--keys a,b] [--wire] [--crack-db -3]"""
import sys, os, argparse, shutil, glob
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import oggx, pocketpops as pp, reference, build_sounds as bs

SR = 44100
DS = os.path.join(os.path.dirname(__file__), '..', 'sources', 'darksignal')
SOURCES = bs.SOURCES; SND = bs.SND; CFG = bs.CFG
ECHO_FOLDER = 'smallcal_echo'

# weapon -> Dark Signal donors: the stereo player shot (folder, takes), the mono npc set (folder), echoes (indices)
DONORS = {
    'pt22': dict(player=('stereo/../mono/vz61', [1, 2, 3]), npc='fiveseven', echo=[1, 4, 7, 10, 13, 16], body_db=-2.0),
    'trejo22': dict(player=('stereo/../mono/vz61', [4, 5, 6]), npc='fiveseven', echo=[2, 5, 8, 11, 14, 17], body_db=-2.0),
    '9galo22': dict(player=('stereo/../mono/vz61', [2, 4, 6]), npc='fiveseven', echo=[3, 6, 9, 12, 15, 18], body_db=-1.0),
    'pt25': dict(player=('stereo/pm', [1, 2, 3]), npc='fiveseven', echo=[1, 5, 9, 13, 17, 3], body_db=-1.0),
    'cvp1908': dict(player=('stereo/fort12', [1, 2, 3]), npc='fiveseven', echo=[2, 6, 10, 14, 18, 4], body_db=-1.0),
    'sav1907': dict(player=('stereo/../mono/vz61', [1, 3, 5, 2]), npc='fiveseven', echo=[3, 7, 11, 15, 1, 5], body_db=0.0),
    'rem51': dict(player=('stereo/pm', [1, 2, 3, 4, 1, 2]), npc='gsh18', echo=[4, 8, 12, 16, 2, 6], body_db=0.0),
    'cpp38': dict(player=('stereo/beretta', [1, 2, 3, 4]), npc='gsh18', echo=[5, 9, 13, 17, 3, 7], body_db=0.0),
}


def db(x):
    return 10 ** (x / 20)


def ds_path(rel):
    return os.path.normpath(os.path.join(DS, rel))


def load(path):
    x = oggx.decode(path).astype(np.float64)
    return x


def header(path):
    h = oggx.info(path)['xray']
    return (h['min'], h['max'], h['volume'], h['type'], h['ai'])


def shape_crack(c, hp=250.0, hold_s=0.025, tau_s=0.035):
    """the recording's crack: highs only (the body supplies the lows), gone within ~150 ms"""
    m = c if c.ndim == 1 else c.mean(1)
    m = pp.highpass(m, hp, 2)
    t = np.arange(len(m)) / SR
    return m * np.where(t > hold_s, np.exp(-(t - hold_s) / tau_s), 1.0)


def mix(body, crack, crack_db, body_db=0.0):
    """Dark Signal body at -1 dBFS (+body_db), the crack aligned at the body's onset at crack_db below the body's peak"""
    body = body / (np.abs(body).max() + 1e-9) * db(-1.0 + body_db)
    c = shape_crack(crack); c = c / (np.abs(c).max() + 1e-9) * db(-1.0 + body_db + crack_db)
    ib = reference.onset(body if body.ndim == 1 else body.mean(1)); ic = reference.onset(c)
    out = body.copy(); start = max(0, ib - ic); n = min(len(c), len(out) - start)
    if out.ndim == 1:
        out[start:start + n] += c[:n]
    else:
        out[start:start + n] += c[:n, None]
    return np.tanh(out / 0.97) * 0.97


def build_weapon(key, w, d, out_dir, profiles, crack_db, log=print):
    folder = os.path.join(out_dir, w['folder']); os.makedirs(folder, exist_ok=True)
    files = {}
    def put(kind, i, x, hdr):
        name = f"{key}_{kind}_{i}"; oggx.write_ogg(x.astype(np.float32), os.path.join(folder, name + '.ogg'), hdr)
        files.setdefault(kind, []).append(name)
    cracks = []
    for take in w['takes']:
        name, ratio = (take, 1.0) if isinstance(take, str) else take
        mono, st, info = pp.shot(os.path.join(SOURCES, name + '.ogg'), profiles)
        cracks.append(mono if ratio == 1.0 else bs.resample(mono, ratio))
    pfolder, ptakes = d['player']
    for i, c in enumerate(cracks):
        p = ds_path(f"{pfolder}/close_{ptakes[i % len(ptakes)]}.ogg")
        put('1p', i + 1, mix(load(p), c, crack_db, d['body_db']), header(p))
    npc = d['npc']
    for kind in ('very_close', 'close'):
        takes = sorted(glob.glob(ds_path(f"mono/{npc}/{kind}_[0-9].ogg")))
        for i, c in enumerate(cracks):
            p = takes[i % len(takes)]
            put(kind, i + 1, mix(load(p), c, crack_db, d['body_db']), header(p))
    for kind in ('close_distance', 'medium_distance'):
        takes = sorted(p for p in glob.glob(ds_path(f"mono/{npc}/*.ogg")) if os.path.basename(p).lower().startswith(kind))
        for i in range(3):
            src = takes[i % len(takes)]; name = f"{key}_{kind}_{i + 1}"
            shutil.copyfile(src, os.path.join(folder, name + '.ogg')); files.setdefault(kind, []).append(name)
    files['echo'] = [f"echo_{n}" for n in d['echo']]
    log(f"  {key}: {len(cracks)} takes; player body {pfolder.split('/')[-1]}, npc body {npc}, body {d['body_db']:+.0f} dB, crack {crack_db:+.0f} dB")
    return files


def install_echoes(out_dir):
    folder = os.path.join(out_dir, ECHO_FOLDER); os.makedirs(folder, exist_ok=True)
    for p in sorted(glob.glob(ds_path('echo/*.ogg'))):
        shutil.copyfile(p, os.path.join(folder, os.path.basename(p)))


def sections(key, w, files):
    p = f"weapons\\{w['folder']}\\"; e = f"weapons\\{ECHO_FOLDER}\\"
    def layer(n, names, prefix):
        return [f"snd_{n}_layer{'' if i == 0 else i} = {prefix}{nm}" for i, nm in enumerate(names)]
    L = layer(1, files['very_close'], p) + layer(2, files['close'], p) + layer(3, files['close_distance'], p) + layer(4, files['medium_distance'], p) + layer(5, files['echo'], e)
    A = layer(1, files['1p'], p) + layer(2, files['echo'], e)
    return (f"\n[smallcal_{key}_snd_shoot]\n" + '\n'.join(L) + f"\n\n[smallcal_{key}_snd_shoot_actor]\n" + '\n'.join(A) + '\n')


def clean_stale(key, w, out_dir, keep):
    folder = os.path.join(out_dir, w['folder']); gone = []
    for fn in sorted(os.listdir(folder)):
        stem = fn[:-4]
        if fn.endswith('.ogg') and stem.startswith(key + '_') and stem not in keep and not stem.endswith('_empty') and any(k in stem for k in ('_shot_', '_very_close_', '_close_', '_medium_distance_', '_1p_', '_mech_')):
            os.remove(os.path.join(folder, fn)); gone.append(fn)
    return gone


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=SND); ap.add_argument('--keys', default=','.join(DONORS)); ap.add_argument('--wire', action='store_true'); ap.add_argument('--crack-db', type=float, default=-3.0)
    a = ap.parse_args()
    profiles = pp.profiles(SOURCES)
    install_echoes(a.out)
    for key in a.keys.split(','):
        w = bs.WEAPONS[key]
        files = build_weapon(key, w, DONORS[key], a.out, profiles, a.crack_db)
        gone = clean_stale(key, w, a.out, {n for v in files.values() for n in v})
        print(key, {k: len(v) for k, v in files.items()}, f"removed {len(gone)} stale" if gone else '')
        if a.wire:
            path = os.path.join(CFG, f"w_{key}.ltx")
            with open(path, 'rb') as f:
                text = f.read().decode('cp1251')
            with open(path, 'wb') as f:
                f.write(bs.wire(key, w, files, text, sections_fn=sections).encode('cp1251'))
