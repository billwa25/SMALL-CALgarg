"""Gunshots: the pack's own recordings are the fire sounds, Dark Signal underneath them.  Each cleaned recording
(sources/pocketpops via tools/pocketpops.py: background noise gated out, onset trimmed, brass hits cut, the hall wash tamed)
is high-passed at 70 Hz against wind rumble and used at full level; a Dark Signal shot (sources/darksignal) is aligned to
its onset a few dB under it for the boom, the action and the room.  No compression, no limiting: levels only, a soft clip
against overs.  Distance layers and echoes are Dark Signal's own files, copied with their headers.

Per weapon:   very_close (npc, mono, one per take) + close (npc, mono, one per take) + close_distance (3) + medium_distance (3)
              + echo (6 of the 18 shared echoes)                                              -> [smallcal_<w>_snd_shoot]
              1p (actor, stereo, one per take) + echo                                          -> [smallcal_<w>_snd_shoot_actor]

python tools/mix_ds.py [--out DIR] [--keys a,b] [--wire] [--ds-db -4]"""
import sys, os, argparse, shutil, glob
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import oggx, pocketpops as pp, reference, build_sounds as bs

SR = 44100
DS = os.path.join(os.path.dirname(__file__), '..', 'sources', 'darksignal')
SOURCES = bs.SOURCES; SND = bs.SND; CFG = bs.CFG
ECHO_FOLDER = 'smallcal_echo'

# weapon -> Dark Signal donors: the stereo player shot (folder, takes), the mono npc set (folder), echoes (indices),
# and how far under the recording the Dark Signal shot sits (dB, added to --ds-db)
DONORS = {
    'pt22': dict(player=('stereo/../mono/vz61', [1, 2, 3]), npc='fiveseven', echo=[1, 4, 7, 10, 13, 16], ds_db=-2.0),
    'trejo22': dict(player=('stereo/../mono/vz61', [4, 5, 6]), npc='fiveseven', echo=[2, 5, 8, 11, 14, 17], ds_db=-2.0),
    '9galo22': dict(player=('stereo/../mono/vz61', [2, 4, 6]), npc='fiveseven', echo=[3, 6, 9, 12, 15, 18], ds_db=-1.0),
    'pt25': dict(player=('stereo/pm', [1, 2, 3]), npc='fiveseven', echo=[1, 5, 9, 13, 17, 3], ds_db=-1.0),
    'cvp1908': dict(player=('stereo/fort12', [1, 2, 3]), npc='fiveseven', echo=[2, 6, 10, 14, 18, 4], ds_db=-1.0),
    'sav1907': dict(player=('stereo/../mono/vz61', [1, 3, 5, 2]), npc='fiveseven', echo=[3, 7, 11, 15, 1, 5], ds_db=0.0),
    'rem51': dict(player=('stereo/pm', [1, 2, 3, 4, 1, 2]), npc='gsh18', echo=[4, 8, 12, 16, 2, 6], ds_db=0.0),
    'cpp38': dict(player=('stereo/beretta', [1, 2, 3, 4]), npc='gsh18', echo=[5, 9, 13, 17, 3, 7], ds_db=0.0),
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


def narrow(st, width=0.5):
    """the recorder's two channels barely correlate: keep half the side signal"""
    mid = st.mean(1); side = (st[:, 0] - st[:, 1]) / 2
    return np.stack([mid + width * side, mid - width * side], 1)


def mix(rec, ds, ds_db):
    """the recording at -1 dBFS, high-passed at 70 Hz (wind rumble; the Dark Signal shot supplies the lows), the Dark Signal
    shot aligned to its onset `ds_db` under it; the file runs as long as the longer of the two"""
    rec = np.stack([pp.highpass(rec[:, c], 70, 4) for c in range(2)], 1) if rec.ndim == 2 else pp.highpass(rec, 70, 4)
    rec = rec / (np.abs(rec).max() + 1e-9) * db(-1.0)
    ds = ds / (np.abs(ds).max() + 1e-9) * db(-1.0 + ds_db)
    if rec.ndim == 2 and ds.ndim == 1:
        ds = np.stack([ds, ds], 1)
    if rec.ndim == 1 and ds.ndim == 2:
        ds = ds.mean(1)
    ir = reference.onset(rec if rec.ndim == 1 else rec.mean(1)); ids = reference.onset(ds if ds.ndim == 1 else ds.mean(1))
    start = ir - ids                                       # where the Dark Signal file begins so that the onsets coincide
    n = max(len(rec), start + len(ds)); out = np.zeros((n,) if rec.ndim == 1 else (n, 2))
    out[:len(rec)] = rec
    a, b = max(0, start), start + len(ds)
    out[a:b] += ds[a - start:]
    return np.tanh(out / 0.97) * 0.97


def build_weapon(key, w, d, out_dir, profiles, ds_db, log=print):
    folder = os.path.join(out_dir, w['folder']); os.makedirs(folder, exist_ok=True)
    files = {}
    def put(kind, i, x, hdr):
        name = f"{key}_{kind}_{i}"; oggx.write_ogg(x.astype(np.float32), os.path.join(folder, name + '.ogg'), hdr)
        files.setdefault(kind, []).append(name)
    monos, stereos = [], []
    for take in w['takes']:
        name, ratio = (take, 1.0) if isinstance(take, str) else take
        mono, st, info = pp.shot(os.path.join(SOURCES, name + '.ogg'), profiles)
        if ratio != 1.0:
            mono, st = bs.resample(mono, ratio), bs.resample(st, ratio)
        monos.append(mono); stereos.append(narrow(st))
    level = ds_db + d['ds_db']
    pfolder, ptakes = d['player']
    for i, st in enumerate(stereos):
        p = ds_path(f"{pfolder}/close_{ptakes[i % len(ptakes)]}.ogg")
        put('1p', i + 1, mix(st, load(p), level), header(p))
    npc = d['npc']
    for kind in ('very_close', 'close'):
        takes = sorted(glob.glob(ds_path(f"mono/{npc}/{kind}_[0-9].ogg")))
        for i, m in enumerate(monos):
            p = takes[i % len(takes)]
            put(kind, i + 1, mix(m, load(p), level), header(p))
    for kind in ('close_distance', 'medium_distance'):
        takes = sorted(p for p in glob.glob(ds_path(f"mono/{npc}/*.ogg")) if os.path.basename(p).lower().startswith(kind))
        for i in range(3):
            src = takes[i % len(takes)]; name = f"{key}_{kind}_{i + 1}"
            shutil.copyfile(src, os.path.join(folder, name + '.ogg')); files.setdefault(kind, []).append(name)
    files['echo'] = [f"echo_{n}" for n in d['echo']]
    log(f"  {key}: {len(monos)} recordings as the fire sounds; Dark Signal {pfolder.split('/')[-1]} (player) / {npc} (npc) at {level:+.0f} dB under them")
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
    ap.add_argument('--out', default=SND); ap.add_argument('--keys', default=','.join(DONORS)); ap.add_argument('--wire', action='store_true'); ap.add_argument('--ds-db', type=float, default=-4.0)
    a = ap.parse_args()
    profiles = pp.profiles(SOURCES)
    install_echoes(a.out)
    for key in a.keys.split(','):
        w = bs.WEAPONS[key]
        files = build_weapon(key, w, DONORS[key], a.out, profiles, a.ds_db)
        gone = clean_stale(key, w, a.out, {n for v in files.values() for n in v})
        print(key, {k: len(v) for k, v in files.items()}, f"removed {len(gone)} stale" if gone else '')
        if a.wire:
            path = os.path.join(CFG, f"w_{key}.ltx")
            with open(path, 'rb') as f:
                text = f.read().decode('cp1251')
            with open(path, 'wb') as f:
                f.write(bs.wire(key, w, files, text, sections_fn=sections).encode('cp1251'))
