"""Layered gunshots for the small-calibre pack, built from the pack's own recordings (sources/pocketpops): every close and
first-person layer is one of the real shots, cleaned and shaped by tools/pocketpops.py and then matched to the measured
reference gunshots (tools/reference.py: pressure pulse from the shot itself, octave balance per time window, envelope,
density); action clicks are cut from the unjam recordings; the tails are echoes of the shot itself; the dry-fire clicks are
the recorded empties, cleaned.

Layers follow the reference pack's structure, each matched to the pack's measurement of that layer type for the weapon's
calibre class (tools/reference_target.json):
             very_close (15-40 m, one per take) + close (20-75 m, one per take) + close_distance (3) + medium_distance (2)
             + mech (semi-autos, 2)                                                            -> [smallcal_<w>_snd_shoot]
             1p (actor, stereo, one per take) + close + close_distance + medium_distance + mech -> [smallcal_<w>_snd_shoot_actor]
             <w>_empty                                                                         -> snd_empty
Every file: 44.1 kHz, X-Ray comment (min, max, volume, type, AI distance).

python tools/build_sounds.py [--out DIR] [--keys a,b] [--wire] [--handling]
    default --out writes into gamedata/sounds/weapons/<folder>/;  --wire rewrites the sound sections of the weapon configs;
    --handling also re-levels the draw/holster/reload/unjam/inspect files"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import oggx
import pocketpops as pp
import reference

SR = 44100
ROOT = os.path.join(os.path.dirname(__file__), '..', 'gamedata')
SND = os.path.join(ROOT, 'sounds', 'weapons')
CFG = os.path.join(ROOT, 'configs', 'items', 'weapons')
SOURCES = os.path.join(os.path.dirname(__file__), '..', 'sources', 'pocketpops')
MARK = '; layered gunshots (tools/build_sounds.py)'

# X-Ray comment values (min, max, volume, game type, AI distance), the ones the user's other packs use
CM = {
    'actor_shot': (5.0, 5.0, 2.0, 2149580800, 160.0),
    'actor_tail': (5.0, 5.0, 1.5, 134217856, 5.0),
    'npc_close': (20.0, 100.0, 2.0, 2149580800, 150.0),
    'npc_mech': (3.0, 35.0, 1.0, 2149580800, 20.0),
    'npc_far': (60.0, 600.0, 2.0, 2149580800, 150.0),
    'npc_tail': (20.0, 350.0, 1.5, 134217856, 75.0),
    'handling': (0.7, 20.0, 1.0, 2147745792, 2.0),
}

# weapon key -> folder, unjam recording (action clicks), calibre profile, semi-auto / full auto,
#               recorded shots used as takes (name or (name, pitch ratio) when a take is shared with another weapon),
#               recorded dry-fire click (name, pitch ratio)
WEAPONS = {
    'pt22': dict(folder='taurus_pts', unjam='pt25_unjam.ogg', cal='22lr', mech=True,
                 takes=['22lr-1', '22lr-2', '22lr-3'], empty=('25acp-empty', 1.05)),
    'pt25': dict(folder='taurus_pts', unjam='pt25_unjam.ogg', cal='25acp', mech=True,
                 takes=['25acp-1', '25acp-2', '25acp3'], empty=('25acp-empty', 1.0)),
    'cvp1908': dict(folder='cvp1908', unjam='cvp1908_unjam.ogg', cal='25acp', mech=True, seed=7,
                    takes=['25acp4', '25acp-5', ('25acp3', 0.97)], empty=('25acp-empty', 0.96)),
    'sav1907': dict(folder='sav1907', unjam='sav1907_unjam.ogg', cal='32acp', mech=True,
                    takes=['32-1', '32-2', '32-3', '32-4'], empty=('32-empty', 1.0)),
    'rem51': dict(folder='rem51', unjam='rem51_unjam.ogg', cal='380acp', mech=True,
                  takes=['380-1', '380-2', '380-3', '380-4', '380-5', '380-6'], empty=('32-empty2', 0.95)),
    'trejo22': dict(folder='trejo22', unjam='trejo22_unjam.ogg', cal='22lr', mech=True, auto=True, seed=3,
                    takes=['22lr-4', '22lr-5', '22lr-6'], empty=('32-empty2', 1.06)),
    '9galo22': dict(folder='9galo22', cal='22lr_rev', mech=False,
                    takes=['22lr-7', '22lr-8', '22lr'], empty=('38s-empty3', 1.12)),
    'cpp38': dict(folder='cpp38', cal='38spl', mech=False,
                  takes=['38s-1', '38s-2', '38s-3', '38s-4'], empty=('38s-empty3', 1.0)),
}

# calibre profile: echo T60 s, far low-pass Hz, action level, reference class ('small' = the pack's .32 ACP machine
# pistol, 'medium' = its 9x18 pistols), low-end pulse relative to that class (dB)
CAL = {
    '22lr': dict(t60=0.8, far_lp=1800, mech_level=0.30, cls='small', lf_db=-3.0),
    '22lr_rev': dict(t60=0.9, far_lp=1800, mech_level=0.0, cls='small', lf_db=-2.0),
    '25acp': dict(t60=0.9, far_lp=1800, mech_level=0.34, cls='small', lf_db=-1.0),
    '32acp': dict(t60=1.0, far_lp=1600, mech_level=0.36, cls='small', lf_db=0.0),
    '380acp': dict(t60=1.1, far_lp=1500, mech_level=0.38, cls='medium', lf_db=0.0),
    '38spl': dict(t60=1.2, far_lp=1400, mech_level=0.0, cls='medium', lf_db=0.5),
}
MECH_GAIN = 0.4           # the reference layers carry their own action noise, so the separate click layer sits well back


# ---------------------------------------------------------------------------------------------------------------------- dsp helpers

def db(x):
    return 10 ** (x / 20)


def fft_gain(x, gain_fn, pad_ms=100):
    """zero-phase filter, padded so the ringing does not wrap around the buffer"""
    p = int(pad_ms * SR / 1000); xp = np.concatenate([np.zeros(p), x, np.zeros(p)])
    n = len(xp); X = np.fft.rfft(xp); f = np.fft.rfftfreq(n, 1 / SR)
    return np.fft.irfft(X * gain_fn(f), n)[p:p + len(x)]


def lowpass(x, fc, order=4):
    return fft_gain(x, lambda f: 1 / np.sqrt(1 + (f / fc) ** (2 * order)))


def highpass(x, fc, order=2):
    return fft_gain(x, lambda f: 1 / np.sqrt(1 + (fc / np.maximum(f, 1e-3)) ** (2 * order)))


def bandpass(x, lo, hi, order=2):
    return lowpass(highpass(x, lo, order), hi, order)


def fade_out(x, ms):
    n = min(len(x), int(ms * SR / 1000)); w = np.ones(len(x)); w[-n:] = np.linspace(1, 0, n) ** 2
    return x * w if x.ndim == 1 else x * w[:, None]


def normalize(x, peak_db=-1.0):
    p = np.abs(x).max()
    return x * (db(peak_db) / p) if p > 0 else x


def soft_clip(x, ceiling=0.98):
    return np.tanh(x / ceiling) * ceiling


def place(dst, src, at_s, gain=1.0):
    i = int(at_s * SR); m = min(len(src), len(dst) - i)
    if m > 0:
        dst[i:i + m] += src[:m] * gain
    return dst


def convolve(x, h):
    n = len(x) + len(h) - 1; N = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(x, N) * np.fft.rfft(h, N), N)[:n]


def resample(x, ratio):
    """pitch / length change by plain interpolation (small ratios only); ratio > 1 is shorter and higher"""
    n = int(len(x) / ratio); idx = np.arange(n) * ratio
    if x.ndim == 1:
        return np.interp(idx, np.arange(len(x)), x)
    return np.stack([np.interp(idx, np.arange(len(x)), x[:, c]) for c in range(x.shape[1])], 1)


# ---------------------------------------------------------------------------------------------------------------------- layers
# Nothing here is a synthesised tone: the shots are the recordings, the action is cut from the unjam recordings, the echoes are
# the shot convolved with a sparse reflection pattern.

def extract_click(x, rng, length_ms=70, skip_ms=150):
    """the sharpest metallic transient of a handling recording (the slide being worked), cut out and cleaned"""
    hp = bandpass(x, 2000, 9000, 2)
    w = int(0.004 * SR); e = np.convolve(hp ** 2, np.ones(w) / w, 'same')
    e[:int(skip_ms * SR / 1000)] = 0
    d = np.diff(e, prepend=0); d[d < 0] = 0
    cands = np.argsort(d)[::-1]
    picks = []
    for i in cands:
        if all(abs(i - j) > int(0.2 * SR) for j in picks):
            picks.append(int(i))
        if len(picks) >= 3:
            break
    i = picks[rng.integers(len(picks))]
    i0 = max(0, i - int(0.0015 * SR)); n = int(length_ms * SR / 1000)
    c = x[i0:i0 + n].copy()
    if len(c) < n:
        c = np.concatenate([c, np.zeros(n - len(c))])
    c = highpass(c, 500, 2)
    t = np.arange(n) / SR
    c *= (1 - np.exp(-t / 0.0007)) * np.where(t > 0.03, np.exp(-(t - 0.03) / 0.012), 1.0)
    return c / (np.abs(c).max() + 1e-9)


def mech_layer(cal, rng, handling, auto=False):
    """the action cycling after the shot, from the weapon's own unjam recording"""
    n = int(0.3 * SR); y = np.zeros(n)
    c1 = extract_click(handling, rng); c2 = extract_click(handling, rng)
    place(y, c1, 0.036 if not auto else 0.02, 0.75)
    place(y, c2, (0.1 if not auto else 0.044) * rng.uniform(0.95, 1.05), 1.0)
    y = fade_out(y, 40)
    return normalize(y, -1.0) * cal['mech_level'] / 0.36


def echo_ir(t60, rng, n_early=14, length=None, stereo=False, lp_start=6000, lp_end=1500, dense=0.2):
    """gunshot echo: sparse, discrete reflections of decreasing brightness over a low dense bed"""
    length = length or int(t60 * 1.4 * SR); t = np.arange(length) / SR
    ch = 2 if stereo else 1
    out = np.zeros((length, ch))
    delays = np.sort(np.concatenate([rng.uniform(0.012, 0.09, n_early // 2), rng.uniform(0.09, t60 * 0.9, n_early - n_early // 2)]))
    for c in range(ch):
        y = np.zeros(length)
        for d in delays:
            i = int(d * SR * (1 + (rng.uniform(-0.03, 0.03) if stereo else 0)))
            w = int(rng.uniform(0.003, 0.009) * SR)
            if i + w >= length:
                continue
            frac = d / max(t60, 1e-3)
            burst = rng.standard_normal(w) * np.hanning(w)
            burst = lowpass(burst, lp_start * (lp_end / lp_start) ** min(frac, 1), 2)
            y[i:i + w] += burst / (np.abs(burst).max() + 1e-9) * np.exp(-6.91 * d / t60) * rng.uniform(0.6, 1.0)
        bed = rng.standard_normal(length) * np.exp(-6.91 * t / t60) * (1 - np.exp(-t / 0.02))
        bed = lowpass(bed, 2500, 2)
        y += dense * bed / (np.abs(bed).max() + 1e-9)
        out[:, c] = y
    out /= np.abs(out).max() + 1e-9
    return out[:, 0] if not stereo else out


def echo_material(shot, t60, rng, lo=200, hi=None, n_early=14, lp_start=6000, lp_end=1500, dense=0.2):
    """the shot's first 150 ms through a sparse echo response: the raw material of a distance layer"""
    ir = echo_ir(t60, rng, n_early=n_early, lp_start=lp_start, lp_end=lp_end, dense=dense)
    n = int(0.15 * SR); head = np.zeros(n); head[:min(n, len(shot))] = shot[:n]
    head = (bandpass(head, lo, hi, 2) if hi else highpass(head, lo, 2)) * np.linspace(1, 0, n) ** 0.7
    y = convolve(head, ir)[:int(t60 * 1.6 * SR)]
    direct = np.zeros(len(y)); direct[:n] = head                         # the direct sound, as the pack's layers keep it
    return fade_out(y + direct, 150)


def close_distance_layer(shot, cal, T, rng, auto=False):
    y = echo_material(shot, cal['t60'] * (0.7 if auto else 1.0) * rng.uniform(0.92, 1.08), rng)
    y, _ = reference.match(y, T, None)
    return y


def medium_distance_layer(shot, cal, T, rng, auto=False):
    y = echo_material(shot, cal['t60'] * 1.3 * (0.7 if auto else 1.0) * rng.uniform(0.92, 1.08), rng, lo=150, hi=cal['far_lp'], n_early=10, lp_start=2500, lp_end=600, dense=0.35)
    y, _ = reference.match(y, T, None)
    return y


def actor_shot(st, rng, width=0.5):
    """the player's own shot: the recording's two channels, narrowed (the recorder's channels barely correlate), plus a few
    early reflections"""
    mid = st.mean(1); side = (st[:, 0] - st[:, 1]) / 2
    L = mid + width * side; R = mid - width * side
    ir = echo_ir(0.3, rng, n_early=6, length=int(0.18 * SR), stereo=True, dense=0.08)
    head = highpass(mid[:int(0.03 * SR)], 300, 2)
    room = np.stack([convolve(head, ir[:, c]) for c in range(2)], 1)
    n = max(len(L), len(room)); y = np.zeros((n, 2)); y[:len(L), 0] = L; y[:len(R), 1] = R
    y[:len(room)] += room * 0.07
    return soft_clip(normalize(y, -1.0))


# ---------------------------------------------------------------------------------------------------------------------- build

def build_weapon(key, w, out_dir, profiles, log=print, targets=None):
    rng = np.random.default_rng(w.get('seed', 11) * 1000 + sum(map(ord, key)))
    cal = CAL[w['cal']]
    targets = targets or reference.load_target()
    T = targets[cal['cls']]
    folder = os.path.join(out_dir, w['folder']); os.makedirs(folder, exist_ok=True)
    auto = w.get('auto', False)
    files = {}
    def put(kind, i, x, cm):
        name = f"{key}_{kind}_{i}"
        oggx.write_ogg(x, os.path.join(folder, name + '.ogg'), cm)
        files.setdefault(kind, []).append(name)
    cleans, stereos = [], []
    for take in w['takes']:
        name, ratio = (take, 1.0) if isinstance(take, str) else take
        mono, st, info = pp.shot(os.path.join(SOURCES, name + '.ogg'), profiles)
        if ratio != 1.0:
            mono, st = resample(mono, ratio), resample(st, ratio)
        cleans.append(mono); stereos.append(st)
        log(f"  {key}: take {name:12s} {len(mono) / SR:.2f}s" + (f"  brass cut at {info['brass_cut_ms']:.0f} ms" if info['brass_cut_ms'] else ''))
    vcs = []
    for i, c in enumerate(cleans):
        y, inf = reference.match(c, T['very_close'], cal['lf_db']); vcs.append(y)
        put('very_close', i + 1, y, reference.HEADERS['very_close'])
        met = reference.metrics(y, T['very_close'])
        log(f"  {key}: very_close_{i + 1} crest {met['crest_db']:.1f} env {met['env_rms_err_db']:.1f} pulse x{inf['thump']:.2f} drive {inf['drive_db']:.1f}")
    for i, c in enumerate(cleans):
        y, _ = reference.match(resample(c, 1.0 + (0.015 if i % 2 else -0.015)), T['close'], cal['lf_db'])
        put('close', i + 1, y, reference.HEADERS['close'])
    for i in range(3):
        put('close_distance', i + 1, close_distance_layer(vcs[i % len(vcs)], cal, T['close_distance'], rng, auto), reference.HEADERS['close_distance'])
    for i in range(2):
        put('medium_distance', i + 1, medium_distance_layer(vcs[i % len(vcs)], cal, T['medium_distance'], rng, auto), reference.HEADERS['medium_distance'])
    if w['mech']:
        handling = oggx.decode(os.path.join(SND, w['folder'], w['unjam']))
        for i in range(2):
            put('mech', i + 1, mech_layer(cal, rng, handling, auto) * MECH_GAIN, CM['npc_mech'])
    for i, st in enumerate(stereos):
        y, _ = reference.match(actor_shot(st, rng), T['very_close'], cal['lf_db'])
        put('1p', i + 1, y, CM['actor_shot'])
    name, ratio = w['empty']
    click, info = pp.empty(os.path.join(SOURCES, name + '.ogg'), profiles)
    if ratio != 1.0:
        click = resample(click, ratio)
    oggx.write_ogg(click, os.path.join(folder, f"{key}_empty.ogg"), CM['handling'])
    files['empty'] = [f"{key}_empty"]
    return files


def clean_stale(key, w, out_dir):
    """remove this tool's earlier generated layers (<key>_shot_*.ogg) that nothing references any more"""
    folder = os.path.join(out_dir, w['folder']); gone = []
    for fn in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
        if fn.startswith(f"{key}_shot_") and fn.endswith('.ogg'):
            os.remove(os.path.join(folder, fn)); gone.append(fn)
    return gone


def sections(key, w, files):
    p = f"weapons\\{w['folder']}\\"
    def layer(n, names):
        return [f"snd_{n}_layer{'' if i == 0 else i} = {p}{nm}" for i, nm in enumerate(names)]
    def block(order):
        L = []; n = 1
        for kind in order:
            if kind in files:
                L += layer(n, files[kind]); n += 1
        return L
    L = block(['very_close', 'close', 'close_distance', 'medium_distance', 'mech'])
    A = block(['1p', 'close', 'close_distance', 'medium_distance', 'mech'])
    return (f"\n[smallcal_{key}_snd_shoot]\n" + '\n'.join(L) + f"\n\n[smallcal_{key}_snd_shoot_actor]\n" + '\n'.join(A) + '\n')


def wire(key, w, files, text):
    """the weapon's config: the layered sections after the marker rewritten, snd_empty added to every section that sets
    snd_shoot.  Lines keep their own terminators"""
    lines = text.splitlines(keepends=True)
    nl = '\r\n' if lines and lines[0].endswith('\r\n') else '\n'
    out = []
    empty_line = f"snd_empty = weapons\\{w['folder']}\\{key}_empty{nl}"
    section = None; has_empty = {}; shoots = {}
    for ln in lines:
        s = ln.strip()
        if s.startswith('['):
            section = s.split(']')[0][1:]
        elif section and s.split('=')[0].strip() == 'snd_empty':
            has_empty[section] = True
        elif section and s.split('=')[0].strip() == 'snd_shoot':
            shoots[section] = True
    section = None
    for ln in lines:
        if ln.strip() == MARK:
            break
        s = ln.strip()
        if s.startswith('['):
            section = s.split(']')[0][1:]
        out.append(ln)
        if section in shoots and not has_empty.get(section) and s.split('=')[0].strip() == 'snd_shoot_actor':
            sep = ln[len(ln.rstrip('\r\n')) - len(ln.rstrip('\r\n').lstrip()):]
            key_part = s.split('=')[0]
            out.append(f"snd_empty{key_part[len('snd_shoot_actor'):]}= weapons\\{w['folder']}\\{key}_empty{nl}")
            has_empty[section] = True
    while out and out[-1].strip() == '':
        out.pop()
    body = ''.join(out)
    if not body.endswith(nl):
        body += nl
    return body + nl + MARK + nl + sections(key, w, files).replace('\n', nl)


HANDLING_LEVELS = {'draw': -6.0, 'holster': -6.0, 'reload': -3.0, 'unjam': -3.0, 'inspect': -3.0}


def fix_handling(out_dir):
    done = []
    for folder in sorted(os.listdir(SND)):
        d = os.path.join(SND, folder)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith('.ogg') or 'shoot' in fn or '_shot_' in fn or fn.endswith('_empty.ogg'):
                continue
            kind = next((k for k in HANDLING_LEVELS if k in fn), None)
            if kind is None:
                continue
            x = oggx.decode(os.path.join(d, fn))
            if x.ndim > 1:
                x = x.mean(axis=1)
            x = pp.declip(x)
            x = normalize(x, HANDLING_LEVELS[kind])
            od = os.path.join(out_dir, folder); os.makedirs(od, exist_ok=True)
            oggx.write_ogg(x, os.path.join(od, fn), CM['handling'])
            done.append(f"{folder}/{fn}")
    return done


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=SND); ap.add_argument('--keys', default=','.join(WEAPONS))
    ap.add_argument('--wire', action='store_true'); ap.add_argument('--handling', action='store_true')
    a = ap.parse_args()
    profiles = pp.profiles(SOURCES); targets = reference.load_target()
    allsec = {}
    for key in a.keys.split(','):
        w = WEAPONS[key]
        files = build_weapon(key, w, a.out, profiles, targets=targets)
        allsec[key] = sections(key, w, files)
        gone = clean_stale(key, w, a.out)
        print(key, {k: len(v) for k, v in files.items()}, f"removed {len(gone)} stale files" if gone else '')
        if a.wire:
            path = os.path.join(CFG, f"w_{key}.ltx")
            with open(path, 'rb') as f:
                text = f.read().decode('cp1251')
            with open(path, 'wb') as f:
                f.write(wire(key, w, files, text).encode('cp1251'))
            print('  wired', os.path.relpath(path))
    with open(os.path.join(a.out, 'sections.ltx'), 'w') as f:
        f.write(''.join(allsec.values()))
    if a.handling:
        print('handling:', len(fix_handling(a.out)), 'files')
