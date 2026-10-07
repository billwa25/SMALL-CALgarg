"""Layered gunshots for the small-calibre pack, built from the pack's own recordings plus synthesised body, crack, action and reverb layers.

Per weapon:  close (npc, mono, 3 takes) + mech (semi-autos, 2 takes) + tail (3 takes) + far (2 takes)   -> [smallcal_<w>_snd_shoot]
             1p (actor, stereo, 3 takes) + mech + 1p tail (stereo, 2 takes)                             -> [smallcal_<w>_snd_shoot_actor]
Every file: 44.1 kHz, X-Ray comment (min, max, volume, type, AI distance).  Handling sounds are declipped, levelled and given the header too.

python tools/build_sounds.py [--out DIR] [--keys a,b]       (default: writes into gamedata/sounds/weapons/<folder>/)"""
import sys, os, argparse, re
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import oggx

SR = 44100
ROOT = os.path.join(os.path.dirname(__file__), '..', 'gamedata')
SND = os.path.join(ROOT, 'sounds', 'weapons')

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

# weapon key -> folder, source shot file, calibre profile, semi-auto (has a cycling action), full auto
WEAPONS = {
    'pt22': dict(folder='taurus_pts', src='22lr_pistol_shoot.ogg', cal='22lr', mech=True),
    'pt25': dict(folder='taurus_pts', src='25acp_shoot.ogg', cal='25acp', mech=True),
    'cvp1908': dict(folder='cvp1908', src='25acp_shoot_striker.ogg', cal='25acp', mech=True, seed=7),
    'sav1907': dict(folder='sav1907', src='32acp_shoot.ogg', cal='32acp', mech=True),
    'rem51': dict(folder='rem51', src='380acp_shoot_striker.ogg', cal='380acp', mech=True),
    'trejo22': dict(folder='trejo22', src='22lr_pistol_shoot.ogg', cal='22lr', mech=True, auto=True, seed=3),
    '9galo22': dict(folder='9galo22', src='rev22lr_shoot.ogg', cal='22lr_rev', mech=False),
    'cpp38': dict(folder='cpp38', src='38spl_shoot.ogg', cal='38spl', mech=False),
}

# calibre profile: body thump (f start, f end, seconds) scaled so that 60-200 Hz sits `low` dB against 500-3000 Hz in the first 120 ms, crack (level, decay ms), reverb T60 s, mid dip dB, far low-pass Hz
CAL = {
    '22lr': dict(high=3.0, thump=(230, 120, 0.045), low=-13.0, crack=(0.55, 10), t60=0.9, dip=-5.0, far_lp=1800, mech_level=0.34),
    '22lr_rev': dict(high=2.5, thump=(210, 110, 0.05), low=-12.0, crack=(0.6, 11), t60=1.0, dip=-5.0, far_lp=1800, mech_level=0.0),
    '25acp': dict(high=2.5, thump=(210, 110, 0.05), low=-12.0, crack=(0.5, 11), t60=1.0, dip=-5.0, far_lp=1800, mech_level=0.38),
    '32acp': dict(high=2.0, thump=(180, 95, 0.06), low=-10.5, crack=(0.5, 12), t60=1.1, dip=-4.5, far_lp=1600, mech_level=0.40),
    '380acp': dict(high=1.5, thump=(160, 80, 0.07), low=-9.0, crack=(0.45, 13), t60=1.2, dip=-4.0, far_lp=1500, mech_level=0.42),
    '38spl': dict(high=0.0, thump=(140, 70, 0.09), low=-7.5, crack=(0.45, 14), t60=1.3, dip=-4.0, far_lp=1400, mech_level=0.0),
}


# ---------------------------------------------------------------------------------------------------------------------- dsp helpers

def db(x):
    return 10 ** (x / 20)


def fft_gain(x, gain_fn):
    n = len(x); X = np.fft.rfft(x); f = np.fft.rfftfreq(n, 1 / SR)
    return np.fft.irfft(X * gain_fn(f), n)


def lowpass(x, fc, order=4):
    return fft_gain(x, lambda f: 1 / np.sqrt(1 + (f / fc) ** (2 * order)))


def highpass(x, fc, order=2):
    return fft_gain(x, lambda f: 1 / np.sqrt(1 + (fc / np.maximum(f, 1e-3)) ** (2 * order)))


def bandpass(x, lo, hi, order=2):
    return lowpass(highpass(x, lo, order), hi, order)


def peaking(x, fc, gain_db, q=1.0):
    """bell EQ (zero phase): gain in dB around fc"""
    g = db(gain_db) - 1
    def fn(f):
        lf = np.log2(np.maximum(f, 1) / fc)
        return 1 + g * np.exp(-0.5 * (lf * q * 1.44) ** 2)
    return fft_gain(x, fn)


def shelf_high(x, fc, gain_db):
    g = db(gain_db) - 1
    return fft_gain(x, lambda f: 1 + g / (1 + (fc / np.maximum(f, 1e-3)) ** 2))


def declip(x, thr=0.97):
    """rebuild hard-clipped runs with a cubic through the samples around them"""
    x = x.astype(np.float64).copy()
    a = np.abs(x) >= thr
    i = 0; n = len(x)
    while i < n:
        if a[i]:
            j = i
            while j < n and a[j]:
                j += 1
            lo, hi = max(0, i - 3), min(n, j + 3)
            idx = np.r_[np.arange(lo, i), np.arange(j, hi)]
            if len(idx) >= 4 and (j - i) <= 400:
                c = np.polyfit(idx - i, x[idx], 3)
                rep = np.polyval(c, np.arange(i, j) - i)
                s = np.sign(x[i])
                rep = s * np.maximum(np.abs(rep), thr)
                x[i:j] = rep
            i = j
        else:
            i += 1
    return x


def trim_onset(x, pre_ms=1.5, thr_db=-30):
    env = np.abs(x); thr = env.max() * db(thr_db)
    i0 = int(np.argmax(env > thr)); i0 = max(0, i0 - int(pre_ms * SR / 1000))
    return x[i0:]


def fade_out(x, ms):
    n = min(len(x), int(ms * SR / 1000)); w = np.ones(len(x)); w[-n:] = np.linspace(1, 0, n) ** 2
    return x * w


def normalize(x, peak_db=-1.0):
    p = np.abs(x).max()
    return x * (db(peak_db) / p) if p > 0 else x


def soft_clip(x, ceiling=0.98):
    return np.tanh(x / ceiling) * ceiling


def thump(f0, f1, dur, rng):
    n = int(dur * SR * 2.5); t = np.arange(n) / SR
    k = np.log(f1 / f0) / dur
    phase = 2 * np.pi * f0 * (np.exp(k * np.minimum(t, dur * 2)) - 1) / k
    env = np.exp(-t / (dur * 0.6)) * (1 - np.exp(-t / 0.0012))
    y = np.sin(phase + rng.uniform(0, 0.4)) * env
    return y


def crack(decay_ms, rng, n=None):
    n = n or int(0.12 * SR); t = np.arange(n) / SR
    noise = rng.standard_normal(n)
    env = np.exp(-t / (decay_ms / 1000)) * (1 - np.exp(-t / 0.0003))
    return bandpass(noise * env, 1800, 9500, 2)


def metallic_click(rng, freqs, decay_ms, n=None):
    n = n or int(0.08 * SR); t = np.arange(n) / SR
    y = np.zeros(n)
    for f, d in zip(freqs, decay_ms):
        ph = rng.uniform(0, 2 * np.pi)
        y += np.sin(2 * np.pi * f * (1 + rng.uniform(-0.03, 0.03)) * t + ph) * np.exp(-t / (d / 1000))
    noise = rng.standard_normal(n) * np.exp(-t / 0.004)
    y = y / (np.abs(y).max() + 1e-9) + 0.6 * bandpass(noise, 1500, 9000) / (np.abs(noise).max() + 1e-9)
    return y * (1 - np.exp(-t / 0.0004))


def place(dst, src, at_s, gain=1.0):
    i = int(at_s * SR); m = min(len(src), len(dst) - i)
    if m > 0:
        dst[i:i + m] += src[:m] * gain
    return dst


def synth_ir(t60, rng, length=None, bright=1.0, stereo=False, early=True):
    """outdoor-ish impulse response: sparse early reflections + dense tail that darkens as it decays"""
    length = length or int(t60 * 1.6 * SR)
    t = np.arange(length) / SR
    ch = 2 if stereo else 1
    out = np.zeros((length, ch))
    for c in range(ch):
        noise = rng.standard_normal(length) * np.exp(-6.91 * t / t60)
        # darken over time: 6 overlapping segments with falling cutoff
        segs = 6; y = np.zeros(length)
        edges = np.linspace(0, length, segs + 1).astype(int)
        for s in range(segs):
            a, b = edges[s], edges[s + 1]
            w = np.zeros(length); w[a:b] = 1
            if s > 0: w[max(0, a - 2000):a] = np.linspace(0, 1, min(2000, a))[-(a - max(0, a - 2000)):]
            if s < segs - 1: w[b:min(length, b + 2000)] = np.linspace(1, 0, min(2000, length - b))
            fc = (6500 * bright) * (0.70 ** s) + 900
            y += lowpass(noise * w, fc, 2)
        y *= (1 - np.exp(-t / 0.004))
        if early:
            for d, g, fc in ((0.023, 0.5, 5000), (0.041, 0.4, 4000), (0.068, 0.35, 3500), (0.11, 0.3, 3000), (0.17, 0.25, 2500), (0.26, 0.2, 2000)):
                d2 = d * rng.uniform(0.9, 1.1); i = int(d2 * SR)
                if i + 400 < length:
                    tap = lowpass(rng.standard_normal(400) * np.exp(-np.arange(400) / 90), fc, 2)
                    y[i:i + 400] += tap * g * (0.6 + 0.4 * rng.random())
        out[:, c] = y
    if stereo:
        m = out.mean(axis=1, keepdims=True); out = 0.55 * m + 0.45 * out      # partly correlated channels
    out /= np.abs(out).max() + 1e-9
    return out[:, 0] if not stereo else out


def convolve(x, h):
    n = len(x) + len(h) - 1; N = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(x, N) * np.fft.rfft(h, N), N)[:n]


def resample(x, ratio):
    """pitch / length change by plain interpolation (small ratios only)"""
    n = int(len(x) / ratio); idx = np.arange(n) * ratio
    return np.interp(idx, np.arange(len(x)), x)


# ---------------------------------------------------------------------------------------------------------------------- layers

def close_shot(src, cal, rng, variant):
    """the recording, declipped, trimmed, re-balanced, with synthesised body and crack"""
    x = declip(src)
    x = trim_onset(x)
    x = highpass(x, 45, 2)
    x = peaking(x, 1000, cal['dip'], q=0.9)
    x = peaking(x, 300, -1.5, q=1.0)
    x = shelf_high(x, 3500, 3.5)
    t = np.arange(len(x)) / SR
    x = x * (1 + 0.5 * np.exp(-t / 0.012))                                   # attack emphasis
    x = x * np.where(t > 0.04, np.exp(-(t - 0.04) / 0.16), 1.0)              # tighter decay
    x = resample(x, 1 + rng.uniform(-0.025, 0.025)) if variant else x
    n = max(len(x), int(0.9 * SR)); y = np.zeros(n); y[:len(x)] = x
    y = y / (np.abs(y).max() + 1e-9)
    cr = crack(cal['crack'][1], rng)
    lvl = cal['crack'][0]
    for _ in range(6):                       # scale the crack so that 3-10 kHz sits `high` dB against 500-3000 Hz in the first 30 ms
        z = place(y.copy(), cr, 0.0, lvl)
        lvl *= db((cal['high'] - high_mid_db(z)) * 0.7)
        lvl = float(np.clip(lvl, 0.0, 1.2))
    y = place(y, cr, 0.0, lvl)
    f0, f1, dur = cal['thump']
    th = thump(f0 * rng.uniform(0.95, 1.05), f1, dur, rng)
    lvl = 0.3
    for _ in range(6):                       # scale the body so that low (60-200 Hz) sits `low` dB against mid (500-3000 Hz) over the first 120 ms
        z = place(y.copy(), th, 0.0005, lvl)
        r = low_mid_db(z)
        lvl *= db((cal['low'] - r) * 0.7)
        lvl = float(np.clip(lvl, 0.03, 1.5))
    y = place(y, th, 0.0005, lvl)
    y = fade_out(y, 60)
    return soft_clip(normalize(y, -1.0))


def band_rms(x, lo, hi):
    X = np.abs(np.fft.rfft(x)) ** 2; f = np.fft.rfftfreq(len(x), 1 / SR)
    return float(np.sqrt(X[(f >= lo) & (f < hi)].sum() / len(x) + 1e-18))


def low_mid_db(x, ms=120):
    h = x[:int(ms * SR / 1000)]
    return 20 * np.log10(band_rms(h, 60, 200) / band_rms(h, 500, 3000))


def high_mid_db(x, ms=30):
    h = x[:int(ms * SR / 1000)]
    return 20 * np.log10(band_rms(h, 3000, 10000) / band_rms(h, 500, 3000))


def mech_layer(cal, rng, auto=False):
    """the action cycling after the shot: slide back, slide forward"""
    n = int(0.32 * SR); y = np.zeros(n)
    back = metallic_click(rng, (2600, 4100, 6300), (9, 7, 5))
    fwd = metallic_click(rng, (1900, 3300, 5200), (14, 10, 6))
    t_back = 0.038 if not auto else 0.022
    t_fwd = (0.105 if not auto else 0.046) * rng.uniform(0.95, 1.05)
    place(y, back, t_back, 0.7); place(y, fwd, t_fwd, 1.0)
    thud = lowpass(rng.standard_normal(int(0.03 * SR)) * np.exp(-np.arange(int(0.03 * SR)) / (0.006 * SR)), 400, 2)
    place(y, thud / (np.abs(thud).max() + 1e-9), t_fwd, 0.5)
    y = fade_out(y, 40)
    return normalize(y, -1.0) * cal['mech_level'] / 0.36


def tail_layer(close, cal, rng, stereo=False, auto=False):
    t60 = cal['t60'] * (0.65 if auto else 1.0) * rng.uniform(0.92, 1.08)
    ir = synth_ir(t60, rng, stereo=stereo, bright=1.0)
    head = highpass(close[:int(0.06 * SR)], 300, 2) * np.linspace(1, 0, int(0.06 * SR)) ** 0.5
    if stereo:
        y = np.stack([convolve(head, ir[:, c]) for c in range(2)], 1)
    else:
        y = convolve(head, ir)
    y = highpass(y, 180, 2) if not stereo else np.stack([highpass(y[:, c], 180, 2) for c in range(2)], 1)
    y = y[:int((t60 * 1.5) * SR)]
    y = fade_out(y, 120) if not stereo else np.stack([fade_out(y[:, c], 120) for c in range(2)], 1)
    return soft_clip(normalize(y, -9.0))


def far_layer(close, cal, rng, auto=False):
    """distant report: low-passed pop with a long dark tail and a slapback or two"""
    t60 = cal['t60'] * 1.1 * (0.6 if auto else 1.0)
    ir = synth_ir(t60, rng, bright=0.5, early=False)
    pop = bandpass(close[:int(0.08 * SR)], 150, cal['far_lp'], 3)
    y = convolve(pop, ir)
    for d, g in ((0.19, 0.3), (0.33, 0.18)):
        y = place(y, lowpass(pop, cal['far_lp'] * 0.7, 3), d * rng.uniform(0.9, 1.1), g)
    y = y[:int(t60 * 1.3 * SR)]
    y = fade_out(y, 150)
    return soft_clip(normalize(y, -17.0))


def actor_shot(close, rng):
    """the player's own shot: stereo width from a small per-channel tilt and a touch of early room (no inter-channel delay, so it sums to mono cleanly)"""
    L = peaking(close, 2500, 1.0, 1.0); R = peaking(close, 2500, -1.0, 1.0)
    L = peaking(L, 500, -0.8, 1.0); R = peaking(R, 500, 0.8, 1.0)
    ir = synth_ir(0.25, rng, length=int(0.16 * SR), stereo=True, bright=1.0)
    head = close[:int(0.03 * SR)]
    room = np.stack([convolve(head, ir[:, c]) for c in range(2)], 1)
    n = max(len(L), len(room)); y = np.zeros((n, 2)); y[:len(L), 0] = L; y[:len(R), 1] = R
    room = np.stack([highpass(room[:, c], 250, 2) for c in range(2)], 1)
    y[:len(room)] += room * 0.05
    return soft_clip(normalize(y, -1.0))


# ---------------------------------------------------------------------------------------------------------------------- build

def build_weapon(key, w, out_dir):
    rng = np.random.default_rng(w.get('seed', 11) * 1000 + sum(map(ord, key)))
    cal = CAL[w['cal']]
    src = oggx.decode(os.path.join(SND, w['folder'], w['src']))
    folder = os.path.join(out_dir, w['folder']); os.makedirs(folder, exist_ok=True)
    auto = w.get('auto', False)
    files = {}
    def put(kind, i, x, cm):
        name = f"{key}_shot_{kind}_{i}"
        oggx.write_ogg(x, os.path.join(folder, name + '.ogg'), CM[cm])
        files.setdefault(kind, []).append(name)
    closes = []
    for i in range(3):
        c = close_shot(src, cal, rng, i); closes.append(c); put('close', i + 1, c, 'npc_close')
    if w['mech']:
        for i in range(2):
            put('mech', i + 1, mech_layer(cal, rng, auto), 'npc_mech')
    for i in range(3):
        put('tail', i + 1, tail_layer(closes[i], cal, rng, auto=auto), 'npc_tail')
    for i in range(2):
        put('far', i + 1, far_layer(closes[i], cal, rng, auto=auto), 'npc_far')
    for i in range(3):
        put('1p', i + 1, actor_shot(closes[i], rng), 'actor_shot')
    for i in range(2):
        put('1p_tail', i + 1, tail_layer(closes[i], cal, rng, stereo=True, auto=auto), 'actor_tail')
    return files


def sections(key, w, files):
    p = f"weapons\\{w['folder']}\\"
    def layer(n, names, extra=''):
        return [f"snd_{n}_layer{'' if i == 0 else i} = {p}{nm}{extra}" for i, nm in enumerate(names)]
    L = []; n = 1
    L += layer(n, files['close']); n += 1
    if 'mech' in files:
        L += layer(n, files['mech']); n += 1
    L += layer(n, files['tail']); n += 1
    L += layer(n, files['far']); n += 1
    A = []; n = 1
    A += layer(n, files['1p']); n += 1
    if 'mech' in files:
        A += layer(n, files['mech']); n += 1
    A += layer(n, files['1p_tail']); n += 1
    return (f"\n[smallcal_{key}_snd_shoot]\n" + '\n'.join(L) + f"\n\n[smallcal_{key}_snd_shoot_actor]\n" + '\n'.join(A) + '\n')


HANDLING_LEVELS = {'draw': -6.0, 'holster': -6.0, 'reload': -3.0, 'unjam': -3.0, 'inspect': -3.0}


def fix_handling(out_dir):
    done = []
    for folder in sorted(os.listdir(SND)):
        d = os.path.join(SND, folder)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith('.ogg') or 'shoot' in fn or '_shot_' in fn:
                continue
            kind = next((k for k in HANDLING_LEVELS if k in fn), None)
            if kind is None:
                continue
            x = oggx.decode(os.path.join(d, fn))
            if x.ndim > 1:
                x = x.mean(axis=1)
            x = declip(x)
            x = normalize(x, HANDLING_LEVELS[kind])
            od = os.path.join(out_dir, folder); os.makedirs(od, exist_ok=True)
            oggx.write_ogg(x, os.path.join(od, fn), CM['handling'])
            done.append(f"{folder}/{fn}")
    return done


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=SND); ap.add_argument('--keys', default=','.join(WEAPONS)); ap.add_argument('--no-handling', action='store_true')
    a = ap.parse_args()
    allsec = {}
    for key in a.keys.split(','):
        files = build_weapon(key, WEAPONS[key], a.out)
        allsec[key] = sections(key, WEAPONS[key], files)
        print(key, {k: len(v) for k, v in files.items()})
    with open(os.path.join(a.out, 'sections.ltx'), 'w') as f:
        f.write(''.join(allsec.values()))
    if not a.no_handling:
        print('handling:', len(fix_handling(a.out)), 'files')
