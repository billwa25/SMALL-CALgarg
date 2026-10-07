"""Layered gunshots for the small-calibre pack, built from the pack's own recordings: body and snap from the blast itself, action clicks cut from the unjam recordings, echoes of the shot for the tails.

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
    'pt22': dict(unjam='pt25_unjam.ogg', folder='taurus_pts', src='22lr_pistol_shoot.ogg', cal='22lr', mech=True),
    'pt25': dict(unjam='pt25_unjam.ogg', folder='taurus_pts', src='25acp_shoot.ogg', cal='25acp', mech=True),
    'cvp1908': dict(unjam='cvp1908_unjam.ogg', folder='cvp1908', src='25acp_shoot_striker.ogg', cal='25acp', mech=True, seed=7),
    'sav1907': dict(unjam='sav1907_unjam.ogg', folder='sav1907', src='32acp_shoot.ogg', cal='32acp', mech=True),
    'rem51': dict(unjam='rem51_unjam.ogg', folder='rem51', src='380acp_shoot_striker.ogg', cal='380acp', mech=True),
    'trejo22': dict(unjam='trejo22_unjam.ogg', folder='trejo22', src='22lr_pistol_shoot.ogg', cal='22lr', mech=True, auto=True, seed=3),
    '9galo22': dict(folder='9galo22', src='rev22lr_shoot.ogg', cal='22lr_rev', mech=False),
    'cpp38': dict(folder='cpp38', src='38spl_shoot.ogg', cal='38spl', mech=False),
}

# calibre profile: `low` = 60-200 Hz against 500-3000 Hz (dB, first 120 ms), `high` = 3-10 kHz against 500-3000 Hz (dB, first 30 ms), echo T60 s, mid dip dB, far low-pass Hz, action level
CAL = {
    '22lr': dict(high=2.0, low=-10.0, t60=0.8, dip=-4.5, far_lp=1800, mech_level=0.30),
    '22lr_rev': dict(high=1.5, low=-9.0, t60=0.9, dip=-4.5, far_lp=1800, mech_level=0.0),
    '25acp': dict(high=1.5, low=-9.0, t60=0.9, dip=-4.5, far_lp=1800, mech_level=0.34),
    '32acp': dict(high=1.0, low=-8.0, t60=1.0, dip=-4.0, far_lp=1600, mech_level=0.36),
    '380acp': dict(high=0.5, low=-7.0, t60=1.1, dip=-3.5, far_lp=1500, mech_level=0.38),
    '38spl': dict(high=-1.0, low=-6.0, t60=1.2, dip=-3.5, far_lp=1400, mech_level=0.0),
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
# Nothing here is a synthesised tone: body, snap, action and echoes are all taken from the recordings themselves.

def onset_click(x, ms=6.0):
    """the recording's own first milliseconds, spectrally flattened: a broadband snap that belongs to this shot"""
    n = int(ms * SR / 1000)
    h = x[:n] * np.hanning(2 * n)[n:]
    X = np.fft.rfft(h, 4 * n)
    mag = np.abs(X); sm = np.convolve(mag, np.ones(9) / 9, 'same') + 1e-6
    y = np.fft.irfft(X / sm, 4 * n)[:n]
    y = highpass(y, 1500, 2)
    return y / (np.abs(y).max() + 1e-9)


def blast_body(x, rng):
    """low end of the muzzle blast: the recording's own first 50 ms below 250 Hz under a short envelope, plus one pressure pulse (no pitch, no ring)"""
    n = int(0.05 * SR); t = np.arange(n) / SR
    low = lowpass(x[:n], 250, 3) * (1 - np.exp(-t / 0.0005)) * np.exp(-t / 0.028)
    f = rng.uniform(75, 100); m = int(SR / f)
    pulse = np.zeros(n); tt = np.arange(m) / SR
    pulse[:m] = np.sin(2 * np.pi * f * tt) * np.hanning(m)
    y = low / (np.abs(low).max() + 1e-9) + 0.3 * pulse
    return y / (np.abs(y).max() + 1e-9)


def close_shot(src, cal, rng, variant):
    """the recording, declipped, trimmed, re-balanced, with its onset sharpened and its blast's low end brought up"""
    x = declip(src)
    x = trim_onset(x)
    x = highpass(x, 45, 2)
    x = peaking(x, 1000, cal['dip'], q=0.9)
    x = peaking(x, 300, -1.0, q=1.0)
    x = shelf_high(x, 3500, 3.0)
    x = resample(x, 1 + rng.uniform(-0.02, 0.02)) if variant else x
    t = np.arange(len(x)) / SR
    x = x * (1 + 0.35 * np.exp(-t / 0.010))
    x = x * np.where(t > 0.05, np.exp(-(t - 0.05) / 0.30), 1.0)
    n = max(len(x), int(0.9 * SR)); y = np.zeros(n); y[:len(x)] = x
    y = y / (np.abs(y).max() + 1e-9)
    t = np.arange(len(y)) / SR
    drive = np.exp(-t / 0.008)                                               # harmonic saturation of the first milliseconds
    y = np.tanh(y * (1 + 2.5 * drive)) / np.tanh(1 + 2.5 * drive) * (1 + 0.15 * drive) + y * 0.0
    y = y / (np.abs(y).max() + 1e-9)
    env = np.exp(-t / 0.025)                                                 # transient-only brightness: high shelf on the first 25 ms
    y = y * (1 - env) + shelf_high(y, 3000, 8.0) * env
    y = y / (np.abs(y).max() + 1e-9)
    click = onset_click(y)
    lvl = 0.5
    for _ in range(8):
        z = place(y.copy(), click, 0.0, lvl)
        lvl *= db((cal['high'] - high_mid_db(z)) * 0.7); lvl = float(np.clip(lvl, 0.0, 2.5))
    y = place(y, click, 0.0, lvl)
    body = blast_body(y, rng)
    lvl = 0.3
    for _ in range(6):
        z = place(y.copy(), body, 0.0, lvl)
        lvl *= db((cal['low'] - low_mid_db(z)) * 0.7); lvl = float(np.clip(lvl, 0.0, 1.5))
    y = place(y, body, 0.0, lvl)
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


def tail_layer(close, cal, rng, stereo=False, auto=False):
    """the shot echoing off the surroundings: the recording's own first 150 ms, convolved with a sparse echo response"""
    t60 = cal['t60'] * (0.65 if auto else 1.0) * rng.uniform(0.92, 1.08)
    ir = echo_ir(t60, rng, stereo=stereo)
    head = highpass(close[:int(0.15 * SR)], 250, 2) * np.linspace(1, 0, int(0.15 * SR)) ** 0.7
    if stereo:
        y = np.stack([convolve(head, ir[:, c]) for c in range(2)], 1)
        y = np.stack([fade_out(highpass(y[:, c], 180, 2), 120) for c in range(2)], 1)
    else:
        y = fade_out(highpass(convolve(head, ir), 180, 2), 120)
    y = y[:int(t60 * 1.4 * SR)]
    return soft_clip(normalize(y, -10.0))


def far_layer(close, cal, rng, auto=False):
    """the distant report: the same echo response, longer and darker, on a band-limited shot"""
    t60 = cal['t60'] * 1.2 * (0.6 if auto else 1.0)
    ir = echo_ir(t60, rng, n_early=10, lp_start=2500, lp_end=600, dense=0.35)
    pop = bandpass(close[:int(0.1 * SR)], 150, cal['far_lp'], 3)
    y = fade_out(convolve(pop, ir)[:int(t60 * 1.3 * SR)], 150)
    return soft_clip(normalize(y, -18.0))


def actor_shot(close, rng):
    """the player's own shot: a small per-channel tilt and a few early reflections, no inter-channel delay"""
    L = peaking(close, 2500, 1.0, 1.0); R = peaking(close, 2500, -1.0, 1.0)
    L = peaking(L, 500, -0.8, 1.0); R = peaking(R, 500, 0.8, 1.0)
    ir = echo_ir(0.3, rng, n_early=6, length=int(0.18 * SR), stereo=True, dense=0.08)
    head = highpass(close[:int(0.03 * SR)], 300, 2)
    room = np.stack([convolve(head, ir[:, c]) for c in range(2)], 1)
    n = max(len(L), len(room)); y = np.zeros((n, 2)); y[:len(L), 0] = L; y[:len(R), 1] = R
    y[:len(room)] += room * 0.07
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
        handling = oggx.decode(os.path.join(SND, w['folder'], w['unjam']))
        for i in range(2):
            put('mech', i + 1, mech_layer(cal, rng, handling, auto), 'npc_mech')
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
