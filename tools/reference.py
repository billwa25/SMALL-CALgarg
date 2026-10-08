"""Reference matching.  The pack's shots are pushed toward the measured properties of a set of reference gunshots the user
supplied: tools/reference_target.json holds the numbers (envelope curve, octave-band balance per time window, crest factor,
peak level).  No reference audio is used or copied; everything added to a shot is made from that shot's own recording.

    T = reference.load_target()
    y = reference.match(x, T, lf_db=0.0)          # x mono (n,) or stereo (n, 2), onset at the start, any level

python tools/reference.py --measure REF.ogg ...   # writes tools/reference_target.json
python tools/reference.py --test FILE.ogg ...     # match and print the metrics against the target"""
import sys, os, json, subprocess, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pocketpops as pp

SR = 44100
ms, db, to_db, rms = pp.ms, pp.db, pp.to_db, pp.rms
BANDS = [(20, 60), (60, 120), (120, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 16000), (16000, 22050)]
WINDOWS = [(0, 10), (10, 40), (40, 150), (150, 400), (0, 150)]
CURVE_MS = 500
TARGET_PATH = os.path.join(os.path.dirname(__file__), 'reference_target.json')


# ---------------------------------------------------------------------------------------------------------------------- analysis

def decode_mono(path):
    r = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-f', 'f32le', '-ac', '1', '-ar', str(SR), '-'], capture_output=True)
    return np.frombuffer(r.stdout, np.float32).astype(np.float64)


def mono(x):
    return x if x.ndim == 1 else x.mean(1)


def lowpass(x, fc, order=4):
    return pp.fft_gain(x, lambda f: 1 / np.sqrt(1 + (f / fc) ** (2 * order)))


def onset(x):
    e = pp.envelope(mono(x), 2)
    return int(np.argmax(e > e.max() * db(-30)))


def band_levels(seg, bands=BANDS):
    """octave-ish band energies in dB relative to the segment's total"""
    seg = mono(seg)
    X = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2; f = np.fft.rfftfreq(len(seg), 1 / SR); tot = X.sum() + 1e-18
    return [10 * np.log10(X[(f >= a) & (f < b)].sum() / tot + 1e-12) for a, b in bands]


def window(x, i0, w):
    return x[i0 + ms(w[0]):i0 + ms(w[1])]


def env_curve(x, i0, length_ms=CURVE_MS):
    """5 ms envelope in dB relative to its peak, 1 ms steps from the onset"""
    e = pp.envelope(mono(x), 5); seg = e[i0:i0 + ms(length_ms)]
    seg = np.concatenate([seg, np.full(max(0, ms(length_ms) - len(seg)), 1e-12)])
    return to_db(seg / e.max())[::ms(1)]


def crest(x, i0):
    """peak over the rms of the first 100 ms, dB (stereo: mean of the channels' own crest factors)"""
    if x.ndim == 2:
        return float(np.mean([crest(x[:, c], i0) for c in range(x.shape[1])]))
    return to_db(np.abs(x).max()) - to_db(rms(x[i0:i0 + ms(100)]))


def measure(paths):
    curves, bands, crests, peaks, wlev = [], [], [], [], []
    for p in paths:
        x = decode_mono(p); i0 = onset(x); pk = np.abs(x).max()
        curves.append(env_curve(x, i0))
        bands.append([band_levels(window(x, i0, w)) for w in WINDOWS])
        wlev.append([to_db(rms(window(x, i0, w)) / pk) for w in WINDOWS])
        crests.append(crest(x, i0)); peaks.append(to_db(pk))
    return dict(files=[os.path.basename(p) for p in paths], env_db=np.median(curves, 0).round(2).tolist(),
                bands_db=np.median(bands, 0).round(2).tolist(), win_rel_db=np.median(wlev, 0).round(2).tolist(),
                crest_db=round(float(np.median(crests)), 2), peak_dbfs=round(float(np.median(peaks)), 2), windows=WINDOWS, bands=BANDS)


def load_target(path=TARGET_PATH):
    with open(path) as f:
        return json.load(f)


def metrics(x, T):
    """how far a shot is from the target: crest, envelope error over 10-200 ms, per-window band errors (dB)"""
    i0 = onset(x); cur = env_curve(x, i0); ref = np.array(T['env_db'])
    env_err = float(np.sqrt(((cur[10:200] - ref[10:200]) ** 2).mean()))
    bands = [band_levels(window(x, i0, w)) for w in WINDOWS]
    berr = [[round(c - r, 1) for c, r in zip(b, rb)] for b, rb in zip(bands, T['bands_db'])]
    return dict(crest_db=round(crest(x, i0), 1), env_rms_err_db=round(env_err, 1), band_err_db=berr, peak_dbfs=round(float(to_db(np.abs(x).max())), 1))


# ---------------------------------------------------------------------------------------------------------------------- processing

def resample(x, ratio):
    n = int(len(x) / ratio); idx = np.arange(n) * ratio
    return np.interp(idx, np.arange(len(x)), x)


def thump(x, length_ms=80):
    """the shot's low-frequency pressure pulse, made from the shot itself: its first 8 ms slowed down twelve times (the crack's
    1-1.5 kHz lands at 85-125 Hz) plus its own low end, both band-passed to 85-125 Hz (a 30 ms pulse this high leaves the
    20-60 Hz band as quiet as the reference's), under a pulse envelope that rises from 10 ms, holds to 32 ms and is 17 dB
    down by 48 ms (the reference's block lasts 50 ms and peaks 8-24 ms in).  Peak-normalised, mono"""
    m = mono(x); n = ms(length_ms); t = np.arange(n) / SR
    slow = pp.highpass(lowpass(resample(m[:ms(8)], 1 / 12.0), 125, 8), 85, 8)
    own = pp.highpass(lowpass(m[:n], 125, 8), 85, 8)
    slow = np.concatenate([slow, np.zeros(max(0, n - len(slow)))])[:n]
    own = np.concatenate([own, np.zeros(max(0, n - len(own)))])[:n]
    core = slow / (np.abs(slow).max() + 1e-9) + 0.5 * own / (np.abs(own).max() + 1e-9)
    a = np.clip((t - 0.010) / 0.008, 0, 1); a = a * a * (3 - 2 * a)
    env = a * np.where(t > 0.032, np.exp(-(t - 0.032) / 0.008), 1.0)
    y = core * env
    return y / (np.abs(y).max() + 1e-9)


def add_thump(x, T, i0, lf_db=0.0):
    """mix the thump in so that the 60-120 Hz share of the 10-40 ms window matches the reference's (plus lf_db).  Done after
    the EQ, which has already set the balance of everything above 120 Hz"""
    th = thump(x)
    target = T['bands_db'][1][1] + lf_db
    lvl = 0.3
    def mixed(l):
        y = x.copy(); n = min(len(th), len(y) - i0)
        if y.ndim == 1:
            y[i0:i0 + n] += th[:n] * l
        else:
            y[i0:i0 + n] += th[:n, None] * l
        return y
    for _ in range(12):
        cur = band_levels(window(mixed(lvl), i0, WINDOWS[1]))[1]
        lvl *= db((target - cur) * 0.6); lvl = float(np.clip(lvl, 0.0, 3.0))
    return mixed(lvl), lvl


LIFT_MAX = [4.0, 8.0] + [12.0] * 8                  # dB of lift allowed per band: little below 120 Hz


def renorm(levels, from_band):
    """band shares (dB) re-normalised over the bands from `from_band` up, so the low end does not sway the balance"""
    lv = np.array(levels, float); keep = lv[from_band:]
    return lv - 10 * np.log10((10 ** (keep / 10)).sum() + 1e-18)


def eq_gains(x, i0, w, T, k, max_db, from_band):
    cur = renorm(band_levels(window(x, i0, w)), from_band); ref = renorm(T['bands_db'][k], from_band)
    return np.array([np.clip(r - c, -max_db, min(max_db, LIFT_MAX[b])) if b >= from_band else 0.0 for b, (c, r) in enumerate(zip(cur, ref))])


def min_phase_ir(gain_fn, n=8192):
    """minimum-phase impulse response with the given magnitude (cepstral method): the EQ then rings after the onset only"""
    f = np.fft.rfftfreq(n, 1 / SR); mag = np.maximum(gain_fn(f), 1e-5)
    cep = np.fft.irfft(np.log(mag), n)
    fold = np.zeros(n); fold[0] = cep[0]; fold[1:n // 2] = 2 * cep[1:n // 2]; fold[n // 2] = cep[n // 2]
    return np.fft.irfft(np.exp(np.fft.rfft(fold)), n)[:n // 2]


def min_phase_highpass(x, fc, order=8):
    """steep causal high-pass (the reference carries almost nothing under 60 Hz)"""
    h = min_phase_ir(lambda f: 1 / np.sqrt(1 + (fc / np.maximum(f, 1e-3)) ** (2 * order)))
    def conv(v):
        n = len(v) + len(h) - 1; N = 1 << (n - 1).bit_length()
        return np.fft.irfft(np.fft.rfft(v, N) * np.fft.rfft(h, N), N)[:len(v)]
    return conv(x) if x.ndim == 1 else np.stack([conv(x[:, c]) for c in range(2)], 1)


def apply_eq(x, gains):
    centers = np.log2([np.sqrt(a * b) for a, b in BANDS])
    def fn(f):
        return db(np.interp(np.log2(np.maximum(f, 1.0)), centers, gains, left=0.0, right=gains[-1]))
    h = min_phase_ir(fn)
    def conv(v):
        n = len(v) + len(h) - 1; N = 1 << (n - 1).bit_length()
        return np.fft.irfft(np.fft.rfft(v, N) * np.fft.rfft(h, N), N)[:len(v)]
    return conv(x) if x.ndim == 1 else np.stack([conv(x[:, c]) for c in range(2)], 1)


def match_eq(x, T, i0, max_db=12.0, passes=2):
    """time-varying octave-band EQ: one static EQ per reference window (0-10, 10-40, 40-150, 150-400 ms), cross-faded in
    time, each bringing that window's balance to the target.  Below 120 Hz the first two windows are left to the thump.
    Two passes, since changing one band's gain moves every band's share"""
    n = len(x); t_ms = (np.arange(n) - i0) / SR * 1000
    wins = WINDOWS[:4]
    edges = [10.0, 40.0, 150.0]                      # cross-fade centres, ms; fade width grows with time
    widths = [6.0, 16.0, 60.0]
    weights = []
    for k in range(len(wins)):
        lo = 1.0 if k == 0 else np.clip((t_ms - edges[k - 1]) / widths[k - 1] + 0.5, 0, 1)
        hi = 1.0 if k == len(wins) - 1 else 1 - np.clip((t_ms - edges[k]) / widths[k] + 0.5, 0, 1)
        weights.append(np.asarray(lo) * np.asarray(hi))
    total = [np.zeros(len(BANDS)) for _ in wins]
    y = x
    for _ in range(passes):
        gains = [eq_gains(y, i0, w, T, k, max_db, 2 if k < 2 else 0) for k, w in enumerate(wins)]
        total = [np.clip(t + g, -max_db, max_db) for t, g in zip(total, gains)]
        y = np.zeros_like(x)
        for g, wgt in zip(total, weights):
            f = apply_eq(x, g)
            y += f * wgt if x.ndim == 1 else f * wgt[:, None]
    return y, total


def match_envelope(x, T, i0, lift_max_db=10.0, cut_max_db=24.0, lift_below_db=None):
    """time gain that brings the 5 ms envelope to the target curve over its length, lifting at most lift_max_db (and, with
    lift_below_db, only where the envelope already sits that far under the peak); beyond the curve the shot may only be cut"""
    m = mono(x)
    e = pp.envelope(m, 5); cur = to_db(e / e.max())
    ref = np.array(T['env_db']); t_ms = np.arange(len(m) - i0) / SR * 1000
    ref_full = np.interp(t_ms, np.arange(len(ref)), ref, right=ref[-1])
    g = np.zeros(len(m)); g[i0:] = ref_full - cur[i0:]
    beyond = np.arange(len(m)) >= i0 + ms(CURVE_MS)
    g = np.where(beyond, np.minimum(g, 0.0), np.clip(g, -cut_max_db, lift_max_db))
    if lift_below_db is not None:
        g = np.where(cur > lift_below_db, np.minimum(g, 0.0), g)
    ramp = np.clip((np.arange(len(m)) - i0 - ms(8)) / ms(4), 0, 1)          # the attack itself is left alone: the reference
    g = g * ramp                                                             # curve's first ms only show its rms window
    k = np.hanning(ms(8)); g = np.convolve(g, k / k.sum(), 'same')
    lin = db(g)
    return x * lin if x.ndim == 1 else x * lin[:, None]


def limiter(x, ceiling, lookahead_ms=1.0, release_ms=30.0):
    """look-ahead peak limiter, channels linked"""
    a = np.abs(x) if x.ndim == 1 else np.abs(x).max(1)
    need = np.minimum(1.0, ceiling / np.maximum(a, 1e-9))
    L = max(2, ms(lookahead_ms))
    padded = np.concatenate([need, np.ones(L)])
    g1 = np.lib.stride_tricks.sliding_window_view(padded, L + 1).min(1)[:len(need)]
    g2 = np.convolve(np.concatenate([np.ones(L - 1), g1]), np.ones(L) / L, 'valid')
    alpha = np.exp(-1.0 / (release_ms / 1000 * SR))
    g = np.empty(len(need)); cur = 1.0
    for n in range(len(need)):
        v = g2[n]
        cur = v if v < cur else v + (cur - v) * alpha
        g[n] = cur
    return x * g if x.ndim == 1 else x * g[:, None]


def densify(x, T, i0, knee=1.5, crest_bias_db=0.0):
    """drive into a soft clipper and the limiter until the first-100 ms crest factor meets the target (plus a bias); peak at
    the target level"""
    ceiling = db(T['peak_dbfs']); goal = T['crest_db'] + crest_bias_db
    x = x / (np.abs(x).max() + 1e-9) * ceiling
    drive_db = max(0.0, crest(x, i0) - goal)
    y = x
    for _ in range(5):
        d = x * db(drive_db)
        d = np.tanh(d * knee / ceiling) * ceiling / np.tanh(knee)
        y = limiter(d, ceiling)
        drive_db = float(np.clip(drive_db + (crest(y, i0) - goal) * 0.8, 0.0, 30.0))
    return y / (np.abs(y).max() + 1e-9) * ceiling, drive_db


def match(x, T, lf_db=0.0):
    """octave balance per window, thump from the shot itself, balance again, a steep 55 Hz high-pass, envelope, density,
    a second envelope correction after the limiter, level.  -> matched shot, info"""
    x = x / (np.abs(x).max() + 1e-9); i0 = onset(x)
    y, gains = match_eq(x, T, i0)
    y, lvl = add_thump(y, T, i0, lf_db)
    y, gains2 = match_eq(y, T, i0, passes=1)                        # re-balance 120 Hz up around the pulse's skirt
    y = min_phase_highpass(y, 55.0)                                 # before the limiter, so its ringing is limited too
    y = match_envelope(y, T, i0)
    y, drive = densify(y, T, i0, crest_bias_db=-1.0)                # the correction below raises the crest by about that much
    y = match_envelope(y, T, i0, lift_max_db=6.0, cut_max_db=5.0, lift_below_db=-8.0)
    ceiling = db(T['peak_dbfs'])
    y = y / (np.abs(y).max() + 1e-9) * ceiling
    return y, dict(thump=round(lvl, 2), eq_db=[[round(float(a + b), 1) for a, b in zip(g, g2)] for g, g2 in zip(gains, gains2)], drive_db=round(drive, 1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--measure', nargs='+'); ap.add_argument('--test', nargs='+'); ap.add_argument('--lf', type=float, default=0.0); ap.add_argument('--out')
    a = ap.parse_args()
    if a.measure:
        T = measure(a.measure)
        with open(TARGET_PATH, 'w') as f:
            json.dump(T, f, indent=1)
        print('target from', T['files'], 'crest', T['crest_db'], 'peak', T['peak_dbfs'])
        print('env 0..200ms/10ms:', ' '.join(f"{v:.0f}" for v in T['env_db'][:200:10]))
        for w, b in zip(T['windows'], T['bands_db']):
            print(f"  {str(w):10s}", ' '.join(f"{v:6.1f}" for v in b))
    if a.test:
        T = load_target()
        for p in a.test:
            x = pp.oggx.decode(p)
            y, info = match(x, T, a.lf)
            m = metrics(y, T)
            print(f"{os.path.basename(p):26s} crest {m['crest_db']:4.1f} (target {T['crest_db']}) env err {m['env_rms_err_db']:4.1f} dB  thump {info['thump']} drive {info['drive_db']}")
            for k, g in enumerate(info['eq_db']):
                print(f"   eq {str(WINDOWS[k]):10s}", ' '.join(f"{v:6.1f}" for v in g))
            cur = env_curve(y, onset(y))
            print('   env target 0..300/10ms:', ' '.join(f"{v:4.0f}" for v in T['env_db'][:300:10]))
            print('   env result 0..300/10ms:', ' '.join(f"{v:4.0f}" for v in cur[:300:10]))
            for w, e in zip(WINDOWS, m['band_err_db']):
                print(f"   {str(w):10s} err", ' '.join(f"{v:6.1f}" for v in e))
            if a.out:
                os.makedirs(a.out, exist_ok=True); pp.oggx.encode(y.astype(np.float32), os.path.join(a.out, os.path.basename(p)))
