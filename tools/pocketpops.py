"""Clean-up of the pocket-pistol recordings (the 'pocketpops' set): spectral denoising from the recordings' own pre-roll (pooled per
calibre), onset trim, brass-casing hits cut off the tail, the sustained room wash pulled down under the crack, decay tightened,
transient emphasised.  Shots come out mono (the cleaner channel) and stereo; dry-fire clicks come out mono.

    profiles = pocketpops.profiles(dir)                # noise profile per group ('22lr', '25acp', '32', '380', '38s')
    mono, stereo, info = pocketpops.shot(path, profiles)
    click, info = pocketpops.empty(path, profiles)

python tools/pocketpops.py SRC_DIR OUT_DIR            # cleaned <name>.ogg / <name>_st.ogg per shot, <name>.ogg per empty"""
import os, sys, re, glob, subprocess
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import oggx

SR = 44100
NFFT, HOP = 1024, 256
WIN = np.sqrt(np.hanning(NFFT + 1)[:-1])


def db(x):
    return 10 ** (x / 20)


def to_db(a):
    return 20 * np.log10(np.maximum(a, 1e-12))


def ms(v):
    return int(v * SR / 1000)


def rms(x):
    return float(np.sqrt((x ** 2).mean() + 1e-18)) if len(x) else 0.0


def group_of(name):
    """'22lr-3', '25acp4', '38s-empty2' -> '22lr', '25acp', '38s'"""
    m = re.match(r'^(\d+[a-z]*)', os.path.basename(name))
    return m.group(1) if m else 'misc'


# ---------------------------------------------------------------------------------------------------------------------- io

def decode_stereo(path):
    """-> (n, 2) float64 at 44.1 kHz"""
    r = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-f', 'f32le', '-ac', '2', '-ar', str(SR), '-'], capture_output=True)
    return np.frombuffer(r.stdout, np.float32).reshape(-1, 2).astype(np.float64)


def declip(x, thr=0.985):
    """rebuild hard-clipped runs with a cubic through the samples around them (mono)"""
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
                x[i:j] = np.sign(x[i]) * np.maximum(np.abs(rep), thr)
            i = j
        else:
            i += 1
    return x


# ---------------------------------------------------------------------------------------------------------------------- analysis

def stft(x):
    pad = np.concatenate([np.zeros(NFFT), x, np.zeros(NFFT)])
    n = 1 + (len(pad) - NFFT) // HOP
    frames = np.stack([pad[i * HOP:i * HOP + NFFT] * WIN for i in range(n)])
    return np.fft.rfft(frames, axis=1)


def istft(X, length):
    frames = np.fft.irfft(X, NFFT, axis=1) * WIN
    out = np.zeros((len(frames) - 1) * HOP + NFFT); norm = np.zeros_like(out)
    for i, fr in enumerate(frames):
        out[i * HOP:i * HOP + NFFT] += fr; norm[i * HOP:i * HOP + NFFT] += WIN ** 2
    out /= np.maximum(norm, 1e-6)
    return out[NFFT:NFFT + length]


def envelope(x, win_ms):
    w = max(1, ms(win_ms))
    return np.sqrt(np.convolve(x ** 2, np.ones(w) / w, 'same') + 1e-18)


def onset_index(x, floor, within_db=-15.0):
    """start of the shot: the first 2 ms window within `within_db` of the loudest one (a weak pre-event such as a hammer fall
    50 ms early is skipped), walked back to the noise, minus 1.5 ms"""
    env = envelope(x, 2); i = int(np.argmax(env > env.max() * db(within_db)))
    thr = max(6 * floor, 0.01 * np.abs(x).max())
    j = i
    while j > 0 and abs(x[j - 1]) > thr:
        j -= 1
    return max(0, min(i, j) - ms(1.5))


def first_event(x, i0, above=5.0):
    """where anything at all starts (a hammer fall, a breath, the shot): the first 2 ms window `above` times the median
    2 ms level of everything before the shot onset `i0`"""
    env = envelope(x, 2); med = np.median(env[:max(i0, ms(10))])
    env[:ms(5)] = 0                                # the first milliseconds of a file are not an event
    return int(np.argmax(env > above * med))


def preroll(x, i0):
    """the noise before anything happens: up to 3 ms before the first event or the shot onset, whichever is earlier"""
    return x[:max(0, min(i0, first_event(x, i0)) - ms(3))]


def noise_profile(segments):
    """pooled noise profile: 85th-percentile magnitude per bin over every frame of the segments, and their broadband rms"""
    mags = np.concatenate([np.abs(stft(s)) for s in segments])
    return np.percentile(mags, 85, axis=0) + 1e-9, rms(np.concatenate(segments))


def profiles(src_dir, min_preroll_ms=100):
    """noise profile per group, from the pre-roll of every recording in the group with at least `min_preroll_ms` before its onset"""
    segs = {}
    for p in sorted(glob.glob(os.path.join(src_dir, '*.ogg'))):
        x = decode_stereo(p).mean(1)
        i0 = onset_index(x, rms(x[-ms(60):]))
        pre = preroll(x, i0)
        if len(pre) >= ms(min_preroll_ms):
            segs.setdefault(group_of(p), []).append(pre[-ms(400):])
    out = {}
    for g, s in segs.items():                     # a recording whose noise is 6 dB off the group's median is not the same noise
        med = np.median([rms(x) for x in s])
        out[g] = noise_profile([x for x in s if abs(to_db(rms(x)) - to_db(med)) <= 6.0])
    return out


def gate_gains(x, nmag, alpha=2.0, floor_db=-40.0, quiet_ratio=3.0):
    """spectral-gate gains (frames, bins): 1 - alpha * noise / signal, floored; frames near the noise energy dropped to the floor;
    smoothed over 5 bins and with a slow release in time"""
    mag = np.abs(stft(x))
    G = np.clip(1 - alpha * nmag[None, :] / (mag + 1e-9), db(floor_db), 1)
    quiet = (mag ** 2).sum(1) < quiet_ratio * (nmag ** 2).sum()
    G[quiet] = db(floor_db)
    k = np.ones(5) / 5
    G = np.apply_along_axis(lambda r: np.convolve(r, k, 'same'), 1, G)
    for t in range(1, len(G)):
        G[t] = np.maximum(G[t], 0.7 * G[t - 1])
    return G


def apply_gains(x, G):
    return istft(stft(x) * G, len(x))


def late_transient(y, from_s, floor_db, jump_db=7.0, rise_db=5.0):
    """index of a transient landing in the decayed tail (a casing hitting the ground): a 30 ms envelope that jumps `jump_db`
    above the minimum of the preceding 100 ms, rising `rise_db` within 25 ms, once the tail is already under `floor_db`
    relative to the peak.  None if there is no such event"""
    env = to_db(envelope(y, 30)); pk = env.max()
    step = ms(1)
    for i in range(max(ms(from_s * 1000), ms(110)), len(y), step):
        prev = env[i - ms(100):i - ms(10)].min()
        if prev - pk <= floor_db and env[i] - prev >= jump_db and env[i] - env[i - ms(25)] >= rise_db:
            return max(ms(60), i - ms(15))
    return None


def end_index(y, end_db, hold_ms=20):
    """one `hold_ms` after the last 10 ms window above `end_db` relative to the peak"""
    env = envelope(y, 10); above = np.nonzero(env > env.max() * db(end_db))[0]
    return min(len(y), (int(above[-1]) if len(above) else len(y)) + ms(hold_ms))


# ---------------------------------------------------------------------------------------------------------------------- shaping

def fft_gain(x, gain_fn, pad_ms=100):
    """zero-phase filter; padded so the filter's ringing around the onset and the end does not wrap around the buffer"""
    p = ms(pad_ms); xp = np.concatenate([np.zeros(p), x, np.zeros(p)])
    n = len(xp); X = np.fft.rfft(xp); f = np.fft.rfftfreq(n, 1 / SR)
    return np.fft.irfft(X * gain_fn(f), n)[p:p + len(x)]


def highpass(x, fc, order=2):
    return fft_gain(x, lambda f: 1 / np.sqrt(1 + (fc / np.maximum(f, 1e-3)) ** (2 * order)))


def peaking(x, fc, gain_db, q=1.0):
    g = db(gain_db) - 1
    return fft_gain(x, lambda f: 1 + g * np.exp(-0.5 * (np.log2(np.maximum(f, 1) / fc) * q * 1.44) ** 2))


def fades(y, in_ms=1.0, out_ms=30.0):
    y = y.copy()
    fi = min(len(y) // 2, ms(in_ms)); y[:fi] *= np.linspace(0, 1, fi) ** 2
    fo = min(len(y) // 2, ms(out_ms)); y[-fo:] *= np.linspace(1, 0, fo) ** 2
    return y


def splice_gap(seg, mc, drop_db=-24.0, back_db=-20.0, search_ms=80):
    """these recordings were made beyond the room's critical distance: the crack decays into near silence and the hall's
    reverberant wash arrives 20-40 ms later, louder than the crack.  The silent gap is cut out (all channels, 3 ms crossfade)
    so the wash follows the crack as a blast body would.  -> seg, gap length in ms (0 when there is none)"""
    m = seg[:, mc]; env = envelope(m, 5); c = env[:ms(20)].max()
    pk = int(np.argmax(env[:ms(20)]))
    fall = np.nonzero(env[pk:ms(search_ms)] < c * db(drop_db))[0]
    if not len(fall):
        return seg, 0.0
    a = pk + int(fall[0])
    back = np.nonzero(env[a:ms(search_ms)] > c * db(back_db))[0]
    if not len(back) or back[0] < ms(4):
        return seg, 0.0
    b = a + int(back[0])
    xf = ms(3)
    out = np.concatenate([seg[:a], seg[b:]])
    w = np.linspace(0, 1, xf)[:, None]
    out[a - xf:a] = seg[a - xf:a] * (1 - w) + seg[b - xf:b] * w
    return out, (b - a) / SR * 1000


def decay_curve(m, crack_ms=25, body_db=-3.0, body_tau=0.035, tail_db=-20.0, tail_tau=0.15, fill_db=-9.0, fill_max_db=12.0):
    """time gain curve: nothing above a target decay.  The first `crack_ms` (the crack, level C = its 10 ms rms peak) are
    untouched; after it the envelope may not exceed C+body_db falling with `body_tau`, nor C+tail_db falling with `tail_tau`
    (the slower of the two wins).  The gain is only ever <= 1 there, so a recording that already decays faster than the target
    is left alone.  The one place the gain goes above 1: a dip under C+fill_db between 6 and 45 ms (the crack dying before the
    blast body arrives) is lifted by up to `fill_max_db`.  -> gain curve, the largest cut in dB"""
    env = envelope(m, 10); t = np.arange(len(m)) / SR; t0 = crack_ms / 1000
    c = env[:ms(crack_ms)].max()
    body = c * db(body_db) * np.exp(-np.maximum(t - t0, 0) / body_tau)
    tail = c * db(tail_db) * np.exp(-np.maximum(t - t0, 0) / tail_tau)
    target = np.maximum(body, tail); target[:ms(crack_ms)] = np.inf
    g = np.minimum(1.0, target / np.maximum(env, 1e-9))
    fill = np.clip(c * db(fill_db) / np.maximum(env, 1e-9), 1.0, db(fill_max_db))
    win = (t >= 0.004) & (t <= 0.045)
    g[win] = np.maximum(g[win], fill[win])
    k = np.hanning(ms(4)); g = np.convolve(g, k / k.sum(), 'same'); g[:ms(4)] = 1.0
    return g, 20 * np.log10(max(g.min(), 1e-6))


def pop(x, attack=0.45):
    """rumble out, slight mud cut, transient emphasis, saturation of the first milliseconds"""
    x = highpass(x, 40, 2)
    x = peaking(x, 350, -1.5, 1.0)
    t = np.arange(len(x)) / SR
    x = x * (1 + attack * np.exp(-t / 0.008))
    drive = np.exp(-t / 0.006)
    return np.tanh(x * (1 + 1.5 * drive)) / np.tanh(1 + 1.5 * drive)


def _prepare(path, profiles, alpha, floor_db, quiet_ratio):
    """decode, declip, locate the onset, match the group noise profile to this file's pre-roll, gate both channels.
    -> x (gated, (n, 2)), i0, mono channel index, noise floor rms, info"""
    x = decode_stereo(path)
    x = np.stack([declip(x[:, 0]), declip(x[:, 1])], 1)
    mid = x.mean(1)
    nmag, nrms = profiles.get(group_of(path)) or noise_profile([mid[-ms(60):]])
    i0 = onset_index(mid, nrms)
    pre = preroll(mid, i0)
    if len(pre) >= ms(100):                       # enough of its own noise: use it rather than the group's
        nmag, nrms = noise_profile([pre]); scale = 1.0
    else:
        scale = float(np.clip(rms(pre) / nrms, 0.5, 2.0)) if len(pre) >= ms(25) else 1.0
    floor = nrms * scale
    i0 = onset_index(mid, floor)
    pre = preroll(mid, i0)
    G = [gate_gains(x[:, c], nmag * scale, alpha, floor_db, quiet_ratio) for c in range(2)]
    y = np.stack([apply_gains(x[:, c], G[c]) for c in range(2)], 1)
    chfloor = [rms(preroll(x[:, c], i0)) if len(pre) >= ms(25) else floor for c in range(2)]
    snr = [np.abs(x[:, c]).max() / max(chfloor[c], 1e-9) for c in range(2)]
    mono_ch = 0 if snr[0] >= snr[1] else 1
    info = dict(onset_ms=i0 / SR * 1000, preroll_ms=len(pre) / SR * 1000, noise_scale=scale, noise_dbfs=to_db(floor), mono_ch=mono_ch, snr_db=to_db(max(snr)))
    return y, i0, mono_ch, floor, info


def shot(path, profiles, max_s=0.9, **shape):
    """-> (mono, stereo, info): cleaned shot, both peak-normalised to -1 dBFS.  `shape` = decay_curve keyword arguments"""
    y, i0, mc, floor, info = _prepare(path, profiles, alpha=2.0, floor_db=-40.0, quiet_ratio=3.0)
    seg, gap = splice_gap(y[i0:i0 + int(max_s * SR) + ms(100)], mc)
    m = seg[:int(max_s * SR), mc]
    cut = late_transient(m, 0.25, -24.0)
    info['brass_cut_ms'] = cut / SR * 1000 if cut else None
    g, gmin = decay_curve(m[:cut] if cut else m, **shape)
    shaped = (m[:cut] if cut else m) * g
    end = end_index(shaped, -66.0)
    out_ms = 60.0 if cut and end >= cut - ms(2) else 40.0
    mono = pop(fades(seg[:end, mc] * g[:end], 1.0, out_ms))
    st = np.stack([pop(fades(seg[:end, c] * g[:end], 1.0, out_ms)) for c in range(2)], 1)
    mono = mono / (np.abs(mono).max() + 1e-9) * db(-1.0)
    st = st / (np.abs(st).max() + 1e-9) * db(-1.0)
    info.update(gap_ms=gap, shape_cut_db=gmin, len_s=len(mono) / SR)
    return mono, st, info


def empty(path, profiles, max_s=0.35):
    """-> (click, info): mono dry-fire click, cleaned and tightened, peak -4 dBFS"""
    y, i0, mc, floor, info = _prepare(path, profiles, alpha=2.5, floor_db=-40.0, quiet_ratio=1.5)
    m = y[i0:i0 + int(max_s * SR), mc]
    cut = late_transient(m, 0.10, -30.0)
    info['brass_cut_ms'] = cut / SR * 1000 if cut else None
    end = end_index(m[:cut] if cut else m, -50.0)
    c = fades(m[:end], 0.5, 12.0 if cut and end >= cut - ms(2) else 25.0)
    c = highpass(c, 300, 2)
    t = np.arange(len(c)) / SR
    c = c * (1 + 0.4 * np.exp(-t / 0.005))
    c = c / (np.abs(c).max() + 1e-9) * db(-4.0)
    info.update(len_s=len(c) / SR)
    return c, info


def describe(y):
    """time to -20 / -40 dB after the peak and the level of the last 20 ms, for the stats table"""
    m = y if y.ndim == 1 else y.mean(1)
    env = to_db(envelope(m, 5)); pk = int(np.argmax(env)); rel = env[pk:] - env[pk]
    t20 = int(np.argmax(rel < -20)) / SR if (rel < -20).any() else None
    t40 = int(np.argmax(rel < -40)) / SR if (rel < -40).any() else None
    return t20, t40, to_db(rms(m[-ms(20):])) - env[pk]


if __name__ == '__main__':
    src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
    prof = profiles(src)
    print('noise profiles:', {g: f"{to_db(r):.1f} dBFS" for g, (n, r) in prof.items()})
    for p in sorted(glob.glob(os.path.join(src, '*.ogg'))):
        name = os.path.splitext(os.path.basename(p))[0]
        if 'empty' in name:
            c, info = empty(p, prof); oggx.encode(c.astype(np.float32), os.path.join(out, name + '.ogg')); y = c
        else:
            m, s, info = shot(p, prof); y = m
            oggx.encode(m.astype(np.float32), os.path.join(out, name + '.ogg')); oggx.encode(s.astype(np.float32), os.path.join(out, name + '_st.ogg'))
        t20, t40, endlvl = describe(y)
        cut = f"cut@{info['brass_cut_ms']:.0f}ms" if info['brass_cut_ms'] else 'no-cut'
        print(f"{name:14s} {'empty' if 'empty' in name else 'shot ':5s} {info['len_s']:.2f}s onset {info['onset_ms']:5.0f}ms pre {info['preroll_ms']:4.0f}ms "
              f"ch{info['mono_ch']} snr {info['snr_db']:4.0f}dB noise {info['noise_dbfs']:5.1f} gap {info.get('gap_ms', 0):4.1f}ms shape {info.get('shape_cut_db', 0):5.1f}dB "
              f"{cut:10s} -20dB@{(t20 or 0)*1000:4.0f}ms -40dB@{(t40 or 0)*1000:4.0f}ms end {endlvl:6.1f}dB")
