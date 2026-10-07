"""Convert any song into a Minecraft-villager "hrmm" cover.

    song ─► excerpt (chorus auto-detect or --start/--dur)
         ─► Mel-Band RoFormer: vocals / instrumental
         ─► RMVPE pitch contour of the vocal
         ─► syllable segmentation (voicing + onsets + energy dips + pitch steps)
         ─► one real-villager "hrmm" per syllable, following the singer's pitch (WORLD)
         ─► remix over the original instrumental

    .venv-conv/bin/python convert.py input/song.mp3 --start 90.3 --dur 32
"""

import argparse
import os
import subprocess
import sys
import warnings

import librosa
import numpy as np
import scipy.signal as ss
from scipy.ndimage import uniform_filter1d
import soundfile as sf

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "third_party", "seed-vc"))
import villager_sing as vs  # noqa: E402
from villager_sing import FPS, SR  # noqa: E402

SEP_MODEL = "vocals_mel_band_roformer.ckpt"
LEAD_MODEL = "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"
RMVPE_PT = os.path.join(HERE, "third_party", "seed-vc", "checkpoints",
                        "models--lj1995--VoiceConversionWebUI", "snapshots",
                        "e6d0c1a17da07c33557852f9dfa2bd44cc75737d", "rmvpe.pt")
PAD = 2.0  # seconds of context separated around the excerpt


# ---------------------------------------------------------------- 1. excerpt

def find_chorus(path, dur):
    """Start time of the loudest, most-repeated `dur`-second window."""
    y, sr = librosa.load(path, sr=22050)
    hop = 2048
    fps = sr / hop
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rec = librosa.segment.recurrence_matrix(
        librosa.feature.stack_memory(chroma, n_steps=8, delay=2),
        mode="affinity", width=int(10 * fps), sym=True).sum(1)
    w = int(dur * fps)
    starts = np.arange(0, len(rms) - w, int(fps))
    score = [rms[s:s + w].mean() * rec[s:s + w].mean() for s in starts]
    return float(starts[int(np.argmax(score))] / fps)


def separate(path, start, dur, work):
    """Cut [start-PAD, start+dur+PAD] and split it into vocals / instrumental."""
    os.makedirs(work, exist_ok=True)
    clip = os.path.join(work, "clip.wav")
    voc, inst = os.path.join(work, "vocals.wav"), os.path.join(work, "instrumental.wav")
    if os.path.exists(voc) and os.path.exists(inst):
        return clip, voc, inst
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(max(0, start - PAD)),
                    "-t", str(dur + 2 * PAD), "-i", path, "-ar", str(SR), clip], check=True)
    subprocess.run([os.path.join(os.path.dirname(sys.executable), "audio-separator"), clip,
                    "-m", SEP_MODEL, "--output_dir", work, "--output_format", "WAV",
                    "--model_file_dir", os.path.join(HERE, "models", "separator"),
                    "--custom_output_names", '{"Vocals": "vocals", "Other": "instrumental"}'],
                   check=True)
    return clip, voc, inst


def split_lead(voc, work):
    """Karaoke model: lead vocal vs. backing vocals (harmonies confuse the pitch tracker)."""
    lead, back = os.path.join(work, "lead.wav"), os.path.join(work, "backing_vocals.wav")
    if not (os.path.exists(lead) and os.path.exists(back)):
        subprocess.run([os.path.join(os.path.dirname(sys.executable), "audio-separator"), voc,
                        "-m", LEAD_MODEL, "--output_dir", work, "--output_format", "WAV",
                        "--model_file_dir", os.path.join(HERE, "models", "separator"),
                        "--custom_output_names",
                        '{"Vocals": "lead", "Instrumental": "backing_vocals"}'], check=True)
    return lead, back


# ---------------------------------------------------------------- 2. pitch

def vocal_f0(y):
    """RMVPE f0 on the WORLD 5 ms grid (0 = unvoiced)."""
    import torch
    from modules.rmvpe import RMVPE
    dev = ("cuda" if torch.cuda.is_available() else
           "mps" if torch.backends.mps.is_available() else "cpu")
    model = RMVPE(RMVPE_PT, is_half=False, device=dev)
    y16 = librosa.resample(y, orig_sr=SR, target_sr=16000)
    f0 = model.infer_from_audio(y16, thred=0.03)  # 10 ms hop
    t_src = np.arange(len(f0)) * 0.01
    n = int(len(y) / SR * FPS)
    t = np.arange(n) / FPS
    voiced = np.interp(t, t_src, (f0 > 0).astype(float)) > 0.5
    logf = np.interp(t, t_src[f0 > 0], np.log(f0[f0 > 0]))
    return np.where(voiced, np.exp(logf), 0.0)


def fill_gaps(mask, max_gap):
    """Close runs of False shorter than max_gap frames that sit between True runs."""
    out = mask.copy()
    idx = np.where(mask)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap:
            out[a:b] = True
    return out


def clean_f0(f0, min_island=0.06, max_dropout=0.045):
    """Bridge tracker dropouts, drop tiny voiced islands, fold octave glitches."""
    v = fill_gaps(f0 > 0, int(max_dropout * FPS))
    idx = np.where(f0 > 0)[0]
    out = np.where(v, np.exp(np.interp(np.arange(len(f0)), idx, np.log(f0[idx]))), 0.0)
    lab = np.cumsum(np.diff(np.r_[0, v.astype(int)]) == 1) * v
    for k in range(1, lab.max() + 1):
        idx = np.where(lab == k)[0]
        if len(idx) < min_island * FPS:
            out[idx] = 0
            continue
        cents = 1200 * np.log2(out[idx])
        med = ss.medfilt(cents, 41 if len(idx) > 41 else (len(idx) // 2) * 2 + 1)
        jump = np.round((cents - med) / 1200)
        out[idx] = 2 ** ((cents - 1200 * jump) / 1200)
    return out


# ---------------------------------------------------------------- 3. syllables

def segment(y, f0, min_len=0.09, pitch_cuts=False):
    """Split voiced vocal into syllable-sized (start, end) frame ranges."""
    n = len(f0)
    hop = int(SR / FPS)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0][:n]
    db = 20 * np.log10(rms + 1e-9)
    db = np.pad(db, (0, n - len(db)), constant_values=-120)
    active = (f0 > 0) & (db > np.percentile(db[f0 > 0], 95) - 30)
    active = fill_gaps(active, int(0.03 * FPS)) & (f0 > 0)

    cuts = set()
    sm = np.convolve(db, np.ones(9) / 9, "same")
    v = np.where(f0 > 0)[0]
    cents = np.interp(np.arange(n), v, 1200 * np.log2(f0[v]))
    # a moving average over ~1 vibrato period (200 ms) cancels vibrato but keeps note steps
    med = uniform_filter1d(cents, 41, mode="nearest")
    # onsets (consonants / re-attacks). On a held vibrato note the onset detector also fires
    # on the amplitude wobble, so keep an onset only if it comes with a dip or a pitch change.
    on = librosa.onset.onset_detect(y=y, sr=SR, hop_length=hop, backtrack=False,
                                    units="frames", delta=0.06, wait=int(0.09 * FPS))
    for i in on:
        lo, hi = max(0, i - 40), min(n, i + 40)
        dip = min(sm[lo:i + 1].max(), sm[i:hi].max()) - sm[max(0, i - 10):i + 10].min()
        pitch_move = abs(med[min(n - 1, i + 16)] - med[max(0, i - 16)])
        voicing_break = not (f0[max(0, i - 6):i + 6] > 0).all()
        if dip >= 5 or pitch_move >= 80 or voicing_break:
            cuts.add(int(i))
    # energy dips: a valley >= 6 dB below the peaks on both sides
    for i in ss.argrelmin(sm, order=12)[0]:
        left, right = sm[max(0, i - 40):i].max(initial=-120), sm[i:i + 40].max(initial=-120)
        if min(left, right) - sm[i] >= 6:
            cuts.add(int(i))
    # pitch steps between held notes (not vibrato): >= 0.8 semitone change within 120 ms
    # Off by default: the reference cover keeps melismas legato (one breath, gliding pitch)
    step = np.abs(med[24:] - med[:-24]) * pitch_cuts
    for i in ss.find_peaks(step, height=80, distance=int(0.12 * FPS))[0]:
        if active[i] and active[i + 24]:
            cuts.add(int(i + 12))

    segs = []
    i = 0
    while i < n:
        if not active[i]:
            i += 1
            continue
        j = i + 1
        while j < n and active[j] and j not in cuts:
            j += 1
        segs.append([i, j])
        i = j
    # merge slivers into the previous syllable when they touch, else drop them
    out = []
    for s, e in segs:
        if e - s < min_len * FPS:
            if out and s - out[-1][1] <= 1:
                out[-1][1] = e
            continue
        out.append([s, e])
    return out, db


# ---------------------------------------------------------------- 4. render

BANK_CLIPS = ["idle1", "idle2", "idle3", "yes1", "yes2", "yes3", "haggle1", "haggle2", "haggle3",
              "no1", "no2", "no3"]
ESSENCE = os.path.join(HERE, "ref", "essence.npz")
# The "essence" of the reference villager cover (see analyze_reference.py / README):
#   timbre   61% of its frames sound like `yes`, 26% `idle`, 10% `haggle`, ~0% grumpy `no`
#   shape    staccato: ~16% of each phrase is silence, syllables start within ~35 ms
#   colour   brighter (3-8 kHz) and less boomy (<300 Hz) than our v2; cleaner harmonics
#   level    villager sits ~0.65x the original singer's level in the mix
STYLE = dict(weights={"yes": 10, "idle": 2, "haggle": 0.5, "no": 0},
             attack=0.02, release=0.04, gap=0.05, gap_frac=0.12, legato_dip=4.0, aspirate=0.02,
             ap_scale=0.75, ap_max=0.45, scoop=1.0, end_fall=0.8, growl=0.15,
             flatten=False, follow_env=False, vowel=0.0, f0_smooth_hz=0.0)
# v4 "clean voice" (map_sounds.py / compare_style.py): their syllables have ~17 dB less
# subharmonic energy, half the pitch jitter, one loudness bump per word, pitch that follows the
# singer only (no clip intonation), and vowel colour that changes word to word.
CLEAN = dict(STYLE, aspirate=0.0, scoop=0.0, end_fall=0.0, growl=0.0, flatten=True,
             follow_env=True, vowel=0.25, f0_smooth_hz=6.0, ap_scale=0.4, ap_max=0.3)


def hum_bank(top_db=10, smooth=False):
    """Single-bump "hrmm" cores: the loudest contiguous region of each villager clip.

    Several clips are two-part ("hm-HMM"); used whole they sound like two notes per syllable.
    """
    bank = []
    for name in BANK_CLIPS:
        f0, sp, ap = vs.world(vs.load_villager(name), fixed_f0=105.0)
        e = np.convolve(vs.frame_energy(sp), np.ones(9) / 9, "same")
        pk = int(np.argmax(e))
        lo = hi = pk
        while lo > 0 and e[lo - 1] > e[pk] - top_db:
            lo -= 1
        while hi < len(e) - 1 and e[hi + 1] > e[pk] - top_db:
            hi += 1
        if hi - lo >= int(0.12 * FPS):
            core = sp[lo:hi + 1]
            bank.append((name.rstrip("0123456789"), lifter(core) if smooth else core, ap[lo:hi + 1]))
    return bank


def envelope(m, attack, release):
    env = np.ones(m)
    na, nr = min(m // 3, int(attack * FPS)), min(m // 3, int(release * FPS))
    env[:na] = np.linspace(0.25, 1, na)
    if nr:
        env[-nr:] = np.linspace(1, 0.25, nr)
    return env


def smooth_f0(f0, hz):
    """Low-pass the pitch track (keeps melody and ~5 Hz vibrato, removes tracker jitter)."""
    v = f0 > 0
    if hz <= 0 or v.sum() < 10:
        return f0
    idx = np.where(v)[0]
    lf = np.interp(np.arange(len(f0)), idx, np.log(f0[idx]))
    b, a = ss.butter(2, hz / (FPS / 2))
    return np.where(v, np.exp(ss.filtfilt(b, a, lf)), 0.0)


def lifter(sp, order=30):
    """Keep only the formant envelope (mel-cepstral order `order`): strips the harmonic ripple
    that rough villager clips / the singer's own pitch leave in the envelope (heard as growl)."""
    import pyworld as pw
    fft = (sp.shape[1] - 1) * 2
    return pw.decode_spectral_envelope(
        np.ascontiguousarray(pw.code_spectral_envelope(np.ascontiguousarray(sp), SR, order)), SR, fft)


def singer_envelope(voc, f0):
    """Formant envelope (log, liftered) of the singer, for vowel colour."""
    import pyworld as pw
    n = len(f0)
    t = np.arange(n) / FPS
    sp = pw.cheaptrick(voc[:int(n / FPS * SR)].astype(np.float64), np.where(f0 > 0, f0, 150.0)[:n],
                       t, SR)
    lsp = np.log(lifter(sp, 20) + 1e-12)
    return lsp - lsp[f0[:len(lsp)] > 0].mean(0)


def render_hum(f0, segs, db, octave=0, seed=0, style=STYLE, eq_db=None, singer_lsp=None):
    rng = np.random.default_rng(seed)
    import pyworld as pw
    nb = pw.get_cheaptrick_fft_size(SR) // 2 + 1
    n = len(f0)
    F0, SP, AP = np.zeros(n), np.full((n, nb), 1e-14), np.ones((n, nb))
    bank = [h for h in hum_bank(smooth=style["flatten"]) if style["weights"].get(h[0], 0) > 0]
    w = np.array([style["weights"][h[0]] for h in bank], float)
    w /= w.sum()
    eq = 10 ** (eq_db / 10) if eq_db is not None else np.ones(nb)
    freqs = np.linspace(0, SR / 2, nb)
    ref_db = np.percentile(db[f0 > 0], 90)
    prev_legato = False
    for k, (s, e) in enumerate(segs):
        # Follow the singer's own articulation: where the voice really stops (consonant,
        # breath) leave a clear gap; where it is legato, run straight into the next syllable.
        legato = False
        if k + 1 < len(segs):
            s2, e2 = segs[k + 1]
            room_ = s2 - s
            dip = min(db[s:e].max(), db[s2:e2].max()) - db[max(s, e - 10):s2 + 10].min()
            legato = s2 - e < 3 and dip < style["legato_dip"]
            if not legato:
                e = min(e, s + max(int(0.06 * FPS), room_ - max(int(style["gap"] * FPS),
                                                                  int(style["gap_frac"] * room_))))
        m = e - s
        _, sp, ap = bank[rng.choice(len(bank), p=w)]
        sp = vs.stretch_core(sp, m)
        if style["flatten"]:  # remove the clip's own loudness bumps: one bump per word
            fe = sp.sum(1, keepdims=True)
            sp = sp / fe * np.median(fe)
        if style["vowel"] > 0 and singer_lsp is not None:
            # one vowel colour per word (constant inside the syllable: a moving envelope
            # smears noise between the harmonics)
            sp = sp * np.exp(style["vowel"] * singer_lsp[s:e].mean(0))
        ap = np.minimum(vs.stretch_core(ap, m) * style["ap_scale"], style["ap_max"])
        na = int(style["aspirate"] * FPS)  # breathy "h" onset: the hit-like attack
        ap[:na, freqs > 1500] = 0.95
        seg_f0 = f0[s:e].copy()
        v = seg_f0 > 0
        seg_f0 = np.exp(np.interp(np.arange(m), np.where(v)[0], np.log(seg_f0[v])))
        t = np.arange(m) / FPS
        bend = -style["scoop"] * np.exp(-t / 0.035)                       # scoop in
        bend += -style["end_fall"] * np.clip((t - t[-1] + 0.07) / 0.07, 0, 1) ** 2  # "hm↘"
        seg_f0 *= 2 ** (octave + bend / 12)
        # per-syllable level from the singer, but a crisp envelope of our own
        if style["follow_env"]:  # loudness shape comes from the singer (as a voice model does)
            g_db = np.clip(uniform_filter1d(db[s:e], 7, mode="nearest") - ref_db, -30, 3)
        else:
            g_db = np.clip(np.median(db[s:e]) - ref_db, -20, 3)
        env = envelope(m, style["attack"] if not prev_legato else 0.008,
                       style["release"] if not legato else 0.008)
        prev_legato = legato
        gain = 10 ** (g_db / 10) * env ** 2
        F0[s:e], SP[s:e], AP[s:e] = seg_f0, sp * gain[:, None] * eq, ap
    y = vs.synth(F0, SP, AP, growl=style["growl"])
    # rumble below the lowest sung note is not voice (it measured as "growl" at f0/2)
    lo = 0.6 * F0[F0 > 0].min() if (F0 > 0).any() else 60.0
    return ss.sosfiltfilt(ss.butter(4, max(40.0, lo), "hp", fs=SR, output="sos"), y)


# ---------------------------------------------------------------- 5. mix

def room(y, wet=0.06, t60=0.45, seed=1):
    rng = np.random.default_rng(seed)
    n = int(t60 * SR)
    ir = rng.normal(0, 1, n) * np.exp(-6.9 * np.arange(n) / n)
    ir = ss.lfilter([1], [1, -0.6], ir)  # darker tail
    wet_sig = ss.fftconvolve(y, ir)[:len(y)]
    return y + wet * wet_sig / (np.abs(wet_sig).max() + 1e-9) * np.abs(y).max()


def active_rms(x):
    x = x if x.ndim == 1 else x.mean(1)
    fr = librosa.feature.rms(y=x, frame_length=2048, hop_length=512)[0]
    return np.sqrt(np.mean(fr[fr > np.percentile(fr, 60)] ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("song")
    ap.add_argument("--start", type=float, help="excerpt start (s); default: auto chorus")
    ap.add_argument("--dur", type=float, default=30.0)
    ap.add_argument("--octave", type=float, default=None,
                    help="octave shift for the hum; default: auto toward villager range")
    ap.add_argument("--vocal-gain", type=float, default=0.65, help="hum level vs original vocal")
    ap.add_argument("--out", default="out")
    ap.add_argument("--no-lead", action="store_true",
                    help="skip the lead/backing-vocal split (track the full vocal stem)")
    ap.add_argument("--backing-gain", type=float, default=0.0,
                    help="mix the original backing vocals back in (0 = villager only)")
    ap.add_argument("--no-eq", action="store_true", help="skip the reference-matched EQ curve")
    ap.add_argument("--style", choices=["clean", "essence"], default="clean",
                    help="clean = v4 voice-model-like hum (default); essence = v3")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override style parameters, e.g. --set ap_scale=0.4 scoop=0")
    ap.add_argument("--legato-dip", type=float, default=STYLE["legato_dip"],
                    help="dB dip in the singer's voice that counts as a stop (lower = choppier)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    name = os.path.splitext(os.path.basename(a.song))[0]
    start = a.start if a.start is not None else find_chorus(a.song, a.dur)
    print(f"excerpt: {start:.1f}s - {start + a.dur:.1f}s")
    work = os.path.join(HERE, "work", f"{name}_{start:.1f}_{a.dur:.0f}")
    clip, voc_p, inst_p = separate(a.song, start, a.dur, work)

    if not a.no_lead:
        voc_p, back_p = split_lead(voc_p, work)
    voc, _ = librosa.load(voc_p, sr=SR, mono=True)
    inst, _ = sf.read(inst_p)
    if not a.no_lead and a.backing_gain > 0:
        back, _ = sf.read(back_p)
        m = min(len(inst), len(back))
        inst = inst[:m] + a.backing_gain * (back[:m] if back.ndim == 2 else back[:m, None])
    orig, _ = sf.read(clip)
    f0 = clean_f0(vocal_f0(voc))
    segs, db = segment(voc, f0)
    med = np.median(f0[f0 > 0])
    # villager clips sit near 105 Hz; move by whole octaves only so the key is kept
    octave = a.octave if a.octave is not None else float(np.clip(np.round(np.log2(160 / med)), -1, 0))
    print(f"vocal median f0 {med:.0f} Hz, {len(segs)} syllables, octave shift {octave:+.0f}")

    eq_db = np.load(ESSENCE)["eq_db"] if os.path.exists(ESSENCE) and not a.no_eq else None
    style = dict(CLEAN if a.style == "clean" else STYLE, legato_dip=a.legato_dip)
    style.update({k: float(v) for k, v in (kv.split("=") for kv in a.set)})
    f0_r = smooth_f0(f0, style["f0_smooth_hz"])
    lsp = singer_envelope(voc, f0) if style["vowel"] > 0 else None
    hum = render_hum(f0_r, segs, db, octave=octave, seed=a.seed, style=style, eq_db=eq_db,
                     singer_lsp=lsp)
    hum = room(hum)
    n = min(len(hum), len(inst))
    hum = hum[:n] * a.vocal_gain * active_rms(voc) / (active_rms(hum) + 1e-9)
    inst = inst[:n] if inst.ndim == 2 else np.stack([inst[:n]] * 2, 1)
    mix = inst + hum[:, None]

    # trim the padding, fade, normalise
    lo = int(min(PAD, start) * SR)
    hi = lo + int(a.dur * SR)
    fade = np.ones(hi - lo)
    fade[:int(0.3 * SR)] = np.linspace(0, 1, int(0.3 * SR))
    fade[-int(1.5 * SR):] = np.linspace(1, 0, int(1.5 * SR))

    def fin(x):
        x = x[lo:hi] * (fade[:, None] if x.ndim == 2 else fade)
        return 0.95 * x / np.abs(x).max()

    tag = f"{a.out}/{name}_villager"
    os.makedirs(a.out, exist_ok=True)
    sf.write(f"{tag}_cover_premaster.wav", fin(mix), SR)
    # master: gentle glue compression + streaming loudness (-14 LUFS, -1 dBTP)
    master = "acompressor=threshold=-18dB:ratio=2:attack=15:release=200:makeup=1,loudnorm=I=-14:TP=-1:LRA=11"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", f"{tag}_cover_premaster.wav",
                    "-af", master, "-ar", str(SR), f"{tag}_cover.wav"], check=True)
    sf.write(f"{tag}_hum_only.wav", fin(hum), SR)
    sf.write(f"{a.out}/{name}_original_excerpt.wav", fin(orig), SR)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", f"{tag}_cover.wav",
                    "-b:a", "192k", f"{tag}_cover.mp3"], check=True)
    np.savez(os.path.join(work, "analysis.npz"), f0=f0, segs=np.array(segs), db=db)
    print("wrote", f"{tag}_cover.wav/.mp3", f"{tag}_hum_only.wav")


if __name__ == "__main__":
    main()
