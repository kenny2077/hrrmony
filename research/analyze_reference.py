"""Reverse-engineer a reference villager cover against the original song.

    .venv-conv/bin/python analyze_reference.py ref/target/villager_rickroll_ai.mp3 \
        input/rickroll.mp3 29.5 31 --ours out/rickroll_villager_hum_only.wav

1. Align the reference to the original (onset-envelope cross-correlation).
2. Separate the reference's villager vocal for the same excerpt.
3. Compare it, and our hum, with the original lead vocal and with every villager clip type.
"""

import argparse
import glob
import os
import re
import subprocess
import sys
import warnings

import librosa
import numpy as np
import pyworld as pw
import soundfile as sf

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import convert  # noqa: E402

SR = 44100


def onset_env(y, sr):
    return librosa.onset.onset_strength(y=y, sr=sr, hop_length=441)  # 10 ms


def align(ref, orig):
    """Seconds to add to an original-song time to get the reference time."""
    a, b = onset_env(ref, 22050), onset_env(orig, 22050)
    a, b = (a - a.mean()) / a.std(), (b - b.mean()) / b.std()
    c = np.correlate(a, b, "full")
    lag = np.argmax(c) - (len(b) - 1)
    return lag * 441 / 22050, c.max() / len(b)


def world_feats(y):
    y = y.astype(np.float64)
    f0, t = pw.harvest(y, SR, f0_floor=60, f0_ceil=800, frame_period=5.0)
    sp = pw.cheaptrick(y, f0, t, SR)
    ap = pw.d4c(y, f0, t, SR)
    return f0, sp, ap


def describe(name, y, lead_f0=None, lead_env=None):
    """Timbre / articulation descriptors of a vocal stem."""
    f0 = convert.vocal_f0(y)
    _, sp, ap = world_feats(y)
    n = min(len(f0), len(sp))
    f0, sp, ap = f0[:n], sp[:n], ap[:n]
    e = 10 * np.log10(sp.sum(1) + 1e-20)
    act = e > np.percentile(e, 95) - 30
    v = (f0 > 0) & act
    freqs = np.linspace(0, SR / 2, sp.shape[1])
    lt = sp[act].mean(0)
    cent = (lt * freqs).sum() / lt.sum()
    band = lambda lo, hi: 10 * np.log10(lt[(freqs >= lo) & (freqs < hi)].sum() / lt.sum())
    ap_mid = ap[v][:, (freqs > 500) & (freqs < 4000)].mean() if v.any() else np.nan
    d = dict(stem=name, voiced_pct=100 * v.sum() / max(1, act.sum()),
             f0_med=np.median(f0[v]) if v.any() else np.nan,
             centroid=cent, low_0_300=band(0, 300), nasal_300_1k=band(300, 1000),
             mid_1k_3k=band(1000, 3000), high_3k_8k=band(3000, 8000),
             aperiodicity_mid=ap_mid)
    if lead_f0 is not None:
        m = min(n, len(lead_f0))
        ok = (f0[:m] > 0) & (lead_f0[:m] > 0)
        c = 1200 * np.log2(f0[:m][ok] / lead_f0[:m][ok])
        d["octave_vs_lead"] = np.median(c) / 1200
        d["pitch_err_c"] = np.median(np.abs(c - 1200 * np.round(np.median(c) / 1200)))
        le = lead_env[:m]
        d["env_corr_lead"] = np.corrcoef(e[:m], le)[0, 1]
        d["voiced_where_lead_voiced"] = 100 * ok.sum() / max(1, (lead_f0[:m] > 0).sum())
    return d, sp[act].mean(0)


def clip_ltas():
    """Long-term spectra of each official villager clip type."""
    out = {}
    for f in sorted(glob.glob(os.path.join(HERE, "ref/villager/*.ogg"))):
        y, _ = librosa.load(f, sr=SR)
        _, sp, _ = world_feats(y)
        e = 10 * np.log10(sp.sum(1) + 1e-20)
        kind = re.sub(r"\d", "", os.path.basename(f)[:-4])
        out.setdefault(kind, []).append(sp[e > e.max() - 20].mean(0))
    return {k: np.mean(v, 0) for k, v in out.items()}


def spec_dist(a, b):
    """Distance between two long-term spectra (shape only, level-normalised, log-bark-ish)."""
    la, lb = np.log(a + 1e-20), np.log(b + 1e-20)
    k = np.hanning(15); k /= k.sum()
    la, lb = np.convolve(la, k, "same"), np.convolve(lb, k, "same")
    sel = slice(5, 400)  # ~100 Hz .. 8.6 kHz
    la, lb = la[sel] - la[sel].mean(), lb[sel] - lb[sel].mean()
    return float(np.sqrt(np.mean((la - lb) ** 2)))


def articulation(y, sr=SR):
    """Staccato-ness: % silence inside phrases, syllable length, attack time."""
    from scipy.ndimage import binary_closing
    hop = int(sr * 0.005)
    db = 20 * np.log10(librosa.feature.rms(y=y, frame_length=1024, hop_length=hop)[0] + 1e-9)
    on = db > np.percentile(db, 95) - 25
    phrase = binary_closing(on, structure=np.ones(80))
    runs, rises, i = [], [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            seg = db[i:j]
            k = int(np.argmax(seg))
            runs.append((j - i) * 0.005)
            rises.append(np.argmax(seg[:k + 1] > seg[k] - 3) * 5.0)
            i = j
        else:
            i += 1
    runs = np.array(runs)
    return dict(gap_pct=100 * (phrase & ~on).sum() / phrase.sum(),
                syll_ms=1000 * np.median(runs[runs > 0.04]), attack_ms=np.median(rises))


def clip_type_mix(y, sr=SR):
    """Share of frames whose MFCCs are closest (diag-Gaussian) to each villager clip type."""
    def mf(x):
        return (librosa.feature.mfcc(y=x, sr=sr, n_mfcc=20, hop_length=1024)[1:],
                librosa.feature.rms(y=x, hop_length=1024)[0])
    types = {}
    for f in sorted(glob.glob(os.path.join(HERE, "ref/villager/*.ogg"))):
        x, _ = librosa.load(f, sr=sr)
        m, r = mf(x)
        types.setdefault(re.sub(r"\d", "", os.path.basename(f)[:-4]), []).append(m[:, r > r.max() * 0.1])
    stats = {k: (np.concatenate(v, 1).mean(1), np.concatenate(v, 1).var(1) + 1e-3)
             for k, v in types.items()}
    m, r = mf(y)
    m = m[:, r > np.percentile(r, 95) * 0.1]
    ll = np.stack([-0.5 * (((m - mu[:, None]) ** 2) / var[:, None] + np.log(var[:, None])).sum(0)
                   for mu, var in stats.values()])
    lab = np.argmax(ll, 0)
    return {k: round(100 * np.mean(lab == i)) for i, k in enumerate(stats)}


def calibrate_eq(ref_y, ours_y, path, max_db=6.0):
    """EQ curve (dB per WORLD bin) that moves our long-term spectrum onto the reference's."""
    lt = []
    for y in (ref_y, ours_y):
        _, sp, _ = world_feats(y)
        e = 10 * np.log10(sp.sum(1) + 1e-20)
        x = sp[e > np.percentile(e, 95) - 25].mean(0)
        lt.append(10 * np.log10(x / x.sum()))
    k = np.hanning(25); k /= k.sum()
    eq = np.clip(np.convolve(lt[0] - lt[1], k, "same"), -max_db, max_db)
    np.savez(path, eq_db=eq)
    freqs = np.linspace(0, SR / 2, len(eq))
    print("EQ written:", ", ".join(f"{int(f)}Hz {eq[np.argmin(np.abs(freqs - f))]:+.1f}dB"
                                    for f in (150, 300, 600, 1000, 2000, 3000, 5000, 8000)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("original")
    ap.add_argument("start", type=float)
    ap.add_argument("dur", type=float)
    ap.add_argument("--ours", nargs="*", default=[])
    ap.add_argument("--calibrate", help="hum wav to fit ref/essence.npz EQ against")
    a = ap.parse_args()

    ref_full, _ = librosa.load(a.reference, sr=22050)
    orig_full, _ = librosa.load(a.original, sr=22050)
    off, strength = align(ref_full, orig_full)
    print(f"alignment: reference = original {off:+.2f}s (xcorr {strength:.2f})")

    work = os.path.join(HERE, "work", f"reference_{a.start:.1f}_{a.dur:.0f}")
    _, rvoc_p, rinst_p = convert.separate(a.reference, a.start + off, a.dur, work)
    owork = os.path.join(HERE, "work", f"{os.path.splitext(os.path.basename(a.original))[0]}"
                                       f"_{a.start:.1f}_{a.dur:.0f}")
    lo = int(convert.PAD * SR)
    cut = lambda p: librosa.load(p, sr=SR, mono=True)[0][lo:lo + int(a.dur * SR)]
    lead = cut(os.path.join(owork, "lead.wav"))
    rvoc = cut(rvoc_p)
    # same backing track? correlate the two separated instrumentals
    ri, oi = cut(rinst_p), cut(os.path.join(owork, "instrumental.wav"))
    print(f"instrumental corr (reference vs original): {np.corrcoef(ri, oi)[0, 1]:.2f}")
    print(f"vocal/instrumental RMS ratio: reference {convert.active_rms(rvoc) / convert.active_rms(ri):.2f}"
          f", original {convert.active_rms(lead) / convert.active_rms(oi):.2f}")
    sf.write(os.path.join(work, "reference_vocal_excerpt.wav"), rvoc, SR)

    if a.calibrate:
        calibrate_eq(rvoc, librosa.load(a.calibrate, sr=SR)[0][:int(a.dur * SR)],
                     os.path.join(HERE, "ref", "essence.npz"))
        return
    lead_d, lead_lt = describe("original lead", lead)
    lead_f0 = convert.vocal_f0(lead)
    _, lsp, _ = world_feats(lead)
    lead_env = 10 * np.log10(lsp.sum(1) + 1e-20)
    rows = [lead_d]
    lts = {"original lead": lead_lt}
    for name, y in [("reference villager", rvoc)] + [
            (os.path.relpath(p, HERE), librosa.load(p, sr=SR)[0][:int(a.dur * SR)]) for p in a.ours]:
        d, lt = describe(name, y, lead_f0, lead_env)
        d.update(articulation(y))
        d.update({f"type_{k}%": v for k, v in clip_type_mix(y).items()})
        rows.append(d)
        lts[name] = lt

    keys = ["voiced_pct", "f0_med", "octave_vs_lead", "pitch_err_c", "voiced_where_lead_voiced",
            "env_corr_lead", "centroid", "low_0_300", "nasal_300_1k", "mid_1k_3k", "high_3k_8k",
            "aperiodicity_mid", "gap_pct", "syll_ms", "attack_ms"] + sorted(
            k for k in rows[1] if k.startswith("type_"))
    print("\n" + "metric".ljust(26) + "".join(r["stem"][:24].rjust(26) for r in rows))
    for k in keys:
        print(k.ljust(26) + "".join(
            (f"{r[k]:26.2f}" if k in r and np.isfinite(r[k]) else " " * 25 + "-") for r in rows))

    clips = clip_ltas()
    print("\nspectral distance to villager clip types (lower = closer)")
    print("stem".ljust(26) + "".join(k.rjust(9) for k in clips))
    for name, lt in lts.items():
        print(name[:24].ljust(26) + "".join(f"{spec_dist(lt, c):9.2f}" for c in clips.values()))


if __name__ == "__main__":
    main()
