"""Production / mix forensics of the reference cover vs the original song (excerpt 29.5-60.5 s).

  - is their instrumental the original recording? (sample-level NCC with lag search, level diff)
  - does Rick's vocal bleed into their mix? (residual after removing the original instrumental)
  - stereo width (side/mid) of mix and vocal stem; vocal-to-instrumental level
  - reverb tail (decay rate after syllable offsets), dynamics (crest factor, loudness range)
writes analysis/mix.json
"""

import json
import os
import subprocess

import librosa
import numpy as np
import scipy.signal as ss

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 44100
W = os.path.join(HERE, "work")


def st(p, off, dur=31.0):
    y, _ = librosa.load(p, sr=SR, mono=False, offset=off, duration=dur)
    return y if y.ndim == 2 else np.stack([y, y])


def best_lag(a, b, maxlag=4410):
    r = ss.correlate(a, b, mode="full", method="fft")
    mid = len(b) - 1
    seg = r[mid - maxlag:mid + maxlag + 1]
    i = int(np.argmax(np.abs(seg)))
    lag = i - maxlag
    return lag, float(seg[i] / (np.linalg.norm(a) * np.linalg.norm(b)))


def width(y):
    m, s = (y[0] + y[1]) / 2, (y[0] - y[1]) / 2
    return float(10 * np.log10((s ** 2).sum() / ((m ** 2).sum() + 1e-12) + 1e-12))


def decay_db_per_s(y, sr=SR):
    """Median slope of the energy envelope in the 30-250 ms after each syllable offset."""
    hop = 220
    e = 20 * np.log10(librosa.feature.rms(y=y, frame_length=1024, hop_length=hop)[0] + 1e-9)
    on = e > np.percentile(e, 95) - 20
    slopes = []
    for i in np.where(on[:-1] & ~on[1:])[0]:
        seg = e[i + 6:i + 50]
        if len(seg) > 20 and seg.max() - seg.min() > 3:
            slopes.append(np.polyfit(np.arange(len(seg)) * hop / sr, seg, 1)[0])
    return float(np.median(slopes)) if slopes else None, len(slopes)


def lufs(path):
    out = subprocess.run(["ffmpeg", "-hide_banner", "-i", path, "-af", "ebur128", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    I = [l for l in out.splitlines() if l.strip().startswith("I:")]
    LRA = [l for l in out.splitlines() if l.strip().startswith("LRA:")]
    return (I[-1].split()[1] if I else None), (LRA[-1].split()[1] if LRA else None)


def main():
    res = {}
    their_mix = st(f"{W}/reference_full/clip.wav", 29.6)
    their_ins = st(f"{W}/reference_full/instrumental.wav", 29.6)
    their_voc = st(f"{W}/reference_full/vocals.wav", 29.6)
    orig_mix = st(os.path.join(HERE, "input/rickroll.mp3"), 29.5)
    orig_ins = st(f"{W}/rickroll_29.5_31/instrumental.wav", 2.0)
    rick = st(f"{W}/rickroll_29.5_31/lead.wav", 2.0)

    # 1. instrumental identity
    lag, r = best_lag(their_ins.mean(0), orig_ins.mean(0))
    lagm, rm = best_lag(their_mix.mean(0), orig_mix.mean(0))
    g = np.dot(their_ins.mean(0), np.roll(orig_ins.mean(0), lag)) / np.dot(orig_ins.mean(0), orig_ins.mean(0))
    M1 = np.log(librosa.feature.melspectrogram(y=their_ins.mean(0), sr=SR, n_mels=64) + 1e-9)
    M2 = np.log(librosa.feature.melspectrogram(y=orig_ins.mean(0), sr=SR, n_mels=64) + 1e-9)
    n = min(M1.shape[1], M2.shape[1])
    melcorr = float(np.corrcoef(M1[:, :n].ravel(), M2[:, :n].ravel())[0, 1])
    res["instrumental"] = dict(sample_lag=lag, lag_ms=1000 * lag / SR, wave_ncc=r, gain=float(g),
                               mix_vs_mix_lag=lagm, mix_vs_mix_ncc=rm, logmel_corr=melcorr)
    # per-band coherence at best lag (is it the same master, or a remake/re-encode?)
    a = their_ins.mean(0)
    b = np.roll(orig_ins.mean(0), lag)
    f, coh = ss.coherence(a, b, fs=SR, nperseg=4096)
    res["instrumental"]["coherence_by_band"] = {f"{lo}-{hi}Hz": float(coh[(f >= lo) & (f < hi)].mean())
                                                for lo, hi in [(30, 150), (150, 500), (500, 2000),
                                                               (2000, 6000), (6000, 16000)]}

    # 2. Rick bleed: residual of their mix after subtracting the aligned, gain-matched original
    #    instrumental, projected on Rick's lead
    rmono = rick.mean(0)
    resid = their_mix.mean(0) - g * b
    lag_r, r_r = best_lag(resid, rmono, maxlag=2205)
    lag_v, r_v = best_lag(their_voc.mean(0), rmono, maxlag=2205)
    res["rick_bleed"] = dict(resid_vs_rick_ncc=r_r, their_vocal_vs_rick_ncc=r_v, lag_ms=1000 * lag_v / SR,
                             note="NCC of an independent signal is ~0.0x; same recording would be >0.3")

    # 3. levels, width, dynamics, reverb
    rms = lambda x: float(np.sqrt(np.mean(x ** 2)))
    res["levels"] = dict(
        their_vocal_to_instr_db=20 * np.log10(rms(their_voc) / rms(their_ins)),
        orig_lead_to_instr_db=20 * np.log10(rms(rick) / rms(orig_ins)),
        their_mix_rms_dbfs=20 * np.log10(rms(their_mix)), orig_mix_rms_dbfs=20 * np.log10(rms(orig_mix)))
    res["width_side_minus_mid_db"] = dict(their_mix=width(their_mix), orig_mix=width(orig_mix),
                                          their_vocal=width(their_voc), rick_lead=width(rick),
                                          their_instr=width(their_ins), orig_instr=width(orig_ins))
    crest = lambda x: float(20 * np.log10(np.abs(x).max() / rms(x)))
    res["crest_db"] = dict(their_vocal=crest(their_voc.mean(0)), rick_lead=crest(rmono))
    ours = {k: librosa.load(p, sr=SR, mono=True, duration=31)[0] for k, p in
            [("ours_v3", f"{HERE}/out/final/rickroll_villager_hum_only.wav"),
             ("ours_v4", f"{HERE}/out/final_v4/rickroll_villager_hum_only.wav")]}
    res["decay_db_per_s_after_offsets"] = {k: decay_db_per_s(v) for k, v in
                                           [("their_vocal", their_voc.mean(0)), ("rick_lead", rmono)] + list(ours.items())}
    res["lufs_I_LRA"] = {"their_full": lufs(f"{HERE}/ref/target/villager_rickroll_ai.mp3"),
                         "orig_full": lufs(f"{HERE}/input/rickroll.mp3"),
                         "ours_v4_cover": lufs(f"{HERE}/out/final_v4/rickroll_villager_cover.wav")}
    print(json.dumps(res, indent=1, default=float))
    json.dump(res, open(os.path.join(HERE, "analysis/mix.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
