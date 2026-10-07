"""Side-by-side "cleanliness" of villager vocals (theirs vs ours).

    .venv-conv/bin/python compare_style.py theirs=work/reference_29.5_31/reference_vocal_excerpt.wav \
        ours=out/final/rickroll_villager_hum_only.wav

per syllable (runs of sound, like map_sounds.syllables):
  single_bump%   one loudness peak per syllable ("one hum per word")
  jitter_c       micro pitch wobble around the smoothed contour
  aperiodicity   breath/noise share 0.5-4 kHz (WORLD D4C)
  subharm_db     energy at f0/2 relative to f0 (growl / roughness)
  timbre_jump    mel-cepstral distance between consecutive syllables (timbre consistency)
  inner_flux     spectral-envelope movement inside a syllable (texture "noise")
"""

import sys
import warnings

import librosa
import numpy as np

warnings.filterwarnings("ignore")
import map_sounds as ms  # noqa: E402


def subharmonic_db(y, f0, act):
    S = np.abs(librosa.stft(y, n_fft=4096, hop_length=int(ms.SR * ms.FP / 1000)))
    freqs = librosa.fft_frequencies(sr=ms.SR, n_fft=4096)
    vals = []
    for i in np.where(act & (f0 > 0))[0][::4]:
        if i >= S.shape[1]:
            break
        h = S[np.argmin(np.abs(freqs - f0[i])), i]
        sub = S[np.argmin(np.abs(freqs - f0[i] / 2)), i]
        vals.append(20 * np.log10((sub + 1e-9) / (h + 1e-9)))
    return float(np.median(vals)) if vals else np.nan


def analyse(path):
    y, _ = librosa.load(path, sr=ms.SR, duration=31)
    f0, mc, ap, e = ms.feats(y)
    segs = ms.syllables(e)
    rows, means = [], []
    for s, t in segs:
        sel = np.zeros(len(e), bool)
        sel[s:t] = True
        d, act = ms.describe(f0, mc, ap, e, sel)
        m = mc[act][:, 1:]
        means.append(m.mean(0))
        d["inner_flux"] = float(np.mean(np.linalg.norm(np.diff(m, axis=0), axis=1))) if len(m) > 2 else 0
        rows.append(d)
    on = e > np.percentile(e, 95) - 25
    jumps = [np.linalg.norm(a - b) for a, b in zip(means[:-1], means[1:])]
    med = lambda k: float(np.median([r[k] for r in rows]))
    return dict(syllables=len(rows), syll_ms=med("dur_ms"),
                single_bump=100 * np.mean([r["bumps"] == 1 for r in rows]),
                jitter_c=med("jitter_c"), aperiodicity=med("aperiodicity"),
                subharm_db=subharmonic_db(y, f0, on), timbre_jump=float(np.median(jumps)),
                inner_flux=med("inner_flux"))


def main():
    res = {}
    for arg in sys.argv[1:]:
        name, path = arg.split("=", 1)
        res[name] = analyse(path)
    keys = list(next(iter(res.values())))
    print("metric".ljust(16) + "".join(n.rjust(14) for n in res))
    for k in keys:
        print(k.ljust(16) + "".join(f"{r[k]:14.2f}" for r in res.values()))


if __name__ == "__main__":
    main()
