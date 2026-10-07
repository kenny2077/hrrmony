"""How faithful / recognisable is a villager cover?

  pitch     RMVPE f0 of the hum vs. the original vocal (octave-shift aware)
  rhythm    onset F1: onsets heard in the hum vs. onsets in the original vocal (+-70 ms)
  melody    chroma correlation of hum vs. original vocal after DTW alignment
  locate    "name that tune" by machine: subsequence-DTW the cover's chroma against every
            full song in the library; the cover is recognised if the best match is the right
            song at the right place. Run for the full cover and for the hum alone (melody only).

    .venv-conv/bin/python faithfulness.py input/rickroll.mp3 29.5 31 [--octave -1]
"""

import argparse
import glob
import os
import sys
import warnings

import librosa
import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import convert  # noqa: E402

SR = 22050
HOP = 2048  # ~93 ms chroma frames


def chroma(y):
    return librosa.feature.chroma_cens(y=y, sr=SR, hop_length=HOP)


def locate(query, refs):
    """Best subsequence-DTW match of `query` chroma in each reference song."""
    q = chroma(query)
    out = []
    for name, y in refs.items():
        D, wp = librosa.sequence.dtw(X=q, Y=chroma(y), metric="cosine", subseq=True)
        cost = D[-1, :] / q.shape[1]
        end = int(np.argmin(cost))
        _, wp_best = librosa.sequence.dtw(X=q, Y=chroma(y)[:, :end + 1], metric="cosine",
                                          subseq=True)
        start = wp_best[-1, 1]
        out.append((name, float(cost[end]), start * HOP / SR))
    return sorted(out, key=lambda r: r[1])


def onset_f1(ref, est, tol=0.07):
    ref, est = list(ref), list(est)
    hit = 0
    for t in ref:
        j = np.argmin(np.abs(np.array(est) - t)) if est else None
        if j is not None and abs(est[j] - t) <= tol:
            hit += 1
            est.pop(j)
    p = hit / max(1, hit + len(est))
    r = hit / max(1, len(ref))
    return 2 * p * r / max(1e-9, p + r), p, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("song")
    ap.add_argument("start", type=float)
    ap.add_argument("dur", type=float)
    ap.add_argument("--octave", type=float, default=-1)
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    name = os.path.splitext(os.path.basename(a.song))[0]
    work = os.path.join(HERE, "work", f"{name}_{a.start:.1f}_{a.dur:.0f}")
    lo = int(min(convert.PAD, a.start) * SR)

    # compare against the clean lead vocal when available (backing harmonies blur onsets/pitch)
    ref_voc = os.path.join(work, "lead.wav")
    ref_voc = ref_voc if os.path.exists(ref_voc) else os.path.join(work, "vocals.wav")
    voc = librosa.load(ref_voc, sr=SR)[0][lo:lo + int(a.dur * SR)]
    hum = librosa.load(f"{a.out}/{name}_villager_hum_only.wav", sr=SR)[0]
    cover = librosa.load(f"{a.out}/{name}_villager_cover.wav", sr=SR)[0]
    orig = librosa.load(f"{a.out}/{name}_original_excerpt.wav", sr=SR)[0]

    # pitch
    f_src = convert.vocal_f0(librosa.resample(voc, orig_sr=SR, target_sr=convert.SR))
    f_hum = convert.vocal_f0(librosa.resample(hum, orig_sr=SR, target_sr=convert.SR))
    n = min(len(f_src), len(f_hum))
    ok = (f_src[:n] > 0) & (f_hum[:n] > 0)
    c = np.abs(1200 * np.log2(f_hum[:n][ok] / (f_src[:n][ok] * 2 ** a.octave)))
    print(f"pitch : median error {np.median(c):.0f} cents, {np.mean(c < 50):.0%} within 50 c, "
          f"{np.mean(c < 100):.0%} within a semitone; "
          f"{ok.sum() / max(1, (f_src[:n] > 0).sum()):.0%} of sung frames hummed")

    # rhythm
    kw = dict(sr=SR, units="time", backtrack=False, delta=0.08)
    f1, p, r = onset_f1(librosa.onset.onset_detect(y=voc, **kw),
                        librosa.onset.onset_detect(y=hum, **kw))
    print(f"rhythm: onset F1 {f1:.2f} (precision {p:.2f}, recall {r:.2f}, +-70 ms)")

    # melody shape: chroma correlation after DTW alignment
    cv, ch = chroma(voc), chroma(hum)
    _, wp = librosa.sequence.dtw(X=cv, Y=ch, metric="cosine")
    corr = np.mean([np.corrcoef(cv[:, i], ch[:, j])[0, 1] for i, j in wp])
    shuffled = np.mean([np.corrcoef(cv[:, i], ch[:, j])[0, 1]
                        for i, j in zip(np.random.default_rng(0).permutation(cv.shape[1]),
                                        range(ch.shape[1]))])
    print(f"melody: chroma corr vs original vocal {corr:.2f} (shuffled baseline {shuffled:.2f})")

    # name-that-tune against every full song in input/
    refs = {os.path.basename(f): librosa.load(f, sr=SR)[0]
            for f in sorted(glob.glob(os.path.join(HERE, "input", "*.mp3")))}
    for label, q in [("original excerpt", orig), ("villager cover", cover),
                     ("villager hum only", hum)]:
        res = locate(q, refs)
        best = res[0]
        song_ok = best[0] == os.path.basename(a.song)
        place_ok = song_ok and abs(best[2] - a.start) < 3
        verdict = "right song, right place" if place_ok else (
            "right song, other repeat" if song_ok else "WRONG SONG")
        others = ", ".join(f"{r[0]} {r[1]:.3f}" for r in res[1:])
        print(f"locate[{label:17s}]: {best[0]} @ {best[2]:.1f}s cost {best[1]:.3f} -> {verdict}"
              f" | other songs: {others}")


if __name__ == "__main__":
    main()
