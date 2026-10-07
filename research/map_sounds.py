"""Label the Minecraft villager-family sound library and map a villager cover onto it.

    .venv-conv/bin/python map_sounds.py catalog
    .venv-conv/bin/python map_sounds.py map <vocal.wav> --name theirs [--words words.json]

catalog  -> ref/library/catalog.csv : every voice clip with entity / event label / acoustics
map      -> work/mapping_<name>.csv : every sung syllable of a villager vocal with its nearest
            library clip (timbre-over-time DTW), intonation match, and cleanliness metrics
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
import warnings

import librosa
import numpy as np
import pyworld as pw
from scipy.signal import find_peaks

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "ref", "library")
SR = 44100
FP = 5.0
NOT_VOICE = re.compile(r"drink|appeared|disappeared|reappeared|cast|fangs|horn|mirror|prepare|"
                       r"step|unfect|remedy")
# in-game event names (sounds.json) for the file stems
EVENT = {"idle": "ambient", "say": "ambient", "haggle": "trade", "hit": "hurt", "hurt": "hurt",
         "yes": "yes", "no": "no", "death": "death", "celebrate": "celebrate"}


def feats(y):
    """WORLD analysis -> f0, mel-cepstrum (timbre, pitch-independent), aperiodicity, energy."""
    y = y.astype(np.float64)
    f0, t = pw.harvest(y, SR, f0_floor=60, f0_ceil=800, frame_period=FP)
    f0_fill = np.where(f0 > 0, f0, 110.0)  # rough voices: keep envelope analysis stable
    sp = pw.cheaptrick(y, f0_fill, t, SR)
    ap = pw.d4c(y, f0_fill, t, SR)
    mc = pw.code_spectral_envelope(sp, SR, 24)
    e = 10 * np.log10(sp.sum(1) + 1e-20)
    freqs = np.linspace(0, SR / 2, sp.shape[1])
    ap_mid = ap[:, (freqs > 500) & (freqs < 4000)].mean(1)
    return f0, mc, ap_mid, e


def describe(f0, mc, ap_mid, e, sel=None):
    """Acoustic summary of a sound (or of frames `sel`)."""
    sel = np.ones(len(e), bool) if sel is None else sel
    act = sel & (e > e[sel].max() - 25)
    f, a, en = f0[act], ap_mid[act], e[act]
    v = f > 0
    sm = np.convolve(en, np.ones(7) / 7, "same")
    bumps = len(find_peaks(sm, prominence=4, distance=int(60 / FP))[0]) if len(sm) > 8 else 1
    cents = 1200 * np.log2(f[v]) if v.sum() > 4 else np.array([0.0])
    smooth = np.convolve(cents, np.ones(9) / 9, "same") if len(cents) > 9 else cents
    return dict(dur_ms=act.sum() * FP, voiced_pct=100 * v.mean() if len(v) else 0,
                f0_med=float(np.median(f[v])) if v.any() else 0.0,
                aperiodicity=float(a.mean()) if len(a) else 1.0,
                jitter_c=float(np.median(np.abs(cents - smooth))) if len(cents) > 9 else 0.0,
                bumps=max(1, bumps)), act


def contour(f0, act, n=16):
    """Normalised pitch shape (semitones re. median), resampled to n points."""
    f = f0[act]
    v = f > 0
    if v.sum() < 4:
        return np.zeros(n)
    st = 12 * np.log2(f[v] / np.median(f[v]))
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(st)), st)


def label_of(path):
    stem = os.path.basename(path)[:-4]
    entity, snd = stem.split("__")
    kind = re.sub(r"\d+$", "", snd)
    return entity, kind, EVENT.get(kind, kind)


def catalog():
    rows, cache = [], {}
    for p in sorted(glob.glob(os.path.join(LIB, "*.ogg"))):
        entity, kind, event = label_of(p)
        if NOT_VOICE.search(kind):
            continue
        y, _ = librosa.load(p, sr=SR)
        f0, mc, ap, e = feats(y)
        d, act = describe(f0, mc, ap, e)
        rows.append(dict(clip=os.path.basename(p)[:-4], entity=entity, sound=kind,
                         event=f"entity.{entity}.{event}", **{k: round(v, 2) for k, v in d.items()}))
        cache[rows[-1]["clip"]] = dict(mc=mc[act][:, 1:], f0c=contour(f0, act), mc_mean=mc[act][:, 1:].mean(0))
    with open(os.path.join(LIB, "catalog.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    np.save(os.path.join(LIB, "catalog_feats.npy"), cache, allow_pickle=True)
    print(f"{len(rows)} voice clips catalogued -> ref/library/catalog.csv")
    return rows


def syllables(e, min_ms=60, top_db=25):
    """Sung syllables = runs of frames within top_db of the loud level (one sound each)."""
    on = e > np.percentile(e, 95) - top_db
    out, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            if (j - i) * FP >= min_ms:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def dtw_cost(a, b):
    D, wp = librosa.sequence.dtw(X=a.T, Y=b.T, metric="euclidean")
    return float(D[-1, -1] / len(wp))


def map_vocal(path, name, words=None, start=0.0, dur=None, time_offset=0.0):
    lib = np.load(os.path.join(LIB, "catalog_feats.npy"), allow_pickle=True).item()
    meta = {r["clip"]: r for r in csv.DictReader(open(os.path.join(LIB, "catalog.csv")))}
    y, _ = librosa.load(path, sr=SR, offset=start, duration=dur)
    f0, mc, ap, e = feats(y)
    segs = syllables(e)
    wl = json.load(open(words)) if words else []
    rows = []
    for s, t in segs:
        sel = np.zeros(len(e), bool)
        sel[s:t] = True
        d, act = describe(f0, mc, ap, e, sel)
        m = mc[act][:, 1:]
        cont = contour(f0, act)
        # timbre-over-time match (DTW of mel-cepstra) against every library clip
        costs = {c: dtw_cost(m, v["mc"]) for c, v in lib.items()}
        best = sorted(costs, key=costs.get)[:3]
        shape = {c: float(np.corrcoef(cont, lib[c]["f0c"])[0, 1]) if cont.std() > 0.05 and
                 lib[c]["f0c"].std() > 0.05 else 0.0 for c in best}
        t0, t1 = (time_offset + start + s * FP / 1000, time_offset + start + t * FP / 1000)
        word = " ".join(w["text"] for w in wl if w["start"] < t1 and w["end"] > t0).strip()
        rows.append(dict(t_start=round(t0, 2), t_end=round(t1, 2), word=word,
                         best=best[0], best_label=meta[best[0]]["event"], best_cost=round(costs[best[0]], 2),
                         second=best[1], third=best[2], intonation_corr=round(shape[best[0]], 2),
                         **{k: round(v, 2) for k, v in d.items()}))
    out = os.path.join(HERE, "work", f"mapping_{name}.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} syllables mapped -> {os.path.relpath(out, HERE)}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["catalog", "map"])
    ap.add_argument("vocal", nargs="?")
    ap.add_argument("--name", default="vocal")
    ap.add_argument("--words")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--dur", type=float)
    ap.add_argument("--time-offset", type=float, default=0.0,
                    help="add to reported times (e.g. an excerpt that starts mid-song)")
    a = ap.parse_args()
    if a.cmd == "catalog":
        catalog()
    else:
        map_vocal(a.vocal, a.name, a.words, a.start, a.dur, a.time_offset)


if __name__ == "__main__":
    sys.exit(main())
