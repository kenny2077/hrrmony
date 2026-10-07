"""Is their vocal "one villager sample per syllable, re-pitched to Rick -8 st"?

Pitch-independent test on WORLD mel-cepstra (timbre trajectories), validated on a POSITIVE
CONTROL whose answer we know:

  control  = for every syllable of Rick's lead pick a random library voice clip, stretch it to the
             syllable, impose Rick's f0 -8 semitones (WORLD) -> we log the true clip per syllable
  For each vocal (control, theirs, ours, RVC outputs, Rick):
    - identification: DTW(mcep) of each syllable vs every library clip -> best clip, margin
    - reuse: DTW nearest-neighbour distance between syllables of the same vocal
  If the method recovers the control's clips, but theirs shows no confident matches and no reuse,
  the per-syllable-sample hypothesis is rejected (and vice versa).
writes analysis/reuse.json
"""

import csv
import glob
import json
import os
import sys

import librosa
import numpy as np
import pyworld as pw

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import map_sounds as ms  # noqa: E402
import villager_sing as vs  # noqa: E402

SR = 44100
FPS = 200
rng = np.random.default_rng(7)

meta = {r["clip"]: r for r in csv.DictReader(open(f"{HERE}/ref/library/catalog.csv"))}
LIB = np.load(f"{HERE}/ref/library/catalog_feats.npy", allow_pickle=True).item()
VOICE = [c for c in LIB if meta[c]["entity"] in ("villager", "wandering_trader", "pillager",
                                                  "vindication_illager", "evocation_illager",
                                                  "illusion_illager", "zombie_villager")]


def make_control():
    w = f"{HERE}/work/rickroll_29.5_31"
    d = np.load(f"{w}/analysis.npz")
    f0, segs = d["f0"], d["segs"]
    lo = 2 * FPS  # drop the 2 s pad
    n = len(f0)
    nb = pw.get_cheaptrick_fft_size(SR) // 2 + 1
    F0, SP, AP = np.zeros(n), np.full((n, nb), 1e-14), np.ones((n, nb))
    truth = []
    cache = {}
    for s, e in segs:
        c = VOICE[rng.integers(len(VOICE))]
        if c not in cache:
            y, _ = librosa.load(f"{HERE}/ref/library/{c}.ogg", sr=SR)
            y, _ = librosa.effects.trim(y, top_db=30)
            _, sp, ap = vs.world(y, fixed_f0=110.0)
            en = vs.frame_energy(sp)
            k = en > en.max() - 25
            cache[c] = (sp[k], ap[k])
        sp, ap = cache[c]
        m = e - s
        seg = f0[s:e]
        v = seg > 0
        if v.sum() < 3:
            continue
        seg = np.exp(np.interp(np.arange(m), np.where(v)[0], np.log(seg[v]))) * 2 ** (-8 / 12)
        F0[s:e], SP[s:e], AP[s:e] = seg, vs.stretch_frames(sp, m), vs.stretch_frames(ap, m)
        truth.append(((s - lo) / FPS, (e - lo) / FPS, c))
    y = pw.synthesize(F0, SP, AP, SR, 5.0)[lo * SR // FPS:(lo + 31 * FPS) * SR // FPS]
    return y.astype(np.float32), truth


CMN = os.environ.get("CMN") == "1"
if CMN:  # cepstral mean normalisation: remove each source's global EQ / channel colour
    _gm = np.concatenate([LIB[c]["mc"] for c in VOICE]).mean(0)
    LIB = {c: dict(v, mc=v["mc"] - _gm) for c, v in LIB.items()}


def syl_feats(y):
    f0, mc, ap, e = ms.feats(y)
    if CMN:
        loud = e > np.percentile(e, 97) - 30
        mc = mc.copy()
        mc[:, 1:] -= mc[loud, 1:].mean(0)
    out = []
    for s, t in ms.syllables(e):
        sel = np.zeros(len(e), bool)
        sel[s:t] = True
        _, act = ms.describe(f0, mc, ap, e, sel)
        m = mc[act][:, 1:]
        if len(m) >= 6:
            out.append((s / FPS, t / FPS, m))
    return out


def identify(sy):
    rows = []
    for t0, t1, m in sy:
        costs = sorted((ms.dtw_cost(m, LIB[c]["mc"]), c) for c in VOICE)
        rows.append(dict(t0=t0, t1=t1, best=costs[0][1], cost=costs[0][0],
                         margin=(costs[1][0] - costs[0][0]) / costs[0][0], top3=[c for _, c in costs[:3]]))
    return rows


def reuse(sy):
    nn = []
    for i, (_, _, a) in enumerate(sy):
        nn.append(min(ms.dtw_cost(a, b) for j, (_, _, b) in enumerate(sy) if j != i))
    return nn


def summarize(name, y, truth=None):
    sy = syl_feats(y)
    ids = identify(sy)
    nn = reuse(sy)
    r = dict(n_syllables=len(sy), id_cost_median=float(np.median([x["cost"] for x in ids])),
             id_margin_median=float(np.median([x["margin"] for x in ids])),
             frac_margin_gt_10pct=float(np.mean([x["margin"] > 0.10 for x in ids])),
             reuse_nn_cost_median=float(np.median(nn)),
             best_clip_hist=dict(sorted({c: sum(x["best"] == c for x in ids) for c in
                                         {x["best"] for x in ids}}.items(), key=lambda kv: -kv[1])[:6]))
    if truth:
        hit1 = hit3 = tot = 0
        for x in ids:
            mid = (x["t0"] + x["t1"]) / 2
            tc = [c for a, b, c in truth if a <= mid <= b]
            if tc:
                tot += 1
                hit1 += x["best"] == tc[0]
                hit3 += tc[0] in x["top3"]
        r.update(control_top1=hit1 / max(1, tot), control_top3=hit3 / max(1, tot), control_n=tot,
                 chance_top1=1 / len(VOICE))
    print(name, json.dumps(r), flush=True)
    return r, ids


def main():
    res = {}
    ctrl, truth = make_control()
    import soundfile as sf
    sf.write(f"{HERE}/analysis/control_sample_per_syllable.wav", ctrl, SR)
    res["CONTROL (known 1 sample/syllable, -8 st)"], _ = summarize("control", ctrl, truth)
    vocals = {"theirs": (f"{HERE}/work/reference_full/vocals.wav", 29.6),
              "rick_lead": (f"{HERE}/work/rickroll_29.5_31/lead.wav", 2.0),
              "ours_v3": (f"{HERE}/out/final/rickroll_villager_hum_only.wav", 0.0),
              "ours_v4": (f"{HERE}/out/final_v4/rickroll_villager_hum_only.wav", 0.0)}
    for p in sorted(glob.glob(f"{HERE}/repro/rvc/*.wav"))[:4]:
        vocals["rvc:" + os.path.basename(p)[:-4]] = (p, 0.0)
    for k, (p, off) in vocals.items():
        y = librosa.load(p, sr=SR, mono=True, offset=off, duration=31)[0]
        res[k], ids = summarize(k, y)
        if k == "theirs":
            res["theirs_per_syllable"] = [dict(t=round(x["t0"] + 29.5, 2), best=x["best"],
                                               label=meta[x["best"]]["event"], cost=round(x["cost"], 2),
                                               margin=round(x["margin"], 3), top3=x["top3"]) for x in ids]
    json.dump(res, open(f"{HERE}/analysis/reuse{'_cmn' if CMN else ''}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
