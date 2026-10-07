"""Differential matched filter: hit/hurt samples present in THEIR stem but not in the ORIGINAL.

The original song's drums correlate with short noisy templates by chance (NCC ~0.6-0.7), so
absolute thresholds mislead. For each candidate peak in their stem we look up the same template
and rate at the aligned time in the original stem; a real inserted sample shows a large gap.
writes analysis/hit_diff.json
"""

import json
import os

import librosa
import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 16000
OFF = 0.1096
DEV = "cuda"
TPL = ["villager__hit1", "villager__hit2", "villager__hit3", "villager__hit4", "villager__death",
       "wandering_trader__hurt1", "wandering_trader__hurt2", "wandering_trader__hurt3", "wandering_trader__hurt4",
       "zombie_villager__hurt1", "zombie_villager__hurt2"]
SFX = ["damage__hit1", "damage__hit2", "damage__hit3", "entity__player__attack__weak1", "entity__player__attack__weak2",
       "entity__player__attack__strong1", "entity__player__attack__strong2", "entity__player__attack__crit1",
       "entity__player__attack__knockback1"]
RATES = [2 ** (k / 24) for k in range(-24, 25)]


def load(p):
    return librosa.load(p, sr=SR, mono=True)[0].astype(np.float64)


def ncc(x, t):
    n, m = len(x), len(t)
    xt = torch.tensor(x, device=DEV)
    tt = torch.tensor(t - t.mean(), device=DEV)
    nfft = 1 << int(np.ceil(np.log2(n + m)))
    num = torch.fft.irfft(torch.fft.rfft(xt, nfft) * torch.fft.rfft(torch.flip(tt, [0]), nfft), nfft)[m - 1:n]
    cs = torch.cumsum(torch.cat([torch.zeros(1, device=DEV, dtype=xt.dtype), xt]), 0)
    cs2 = torch.cumsum(torch.cat([torch.zeros(1, device=DEV, dtype=xt.dtype), xt ** 2]), 0)
    s1, s2 = cs[m:] - cs[:-m], cs2[m:] - cs2[:-m]
    var = torch.clamp(s2 - s1 ** 2 / m, min=1e-12)
    r = num / (torch.sqrt(var) * torch.linalg.norm(tt))
    loud = s2 / m > (cs2[-1] / n) * 1e-3
    return torch.where(loud, r, torch.zeros_like(r)).cpu().numpy()


def main():
    pairs = {"vocal": ("work/reference_full/vocals.wav", "work/original_full/vocals.wav"),
             "instrumental": ("work/reference_full/instrumental.wav", "work/original_full/instrumental.wav"),
             "mix": ("work/reference_full/clip.wav", "work/original_full/clip.wav")}
    res = {}
    for stem, (a, b) in pairs.items():
        X, Y = load(f"{HERE}/{a}"), load(f"{HERE}/{b}")
        found = []
        for name in TPL + SFX:
            folder = "ref/library" if name in TPL else "ref/sfx"
            t = librosa.load(f"{HERE}/{folder}/{name}.ogg", sr=SR)[0]
            t, _ = librosa.effects.trim(t, top_db=40)
            for rate in RATES:
                tr = librosa.resample(t, orig_sr=SR, target_sr=int(round(SR / rate))).astype(np.float64)
                rx, ry = ncc(X, tr), np.abs(ncc(Y, tr))
                cand = np.where(rx > 0.6)[0]
                if not len(cand):
                    continue
                # local maxima, 100 ms apart
                cand = cand[np.argsort(-rx[cand])]
                taken = []
                for i in cand:
                    if any(abs(i - j) < 0.1 * SR for j in taken):
                        continue
                    taken.append(i)
                    j = int(i - OFF * SR)
                    w = int(0.02 * SR)
                    orig = float(ry[max(0, j - w):j + w].max()) if 0 <= j < len(ry) else 0.0
                    found.append(dict(t=round(i / SR, 3), tpl=name, rate=round(rate, 3), ncc=round(float(rx[i]), 3),
                                      orig_same_spot=round(orig, 3), gap=round(float(rx[i]) - orig, 3)))
                    if len(taken) >= 20:
                        break
        found.sort(key=lambda d: -d["gap"])
        res[stem] = dict(n_gap_gt_0_25=sum(d["gap"] > 0.25 for d in found),
                         n_gap_gt_0_4=sum(d["gap"] > 0.4 for d in found), top=found[:25])
        print(stem, res[stem]["n_gap_gt_0_25"], res[stem]["n_gap_gt_0_4"], res[stem]["top"][:8], flush=True)
    json.dump(res, open(f"{HERE}/analysis/hit_diff.json", "w"), indent=1)


if __name__ == "__main__":
    main()
