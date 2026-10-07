"""Per-burst nearest sound + do RVC villager models also turn consonants into "hit" bursts?"""

import glob
import json
import os
import sys

import librosa
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "analysis"))
import bursts as B  # noqa: E402

SR = B.SR


def groups():
    g = {}
    for p in glob.glob(f"{HERE}/ref/library/*.ogg") + glob.glob(f"{HERE}/ref/sfx/entity__player__attack__*.ogg") \
            + glob.glob(f"{HERE}/ref/sfx/damage__hit*.ogg"):
        name = os.path.basename(p)[:-4]
        y = librosa.load(p, sr=SR)[0]
        S, mel, e, flat, hf = B.frames(y)
        g.setdefault(name.rstrip("0123456789"), []).append(B.shape(mel[:, e > e.max() - 20]))
    return {k: np.mean(v, 0) for k, v in g.items()}


def per_burst(path, off, dur, G):
    y = librosa.load(path, sr=SR, mono=True, offset=off, duration=dur)[0]
    bs, mel, e, f0, hf = B.bursts(y)
    near = []
    for i, j in bs:
        s = B.shape(mel[:, i:j])
        near.append(min(G, key=lambda k: np.sqrt(np.mean((s - G[k]) ** 2))))
    tot = len(near)
    hist = {k: round(near.count(k) / max(1, tot), 2) for k in sorted(set(near), key=near.count, reverse=True)[:5]}
    return dict(bursts_per_min=tot / (dur / 60), nearest=hist)


def main():
    G = groups()
    res = {"theirs_full": per_burst(f"{HERE}/work/reference_full/vocals.wav", 0, 212, G),
           "theirs_excerpt": per_burst(f"{HERE}/work/reference_full/vocals.wav", 29.6, 31, G),
           "rick_excerpt": per_burst(f"{HERE}/work/rickroll_29.5_31/lead.wav", 2.0, 31, G)}
    for p in sorted(glob.glob(f"{HERE}/repro/rvc/*_-8.wav") + glob.glob(f"{HERE}/repro/rvc_fullvox/*.wav")):
        res[os.path.relpath(p, f"{HERE}/repro")] = per_burst(p, 0, 31, G)
    print(json.dumps(res, indent=1))
    json.dump(res, open(f"{HERE}/analysis/bursts_rvc.json", "w"), indent=1)


if __name__ == "__main__":
    main()
