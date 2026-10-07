"""The "hit sound" test, part 2: unvoiced noise bursts inside their villager vocal.

Hypothesis: a villager-trained voice-conversion model renders Rick's consonants (s, t, k, g, h,
"hurt", "desert"...) as short noisy bursts in villager timbre - and the only noisy sounds a
villager has are its hit/hurt clips, so the bursts sound like "villager being hit".

  1. detect unvoiced, loud, noisy bursts in their full vocal stem
  2. are they aligned with Rick's consonants (vs a time-shuffled baseline)?
  3. which library sound do the bursts resemble (log-mel spectral shape)?
  4. same detector on our vocals
writes analysis/bursts.json, analysis/their_bursts.wav, analysis/bursts_spec.png
"""

import glob
import json
import os
import sys

import librosa
import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "third_party", "seed-vc"))
import convert  # noqa: E402

SR = 44100
HOP = 441  # 10 ms
OFF = 0.1096  # reference time = original time + OFF (onset xcorr + sample-level instrumental lag)
MEL = librosa.filters.mel(sr=SR, n_fft=2048, n_mels=48, fmin=80, fmax=12000)


def frames(y):
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP)) ** 2
    mel = MEL @ S
    e = 10 * np.log10(S.sum(0) + 1e-12)
    flat = librosa.feature.spectral_flatness(S=np.sqrt(S))[0]
    fr = librosa.fft_frequencies(sr=SR, n_fft=2048)
    hf = 10 * np.log10(S[fr > 3000].sum(0) + 1e-12) - e  # share of energy above 3 kHz
    return S, mel, e, flat, hf


def f0_10ms(y):
    f = convert.vocal_f0(y)  # 5 ms grid
    return f[::2]


def runs(mask, min_len=3):
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            if j - i >= min_len:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def bursts(y):
    S, mel, e, flat, hf = frames(y)
    f0 = f0_10ms(y)
    n = min(len(e), len(f0))
    loud = e[:n] > np.percentile(e[:n], 97) - 30
    noisy = (f0[:n] == 0) & loud & (hf[:n] > -12)
    return runs(noisy, 3), mel[:, :n], e[:n], f0[:n], hf[:n]


def shape(mel_cols):
    v = np.log(mel_cols.mean(1) + 1e-12)
    return v - v.mean()


def main():
    res = {}
    theirs = librosa.load(f"{HERE}/work/reference_full/vocals.wav", sr=SR, mono=True)[0]
    rick = librosa.load(f"{HERE}/work/original_full/vocals.wav", sr=SR, mono=True)[0]
    rick = np.concatenate([np.zeros(int(OFF * SR), np.float32), rick])  # into reference time
    b_t, mel_t, e_t, f0_t, hf_t = bursts(theirs)
    b_r, mel_r, e_r, f0_r, hf_r = bursts(rick)
    dur_min = len(theirs) / SR / 60
    res["their_bursts"] = dict(count=len(b_t), per_min=len(b_t) / dur_min,
                               median_ms=float(np.median([(j - i) * 10 for i, j in b_t])),
                               share_of_loud_time=float(sum(j - i for i, j in b_t) /
                                                        (e_t > np.percentile(e_t, 97) - 30).sum()))
    res["rick_consonant_bursts"] = dict(count=len(b_r), per_min=len(b_r) / dur_min)

    # 2. alignment with Rick's consonants (+-50 ms), vs circularly shifted baseline
    rc = np.zeros(len(e_t) + 200, bool)
    for i, j in b_r:
        rc[max(0, i - 5):j + 5] = True

    def hit_rate(bs, shift=0):
        return float(np.mean([rc[((i + shift) % len(e_t)):((j + shift) % len(e_t)) + 1].any()
                              if (i + shift) % len(e_t) < (j + shift) % len(e_t) else False for i, j in bs]))
    base = [hit_rate(b_t, s) for s in range(300, 6000, 397)]
    res["alignment"] = dict(frac_their_bursts_on_rick_consonants=hit_rate(b_t),
                            shuffled_baseline_mean=float(np.mean(base)), shuffled_baseline_max=float(np.max(base)))

    # 3. what do the bursts sound like? log-mel shape distance to library clip groups
    groups = {}
    for p in glob.glob(f"{HERE}/ref/library/*.ogg") + glob.glob(f"{HERE}/ref/sfx/entity__player__attack__*.ogg") \
            + glob.glob(f"{HERE}/ref/sfx/damage__hit*.ogg"):
        name = os.path.basename(p)[:-4]
        if name.startswith("mob__"):
            continue
        y = librosa.load(p, sr=SR)[0]
        S, mel, e, flat, hf = frames(y)
        loud = e > e.max() - 20
        g = name.rstrip("0123456789")
        groups.setdefault(g, []).append(shape(mel[:, loud]))
    burst_shape = shape(np.concatenate([mel_t[:, i:j] for i, j in b_t], 1))
    rick_shape = shape(np.concatenate([mel_r[:, i:j] for i, j in b_r], 1))
    voiced_shape = shape(mel_t[:, (f0_t > 0) & (e_t > np.percentile(e_t, 97) - 30)])
    dist = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    allg = {g: np.mean(v, 0) for g, v in groups.items()}
    allg["RICK consonants"] = rick_shape
    allg["THEIR voiced villager"] = voiced_shape
    res["burst_nearest_sounds"] = sorted(((g, round(dist(burst_shape, s), 3)) for g, s in allg.items()),
                                         key=lambda x: x[1])[:12]
    res["burst_dist_to_villager_hit"] = dist(burst_shape, allg["villager__hit"])
    res["rick_consonant_dist_to_villager_hit"] = dist(rick_shape, allg["villager__hit"])

    # 4. our vocals, same detector
    for k, p in [("ours_v3", f"{HERE}/out/final/rickroll_villager_hum_only.wav"),
                 ("ours_v4", f"{HERE}/out/final_v4/rickroll_villager_hum_only.wav")]:
        y = librosa.load(p, sr=SR, mono=True)[0]
        b, *_ = bursts(y)
        res[f"{k}_bursts_per_min"] = len(b) / (len(y) / SR / 60)
    their_ex = theirs[int(29.6 * SR):int(60.6 * SR)]
    b, *_ = bursts(their_ex)
    res["theirs_excerpt_bursts_per_min"] = len(b) / (31 / 60)

    # listening snippet: 12 bursts (with 60 ms context each side) separated by 250 ms silence
    pieces = []
    for i, j in b_t[::max(1, len(b_t) // 12)][:12]:
        pieces += [theirs[max(0, (i - 6) * HOP):(j + 6) * HOP], np.zeros(int(0.25 * SR), np.float32)]
    sf.write(f"{HERE}/analysis/their_bursts.wav", np.concatenate(pieces), SR)
    res["burst_examples_s"] = [(round(i / 100, 2), (j - i) * 10) for i, j in b_t[:40]]

    # spectrogram: chorus 43-50 s ref time with their bursts (red) and Rick consonants (cyan)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t0, t1 = 43.0, 50.0
    fig, ax = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    for a, y, bs, nm in [(ax[0], theirs, b_t, "their villager vocal"), (ax[1], rick, b_r, "Rick vocal (shifted to ref time)")]:
        seg = y[int(t0 * SR):int(t1 * SR)]
        D = librosa.amplitude_to_db(np.abs(librosa.stft(seg, n_fft=2048, hop_length=256)), ref=np.max)
        a.imshow(D, origin="lower", aspect="auto", extent=[t0, t1, 0, SR / 2], vmin=-70, cmap="magma")
        a.set_ylim(0, 10000)
        for i, j in bs:
            if t0 <= i / 100 <= t1:
                a.axvspan(i / 100, j / 100, color="cyan" if a is ax[1] else "red", alpha=0.3)
        a.set_title(nm)
    plt.tight_layout()
    plt.savefig(f"{HERE}/analysis/bursts_spec.png", dpi=60)
    print(json.dumps(res, indent=1, default=float))
    json.dump(res, open(f"{HERE}/analysis/bursts.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
