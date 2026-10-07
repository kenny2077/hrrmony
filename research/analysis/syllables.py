"""How are their syllables made?  (voice conversion vs pasted samples)

Excerpt 29.5-60.5 s (original time). Compares their villager vocal, our v3/v4 hum, and Rick's lead.
  - ASR: do words survive?  (VC keeps phonetic content; pasted hums do not)
  - pitch: offset vs Rick, and whether *micro* contour (vibrato, scoops) inside each syllable
    follows Rick (VC) or the template clip (pasting)
  - consonants: high-band noise where Rick sings fricatives / plosives
  - sample reuse: waveform NCC between their syllables (pasting reuses identical waveforms)
  - speaker embedding (CAM++) similarity to villager / trader / illager / Rick
writes analysis/syllables.json
"""

import glob
import json
import os
import re
import sys

import librosa
import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "third_party", "seed-vc"))
import convert  # noqa: E402  (vocal_f0 = RMVPE on the 5 ms grid)

SR = 44100
FPS = 200


def cut(p, off, dur=31.0, sr=SR):
    return librosa.load(p, sr=sr, mono=True, offset=off, duration=dur)[0]


def runs(mask, min_len=12):
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


def main():
    res = {}
    W = os.path.join(HERE, "work")
    stems = {
        "rick_lead": cut(f"{W}/rickroll_29.5_31/lead.wav", 2.0),
        "theirs": cut(f"{W}/reference_full/vocals.wav", 29.6),
        "ours_v3": cut(f"{HERE}/out/final/rickroll_villager_hum_only.wav", 0.0),
        "ours_v4": cut(f"{HERE}/out/final_v4/rickroll_villager_hum_only.wav", 0.0),
    }
    # ---------- ASR
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=0)
    res["asr"] = {k: asr(librosa.resample(y, orig_sr=SR, target_sr=16000),
                         return_timestamps=True, generate_kwargs={"language": "en", "task": "transcribe"})["text"]
                  for k, y in stems.items()}
    print(json.dumps(res["asr"], indent=1), flush=True)

    # ---------- pitch
    f0 = {k: convert.vocal_f0(y) for k, y in stems.items()}
    n = min(len(v) for v in f0.values())
    f0 = {k: v[:n] for k, v in f0.items()}
    rick = f0["rick_lead"]
    pitch = {}
    for k in ("theirs", "ours_v3", "ours_v4"):
        both = (rick > 0) & (f0[k] > 0)
        c = 1200 * np.log2(f0[k][both] / rick[both])
        off = float(np.median(c))
        # micro-contour: inside each common voiced run, correlation of detrended contours
        cors, devs = [], []
        for s, e in runs(both, 30):
            a = 1200 * np.log2(f0[k][s:e])
            b = 1200 * np.log2(rick[s:e])
            a, b = a - np.convolve(a, np.ones(25) / 25, "same"), b - np.convolve(b, np.ones(25) / 25, "same")
            a, b = a[12:-12], b[12:-12]
            if len(a) > 10 and a.std() > 1 and b.std() > 1:
                cors.append(np.corrcoef(a, b)[0, 1])
            d = 1200 * np.log2(f0[k][s:e] / rick[s:e]) - off
            devs.append(np.median(np.abs(d)))
        hist, edges = np.histogram(c, bins=np.arange(-2450, 1250, 100))
        pitch[k] = dict(offset_cents=off, offset_semitones=off / 100,
                        top_offsets=[(int(edges[i] + 50), int(hist[i])) for i in np.argsort(-hist)[:3]],
                        micro_contour_corr_median=float(np.median(cors)) if cors else None,
                        n_runs=len(cors),
                        abs_dev_from_constant_offset_c=float(np.median(devs)) if devs else None,
                        voiced_where_rick_voiced=float(both.sum() / max(1, (rick > 0).sum())))
    res["pitch"] = pitch
    print(json.dumps(pitch, indent=1), flush=True)

    # ---------- consonants: 4-10 kHz energy at Rick's unvoiced-but-loud frames vs vowel frames
    def band_db(y, lo, hi):
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=SR // FPS)) ** 2
        fr = librosa.fft_frequencies(sr=SR, n_fft=2048)
        return 10 * np.log10(S[(fr >= lo) & (fr < hi)].sum(0) + 1e-12)
    rhi = band_db(stems["rick_lead"], 4000, 10000)[:n]
    rall = band_db(stems["rick_lead"], 80, 10000)[:n]
    fric = (rick == 0) & (rhi > np.percentile(rhi, 85)) & (rhi > rall - 6)
    vowel = (rick > 0) & (rall > np.percentile(rall, 60))
    cons = {"n_fricative_frames": int(fric.sum())}
    for k in ("rick_lead", "theirs", "ours_v3", "ours_v4"):
        hi = band_db(stems[k], 4000, 10000)[:n]
        cons[k] = dict(hf_at_rick_fricatives_minus_hf_at_vowels_db=float(np.median(hi[fric]) - np.median(hi[vowel])))
    res["consonants"] = cons
    print(json.dumps(cons, indent=1), flush=True)

    # ---------- sample reuse among their syllables (waveform NCC, same-length windows)
    def reuse(y, f0k):
        rs = [(s, e) for s, e in runs(f0k > 0, 30)]
        segs = [y[int(s / FPS * SR):int(s / FPS * SR) + int(0.15 * SR)] for s, e in rs]
        segs = [g for g in segs if len(g) == int(0.15 * SR)]
        best = []
        for i, a in enumerate(segs):
            m = 0.0
            for j, b in enumerate(segs):
                if i == j:
                    continue
                r = np.correlate(a - a.mean(), b - b.mean(), "full")
                r /= (np.linalg.norm(a - a.mean()) * np.linalg.norm(b - b.mean()) + 1e-9)
                m = max(m, float(r.max()))
            best.append(m)
        return dict(n=len(segs), median_best_match=float(np.median(best)) if best else None,
                    frac_above_0_8=float(np.mean(np.array(best) > 0.8)) if best else None)
    res["reuse"] = {k: reuse(stems[k], f0[k]) for k in ("theirs", "ours_v3", "ours_v4", "rick_lead")}
    print(json.dumps(res["reuse"], indent=1), flush=True)

    # ---------- speaker embedding
    from modules.campplus.DTDNN import CAMPPlus
    import torchaudio
    cam = CAMPPlus(feat_dim=80, embedding_size=192)
    cam.load_state_dict(torch.load(os.path.join(HERE, "third_party/seed-vc/campplus_cn_common.bin"), map_location="cpu"))
    cam.eval()

    def emb(y16):
        f = torchaudio.compliance.kaldi.fbank(torch.tensor(y16)[None], num_mel_bins=80, dither=0, sample_frequency=16000)
        f = f - f.mean(0, keepdim=True)
        with torch.no_grad():
            e = cam(f[None])[0]
        return (e / e.norm()).numpy()
    groups = {}
    for p in glob.glob(os.path.join(HERE, "ref/library/*.ogg")):
        ent, snd = os.path.basename(p)[:-4].split("__")
        if re.search(r"drink|appeared|cast|fangs|horn|mirror|prepare|step|unfect|remedy", snd):
            continue
        groups.setdefault(ent, []).append(emb(librosa.load(p, sr=16000)[0]))
        groups.setdefault(f"{ent}.{re.sub(r'[0-9]+$', '', snd)}", []).append(groups[ent][-1])
    cents = {g: (np.mean(v, 0) / np.linalg.norm(np.mean(v, 0))) for g, v in groups.items()}
    cents["rick_lead"] = emb(librosa.resample(stems["rick_lead"], orig_sr=SR, target_sr=16000))
    sims = {}
    for k in ("theirs", "ours_v3", "ours_v4"):
        e = emb(librosa.resample(stems[k], orig_sr=SR, target_sr=16000))
        s = {g: float(e @ c) for g, c in cents.items()}
        sims[k] = dict(sorted(s.items(), key=lambda kv: -kv[1])[:8])
    res["speaker_sim_top"] = sims
    print(json.dumps(sims, indent=1), flush=True)
    json.dump(res, open(os.path.join(HERE, "analysis/syllables.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
