"""Score generated villager vocals.

  villager_sim  cosine similarity of CAM++ speaker embeddings to the real villager clips
  wer           word error rate of Whisper transcription vs. the sung lyrics
  pitch_err     median |cents| between sung f0 and the melody (voiced frames)

Run with the Seed-VC venv (it has torch/transformers):
    third_party/seed-vc/.venv/bin/python evaluate.py songs/happy_birthday.json out/*_vocal.wav
"""

import glob
import json
import os
import re
import sys

import librosa
import numpy as np
import torch
import torchaudio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "third_party", "seed-vc"))
from modules.campplus.DTDNN import CAMPPlus  # noqa: E402


def load_campplus():
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("funasr/campplus", "campplus_cn_common.bin",
                           cache_dir=os.path.join(HERE, "third_party", "seed-vc", "checkpoints"))
    m = CAMPPlus(feat_dim=80, embedding_size=192)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval()


def embed(model, y16):
    feat = torchaudio.compliance.kaldi.fbank(torch.tensor(y16, dtype=torch.float32)[None],
                                             num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(0, keepdim=True)
    with torch.no_grad():
        e = model(feat[None])[0]
    return (e / e.norm()).numpy()


def wer(ref, hyp):
    r, h = ref, hyp
    d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        prev, d[0] = d.copy(), i
        for j in range(1, len(h) + 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(h)] / len(r)


def norm(s):
    return re.sub(r"[^a-z ]", "", s.lower().replace("-", " ")).split()


def pitch_error(y, sr, song):
    spb = 60.0 / song["bpm"]
    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C5"), sr=sr, frame_length=2048, hop_length=256)
    t = np.arange(len(f0)) * 256 / sr
    target = np.full(len(f0), np.nan)
    shift = song.get("vocal_transpose", 0)
    for n in song["notes"]:
        a, b = n["t"] * spb + 0.1, (n["t"] + n["b"]) * spb - 0.05  # skip glide-in
        target[(t >= a) & (t < b)] = librosa.midi_to_hz(n["m"] + shift)
    ok = ~np.isnan(f0) & ~np.isnan(target)
    if ok.sum() < 10:
        return float("nan"), 0.0
    c = 1200 * np.abs(np.log2(f0[ok] / target[ok]))
    c = np.minimum(c, np.abs(c - 1200))  # forgive octave errors from the tracker
    return float(np.median(c)), float(ok.sum() / (~np.isnan(target)).sum())


def phrases(song, sr, n):
    """Sample ranges of each phrase; a note of 2+ beats ends a phrase."""
    spb = 60.0 / song["bpm"]
    cuts = [song["notes"][0]["t"]] + [m["t"] for p, m in zip(song["notes"], song["notes"][1:])
                                      if p["b"] >= 2]
    edges = [int((c * spb - 0.25) * sr) for c in cuts] + [n]
    return list(zip(edges[:-1], edges[1:]))


def main():
    song = json.load(open(sys.argv[1]))
    files = sys.argv[2:]
    # syllables flagged "join" continue the word ("Hap"+"py")
    text = "".join(n["lyric"] + ("" if n.get("join") else " ") for n in song["notes"])
    ref_text = " ".join(norm(text))

    cam = load_campplus()
    vill = [librosa.load(f, sr=16000)[0] for f in sorted(glob.glob(os.path.join(HERE, "ref/villager/*.ogg")))
            if re.search(r"(idle|yes|no|haggle)", f)]
    v_emb = np.mean([embed(cam, y) for y in vill], 0)
    v_emb /= np.linalg.norm(v_emb)
    # baseline: how similar real villager clips are to their own centroid
    self_sim = np.mean([embed(cam, y) @ v_emb for y in vill])

    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-small", device="cpu",
                   model_kwargs={"cache_dir": os.path.join(HERE, "third_party", "seed-vc", "checkpoints", "hf_cache")})

    print(f"reference lyric: {ref_text}")
    print(f"real villager clips -> centroid sim: {self_sim:.3f}\n")
    print(f"{'file':52s} {'vill_sim':>8s} {'WER':>6s} {'pitch¢':>7s} {'voiced':>6s}  transcript")
    for f in files:
        y16, _ = librosa.load(f, sr=16000)
        sim = float(embed(cam, y16) @ v_emb)
        # transcribe phrase by phrase: Whisper tends to stop early on long sung clips
        hyp = " ".join(asr(y16[a:b].copy(), generate_kwargs={"language": "en", "task": "transcribe"})["text"]
                       for a, b in phrases(song, 16000, len(y16)))
        w = wer(ref_text.split(), norm(hyp))
        y, sr = librosa.load(f, sr=22050)
        pe, vr = pitch_error(y, sr, song)
        print(f"{os.path.relpath(f):52s} {sim:8.3f} {w:6.2f} {pe:7.1f} {vr:6.2f}  {hyp.strip()[:70]}")


if __name__ == "__main__":
    main()
