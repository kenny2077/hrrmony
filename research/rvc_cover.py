"""Villager cover via RVC voice conversion: the reverse-engineered recipe of the reference cover.

    .venv-conv/bin/python rvc_cover.py input/song.mp3 [--start 29.5 --dur 31] [--shift -8]

1. Mel-Band RoFormer -> full vocal stem (lead + backing) and instrumental
2. RVC villager model (r3gm/villager), rmvpe pitch, index 0.75, protect 0.33, shift -8 st
   (consonants come out as the villager "hurt" bursts; vowels as one clean hum per syllable)
3. vocal post: 4th-order high-pass 120 Hz, -8 dB high shelf above 5 kHz, mono, dry, no compressor
4. mix ~4 dB under the instrumental, master to -16 LUFS (light, no heavy limiting)

Runs on the GPU laptop. Needs `.venv-rvc` (infer-rvc-python) next to `.venv-conv`.
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import scipy.signal as ss
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import convert  # noqa: E402  (separation helpers)

SR = 44100
MODEL = os.path.join(HERE, "models", "rvc", "r3gm_villager")
RVC_PY = os.path.join(HERE, ".venv-rvc", "bin", "python")

RVC_SNIPPET = r"""
import shutil, sys, warnings
warnings.filterwarnings("ignore")
from infer_rvc_python import BaseLoader
src, dst, model, index, shift = sys.argv[1:6]
c = BaseLoader(only_cpu=False, hubert_path=None, rmvpe_path=None)
c.apply_conf(tag="v", file_model=model, pitch_algo="rmvpe", pitch_lvl=int(shift), file_index=index,
             index_influence=0.75, respiration_median_filtering=3, envelope_ratio=0.25,
             consonant_breath_protection=0.33)
tmp = dst + ".in.wav"; shutil.copy(src, tmp)
out = c([tmp], ["v"], overwrite=True, parallel_workers=1, type_output="wav", show_progress=False)
shutil.move(out[0] if isinstance(out, list) else out, dst)
"""


def rvc(src, dst, shift):
    subprocess.run([RVC_PY, "-c", RVC_SNIPPET, src, dst, os.path.join(MODEL, "model.pth"),
                    os.path.join(MODEL, "model.index"), str(shift)], check=True)


def post_eq(y):
    y = ss.sosfiltfilt(ss.butter(4, 120, "hp", fs=SR, output="sos"), y)
    # -8 dB high shelf from ~5 kHz: low-pass branch + attenuated high branch
    lo = ss.sosfiltfilt(ss.butter(2, 5000, "lp", fs=SR, output="sos"), y)
    return lo + 10 ** (-8 / 20) * (y - lo)


def rms(x):
    x = x if x.ndim == 1 else x.mean(1)
    return np.sqrt(np.mean(x[np.abs(x) > 1e-4] ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("song")
    ap.add_argument("--start", type=float, default=None, help="excerpt start (s); default whole song")
    ap.add_argument("--dur", type=float, default=None)
    ap.add_argument("--shift", type=int, default=-8,
                    help="-8 = reference sound (a major third off key); -12 = in key")
    ap.add_argument("--vocal-db", type=float, default=-3.9, help="vocal level vs instrumental")
    ap.add_argument("--out", default="out/rvc")
    a = ap.parse_args()

    name = os.path.splitext(os.path.basename(a.song))[0]
    if a.start is None:
        dur = float(subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", a.song]))
        start, length, pad = 0.0, dur, 0.0
    else:
        start, length, pad = a.start, a.dur or 30.0, convert.PAD
    work = os.path.join(HERE, "work", f"{name}_{start:.1f}_{length:.0f}")
    _, voc_p, inst_p = convert.separate(a.song, start, length, work)
    conv_p = os.path.join(work, f"rvc_villager_{a.shift:+d}.wav")
    if not os.path.exists(conv_p):
        rvc(voc_p, conv_p, a.shift)

    v, _ = sf.read(conv_p)
    v = v.mean(1) if v.ndim == 2 else v
    if _ != SR:
        import librosa
        v = librosa.resample(v, orig_sr=_, target_sr=SR)
    inst, _ = sf.read(inst_p)
    inst = inst if inst.ndim == 2 else np.stack([inst] * 2, 1)
    v = post_eq(v)
    n = min(len(v), len(inst))
    v, inst = v[:n], inst[:n]
    v *= rms(inst) / rms(v) * 10 ** (a.vocal_db / 20)
    mix = inst + v[:, None]  # mono vocal, dry

    lo = int(min(pad, start) * SR)
    hi = lo + int(length * SR) if a.start is not None else n
    mix, v = mix[lo:hi], v[lo:hi]
    if a.start is not None:  # fades on excerpts
        f = np.ones(len(mix))
        f[:int(0.3 * SR)] = np.linspace(0, 1, int(0.3 * SR))
        f[-int(1.5 * SR):] = np.linspace(1, 0, int(1.5 * SR))
        mix, v = mix * f[:, None], v * f
    os.makedirs(a.out, exist_ok=True)
    tag = f"{a.out}/{name}{'' if a.start is None else f'_{start:g}s'}_villager_rvc{a.shift:+d}"
    sf.write(f"{tag}_premaster.wav", 0.9 * mix / np.abs(mix).max(), SR)
    sf.write(f"{tag}_vocal.wav", 0.9 * v / np.abs(v).max(), SR)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", f"{tag}_premaster.wav", "-af",
                    "loudnorm=I=-16:TP=-1:LRA=11", "-ar", str(SR), f"{tag}.wav"], check=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", f"{tag}.wav", "-b:a", "192k",
                    f"{tag}.mp3"], check=True)
    print("wrote", f"{tag}.mp3")


if __name__ == "__main__":
    main()
