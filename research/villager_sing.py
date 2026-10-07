"""Make a Minecraft villager sing a melody.

Two vocal modes, both built on the WORLD vocoder (f0 / spectral envelope / aperiodicity):

  hum     Re-pitch and time-stretch the real villager "hrmm" clips to each note.
          Formants stay put (unlike naive resampling), so it keeps the villager timbre.
  lyrics  Build a sung guide vocal from per-syllable TTS (macOS `say`), force it onto
          the melody, then "villager-ify" it: transfer the villager long-term spectral
          envelope, add breath/roughness, scoop-up pitch and subharmonic growl.

A note-block style backing track is rendered from the song's chord list.

    python villager_sing.py songs/happy_birthday.json --out out
"""

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import warnings

import librosa
import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")
import pyworld as pw  # noqa: E402

SR = 44100
FP = 5.0  # WORLD frame period, ms
FPS = 1000.0 / FP
HERE = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.join(HERE, "ref", "villager")
CACHE = os.path.join(HERE, "cache")
# Clips whose voiced "hrmm" core is clean enough to reuse as a sustained note.
HUM_CLIPS = ["idle1", "idle2", "yes2", "haggle3", "yes3", "haggle2"]


def midi_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


def world(y, f0_floor=60.0, f0_ceil=600.0, fixed_f0=None):
    y = y.astype(np.float64)
    f0, t = pw.harvest(y, SR, f0_floor=f0_floor, f0_ceil=f0_ceil, frame_period=FP)
    if fixed_f0 is not None:  # rough/creaky voices: harvest drops frames, keep them voiced
        f0 = np.where(f0 > 0, f0, fixed_f0)
    sp = pw.cheaptrick(y, f0, t, SR)
    ap = pw.d4c(y, f0, t, SR)
    return f0, sp, ap


def frame_energy(sp):
    return 10 * np.log10(sp.sum(1) + 1e-20)


def stretch_frames(x, n):
    """Resample a (frames, bins) array to n frames by linear interpolation."""
    if n <= 0:
        return x[:0]
    if len(x) == 1:
        return np.repeat(x, n, 0)
    pos = np.linspace(0, len(x) - 1, n)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, len(x) - 1)
    w = (pos - i0)[:, None]
    return x[i0] * (1 - w) + x[i1] * w


def stretch_core(x, n, edge=0.2):
    """Stretch to n frames, keeping attack/release edges at natural speed."""
    if n <= len(x):
        return stretch_frames(x, n)
    e = max(1, int(len(x) * edge))
    head, mid, tail = x[:e], x[e:len(x) - e], x[len(x) - e:]
    if len(mid) < 2:
        return stretch_frames(x, n)
    return np.concatenate([head, stretch_frames(mid, n - len(head) - len(tail)), tail])


# ---------------------------------------------------------------- pitch contour

def note_f0(n, hz, prev_hz, rng, scoop=1.5, vib_depth=0.25, jitter=0.012):
    """Target f0 for one note: portamento/scoop in, delayed vibrato, jitter."""
    t = np.arange(n) / FPS
    cents = np.zeros(n)
    start = -scoop if prev_hz is None else 12 * np.log2(prev_hz / hz)
    start = min(start, -scoop) if scoop > 0 else start
    cents += start * np.exp(-t / 0.045)  # glide into the note (semitones)
    vib_on = np.clip((t - 0.28) / 0.2, 0, 1)
    cents += vib_depth * vib_on * np.sin(2 * np.pi * 5.3 * t + rng.uniform(0, 6.28))
    drift = np.convolve(rng.normal(0, 1, n), np.ones(15) / 15, "same")
    f0 = hz * 2 ** (cents / 12.0) * (1 + jitter * drift / (np.std(drift) + 1e-9))
    return f0


# ---------------------------------------------------------------- villager reference

def load_villager(name):
    y, _ = librosa.load(os.path.join(REF_DIR, name + ".ogg"), sr=SR)
    return y


def villager_profile():
    """Long-term average log spectral envelope + aperiodicity of real villager voice."""
    sps, aps = [], []
    for name in HUM_CLIPS:
        f0, sp, ap = world(load_villager(name), fixed_f0=105.0)
        e = frame_energy(sp)
        keep = e > e.max() - 18
        sps.append(np.log(sp[keep]))
        aps.append(ap[keep])
    return np.concatenate(sps).mean(0), np.concatenate(aps).mean(0)


def hum_source(name):
    f0, sp, ap = world(load_villager(name), fixed_f0=105.0)
    e = frame_energy(sp)
    idx = np.where(e > e.max() - 22)[0]
    return sp[idx[0]:idx[-1] + 1], ap[idx[0]:idx[-1] + 1]


# ---------------------------------------------------------------- TTS syllables

def tts_syllable(text, voice):
    os.makedirs(CACHE, exist_ok=True)
    key = hashlib.md5(f"{voice}|{text}".encode()).hexdigest()[:12]
    path = os.path.join(CACHE, f"{key}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return d["f0"], d["sp"], d["ap"]
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "s.wav")
        subprocess.run(["say", "-v", voice, "-o", wav, "--file-format=WAVE",
                        f"--data-format=LEI16@{SR}", text], check=True)
        y, _ = sf.read(wav)
    y, _ = librosa.effects.trim(y, top_db=35)
    f0, sp, ap = world(y, f0_floor=60, f0_ceil=400)
    np.savez(path, f0=f0, sp=sp, ap=ap)
    return f0, sp, ap


def syllable_segments(f0, sp):
    """Split a syllable into onset consonants / voiced nucleus / coda."""
    e = frame_energy(sp)
    active = np.where(e > e.max() - 40)[0]
    a0, a1 = active[0], active[-1] + 1
    voiced = np.where(f0[a0:a1] > 0)[0] + a0
    if len(voiced) == 0:
        return slice(a0, a0), slice(a0, a1), slice(a1, a1)
    v0, v1 = voiced[0], voiced[-1] + 1
    return slice(a0, v0), slice(v0, v1), slice(v1, a1)


# ---------------------------------------------------------------- vocal builders

def song_frames(song):
    spb = 60.0 / song["bpm"]
    total = int(song["end_beat"] * spb * FPS) + 1
    return spb, total


def build_vocal(song, mode, voice="Daniel", seed=0, **kw):
    rng = np.random.default_rng(seed)
    spb, total = song_frames(song)
    nb = pw.get_cheaptrick_fft_size(SR) // 2 + 1
    F0 = np.zeros(total)
    SP = np.full((total, nb), 1e-14)
    AP = np.ones((total, nb))
    shift = song.get("vocal_transpose", 0) + kw.get("transpose", 0)
    hums = [hum_source(n) for n in HUM_CLIPS] if mode == "hum" else None
    prev_hz = None
    notes = song["notes"]
    for i, note in enumerate(notes):
        start = int(note["t"] * spb * FPS)
        n = int(note["b"] * spb * FPS)
        # leave a short breath before the next phrase / repeated-note re-articulation
        nxt = notes[i + 1]["t"] if i + 1 < len(notes) else None
        if nxt is None or nxt - (note["t"] + note["b"]) > 0.01:
            n = int(n * 0.9)
        hz = midi_hz(note["m"] + shift)
        if mode == "hum":
            sp, ap = hums[i % len(hums)]
            sp = stretch_core(sp, n)
            ap = np.minimum(stretch_core(ap, n), kw.get("ap_max", 0.55))
            f0 = note_f0(n, hz, prev_hz, rng, scoop=kw.get("scoop", 2.0))
            seg = slice(start, start + n)
            F0[seg], SP[seg], AP[seg] = f0, sp, ap
        else:
            f0s, sps, aps = tts_syllable(note["say"], voice)
            on, nuc, cod = syllable_segments(f0s, sps)
            n_on = min(on.stop - on.start, int(0.12 * FPS))
            n_cod = min(cod.stop - cod.start, int(0.35 * n))
            n_nuc = max(n - n_cod, 4)
            src_sp = np.concatenate([sps[on][-n_on:] if n_on else sps[:0],
                                     stretch_core(sps[nuc], n_nuc), sps[cod][:n_cod]])
            src_ap = np.concatenate([aps[on][-n_on:] if n_on else aps[:0],
                                     stretch_core(aps[nuc], n_nuc), aps[cod][:n_cod]])
            voiced = np.concatenate([f0s[on][-n_on:] > 0 if n_on else np.zeros(0, bool),
                                     np.ones(n_nuc, bool), f0s[cod][:n_cod] > 0])
            f0 = np.zeros(len(src_sp))
            f0[n_on:] = note_f0(len(src_sp) - n_on, hz, prev_hz, rng,
                                scoop=kw.get("scoop", 1.0), vib_depth=0.2)
            f0[:n_on] = f0[n_on] if n_on else 0
            f0 *= voiced
            s0 = max(0, start - n_on)  # consonants lead so the vowel lands on the beat
            seg = slice(s0, s0 + len(src_sp))
            F0[seg], SP[seg], AP[seg] = f0[: seg.stop - seg.start], src_sp, src_ap
        prev_hz = hz
    return F0, SP, AP


def villagerify(F0, SP, AP, profile, amount=1.0, formant=1.0, breath=0.12):
    """Push a sung guide vocal toward the villager timbre (spectral-envelope transfer)."""
    v_logsp, v_ap = profile
    voiced = F0 > 0
    logsp = np.log(SP)
    if formant != 1.0:  # warp the envelope's frequency axis (formant shift)
        bins = np.arange(logsp.shape[1])
        src = np.clip(bins / formant, 0, bins[-1])
        logsp = np.stack([np.interp(src, bins, r) for r in logsp])
    g_logsp = logsp[voiced].mean(0)
    # smooth both long-term spectra so we move the timbre, not the formant detail
    k = np.hanning(31); k /= k.sum()
    diff = np.convolve(v_logsp - g_logsp, k, "same")
    logsp[voiced] += amount * diff
    ap = AP.copy()
    ap[voiced] = np.clip((1 - amount) * ap[voiced] + amount * (0.6 * ap[voiced] + 0.4 * v_ap)
                         + breath, 0, 0.95)
    return F0, np.exp(logsp), ap


def synth(F0, SP, AP, growl=0.0):
    y = pw.synthesize(F0, SP, AP, SR, FP)
    if growl > 0:  # period-doubling AM -> subharmonics: the "hrrr" in hrmm
        f0_t = np.interp(np.arange(len(y)) / SR, np.arange(len(F0)) / FPS, F0)
        phase = np.cumsum(2 * np.pi * (f0_t / 2) / SR)
        y = y * (1 - growl * 0.5 * (1 + np.sin(phase)))
    return y


# ---------------------------------------------------------------- backing track

def pluck(hz, dur, sr=SR):
    """Note-block harp: bright pluck with fast exponential decay."""
    t = np.arange(int(dur * sr)) / sr
    y = sum(a * np.sin(2 * np.pi * hz * h * t) * np.exp(-t * (4 + 3 * h))
            for h, a in [(1, 1), (2, 0.5), (3, 0.25), (4, 0.12)])
    att = np.minimum(t / 0.003, 1)
    return y * att


def backing(song, n_samples):
    spb = 60.0 / song["bpm"]
    out = np.zeros(n_samples + SR * 2)
    chords = song["chords"]
    third = {"maj": 4, "min": 3}
    for i, c in enumerate(chords):
        end = chords[i + 1]["t"] if i + 1 < len(chords) else song["end_beat"]
        r = c["root"]
        tones = [r + 12, r + 12 + third[c["q"]], r + 19]
        b = c["t"]
        while b < end - 1e-6:
            pos = int(b * spb * SR)
            on_down = abs((b % song["beats_per_bar"])) < 1e-6
            if on_down or b == c["t"]:
                y = pluck(midi_hz(r - 12), 1.2) * 0.9
                out[pos:pos + len(y)] += y
            else:
                for m in tones:
                    y = pluck(midi_hz(m), 0.6) * 0.28
                    out[pos:pos + len(y)] += y
            b += 1
    return out[:n_samples]


def mix(vocal, back, back_gain=0.35):
    n = max(len(vocal), len(back))
    v = np.pad(vocal, (0, n - len(vocal)))
    b = np.pad(back, (0, n - len(back)))
    v = v / (np.abs(v).max() + 1e-9)
    b = b / (np.abs(b).max() + 1e-9)
    y = v + back_gain * b
    return 0.89 * y / np.abs(y).max()


def write(path, y):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sf.write(path, (0.95 * y / (np.abs(y).max() + 1e-9)).astype(np.float32), SR)
    print("wrote", os.path.relpath(path))


SEEDVC_DIR = os.path.join(HERE, "third_party", "seed-vc")


def seedvc_convert(src, out_path, steps=30, cfg=0.7):
    """Zero-shot singing voice conversion of `src` to the villager timbre with Seed-VC."""
    ref = os.path.join(HERE, "ref", "villager_reference.wav")
    with tempfile.TemporaryDirectory() as td:
        env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="1", HF_HUB_DISABLE_PROGRESS_BARS="1")
        subprocess.run([os.path.join(SEEDVC_DIR, ".venv", "bin", "python"), "inference.py",
                        "--source", os.path.abspath(src), "--target", ref, "--output", td,
                        "--diffusion-steps", str(steps), "--inference-cfg-rate", str(cfg),
                        "--f0-condition", "True", "--auto-f0-adjust", "False",
                        "--semi-tone-shift", "0", "--fp16", "False"],
                       cwd=SEEDVC_DIR, env=env, check=True)
        y, _ = librosa.load(os.path.join(td, os.listdir(td)[0]), sr=SR)
    write(out_path, y)
    return y


def layer(lead, hum, gain):
    """Lead vocal on top, real-villager hum doubling underneath for extra "hrmm"."""
    n = min(len(lead), len(hum))
    return lead[:n] / np.abs(lead).max() + gain * hum[:n] / np.abs(hum).max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("song")
    ap.add_argument("--out", default="out")
    ap.add_argument("--voice", default="Daniel", help="macOS `say` voice for the lyric guide")
    ap.add_argument("--transpose", type=int, default=0)
    # DSP defaults picked by the sweep in README (best villager-sim at phrase WER <= 0.06)
    ap.add_argument("--amount", type=float, default=1.5, help="villager envelope transfer")
    ap.add_argument("--formant", type=float, default=0.85)
    ap.add_argument("--growl", type=float, default=0.4)
    ap.add_argument("--breath", type=float, default=0.25)
    ap.add_argument("--hum-layer", type=float, default=0.35, help="hum level under the lyrics")
    ap.add_argument("--seedvc", action="store_true", help="also run Seed-VC (best quality, slow)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    song = json.load(open(a.song))
    stem = os.path.join(a.out, os.path.splitext(os.path.basename(a.song))[0])
    profile = villager_profile()

    hum = synth(*build_vocal(song, "hum", seed=a.seed, transpose=a.transpose), growl=0.25)
    guide = build_vocal(song, "lyrics", voice=a.voice, seed=a.seed, transpose=a.transpose)
    lyr = synth(*villagerify(*guide, profile, amount=a.amount, formant=a.formant,
                             breath=a.breath), growl=a.growl)
    back = backing(song, len(hum))

    write(f"{stem}_guide_vocal.wav", synth(*guide))
    write(f"{stem}_villager_hum_vocal.wav", hum)
    write(f"{stem}_villager_lyrics_vocal.wav", lyr)
    write(f"{stem}_backing.wav", back)
    write(f"{stem}_villager_hum_mix.wav", mix(hum, back))
    write(f"{stem}_villager_lyrics_mix.wav", mix(layer(lyr, hum, a.hum_layer), back))
    if a.seedvc:
        svc = seedvc_convert(f"{stem}_guide_vocal.wav", f"{stem}_seedvc_vocal.wav")
        write(f"{stem}_FINAL_seedvc_mix.wav", mix(layer(svc, hum, a.hum_layer), back))


if __name__ == "__main__":
    main()
