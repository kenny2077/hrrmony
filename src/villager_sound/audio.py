"""Small audio I/O helpers built on ffmpeg and soundfile."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import SAMPLE_RATE


class FFmpegMissingError(RuntimeError):
    pass


def require_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe or not shutil.which("ffprobe"):
        raise FFmpegMissingError(
            "ffmpeg/ffprobe not found on PATH. Install it (e.g. `brew install ffmpeg`, "
            "`sudo apt install ffmpeg`, or a static build) and try again."
        )
    return exe


def duration(path: str | Path) -> float:
    require_ffmpeg()
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)]
    )
    return float(out.strip())


def decode(src: str | Path, dst: str | Path, start: float = 0.0, length: float | None = None) -> Path:
    """Decode any input format to a 44.1 kHz stereo WAV, optionally cutting a window."""
    require_ffmpeg()
    cmd = ["ffmpeg", "-loglevel", "error", "-y"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    if length is not None:
        cmd += ["-t", f"{length:.3f}"]
    cmd += ["-i", str(src), "-ac", "2", "-ar", str(SAMPLE_RATE), str(dst)]
    subprocess.run(cmd, check=True)
    return Path(dst)


def read(path: str | Path, mono: bool = False) -> np.ndarray:
    """Read audio as float32 at SAMPLE_RATE (resampling if needed)."""
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        import librosa

        y = librosa.resample(y.T, orig_sr=sr, target_sr=SAMPLE_RATE).T
    if mono:
        return y.mean(axis=1)
    if y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)
    return y


def write(path: str | Path, y: np.ndarray) -> Path:
    sf.write(str(path), y, SAMPLE_RATE, subtype="PCM_16")
    return Path(path)


def master(src: str | Path, dst_stem: str | Path, lufs: float = -16.0,
           formats: tuple[str, ...] = ("mp3", "wav")) -> dict[str, Path]:
    """Loudness-normalise (EBU R128, -1 dBTP) and export each requested format."""
    require_ffmpeg()
    out: dict[str, Path] = {}
    wav = Path(f"{dst_stem}.wav")
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-af",
         f"loudnorm=I={lufs}:TP=-1:LRA=11", "-ar", str(SAMPLE_RATE), str(wav)],
        check=True,
    )
    if "wav" in formats:
        out["wav"] = wav
    if "mp3" in formats:
        mp3 = Path(f"{dst_stem}.mp3")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(wav), "-b:a", "192k",
                        str(mp3)], check=True)
        out["mp3"] = mp3
    if "wav" not in formats:
        wav.unlink(missing_ok=True)
    return out
