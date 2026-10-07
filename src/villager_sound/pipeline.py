"""End-to-end: song -> Minecraft villager cover."""

from __future__ import annotations

import hashlib
import logging
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from . import audio, mix, segment, separate, voice
from .config import DEFAULT_SHIFT, DEFAULT_VOICE, SAMPLE_RATE, home

log = logging.getLogger(__name__)

Mode = Literal["hook", "full"]
# stage name, fraction of the job done (0..1), human message
Progress = Callable[[str, float, str], None]

PAD = 2.0  # seconds of context separated around a hook so the cut edges stay clean


@dataclass
class CoverOptions:
    mode: Mode = "hook"            # "hook": best 30 s (chorus); "full": entire song
    duration: float = 30.0         # hook length in seconds
    start: float | None = None     # force the hook start instead of auto-detecting it
    shift: int = DEFAULT_SHIFT     # semitones; -8 classic villager sound, -12 in key
    voice: str = DEFAULT_VOICE
    vocal_db: float = mix.VOCAL_DB
    loudness: float = -16.0        # LUFS
    formats: tuple[str, ...] = ("mp3", "wav")
    keep_stems: bool = False
    device: str = "auto"


@dataclass
class CoverResult:
    outputs: dict[str, Path]
    start: float
    duration: float
    stems: dict[str, Path] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["outputs"] = {k: str(v) for k, v in self.outputs.items()}
        d["stems"] = {k: str(v) for k, v in self.stems.items()}
        return d


def _noop(stage: str, frac: float, msg: str) -> None:
    log.info("[%3.0f%%] %s: %s", frac * 100, stage, msg)


def _workdir(src: Path, start: float, length: float) -> Path:
    h = hashlib.sha1(f"{src.resolve()}|{src.stat().st_size}|{start:.2f}|{length:.2f}".encode())
    d = home() / "work" / h.hexdigest()[:16]
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_cover(song: str | Path, output_dir: str | Path = ".", options: CoverOptions | None = None,
               progress: Progress | None = None) -> CoverResult:
    """Turn `song` (any format ffmpeg reads) into a villager cover in `output_dir`."""
    opts = options or CoverOptions()
    say = progress or _noop
    src = Path(song)
    if not src.exists():
        raise FileNotFoundError(src)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t: dict[str, float] = {}
    t0 = time.perf_counter()

    say("analyze", 0.02, "Reading the song")
    total = audio.duration(src)
    if opts.mode == "full":
        start, length, pad_l = 0.0, total, 0.0
    else:
        length = min(opts.duration, total)
        if opts.start is not None:
            start = max(0.0, min(opts.start, total - length))
        else:
            say("analyze", 0.05, "Finding the hook (loudest, most repeated part)")
            start = segment.find_hook(src, length)
        pad_l = min(PAD, start)
    t["analyze"] = time.perf_counter() - t0

    work = _workdir(src, start, length)
    clip = work / "clip.wav"
    if not clip.exists():
        audio.decode(src, clip, start - pad_l, length + pad_l + (PAD if opts.mode == "hook" else 0))

    t1 = time.perf_counter()
    say("separate", 0.15, "Separating vocals from the music")
    vocals_p, inst_p = separate.separate(clip, work)
    t["separate"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    say("convert", 0.55, "Villagerizing the vocal")
    vocal = audio.read(vocals_p, mono=True)
    villager = voice.convert(vocal, opts.shift, opts.voice, opts.device)
    villager = mix.vocal_eq(villager)
    t["convert"] = time.perf_counter() - t2

    t3 = time.perf_counter()
    say("master", 0.85, "Mixing and mastering")
    inst = audio.read(inst_p)
    full = mix.mix(villager, inst, opts.vocal_db)
    a = int(pad_l * SAMPLE_RATE)
    b = a + int(length * SAMPLE_RATE)
    cut, vox = full[a:b], villager[a:b]
    if opts.mode == "hook":
        env = mix.fades(len(cut))
        cut, vox = cut * env[:, None], vox * env[: len(vox)]
    name = f"{src.stem}_villager" + (f"_{start:.0f}s" if opts.mode == "hook" else "")
    with tempfile.TemporaryDirectory() as td:
        pre = audio.write(Path(td) / "premaster.wav", cut)
        outputs = audio.master(pre, out_dir / name, opts.loudness, opts.formats)
        # the same excerpt of the original, loudness-matched, for A/B listening
        orig = audio.read(clip)[a:b]
        if opts.mode == "hook":
            orig = orig * mix.fades(len(orig))[:, None]
        ref = audio.write(Path(td) / "original.wav", orig)
        outputs["original"] = audio.master(ref, out_dir / f"{name}_original", opts.loudness,
                                           ("mp3",))["mp3"]
    stems: dict[str, Path] = {}
    if opts.keep_stems:
        peak = np.abs(vox).max() or 1.0
        stems["villager_vocal"] = audio.write(out_dir / f"{name}_vocal.wav", 0.9 * vox / peak)
        stems["instrumental"] = audio.write(out_dir / f"{name}_instrumental.wav", inst[a:b])
    t["master"] = time.perf_counter() - t3
    t["total"] = time.perf_counter() - t0
    say("done", 1.0, f"Done in {t['total']:.0f} s")
    return CoverResult(outputs=outputs, start=start, duration=length, stems=stems, timings=t)
