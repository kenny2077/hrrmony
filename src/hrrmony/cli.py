"""`hrrmony` command line."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

from . import __version__
from .config import DEFAULT_SHIFT, DEFAULT_VOICE, VOICES, home, parse_shift
from .mix import VOCAL_DB


def _shift(value: str) -> int:
    try:
        return parse_shift(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def _seconds(value: str) -> float:
    v = float(value)
    if v <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return v


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hrrmony",
                                description="Turn any song into a Minecraft villager cover.")
    p.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="show debug logs")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("cover", help="make a villager cover of a song")
    c.add_argument("song", type=Path, help="audio file (mp3, wav, flac, m4a, ...)")
    g = c.add_mutually_exclusive_group()
    g.add_argument("--hook", dest="mode", action="store_const", const="hook",
                   help="best ~30 s of the song (default)")
    g.add_argument("--full", dest="mode", action="store_const", const="full",
                   help="the entire song")
    c.set_defaults(mode="hook")
    c.add_argument("-d", "--duration", type=_seconds, default=30.0, help="hook length in seconds")
    c.add_argument("-s", "--start", type=float, help="hook start in seconds (skip auto-detect)")
    c.add_argument("--shift", type=_shift, default=DEFAULT_SHIFT,
                   help="semitones, or 'classic' (-8, default) / 'in-key' (-12)")
    c.add_argument("--voice", default=DEFAULT_VOICE, choices=sorted(VOICES))
    c.add_argument("--vocal-db", type=float, default=VOCAL_DB, help="vocal level vs instrumental")
    c.add_argument("--loudness", type=float, default=-16.0, help="target LUFS")
    c.add_argument("-o", "--output", type=Path, default=Path("."), help="output directory")
    c.add_argument("--format", dest="formats", action="append", choices=["mp3", "wav"],
                   help="output format(s); default mp3 and wav")
    c.add_argument("--stems", action="store_true", help="also save villager vocal + instrumental")
    c.add_argument("--cpu", action="store_true", help="force CPU even if a GPU is available")
    c.add_argument("--json", action="store_true", help="print the result as JSON")

    s = sub.add_parser("serve", help="start the local web app")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=7860)

    sub.add_parser("doctor", help="check ffmpeg, GPU and model cache")
    v = sub.add_parser("voices", help="list voices, or download one with --download")
    v.add_argument("--download", metavar="NAME", help="pre-fetch a voice model")
    return p


def cmd_cover(a: argparse.Namespace) -> int:
    from .pipeline import CoverOptions, make_cover

    opts = CoverOptions(mode=a.mode, duration=a.duration, start=a.start, shift=a.shift,
                        voice=a.voice, vocal_db=a.vocal_db, loudness=a.loudness,
                        formats=tuple(a.formats or ("mp3", "wav")), keep_stems=a.stems,
                        device="cpu" if a.cpu else "auto")

    def progress(stage: str, frac: float, msg: str) -> None:
        if not a.json:
            print(f"  {'█' * int(frac * 20):<20} {frac:4.0%}  {msg}", file=sys.stderr)

    res = make_cover(a.song, a.output, opts, progress)
    if a.json:
        print(json.dumps(res.to_dict(), indent=2))
    else:
        where = "full song" if a.mode == "full" else f"{res.start:.1f}–{res.start + res.duration:.1f} s"
        print(f"\n✓ villager cover ({where})")
        for kind, path in {**res.outputs, **res.stems}.items():
            print(f"  {kind:<15} {path}")
    return 0


def cmd_serve(a: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("The web app needs the 'web' extra: pip install 'hrrmony[web]'", file=sys.stderr)
        return 1
    import os

    from .server.app import LOOPBACK_HOSTS, create_app

    if a.host in LOOPBACK_HOSTS:
        allowed = LOOPBACK_HOSTS
    else:
        env = os.environ.get("HRRMONY_ALLOWED_HOSTS", "")
        allowed = frozenset(h.strip().lower() for h in env.split(",") if h.strip()) or None
        print(f"warning: listening on {a.host}. The web app has no login; anyone who can reach this "
              "port can upload songs and use your GPU. Put it behind an authenticating proxy.",
              file=sys.stderr)
    print(f"hrrmony {__version__} → http://{a.host}:{a.port}")
    uvicorn.run(create_app(allowed_hosts=allowed), host=a.host, port=a.port, log_level="warning")
    return 0


def cmd_doctor(_: argparse.Namespace) -> int:
    ok = True

    def line(good: bool, label: str, detail: str) -> None:
        nonlocal ok
        ok &= good
        print(f"  {'✓' if good else '✗'} {label:<14} {detail}")

    print(f"hrrmony {__version__}")
    line(bool(shutil.which("ffmpeg") and shutil.which("ffprobe")), "ffmpeg",
         shutil.which("ffmpeg") or "not found: install ffmpeg")
    try:
        import torch

        gpu = torch.cuda.is_available()
        line(True, "torch", f"{torch.__version__} · " + (
            f"CUDA {torch.cuda.get_device_name(0)}" if gpu else "CPU only (works, ~5-10x slower)"))
    except ImportError:
        line(False, "torch", "not installed: pip install 'hrrmony[cpu]' or '[gpu]'")
    for mod, extra in [("audio_separator", "cpu|gpu"), ("faiss", "cpu|gpu"), ("fastapi", "web")]:
        try:
            __import__(mod)
            line(True, mod, "ok")
        except ImportError:
            line(mod == "fastapi", mod, f"missing (extra: {extra})")
    cache = home()
    size = sum(f.stat().st_size for f in cache.rglob("*") if f.is_file()) / 1e9
    line(True, "cache", f"{cache} ({size:.1f} GB)")
    return 0 if ok else 1


def cmd_voices(a: argparse.Namespace) -> int:
    if a.download:
        from .voice import rmvpe_file, voice_files

        model, index = voice_files(a.download)
        print(f"✓ {a.download}: {model}\n  index: {index}\n  pitch: {rmvpe_file()}")
        return 0
    for v in VOICES.values():
        print(f"{v.name:<10} {v.description}\n{'':<10} {v.source_url}")
    return 0


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    return {"cover": cmd_cover, "serve": cmd_serve, "doctor": cmd_doctor,
            "voices": cmd_voices}[a.command](a)


if __name__ == "__main__":
    sys.exit(main())
