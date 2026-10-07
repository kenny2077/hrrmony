"""Paths, model registry and tunable defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

SAMPLE_RATE = 44100


def home() -> Path:
    """Cache/config root. Override with ``VILLAGER_SOUND_HOME``."""
    root = os.environ.get("VILLAGER_SOUND_HOME") or Path.home() / ".cache" / "villager-sound"
    path = Path(root).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir() -> Path:
    path = home() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


# Mel-Band RoFormer vocal model (Kimberley Jensen), fetched by audio-separator on first use.
SEPARATOR_MODEL = "vocals_mel_band_roformer.ckpt"


@dataclass(frozen=True)
class Voice:
    """An RVC v2 voice model hosted on the Hugging Face Hub."""

    name: str
    repo: str
    model_file: str
    index_file: str
    description: str
    source_url: str


VOICES: dict[str, Voice] = {
    "villager": Voice(
        name="villager",
        repo="r3gm/villager",
        model_file="model.pth",
        index_file="model.index",
        description="Community Minecraft villager RVC v2 model; closest match to the reference "
        "'AI generated' villager covers (see docs/how-it-works.md).",
        source_url="https://huggingface.co/r3gm/villager",
    ),
}
DEFAULT_VOICE = "villager"

# Shift presets, in semitones. -8 reproduces the popular villager-cover sound (it lands a major
# third off the backing track's key); -12 keeps the vocal in key.
SHIFT_PRESETS = {"classic": -8, "in-key": -12}
DEFAULT_SHIFT = SHIFT_PRESETS["classic"]
