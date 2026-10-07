"""Paths, model registry and tunable defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

SAMPLE_RATE = 44100


def home() -> Path:
    """Cache/config root. Override with ``HRRMONY_HOME``."""
    root = os.environ.get("HRRMONY_HOME") or Path.home() / ".cache" / "hrrmony"
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
    revision: str  # pinned commit: model files are pickles, never follow a moving branch


VOICES: dict[str, Voice] = {
    "villager": Voice(
        name="villager",
        repo="r3gm/villager",
        model_file="model.pth",
        index_file="model.index",
        description="Community Minecraft villager RVC v2 model; closest match to the reference "
        "'AI generated' villager covers (see docs/how-it-works.md).",
        source_url="https://huggingface.co/r3gm/villager",
        revision="8db3b84dca04b46dec60b9d22fb2a66eed87d9b5",
    ),
}
DEFAULT_VOICE = "villager"

# Shift presets, in semitones. -8 reproduces the popular villager-cover sound (it lands a major
# third off the backing track's key); -12 keeps the vocal in key.
SHIFT_PRESETS = {"classic": -8, "in-key": -12}
DEFAULT_SHIFT = SHIFT_PRESETS["classic"]
MAX_SHIFT = 24


def parse_shift(value: str | int) -> int:
    """'classic' / 'in-key' or an integer number of semitones within +-MAX_SHIFT."""
    if isinstance(value, str) and value in SHIFT_PRESETS:
        return SHIFT_PRESETS[value]
    try:
        semis = int(value)
    except (TypeError, ValueError):
        names = ", ".join(SHIFT_PRESETS)
        raise ValueError(f"shift must be one of {names} or a number of semitones") from None
    if not -MAX_SHIFT <= semis <= MAX_SHIFT:
        raise ValueError(f"shift must be between -{MAX_SHIFT} and {MAX_SHIFT} semitones")
    return semis
