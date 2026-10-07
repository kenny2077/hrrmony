"""RVC voice conversion: the singer's vocal -> a Minecraft villager.

Why RVC (and not pasted villager samples)? The popular villager covers keep the singer's
phrasing and pitch exactly and turn consonants into short, hit-like villager noise bursts.
That is what a voice-conversion model trained on villager sounds does; see
docs/how-it-works.md for the measurements.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import DEFAULT_VOICE, SAMPLE_RATE, VOICES, Voice, models_dir

log = logging.getLogger(__name__)

# Conversion settings matched to the reference covers (others barely mattered in the sweep).
RVC_SETTINGS = dict(pitch_algo="rmvpe", index_influence=0.75, respiration_median_filtering=3,
                    envelope_ratio=0.25, consonant_breath_protection=0.33)
RMVPE_REPO, RMVPE_FILE = "r3gm/sonitranslate_voice_models", "rmvpe.pt"


def voice_files(voice: str | Voice = DEFAULT_VOICE) -> tuple[Path, Path]:
    """Download (once) and return the (model.pth, model.index) of a registered voice."""
    from huggingface_hub import hf_hub_download

    v = VOICES[voice] if isinstance(voice, str) else voice
    target = models_dir() / "voices" / v.name
    model = hf_hub_download(v.repo, v.model_file, local_dir=target)
    index = hf_hub_download(v.repo, v.index_file, local_dir=target)
    return Path(model), Path(index)


def rmvpe_file() -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(RMVPE_REPO, RMVPE_FILE, local_dir=models_dir() / "pitch"))


@lru_cache(maxsize=1)
def _converter(device: str):
    from ._vendor.infer_rvc_python.main import BaseLoader

    return BaseLoader(only_cpu=(device == "cpu"), hubert_path=None, rmvpe_path=str(rmvpe_file()))


def convert(vocal: np.ndarray, shift: int, voice: str = DEFAULT_VOICE, device: str = "auto") -> np.ndarray:
    """Convert a mono vocal at SAMPLE_RATE; returns mono float32 at SAMPLE_RATE."""
    import librosa

    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, index = voice_files(voice)
    conv = _converter(device)
    tag = f"{voice}{shift:+d}"
    conv.apply_conf(tag=tag, file_model=str(model), file_index=str(index), pitch_lvl=int(shift),
                    **RVC_SETTINGS)
    out, sr = conv.generate_from_cache(audio_data=(vocal.astype(np.float32), SAMPLE_RATE), tag=tag)
    out = np.asarray(out, dtype=np.float32)
    if np.abs(out).max() > 2.0:  # int16-scaled output
        out = out / 32768.0
    if sr != SAMPLE_RATE:
        out = librosa.resample(out, orig_sr=sr, target_sr=SAMPLE_RATE)
    return out
