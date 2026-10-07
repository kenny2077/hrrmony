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
# Pinned commits for everything we load (RVC/RMVPE checkpoints are PyTorch pickles).
RMVPE_REPO, RMVPE_FILE = "r3gm/sonitranslate_voice_models", "rmvpe.pt"
RMVPE_REVISION = "cbe63a1ea0a4eab00a983c54c4784c1bd9d084c8"
HUBERT_REPO, HUBERT_REVISION = "r3gm/hubert_base", "10ca3ff6e99b8cc6932753989bda20bd4674644e"


def voice_files(voice: str | Voice = DEFAULT_VOICE) -> tuple[Path, Path]:
    """Download (once) and return the (model.pth, model.index) of a registered voice."""
    from huggingface_hub import hf_hub_download

    v = VOICES[voice] if isinstance(voice, str) else voice
    target = models_dir() / "voices" / v.name
    model_p, index_p = target / v.model_file, target / v.index_file
    if model_p.exists() and index_p.exists():  # revisions are pinned: a local copy is final
        return model_p, index_p
    model = hf_hub_download(v.repo, v.model_file, revision=v.revision, local_dir=target)
    index = hf_hub_download(v.repo, v.index_file, revision=v.revision, local_dir=target)
    return Path(model), Path(index)


def rmvpe_file() -> Path:
    from huggingface_hub import hf_hub_download

    local = models_dir() / "pitch" / RMVPE_FILE
    if local.exists():
        return local
    return Path(hf_hub_download(RMVPE_REPO, RMVPE_FILE, revision=RMVPE_REVISION,
                                local_dir=models_dir() / "pitch"))


def hubert_dir() -> Path:
    """HuBERT content encoder (safetensors), pinned and cached next to the other models."""
    from huggingface_hub import snapshot_download

    local = models_dir() / "hubert"
    if (local / "config.json").exists() and any(local.glob("*.safetensors")):
        return local
    return Path(snapshot_download(HUBERT_REPO, revision=HUBERT_REVISION,
                                  allow_patterns=["config.json", "*.safetensors"],
                                  local_dir=models_dir() / "hubert"))


@lru_cache(maxsize=1)
def _converter(device: str):
    from ._vendor.infer_rvc_python.main import BaseLoader

    return BaseLoader(only_cpu=(device == "cpu"), hubert_path=str(hubert_dir()),
                      rmvpe_path=str(rmvpe_file()))


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
