"""Vocal / instrumental separation with Mel-Band RoFormer (via audio-separator)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .config import SEPARATOR_MODEL, models_dir


@lru_cache(maxsize=1)
def _separator(model: str, output_dir: str):
    from audio_separator.separator import Separator

    sep = Separator(output_dir=output_dir, output_format="WAV",
                    model_file_dir=str(models_dir() / "separator"))
    sep.load_model(model_filename=model)
    return sep


def separate(clip: str | Path, workdir: str | Path, model: str = SEPARATOR_MODEL) -> tuple[Path, Path]:
    """Split `clip` into (vocals.wav, instrumental.wav) inside `workdir`.

    The full vocal stem (lead + backing) is kept on purpose: converting it matches the
    reference villager covers better than the lead alone.
    """
    workdir = Path(workdir)
    vocals, inst = workdir / "vocals.wav", workdir / "instrumental.wav"
    if vocals.exists() and inst.exists():
        return vocals, inst
    sep = _separator(model, str(workdir))
    sep.output_dir = str(workdir)
    if getattr(sep, "model_instance", None) is not None:
        sep.model_instance.output_dir = str(workdir)
    sep.separate(str(clip), custom_output_names={"Vocals": "vocals", "Other": "instrumental",
                                                 "Instrumental": "instrumental"})
    if not (vocals.exists() and inst.exists()):
        raise RuntimeError(f"separation did not produce {vocals.name} / {inst.name} in {workdir}")
    return vocals, inst
