"""Post-processing and mixing, matched to the reference villager covers.

Measured on the reference: villager vocal is mono, dry, uncompressed, ~3.9 dB under the
original instrumental, with 50-100 Hz cut by ~10 dB and the top above ~4-5 kHz darkened.
"""

from __future__ import annotations

import numpy as np
import scipy.signal as ss

from .config import SAMPLE_RATE

HIGHPASS_HZ = 120.0
SHELF_HZ = 5000.0
SHELF_DB = -8.0
VOCAL_DB = -3.9
MAX_OVER_SINGER_DB = 3.0  # never louder than the original singer + this


def vocal_eq(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """4th-order high-pass at 120 Hz plus a -8 dB high shelf above ~5 kHz."""
    y = ss.sosfiltfilt(ss.butter(4, HIGHPASS_HZ, "hp", fs=sr, output="sos"), y)
    low = ss.sosfiltfilt(ss.butter(2, SHELF_HZ, "lp", fs=sr, output="sos"), y)
    return low + 10 ** (SHELF_DB / 20) * (y - low)


def rms(y: np.ndarray) -> float:
    """RMS over non-silent samples (mono or stereo)."""
    x = y.mean(axis=1) if y.ndim == 2 else y
    x = x[np.abs(x) > 1e-4]
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0


def fades(n: int, fade_in: float = 0.3, fade_out: float = 1.5, sr: int = SAMPLE_RATE) -> np.ndarray:
    env = np.ones(n, dtype=np.float32)
    a, b = min(n // 2, int(fade_in * sr)), min(n // 2, int(fade_out * sr))
    if a:
        env[:a] = np.linspace(0, 1, a)
    if b:
        env[-b:] = np.linspace(1, 0, b)
    return env


def mix(vocal: np.ndarray, instrumental: np.ndarray, vocal_db: float = VOCAL_DB,
        reference_vocal: np.ndarray | None = None) -> np.ndarray:
    """Centre the (mono) villager vocal `vocal_db` dB relative to the instrumental's level.

    With `reference_vocal` (the separated original vocal) the gain is capped so the villager is
    never more than MAX_OVER_SINGER_DB above the original singer: near-instrumental tracks stay
    near-instrumental instead of having separation residue blown up.
    """
    n = min(len(vocal), len(instrumental))
    v, inst = vocal[:n], instrumental[:n]
    if inst.ndim == 1:
        inst = np.stack([inst, inst], axis=1)
    level, own = rms(inst), rms(v)
    if level > 0 and own > 0:
        gain = (level / own) * 10 ** (vocal_db / 20)
        if reference_vocal is not None:
            singer = rms(reference_vocal[:n])
            gain = min(gain, singer * 10 ** (MAX_OVER_SINGER_DB / 20) / own)
        v = v * gain
    out = inst + v[:, None]
    peak = np.abs(out).max()
    return out / peak * 0.9 if peak > 0 else out
