"""Pick the most recognisable window of a song (the "hook")."""

from __future__ import annotations

from pathlib import Path

import numpy as np

ANALYSIS_SR = 22050
HOP = 2048


def hook_scores(y: np.ndarray, sr: int, window: float) -> tuple[np.ndarray, np.ndarray]:
    """Score every 1-second start position: loudness x how often the harmony repeats.

    Choruses are both the loudest and the most repeated part of most pop songs, so the
    product of mean RMS and mean chroma self-similarity peaks on them.
    """
    import librosa

    fps = sr / HOP
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP)
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    n = min(chroma.shape[1], len(rms))
    chroma, rms = chroma[:, :n], rms[:n]
    rec = librosa.segment.recurrence_matrix(
        librosa.feature.stack_memory(chroma, n_steps=8, delay=2),
        mode="affinity", width=max(1, min(int(10 * fps), (n - 1) // 2 - 1)), sym=True,
    ).sum(axis=1)
    w = max(1, int(window * fps))
    starts = np.arange(0, max(1, n - w), max(1, int(fps)))
    score = np.array([rms[s:s + w].mean() * rec[s:s + w].mean() for s in starts])
    return starts / fps, score


def snap_to_beat(y: np.ndarray, sr: int, t: float) -> float:
    """Move a start time back onto the nearest preceding beat (<= 1 s away)."""
    import librosa

    _, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    before = beats[(beats <= t + 0.05) & (beats >= t - 1.0)]
    return float(before[-1]) if len(before) else float(t)


def find_hook(path: str | Path, window: float = 30.0) -> float:
    """Start time (s) of the best `window`-second excerpt of the song at `path`."""
    import librosa

    y, sr = librosa.load(str(path), sr=ANALYSIS_SR, mono=True)
    if len(y) / sr <= window:
        return 0.0
    starts, score = hook_scores(y, sr, window)
    best = float(starts[int(np.argmax(score))])
    return max(0.0, snap_to_beat(y, sr, best))
