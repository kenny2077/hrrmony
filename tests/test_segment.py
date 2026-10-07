import numpy as np
import soundfile as sf

from villager_sound.segment import ANALYSIS_SR, find_hook, hook_scores


def chord(freqs, seconds, sr=ANALYSIS_SR, amp=0.3):
    t = np.arange(int(seconds * sr)) / sr
    return amp * sum(np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs)


def make_song(rng, sr=ANALYSIS_SR):
    """Quiet, non-repeating noise around a loud, repeated I-V-vi-IV 'chorus' at 40-70 s."""
    verse = 0.02 * rng.standard_normal(40 * sr)
    prog = [(261.6, 329.6, 392.0), (392.0, 493.9, 587.3), (440.0, 523.3, 659.3), (349.2, 440.0, 523.3)]
    bar = np.concatenate([chord(c, 1.875) for c in prog])  # 7.5 s
    chorus = np.tile(bar, 4)                                 # 30 s
    outro = 0.02 * rng.standard_normal(30 * sr)
    return np.concatenate([verse, chorus, outro]).astype(np.float32)


def test_hook_scores_peak_on_loud_repeated_section(rng):
    y = make_song(rng)
    starts, score = hook_scores(y, ANALYSIS_SR, 20.0)
    best = starts[int(np.argmax(score))]
    assert 35 <= best <= 55


def test_find_hook_lands_in_chorus(tmp_path, rng):
    path = tmp_path / "song.wav"
    sf.write(path, make_song(rng), ANALYSIS_SR)
    start = find_hook(path, 20.0)
    assert 35 <= start <= 55


def test_short_song_starts_at_zero(tmp_path, rng):
    path = tmp_path / "short.wav"
    sf.write(path, chord((440,), 10.0), ANALYSIS_SR)
    assert find_hook(path, 30.0) == 0.0


def test_song_barely_longer_than_window(tmp_path, rng):
    """Regression: the repetition width must shrink for short inputs."""
    path = tmp_path / "twelve.wav"
    sf.write(path, make_song(rng)[: 12 * ANALYSIS_SR], ANALYSIS_SR)
    assert 0.0 <= find_hook(path, 8.0) <= 4.0
