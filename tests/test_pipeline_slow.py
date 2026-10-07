"""Full pipeline with real models. Opt-in: `pytest -m slow` (downloads ~1.2 GB on first run)."""

import numpy as np
import pytest
import soundfile as sf

from hrrmony.config import SAMPLE_RATE as SR

pytestmark = pytest.mark.slow


def test_make_cover_end_to_end(tmp_path):
    pytest.importorskip("audio_separator")
    from hrrmony.pipeline import CoverOptions, make_cover

    # a sung-ish vowel: harmonic stack with vibrato over a quiet chord
    t = np.arange(12 * SR) / SR
    f0 = 220 * 2 ** (0.3 / 12 * np.sin(2 * np.pi * 5.5 * t))
    phase = 2 * np.pi * np.cumsum(f0) / SR
    voice = sum(np.sin(k * phase) / k for k in range(1, 12)) * 0.2
    backing = 0.05 * np.sin(2 * np.pi * 110 * t)
    song = tmp_path / "song.wav"
    sf.write(song, np.stack([voice + backing] * 2, axis=1), SR)

    res = make_cover(song, tmp_path / "out", CoverOptions(mode="hook", duration=8.0))
    assert res.outputs["mp3"].stat().st_size > 10_000
    y, sr = sf.read(res.outputs["wav"])
    assert sr == SR and abs(len(y) / SR - 8.0) < 0.2
