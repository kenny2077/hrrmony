import numpy as np

from villager_sound import mix
from villager_sound.config import SAMPLE_RATE as SR


def tone(hz, seconds=2.0, amp=0.5):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def level_db(y):
    core = y[len(y) // 4: -len(y) // 4]
    return 20 * np.log10(np.sqrt(np.mean(core ** 2)) + 1e-12)


def test_vocal_eq_cuts_rumble_keeps_voice_darkens_top():
    assert level_db(mix.vocal_eq(tone(50))) - level_db(tone(50)) < -12
    assert abs(level_db(mix.vocal_eq(tone(800))) - level_db(tone(800))) < 1.0
    assert level_db(mix.vocal_eq(tone(12000))) - level_db(tone(12000)) < -5


def test_mix_places_vocal_relative_to_instrumental():
    inst = np.stack([tone(220, amp=0.3)] * 2, axis=1)
    vocal = tone(440, amp=0.9)
    out = mix.mix(vocal, inst, vocal_db=-3.9)
    assert out.shape == inst.shape
    assert np.abs(out).max() <= 0.9 + 1e-6
    # separate the two components by frequency and compare their levels
    spec = np.abs(np.fft.rfft(out[:, 0]))
    f = np.fft.rfftfreq(len(out), 1 / SR)
    ratio = 20 * np.log10(spec[np.argmin(abs(f - 440))] / spec[np.argmin(abs(f - 220))])
    assert -5.0 < ratio < -2.8


def test_mix_handles_mono_instrumental_and_length_mismatch():
    out = mix.mix(tone(440, 1.0), tone(220, 1.5))
    assert out.shape == (SR, 2)


def test_fades():
    env = mix.fades(SR * 4)
    assert env[0] == 0 and env[-1] == 0 and env[SR * 2] == 1
