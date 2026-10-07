import numpy as np
import pytest


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Never touch the real ~/.cache/villager-sound from tests."""
    monkeypatch.setenv("VILLAGER_SOUND_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


@pytest.fixture
def rng():
    return np.random.default_rng(0)
