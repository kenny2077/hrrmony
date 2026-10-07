import pytest

from hrrmony import __version__
from hrrmony.cli import build_parser, main


def parse(*args):
    return build_parser().parse_args(list(args))


def test_cover_defaults_to_hook_classic():
    a = parse("cover", "song.mp3")
    assert a.mode == "hook"
    assert a.shift == -8
    assert a.duration == 30.0
    assert a.start is None


def test_full_song_and_in_key_preset():
    a = parse("cover", "song.mp3", "--full", "--shift", "in-key")
    assert a.mode == "full"
    assert a.shift == -12


def test_numeric_shift():
    assert parse("cover", "x.wav", "--shift", "-5").shift == -5


def test_bad_shift_is_rejected():
    with pytest.raises(SystemExit):
        parse("cover", "x.wav", "--shift", "loud")


def test_hook_and_full_are_exclusive():
    with pytest.raises(SystemExit):
        parse("cover", "x.wav", "--hook", "--full")


def test_version(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])
    assert __version__ in capsys.readouterr().out


def test_voices_lists_villager(capsys):
    assert main(["voices"]) == 0
    assert "villager" in capsys.readouterr().out
