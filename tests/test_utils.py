"""Tests covering the `utils.py` module."""
import pathlib

from mltype.utils import get_cache_dir


def test_get_cache_dir(monkeypatch, tmpdir):
    tmpdir_ = pathlib.Path(str(tmpdir))

    assert tmpdir_ == get_cache_dir(tmpdir_)

    cache_dir_before_mp = get_cache_dir()

    assert tmpdir_ != cache_dir_before_mp

    monkeypatch.setenv("HOME", str(tmpdir))

    cache_dir_after_mp = get_cache_dir()

    assert cache_dir_before_mp != cache_dir_after_mp
    assert cache_dir_after_mp == tmpdir_ / ".mltype"
