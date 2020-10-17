"""Tests covering the `utils.py` module."""
import pathlib
from unittest.mock import Mock

import pytest

from mltype.utils import get_cache_dir, get_mlflow_artifacts_path, print_section


def test_get_cache_dir(monkeypatch, tmpdir):
    tmpdir_ = pathlib.Path(str(tmpdir))

    assert tmpdir_ == get_cache_dir(tmpdir_)

    cache_dir_before_mp = get_cache_dir()

    assert tmpdir_ != cache_dir_before_mp

    monkeypatch.setenv("HOME", str(tmpdir))

    cache_dir_after_mp = get_cache_dir()

    assert cache_dir_before_mp != cache_dir_after_mp
    assert cache_dir_after_mp == tmpdir_ / ".mltype"


def test_get_mlflow_artifacts_path():
    path_str = "file:/path/to/whatever"

    fake_info = Mock()
    fake_info.info.artifact_uri = path_str

    client = Mock()
    client.get_run.return_value = fake_info

    res = get_mlflow_artifacts_path(client, "1234")

    assert res == pathlib.Path(path_str[5:])


@pytest.mark.parametrize("drop_end", [True, False])
@pytest.mark.parametrize("add_ts", [True, False])
def test_print_section(capsys, drop_end, add_ts):
    # error
    with pytest.raises(ValueError):
        with print_section(
            "name", fill_char="*-", drop_end=drop_end, add_ts=add_ts
        ):
            pass

    # clear cache
    _ = capsys.readouterr()

    fill_char = "@"
    with print_section(
        "name", fill_char=fill_char, drop_end=drop_end, add_ts=add_ts
    ):
        pass

    captured = capsys.readouterr()
    assert f"{fill_char}name" in captured.out
