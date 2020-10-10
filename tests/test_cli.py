"""Collections of tests of the CLI."""
import pathlib

from click.testing import CliRunner
import pytest

import mltype.cli.cli


@pytest.mark.parametrize(
    "cmd", ["file", "list", "raw", "replay", "sample", "train"]
)
def test_help(cmd):
    fun = getattr(mltype.cli.cli, cmd)

    runner = CliRunner()
    result = runner.invoke(fun, "--help")

    assert result.exit_code == 0


def test_list(tmpdir, monkeypatch):
    new_home = pathlib.Path(str(tmpdir))
    path_languages = new_home / ".mltype" / "languages"
    path_languages.mkdir(parents=True)

    monkeypatch.setenv("HOME", str(new_home))

    runner = CliRunner()
    mlt_list = getattr(mltype.cli.cli, "list")
    result = runner.invoke(mlt_list, [])

    assert result.exit_code == 0
    assert result.output == ""

    (path_languages / "a").touch()
    (path_languages / "b").touch()
    (path_languages / "c").mkdir()

    result = runner.invoke(mlt_list, [])

    assert result.exit_code == 0
    assert result.output == "a\nb\n"
