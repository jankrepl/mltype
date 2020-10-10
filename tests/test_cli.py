"""Collections of tests of the CLI."""
import pathlib

from click.testing import CliRunner

from mltype.cli.cli import list as mlt_list


def test_list(tmpdir, monkeypatch):
    new_home = pathlib.Path(str(tmpdir))
    path_languages = new_home / ".mltype" / "languages"
    path_languages.mkdir(parents=True)

    monkeypatch.setenv("HOME", str(new_home))

    runner = CliRunner()
    result = runner.invoke(mlt_list, [])

    assert result.exit_code == 0
    assert result.output == ""

    (path_languages / "a").touch()
    (path_languages / "b").touch()
    (path_languages / "c").mkdir()

    result = runner.invoke(mlt_list, [])

    assert result.exit_code == 0
    assert result.output == "a\nb\n"
