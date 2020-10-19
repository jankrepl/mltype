"""Collections of tests of the CLI."""
import pathlib
from unittest.mock import Mock

from click.testing import CliRunner
import pytest

import mltype.cli.cli


def command_composer(args, options, use_long=True):
    """Compose arguments and options.

    Parameters
    ----------
    args : list or tuple
        Iterable of strings that will be joined on a space.

    options : list
        List of 3 element tuples where the elements are the following:
            * short_option  # str
            * long_option  # str
            * value  # str or bool
    """
    s = " ".join(args) + " "

    elements = []
    for short_op, long_op, v in options:
        if v is False:
            continue

        dash = "--" if use_long else "-"
        name = long_op if use_long else short_op
        value = "" if v is True else f" {v}"
        elements.append(f"{dash}{name}{value}")

    s += " ".join(elements)

    return s


@pytest.mark.parametrize(
    "cmd", ["cli", "file", "ls", "random", "raw", "replay", "sample", "train"]
)
def test_help(cmd):
    fun = getattr(mltype.cli.cli, cmd)

    runner = CliRunner()
    result = runner.invoke(fun, "--help")

    assert result.exit_code == 0


def test_ls(tmpdir, monkeypatch):
    ls = getattr(mltype.cli.cli, "ls")

    new_home = pathlib.Path(str(tmpdir))
    path_languages = new_home / ".mltype" / "languages"
    path_languages.mkdir(parents=True)

    monkeypatch.setenv("HOME", str(new_home))

    runner = CliRunner()
    result = runner.invoke(ls, [])

    assert result.exit_code == 0
    assert result.output == ""

    (path_languages / "a").touch()
    (path_languages / "b").touch()
    (path_languages / "c").mkdir()

    result = runner.invoke(ls, [])

    assert result.exit_code == 0
    assert result.output == "a\nb\n"


@pytest.mark.parametrize("force_perfect", [True, False])
@pytest.mark.parametrize("instant_death", [True, False])
@pytest.mark.parametrize("n_chars", [34, 52])
@pytest.mark.parametrize("output_file", ["some/path", "other/path"])
@pytest.mark.parametrize("use_long", [True, False])
@pytest.mark.parametrize("target_wpm", [33, 55])
def test_random(
    tmpdir,
    monkeypatch,
    force_perfect,
    instant_death,
    n_chars,
    output_file,
    use_long,
    target_wpm,
):
    raw = getattr(mltype.cli.cli, "random")

    fake_main_basic = Mock()
    monkeypatch.setattr("mltype.interactive.main_basic", fake_main_basic)

    runner = CliRunner()
    options = [
        ("f", "force-perfect", force_perfect),
        ("i", "instant-death", instant_death),
        ("n", "n-chars", n_chars),
        ("o", "output-file", output_file),
        ("t", "target-wpm", target_wpm),
    ]

    command = command_composer(("abc",), options, use_long=use_long)
    print(command)  # to know why it failed

    result = runner.invoke(raw, command)

    assert result.exit_code == 0
    fake_main_basic.assert_called_once()

    call = fake_main_basic.call_args

    assert set(call.args[0]).issubset(set("abc"))

    assert call.kwargs == {
        "force_perfect": force_perfect,
        "instant_death": instant_death,
        "output_file": output_file,
        "target_wpm": target_wpm,
    }


@pytest.mark.parametrize("force_perfect", [True, False])
@pytest.mark.parametrize("instant_death", [True, False])
@pytest.mark.parametrize("output_file", ["some/path", "other/path"])
@pytest.mark.parametrize("raw_string", [True, False])
@pytest.mark.parametrize("use_long", [True, False])
@pytest.mark.parametrize("target_wpm", [33, 55])
def test_raw(
    tmpdir,
    monkeypatch,
    force_perfect,
    instant_death,
    output_file,
    raw_string,
    use_long,
    target_wpm,
):
    raw = getattr(mltype.cli.cli, "raw")

    fake_main_basic = Mock()
    monkeypatch.setattr("mltype.interactive.main_basic", fake_main_basic)

    runner = CliRunner()
    options = [
        ("f", "force-perfect", force_perfect),
        ("i", "instant-death", instant_death),
        ("o", "output-file", output_file),
        ("r", "raw-string", raw_string),
        ("t", "target-wpm", target_wpm),
    ]

    command = command_composer(("Hello",), options, use_long=use_long)
    print(command)  # to know why it failed

    result = runner.invoke(raw, command)

    assert result.exit_code == 0
    fake_main_basic.assert_called_once()

    call = fake_main_basic.call_args

    assert call.args == ("Hello",)
    assert call.kwargs == {
        "force_perfect": force_perfect,
        "instant_death": instant_death,
        "output_file": output_file,
        "target_wpm": target_wpm}


@pytest.mark.parametrize("force_perfect", [True, False])
@pytest.mark.parametrize("instant_death", [True, False])
@pytest.mark.parametrize("overwrite", [True, False])
@pytest.mark.parametrize("use_long", [True, False])
@pytest.mark.parametrize("target_wpm", [33, 55])
def test_replay(
    tmpdir,
    monkeypatch,
    force_perfect,
    instant_death,
    overwrite,
    use_long,
    target_wpm,
):
    raw = getattr(mltype.cli.cli, "replay")

    fake_main_replay = Mock()
    monkeypatch.setattr("mltype.interactive.main_replay", fake_main_replay)

    runner = CliRunner()
    options = [
        ("f", "force-perfect", force_perfect),
        ("i", "instant-death", instant_death),
        ("w", "overwrite", overwrite),
        ("t", "target-wpm", target_wpm),
    ]

    command = command_composer(("aa",), options, use_long=use_long)
    print(command)  # to know why it failed

    result = runner.invoke(raw, command)

    assert result.exit_code == 0
    fake_main_replay.assert_called_once()

    call = fake_main_replay.call_args

    assert call.args == tuple()
    assert call.kwargs == {
        "replay_file": "aa",
        "force_perfect": force_perfect,
        "instant_death": instant_death,
        "overwrite": overwrite,
        "target_wpm": target_wpm}

