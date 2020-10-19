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
        if v is False or v is None:
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

@pytest.mark.parametrize("end_line", [None, 5, 10])
@pytest.mark.parametrize("force_perfect", [True])
@pytest.mark.parametrize("include_whitespace", [False])
@pytest.mark.parametrize("instant_death", [True])
@pytest.mark.parametrize("n_lines", [None, 4])
@pytest.mark.parametrize("output_file", ["other/path"])
@pytest.mark.parametrize("random_state", [None, 7])
@pytest.mark.parametrize("start_line", [None, 9])
@pytest.mark.parametrize("use_long", [True, False])
@pytest.mark.parametrize("target_wpm", [33, 55])
def test_file(
    tmpdir,
    monkeypatch,
    end_line,
    force_perfect,
    include_whitespace,
    instant_death,
    n_lines,
    output_file,
    random_state,
    start_line,
    use_long,
    target_wpm,
):
    file_path = pathlib.Path(str(tmpdir)) / "texts.txt"
    file_path.write_text("\n".join(30 * ["sds"]))

    file_ = getattr(mltype.cli.cli, "file")

    fake_main_basic = Mock()
    monkeypatch.setattr("mltype.interactive.main_basic", fake_main_basic)

    runner = CliRunner()
    options = [
        ("e", "end-line", end_line),
        ("f", "force-perfect", force_perfect),
        ("i", "instant-death", instant_death),
        ("l", "n-lines", n_lines),
        ("o", "output-file", output_file),
        ("r", "random-state", random_state),
        ("s", "start-line", start_line),
        ("t", "target-wpm", target_wpm),
        ("w", "include-whitespace", include_whitespace)
    ]

    command = command_composer((str(file_path),), options, use_long=use_long)
    print(command)  # to know why it failed

    result = runner.invoke(file_, command)

    mode_exact = start_line is not None and end_line is not None
    mode_random = n_lines is not None

    if not (mode_exact ^ mode_random):
        assert result.exit_code != 0
        return

    if mode_exact:
        if random_state is not None:
            assert result.exit_code != 0
            return

        if start_line >= end_line:
            assert result.exit_code != 0
            return

    print(result.output)
    assert result.exit_code == 0
    fake_main_basic.assert_called_once()

    call = fake_main_basic.call_args

    assert isinstance(call.args[0], str)

    assert call.kwargs == {
        "force_perfect": force_perfect,
        "instant_death": instant_death,
        "output_file": output_file,
        "target_wpm": target_wpm,
    }

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
    random = getattr(mltype.cli.cli, "random")

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

    result = runner.invoke(random, command)

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
    replay = getattr(mltype.cli.cli, "replay")

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

    result = runner.invoke(replay, command)

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

@pytest.mark.parametrize("force_perfect", [True, False])
@pytest.mark.parametrize("instant_death", [True, False])
@pytest.mark.parametrize("n_chars", [3, 12])
@pytest.mark.parametrize("output_file", ["some/path", "other/path"])
@pytest.mark.parametrize("random_state", [6, 8])
@pytest.mark.parametrize("use_long", [True, False])
@pytest.mark.parametrize("target_wpm", [33, 55])
def test_sample(
    tmpdir,
    monkeypatch,
    force_perfect,
    instant_death,
    n_chars,
    output_file,
    random_state,
    use_long,
    target_wpm,
):
    # prevent parametrize from being huge
    verbose = True
    starting_text = "theeere"
    top_k = 32

    sample = getattr(mltype.cli.cli, "sample")

    fake_main_basic = Mock()
    fake_load_model = Mock(return_value=(Mock(), Mock()))
    fake_sample_text = Mock(return_value="amazing")

    monkeypatch.setattr("mltype.interactive.main_basic", fake_main_basic)
    monkeypatch.setattr("mltype.ml.load_model", fake_load_model)
    monkeypatch.setattr("mltype.ml.sample_text", fake_sample_text)

    runner = CliRunner()
    options = [
        ("f", "force-perfect", force_perfect),
        ("i", "instant-death", instant_death),
        ("k", "top-k", top_k),
        ("n", "n-chars", n_chars),
        ("o", "output-file", output_file),
        ("r", "random-state", random_state),
        ("s", "starting-text", starting_text),
        ("t", "target-wpm", target_wpm),
        ("v", "verbose", verbose),
    ]

    command = command_composer(("something",), options, use_long=use_long)
    print(command)  # to know why it failed

    result = runner.invoke(sample, command)
    print(result.output)

    assert result.exit_code == 0

    fake_sample_text.assert_called_once()

    call_1 = fake_sample_text.call_args

    assert call_1.args[0] == n_chars

    assert call_1.kwargs == {
        "initial_text": starting_text,
        "random_state": random_state,
        "top_k": top_k,
        "verbose": verbose}

    fake_main_basic.assert_called_once()

    call_2 = fake_main_basic.call_args

    assert call_2.args == ("amazing",)
    assert call_2.kwargs == {
        "force_perfect": force_perfect,
        "instant_death": instant_death,
        "output_file": output_file,
        "target_wpm": target_wpm}

