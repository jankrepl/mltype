"""Collections of tests of the CLI."""
import pathlib
from unittest.mock import Mock

from click.testing import CliRunner
import pytest

import mltype.cli


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
    fun = getattr(mltype.cli, cmd)

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
@pytest.mark.parametrize("target_wpm", [55])
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

    file_ = getattr(mltype.cli, "file")

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
        ("w", "include-whitespace", include_whitespace),
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

    assert isinstance(call[0][0], str)

    assert call[1] == {
        "force_perfect": force_perfect,
        "instant_death": instant_death,
        "output_file": output_file,
        "target_wpm": target_wpm,
    }


@pytest.mark.parametrize("dir_exists", [True, False])
def test_ls(tmpdir, monkeypatch, dir_exists):
    ls = getattr(mltype.cli, "ls")

    new_home = pathlib.Path(str(tmpdir))
    path_languages = new_home / ".mltype" / "languages"

    if dir_exists:
        path_languages.mkdir(parents=True)

    monkeypatch.setenv("HOME", str(new_home))

    runner = CliRunner()
    result = runner.invoke(ls, [])

    assert result.exit_code == 0
    assert result.output == ""

    if not dir_exists:
        return

    (path_languages / "a").touch()
    (path_languages / "b").touch()
    (path_languages / "c").mkdir()

    result = runner.invoke(ls, [])

    assert result.exit_code == 0
    assert result.output == "a\nb\n"


@pytest.mark.parametrize("force_perfect", [True])
@pytest.mark.parametrize("instant_death", [True])
@pytest.mark.parametrize("n_chars", [52])
@pytest.mark.parametrize("output_file", ["some/path"])
@pytest.mark.parametrize("use_long", [True, False])
@pytest.mark.parametrize("target_wpm", [55])
def test_random(
    monkeypatch,
    force_perfect,
    instant_death,
    n_chars,
    output_file,
    use_long,
    target_wpm,
):
    random = getattr(mltype.cli, "random")

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

    assert set(call[0][0]).issubset(set("abc"))

    assert call[1] == {
        "force_perfect": force_perfect,
        "instant_death": instant_death,
        "output_file": output_file,
        "target_wpm": target_wpm,
    }


@pytest.mark.parametrize("force_perfect", [True])
@pytest.mark.parametrize("instant_death", [True])
@pytest.mark.parametrize("output_file", ["other/path"])
@pytest.mark.parametrize("raw_string", [False])
@pytest.mark.parametrize("use_long", [True, False])
@pytest.mark.parametrize("target_wpm", [33])
def test_raw(
    monkeypatch,
    force_perfect,
    instant_death,
    output_file,
    raw_string,
    use_long,
    target_wpm,
):
    raw = getattr(mltype.cli, "raw")

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

    assert call[0] == ("Hello",)
    assert call[1] == {
        "force_perfect": force_perfect,
        "instant_death": instant_death,
        "output_file": output_file,
        "target_wpm": target_wpm,
    }


@pytest.mark.parametrize("force_perfect", [True])
@pytest.mark.parametrize("instant_death", [True])
@pytest.mark.parametrize("overwrite", [True])
@pytest.mark.parametrize("use_long", [True, False])
@pytest.mark.parametrize("target_wpm", [55])
def test_replay(
    monkeypatch,
    force_perfect,
    instant_death,
    overwrite,
    use_long,
    target_wpm,
):
    replay = getattr(mltype.cli, "replay")

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

    assert call[0] == tuple()
    assert call[1] == {
        "replay_file": "aa",
        "force_perfect": force_perfect,
        "instant_death": instant_death,
        "overwrite": overwrite,
        "target_wpm": target_wpm,
    }


@pytest.mark.parametrize("force_perfect", [True])
@pytest.mark.parametrize("instant_death", [True])
@pytest.mark.parametrize("n_chars", [3])
@pytest.mark.parametrize("output_file", ["other/path"])
@pytest.mark.parametrize("random_state", [8])
@pytest.mark.parametrize("use_long", [True, False])
@pytest.mark.parametrize("target_wpm", [33])
def test_sample(
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

    sample = getattr(mltype.cli, "sample")

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

    assert call_1[0][0] == n_chars

    assert call_1[1] == {
        "initial_text": starting_text,
        "random_state": random_state,
        "top_k": top_k,
        "verbose": verbose,
    }

    fake_main_basic.assert_called_once()

    call_2 = fake_main_basic.call_args

    assert call_2[0] == ("amazing",)
    assert call_2[1] == {
        "force_perfect": force_perfect,
        "instant_death": instant_death,
        "output_file": output_file,
        "target_wpm": target_wpm,
    }


@pytest.mark.parametrize("batch_size", [34])
@pytest.mark.parametrize("checkpoint_path", ["some_path"])
@pytest.mark.parametrize("dense_size", [30])
@pytest.mark.parametrize("extensions", [".py"])
@pytest.mark.parametrize("fill_strategy", ["zeros"])
@pytest.mark.parametrize("gpus", [4])
@pytest.mark.parametrize("hidden_size", [3])
@pytest.mark.parametrize("illegal_chars", ["abc"])
@pytest.mark.parametrize("n_layers", [7])
@pytest.mark.parametrize("use_mlflow", [True])
@pytest.mark.parametrize("max_epochs", [6])
@pytest.mark.parametrize("output_path", ["some/path"])
@pytest.mark.parametrize("early_stopping", [True])
@pytest.mark.parametrize("use_long", [True, False])
@pytest.mark.parametrize("train_test_split", [0.8])
@pytest.mark.parametrize("vocab_size", [5])
@pytest.mark.parametrize("window_size", [1])
def test_train(
    tmpdir,
    monkeypatch,
    batch_size,
    checkpoint_path,
    dense_size,
    extensions,
    fill_strategy,
    gpus,
    hidden_size,
    illegal_chars,
    n_layers,
    use_mlflow,
    max_epochs,
    output_path,
    early_stopping,
    use_long,
    train_test_split,
    vocab_size,
    window_size,
):
    train = getattr(mltype.cli, "train")

    path_dir = pathlib.Path(str(tmpdir))
    path_file = path_dir / "1.txt"
    path_file.write_text("HELLOOOO THEREEEE")

    fake_run_train = Mock()
    monkeypatch.setattr("mltype.ml.run_train", fake_run_train)

    runner = CliRunner()
    options = [
        ("b", "batch-size", batch_size),
        ("c", "checkpoint-path", checkpoint_path),
        ("d", "dense-size", dense_size),
        ("e", "extensions", extensions),
        ("f", "fill-strategy", fill_strategy),
        ("g", "gpus", gpus),
        ("h", "hidden-size", hidden_size),
        ("i", "illegal-chars", illegal_chars),
        ("l", "n-layers", n_layers),
        ("m", "use-mlflow", use_mlflow),
        ("n", "max-epochs", max_epochs),
        ("o", "output-path", output_path),
        ("s", "early-stopping", early_stopping),
        ("t", "train-test-split", train_test_split),
        ("v", "vocab-size", vocab_size),
        ("w", "window-size", window_size),
    ]

    command = command_composer(
        (str(path_dir), str(path_file), "naame"), options, use_long=use_long
    )

    print(command)  # to know why it failed

    result = runner.invoke(train, command)
    print(result.output)

    assert result.exit_code == 0

    fake_run_train.assert_called_once()

    call = fake_run_train.call_args

    assert isinstance(call[0][0], list)
    assert call[0][1] == "naame"

    assert call[1] == dict(
        batch_size=batch_size,
        checkpoint_path=checkpoint_path,
        dense_size=dense_size,
        fill_strategy=fill_strategy,
        gpus=gpus,
        hidden_size=hidden_size,
        illegal_chars=illegal_chars,
        n_layers=n_layers,
        use_mlflow=use_mlflow,
        max_epochs=max_epochs,
        output_path=output_path,
        early_stopping=early_stopping,
        train_test_split=train_test_split,
        vocab_size=vocab_size,
        window_size=window_size,
    )
