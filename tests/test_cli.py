"""Collections of tests of the CLI."""
from configparser import ConfigParser
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


@pytest.mark.parametrize("from_config", [True, False])
@pytest.mark.parametrize("dir_exists", [True, False])
def test_ls(tmpdir, monkeypatch, dir_exists, from_config):
    # tmpdir
    # ---- .mltype/
    # --------- config.ini  # if from_config
    # --------- custom/  # if from_config and dir_exists
    # --------- langugages/  # if not from_config and dir_exists
    ls = getattr(mltype.cli, "ls")

    tmpdir = pathlib.Path(str(tmpdir))
    path_cache = tmpdir / ".mltype"
    path_models = path_cache / ("custom" if from_config else "languages")

    path_cache.mkdir()

    if dir_exists:
        path_models.mkdir(parents=True)

    if from_config:
        path_config = path_cache / "config.ini"
        with path_config.open("w") as f:
            cp = ConfigParser()
            cp["general"] = {"models_dir": str(path_models)}
            cp.write(f)

    def fake_get_cache_dir(path=None):
        return pathlib.Path(path) if path is not None else path_cache

    monkeypatch.setattr("mltype.utils.get_cache_dir", fake_get_cache_dir)

    runner = CliRunner()
    result = runner.invoke(ls, [])

    assert result.exit_code == 0
    assert result.output == ""

    if not dir_exists:
        return

    (path_models / "a").touch()
    (path_models / "b").touch()
    (path_models / "c").mkdir()

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


class TestConfigFile:
    def test_nonexistent_config(self, tmpdir):
        """Get an error when the custom config file does not exist.

        Only run for ls but the logic is similar for all of them.
        """
        config_path = pathlib.Path(tmpdir) / "config.ini"

        ls_fun = getattr(mltype.cli, "ls")
        runner = CliRunner()
        result = runner.invoke(ls_fun, [f"--config={config_path}"])
        assert result.exit_code != 0

        config_path.touch()
        result = runner.invoke(ls_fun, [f"--config={config_path}"])
        assert result.exit_code == 0

    def test_implicit_goes_to_cache_dir(self, monkeypatch, tmpdir):
        """Check not providing config explicitly defaults to cache dir.

        Only run for ls but the logic is similar for all of them.
        """
        cache_dir = pathlib.Path(str(tmpdir)) / ".mltype"
        monkeypatch.setattr("mltype.utils.get_cache_dir", lambda: cache_dir)

        ls_fun = getattr(mltype.cli, "ls")
        runner = CliRunner()
        result = runner.invoke(ls_fun, [])

        assert result.exit_code == 0

    @pytest.mark.parametrize("location", ["inside_cache", "outside_cache"])
    def test_overall(self, location, monkeypatch, tmpdir):
        """Check whether the readinig of config parameters works.

        Done with raw however kind of covers all.
        """
        # Path preparations
        tmpdir = pathlib.Path(str(tmpdir))
        cache_dir = tmpdir / "cache_dir"
        other_dir = tmpdir / "other"

        cache_dir.mkdir()
        other_dir.mkdir()

        if location == "inside_cache":
            config_file_path = cache_dir / "config.ini"
        elif location == "outside_cache":
            config_file_path = other_dir / "config.ini"
        else:
            raise ValueError()

        config_file_path.touch()

        # More preparations
        monkeypatch.setattr("mltype.utils.get_cache_dir", lambda: cache_dir)

        raw_fn = getattr(mltype.cli, "raw")

        fake_main_basic = Mock()

        monkeypatch.setattr("mltype.interactive.main_basic", fake_main_basic)

        runner = CliRunner()

        # Experiment 1 - only CLI
        inputs_1 = ["whatever", "--target-wpm", "12"]
        if location == "outside_cache":
            inputs_1.extend(["--config", str(config_file_path)])

        result_1 = runner.invoke(raw_fn, inputs_1)

        assert result_1.exit_code == 0

        assert fake_main_basic.call_count == 1
        kwargs_1 = fake_main_basic.call_args[1]
        assert kwargs_1["target_wpm"] == 12

        # Experiment 2 - nowhere
        inputs_2 = ["whatever"]
        if location == "outside_cache":
            inputs_2.extend(["--config", str(config_file_path)])

        result_2 = runner.invoke(raw_fn, inputs_2)

        assert result_2.exit_code == 0

        assert fake_main_basic.call_count == 2
        kwargs_2 = fake_main_basic.call_args[1]
        assert kwargs_2["target_wpm"] is None  # that is the default

        # Experiment 3 - only in config (correctly specified)
        with config_file_path.open("w") as f:
            config = ConfigParser()
            config["raw"] = {}
            config["raw"]["target_wpm"] = "44"
            config.write(f)
        inputs_3 = ["whatever"]
        if location == "outside_cache":
            inputs_3.extend(["--config", str(config_file_path)])

        result_3 = runner.invoke(raw_fn, inputs_3)

        assert result_3.exit_code == 0

        assert fake_main_basic.call_count == 3
        kwargs_3 = fake_main_basic.call_args[1]
        assert kwargs_3["target_wpm"] == 44

        # Experiment 4 - CLI has preference over config
        config_file_path.unlink()  # just to make it explicit
        with config_file_path.open("w") as f:
            config = ConfigParser()
            config["raw"] = {}
            config["raw"]["target_wpm"] = "44"
            config.write(f)
        inputs_4 = ["whatever", "--target-wpm", "13"]
        if location == "outside_cache":
            inputs_4.extend(["--config", str(config_file_path)])

        result_4 = runner.invoke(raw_fn, inputs_4)

        assert result_4.exit_code == 0

        assert fake_main_basic.call_count == 4
        kwargs_4 = fake_main_basic.call_args[1]
        assert kwargs_4["target_wpm"] == 13

        # Experiment 5 - misspelled in config (taking a default)
        config_file_path.unlink()  # just to make it explicit
        with config_file_path.open("w") as f:
            config = ConfigParser()
            config["raw"] = {}
            config["raw"]["target-wpm"] = "48"  # hyphen not allowed in config
            config.write(f)
        inputs_5 = ["whatever"]
        if location == "outside_cache":
            inputs_5.extend(["--config", str(config_file_path)])

        result_5 = runner.invoke(raw_fn, inputs_5)

        assert result_5.exit_code == 0

        assert fake_main_basic.call_count == 5
        kwargs_5 = fake_main_basic.call_args[1]
        assert kwargs_5["target_wpm"] is None  # that is the default

        # Experiment 6 - another misspelled in config (taking a default)
        config_file_path.unlink()  # just to make it explicit
        with config_file_path.open("w") as f:
            config = ConfigParser()
            config["raw"] = {}
            config["raw"]["--target_wpm"] = "49"  # -- not supported in config
            config.write(f)
        inputs_6 = ["whatever"]
        if location == "outside_cache":
            inputs_6.extend(["--config", str(config_file_path)])

        result_6 = runner.invoke(raw_fn, inputs_6)

        assert result_6.exit_code == 0

        assert fake_main_basic.call_count == 6
        kwargs_6 = fake_main_basic.call_args[1]
        assert kwargs_6["target_wpm"] is None  # that is the default

        # Experiment 7 - case insensitive config
        config_file_path.unlink()  # just to make it explicit
        with config_file_path.open("w") as f:
            config = ConfigParser()
            config["raw"] = {}
            config["raw"]["tArgEt_wPm"] = "22"  # -- not supported in config
            config.write(f)
        inputs_7 = ["whatever"]
        if location == "outside_cache":
            inputs_7.extend(["--config", str(config_file_path)])

        result_7 = runner.invoke(raw_fn, inputs_7)

        assert result_7.exit_code == 0

        assert fake_main_basic.call_count == 7
        kwargs_7 = fake_main_basic.call_args[1]
        assert kwargs_7["target_wpm"] is 22  # that is the default
