"""Command line interface."""
from configparser import ConfigParser
from collections import Counter
import pathlib
import pprint
import warnings
import sys

import click
import click_config_file

from mltype.utils import (
    get_config_file_path,
    set_config_file_path,
)

warnings.filterwarnings("ignore")


def provider(file_path, cmd_name):
    """Read a section from a config file.

    This function is used for the `click_config_file`. The desired behavior
    is that by default we are reading the config file from
    `~/.mltype/config.ini`. If the user manually specifices the `--config`
    option (available in all subcommands) then one can provide a custom path.

    Importantly, one needs to use the full option names and replace
    "-" with "_". For example, `t = 50` and `target-wpm = 50 are invalid` and
    `target_wpm = 50` is valid.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to where the config file is located.

    cmd_name : str
        Name of the command.

    Returns
    -------
    dict
        Keys represent the command line options and the values
        are their desired values.
    """
    # Drop the default behaviour
    provided_path = pathlib.Path(file_path)
    default_click_path = pathlib.Path(click.get_app_dir(cmd_name)) / "config"

    if provided_path == default_click_path:
        # The user did not provide custom config file via --config
        used_path = get_config_file_path()
        set_config_file_path(None)  # None == defaulting to cache dir
        if not used_path.exists():
            return {}
    else:
        # The user did provide custom config file via --config
        used_path = provided_path
        # Enforce existence of custom config file
        if not used_path.exists():
            raise FileNotFoundError("The configuration file not found.")
        # Redefine actual config file path for non-CLI settings
        set_config_file_path(used_path)

    cp = ConfigParser()
    cp.read(used_path)

    if cmd_name in cp.sections():
        return dict(cp[cmd_name])
    else:
        return {}


@click.group()
def cli():  # noqa: D400
    """Tool for improving typing speed and accuracy"""


@cli.command()
@click.argument("path", type=click.File("r"))
@click.option(
    "-e",
    "--end-line",
    type=int,
    help="The end line of the excerpt to use. Needs to be used together with "
    "start-line.",
)
@click.option(
    "-f",
    "--force-perfect",
    is_flag=True,
    help="All characters need to be typed correctly",
)
@click.option(
    "-i",
    "--instant-death",
    is_flag=True,
    help="End game after the first mistake",
)
@click.option(
    "-l",
    "--n-lines",
    type=int,
    help="Number of consecutive lines to be selected at random. Cannot be "
    "used together with start-line and end-line.",
)
@click.option(
    "-o",
    "--output-file",
    type=click.Path(),
    help="Path to where to save the result file",
)
@click.option("-r", "--random-state", type=int)
@click.option(
    "-s",
    "--start-line",
    type=int,
    help="The start line of the excerpt to use. needs to be used together "
    "with end-line.",
)
@click.option(
    "-t",
    "--target-wpm",
    type=int,
    help="The desired speed to be shown as a guide",
)
@click.option(
    "-w",
    "--include-whitespace",
    is_flag=True,
    help="Include whitespace characters",
)
@click_config_file.configuration_option(provider=provider, implicit=True)
def file(
    path,
    start_line,
    end_line,
    n_lines,
    random_state,
    force_perfect,
    instant_death,
    output_file,
    target_wpm,
    include_whitespace,
):  # noqa: D400
    """Type text from a file"""  # noqa: D400
    import numpy as np

    from mltype.interactive import main_basic

    # validation
    mode_exact = start_line is not None and end_line is not None
    mode_random = n_lines is not None

    if not (mode_exact ^ mode_random):
        raise ValueError(
            "One can either provide start and end "
            "line or the n_lines to be randomly selected."
        )

    all_lines = [line for line in path.readlines()]
    if not include_whitespace:
        all_lines = [f"{line.strip()} " for line in all_lines]

    n_all_lines = len(all_lines)

    if mode_exact:
        if random_state is not None:
            raise ValueError(
                "One can only use random state in combination with n-lines"
            )

        if not 0 <= start_line < end_line < len(all_lines):
            raise ValueError(
                f"Selected lines fall outside of the range (0, {n_all_lines})"
                " or are in a wrong order."
            )

    if mode_random:
        if random_state is not None:
            np.random.seed(random_state)

        start_line = np.random.randint(n_all_lines - n_lines)
        end_line = start_line + n_lines

    selected_lines = all_lines[start_line:end_line]
    main_basic(
        "".join(selected_lines),
        force_perfect=force_perfect,
        output_file=output_file,
        instant_death=instant_death,
        target_wpm=target_wpm,
    )


@cli.command()
@click_config_file.configuration_option(provider=provider, implicit=True)
def ls():  # noqa: D400
    """List all language models"""
    from mltype.utils import get_cache_dir, get_config_file

    cp = get_config_file()
    try:
        predefined_path = cp["general"]["models_dir"]
        languages_dir = get_cache_dir(predefined_path)

    except KeyError:
        languages_dir = get_cache_dir() / "languages"

    if not languages_dir.exists():
        return

    all_names = sorted([x.name for x in languages_dir.iterdir() if x.is_file()])
    for name in all_names:
        print(name)


@cli.command()
@click.argument("path", nargs=-1, type=click.Path())
@click.argument("model_name", type=str)
@click.option(
    "-b",
    "--batch-size",
    type=int,
    default=64,
    help="Number of samples in a batch",
    show_default=True,
)
@click.option(
    "-c",
    "--checkpoint-path",
    type=click.Path(),
    help="Load a checkpoint and continue training it",
)
@click.option(
    "-d",
    "--dense-size",
    type=int,
    default=128,
    help="Size of the dense layer",
    show_default=True,
)
@click.option(
    "-e",
    "--extensions",
    type=str,
    help="Comma-separated list of allowed extensions",
)
@click.option(
    "-f",
    "--fill-strategy",
    type=str,
    default="skip",
    help="Either zeros or skip. Determines how to deal with out of vocabulary "
    "characters",
    show_default=True,
)
@click.option(
    "-g",
    "--gpus",
    type=int,
    help="Number of gpus. If not specified, then none. If -1, then all.",
    show_default=True,
)
@click.option(
    "-h",
    "--hidden-size",
    type=int,
    default=32,
    help="Size of the hidden state",
    show_default=True,
)
@click.option(
    "-i",
    "--illegal-chars",
    type=str,
    help="Characters to exclude from the vocabulary",
    show_default=True,
)
@click.option(
    "-l",
    "--n-layers",
    type=int,
    default=1,
    help="Number of layers in the recurrent network",
    show_default=True,
)
@click.option(
    "-m",
    "--use-mlflow",
    is_flag=True,
    help="Use MLFlow for logging",
    show_default=True,
)
@click.option(
    "-n",
    "--max-epochs",
    type=int,
    default=10,
    help="Maximum number of epochs",
    show_default=True,
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(),
    help="Custom path where to save the trained models and logging details. "
    "If not provided it defaults to `~/.mltype`.",
)
@click.option(
    "-s",
    "--early-stopping",
    is_flag=True,
    help="Enable early stopping based on validation loss",
    show_default=True,
)
@click.option(
    "-t",
    "--train-test-split",
    type=float,
    default=0.9,
    help="Train test split - value between (0, 1)",
    show_default=True,
)
@click.option(
    "-v",
    "--vocab-size",
    type=int,
    default=50,
    help="Number of most frequent characters to include in the vocabulary",
    show_default=True,
)
@click.option(
    "-w",
    "--window-size",
    type=int,
    default=50,
    help="Number of previous characters to consider for prediction",
    show_default=True,
)
@click_config_file.configuration_option(provider=provider, implicit=True)
def train(
    path,
    model_name,
    checkpoint_path,
    extensions,
    fill_strategy,
    illegal_chars,
    gpus,
    batch_size,
    dense_size,
    hidden_size,
    max_epochs,
    early_stopping,
    n_layers,
    output_path,
    train_test_split,
    use_mlflow,
    vocab_size,
    window_size,
):  # noqa: D400
    """Train a language"""
    params = locals()
    from mltype.data import file2text, folder2text
    from mltype.ml import run_train
    from mltype.utils import print_section

    with print_section(" Parameters ", drop_end=True):
        pprint.pprint(params)

    all_texts = []
    with print_section(" Reading file(s) ", drop_end=True):
        for p in path:
            path_p = pathlib.Path(str(p))

            if not path_p.exists():
                raise ValueError(
                    "The provided path does not exist"
                )  # pragma: no cover

            if path_p.is_file():
                texts = [file2text(path_p)]
            elif path_p.is_dir():
                valid_extensions = (
                    extensions.split(",") if extensions is not None else None
                )
                texts = folder2text(path_p, valid_extensions=valid_extensions)
            else:
                ValueError("Unrecognized object")  # pragma: no cover

            all_texts.extend(texts)

    if not all_texts:
        raise ValueError("Did not manage to read any text")  # pragma: no cover

    run_train(
        all_texts,
        model_name,
        max_epochs=max_epochs,
        window_size=window_size,
        batch_size=batch_size,
        vocab_size=vocab_size,
        fill_strategy=fill_strategy,
        illegal_chars=illegal_chars,
        train_test_split=train_test_split,
        hidden_size=hidden_size,
        dense_size=dense_size,
        n_layers=n_layers,
        use_mlflow=use_mlflow,
        early_stopping=early_stopping,
        gpus=gpus,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
    )
    print("Done")


@cli.command()
@click.argument("characters")
@click.option(
    "-f",
    "--force-perfect",
    is_flag=True,
    help="All characters need to be typed correctly",
)
@click.option(
    "-i",
    "--instant-death",
    is_flag=True,
    help="End game after the first mistake",
)
@click.option(
    "-n",
    "--n-chars",
    type=int,
    default=100,
    show_default=True,
    help="Number of characters to generate",
)
@click.option(
    "-o",
    "--output-file",
    type=click.Path(),
    help="Path to where to save the result file",
)
@click.option(
    "-t",
    "--target-wpm",
    type=int,
    help="The desired speed to be shown as a guide",
)
@click_config_file.configuration_option(provider=provider, implicit=True)
def random(
    characters, force_perfect, instant_death, n_chars, output_file, target_wpm
):  # noqa: D400
    """Sample characters randomly from a vocabulary"""
    import numpy as np

    from mltype.interactive import main_basic

    c = Counter(characters)
    vocabulary = list(c.keys())
    counts = np.array([c[x] for x in vocabulary])

    p = counts / counts.sum()

    text = "".join(np.random.choice(vocabulary, size=n_chars, p=p))

    main_basic(
        text,
        force_perfect=force_perfect,
        output_file=output_file,
        instant_death=instant_death,
        target_wpm=target_wpm,
    )


@cli.command()
@click.argument("text")
@click.option(
    "-f",
    "--force-perfect",
    is_flag=True,
    help="All characters need to be typed correctly",
)
@click.option(
    "-i",
    "--instant-death",
    is_flag=True,
    help="End game after the first mistake",
)
@click.option(
    "-o",
    "--output-file",
    type=click.Path(),
    help="Path to where to save the result file",
)
@click.option(
    "-r",
    "--raw-string",
    is_flag=True,
    help="If active, then newlines and tabs are not seen as special characters",
)
@click.option(
    "-t",
    "--target-wpm",
    type=int,
    help="The desired speed to be shown as a guide",
)
@click_config_file.configuration_option(provider=provider, implicit=True)
def raw(
    text, force_perfect, output_file, instant_death, target_wpm, raw_string
):  # noqa: D400
    """Provide text manually"""
    import codecs

    from mltype.interactive import main_basic

    if not raw_string:
        text = codecs.decode(text, "unicode_escape")

    main_basic(
        text,
        force_perfect=force_perfect,
        output_file=output_file,
        instant_death=instant_death,
        target_wpm=target_wpm,
    )


@cli.command()
@click.argument("replay-file", type=click.Path())
@click.option(
    "-f",
    "--force-perfect",
    is_flag=True,
    help="All characters need to be typed correctly",
)
@click.option(
    "-i",
    "--instant-death",
    is_flag=True,
    help="End game after the first mistake",
)
@click.option(
    "-t",
    "--target-wpm",
    type=int,
    help="The desired speed to be shown as a guide",
)
@click.option(
    "-w", "--overwrite", is_flag=True, help="Overwrite in place if faster"
)
@click_config_file.configuration_option(provider=provider, implicit=True)
def replay(
    replay_file, force_perfect, instant_death, overwrite, target_wpm
):  # noqa: D400
    """Compete against a past performance"""
    from mltype.interactive import main_replay

    main_replay(
        replay_file=replay_file,
        force_perfect=force_perfect,
        instant_death=instant_death,
        overwrite=overwrite,
        target_wpm=target_wpm,
    )


@cli.command()
@click.argument("model-name")
@click.option(
    "-f",
    "--force-perfect",
    is_flag=True,
    help="All characters need to be typed correctly",
)
@click.option(
    "-i",
    "--instant-death",
    is_flag=True,
    help="End game after the first mistake",
)
@click.option(
    "-k",
    "--top-k",
    type=int,
    help="Consider the top k most probable characters",
)
@click.option(
    "-n",
    "--n-chars",
    type=int,
    default=100,
    show_default=True,
    help="Number of characters to generate",
)
@click.option(
    "-o",
    "--output-file",
    type=click.Path(),
    help="Path to where to save the result file",
)
@click.option(
    "-r",
    "--random-state",
    type=int,
    help="Random state for reproducible results",
)
@click.option(
    "-s",
    "--starting-text",
    type=str,
    help="Initial text used as a starting condition",
)
@click.option(
    "-t",
    "--target-wpm",
    type=int,
    help="The desired speed to be shown as a guide",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Show progressbar when generating text",
)
@click_config_file.configuration_option(provider=provider, implicit=True)
def sample(
    model_name,
    n_chars,
    force_perfect,
    instant_death,
    random_state,
    output_file,
    target_wpm,
    verbose,
    starting_text,
    top_k,
):  # noqa: D400
    """Sample text from a language"""
    sys.modules["mlflow"] = None
    from mltype.interactive import main_basic
    from mltype.ml import load_model, sample_text
    from mltype.utils import get_cache_dir, get_config_file

    cp = get_config_file()
    try:
        predefined_path = cp["general"]["models_dir"]
        languages_dir = get_cache_dir(predefined_path)

    except KeyError:
        languages_dir = get_cache_dir() / "languages"

    model_path = languages_dir / model_name

    network, vocabulary = load_model(model_path)
    text = sample_text(
        n_chars,
        network,
        vocabulary,
        initial_text=starting_text,
        random_state=random_state,
        top_k=top_k,
        verbose=verbose,
    )

    main_basic(
        text,
        force_perfect=force_perfect,
        output_file=output_file,
        instant_death=instant_death,
        target_wpm=target_wpm,
    )
