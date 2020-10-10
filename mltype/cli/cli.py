"""Command line interface."""
import pathlib
import warnings

import click

warnings.filterwarnings("ignore")


@click.group()
def cli():
    """Tool for improving typing speed and accuracy."""
    pass


@cli.command()
@click.argument("path", type=click.File("r"))
@click.option(
    "-e",
    "--end_line",
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
    "--n_lines",
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
    "--start_line",
    type=int,
    help="the start line of the excerpt to use. needs to be used together "
    "with end-line.",
)
@click.option(
    "-t",
    "--target-wpm",
    type=int,
    help="The desired speed to be shown as a guide",
)
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
):
    """Type text from a file"""
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

    all_lines = [line.strip() for line in path.readlines()]
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
        " ".join(selected_lines),
        force_perfect=force_perfect,
        output_file=output_file,
        instant_death=instant_death,
        target_wpm=target_wpm,
    )


@cli.command()
def list():
    """List all language models"""
    from mltype.utils import get_cache_dir

    languages_dir = get_cache_dir() / "languages"

    if not languages_dir.exists():
        return

    all_names = sorted([x.name for x in languages_dir.iterdir() if x.is_file()])
    for name in all_names:
        print(name)


@cli.command()
@click.argument("path", type=click.Path())
@click.argument("model_name", type=click.Path())
@click.option(
    "-b",
    "--batch-size",
    type=int,
    default=64,
    help="Number of samples in a batch",
    show_default=True,
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
    help="Either zeros or skip. Determines how deal with out of vocabulary "
    "characters",
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
    "-l",
    "--n-layers",
    type=int,
    default=1,
    help="Number of layers in the recurrent network",
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
def train(
    path,
    model_name,
    extensions,
    fill_strategy,
    batch_size,
    dense_size,
    hidden_size,
    max_epochs,
    n_layers,
    train_test_split,
    vocab_size,
    window_size,
):
    """Train a language"""
    from mltype.data import file2text, folder2text
    from mltype.ml import run_train

    path_p = pathlib.Path(str(path))

    if not path_p.exists():
        raise ValueError("The provided path does not exist")

    if path_p.is_file():
        text = file2text(path_p)
    elif path_p.is_dir():
        valid_extensions = (
            extensions.split(",") if extensions is not None else None
        )
        text = folder2text(path_p, valid_extensions=valid_extensions)
    else:
        ValueError("Unrecoggnized object")

    run_train(
        text,
        model_name,
        max_epochs=max_epochs,
        window_size=window_size,
        batch_size=batch_size,
        vocab_size=vocab_size,
        fill_strategy=fill_strategy,
        train_test_split=train_test_split,
        hidden_size=hidden_size,
        dense_size=dense_size,
        n_layers=n_layers,
    )
    print(len(text))
    print("Done")


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
    "-t",
    "--target_wpm",
    type=int,
    help="The desired speed to be shown as a guide",
)
def raw(text, force_perfect, output_file, instant_death, target_wpm):
    """Provide text manually"""
    from mltype.interactive import main_basic

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
    "--target_wpm",
    type=int,
    help="The desired speed to be shown as a guide",
)
@click.option("-w", "--overwrite", is_flag=True)
def replay(replay_file, force_perfect, instant_death, overwrite, target_wpm):
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
    "--target_wpm",
    type=int,
    help="The desired speed to be shown as a guide",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Show progressbar when generating text",
)
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
):
    """Sample text from a language"""
    from mltype.interactive import main_basic
    from mltype.ml import load_model, sample_text_no_window
    from mltype.utils import get_cache_dir

    model_folder = get_cache_dir() / "languages" / model_name

    network, vocabulary = load_model(model_folder)
    text = sample_text_no_window(
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


if __name__ == "__main__":
    cli()
