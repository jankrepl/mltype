"""Collection of utility functions."""
from configparser import ConfigParser
from contextlib import contextmanager
from datetime import datetime
import pathlib
import shutil

CONFIG_FILE_PATH = None


def get_cache_dir(predefined_path=None):
    """Get the cache directory path and potentially create it.

    If no predefined path provided, we simply take `~/.mltype`.
    Note that if one changes the `os.environ["home"]` dynamically
    it will influence the output of this function. this is done
    on purpose to simplify testing.

    Parameters
    ----------
    predefined_path : None or pathlib.Path or str
        If provided, we just return the same path. We potentially
        create the directory if it does not exist. If it is not
        provided we use `$HOME/.mltype`.

    Returns
    -------
    path : pathlib.Path
        Path to where the caching directory is located.
    """
    if predefined_path is not None:
        path = pathlib.Path(str(predefined_path))
    else:
        path = pathlib.Path.home() / ".mltype"

    path.mkdir(parents=True, exist_ok=True)

    return path


def get_config_file():
    """Get config file.

    Returns
    -------
    ConfigParser
        Instance of the configuration parser.
    """
    cp = ConfigParser()
    cp.read(get_config_file_path())  # if file does not exist no error

    return cp


def get_config_file_path():
    """Get path to the configuration file.

    Returns
    -------
    pathlib.Path
        Position of the configuration file. Note that
        it will depend on the global variable `CONFIG_FILE_PATH`
        that can get modified by the CLI. However, by default
        we put it into `~/.mltype/config.ini`.

    """
    return CONFIG_FILE_PATH or get_cache_dir() / "config.ini"


def set_config_file_path(path):
    """Set path to the configuration file."""
    global CONFIG_FILE_PATH  # oops, did not find a better solution
    CONFIG_FILE_PATH = path


def get_mlflow_artifacts_path(client, run_id):
    """Get path to where the artifacts are located.

    The benefit is that we can log any file into it and even
    create a custom folder hierarachy.

    Parameters
    ----------
    client : mlflow.tracking.MlflowClient
        Client.

    run_id : str
        Unique identifier of a run.

    Returns
    -------
    path : pathlib.Path
        Path to the root folder of artifacts.
    """
    artifacts_uri = client.get_run(run_id).info.artifact_uri
    path_str = artifacts_uri.partition("file:")[2]
    path = pathlib.Path(path_str)

    return path


@contextmanager
def print_section(name, fill_char="=", drop_end=False, add_ts=True):
    """Print nice section blocks.

    Parameters
    ----------
    name : str
        Name of the section.

    fill_char : str
        Character to be used for filling the line.

    drop_end : bool
        If True, the ending line is not printed.

    add_ts : bool
        We add a time step to the heading.
    """
    if len(fill_char) != 1:
        raise ValueError("The fill_char needs to have exactly one character")

    if add_ts:
        ts = datetime.now().strftime("%H:%M:%S")
        title = f"{name}| {ts} "
    else:
        title = name

    width, _ = shutil.get_terminal_size()
    print(title.center(width, fill_char))

    yield

    if not drop_end:
        print(width * fill_char)
