"""Collection of utility functions."""
from contextlib import contextmanager
from datetime import datetime
import pathlib
import shutil


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
    path = pathlib.Path(get_config(predefined_path or pathlib.Path.home() / ".mltype", "cache_dir"))
    path.mkdir(parents=True, exist_ok=True)

    return path


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

def get_config(path, property):
    from configparser import ConfigParser
    config_object = ConfigParser()
    config_path = path / "config.ini"
    if (config_path).is_file():
        config_object.read(config_path)
        return pathlib.Path(config_object["GENERAL"][property])
    else:
        config_object["GENERAL"] = {
            "cache_dir": path
        }
        with open(config_path, 'w') as conf:
            config_object.write(conf)
        return path
        
