"""Collection of utility functions."""
import pathlib


def get_cache_dir(predefined_path=None):
    """Get the cache directory path and potentially create it.

    If no predefined path provided, we simply take `~/.mltype`.
    Note that if one changes the `os.environ["home"]` dynamically
    it will influence the output of this function. this is done
    on purpose to simplify testing.

    Parameters
    ----------
    predefined_path : None or pathlib.Path
        If provided, we just return the same path. We potentially
        create the directory if it does not exist. If it is not
        provided we use `$HOME/.mltype`.

    Returns
    -------
    path : pathlib.Path
        Path to where the caching directory is located.
    """
    if predefined_path is not None:
        path = predefined_path
    else:
        path = pathlib.Path.home() / ".mltype"

    path.mkdir(parents=True, exist_ok=True)

    return path
