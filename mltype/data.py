"""Data creating and managing."""
import pathlib
import sys


def file2text(filepath):
    """Read all lines of a file into a string.

    Parameters
    ----------
    filepath : pathlib.Path
        Path to the file

    Returns
    -------
    text : str
        All the text found in the input file.
    """
    with filepath.open("r") as f:
        texts = [l.strip() for l in f.readlines()]

    return " ".join(texts)


def folder2text(folderpath, valid_extensions=None):
    """Collect all files recursively and read into a string."""
    texts = []

    for p in folderpath.rglob("*"):
        if not p.is_file():
            continue

        if valid_extensions is not None and p.suffix not in valid_extensions:
            continue
        try:
            texts.append(file2text(p))
        except UnicodeDecodeError:
            continue
        print(p.name)

    return " ".join(texts)
