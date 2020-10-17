"""Data creating and managing."""


def file2text(filepath, verbose=True):
    """Read all lines of a file into a string.

        Note that we destroy all the new line characters
        and all the whitespace charecters on both ends
        of the line. Note that this is very radical
        for source code of programming languages or
        similar.

    Parameters
    ----------
    filepath : pathlib.Path
        Path to the file

    verbose : bool
        If True, we print the name of the file.

    Returns
    -------
    text : str
        All the text found in the input file.
    """
    with filepath.open("r") as f:
        texts = [line.strip() for line in f.readlines()]
        texts = [x for x in texts if x and not x.isspace()]

    if verbose:
        print(filepath.name)

    return " ".join(texts)


def folder2text(folderpath, valid_extensions=None):
    """Collect all files recursively and read into a list of strings."""
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

    return texts
