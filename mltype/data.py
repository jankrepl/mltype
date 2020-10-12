"""Data creating and managing."""


def file2text(filepath, keep_new_lines=True):
    """Read all lines of a file into a string.

    Parameters
    ----------
    filepath : pathlib.Path
        Path to the file

    keep_new_lines : bool
        If True, the final text might contain new_line
        characters.

    Returns
    -------
    text : str
        All the text found in the input file.
    """
    with filepath.open("r") as f:
        text = f.read()

        if not keep_new_lines:
            text = text.replace('\n', '')

        return text


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
