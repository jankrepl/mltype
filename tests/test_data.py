"""Collection of tests covering data.py."""
import pathlib

from mltype.data import file2text, folder2text


def test_file2text(tmpdir):
    path = pathlib.Path(str(tmpdir)) / "file.txt"
    text = "    1st line\n2nd line    \n    3rd line\n\n"

    path.write_text(text)

    res = file2text(path, verbose=True)

    assert res == "1st line 2nd line 3rd line"


def test_folder2text(tmpdir):
    path_root = pathlib.Path(str(tmpdir))

    # - a.py
    # - b
    # --- c.py
    # --- d.bin
    # --- e
    # ------ f.py
    # ------ g.txt
    # - h.js

    # CREATION
    paths = {
        "a": path_root / "a.py",
        "c": path_root / "b" / "c.py",
        "d": path_root / "b" / "d.bin",
        "f": path_root / "b" / "e" / "f.py",
        "g": path_root / "b" / "e" / "g.txt",
        "h": path_root / "h.js",
    }

    for text, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix != ".bin":
            path.write_text(text)

    paths["d"].write_bytes(bytearray([222, 223, 224]))

    assert set(folder2text(path_root)) == {"a", "c", "f", "g", "h"}
    assert set(folder2text(path_root, ".py")) == {"a", "c", "f"}
    assert set(folder2text(path_root, ".txt")) == {"g"}
    assert set(folder2text(path_root, ".js")) == {"h"}
    assert set(folder2text(path_root, ".bin")) == set()
