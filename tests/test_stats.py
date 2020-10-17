"""Collection of tests covering stats.py"""

from mltype.base import TypedText
from mltype.stats import times_per_character


def test_time_per_character():
    text = "hello"
    tt = TypedText(text)

    tt.type_character(0, "h")  # first caracter does not count
    tt.type_character(1, "e")
    tt.type_character(2, "l")
    tt.type_character(3, "l")
    tt.type_character(4, "m")  # mistake
    tt.type_character(4)  # backspace

    stats = times_per_character(tt)

    assert len(stats) == 2
    assert {"e", "l"} ==  set(stats.keys())
    assert len(stats["e"]) == 1
    assert len(stats["l"]) == 2
