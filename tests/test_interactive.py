"""Collection of tests covering the `interactive.py` module."""
from unittest.mock import Mock, MagicMock

from mltype.base import TypedText
from mltype.interactive import main_basic


def test_strip(monkeypatch):
    fake_curses = Mock()
    tt = TypedText("Hello")
    fake_curses.wrapper.return_value = tt
    monkeypatch.setattr("mltype.interactive.curses", fake_curses)

    fake_text = MagicMock(spec=str)

    main_basic(fake_text, None, None, None, None)
    fake_text.strip.assert_called_once()
