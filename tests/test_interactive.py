"""Collection of tests covering the `interactive.py` module."""
from unittest.mock import Mock, MagicMock

import hecate.hecate
import pytest

from mltype.base import TypedText
from mltype.interactive import main_basic

font_colors = {
    "black": 30,
    "red": 31,
    "green": 32,
    "yellow": 33,
    "blue": 34,
    "magenta": 35,
    "cyan": 36,
    "white": 37,
    "reset": 0,
}


background_colors = {
    "black": 40,
    "red": 41,
    "green": 42,
    "yellow": 43,
    "blue": 44,
    "magenta": 45,
    "cyan": 46,
    "white": 47,
}

DEFAULT_BACKGROUND = "black"


def esc(x):
    return f"\033[{x}m"

class CustomRunner(hecate.hecate.Runner):
    """Add convenience methods and color screenshotting."""
    def await_ctext(self, text, timeout=None):
        for _ in self.poll_until_timeout(timeout):
            screen = self.cscreenshot()
            if text in screen:
                return
        raise Timeout("Timeout while waiting for text %r to appear" % (text,))

    def cscreenshot(self, pane=0):
        """Screenshot with color."""
        buf = self.tmux.a_buffer()
        self.tmux.execute_command("capture-pane", "-e", "-b", buf, "-t", pane)
        return self.tmux.get_buffer(buf)

    def get_cursor_position(self):
        """Get current position of the cursor."""

    def get_line(self, i):
        """Get all the text on a given line."""

    def resize(self, height, width):
        """Resize the pane"""

def capture_pane_color(self, pane):
    """Screenshot the entire tmux pane with colors.

    We patch the original `hackate` method with this one
    in order to be capture colors too.

    Parameters
    ----------
    pane : int
        Number of the pane.

    Return
    ------
    str
        The screnshot with colors.
    """
    buf = self.a_buffer()
    self.execute_command("capture-pane", "-e", "-b", buf, "-t", pane)
    return self.get_buffer(buf)


def test_strip(monkeypatch):
    fake_curses = Mock()
    tt = TypedText("Hello")
    fake_curses.wrapper.return_value = tt
    monkeypatch.setattr("mltype.interactive.curses", fake_curses)

    fake_text = MagicMock(spec=str)

    main_basic(fake_text, None, None, None, None)
    fake_text.rstrip.assert_called_once()


class TestHecate:
    @pytest.mark.skipif(not RUN_HECATE, reason="Hecate is not installed")
    def test_basic(self, monkeypatch):

        with CustomRunner("mlt", "raw", "inside") as r:
            # initial check
            s_initial = esc(font_colors["white"])
            s_initial += esc(background_colors[DEFAULT_BACKGROUND])
            s_initial += "inside"
            r.await_ctext(s_initial, 1)

            # type i
            r.write("i")
            s = esc(font_colors["green"])
            s += esc(background_colors[DEFAULT_BACKGROUND])
            s += "i"
            s += esc(font_colors["white"])
            s = "nside"
            r.await_ctext(s, 1)

            # backspace
            r.press("BSpace")
            r.await_ctext(s_initial, 1)

            # type insa
            r.write("insa")
            s = esc(font_colors["green"])
            s += esc(background_colors[DEFAULT_BACKGROUND])
            s += "ins"
            s = esc(font_colors["white"])
            s += esc(background_colors["red"])
            s += "i"
            s += esc(background_colors[DEFAULT_BACKGROUND])
            s += "de"
            r.await_ctext(s, 1)

            # Finalize
            r.press("BSpace")
            r.write("ide")

            r.await_ctext("== Statistics ==")
