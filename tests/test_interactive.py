"""Collection of tests covering the `interactive.py` module."""
from datetime import datetime
from unittest.mock import Mock, MagicMock

try:
    import hecate.hecate
except ImportError:
    hecate = Mock()

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


def gen_fb(font=None, background=None):
    """Generate font color and background color escape sequence."""

    s = ""
    if font is not None:
        s += esc(font_colors[font])

    if background is not None:
        s += esc(background_colors[background])

    return s


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
        x_s, y_s = self.tmux.execute_command(
            "display", "-t0", "-p", "#{cursor_x}\t#{cursor_y}"
        ).split()
        return int(x_s), int(y_s)

    def record_cscreenshots(self, n_seconds=1, pane=0):
        all_ss = set()
        start = datetime.now()

        while (datetime.now() - start).total_seconds() < n_seconds:
            all_ss.add(self.cscreenshot())

        return all_ss


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


@pytest.mark.skipif(isinstance(hecate, Mock), reason="Hecate is not installed")
class TestHecate:
    def test_basic(self):
        with CustomRunner("mlt", "raw", "inside") as r:
            assert r.get_cursor_position() == (0, 0)
            # initial check
            s_initial = esc(font_colors["white"])
            s_initial += esc(background_colors[DEFAULT_BACKGROUND])
            s_initial += "inside"
            r.await_ctext(s_initial, 1)

            # type i
            r.write("i")
            s = gen_fb("green", DEFAULT_BACKGROUND)
            s += "i"
            s += gen_fb("white")
            s = "nside"
            r.await_ctext(s, 1)
            assert r.get_cursor_position() == (1, 0)

            # backspace
            r.press("BSpace")
            r.await_ctext(s_initial, 1)
            assert r.get_cursor_position() == (0, 0)

            # type insa
            r.write("insa")
            s = gen_fb("green", DEFAULT_BACKGROUND)
            s += "ins"
            s += gen_fb("white", "red")
            s += "i"
            s += gen_fb(None, DEFAULT_BACKGROUND)
            s += "de"
            r.await_ctext(s, 1)
            assert r.get_cursor_position() == (4, 0)

            # Finalize
            r.press("BSpace")
            r.write("ide")

            r.await_ctext("== Statistics ==")

    def test_target_speed(self):
        with CustomRunner("mlt", "raw", "hello", "-t", "120") as r:
            r.await_ctext(gen_fb("white", DEFAULT_BACKGROUND) + "hello")
            assert r.get_cursor_position() == (0, 0)

            r.press("BSpace")  # this triggers the marker
            r.await_ctext(esc(background_colors["magenta"]))

            all_ss = r.record_cscreenshots(0.5)

            # the first one is excluded because the cursor is there
            assert len(all_ss) == 4
            l = [esc(background_colors["magenta"]) in x for x in all_ss]
            assert all(l)
