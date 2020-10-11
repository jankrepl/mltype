"""Module implementing interaction logic."""
import bisect
import curses
import pathlib
import shutil

from mltype.base import TypedText
from mltype.base import STATUS_BACKSPACE, STATUS_CORRECT, STATUS_WRONG


class Cursor:
    """Utility class that can locate and modify the position of a cursor."""
    def __init__(self, stdscr):
        self.stdscr = stdscr

    @property
    def pos(self):
        """Current position (x, y) of the cursors."""
        return self.stdscr.getyx()

    def get_char(self):
        """Get the character the cursor is standing on."""
        y, x = self.pos
        return chr(self.stdscr.inch(y, x) & 0xFF)

    def move_abs(self, y, x):
        """Move absolutely to cooordinates.

        Note that if the column coordinate x is out of the
        screen then we automatically move to differnt row.

        Paramaters
        ----------
        y, x : int
            New coordinates where to move the cursor to.

        """
        max_y, max_x = self.stdscr.getmaxyx()
        delta_y, new_x = divmod(x, max_x)
        new_y = max(y + delta_y, 0)
        self.stdscr.move(new_y, new_x)

    def move_rel(self, delta_y, delta_x):
        """Move relative to the current cursor.

        Parameters
        ----------
        delta_y, delta_x : int
            Relative shifts in a respective direction
        """
        y, x = self.pos
        self.move_abs(y + delta_y, x + delta_x)

    def jump_line_start(self):
        """Jump to the start of the line."""
        y, _ = self.pos

        self.move_abs(y, 0)

    def jump_start(self):
        """Jump to the start of the window (upper left)."""
        self.move_abs(0, 0)

    def jump_end(self):
        """Jump the the end of the window (bottom right)."""
        height, width = self.stdscr.getmaxyx()
        self.move_abs(height - 1, width - 1)

    def new_line(self, n=1):
        """Shift by n rows."""
        self.move_rel(n, 0)

    def shift(self, n=1):
        """Shift by n columns."""
        self.move_rel(0, n)


class Pen:
    def __init__(self, font, background, i):
        self.font = font
        self.background = background
        self.i = i

        self._register()

    def addch(self, stdscr, y, x, text):
        stdscr.addch(y, x, text, curses.color_pair(self.i))

    def _register(self):
        curses.init_pair(self.i, self.font, self.background)


class TypedTextWriter:
    """Curses writer that uses the TypedText object.

    We make an assumption that the x and y position of the starting
    character stay the same.

    Parameters
    ----------
    tt : TypedText
        Text that the user is going to type.

    stdscr : curses.Window
        Main curses window.

    y_start, x_start : int
        Coordinates of the first character.

    replay_tt : TypedText or None
        If provided, it represents a previously typed text that
        we want to dynamically plot together with the current
        typing.

    Attributes
    ----------
    current_ix : int
        Represents the index of the character of `self.tt.text` that we
        are about to type. Note this is exactly the character on which
        the cursor will be lying.

    pens : dict
        The keys are integers representing different statuses. The values
        are `Pen` objects representing how to format a character with
        a given status. Note that if `replay_tt` is provided we add a new
        entry "replay" and it represents the style of replay character.

    replay_uactions : list
        The unrolled actions of the replay.

    replay_elapsed: list
        The same length as `replay_uactions`. It stores the
        elapsed times (since the start) of all the actions. Note that
        it is going to be sorted in an ascending order and we can do
        binary search on it.

    target_wpm : int or None
        If specified, we display the uniform run that leads to that speed.
    """

    def __init__(
        self, tt, stdscr, y_start=0, x_start=0, replay_tt=None, target_wpm=None
    ):
        self.tt = tt
        self.stdscr = stdscr
        self.y_start = y_start
        self.x_start = x_start
        self.replay_tt = replay_tt
        self.target_wpm = target_wpm

        self.current_ix = 0
        self.cursor = Cursor(stdscr)  # utility that will help us jump around

        self.pens = {
            STATUS_BACKSPACE: Pen(curses.COLOR_WHITE, curses.COLOR_BLACK, 1),
            STATUS_CORRECT: Pen(curses.COLOR_GREEN, curses.COLOR_BLACK, 2),
            STATUS_WRONG: Pen(curses.COLOR_WHITE, curses.COLOR_RED, 3),
        }

        if self.replay_tt is not None:
            self._validate_replay()

            self.replay_uactions = self.replay_tt.unroll_actions()
            self.replay_elapsed = [
                (x[1].ts - self.replay_tt.start_ts).total_seconds()
                for x in self.replay_uactions
            ]
            self.pens["replay"] = Pen(curses.COLOR_WHITE, curses.COLOR_BLUE, 4)
        if self.target_wpm is not None:
            self.pens["target"] = Pen(
                curses.COLOR_WHITE, curses.COLOR_MAGENTA, 5
            )

    def _validate_replay(self):
        """Check that the replay is compatible with the current text."""

        if self.replay_tt.text != self.tt.text:
            raise ValueError("The replay text and text do not agree.")

        if self.replay_tt.start_ts is None:
            raise ValueError("The replay was never started")

    def render(self):
        # self.stdscr.clear()

        i_start, _, width = self.screen_status

        if self.replay_tt is not None:
            aix_replay = bisect.bisect_left(
                self.replay_elapsed, self.tt.elapsed_seconds
            )
            aix_replay = min(self.replay_tt.n_actions - 1, aix_replay)
            i_replay = self.replay_uactions[aix_replay][0]

        if self.target_wpm is not None:
            i_target = self.tt.elapsed_seconds * 5 * self.target_wpm / 60
            i_target = min(self.tt.n_characters - 1, int(i_target))

        # rended text
        for i, (alist, ch) in enumerate(zip(self.tt.actions, self.tt.text)):
            y, x = divmod(i_start + i, width)

            if i == self.current_ix or not alist:
                # character that we stand on needs to have backspace styling
                status = STATUS_BACKSPACE  # same styling
            else:
                status = alist[-1].status

            if self.replay_tt is not None and i == i_replay != self.current_ix:
                if status in {STATUS_BACKSPACE, STATUS_CORRECT}:
                    # Make sure the normal cursor is visible
                    status = "replay"

            if self.target_wpm is not None and i == i_target != self.current_ix:
                if status in {STATUS_BACKSPACE, STATUS_CORRECT}:
                    # Make sure the normal cursor is visible
                    status = "target"

            self.pens[status].addch(self.stdscr, y, x, ch)

        # render cursor
        self.cursor.move_abs(self.y_start, self.x_start + self.current_ix)

        self.stdscr.refresh()

    def process_character(self):
        """The integer."""
        try:
            char_typed_ = self.stdscr.getch()
        except curses.error:
            return

        # Action characters handeling
        if char_typed_ in {127, curses.KEY_BACKSPACE}:
            try:
                self.tt.type_character(self.current_ix)
            except IndexError:
                return

            self.current_ix = max(0, self.current_ix - 1)
            return

        elif char_typed_ in {curses.KEY_RESIZE}:
            self.stdscr.clear()
            return

        # See it as a non-system character
        try:
            char_typed = chr(char_typed_)
        except ValueError:
            return

        try:
            self.tt.type_character(self.current_ix, char_typed)
        except IndexError:
            return

        self.current_ix = min(self.tt.n_characters - 1, self.current_ix + 1)

    @property
    def screen_status(self):
        """The starting position of our text."""
        height, width = self.stdscr.getmaxyx()
        i_start = self.y_start * width + self.x_start

        return i_start, height, width


def run_loop(
    stdscr,
    text,
    force_perfect=True,
    replay_tt=None,
    instant_death=False,
    target_wpm=None,
):
    """Run curses loop - actual implementation."""

    tt = TypedText(text)
    writer = TypedTextWriter(
        tt, stdscr, replay_tt=replay_tt, target_wpm=target_wpm
    )

    while not tt.check_finished(force_perfect=force_perfect):
        writer.render()

        writer.process_character()

        if instant_death and tt.n_wrong_characters > 0:
            tt.end_ts = tt.start_ts
            break

    return tt


def main_basic(text, force_perfect, output_file, instant_death, target_wpm):
    """Run main curses loop with no previous replay

    Parameters
    ----------
    force_perfect : bool
        If True, then one cannot finish typing before all characters
        are typed without any mistakes.

    output_file : str or pathlib.Path or None
        If ``pathlib.Path`` then we store the typed text in this file.
        If None, no saving is taking place.

    instant_death : bool
        If active, the first mistake will end the game.

    target_wpm : int or None
        The desired speed to be displayed as a guide.
    """
    text_stripped = text.strip()

    tt = curses.wrapper(
        run_loop,
        text_stripped,
        force_perfect=force_perfect,
        replay_tt=None,
        instant_death=instant_death,
        target_wpm=target_wpm,
    )

    if output_file is not None:
        tt.save(pathlib.Path(output_file))

    width, _ = shutil.get_terminal_size()
    print(" Statistics ".center(width, "="))
    print(f"Accuracy: {tt.compute_accuracy():.1f}\nWPM: {tt.compute_wpm():.1f}")
    print(width * "=")


def main_replay(
    replay_file, force_perfect, overwrite, instant_death, target_wpm
):
    """Run main curses loop with a replay.

    Parameters
    ----------
    force_perfect : bool
        If True, then one cannot finish typing before all characters
        are typed without any mistakes.


    overwrite : bool
        If True, the replay file will be overwritten in case
        we are faster than it.

    replay_file : str or pathlib.Path
        Typed text in this file from some previous game.

    instant_death : bool
        If active, the first mistake will end the game.

    target_wpm : None or int
        The desired speed to be shown as guide.
    """
    replay_file = pathlib.Path(replay_file)
    replay_tt = TypedText.load(replay_file)

    if not replay_tt.check_finished():
        raise ValueError("The checkpoint file contains unfinished text")

    tt = curses.wrapper(
        run_loop,
        replay_tt.text,
        force_perfect=force_perfect,
        replay_tt=replay_tt,
        instant_death=instant_death,
        target_wpm=target_wpm,
    )

    wpm_replay = replay_tt.compute_wpm()
    wpm_new = tt.compute_wpm()

    width, _ = shutil.get_terminal_size()
    print(" Statistics ".center(width, "="))
    print(f"Old WPM: {wpm_replay:.1f}\nNew WPM: {wpm_new:.1f}")

    if wpm_new > wpm_replay:
        print("Congratulations!")
        if overwrite:
            print("Updating the checkpoint file")
            tt.save(replay_file)
    else:
        print("You lost!")
    print(width * "=")
