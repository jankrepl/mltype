"""Module implementing interaction logic."""
import bisect
import curses
import pathlib

from mltype.base import TypedText
from mltype.base import STATUS_BACKSPACE, STATUS_CORRECT, STATUS_WRONG
from mltype.utils import get_config_file, print_section


class Cursor:
    """Utility class that can locate and modify the position of a cursor."""

    def __init__(self, stdscr):
        self.stdscr = stdscr

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


class Pen:
    """Represents background and font color."""

    def __init__(self, font, background, i):
        self.font = font
        self.background = background
        self.i = i

        self._register()

    def addch(self, stdscr, y, x, text):
        """Add a single character.

        Parameters
        ----------
        stdscr : curses.Window
            Window in which we add the character.

        y, x : int
            Position of the character.

        text : str
            Single element string representing the character.
        """
        stdscr.addch(y, x, text, curses.color_pair(self.i))

    def addstr(self, stdscr, y, x, text):
        """Add a string.

        Parameters
        ----------
        stdscr : curses.Window
            Window in which we add the character.

        y, x : int
            Position of the string.

        text : str
            String to put to the screen.
        """
        stdscr.addstr(y, x, text, curses.color_pair(self.i))

    def _register(self):
        """Register colors with curses."""
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

        curses.use_default_colors()  # allow using terminal colors
        colors = self._get_colors()

        self.pens = {
            STATUS_BACKSPACE: Pen(
                colors["color_default_foreground"],
                colors["color_default_background"],
                1,
            ),
            STATUS_CORRECT: Pen(
                colors["color_correct_foreground"],
                colors["color_correct_background"],
                2,
            ),
            STATUS_WRONG: Pen(
                colors["color_wrong_foreground"],
                colors["color_wrong_background"],
                3,
            ),
            "replay": Pen(
                colors["color_replay_foreground"],
                colors["color_replay_background"],
                4,
            ),
            "target": Pen(
                colors["color_target_foreground"],
                colors["color_target_background"],
                5,
            ),
        }

        stdscr.bkgd(" ", curses.color_pair(1))

        if self.replay_tt is not None:
            self._validate_replay()

            self.replay_uactions = self.replay_tt.unroll_actions()
            self.replay_elapsed = [
                (x[1].ts - self.replay_tt.start_ts).total_seconds()
                for x in self.replay_uactions
            ]

    def _validate_replay(self):
        """Check that the replay is compatible with the current text."""
        if self.replay_tt.text != self.tt.text:
            raise ValueError("The replay text and text do not agree.")

        if self.replay_tt.start_ts is None:
            raise ValueError("The replay was never started")

    def render(self):
        """Render the entire screen."""
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
        i_print = i_start
        current_ix_print = i_start
        for i, (alist, ch) in enumerate(zip(self.tt.actions, self.tt.text)):
            y, x = divmod(i_print, width)

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

            if i == self.current_ix:
                current_ix_print = i_print

            if ch == "\n":
                i_print += width - (i_print % width)
                self.pens[status].addch(self.stdscr, y, x, " ")
            elif ch == "\t":
                i_print += 4
                self.pens[status].addstr(self.stdscr, y, x, 4 * " ")
            else:
                i_print += 1
                self.pens[status].addch(self.stdscr, y, x, ch)

        # render cursor
        self.cursor.move_abs(self.y_start, self.x_start + current_ix_print)

        self.stdscr.refresh()

    def process_character(self):
        """Process an entered character."""
        try:
            char_typed_ = self.stdscr.getch()
        except curses.error:
            return

        # Action characters handeling
        if char_typed_ == -1:
            # no key typed:
            return

        elif char_typed_ in {8, 127, curses.KEY_BACKSPACE}:
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
        """Get screen information.

        Returns
        -------
        i_start : int
            Integer representing the number of cells away from the start
            we are.

        height, width : int
            Height, width of the screen. Note that user my resize during
            a session.
        """
        height, width = self.stdscr.getmaxyx()
        i_start = self.y_start * width + self.x_start

        return i_start, height, width

    @staticmethod
    def _get_colors():
        """Get colors from the config file.

        If not present we define some reasonable defaults.

        Returns
        -------
        dict
            The keys are the names of the colors (in plain English).
            The values are curses tokens for those colors.
        """
        mapping = {
            "terminal": -1,
            "black": curses.COLOR_BLACK,
            "red": curses.COLOR_RED,
            "green": curses.COLOR_GREEN,
            "yellow": curses.COLOR_YELLOW,
            "blue": curses.COLOR_BLUE,
            "magenta": curses.COLOR_MAGENTA,
            "cyan": curses.COLOR_CYAN,
            "white": curses.COLOR_WHITE,
        }

        cp = get_config_file()

        if "general" in cp.sections():
            general = cp["general"]
        else:
            general = {}

        defaults = {
            "color_default_background": "black",
            "color_default_foreground": "white",
            "color_correct_background": "black",
            "color_correct_foreground": "green",
            "color_wrong_background": "red",
            "color_wrong_foreground": "white",
            "color_target_background": "magenta",
            "color_target_foreground": "white",
            "color_replay_background": "blue",
            "color_replay_foreground": "white",
        }

        colors = {}

        for name, default in defaults.items():
            color = general.get(name, default)

            if color not in mapping:
                raise KeyError(f"Unsupported color {color}")

            colors[name] = mapping[color]

        return colors


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

    # Curses settings
    stdscr.nodelay(1)  # makes getch non-blocking

    while not tt.check_finished(force_perfect=force_perfect):
        writer.render()

        writer.process_character()

        if instant_death and tt.n_wrong_characters > 0:
            tt.end_ts = tt.start_ts
            break

    return tt


def main_basic(text, force_perfect, output_file, instant_death, target_wpm):
    """Run main curses loop with no previous replay.

    Parameters
    ----------
    force_perfect : bool
        If True, then one cannot finish typing before all characters are
        typed without any mistakes.

    output_file : str or pathlib.Path or None
        If ``pathlib.Path`` then we store the typed text in this file.
        If None, no saving is taking place.

    instant_death : bool
        If active, the first mistake will end the game.

    target_wpm : int or None
        The desired speed to be displayed as a guide.
    """
    text_stripped = text.rstrip()

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

    with print_section(" Statistics ", fill_char="=", add_ts=False):
        print(
            f"Accuracy: {tt.compute_accuracy():.1%}\n"
            f"WPM: {tt.compute_wpm():.1f}"
        )


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

    with print_section(" Statistics ", fill_char="=", add_ts=False):
        print(f"Old WPM: {wpm_replay:.1f}\nNew WPM: {wpm_new:.1f}")

        if wpm_new > wpm_replay:
            print("Congratulations!")
            if overwrite:
                print("Updating the checkpoint file")
                tt.save(replay_file)
        else:
            print("You lost!")
