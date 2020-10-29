"""Building blocks."""

from datetime import datetime
import pickle


STATUS_BACKSPACE = 1
STATUS_CORRECT = 2
STATUS_WRONG = 3


class Action:
    """Representation of one keypress.

    Parameters
    ----------
    pressed_key : str
        What key was pressed. We define a convention that pressing
        a backspace will be represented as `pressed_key=None`.

    status : int
        What was the status AFTER pushing the key. It should be one
        of the following integers:
            * STATUS_BACKSPACE
            * STATUS_CORRECT
            * STATUS_WRONG

    ts : datetime
        The timestamp corresponding to this action.
    """

    def __init__(self, pressed_key, status, ts):
        if pressed_key is not None and len(pressed_key) != 1:
            raise ValueError("The pressed key needs to be a single character")
        self.pressed_key = pressed_key
        self.status = status
        self.ts = ts

    def __eq__(self, other):
        """Check whether equal."""
        if not isinstance(other, self.__class__):
            return False

        return (
            self.pressed_key == other.pressed_key
            and self.status == other.status
            and self.ts == other.ts
        )


class TypedText:
    """Abstraction that represenets the text that needs to be typed.

    Parameters
    ----------
    text : str
        Text that needs to be typed.

    Attributes
    ----------
    actions : list
        List of lists of Action instances of length equal to `len(text)`.
        It logs per character all actions that have been taken on it.

    start_ts : datetime or None
        Timestamp of when the first action was performed (not the
        time of initialization).

    end_ts : datetime or None
        Timestamp of when the last action was taken. Note that
        it is the action that lead to the text being correctly typed
        in it's entirity.
    """

    def __init__(self, text):
        self.text = text

        self.actions = [[] for _ in range(len(text))]

        self.start_ts = None
        self.end_ts = None

    @classmethod
    def load(cls, path):
        """Load a pickled file.

        Parameters
        ----------
        path : pathlib.Path
            Path to the pickle file.

        Returns
        -------
        typed_text : TypedText
            Instance of the ``TypedText``
        """
        with path.open("rb") as f:
            text, actions, start_ts, end_ts = pickle.load(f)

        typed_text = cls(text)
        typed_text.actions = actions
        typed_text.start_ts = start_ts
        typed_text.end_ts = end_ts

        return typed_text

    def __eq__(self, other):
        """Check if equal.

        Not considering start and end timestamps.
        """
        if not isinstance(other, self.__class__):
            return False

        return self.text == other.text and self.actions == other.actions

    def _n_characters_with_status(self, status):
        """Count the number of characters with a given status.

        Parameters
        ----------
        status : str
            The status we look for in the character.

        Returns
        -------
        The number of characters with status `status`.
        """
        return len([x for x in self.actions if x and x[-1].status == status])

    @property
    def elapsed_seconds(self):
        """Get the number of seconds elapsed from the first action."""
        if self.start_ts is None:
            return 0

        end_ts = self.end_ts or datetime.now()

        return (end_ts - self.start_ts).total_seconds()

    @property
    def n_actions(self):
        """Get the number of actions that have been taken."""
        return sum(len(x) for x in self.actions)

    @property
    def n_characters(self):
        """Get the number of characters in the text."""
        return len(self.text)

    @property
    def n_backspace_actions(self):
        """Get the number of backspace actions."""
        return sum(
            sum(1 for a in x if a.status == STATUS_BACKSPACE)
            for x in self.actions
        )

    @property
    def n_backspace_characters(self):
        """Get the number of characters that have been backspaced."""
        return self._n_characters_with_status(STATUS_BACKSPACE)

    @property
    def n_correct_characters(self):
        """Get the number of characters that have been typed correctly."""
        return self._n_characters_with_status(STATUS_CORRECT)

    @property
    def n_untouched_characters(self):
        """Get the number of characters that have not been touched yet."""
        return len([x for x in self.actions if not x])

    @property
    def n_wrong_characters(self):
        """Get the number of characters that have been typed wrongly."""
        return self._n_characters_with_status(STATUS_WRONG)

    def compute_accuracy(self):
        """Compute the accuracy of the typing."""
        try:
            acc = self.n_correct_characters / (
                self.n_actions - self.n_backspace_actions
            )
        except ZeroDivisionError:
            acc = 0

        return acc

    def compute_cpm(self):
        """Compute characters per minute."""
        try:
            cpm = 60 * self.n_correct_characters / self.elapsed_seconds

        except ZeroDivisionError:
            # We actually set self.end_ts = self.start_ts in instant death
            cpm = 0

        return cpm

    def compute_wpm(self, word_size=5):
        """Compute words per minute."""
        return self.compute_cpm() / word_size

    def check_finished(self, force_perfect=True):
        """Determine whether the typing has been finished successfully.

        Parameters
        ----------
        force_perfect : bool
            If True, one can only finished if all the characters were typed
            correctly. Otherwise, all characters need to be either correct
            or wrong.

        """
        if force_perfect:
            return self.n_correct_characters == self.n_characters
        else:
            return (
                self.n_correct_characters + self.n_wrong_characters
                == self.n_characters
            )

    def save(self, path):
        """Save internal state of this TypedText.

        Can be loaded via the class method ``load``.

        Parameters
        ----------
        path : pathlib.Path
            Where the .rlt file will be store.
        """
        with path.open("wb") as f:
            all_obj = (self.text, self.actions, self.start_ts, self.end_ts)
            pickle.dump(all_obj, f)

    def type_character(self, i, ch=None):
        """Type one single character.

        Parameters
        ----------
        i : int
            Index of the character in the text.

        ch : str or None
            The character that was typed. Note that if None then we assume
            that the user used backspace.
        """
        if not (0 <= i < self.n_characters):
            raise IndexError(f"The index {i} is outside of the text.")

        ts = datetime.now()

        # check if it is the first action
        if self.start_ts is None:
            self.start_ts = ts

        # check if it is a backspace
        if ch is None:
            self.actions[i].append(Action(ch, STATUS_BACKSPACE, ts))
            return

        # check if the characters agree
        if ch == self.text[i]:
            self.actions[i].append(Action(ch, STATUS_CORRECT, ts))
        else:
            self.actions[i].append(Action(ch, STATUS_WRONG, ts))

        # check whether finished
        if self.check_finished(force_perfect=False):
            self.end_ts = ts

    def unroll_actions(self):
        """Export actions in an order they appeared.

        Returns
        -------
        res : list
            List of tuples of `(ix_char, Action(..))`
        """
        return sorted(
            [(i, a) for i, x in enumerate(self.actions) for a in x],
            key=lambda x: x[1].ts,
        )
