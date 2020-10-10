"""Building blocks."""

from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
import pickle

import numpy as np
from tqdm import tqdm


class CharacterSampler(ABC):
    """Generates a character given previous characters."""

    @abstractmethod
    def sample(self, prev_s=None, random_state=None):
        """Generate a character given previous string.

        Parameters
        ----------
        prev_s : str or None
            If speficied, then an initial condition to be used to
            sample new character.

        random_state : str or None
            If specified, one can get reproducible results.

        Returns
        -------
        new_ch : str
            A sampled character (=one element string).
        """

    def sample_many(self, n_chars, prev_s=None, random_state=None):
        """Generate a sequence of characters given previous string.

        Parameters
        ----------
        n_chars : int
            Number of characters to generate.

        prev_s : str or None
            If speficied, then an initial condition to be used to
            sample new character.

        random_state : str or None
            If specified, one can get reproducible results.
        """
        prev_s = prev_s or ""
        new_s = ""

        if random_state is not None:
            np.random.seed(random_state)

        while len(new_s) < n_chars:
            new_s += self.sample(prev_s=prev_s + new_s, random_state=None)

        return new_s


class Language(CharacterSampler):
    """Represents a specific language with conditional probabilities.

    Parameters
    ----------
    vocab : str
        A collection of characters that are contained in the language.
        Note that it is ordered based on the `ord`.

    probs : np.ndarray
        Numpy array of dimension `N + 1` where `N` is the number
        of previous characters we condition on. Note that it holds
        conditional distributions.
    """

    @classmethod
    def load(cls, path):
        """Load a language from a serialized file."""

    @classmethod
    def generate(
        cls,
        texts,
        window_size=1,
        min_char_count=3,
        forbidden_chars=None,
        verbose=False,
    ):
        """Generate a language from a list of texts."""
        counter = defaultdict(int)
        for text in texts:
            for ch in text:
                counter[ch] += 1

        vocab = "".join(
            sorted(
                [
                    ch
                    for ch, count in counter.items()
                    if count >= min_char_count
                ]
            )
        )

        n_chars = len(vocab)
        ch2ix = {ch: ix for ix, ch in enumerate(vocab)}

        probs = np.ones([n_chars] * (window_size + 1)) * np.finfo(float).eps

        if verbose:
            iterable = tqdm(texts)
        else:
            iterable = texts

        for text in iterable:
            for i in range(window_size, len(text)):
                try:
                    coords = tuple(
                        ch2ix[ch] for ch in text[i - window_size: i + 1]
                    )
                except KeyError:
                    # contains a character that did not make it to the vocab
                    continue

                probs[coords] += 1

        probs /= probs.sum(axis=-1, keepdims=True)
        probs[np.isnan(probs)] = 0  # replace

        return cls(vocab, probs=probs)

    def __init__(self, vocab, probs):
        self.vocab = vocab
        self.probs = probs

        self._ch2ix = {ch: ix for ix, ch in enumerate(self.vocab)}
        self._vocab_l = list(vocab)

    @property
    def window_size(self):
        """Get the size of the window."""
        return self.probs.ndim - 1

    @property
    def n_chars(self):
        """Get number of available characters."""
        return len(self.vocab)

    def get_char_position(self, ch):
        return self._ch2ix[ch]

    def sample(self, prev_s=None, random_state=None):
        """Sample one character."""
        prev_s = prev_s or ""

        if random_state is not None:
            np.random.seed(random_state)

        if len(prev_s) < self.window_size:
            raise NotImplementedError()

        if self.window_size:
            coords = tuple(
                self._ch2ix[ch] for ch in prev_s[-self.window_size:]
            )
        else:
            coords = tuple()

        return np.random.choice(self._vocab_l, p=self.probs[coords])


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
        What was the status AFTER pusing the key. It should be one
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
        """The number of seconds elapsed from the first action."""
        if self.start_ts is None:
            return 0

        end_ts = self.end_ts or datetime.now()

        return (end_ts - self.start_ts).total_seconds()

    @property
    def n_actions(self):
        """The number of actions that have been taken."""
        return sum(len(x) for x in self.actions)

    @property
    def n_characters(self):
        """The number of characters in the text."""
        return len(self.text)

    @property
    def n_backspace_actions(self):
        """The number of backspace actions."""
        return sum(
            sum(1 for a in x if a.status == STATUS_BACKSPACE)
            for x in self.actions
        )

    @property
    def n_backspace_characters(self):
        """The number of characters that have been backspaced."""
        return self._n_characters_with_status(STATUS_BACKSPACE)

    @property
    def n_correct_characters(self):
        """The number of characters that have been typed correctly."""
        return self._n_characters_with_status(STATUS_CORRECT)

    @property
    def n_untouched_characters(self):
        """The number of characters that have not been touched yet."""
        return len([x for x in self.actions if not x])

    @property
    def n_wrong_characters(self):
        """The number of characters that have been typed wrongly."""
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
        """Determined whether the typing has been finished successfully.

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
            The characgter that was typed. Note that if None then we assume
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
        if self.check_finished():
            self.end_ts = ts

    def unroll_actions(self):
        """Export actions in an order they appeared

        Returns
        -------
        res : list
            List of tuples of `(ix_char, Action(..))`
        """
        return sorted(
            [(i, a) for i, x in enumerate(self.actions) for a in x],
            key=lambda x: x[1].ts,
        )


class TypedText_:
    """Abstraction that represents the text that needs to be typed.

    Parameters
    ----------
    text : str
        Some text that needs to be typed.

    blocking_mode : bool
        If activated, one cannot write correct characters
        if there is a mistake before.

    Attributes
    ----------
    status : list
        List of the same length as the `text` string. It stores
        per character what the status is.

    times : list
        List of the same length as the `text` string. It storesjjjjjj
    """

    def __init__(self, text, blocking_mode=True):
        self.text = text
        self.blocking_mode = blocking_mode

        self.status = len(text) * [STATUS_BACKSPACE]

        self.times = len(text) * [None]
        self.start_ts = None  # used to compute CPM/WPM
        self.end_ts = None

    def __len__(self):
        return len(self.text)

    def __setitem__(self, i, status):
        # check if it is the very first character
        ts = datetime.now()

        if self.start_ts is None:
            self.start_ts = ts

        self.times[i] = status
        self.status[i] = status

        if self.is_done():
            self.end_ts = ts

    @property
    def n_correct(self):
        """Count the number of correct entries."""
        return len([1 for x in self.status if x == STATUS_CORRECT])

    def compute_cpm(self):
        """Compute current characters per minute."""
        end_ts = self.end_ts or datetime.now()
        start_ts = self.start_ts or end_ts
        n_seconds = (end_ts - start_ts).total_seconds()

        try:
            cpm = 60 * (self.n_correct / n_seconds)
        except ZeroDivisionError:
            cpm = 0

        return cpm

    def compute_wpm(self, word_size=5):
        """Compute current words per minute."""
        return self.compute_cpm() / word_size

    def is_done(self):
        return all(x == STATUS_CORRECT for x in self.status)

    def is_untouched(self):
        return all(x == STATUS_BACKSPACE for x in self.status)

    def set_correct(self, i):
        self[i] = STATUS_CORRECT

    def set_normal(self, i):
        self[i] = STATUS_BACKSPACE

    def set_wrong(self, i):
        self[i] = STATUS_WRONG
