from datetime import datetime
import pathlib

import pytest

from mltype.base import (
    Action,
    STATUS_CORRECT,
    STATUS_WRONG,
    TypedText,
)


class TestAction:
    def test_init(self):
        with pytest.raises(ValueError):
            Action("wa", STATUS_CORRECT, datetime.now())

    def test_equality(self):
        ts_1 = datetime.now()
        ts_2 = datetime.now()

        assert Action("b", STATUS_CORRECT, ts_1) != "wrong_type"

        assert Action("a", STATUS_CORRECT, ts_1) == Action(
            "a", STATUS_CORRECT, ts_1
        )
        assert Action("a", STATUS_WRONG, ts_1) != Action(
            "a", STATUS_CORRECT, ts_1
        )
        assert Action("a", STATUS_CORRECT, ts_2) != Action(
            "a", STATUS_CORRECT, ts_1
        )
        assert Action("b", STATUS_CORRECT, ts_1) != Action(
            "a", STATUS_CORRECT, ts_1
        )


class TestTypedText:
    def test_basic(self):
        tt = TypedText("hello")

        assert tt == tt
        assert tt != "wrong type"

        # nothing typed
        assert tt.elapsed_seconds == 0

        assert tt.n_characters == 5
        assert tt.n_correct_characters == 0
        assert tt.n_untouched_characters == 5
        assert tt.n_backspace_characters == 0
        assert tt.n_wrong_characters == 0

        assert tt.n_actions == 0
        assert tt.n_backspace_actions == 0

        assert tt.compute_accuracy() == 0

        assert not tt.check_finished()
        assert not tt.unroll_actions()

        # type start
        tt.type_character(0, "h")
        tt.type_character(1, "w")
        tt.type_character(1, None)
        tt.type_character(1, "e")

        assert tt.elapsed_seconds > 0

        assert tt.n_characters == 5
        assert tt.n_correct_characters == 2
        assert tt.n_untouched_characters == 3
        assert tt.n_backspace_characters == 0
        assert tt.n_wrong_characters == 0

        assert tt.n_actions == 4
        assert tt.n_backspace_actions == 1

        assert tt.compute_accuracy() == 2 / 3

        assert not tt.check_finished()
        assert len(tt.unroll_actions()) == 4

        # finish typing
        tt.type_character(2, "l")
        tt.type_character(3, "l")
        tt.type_character(4, "a")
        tt.type_character(4)
        tt.type_character(4, "o")

        assert tt.elapsed_seconds == tt.elapsed_seconds > 0

        assert tt.n_characters == 5
        assert tt.n_correct_characters == 5
        assert tt.n_untouched_characters == 0
        assert tt.n_backspace_characters == 0
        assert tt.n_wrong_characters == 0

        assert tt.n_actions == 4 + 5
        assert tt.n_backspace_actions == 2

        assert tt.compute_accuracy() == 5 / 7

        assert tt.check_finished()
        assert tt.check_finished(force_perfect=False)

        assert tt == tt

        ua = tt.unroll_actions()
        assert len(ua) == 9
        assert ua[0][0] == 0
        assert ua[-1][0] == 4

        with pytest.raises(IndexError):
            tt.type_character(5)

    def test_save_and_load(self, tmpdir):
        path_dir = pathlib.Path(str(tmpdir))
        path_file = path_dir / "cool.rlt"

        tt = TypedText("Hello")

        # finish typing
        tt.type_character(0, "H")
        tt.type_character(1, "w")
        tt.type_character(1)
        tt.type_character(1, "e")
        tt.type_character(2, "l")
        tt.type_character(3, "l")

        tt.save(path_file)
        assert tt == TypedText.load(path_file)
