import pathlib
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from mltype.ml import (
    SingleCharacterLSTM,
    create_data_language,
    load_model,
    save_model,
)


class TestCreateDataLanguage:
    def test_exceptions(self):
        with pytest.raises(ValueError):
            create_data_language("great", [])

        with pytest.raises(ValueError):
            create_data_language("great", ["g", "r", "g"])

        with pytest.raises(ValueError):
            create_data_language("great", [chr(i) for i in range(255)])

    def test_zeros(self, monkeypatch):
        fake_tqdm = Mock()
        fake_tqdm.tqdm.side_effect = lambda x: x
        monkeypatch.setattr("mltype.ml.tqdm", fake_tqdm)

        text = "world"
        vocabulary = ["d", "l", "o", "w"]  # r is missing
        vocab_size = len(vocabulary)

        X, y, indices = create_data_language(
            text, vocabulary, window_size=2, fill_strategy="zeros", verbose=True
        )

        X_true = np.array(
            [
                [vocab_size, vocab_size],
                [vocab_size, 3],
                [3, 2],
                [2, vocab_size],
                [vocab_size, 1],
            ],
            dtype=np.int8,
        )

        y_true = np.array(
            [
                3,
                2,
                vocab_size,
                1,
                0,
            ],
            dtype=np.int8,
        )

        indices_true = np.arange(len(text))

        # asserts
        fake_tqdm.tqdm.assert_called_once()

        np.testing.assert_array_equal(X, X_true)
        np.testing.assert_array_equal(y, y_true)
        np.testing.assert_array_equal(indices, indices_true)

    def test_skip(self):
        text = "worldw"  # the only condition "ld" -> "w" is goning to make it
        vocabulary = ["d", "l", "o", "w"]  # r is missing

        X, y, indices = create_data_language(
            text, vocabulary, window_size=2, fill_strategy="skip"
        )

        X_true = np.array([[1, 0]], dtype=np.int8)

        y_true = np.array([3], dtype=np.int8)

        indices_true = np.array([5])

        np.testing.assert_array_equal(X, X_true)
        np.testing.assert_array_equal(y, y_true)
        np.testing.assert_array_equal(indices, indices_true)

    @pytest.mark.parametrize("window_size", [1, 4])
    @pytest.mark.parametrize("fill_strategy", ["zeros", "skip"])
    def test_empty_text(self, window_size, fill_strategy):
        """Make sure the dimensions are correct anyway.

        It is essential for being able to concatenate with other
        features and targets.
        """
        text = ""
        vocabulary = ["a", "b"]

        X, y, indices = create_data_language(
            text,
            vocabulary,
            window_size=window_size,
            fill_strategy=fill_strategy,
        )

        assert X.shape == (0, window_size)
        assert y.shape == (0,)


class TestSingleCharacterLSTM:
    def test_basic(self, tmpdir):
        vocabulary = ["a", "b", "c", "d"]
        hparams = {
            "vocab_size": 4,
            "hidden_size": 5,
            "n_layers": 1,
            "dense_size": 12,
        }

        network = SingleCharacterLSTM(**hparams)

        assert isinstance(network, torch.nn.Module)
        assert network.hparams == hparams

        checkpoint_path = pathlib.Path(str(tmpdir)) / "chp.pt"

        save_model(network, vocabulary, checkpoint_path)

        network_loaded, vocabulary_loaded = load_model(checkpoint_path)

        assert network_loaded.hparams == hparams
        assert vocabulary_loaded == vocabulary

    def test_forward_pass(self):
        batch_size = 3
        window_size = 2
        hparams = {
            "vocab_size": 4,
            "hidden_size": 5,
            "n_layers": 1,
            "dense_size": 12,
        }
        network = SingleCharacterLSTM(**hparams)

        # normal mode
        x1 = torch.rand(batch_size, window_size, hparams["vocab_size"])
        o1, h_n1, c_n1 = network(x1)

        assert torch.is_tensor(o1)

        assert o1.shape == (batch_size, hparams["vocab_size"])
        assert h_n1.shape == (
            hparams["n_layers"],
            batch_size,
            hparams["hidden_size"],
        )
        assert c_n1.shape == (
            hparams["n_layers"],
            batch_size,
            hparams["hidden_size"],
        )

        # continuation mode
        x2 = torch.rand(batch_size, 1, hparams["vocab_size"])
        h = torch.rand(hparams["n_layers"], batch_size, hparams["hidden_size"])
        c = torch.rand(hparams["n_layers"], batch_size, hparams["hidden_size"])

        o2, h_n2, c_n2 = network(x2, h=h, c=c)

        assert torch.is_tensor(o2)
        assert torch.is_tensor(h_n2)
        assert torch.is_tensor(c_n2)

        assert o2.shape == (batch_size, hparams["vocab_size"])
        assert h_n2.shape == (
            hparams["n_layers"],
            batch_size,
            hparams["hidden_size"],
        )
        assert c_n2.shape == (
            hparams["n_layers"],
            batch_size,
            hparams["hidden_size"],
        )


class SampleCharacter:
    pass
