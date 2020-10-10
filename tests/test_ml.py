import pathlib

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

    def test_zeros(self):
        text = "world"
        vocabulary = ["d", "l", "o", "w"]  # r is missing

        X, y, indices = create_data_language(
            text, vocabulary, window_size=2, fill_strategy="zeros"
        )

        X_true = np.array(
            [
                [[0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 1]],
                [[0, 0, 0, 1], [0, 0, 1, 0]],
                [[0, 0, 1, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 1, 0, 0]],
            ],
            dtype=np.bool,
        )

        y_true = np.array(
            [
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 0],
                [0, 1, 0, 0],
                [1, 0, 0, 0],
            ],
            dtype=np.bool,
        )

        indices_true = np.arange(len(text))

        np.testing.assert_array_equal(X, X_true)
        np.testing.assert_array_equal(y, y_true)
        np.testing.assert_array_equal(indices, indices_true)

    def test_skip(self):
        text = "worldw"  # the only condition "ld" -> "w" is goning to make it
        vocabulary = ["d", "l", "o", "w"]  # r is missing

        X, y, indices = create_data_language(
            text, vocabulary, window_size=2, fill_strategy="skip"
        )

        X_true = np.array([[[0, 1, 0, 0], [1, 0, 0, 0]]], dtype=np.bool)

        y_true = np.array([[0, 0, 0, 1]], dtype=np.bool)

        indices_true = np.array([5])

        np.testing.assert_array_equal(X, X_true)
        np.testing.assert_array_equal(y, y_true)
        np.testing.assert_array_equal(indices, indices_true)


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
