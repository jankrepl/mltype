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
