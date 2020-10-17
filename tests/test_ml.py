import pathlib
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from mltype.ml import (
    SingleCharacterLSTM,
    LanguageDataset,
    create_data_language,
    load_model,
    sample_char,
    sample_text,
    save_model,
    text2features
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
        assert indices.shape == (0,)

class TestText2Features:
    def test_basic(self):
        text = "aabd"
        vocabulary = ["a", "b", "c"]


        res_true = np.array([
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0]], dtype=np.bool
        )
        res = text2features(text, vocabulary)

        np.testing.assert_array_equal(res, res_true)

class TestSampleChar:
    def test_error(self):
        with pytest.raises(ValueError):
            sample_char(Mock(spec=torch.nn.Module), ["a", "b"],
                        previous_char="aasdfsadfsa")

    @pytest.mark.parametrize("top_k", [None, 2])
    @pytest.mark.parametrize("random_state", [None, 3])
    @pytest.mark.parametrize("previous_char", [None, "s"])
    def test_basic(self, top_k, random_state, previous_char):
        network = Mock(spec=torch.nn.Module)
        network.return_value = torch.tensor([[0, 1, 0]]), "h", "c"
        vocabulary = ["a", "b", "c"]

        ch, h, c = sample_char(network,
                         vocabulary,
                         previous_char=previous_char,
                         random_state=random_state,
                         top_k=top_k)

        assert ch == "b"
        assert h == "h"
        assert c == "c"

class TestSampleText:
    @pytest.mark.parametrize("n_chars", [0, 2, 5])
    def test_overall(self, monkeypatch, n_chars):
        fake_sample_char = Mock()
        fake_sample_char.return_value = "b", None, None
        fake_tqdm = Mock()
        fake_tqdm.tqdm.side_effect = lambda x: x

        monkeypatch.setattr("mltype.ml.sample_char", fake_sample_char)
        monkeypatch.setattr("mltype.ml.tqdm", fake_tqdm)

        res = sample_text(n_chars,
                          Mock(spec=torch.nn.Module),
                          ["a"],
                          random_state=1,
                          verbose=True)

        assert res == n_chars * "b"

class TestLanguageDataset:
    def test_overall(self):
        vocabulary = ["a", "b", "c"]
        window_size = 3

        X = np.array([[0, 0, 3], [1, 0, 2]], dtype=np.int8)
        y = np.array([2, 0], dtype=np.int8)
        transform = lambda x,y: (x,y)

        ld = LanguageDataset(X, y, vocabulary, transform=transform)

        assert len(ld) == 2

        X_sample, y_sample, vocabulary_ = ld[0]

        X_sample_true = np.array(
            [
                [1, 0, 0],
                [1, 0, 0],
                [0, 0, 0]

        ], dtype=np.float32)

        y_sample_true = np.array([0, 0, 1], dtype=np.float32)

        assert vocabulary_ == vocabulary
        np.testing.assert_array_equal(X_sample, X_sample_true)
        np.testing.assert_array_equal(y_sample, y_sample_true)

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

