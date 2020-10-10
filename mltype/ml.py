"""Machine learning utilities."""
from collections import Counter, defaultdict
import importlib

import numpy as np
import pytorch_lightning as pl
import torch
import tqdm

from mltype.utils import get_cache_dir


def create_data_language(
    text, vocabulary, window_size=2, fill_strategy="zeros", verbose=False
):
    """Create a supervised dataset for the character-lever language model.

    Parameters
    ----------
    text : str
       Some text.

    vocabulary : list
        Unique list of supported characters. Their corresponding indices
        are going to be used for the one hot encoding.

    window_size : int
        The number of previous characters to condition on.

    fill_strategy : str, {"skip", "zeros"}
        Strategy for handling initial characters and unknown characters.

    verbose : bool
        If True, progress bar is showed.

    Returns
    -------
    X : np.ndarray
        Features of shape `(len(text), window_size, len(vocabulary)` if
        the `fill_strategy="zeros"`. Otherwise the first dimension smaller.
        The dtype is `np.bool`.

    y : np.ndarray
        Targets of shape `(len(text), len(vocabulary))` if the
        `fill_strategy="zeros"`. Otherwise the first dimension smaller.
        The dtype is `np.bool`.

    indices : np.ndarray
        For each sample an index of the character we are trying to predict.
        Note that for `fill_strategy="zeros"` it is going to be
        `np.arange(len(text))`. However, for different strategies might
        have gaps. It helps us to keep track of the sample - character
        correspondence.
    """
    if not vocabulary:
        raise ValueError("The vocabulary is empty.")

    if len(vocabulary) != len(set(vocabulary)):
        raise ValueError("There are duplicates in the vocabulary.")

    vocab_size = len(vocabulary)
    text_size = len(text)

    ch2ix = defaultdict(lambda: vocab_size)
    ch2ix.update({ch: ix for ix, ch in enumerate(vocabulary)})

    ohv_matrix = np.eye(vocab_size, dtype=np.bool)
    ohv_matrix = np.concatenate(
        [ohv_matrix, np.zeros((1, vocab_size), dtype=np.bool)], axis=0
    )

    text_l = window_size * [None] + list(text)

    X_lines = []
    y_lines = []
    indices_lines = []

    iterable = range(text_size)
    if verbose:
        iterable = tqdm.tqdm(iterable)

    for i in iterable:
        feature_ixs = [
            ch2ix[text_l[i + offset]] for offset in range(window_size)
        ]
        target_ix = ch2ix[text_l[i + window_size]]

        if fill_strategy == "skip":
            if vocab_size in feature_ixs or vocab_size == target_ix:
                continue

        X_lines.append(ohv_matrix[feature_ixs].copy())
        y_lines.append(ohv_matrix[target_ix].copy())
        indices_lines.append(i)

    X = np.array(X_lines)
    y = np.array(y_lines)
    indices = np.array(indices_lines)

    return X, y, indices


def text2features(text, vocabulary):
    """Create per character one hot encoding.

    Note that we employ the zeros strategy out of vocabulary characters.

    Parameters
    ----------
    text : str
        Text.

    vocabulary : list
        Vocabulary to be used for the endcoding.

    Returns
    -------
    res : np.ndarray
        Array of shape `(len(text), len(vocabulary)` of boolean dtype.
        Each row represents the one hot encoding of the respective character.
        Note that out of vocabulary characters are encoding with a zero
        vector.
    """
    text_size = len(text)
    vocab_size = len(vocabulary)

    ch2ix = {ch: ix for ix, ch in enumerate(vocabulary)}

    output = np.zeros((text_size, vocab_size), dtype=np.bool)
    for i, ch in enumerate(text):
        try:
            output[i, ch2ix[ch]] = True
        except IndexError:
            pass

    return output


def sample_char(
    network, vocabulary, initial_text=None, random_state=None, top_k=None
):
    """Sample a character given network probability prediciton.

    Parameters
    ----------
    network : torch.nn.Module
        Trained neural network that outputs a probability distribution over
        `vocabulary`.

    vocabulary : list
        List of unique characters.

    initial_text : None or str
        Initial text to condition on.

    random_state : None or int
        Guarantees reproducibility.

    top_k : None or int
        If specified, we only sample from the top k most probably characters.
        Otherwise all of them.

    Returns
    -------
    ch : str
        A character from the vocabulary.
    """
    if initial_text:
        features = text2features(initial_text, vocabulary)
    else:
        features = np.zeros((1, len(vocabulary)), dtype=np.bool)

    features = features[None, ...]  # add batch dimension

    if random_state is not None:
        np.random.seed(random_state)

    probs = (
        network(torch.from_numpy(features).to(torch.float32))[0]
        .detach()
        .numpy()
    )

    if top_k is not None:
        probs_new = np.zeros_like(probs)
        top_k_indices = probs.argsort()[-top_k:]
        probs_new[top_k_indices] = probs[top_k_indices]

        probs = probs_new / probs_new.sum()

    return np.random.choice(vocabulary, p=probs)


def sample_text(
    n_chars,
    network,
    vocabulary,
    initial_text=None,
    window_size=5,
    random_state=None,
    top_k=None,
    verbose=False,
):
    """Sample text by unrolling character by character predictions.

    Parameters
    ----------
    n_chars : int
            Number of characters to sample.

    network : torch.nn.Module
            Pretrained character level network.

    vocabulary : list
            List of unique characters.

    initial_text : None or str
            If specified, initial text to condition based on.

    window_size : int
            How big is the conditioning window (at maximum).

    random_state : None or int
            Allows reproducibility.

    top_k : None or int
            If specified, we only sample from the top k most probably characters.
            Otherwise all of them.

    verbose : bool
            Controls verbosity.

    Returns
    -------
    text : str
            Generated text of length `n_chars + len(initial_text)`.
    """
    res = initial_text or ""

    iterable = range(n_chars)
    if verbose:
        iterable = tqdm.tqdm(iterable)

    if random_state is not None:
        np.random.seed(random_state)

    for _ in iterable:
        res += sample_char(
            network, vocabulary, initial_text=res[-window_size:], top_k=top_k
        )

    return res


class LanguageDataset(torch.utils.data.Dataset):
    """Language dataset."""

    def __init__(self, X, y, indices=None, vocabulary=None, transform=None):
        self.X = X
        self.y = y
        self.indices = indices
        self.vocubulary = vocabulary
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, ix):
        X_sample = torch.from_numpy(self.X[ix])
        y_sample = torch.from_numpy(self.y[ix])

        if self.transform is not None:
            X_sample, y_sample = self.transform(X_sample, y_sample)

        return X_sample, y_sample


class SingleCharacterLSTM(pl.LightningModule):
    """Single character recurrent neural network.

    Given some string of characters, we generate the probability distribution
    of the next character.

    The architecture starts with an LSTM (`hidden_size`, `n_layers`,
     `vocab_size`) network and then we feed the last hidden state to a fully
    connected network with one hidden layer (`dense_size`).

    Parameters
    ----------
    vocab_size : int
            Size of the vocabulary. Necessary since we are encoding each character
            as a one hot vector.

    hidden_size : int
            Hidden size of the recurrent cell.

    n_layers : int
            Number of layers in the recurrent network.

    dense_size : int
            Size of the single layer of the feed forward network.

    Attributes
    ----------
    rnn_layer : torch.nn.Module
            The recurrent network layer.

    linear_layer1 : torch.nn.Module
            Linear layer connecting the last hidden state and the single
            layer of the feedforward network.

    linear_layer2 : torch.nn.Module
            Linear layer connecting the single layer of the feedforward network
            with the output (of size `vocabulary_size`).

    activation_layer :  torch.nn.Module
            Softmax layer making sure we get a probability distribution.
    """

    def __init__(self, vocab_size, hidden_size=16, n_layers=1, dense_size=128):
        super().__init__()
        self.save_hyperparameters()
        self.rnn_layer = torch.nn.LSTM(
            input_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
        )
        self.linear_layer1 = torch.nn.Linear(hidden_size, dense_size)
        self.linear_layer2 = torch.nn.Linear(dense_size, vocab_size)

        self.activation_layer = torch.nn.Softmax(dim=1)

    def forward(self, x):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
                Input features of shape `(batch_size, window_size, vocab_size)`.
                Note that the provided `vocab_size` needs to be equal to the one
                provided in the constructor. The remaining dimensions (`batch_size`
                and `window_size`) can be any positive integers.

        Returns
        -------
        probs : torch.Tensor
                Tensor of shape `(batch_size, vocab_size)`. For each sample
                it represents the probability distribution over all characters
                in the vocabulary.
        """
        _, (h_n, _) = self.rnn_layer(x)
        average_h_n = h_n.mean(dim=0)
        x = self.linear_layer1(average_h_n)
        logits = self.linear_layer2(x)
        probs = self.activation_layer(logits)

        return probs

    def training_step(self, batch, batch_idx):
        """Implement training step.

        Necessary for pytorch-lightning.

        Parameters
        ----------
        batch : tuple
                Batch of training samples. The exact definition depends
                on the dataloader.

        batch_idx : idx
                Index of the batch.

        Returns
        -------
        loss : torch.Tensor
                Tensor of shape `(batch_size)` representing a per sample loss.
        """
        x, y = batch
        x, y = x.to(torch.float32), y.to(torch.float32)
        probs = self.forward(x)
        loss = torch.nn.functional.binary_cross_entropy(probs, y)

        result = pl.TrainResult(minimize=loss)
        result.log("train_loss", loss, prog_bar=True)

        return result

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(torch.float32), y.to(torch.float32)
        probs = self.forward(x)
        loss = torch.nn.functional.binary_cross_entropy(probs, y)

        result = pl.EvalResult()
        result.log("val_loss", loss, prog_bar=False)

        return result

    def configure_optimizers(self):
        """Configure optimizers.

        Necessary for pytorch-lightning.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        return optimizer


def run_train(
    text,
    name,
    max_epochs=10,
    window_size=50,
    batch_size=32,
    vocab_size=None,
    fill_strategy="skip",
    train_test_split=0.5,
    hidden_size=32,
    dense_size=32,
    n_layers=1,
):
    output_path = get_cache_dir() / "languages" / name

    if output_path.exists():
        raise FileExistsError(f"The model {name} already exists")

    vocabulary = sorted(
        [x[0] for x in Counter(text).most_common()][:vocab_size]
    )  # works for None
    vocab_size = len(vocabulary)

    X, y, indices = create_data_language(
        text,
        vocabulary,
        window_size=window_size,
        verbose=True,
        fill_strategy=fill_strategy,
    )

    splix_ix = int(len(X) * train_test_split)

    dataset = LanguageDataset(X, y)

    dataloader_t = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.DataLoader(np.arange(splix_ix)),
    )

    dataloader_v = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.DataLoader(np.arange(splix_ix, len(X))),
    )

    network = SingleCharacterLSTM(
        vocab_size,
        hidden_size=hidden_size,
        dense_size=dense_size,
        n_layers=n_layers,
    )

    logger = pl.loggers.MLFlowLogger(
        "mltype", save_dir=get_cache_dir() / "logs" / "mlruns"
    )
    logger.log_hyperparams(
        {
            "fill_strategy": fill_strategy,
            "model_name": name,
            "train_test_split": train_test_split,
            "vocab_size": vocab_size,
            "window_size": window_size,
        }
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        early_stop_callback=pl.callbacks.EarlyStopping(monitor="val_loss"),
    )
    trainer.fit(network, dataloader_t, dataloader_v)

    save_model(network, vocabulary, output_path)


def load_model(path):
    """Load serialized model and vocabulary.

    Parameters
    ----------
    path : pathlib.Path
            Path to where the file lies. This file was created by
            `save_model` method.

    Returns
    -------
    model_inst : SingleCharacterLSTM
            Instance of the model.

    vocabulary : list
            Corresponding vocabulary.
    """
    output_dict = torch.load(path)

    kwargs = output_dict["kwargs"]
    model_class_name = output_dict["model_class_name"]
    state_dict = output_dict["state_dict"]
    vocabulary = output_dict["vocabulary"]

    model_class = getattr(
        importlib.import_module("mltype.ml"), model_class_name
    )
    model_inst = model_class(**kwargs)
    model_inst.load_state_dict(state_dict)

    return model_inst, vocabulary


def save_model(model, vocabulary, path):
    """Serialize a model.

    Note that we require that the model has a property `hparams` that
    we can unpack into the constructor of the class and get the same
    network architecture.

    Parameters
    ----------
    model : SingleCharacterLSTM
            Torch model to be saved. Additionally, we require that it has
            the `hparams` property that contains all necessary hyperparameters
            to instantiate the model.

    vocabulary : list
            The corresponding vocabulary.

    path : pathlib.Path
            Path to the file that will whole the serialized object.
    """

    output_dict = {
        "kwargs": model.hparams,
        "model_class_name": model.__class__.__name__,
        "state_dict": model.state_dict(),
        "vocabulary": vocabulary,
    }

    torch.save(output_dict, path)
