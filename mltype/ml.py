"""Machine learning utilities."""
from collections import Counter, defaultdict
from datetime import datetime
import importlib
import pathlib
import warnings

import numpy as np
import pytorch_lightning as pl
import torch
import tqdm

from mltype.utils import get_cache_dir, get_mlflow_artifacts_path, print_section

warnings.filterwarnings("ignore")


def create_data_language(
    text, vocabulary, window_size=2, fill_strategy="zeros", verbose=False
):
    """Create a supervised dataset for the characte/-lever language model.

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
        Features array of shape `(len(text), window_size)` if
        `fill_strategy=zeros`, otherwise it might be shorter. The dtype is
        `np.int8`. If applicable, the integer `(len(vocabulary))` represnts
        a zero vector (out of vocabulary token).

    y : np.ndarray
        Targets array of shape `(len(text),)` if `fill_strategy=zeros`,
        otherwise it might be shorter. The dtype is `np.int8`.

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

    if vocab_size >= 255:  # we need to use one integer for out of vocabulary
        raise ValueError("The maximum vocabulary size is 255")

    text_size = len(text)

    ch2ix = defaultdict(lambda: vocab_size)
    ch2ix.update({ch: ix for ix, ch in enumerate(vocabulary)})

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

        X_lines.append(feature_ixs)
        y_lines.append(target_ix)
        indices_lines.append(i)

    if not X_lines:
        X = np.empty((0, window_size), dtype=np.int8)
        y = np.empty((0,), dtype=np.int8)

    else:
        X = np.array(X_lines, dtype=np.int8)
        y = np.array(y_lines, dtype=np.int8)

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
        except KeyError:
            pass

    return output


def sample_char(
    network,
    vocabulary,
    h=None,
    c=None,
    previous_chars=None,
    random_state=None,
    top_k=None,
    device=None,
):
    """Sample a character given network probability prediciton (with a state).

    Parameters
    ----------
    network : torch.nn.Module
        Trained neural network that outputs a probability distribution over
        `vocabulary`.

    vocabulary : list
        List of unique characters.

    h, c : torch.Tensor
        Hidden states with shape `(n_layers, batch_size=1, hidden_size)`.
        Note that if both of them are None we are at the very first character.

    previous_chars : None or str
        Previous charaters. None or and empty string if we are at the very
        first character.

    random_state : None or int
        Guarantees reproducibility.

    top_k : None or int
        If specified, we only sample from the top k most probably characters.
        Otherwise all of them.

    device : None or torch.device
        By default `torch.device("cpu")`.

    Returns
    -------
    ch : str
        A character from the vocabulary.
    """
    device = device or torch.device("cpu")

    if previous_chars:
        features = text2features(previous_chars, vocabulary)
    else:
        features = np.zeros((1, len(vocabulary)), dtype=np.bool)

    network.eval()

    features = features[None, ...]  # add batch dimension

    if random_state is not None:
        np.random.seed(random_state)

    x = torch.from_numpy(features).to(dtype=torch.float32, device=device)
    out, h_n, c_n = network(x, h, c)
    probs = out[0].detach().cpu().numpy()

    if top_k is not None:
        probs_new = np.zeros_like(probs)
        top_k_indices = probs.argsort()[-top_k:]
        probs_new[top_k_indices] = probs[top_k_indices]

        probs = probs_new / probs_new.sum()

    return np.random.choice(vocabulary, p=probs), h_n, c_n


def sample_text(
    n_chars,
    network,
    vocabulary,
    initial_text=None,
    random_state=None,
    top_k=None,
    verbose=False,
    device=None,
):
    """Sample text by unrolling character by character predictions.

    Note that keep the pass hidden states with each character prediciton
    and there is not need to specify a window.

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

    random_state : None or int
            Allows reproducibility.

    top_k : None or int
            If specified, we only sample from the top k most probable
            characters. Otherwise all of them.

    verbose : bool
            Controls verbosity.

    device : None or torch.device
        By default `torch.device("cpu")`.

    Returns
    -------
    text : str
            Generated text of length `n_chars + len(initial_text)`.
    """
    device = device or torch.device("cpu")
    network.eval()
    initial_text = initial_text or ""
    res = initial_text
    h, c = None, None

    iterable = range(n_chars)
    if verbose:
        iterable = tqdm.tqdm(iterable)

    if random_state is not None:
        np.random.seed(random_state)

    for _ in iterable:
        previous_chars = initial_text if res == initial_text else res[-1]
        new_ch, h, c = sample_char(
            network,
            vocabulary,
            h=h,
            c=c,
            previous_chars=previous_chars,
            top_k=top_k,
            device=device,
        )
        res += new_ch

    return res


class LanguageDataset(torch.utils.data.Dataset):
    """Language dataset.

    All the inputs of this class should be generated via
    `create_data_language`.

    Parameters
    ----------
    X : np.ndarray
        Array of shape (n_samples, window_size) of dtype `np.int8`.
        It represents the features.

    y : np.ndarray
        Array of shape (n_samples,) of dtype `np.int8`.
        It represents the targets

    vocabulary : list
        List of characters in the vocabulary.

    transform : callable or None
        Some callable that inputs `X` and `y` and returns some
        modified instances of them.

    Attributes
    ----------
    ohv_matrix : np.ndarray
        Matrix of shape `(vocab_size + 1, vocab_size)`. The submatrix
        `ohv_matrix[:vocab_size, :]` is an identity matrix and is used
        for fast creation of one hot vectors. The last row of `ohv_matrix`
        is a zero vector and is used for encoding out-of-vocabulary characters.
    """

    def __init__(self, X, y, vocabulary, transform=None):
        self.X = X
        self.y = y
        self.vocabulary = vocabulary
        self.transform = transform

        vocab_size = len(vocabulary)

        ch2ix = defaultdict(lambda: vocab_size)
        ch2ix.update({ch: ix for ix, ch in enumerate(vocabulary)})

        ohv_matrix = np.eye(vocab_size, dtype=np.float32)
        self.ohv_matrix = np.concatenate(
            [ohv_matrix, np.zeros((1, vocab_size), dtype=np.float32)], axis=0
        )

    def __len__(self):
        """Compute the number of samples."""
        return len(self.X)

    def __getitem__(self, ix):
        """Get a single sample.

        Parameters
        ----------
        ix : int
            Index od the sample.

        Returns
        -------
        X_sample : np.ndarray
            Array of shape `(window_size, vocab_size)` where each
            row is either an one hot vector (inside of vocabulary character) or
            a zero vector (out of vocabulary character).

        y_sample : np.ndarray
            Array of shape `(vocab_size,)` representing either the one hot
            encoding of the character to be predicted (inside of vocabulary
            character) or a zero vector (out of vocabulary character).

        vocabulary : list
            The vocabulary. The reason why we want to provide this too
            is to have access to it during validation.
        """
        X_sample = torch.from_numpy(self.ohv_matrix[self.X[ix]])
        y_sample = torch.from_numpy(self.ohv_matrix[self.y[ix]])

        if self.transform is not None:
            X_sample, y_sample = self.transform(X_sample, y_sample)

        # unfortunatelly vocab will get collated to a batch, but whatever
        return X_sample, y_sample, self.vocabulary


class SingleCharacterLSTM(pl.LightningModule):
    """Single character recurrent neural network.

    Given some string of characters, we generate the probability distribution
    of the next character.

    Architecture starts with an LSTM (`hidden_size`, `n_layers`, `vocab_size`)
    network and then we feed the last hidden state to a fully
    connected network with one hidden layer (`dense_size`).

    Parameters
    ----------
    vocab_size : int
            Size of the vocabulary. Necessary since we are encoding each
            character as a one hot vector.

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

    def forward(self, x, h=None, c=None):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape `(batch_size, window_size, vocab_size)`.
            Note that the provided `vocab_size` needs to be equal to the one
            provided in the constructor. The remaining dimensions
            (`batch_size` and `window_size`) can be any positive integers.

        h, c : torch.Tensor
            Hidden states of shape `(n_layers, batch_size, hidden_size)`. Note
            that if provided we enter a continuation mode. In this case
            to generate the prediction we just use the last character and the
            hidden state for the prediction. Note that in this case
            we enforce that `x.shape=(batch_size, 1, vocab_size)`.

        Returns
        -------
        probs : torch.Tensor
            Tensor of shape `(batch_size, vocab_size)`. For each sample
            it represents the probability distribution over all characters
            in the vocabulary.

        h_n, c_n : torch.Tensor
            New Hidden states of shape `(n_layers, batch_size, hidden_size)`.
        """
        continuation_mode = h is not None and c is not None

        if continuation_mode:
            if not (x.ndim == 3 and x.shape[1] == 1):
                raise ValueError("Wrong input for the continuation mode")

            _, (h_n, c_n) = self.rnn_layer(x, (h, c))

        else:
            _, (h_n, c_n) = self.rnn_layer(x)

        average_h_n = h_n.mean(dim=0)
        x = self.linear_layer1(average_h_n)
        logits = self.linear_layer2(x)
        probs = self.activation_layer(logits)

        return probs, h_n, c_n

    def training_step(self, batch, batch_idx):
        """Run training step.

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
                Tensor scalar representing the mean binary cross entropy
                over the batch.
        """
        x, y, _ = batch
        probs, _, _ = self.forward(x)
        loss = torch.nn.functional.binary_cross_entropy(probs, y)

        self.log("train_loss", loss, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        """Run validation step.

        Optional for pytorch-lightning.

        Parameters
        ----------
        batch : tuple
            Batch of validation samples. The exact definition depends
            on the dataloader.


        batch_idx : idx
            Index of the batch.

        Returns
        -------
        vocabulary : list
            Vocabulary in order to have access in
            `validation_epoch_end`.

        """
        x, y, vocabulary = batch
        probs, _, _ = self.forward(x)
        loss = torch.nn.functional.binary_cross_entropy(probs, y)

        self.log("val_loss", loss, prog_bar=True)

        return vocabulary

    def validation_epoch_end(self, outputs):
        """Run epoch end validation logic.

        We sample 5 times 100 characters from the current network. We
        then print to the standard output.

        Parameters
        ----------
        outputs : list
            List of batches that were collected over the validation
            set with `validation_step`.

        """
        if self.logger is None:
            return

        vocabulary = np.array(outputs[-1])[:, 0]

        n_samples = 5
        n_chars = 100

        lines = [
            sample_text(n_chars, self, vocabulary, device=self.device)
            for _ in range(n_samples)
        ]
        text = "\n".join(lines)

        artifacts_path = get_mlflow_artifacts_path(
            self.logger.experiment, self.logger.run_id
        )

        output_path = artifacts_path / f"{datetime.now()}.txt"

        output_path.write_text(text)

    def configure_optimizers(self):
        """Configure optimizers.

        Necessary for pytorch-lightning.

        Returns
        -------
        optimizer : Optimizer
            The chosen optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        return optimizer


def run_train(
    texts,
    name,
    max_epochs=10,
    window_size=50,
    batch_size=32,
    vocab_size=None,
    fill_strategy="skip",
    illegal_chars="",
    train_test_split=0.5,
    hidden_size=32,
    dense_size=32,
    n_layers=1,
    checkpoint_path=None,
    output_path=None,
    use_mlflow=True,
    early_stopping=True,
    gpus=None,
):
    """Run the training loop.

    Note that the parameters are also explained in the cli of `mlt train`.

    Parameters
    ----------
    texts : list
        List of str representing all texts we would like to train on.

    name : str
        Name of the model. This name is only used when we save the model -
        it is not hardcoded anywhere in the serialization.

    max_epochs : int
        Maximum number of epochs. Note that the number of actual epochs
        can be lower if we activate the `early_stopping` flag.

    window_size : int
        Number of previous characters to consider when predicting the next
        character. The higher the number the longer the memory we are
        enforcing. Howerever, at the same time, the training becomes slower.

    batch_size : int
        Number of samples in one batch.

    vocab_size : int
        Maximum number of characters to be put in the vocabulary. Note that
        one can explicityly exclude characters via `illegal_chars`. The higher
        this number the bigger the feature vectors are and the slower the
        training.

    fill_strategy : str, {"zeros", "skip"}
        Determines how to deal with out of vocabulary characters. When
        "zeros" then we simply encode them as zero vectors. If "skip", we
        skip a given sample if any of the characters in the window or the
        predicted character are not in the vocabulary.

    illegal_chars : str or None
        If specified, then each character of the str represents a forbidden
        character that we do not put in the vocabulary.

    train_test_split : float
        Float in the range (0, 1) representing the percentage of the training
        set with respect to the entire dataset.

    hidden_size : int
        Hidden size of LSTM cells (equal in all layers).

    dense_size : int
        Size of the dense layer that is bridging the hidden state outputted
        by the LSTM and the final output probabilities over the vocabulary.

    n_layers : int
        Number of layers inside of the LSTM.

    checkpoint_path : None or pathlib.Path or str
        If specified, it is pointing to a checkpoint file (generated
        by Pytorch-lightning). This file does not contain the vocabulary.
        It can be used to continue the training.

    output_path : None or pathlib.Path or str
        If specified, it is an alternative output folder when the trained
        models and logging information will be stored. If not specified
        the output folder is by default set to `~/.mltype`.

    use_mlflow : bool
        If active, than we use mlflow for logging of training and validation
        loss. Additionally, at the end of each epoch we generate a few
        sample texts to demonstrate how good/bad the current network is.

    early_stopping : bool
        If True, then we monitor the validation loss and if it does not
        improve for a certain number of epochs then we stop the traning.

    gpus : int or None
        If None or 0, no GPUs are used (only CPUs). Otherwise, it represents
        the number of GPUs to be used (using the data parallelization
        strategy).
    """
    illegal_chars = illegal_chars or ""

    cache_dir = get_cache_dir(output_path)
    languages_path = cache_dir / "languages" / name
    checkpoints_path = cache_dir / "checkpoints" / name

    if languages_path.exists():
        raise FileExistsError(f"The model {name} already exists")

    with print_section(" Computing vocabulary ", drop_end=True):
        vocabulary = sorted(
            [
                x[0]
                for x in Counter("".join(texts)).most_common()
                if x[0] not in illegal_chars
            ][:vocab_size]
        )  # works for None
        vocab_size = len(vocabulary)
        print(f"# characters: {vocab_size}")
        print(vocabulary)

    with print_section(" Creating training set ", drop_end=True):
        X_list = []
        y_list = []
        for text in tqdm.tqdm(texts):
            X_, y_, _ = create_data_language(
                text,
                vocabulary,
                window_size=window_size,
                verbose=False,
                fill_strategy=fill_strategy,
            )
            X_list.append(X_)
            y_list.append(y_)
        X = np.concatenate(X_list, axis=0) if len(X_list) != 1 else X_list[0]
        y = np.concatenate(y_list, axis=0) if len(y_list) != 1 else y_list[0]

        print(f"X.dtype={X.dtype}, y.dtype={y.dtype}")

        split_ix = int(len(X) * train_test_split)
        indices = np.random.permutation(len(X))
        train_indices = indices[:split_ix]
        val_indices = indices[split_ix:]
        print(f"Train: {len(train_indices)}\nValidation: {len(val_indices)}")

    dataset = LanguageDataset(X, y, vocabulary=vocabulary)

    dataloader_t = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_indices),
    )

    dataloader_v = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(val_indices),
    )

    if checkpoint_path is None:
        network = SingleCharacterLSTM(
            vocab_size,
            hidden_size=hidden_size,
            dense_size=dense_size,
            n_layers=n_layers,
        )
    else:
        print(f"Loading a checkpointed network: {checkpoint_path}")
        network = SingleCharacterLSTM.load_from_checkpoint(str(checkpoint_path))

    chp_name_template = str(checkpoints_path / "{epoch}-{val_loss:.3f}")
    chp_callback = pl.callbacks.ModelCheckpoint(
        filepath=chp_name_template,
        save_last=True,  # last epoch always there
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
        save_weights_only=False,
    )
    callbacks = []

    if use_mlflow:
        print("Logging with MLflow")
        logger = pl.loggers.MLFlowLogger(
            "mltype", save_dir=get_cache_dir(output_path) / "logs" / "mlruns"
        )
        print(f"Run ID: {logger.run_id}")

        logger.log_hyperparams(
            {
                "fill_strategy": fill_strategy,
                "model_name": name,
                "train_test_split": train_test_split,
                "vocab_size": vocab_size,
                "window_size": window_size,
            }
        )
    else:
        logger = None

    if early_stopping:
        print("Activating early stopping")
        callbacks.append(
            pl.callbacks.EarlyStopping(monitor="val_loss", verbose=True)
        )

    with print_section(" Training ", drop_end=True):
        trainer = pl.Trainer(
            gpus=gpus,
            max_epochs=max_epochs,
            logger=logger,
            callbacks=callbacks,
            checkpoint_callback=chp_callback,
            replace_sampler_ddp=False,
        )
        trainer.fit(network, dataloader_t, dataloader_v)

    with print_section(" Saving the model ", drop_end=False):
        if chp_callback.best_model_path:
            print(f"Using the checkpoint {chp_callback.best_model_path}")
            network = SingleCharacterLSTM.load_from_checkpoint(
                chp_callback.best_model_path
            )
        else:
            print("No checkpoint found, using the current network")

        print(f"The final model is saved to: {languages_path}")
        save_model(network, vocabulary, languages_path)


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
            Instance of the model. Note that all of its parameters
            will be lying on a CPU.

    vocabulary : list
            Corresponding vocabulary.
    """
    output_dict = torch.load(path, map_location=torch.device("cpu"))

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
    network architecture. This is automatically the case if we subclass
    from `pl.LightningModule`.

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

    path_parent = pathlib.Path(path).parent
    path_parent.mkdir(parents=True, exist_ok=True)

    torch.save(output_dict, path)
