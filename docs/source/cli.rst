Command Line Interface
======================
The command line interface (CLI) is the primary way of using
:code:`mltype`. After installation, one can use the entrypoint
:code:`mlt` that is going to be in the path.

.. code-block:: bash

	$ mlt
	Usage: mlt [OPTIONS] COMMAND [ARGS]...

	  Tool for improving typing speed and accuracy

	Options:
	  --help  Show this message and exit.

	Commands:
	  file    Type text from a file.
	  ls      List all language models
	  random  Sample characters randomly from a provided vocabulary
	  raw     Provide text manually
	  replay  Compete against a past performance
	  sample  Sample text from a language
	  train   Train a language


Note that :code:`mltype` uses the folder :code:`~/.mltype` (in the home 
directory) for storing all relevant data. See below the usual structure.

.. code-block:: bash

   - .mltype/
      - config.ini
      - checkpoints/
          - a/  # training checkpoints of model a
          - b/  # training checkpoints of model b
      - languages/
          - a  # some model
          - b  # some other model
          ...
      - logs/
         .. 


.. _file:

file
----
Type random (or fixed) lines from a text file. This command has
two main modes: 

1. **Random lines** - Select random consecutive lines. One needs to specify
   :code:`--n-lines` and optionally the :code:`random-state` (for
   reproducibility).

2. **Fixed lines** - One needs to specify :code:`--start-line` and
   :code:`--end-line`.


Arguments
~~~~~~~~~
- :code:`PATH` - Path to the text file to read from

Options
~~~~~~~
- :code:`-e, --end-line` :code:`INTEGER` - The end line of the excerpt to use.
  Needs to be used together with start-line.

- :code:`-f, --force-perfect` - All characters need to be typed correctly

- :code:`-i, --instant-death` - End game after the first mistake

- :code:`-l, --n-lines` :code:`INTEGER` - Number of consecutive lines to be
  selected at random. Cannot be used together with start-line and end-line.

- :code:`-o, --output-file` :code:`PATH` - Path to where to save the result file

- :code:`-r, --random-state` :code:`INTEGER`

- :code:`-s, --start-line` :code:`INTEGER` - the start line of the excerpt to
  use. needs to be used together with end-line.

- :code:`-t, --target-wpm` :code:`INTEGER` - The desired speed to be shown as
  a guide

- :code:`-w, --include-whitespace` - Include whitespace characters.


Examples
~~~~~~~~
Let us first create a text file

.. code-block:: bash

    echo $'zeroth\nfirst\nsecond\nthird\nfourth\nfifth\nsixth' > text.txt
    cat text.txt

.. code-block:: text

    zeroth
    first
    second
    third
    fourth
    fifth
    sixth

To select contiguous lines randomly, one can to specify :code:`-l, --n_lines` 
representing the number of lines to use.


.. code-block:: bash

   mlt file -l 2 text.txt 

Which would open the typing interface with 2 random contiguous lines

:: 

   second third

The other option would be to use the deterministic mode and 
select the starting and ending line manually

.. code-block:: bash

   mlt file -s 0 -e 3 text.txt

:: 
  
   zeroth first second

As multiple commands, one can specify a target speed and an output file.
Note that we follow the Python convention - line counting starts from
zero and the intervals contain the starting line but not the ending
one.

Note that one can keep the whitespace characters (including newlines)
in the text by adding the :code:`-w, --include_whitespace` option

.. code-block:: bash

   mlt file -l 2 -w text.txt

:: 

    second
    third

.. _ls:

ls
--
List available language models. One can use them with :ref:`sample`.

Please check the official github to download pretrained models - 
`mltype github <https://github.com/jankrepl/mltype>`_.


.. note::

   :code:`mlt ls` simply lists all the files present
   in :code:`~.mltype/languages`.

Examples
~~~~~~~~

.. code-block:: bash

   mlt ls

.. code-block:: text

   python
   some_amazing_model
   wikipedia

.. _random:

random
------
Generate random sequence of characters based on provided counts. 
The absolute counts are converted to relative counts (probability distribution)
that we sample from.


.. note::

   :code:`mlt random` samples characters independently unlike
   :code:`mlt sample` which conditions on previous characters.

Arguments
~~~~~~~~~
- :code:`CHARACTERS` - Characters to include in the vocabulary. The higher
  the number of occurances of a given character the higher the probabilty
  of this character being sampled.

Options
~~~~~~~
- :code:`-f, --force-perfect` - All characters need to be typed correctly

- :code:`-i, --instant-death` - End game after the first mistake

- :code:`-n, --n-chars` :code:`INTEGER` - Number of characters to sample

- :code:`-o, --output-file` :code:`PATH` - Path to where to save the result file

- :code:`-t, --target-wpm` :code:`INTEGER` - The desired speed to be shown as
  a guide


Examples
~~~~~~~~
Let's say we want to practise typing of digits. However, we would like to spend
more time on 5's and 6's since they are harder. 

.. code-block:: bash

    mlt random "123455556666789    "

This would give us something like this.

::

    546261561 3566  53 5496 556659554 435 1386559569  5 85641553465118589 

We see that the most frequent characters are 5's, 6's and spaces.


.. _raw:

raw
---
Provide text manually.

Arguments
~~~~~~~~~
- :code:`TEXT` - Text to be transfered to the typing interface

Options
~~~~~~~
- :code:`-f, --force-perfect` - All characters need to be typed correctly

- :code:`-i, --instant-death` - End game after the first mistake

- :code:`-o, --output-file` :code:`PATH` - Path to where to save the result file

- :code:`-r, --raw-string` - If active, then newlines and tabs are not seen as
  special characters

- :code:`-t, --target-wpm` :code:`INTEGER` - The desired speed to be shown as
  a guide


Examples
~~~~~~~~
Let's say we have some text in the clipboard that we just paste and type. 
Additionally, we want to see the 80 word per minute (WPM) marker. Lastly,
no errors are acceptable—instant death mode.

.. code-block:: bash

    mlt raw -i -t 80 "Hello world I will write you quickly"

::

    Hello world I will write you quickly 


replay
------
Play against a past performance. To save a past
performance one can use the option :code:`-o, --output_file` of the following
commands 

- :ref:`file`
- :ref:`random`
- :ref:`raw`
- :ref:`sample`

Arguments
~~~~~~~~~
- :code:`REPLAY_FILE` - Past performance to play against

Options
~~~~~~~
- :code:`-f, --force-perfect` - All characters need to be typed correctly

- :code:`-i, --instant-death` - End game after the first mistake

- :code:`-t, --target-wpm` :code:`INTEGER` - The desired speed to be shown as
  a guide

- :code:`-w, --overwrite` :code:`PATH` - Overwrite in place if faster

Examples
~~~~~~~~
We ran :code:`mlt sample ... -o replay_file` and we are not particularly happy
about the performance. We would like to replay the same text and try to
improve our speed. In case we do, we would like the :code:`replay_file` to be
updated automatically (using the :code:`-w, --overwrite` option).

.. code-block:: bash

    mlt replay -w replay_file

:: 

    Some text we already typed before.


.. _sample:

sample
------
Generate text using a character-level language model.

.. note::

    As opposed to :code:`mlt random`, the :code:`mlt sample` command
    is taking into consideration all the previous characters and
    therefore could generate more realistic text.

To see all the available models use :ref:`ls`. Please
check the official github to download pretrained models - 
`mltype github <https://github.com/jankrepl/mltype>`_.

Arguments
~~~~~~~~~
- :code:`MODEL_NAME` - Name of the language model

Options
~~~~~~~
- :code:`-f, --force-perfect` - All characters need to be typed correctly

- :code:`-i, --instant-death` - End game after the first mistake

- :code:`-k, --top-k` :code:`INTEGER`  - Consider only the top k most probable
  characters

- :code:`-n, --n-chars` :code:`INTEGER` - Number of characters to generate

- :code:`-o, --output-file` :code:`PATH` - Path to where to save the result file

- :code:`-r, --random-state` :code:`INTEGER` - Random state for reproducible
  results

- :code:`-s, --starting-text` :code:`TEXT` - Initial text used as a starting
  condition

- :code:`-t, --target-wpm` :code:`INTEGER` - The desired speed to be shown as
  a guide

- :code:`-v, --verbose` Show progressbar when generating text


Examples
~~~~~~~~
We want to practise typing Python without having to worry about having real
source code. Assuming we have a decent language model for Python (see
:ref:`train`) called :code:`amazing_python_model` then we can do the following

.. code-block:: bash

   mlt sample amazing_python_model


::

    spatial_median(X, method="lar", call='Log', Cov']) glm.fit(X, y) assert_all
    close(ref_no_encoded_c


Maybe we would like to give the model some initial text
and let it complete it for us.

.. code-block:: bash

    mlt sample -s "@pytest.mark.parametrize" amazing_python_model

::

    @pytest.mark.parametrize('solver', ['sparse_cg', 'sag', 'saga']) 
    @pytest.mark.parametrize('copy_X', ['not a number', -0.10]]   

.. _train:

train
-----
Train a character-level language model. The trained model can
then be used with :ref:`sample`.

In the background, we use an LSTM and feedforward network architecture
to achieve this task. The user can set most of the important hyperparameters
via the CLI options. Note that one can train without a GPU, however, 
to get access to bigger networks and faster training (~minutes/hours) GPUs
are recommended.

Arguments
~~~~~~~~~
- :code:`PATH_1`, :code:`PATH_2`, ... - Paths to files or folders containing
  text to be trained on

- :code:`MODEL_NAME` - Name of the trained model

Options
~~~~~~~
- :code:`-b, --batch-size` :code:`INTEGER` - Number of samples in a batch

- :code:`-c, --checkpoint-path` :code:`PATH` - Load a checkpoiont and continue training it

- :code:`-d, --dense-size` :code:`INTEGER` - Size of the dense layer

- :code:`-e, --extensions` :code:`TEXT` - Comma-separated list of allowed extensions

- :code:`-f, --fill-strategy` :code:`TEXT` - Either zeros or skip. Determines how to deal
  with out of vocabulary characters

- :code:`-g, --gpus` :code:`INTEGER` - Number  of gpus. In not specified, then none.
  If -1, then all.

- :code:`-h, --hidden_size` :code:`INTEGER` - Size of the hidden state

- :code:`-i, --illegal-chars` :code:`TEXT` - Characters to exclude from the
  vocabulary.

- :code:`-l, --n-layers` :code`INTEGER` - Number of layesr in the recurrent
  network

- :code:`-m, --use-mlflow` - Use MLFlow for logging

- :code:`-n, --max-epochs` :code:`INTEGER` - Maximum number of epochs

- :code:`-o, --output-path` :code:`PATH` - Custom path where to save the
  trained models and logging details. If not provided it defaults to
  `~/.mltype`.

- :code:`-s, --early-stopping` - Enable early stopping based on validation
  loss

- :code:`-t, --train-test-split` :code:`FLOAT` - Train test split - value between (0, 1) 

- :code:`-v, --vocab-size` :code:`INTEGER` - Number of the most frequent
  characters to include in the vocabulary

- :code:`-w, --window-size` :code:`INTEGER` - Number of previous characters
  to consider for prediction

Examples
~~~~~~~~
Let's assume we have a book in fulltext saved in the :code:`book.txt` file. Our
goal would be to train a model that learns the language used in this book
and is able to sample new pieces of text that resemble the original. 

See below a list of hyperparameters that work reasonably well and the
training can be done in a few hours (on a GPU)

- :code:`--batch-size` 128
- :code:`--dense-size` 1024
- :code:`--early-stopping` 
- :code:`--gpus` 1
- :code:`--hidden-size` 512
- :code:`--max-epochs` 10
- :code:`--n-layers` 3
- :code:`--vocab-size` 70
- :code:`--window-size` 100

So overall the commands looks like

.. code-block:: bash

   mlt train book.txt cool_model -n 3 -s -g 1 -b 128 -l 3 -h 512 -d 1024 -w 100 -v 80 


During the training, one can see progress bars and the training and 
validation loss (using :code:`pytorch-lightning` in the background).
Once the training is done, the best model (based the validation loss)
will be stored in :code:`~/.mltype/languages/cool_model`.

There are several important customizatons that one should be aware of.

**Using MLflow**

If one wants to get more training progress information theere is a flag
:code:`--use-mlflow` (requiring :code:`mlflow` being installed). To launch
the ui run the following commands

.. code-block:: bash

   cd ~/.mltype/logs
   mlflow ui 


**Multiple files**

:code:`mlt train` supports training from multiple files and folders.
This is really useful if we want to recursively create a training
set of all files in a given folder (e.g. github repository). Additionally,
one can use the :code:`--extensions` to control what files are considered
when traversing  a folder.

.. code-block:: bash

   mlt train main.py folder_with_a_lot_of_files model --extensions ".py"

The above command will create a training set out of all files inside 
of the :code:`folder_with_a_lot_of_files` folder having the
".py" suffix and also the `main.py`.


**Excluding undesirable characters**

If the input files contain some characters that we do not want the model
to have in its vocabulary, we can simply use the :code:`--illegal-chars`
option. Internally, when an out of vocabulary character is encounter, there
are two strategies to handle this (controled via :code:`--fill-strategy`)

- **zeros** - vector of zeros is used
- **skip** - only consider samples that do not have out of vocabulary
  characters anywhere in their window

.. code-block:: bash

  mlt train book.txt cool_model --illegal-chars "~{}`[]"



Configuration file
------------------
:code:`mltype` supports a configuration file that can be used for the following
tasks.

1. Setting reasonable defaults for any of the CLI commands
2. Defining custom parameters that cannot be set via the CLI

The configuration file is optional and one does not have to create it. By default
it should be located under :code:`~/.mltype/config.ini`. One can also pass it
dynamically via the :code:`--config` option available for all commands.

See below an example configuration file.

.. code-block:: bash

    [general]
    models_dir = /home/my_models
    color_default_background = terminal
    color_wrong_foreground = yellow

    [sample]
    # one needs to use underscores instead of hyphens
    n_chars = 500
    target_wpm = 70

    [raw]
    instant_death = True

General section
~~~~~~~~~~~~~~~
The :code:`general` section can be used for defining special parameters
that cannot be set via the options of the CLI. Below is a complete list
of valid parameters.

* :code:`models_dir`: Alternative location of the language models. The
  default directory is :code:`~/.mltype/languages`. It influences the
  behavior of :code:`ls` and :code:`sample`.

* :code:`color_default_background`: Background color of a default character.
  Note that it is either the character that has not been typed yet or that
  was backspaced (error correction).
* :code:`color_default_foreground`:  Foreground (font) color of a default
  character
* :code:`color_correct_background`: Background color of a correct character
* :code:`color_correct_foreground`: Foreground color of a correct character
* :code:`color_wrong_background`: Background color of wrong character
* :code:`color_wrong_foreground`: Foreground color of a wrong character
* :code:`color_replay_background`: Background color of a replay character
* :code:`color_replay_foreground`: Foreground color of a replay character
* :code:`color_target_background`: Background color of a target character
* :code:`color_target_foreground`: Foreground color of a target character

.. note::

   **Available colors**

   * :code:`terminal` - the color is inherited from the terminal
   * :code:`black`
   * :code:`red`
   * :code:`green`
   * :code:`yellow`
   * :code:`blue`
   * :code:`magenta`
   * :code:`cyan`
   * :code:`white`

Other sections
~~~~~~~~~~~~~~
All the other sections are identical to the commands names, that is

* :code:`file`
* :code:`ls`
* :code:`random`
* :code:`raw`
* :code:`replay`
* :code:`sample`
* :code:`train`

Note that if the same option is specified both in the configuartion file
and the CLI option the CLI value will have preference.

.. note::
    **Formatting rules**

    * The section names and parameter names are case insensitive
    * One needs to use underscores instead of hyphens



   
