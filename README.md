<p align="center">
  <img src="https://user-images.githubusercontent.com/18519371/97606153-c19c2700-1a0f-11eb-9faf-876f266b4585.png">
</p>

![mltype](https://github.com/jankrepl/mltype/workflows/mltype/badge.svg)

Command line tool for improving typing speed and accuracy. The main goal is
to help programmers practise programming languages.

# Demo
<p align="center">
  <img src="https://i.imgur.com/uz1046g.gif">
</p>

# Installation
```bash
pip install --upgrade mltype
```

# Main features
### Text generation

- Using neural networks to generate text. One can use
  pretrained networks (see below) or train new ones from scratch.
- Alternatively, one can read text from a file or provide it manually

### Typing interface
- Dead simple (implemented in `curses`)
- Basic statistics - WPM and accuracy
- Setting target speed
- Playing against past performances

# Documentation and usage
- Detailed documentation: https://mltype.readthedocs.io/en/latest/index.html.
- GIF examples: https://mltype.readthedocs.io/en/latest/source/examples.html. 

The entrypoint is `mlt`

```bash
$ mlt
Usage: mlt [OPTIONS] COMMAND [ARGS]...

  Tool for improving typing speed and accuracy

Options:
  --help  Show this message and exit.

Commands:
  file    Type text from a file
  ls      List all language models
  random  Sample characters randomly from a vocabulary
  raw     Provide text manually
  replay  Compete against a past performance
  sample  Sample text from a language
  train   Train a language
```

# Pretrained models
See below a list of pretrained models. They are stored on a google drive
and one needs to download the entire archive.

|         Name         	|                               Info                              	| Link                                                                                       	|
|:--------------------:	|:---------------------------------------------------------------:	|--------------------------------------------------------------------------------------------	|
| C++                  	| Trained on https://github.com/TheAlgorithms/C-Plus-Plus         	| [link](https://drive.google.com/file/d/1ea49gaUWJea_-nnT4aI2TpwfG5OLQdlw/view?usp=sharing) 	|
| CPython              	| Trained on https://github.com/python/cpython/tree/master/Python 	| [link](https://drive.google.com/file/d/1aKnOkxcSYdpXYCB6yMOpbGJIw2ribVEq/view?usp=sharing) 	|
| Crime and Punishment 	| Trained on http://www.gutenberg.org/ebooks/2554                 	| [link](https://drive.google.com/file/d/1-KgO-9X3z-Xr2SLAgAI_Ijllw7L9MFpN/view?usp=sharing) 	|
| Dracula              	| Trained on http://www.gutenberg.org/ebooks/345                  	| [link](https://drive.google.com/file/d/1Fx2cZ4gOaioJymsUCY_Q620Yk53bZQeK/view?usp=sharing) 	|
| JavaScript           	| Trained on https://github.com/trekhleb/javascript-algorithms    	| [link](https://drive.google.com/file/d/1npW4YN7y2d4Id0WhXVnT_0--slmPEfW0/view?usp=sharing) 	|
| Python               	| Trained on https://github.com/TheAlgorithms/Python              	| [link](https://drive.google.com/file/d/14W-Ymi-h6jqNyqM5yGXyzwG25J3zzdn3/view?usp=sharing) 	|
| Scikit-learn         	| Trained on https://github.com/scikit-learn/scikit-learn         	| [link](https://drive.google.com/file/d/1Hl_DcXOSH8B6IxJ9fHBmoSkEOXFQ1q86/view?usp=sharing) 	|


Once you download the file, you will need to place it in `~/.mltype/languages`.
Note that if the folder does not exist you will have to create it. The file name
can be changed to whatevery you like. This name will then be used to
refer to the model.

To verify that the model was downloaded succesfully, try to sample from it.
**Note that this might take 20+ seconds the first time around.**

```bash
mlt sample my_new_model
```

# Credits
This project is very much motivated by the The Unreasonable Effectiveness of 
Recurrent Neural Networks by Andrej Karpathy.
