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
### Python environment
```bash
pip install --upgrade mltype
```

### Docker
Make sure that Docker and Docker Compose are installed.
```
docker-compose run --rm mltype
```
You will get a shell in a running container and the `mlt` command should be
available. 

See the documentation for more information.

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

The entrypoint is `mlt`. To get information on how to use the subcommands
use the `--help` flag (e.g. `mlt file --help`).

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
See below for a list of pretrained models. They are stored on a google drive
and one needs to download the entire archive.

|         Name         	|                               Info                              	| Link                                                                                       	|
|:--------------------:	|:---------------------------------------------------------------:	|--------------------------------------------------------------------------------------------	|
| C++                  	| Trained on https://github.com/TheAlgorithms/C-Plus-Plus         	| [link](https://drive.google.com/file/d/1ea49gaUWJea_-nnT4aI2TpwfG5OLQdlw/view?usp=sharing) 	|
| C#                  	| Trained on https://github.com/TheAlgorithms/C-Sharp           	| [link](https://drive.google.com/file/d/1_SEze1jUN0YKZDx-WOEucZ-y9pAc5SLX/view?usp=sharing) 	|
| CPython              	| Trained on https://github.com/python/cpython/tree/master/Python 	| [link](https://drive.google.com/file/d/1aKnOkxcSYdpXYCB6yMOpbGJIw2ribVEq/view?usp=sharing) 	|
| Crime and Punishment 	| Trained on http://www.gutenberg.org/ebooks/2554                 	| [link](https://drive.google.com/file/d/1-KgO-9X3z-Xr2SLAgAI_Ijllw7L9MFpN/view?usp=sharing) 	|
| Dracula              	| Trained on http://www.gutenberg.org/ebooks/345                  	| [link](https://drive.google.com/file/d/1Fx2cZ4gOaioJymsUCY_Q620Yk53bZQeK/view?usp=sharing) 	|
| Elixir              	| Trained on https://github.com/phoenixframework/phoenix                | [link](https://drive.google.com/file/d/19fgnGp3oyFEBvy2HtsMZjnzrUy7L2qWa/view?usp=sharing) 	|
| Go            	| Trained on https://github.com/TheAlgorithms/Go    	                | [link](https://drive.google.com/file/d/1VOw0zCa4xRgfeRz41loPkQRwcWadeV-o/view?usp=sharing) 	|
| Haskell            	| Trained on https://github.com/jgm/pandoc      	                | [link](https://drive.google.com/file/d/16IAu3iMhuHIyiYsw6OwkCKtBy3qOVWd0/view?usp=sharing) 	|
| Java            	| Trained on https://github.com/TheAlgorithms/Java    	                | [link](https://drive.google.com/file/d/1-08erirNC1GbuRLcFzQIfjwbhB40_wiA/view?usp=sharing) 	|
| JavaScript           	| Trained on https://github.com/trekhleb/javascript-algorithms    	| [link](https://drive.google.com/file/d/1npW4YN7y2d4Id0WhXVnT_0--slmPEfW0/view?usp=sharing) 	|
| Kotlin           	| Trained on https://github.com/square/leakcanary    	                | [link](https://drive.google.com/file/d/1Ra5P4uQXOd87zimMqg8eyB6ONei9km6h/view?usp=sharing) 	|
| Lua           	| Trained on https://github.com/nmap/nmap       	                | [link](https://drive.google.com/file/d/1MusGCJfepW9Q1G7LqgnFL_7UEc8tpsVP/view?usp=sharing) 	|
| Perl           	| Trained on https://github.com/mojolicious/mojo       	                | [link](https://drive.google.com/file/d/14mKERFDa2SiH7vSweymk6f2d2ULSKuZ1/view?usp=sharing) 	|
| PHP           	| Trained on https://github.com/symfony/symfony    	                | [link](https://drive.google.com/file/d/17xl168mPjeL1w8-k195RDPavOxZWUY96/view?usp=sharing) 	|
| Python               	| Trained on https://github.com/TheAlgorithms/Python              	| [link](https://drive.google.com/file/d/14W-Ymi-h6jqNyqM5yGXyzwG25J3zzdn3/view?usp=sharing) 	|
| R               	| Trained on https://github.com/tidyverse/ggplot2              	        | [link](https://drive.google.com/file/d/1TgZojc2--ej4UC1ksAShDLUtDBWYf7xQ/view?usp=sharing) 	|
| Ruby               	| Trained on https://github.com/jekyll/jekyll              	        | [link](https://drive.google.com/file/d/1UMhlpI9a4Fpni1k1jQrePa8NOD8n6urq/view?usp=sharing) 	|
| Rust               	| Trained on https://github.com/rust-lang/rust/tree/master/compiler     | [link](https://drive.google.com/file/d/1CpzIs4EytZGfij7v-oltKGg4q8uepfVI/view?usp=sharing) 	|
| Scala               	| Trained on https://github.com/apache/spark/tree/master/mllib          | [link](https://drive.google.com/file/d/1ojYNJOIvWjPO3Nc9BCcY_NRrxyumAKeh/view?usp=sharing) 	|
| Scikit-learn         	| Trained on https://github.com/scikit-learn/scikit-learn         	| [link](https://drive.google.com/file/d/1Hl_DcXOSH8B6IxJ9fHBmoSkEOXFQ1q86/view?usp=sharing) 	|
| Swift         	| Trained on https://github.com/raywenderlich/swift-algorithm-club      | [link](https://drive.google.com/file/d/1f6TQQL7lvWRlq7t17B0qn-fN0CRZU71h/view?usp=sharing) 	|


Once you download the file, you will need to place it in `~/.mltype/languages`.
Note that if the folder does not exist you will have to create it. The file name
can be changed to whatever you like. This name will then be used to
refer to the model.

To verify that the model was downloaded succesfully, try to sample from it.
**Note that this might take 20+ seconds the first time around.**

```bash
mlt sample my_new_model
```

Feel free to create an issue if you want me to train a model for you. Note
that you can also do it yourself easily by reading the documentation (`mlt 
train`) and getting a GPU on Google Colab (click the badge below for a ready to
use notebook).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1UfcE0qaAtq2SqSZRXbFUF8yfGgQaWZ5K?usp=sharing)


# Credits
This project is very much motivated by the The Unreasonable Effectiveness of 
Recurrent Neural Networks by Andrej Karpathy.
