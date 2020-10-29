<p align="center">
  <img src="https://user-images.githubusercontent.com/18519371/97606153-c19c2700-1a0f-11eb-9faf-876f266b4585.png">
</p>

# Description
Command line tool for improving typing speed and accuracy. The main goal is
to help programmers practise programming languages.

# Demo
<p align="center">
  <img src="https://i.imgur.com/Gdmctcl.gif">
</p>

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

# Documentation
For more detailed information see https://mltype.readthedocs.io. 

# Pretrained models

|         Name         	|                               Info                              	| Link                                                                                       	|
|:--------------------:	|:---------------------------------------------------------------:	|--------------------------------------------------------------------------------------------	|
| C++                  	| Trained on https://github.com/TheAlgorithms/C-Plus-Plus         	| [link](https://drive.google.com/file/d/1ea49gaUWJea_-nnT4aI2TpwfG5OLQdlw/view?usp=sharing) 	|
| CPython              	| Trained on https://github.com/python/cpython/tree/master/Python 	| [link](https://drive.google.com/file/d/1aKnOkxcSYdpXYCB6yMOpbGJIw2ribVEq/view?usp=sharing) 	|
| Crime and Punishment 	| Trained on http://www.gutenberg.org/ebooks/2554                 	| [link](https://drive.google.com/file/d/1-KgO-9X3z-Xr2SLAgAI_Ijllw7L9MFpN/view?usp=sharing) 	|
| Dracula              	| Trained on http://www.gutenberg.org/ebooks/345                  	| [link](https://drive.google.com/file/d/1Fx2cZ4gOaioJymsUCY_Q620Yk53bZQeK/view?usp=sharing) 	|
| JavaScript           	| Trained on https://github.com/trekhleb/javascript-algorithms    	| [link](https://drive.google.com/file/d/1npW4YN7y2d4Id0WhXVnT_0--slmPEfW0/view?usp=sharing) 	|
| Python               	| Trained on https://github.com/TheAlgorithms/Python              	| [link](https://drive.google.com/file/d/14W-Ymi-h6jqNyqM5yGXyzwG25J3zzdn3/view?usp=sharing) 	|
| Scikit-learn         	| Trained on https://github.com/scikit-learn/scikit-learn         	| [link](https://drive.google.com/file/d/1Hl_DcXOSH8B6IxJ9fHBmoSkEOXFQ1q86/view?usp=sharing) 	|


Once you download the file, you will need to place it in `~/.mltype/langagues`. 
Note that if the folder does not exist you will have to create it. The file name
can be changed to whatevery you like. This name will then be used to
refer to the model.

To verify that the model was downloaded succesfully, try to sample from it.

```bash
mlt sample my_new_model
```
