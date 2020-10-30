Examples
========

Hello world
-----------
The :code:`raw` command allows to specify custom text.


.. image:: https://i.imgur.com/k0Msvk9.gif
    :align: center

Setting a target speed
----------------------
One can set a target speed via the flag :code:`-t, --target-wpm`.
This speed will be interactively shown with a new cursor. One
can use it with the following commands

- :code:`file`
- :code:`random`
- :code:`raw`
- :code:`replay`
- :code:`sample`

.. image:: https://i.imgur.com/ngu4sKT.gif
    :align: center

Competing against yourself
--------------------------
The command :code:`replay` can be used for playing against
a past performance. All of the below commands support
the flag :code:`-o, --output-file` that will store a single
performance to a file.

- :code:`file`
- :code:`random`
- :code:`raw`
- :code:`sample`

This file can then be provided as an argument to :code:`replay`.
The previous performance will be marked by an additional
cursor. 


.. image:: https://i.imgur.com/2yeNtc5.gif
    :align: center


If me manage to be faster than our previous performance,
the file is not going to be overwritten by default. However, one
can allow for overwriting via the :code:`-w, --overwrite`
flag. Note that the file will be only overwritten
if we improve. If we use this flag we essentially always
compete against our all time best and each new record will
lead to an update.

.. image:: https://i.imgur.com/4haqVCZ.gif
    :align: center


Selecting top k characters
--------------------------
The :code:`sample` command is using a neural network in the
background. The characters are generated one by one
based on some probability distribution over all
possible characters (vocabulary). One can use the
option :code:`-k, --top-k` to only sample from the
most k probable characters. In general, the lower
the k the more conservative the next character
predictions are.

.. image:: https://i.imgur.com/rDiKmvJ.gif
    :align: center

Provide initial text for sampling
---------------------------------
The :code:`sample` provides an option :code:`-s, --starting-text`
through which one can specify the starting text. This
way one can decide roughly on the topic of the text.

.. image:: https://i.imgur.com/uz1046g.gif
    :align: center

Independent generation of characters
------------------------------------
The :code:`random` command allows for generating of random
characters based on provided frequency distribution. Note
that as opposed to :code:`sample` the previous characters
are not taken into account when generating a new one.

.. image:: https://i.imgur.com/ILmyQ5w.gif
    :align: center
