Installation
============

The simplest way to install :code:`mltype` is via PyPI

.. code-block:: bash

    pip install mltype

To get the latest version or potentially help with developlment,
clone the github repository

.. code-block:: bash

    git clone https://github.com/jankrepl/mltype.git
    cd mltype
    pip install -e .

Extra dependencies
------------------
One can use the following sytax to install extra dependencies

.. code-block:: bash

    pip install -e .[GROUP]

Below are the available groups with

* :code:`dev` - development tools
* :code:`hecate` - tools for running optional curses tests
* :code:`mlflow` - optional tracking tool to visualize training progress
