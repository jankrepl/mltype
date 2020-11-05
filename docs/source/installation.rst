Installation
============

Pip
---

The simplest way to install :code:`mltype` is via PyPI

.. code-block:: bash

    pip install mltype

To get the latest version or potentially help with developlment,
clone the github repository

.. code-block:: bash

    git clone https://github.com/jankrepl/mltype.git
    cd mltype
    pip install -e .

Docker
------

Alternatively, you can also use Docker Compose. Run the following command

.. code-block:: bash

    docker-compose run --rm mltype

This command builds the :code:`mltype_img` image and then runs it while mounting
your home directory inside of the container.

One does not have to use Docker Compose and instead reacreate the above with the
following commands

.. code-block:: bash

   docker build -t mltype_img .
   docker run --rm -it -v $HOME:/root mltype_img

Note that if you want to have access to GPUs (can be used for training) you
need to add the :code:`--gpus all` flag to :code:`docker run`.

Lastly, we recommend adding a custom user to prevent having
root priveleges inside of the container.


Extra dependencies
------------------
One can use the following sytax to install extra dependencies

.. code-block:: bash

    pip install -e .[GROUP]

Below are the available groups with

* :code:`dev` - development tools
* :code:`mlflow` - optional tracking tool to visualize training progress

Note that for some tests (optional) we use :code:`hecate`. To install it run

.. code-block:: bash

    pip install hecate@git+http://github.com/DRMacIver/hecate#25f3260
