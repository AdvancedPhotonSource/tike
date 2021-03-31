#########################
Installation Instructions
#########################

****************************************
From the conda-forge channel using Conda
****************************************

In theory, `tike` is available via conda from the conda-forge channel.
However, this distribution will only be updated sporadically and is not
recommended.

********************
From the source code
********************

The build and runtime requirements are listed in `requirements.txt`.

Install the package using typical installation methods: navigate to the
directory with `setup.py` and ask `pip` to install tike.

.. code-block:: bash

  pip install .

The `-e` option for `pip install` makes the installation editable; this means
whenever you import `tike`, any changes that you make to the source code will be
included.
