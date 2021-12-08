#########################
Installation Instructions
#########################

Tike is build on the `CuPy <https://github.com/cupy/cupy/>`_ framework which uses
NVidia CUDA to accelerate computation. Thus, a CUDA compatible GPU on Windows_x64 or
Linux_x64 is required. Other platforms are not supported at this time.

****************************************
From the conda-forge channel using Conda
****************************************

In theory, `tike` is available via conda from the conda-forge channel.
However, this distribution will only be updated sporadically and is not
recommended at this time.

********************
From the source code
********************

The build and runtime requirements are listed in `requirements.txt`. Install these
packages before installing tike; pip will not install them automatically.

Install the package using typical installation methods: navigate to the
directory with `setup.cfg` and ask `pip` to install tike.

.. code-block:: bash

  pip install .

The `-e` option for `pip install` makes the installation editable; this means
whenever you import `tike`, any changes that you make to the source code will be
included.
