#########################
Installation Instructions
#########################

Tike is build on the `CuPy <https://github.com/cupy/cupy/>`_ framework which uses
NVidia CUDA to accelerate computation. Thus, a CUDA compatible GPU on Windows_x64 or
Linux_x64 is required. Other platforms are not supported at this time.

******************************************************
From the conda-forge channel using Conda (recommended)
******************************************************

Tike is available via conda from the conda-forge channel. This distribution is
updated whenever there is a tagged release. This project is still below
version 1.0, so expect API breaking changes at every MINOR version.

*************************************
From the source code (for developers)
*************************************

The build and runtime requirements are listed together in `requirements.txt`.
Install these packages before installing tike using conda.

Install the package using typical installation methods: navigate to the
directory with `setup.cfg` and ask `pip` to install tike.

.. code-block:: bash

  pip install . --no-deps

The `-e` option for `pip install` makes the installation editable; this means
whenever you import `tike`, any changes that you make to the source code will be
included.
