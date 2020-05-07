#########################
Installation Instructions
#########################

****************************
From the conda-forge channel
****************************

At this time, `tike` can only be installed from source.

********************
From the source code
********************

The build and runtime requirements are listed in `/meta.yaml`, the conda-build
recipe.

Install the package using typical installation methods.

.. code-block:: bash

  pip install .

The `-e` option for `pip install` makes the installation editable; this means
whenever you import `tike`, any changes that you make to the source code will be
included.

In order to use the CUDA accelerated versions of the solvers, install the
`CuPy <https://cupy.chainer.org/>`_ package. Then, set the
following environment variable using `export` or `set`.

.. code-block:: bash

  export TIKE_BACKEND=cupy
