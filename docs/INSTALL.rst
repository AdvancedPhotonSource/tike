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

Install the package using typical installation methods.

.. code-block:: bash

  $ pip install .

The `-e` option for `pip install` makes the installation editable; this means
whenever you import `tike`, any changes that you make to the source code will be
included.

In order to use the CUDA accelerated versions of the solvers. First, install the
`libtike-cufft <https://github.com/carterbox/ptychocg>`_ package. Then, set the
environment variables using `export` or `set`.

.. code-block:: bash

  export TIKE_PTYCHO_BACKEND=cudafft`
  export TIKE_TOMO_BACKEND=cudafft`
