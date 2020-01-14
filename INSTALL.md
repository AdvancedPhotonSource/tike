# Installation Instructions

At this time, `tike` can only be installed from source.

## Installation from the source code

Install the package using typical installation methods.

```
$ pip install .
```

The `-e` option for `pip install` makes the installation editable; this means
whenever you import `tike`, any changes that you make to the source code will be
included.

In order to use the CuPy accelerated versions of the solvers. First, install the
[`libtike-cufft`](https://github.com/carterbox/ptychocg) package. Then, set the
environment variables `TIKE_PTYCHO_BACKEND=cudafft` and
`TIKE_TOMO_BACKEND=cudafft`.
