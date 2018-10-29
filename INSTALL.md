# Installation Instructions

At this time, `tike` can only be installed from source.

## Installation from the source code

`tike` installation has two steps: building the c extension shared library and installing the python package.

First, build the shared library by using the build script:

```
$ python build.py
```

This will compile a shared library, `libtike`(`.so` / `.dylib` / `.dll`) (Linux, MacOS, Windows) and install it to `tike/sharedlibs`.

For Windows, both MinGW-64 and `make` need to be installed and placed in the PATH, so that the appropriate `gcc.exe` (that is, one that supports C99) and `make.exe` can be found. For Anaconda Python on Windows, adding the `conda` packages `MinGW` and `make` provides these resources.

Next, install the python package to your current environment by using `pip`:

```
$ pip install -e .
```

We recommend `pip` for installation because it includes metadata with the installation that makes it easy to uninstall or update the package later. Calling `setup.py` directly does not create this metadata. The `-e` option for `pip install` makes the installation editable; this means whenever you import `tike`, any changes that you make to the source code will be included.
