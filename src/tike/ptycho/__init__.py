"""Provides ptychography solvers.

The reference implementation uses NumPy's FFT library. Select a non-default
backend by setting the TIKE_PTYCHO_BACKEND environment variable.

Coordinate Systems
------------------

`v, h` are the horizontal and vertical directions perpendicular
to the probe direction where positive directions are to the right and up.

Functions
---------

Each function in this module should have the following interface:

Parameters
----------
data :  (T, P,    V, H) :py:class:`numpy.array` float32
    An array of detector intensities for each of the `P` positions at `T`
    viewing angles. The grid of each detector is `H` pixels wide
    (the horizontal direction) and `V` pixels tall (the vertical direction).
probe : (T, P, M, V, H) :py:class:`numpy.array` complex64
    The illuminations of the probes.
psi :   (T,       V, H) :py:class:`numpy.array` complex64
    The object transmission function.
scan :  (T, P,       2) :py:class:`numpy.array` float32
    The scanning positions with vertical coordinate listed before horizontal
    coordinates.
kwargs : :py:class:`dict`
    Keyword arguments specific to this function. `**kwargs` should always be
    included so that extra parameters are ignored instead of raising an error.

"""
import os

# Search available entry points for requested backend. Must set the
# PtychoBackend variable BEFORE importing the rest of the module.
if "TIKE_PTYCHO_BACKEND" in os.environ:
    import pkg_resources
    _ptycho_backend_options = {}
    for _entry_point in pkg_resources.iter_entry_points('tike.PtychoBackend'):
        _ptycho_backend_options[_entry_point.name] = _entry_point.load()
    _requested_backend = os.environ["TIKE_PTYCHO_BACKEND"]
    if _requested_backend in _ptycho_backend_options:
        PtychoBackend = _ptycho_backend_options[_requested_backend]
    else:
        raise ImportError(
            "Cannot set PtychoBackend to '{}'. "
            "Available options are: {}".format(_requested_backend,
                                               _ptycho_backend_options)
        )
else:
    from tike.operators import Ptycho as PtychoBackend

from .ptycho import *  # noqa
from .solvers import *  # noqa
