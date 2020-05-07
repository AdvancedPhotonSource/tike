"""Provides tomography solvers.

The reference implementation uses NumPy's FFT library. Select a non-default
backend by setting the TIKE_TOMO_BACKEND environment variable.

Coordinate Systems
------------------

`theta, v, h`. `v, h` are the horizontal vertical directions perpendicular
to the probe direction where positive directions are to the right and up.
`theta` is the rotation angle around the vertical reconstruction
space axis, `z`. `z` is parallel to `v`, and uses the right hand rule to
determine reconstruction space coordinates `z, x, y`. `theta` is measured
from the `x` axis, so when `theta = 0`, `h` is parallel to `y`.

Functions
---------

Each public function in this module should have the following interface:

Parameters
----------
obj : (Z, X, Y, P) :py:class:`numpy.array` float32
    An array of material properties. The first three dimensions `Z, X, Y`
    are spatial dimensions. The fourth dimension, `P`,  holds properties at
    each grid position: refractive indices, attenuation coefficents, etc.
integrals : (M, V, H, P) :py:class:`numpy.array` float32
    Integrals across the `obj` for each of the `probe` rays and
    P parameters.
theta, v, h : (M, ) :py:class:`numpy.array` float32
    The min corner (theta, v, h) of the `probe` for each measurement.
kwargs
    Keyword arguments specific to this function. `**kwargs` should always be
    included so that extra parameters are ignored instead of raising an error.

"""
import os

# Search available entry points for requested backend. Must set the
# TomoBackend variable BEFORE importing the rest of the module.
if "TIKE_TOMO_BACKEND" in os.environ:
    import pkg_resources
    _backend_options = {}
    for _entry_point in pkg_resources.iter_entry_points('tike.TomoBackend'):
        _backend_options[_entry_point.name] = _entry_point.load()
    _requested_backend = os.environ["TIKE_TOMO_BACKEND"]
    if _requested_backend in _backend_options:
        TomoBackend = _backend_options[_requested_backend]
    else:
        raise ImportError(
            "Cannot set TomoBackend to '{}'. "
            "Available options are: {}".format(_requested_backend,
                                               _backend_options)
        )
else:
    from tike.operators import Tomo as TomoBackend

from tike.tomo.tomo import *  # noqa
from tike.tomo.solvers import *  # noqa
