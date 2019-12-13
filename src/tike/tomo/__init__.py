"""This module provides tomography solvers.

The tomography solvers are generic and can use any backend that provides an
interface that matches `TomoCore`.

The reference implementation uses NumPy's FFT library. Select a non-default
backend by setting the TIKE_TOMO_BACKEND environment variable.
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
    from tike.tomo._core.numpy import TomoNumPyFFT as TomoBackend

from tike.tomo.tomo import *  # noqa
from tike.tomo.solvers import *  # noqa
