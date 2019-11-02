"""This module provides ptychography solvers.

The ptychography solvers are generic and can use any backend that provides an
interface that matches `PtychoCore`.

The reference implementation uses NumPy's FFT library. Select a non-default
backend by setting the TIKE_PTYCHO_BACKEND environment variable.
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
    from tike.ptycho._core.numpy import PtychoNumPyFFT as PtychoBackend

from tike.ptycho.ptycho import *
from tike.ptycho.solvers import *
