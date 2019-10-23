"""This module provides ptychography solvers.

The ptychography solvers are generic and can use any backend that provides an
interface that matches `tike.templates.PtychoCore`.

The reference implementation uses NumPy's FFT library. Select a non-default
backend by setting the TIKE_PTYCHO_BACKEND environment variable.
"""
import os

# Search available entry points for requested backend. Must set the
# PtychoBackend variable BEFORE importing the rest of the module.
if "TIKE_PTYCHO_BACKEND" in os.environ:
    # something = os.environ["TIKE_PTYCHO_BACKEND"]
    # from something import something as PtychoBackend
    raise ImportError("Cannot set custom backend yet.")
else:
    from tike.ptycho.numpy import PtychoNumPyFFT as PtychoBackend

from tike.ptycho.ptycho import *
