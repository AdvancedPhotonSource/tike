"""Functions for ptychography."""
from .fresnel import *
from .object import ObjectOptions
from .position import check_allowed_positions, PositionOptions
from .probe import ProbeOptions
from .ptycho import *
from .solvers import *

# NOTE: The docstring below holds reference docstring that can be used to fill
# in documentation of new functions.
"""
Parameters
----------
data : (..., FRAME, WIDE, HIGH) float32
    The intensity (square of the absolute value) of the propagated wavefront;
    i.e. what the detector records.
comm : :py:class:`tike.communicators.Comm`
    An object which manages communications between both GPUs and nodes.
eigen_probe : (..., 1, EIGEN, SHARED, WIDE, HIGH) complex64
    The eigen probes for all positions.
eigen_weights : (..., POSI, EIGEN, SHARED) float32
    The relative intensity of the eigen probes at each position.
op : :py:class:`tike.operators.Ptycho`
    A ptychography operator. Provides forward and adjoint operations.
psi : (..., WIDE, HIGH) complex64
    The wavefront modulation coefficients of the object.
probe : (..., 1, 1, SHARED, WIDE, HIGH) complex64
    The shared complex illumination function amongst all positions.
scan : (..., POSI, 2) float32
    Coordinates of the minimum corner of the probe grid for each
    measurement in the coordinate system of psi. Coordinate order consistent
    with WIDE, HIGH order.

"""
