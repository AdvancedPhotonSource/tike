"""Defines a tomography operator based on the NumPy FFT module."""

from .operator import Operator


class Tomo(Operator):
    """A base class for tomography solvers.

    This class is a context manager which provides the basic operators required
    to implement a tomography solver. Specific implementations of this class
    can either inherit from this class or just provide the same interface.

    Solver implementations should inherit from TomoBackend which is an alias
    for whichever TomoCore implementation is selected at import time.

    Attributes
    ----------
    ntheta : int
        The number of projections.
    n, nz : int
        The pixel width and height of the projection.

    Parameters
    ----------
    obj : (nz, n, n) complex64
        The complex object to be transformed or recovered.
    tomo : (ntheta, nz, n) complex64
        The radon transform of `obj`.
    angles : (ntheta, ) float32
        The radian angles at which the radon transform is sampled.
    centers : (nz, ) float32
        The center of rotation in `obj` pixels for each slice along z.

    """

    array_module = None
    asnumpy = None

    def __init__(self, angles, ntheta, nz, n, centers):
        """Please see help(TomoCore) for more info."""

    def __enter__(self):
        """Return self at start of a with-block."""
        return self

    def __exit__(self, type, value, traceback):
        """Free GPU memory due at interruptions or with-block exit."""
        pass

    def run(self, tomo, obj, **kwargs):
        """Implement a specific tomography solving algorithm.

        See help(TomoCore) for more information.
        """
        raise NotImplementedError("Cannot run a base class.")

    def fwd(self, obj, **kwargs):
        """Perform the forward Radon transform (R).

        See help(TomoCore) for more information.
        """
        raise NotImplementedError("Cannot run a base class.")

    def adj(self, tomo, **kwargs):
        """Perform the adjoint Radon transform (R*).

        See help(TomoCore) for more information.
        """
        raise NotImplementedError("Cannot run a base class.")
