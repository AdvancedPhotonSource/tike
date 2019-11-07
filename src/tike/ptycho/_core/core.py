"""This module defines template classes for the solvers.

All of the solvers, tomography and ptychography, rely on core operators
including forward and adjunct operators. These core operators may have multiple
implementations based on different backends e.g. CUDA, OpenCL, NumPy. The
classes in this module prescribe an interface upon which specific solvers are
based. In this way, multiple solvers (e.g. E-Pi, gradient descent) implemented
in Python can share the same core operators and can be upgraded to better
operators in the future.

"""


class PtychoCore(object):
    """A base class for ptychography solvers.

    This class is a context manager which provides the basic operators required
    to implement a ptychography solver. Specific implementations of this class
    can either inherit from this class or just provide the same interface.

    Solver implementations should inherit from PtychoBacked which is an alias
    for whichever PtychoCore implementation is selected at import time.

    Attributes
    ----------
    nscan : int
        The number of scan positions at each angular view. (Assumed to be the
        same for all views.)
    probe_shape : int
        The pixel width and height of the (square) probe illumination.
    detector_shape : int
        The pixel width and height of the (square) detector grid.
    nz, n : int
        The pixel width and height of the reconstructed grid.
    ntheta : int
        The number of angular partitions of the data.
    ptheta : int
        The number of angular partitions to process together
        simultaneously.

    Parameters
    ----------
    psi : (ntheta, nz, n) complex64
        The complex wavefront modulation of the object.
    probe : (ntheta, probe_shape, probe_shape) complex64
        The complex illumination function.
    data, farplane : (ntheta, nscan, detector_shape, detector_shape) complex64
        data is the square of the absolute value of `farplane`. `data` is the
        intensity of the `farplane`.
    scan : (ntheta, nscan, 2) float32
        Coordinates of the minimum corner of the probe grid for each
        measurement in the coordinate system of psi. Vertical coordinates
        first, horizontal coordinates second.

    """

    array_module = None

    def __init__(self, detector_shape, probe_shape, nscan, nz, n, ntheta=1):
        """Please see help(PtychoCore) for more info."""
        self.nscan = nscan
        self.probe_shape = probe_shape
        self.detector_shape = detector_shape
        self.nz = nz
        self.n = n
        self.ntheta = ntheta

    def __enter__(self):
        """Return self at start of a with-block."""
        return self

    def __exit__(self, type, value, traceback):
        """Gracefully handle interruptions or with-block exit.

        Do things like deallocating GPU memory.
        """
        pass

    def run(self, data, probe, scan, psi, **kwargs):
        """Implement a specific ptychography solving algorithm.

        See help(PtychoCore) for more information.
        """
        raise NotImplementedError("Cannot run a base class.")

    def fwd(self, farplane, probe, scan, psi, **kwargs):
        """Perform the forward ptychography transform (FQ).

        See help(PtychoCore) for more information.
        """
        raise NotImplementedError("Cannot run a base class.")

    def adj(self, farplane, probe, scan, psi, **kwargs):
        """Perform the fixed probe adjoint ptychography transform (Q*F*).

        See help(PtychoCore) for more information.
        """
        raise NotImplementedError("Cannot run a base class.")

    def adj_probe(self, farplane, probe, scan, psi, **kwargs):
        """Perform the fixed object adjoint ptychography transform (O*F*).

        See help(PtychoCore) for more information.
        """
        raise NotImplementedError("Cannot run a base class.")
