"""This module defines a pytchography operator based on the NumPy FFT module."""

from .operator import Operator
from .propagation import Propagation
from .convolution import Convolution

class Ptycho(Operator):
    """A base class for ptychography solvers.

    This class is a context manager which provides the basic operators required
    to implement a ptychography solver. Specific implementations of this class
    can either inherit from this class or just provide the same interface.

    Solver implementations should inherit from PtychoBacked which is an alias
    for whichever Ptycho implementation is selected at import time.

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

    def __init__(self, detector_shape, probe_shape, nscan, nz, n, ntheta=1,
                 **kwargs):  # noqa: D102
        """Please see help(Ptycho) for more info."""
        super(Ptycho, self).__init__(**kwargs)
        self.propagation = Propagation(detector_shape, probe_shape, **kwargs)
        self.diffraction = Convolution(detector_shape, probe_shape, nscan, nz, n, ntheta=1,
                     **kwargs)
        self.nscan = nscan
        self.probe_shape = probe_shape
        self.detector_shape = detector_shape
        self.nz = nz
        self.n = n
        self.ntheta = ntheta

    def fwd(self, probe, scan, psi, **kwargs):  # noqa: D102
        probe = probe.reshape(
            (self.ntheta, -1, self.probe_shape, self.probe_shape))
        nearplane = self.diffraction.fwd(psi=psi, scan=scan) * probe
        farplane = self.propagation.fwd(nearplane)
        assert farplane.shape == (self.ntheta, self.nscan, self.detector_shape,
                                  self.detector_shape)
        return farplane

    def adj(self, farplane, probe, scan, **kwargs):  # noqa: D102
        xp = self.array_module
        probe = probe.reshape(
            (self.ntheta, -1, self.probe_shape, self.probe_shape))
        nearplane = self.propagation.adj(farplane)
        psi = self.diffraction.adj(nearplane=nearplane * xp.conj(probe),
                                   scan=scan)
        assert psi.shape == (self.ntheta, self.nz, self.n)
        return psi

    def adj_probe(self, farplane, scan, psi, **kwargs):  # noqa: D102
        xp = self.array_module
        psi_patches = self.diffraction.fwd(psi=psi, scan=scan)
        nearplane = self.propagation.adj(farplane=farplane)
        probe = xp.sum(nearplane * xp.conj(psi_patches), axis=1)
        assert probe.shape == (self.ntheta, self.probe_shape, self.probe_shape)
        return probe
