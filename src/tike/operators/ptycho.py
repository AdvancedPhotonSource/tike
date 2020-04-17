"""Defines a pytchography operator based on the NumPy FFT module."""

import numpy as np

from .operator import Operator
from .propagation import Propagation
from .convolution import Convolution


class Ptycho(Operator):
    """A Ptychography operator.

    Compose a diffraction and propagation operator to simulate the interaction
    of an illumination wavefront with an object followed by the propagation of
    the wavefront to a detector plane.

    Attributes
    ----------
    nscan : int
        The number of scan positions at each angular view.
    fly : int
        The number of consecutive scan positions that describe a fly scan.
    nmode : int
        The number of probe modes per scan position.
    probe_shape : int
        The pixel width and height of the (square) probe illumination.
    detector_shape : int
        The pixel width and height of the (square) detector grid.
    nz, n : int
        The pixel width and height of the reconstructed grid.
    ntheta : int
        The number of angular partitions of the data.
    model : string
        The type of noise model to use for the cost functions.
    propagation : Operator
        The wave propagation operator being used.
    diffraction : Operator
        The object probe interaction operator being used.

    Parameters
    ----------
    psi : (ntheta, nz, n) complex64
        The complex wavefront modulation of the object.
    probe : (ntheta, nscan // fly, fly, nmode, probe_shape, probe_shape) complex64
        The complex illumination function.
    nearplane: (ntheta, nscan // fly, fly, nmode, probe_shape, probe_shape) complex64
        The wavefronts after exiting the object.
    farplane: (ntheta, nscan // fly, fly, nmode, detector_shape, detector_shape) complex64
        The wavefronts hitting the detector respectively.
    data : (ntheta, nscan, detector_shape, detector_shape) complex64
        data is the square of the absolute value of `farplane`. `data` is the
        intensity of the `farplane`.
    scan : (ntheta, nscan, 2) float32
        Coordinates of the minimum corner of the probe grid for each
        measurement in the coordinate system of psi. Vertical coordinates
        first, horizontal coordinates second.

    """

    def __init__(self, detector_shape, probe_shape, nscan, nz, n,
                 ntheta=1, model='gaussian', nmode=1, fly=1,
                 propagation=Propagation,
                 diffraction=Convolution,
                 **kwargs):  # noqa: D102 yapf: disable
        """Please see help(Ptycho) for more info."""
        self.propagation = propagation(ntheta * nscan * nmode, detector_shape,
                                       probe_shape, model=model, fly=fly,
                                       nmode=nmode,
                                       **kwargs)  # yapf: disable
        self.diffraction = diffraction(probe_shape, nscan, nz, n, ntheta,
                                       model=model, fly=fly, nmode=nmode,
                                       **kwargs)  # yapf: disable
        self.nscan = nscan
        self.probe_shape = probe_shape
        self.detector_shape = detector_shape
        self.nz = nz
        self.n = n
        self.ntheta = ntheta
        self.fly = fly
        self.nmode = nmode

    def __enter__(self):
        self.propagation.__enter__()
        self.diffraction.__enter__()
        return self

    def __exit__(self, type, value, traceback):
        self.propagation.__exit__(type, value, traceback)
        self.diffraction.__exit__(type, value, traceback)

    def fwd(self, probe, scan, psi, **kwargs):  # noqa: D102
        nearplane = self.diffraction.fwd(psi=psi, scan=scan, probe=probe)
        farplane = self.propagation.fwd(nearplane)
        return farplane

    def adj(self, farplane, probe, scan, **kwargs):  # noqa: D102
        nearplane = self.propagation.adj(farplane)
        return self.diffraction.adj(nearplane=nearplane, probe=probe, scan=scan)

    def adj_probe(self, farplane, scan, psi, **kwargs):  # noqa: D102
        nearplane = self.propagation.adj(farplane=farplane)
        return self.diffraction.adj_probe(psi=psi, scan=scan,
                                          nearplane=nearplane)  # yapf: disable

    def cost(self, data, psi, scan, probe):  # noqa: D102
        farplane = self.fwd(psi=psi, scan=scan, probe=probe)
        return self.propagation.cost(data, farplane)

    def grad(self, data, psi, scan, probe):  # noqa: D102
        farplane = self.fwd(psi=psi, scan=scan, probe=probe)
        data_diff = self.propagation.grad(data, farplane)
        return self.adj(farplane=data_diff, probe=probe, scan=scan)

    def grad_probe(self, data, psi, scan, probe):  # noqa: D102
        farplane = self.fwd(psi=psi, scan=scan, probe=probe)
        data_diff = self.propagation.grad(data, farplane)
        return self.adj_probe(farplane=data_diff, psi=psi, scan=scan)
