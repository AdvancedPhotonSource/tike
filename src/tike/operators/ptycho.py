"""Defines a pytchography operator based on the NumPy FFT module."""

import numpy as np

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

    def __init__(self, detector_shape, probe_shape, nscan, nz, n,
                 ntheta=1, model='gaussian', nmode=1, fly=1,
                 propagation=Propagation,
                 diffraction=Convolution,
                 **kwargs):  # noqa: D102 yapf: disable
        """Please see help(Ptycho) for more info."""
        self.propagation = propagation(
            nwaves=ntheta * nscan * nmode,
            probe_shape=probe_shape,
            detector_shape=detector_shape,
            model=model,
            fly=fly,
            nmode=nmode,
            **kwargs,
        )
        self.diffraction = diffraction(
            probe_shape=probe_shape,
            detector_shape=detector_shape,
            nscan=nscan,
            nz=nz,
            n=n,
            ntheta=ntheta,
            model=model,
            fly=fly,
            nmode=nmode,
            **kwargs,
        )
        # TODO: Replace these with @property functions
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

    def fwd(self, probe, scan, psi, **kwargs):
        nearplane = self.diffraction.fwd(psi=psi, scan=scan, probe=probe)
        farplane = self.propagation.fwd(nearplane, overwrite=True)
        return farplane

    def adj(self, farplane, probe, scan, overwrite=False, **kwargs):
        nearplane = self.propagation.adj(farplane, overwrite=overwrite)
        return self.diffraction.adj(nearplane=nearplane,
                                    probe=probe,
                                    scan=scan,
                                    overwrite=True)

    def adj_probe(self, farplane, scan, psi, overwrite=False, **kwargs):
        nearplane = self.propagation.adj(farplane=farplane, overwrite=overwrite)
        return self.diffraction.adj_probe(psi=psi,
                                          scan=scan,
                                          nearplane=nearplane,
                                          overwrite=True)

    def _compute_intensity(self, data, psi, scan, probe):
        intensity = 0
        for mode in np.split(probe, probe.shape[-3], axis=-3):
            farplane = self.fwd(psi=psi, scan=scan, probe=mode)
            intensity += np.sum(np.square(np.abs(
                farplane.reshape(*data.shape[:2], -1, *data.shape[2:]))),
                axis=2,
            )  # yapf: disable
        return intensity

    def cost(self, data, psi, scan, probe):
        intensity = self._compute_intensity(data, psi, scan, probe)
        return self.propagation.cost(data, intensity)

    def grad(self, data, psi, scan, probe):
        intensity = self._compute_intensity(data, psi, scan, probe)
        grad_obj = self.xp.zeros_like(psi)
        for mode in np.split(probe, probe.shape[-3], axis=-3):
            # TODO: Pass obj through adj() instead of making new obj inside
            grad_obj += self.adj(
                farplane=self.propagation.grad(
                    data,
                    self.fwd(psi=psi, scan=scan, probe=mode),
                    intensity,
                ),
                probe=mode,
                scan=scan,
                overwrite=True,
            )
        return grad_obj

    def grad_probe(self, data, psi, scan, probe, m=0):
        intensity = self._compute_intensity(data, psi, scan, probe)
        return self.adj_probe(
            farplane=self.propagation.grad(
                data,
                self.fwd(psi=psi, scan=scan, probe=probe[..., m:m+1, :, :]),
                intensity,
            ),
            psi=psi,
            scan=scan,
            overwrite=True,
        )
