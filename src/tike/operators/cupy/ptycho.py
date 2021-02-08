"""Defines a ptychography operator based on the CuPy FFT module."""

__author__ = "Daniel Ching, Viktor Nikitin"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."

import concurrent.futures as cf

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
    psi : (..., nz, n) complex64
        The complex wavefront modulation of the object.
    probe : complex64
        The complex (..., nscan, 1, 1, probe_shape,
        probe_shape) illumination function.
    mode : complex64
        A single (..., nscan, 1, 1, probe_shape, probe_shape)
        probe mode.
    nearplane, farplane: complex64
        The (..., nscan, 1, 1, detector_shape, detector_shape)
        wavefronts exiting the object and hitting the detector respectively.
    data, intensity : float32
        The (..., nframe, detector_shape, detector_shape)
        square of the absolute value of `farplane` summed over `fly` and
        `modes`.
    scan : (..., nscan, 2) float32
        Coordinates of the minimum corner of the probe grid for each
        measurement in the coordinate system of psi. Vertical coordinates
        first, horizontal coordinates second.

    """

    def __init__(self, detector_shape, probe_shape, nz, n,
                 ntheta=1, model='gaussian',
                 propagation=Propagation,
                 diffraction=Convolution,
                 **kwargs):  # noqa: D102 yapf: disable
        """Please see help(Ptycho) for more info."""
        self.propagation = propagation(
            detector_shape=detector_shape,
            model=model,
            **kwargs,
        )
        self.diffraction = diffraction(
            probe_shape=probe_shape,
            detector_shape=detector_shape,
            nz=nz,
            n=n,
            ntheta=ntheta,
            model=model,
            **kwargs,
        )
        # TODO: Replace these with @property functions
        self.probe_shape = probe_shape
        self.detector_shape = detector_shape
        self.nz = nz
        self.n = n
        self.ntheta = ntheta

    def __enter__(self):
        self.propagation.__enter__()
        self.diffraction.__enter__()
        return self

    def __exit__(self, type, value, traceback):
        self.propagation.__exit__(type, value, traceback)
        self.diffraction.__exit__(type, value, traceback)

    def fwd(self, probe, scan, psi, **kwargs):
        return self.propagation.fwd(
            self.diffraction.fwd(
                psi=psi,
                scan=scan,
                probe=probe[..., 0, 0, :, :],
            ),
            overwrite=True,
        )[..., None, None, :, :]

    def adj(self, farplane, probe, scan, overwrite=False, **kwargs):
        return self.diffraction.adj(
            nearplane=self.propagation.adj(
                farplane,
                overwrite=overwrite,
            )[..., 0, 0, :, :],
            probe=probe[..., 0, 0, :, :],
            scan=scan,
            overwrite=True,
        )

    def adj_probe(self, farplane, scan, psi, overwrite=False, **kwargs):
        return self.diffraction.adj_probe(
            psi=psi,
            scan=scan,
            nearplane=self.propagation.adj(
                farplane=farplane,
                overwrite=overwrite,
            )[..., 0, 0, :, :],
            overwrite=True,
        )[..., None, None, :, :]

    def _compute_intensity(self, data, psi, scan, probe, n=-1, mode=None):
        """Compute detector intensities replacing the nth probe mode"""
        intensity = 0
        for m in range(probe.shape[-3]):
            intensity += np.sum(
                np.square(np.abs(self.fwd(
                    psi=psi,
                    scan=scan,
                    probe=mode if m == n else probe[..., m:m + 1, :, :],
                ).reshape(*data.shape[:2], -1, *data.shape[2:]))),
                axis=2,
            )  # yapf: disable
        return intensity

    def cost(self, data, psi, scan, probe, n=-1, mode=None) -> float:
        intensity = self._compute_intensity(data, psi, scan, probe, n, mode)
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
            ) / probe.shape[-3]
        return grad_obj

    def grad_probe(self, data, psi, scan, probe, n=-1, mode=None):
        intensity = self._compute_intensity(data, psi, scan, probe, n, mode)
        return self.adj_probe(
            farplane=self.propagation.grad(
                data,
                self.fwd(
                    psi=psi,
                    scan=scan,
                    probe=mode if mode is not None else probe,
                ),
                intensity,
            ),
            psi=psi,
            scan=scan,
            overwrite=True,
        )
