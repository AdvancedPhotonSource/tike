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
        The complex (..., nscan, 1, nprobe, probe_shape, probe_shape) or
        (..., 1, 1, nprobe, probe_shape, probe_shape) illumination
        function.
    nearplane, farplane: complex64
        The (..., nscan, 1, nprobe, detector_shape, detector_shape)
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
                probe=probe[..., 0, :, :, :],
            ),
            overwrite=True,
        )[..., None, :, :, :]

    def adj(self, farplane, probe, scan, psi=None, overwrite=False, **kwargs):
        return self.diffraction.adj(
            nearplane=self.propagation.adj(
                farplane,
                overwrite=overwrite,
            )[..., 0, :, :, :],
            probe=probe[..., 0, :, :, :],
            scan=scan,
            overwrite=True,
            psi=psi,
        )

    def adj_probe(self, farplane, scan, psi, overwrite=False, **kwargs):
        return self.diffraction.adj_probe(
            psi=psi,
            scan=scan,
            nearplane=self.propagation.adj(
                farplane=farplane,
                overwrite=overwrite,
            )[..., 0, :, :, :],
            overwrite=True,
        )[..., None, :, :, :]

    def _compute_intensity(self, data, psi, scan, probe, n=-1, mode=None):
        """Compute detector intensities replacing the nth probe mode"""
        farplane = self.fwd(
            psi=psi,
            scan=scan,
            probe=probe,
        )
        return self.xp.sum(
            (farplane * farplane.conj()).real,
            axis=(2, 3),
        ), farplane

    def cost(self, data, psi, scan, probe, n=-1, mode=None) -> float:
        intensity, _ = self._compute_intensity(data, psi, scan, probe)
        return self.propagation.cost(data, intensity)

    def grad_psi(self, data, psi, scan, probe):
        intensity, farplane = self._compute_intensity(data, psi, scan, probe)
        grad_obj = self.xp.zeros_like(psi)
        grad_obj = self.adj(
            farplane=self.propagation.grad(
                data,
                farplane,
                intensity,
            ),
            probe=probe,
            scan=scan,
            psi=grad_obj,
            overwrite=True,
        )
        return grad_obj

    def grad_probe(self, data, psi, scan, probe, n=-1, mode=None):
        intensity, farplane = self._compute_intensity(data, psi, scan, probe)
        # Use the average gradient for all probe positions
        return self.xp.mean(
            self.adj_probe(
                farplane=self.propagation.grad(
                    data,
                    farplane,
                    intensity,
                ),
                psi=psi,
                scan=scan,
                overwrite=True,
            ),
            axis=1,
            keepdims=True,
        )
