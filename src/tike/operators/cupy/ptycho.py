"""Defines a ptychography operator based on the CuPy FFT module."""

__author__ = "Daniel Ching, Viktor Nikitin"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."

import typing

import numpy.typing as npt
import numpy as np
import cupy as cp

from .operator import Operator
from .propagation import Propagation
from .multislice import Multislice, SingleSlice
from . import objective


@cp.fuse()
def _intensity_from_farplane(farplane):
    return cp.sum(
        cp.real(farplane * cp.conj(farplane)),
        axis=tuple(range(1, farplane.ndim - 2)),
    )


class Ptycho(Operator):
    """A Ptychography operator.

    Compose a diffraction and propagation operator to simulate the interaction
    of an illumination wavefront with an object followed by the propagation of
    the wavefront to a detector plane.


    Parameters
    ----------
    detector_shape : int
        The pixel width and height of the (square) detector grid.
    d, nz, n : int
        The pixel depth, width, and height of the reconstructed grid.
    probe_shape : int
        The pixel width and height of the (square) probe illumination.
    propagation : :py:class:`Operator`
        The wave propagation operator being used.
    diffraction : :py:class:`Operator`
        The object probe interaction operator being used.
    data : (FRAME, WIDE, HIGH) float32
        The intensity (square of the absolute value) of the propagated
        wavefront; i.e. what the detector records.
    farplane: (POSI, 1, SHARED, detector_shape, detector_shape) complex64
        The wavefronts hitting the detector respectively.
    probe : {(1, 1, SHARED, WIDE, HIGH), (POSI, 1, SHARED, WIDE, HIGH)} complex64
        The complex illumination function.
    psi : (DEPTH, WIDE, HIGH) complex64
        The wavefront modulation coefficients of the object.
    scan : (POSI, 2) float32
        Coordinates of the minimum corner of the probe grid for each
        measurement in the coordinate system of psi. Coordinate order
        consistent with WIDE, HIGH order.


    .. versionchanged:: 0.25.0 Removed the model and ntheta parameters.
    .. versionchanged:: 0.26.0 Added depth dimension to psi array

    """

    def __init__(
        self,
        detector_shape: int,
        probe_shape: int,
        nz: int,
        n: int,
        propagation: typing.Type[Propagation] = Propagation,
        diffraction: typing.Type[Multislice] = Multislice,
        norm: str = 'ortho',
        **kwargs,
    ):
        """Please see help(Ptycho) for more info."""
        self.propagation = propagation(
            detector_shape=detector_shape,
            norm=norm,
            **kwargs,
        )
        self.diffraction = diffraction(
            probe_shape=probe_shape,
            detector_shape=detector_shape,
            nz=nz,
            n=n,
            **kwargs,
        )
        # TODO: Replace these with @property functions
        self.probe_shape = probe_shape
        self.detector_shape = detector_shape
        self.nz = nz
        self.n = n

    def __enter__(self):
        self.propagation.__enter__()
        self.diffraction.__enter__()
        return self

    def __exit__(self, type, value, traceback):
        self.propagation.__exit__(type, value, traceback)
        self.diffraction.__exit__(type, value, traceback)

    def fwd(
        self,
        probe: npt.NDArray[np.csingle],
        scan: npt.NDArray[np.single],
        psi: npt.NDArray[np.csingle],
        **kwargs,
    ) -> npt.NDArray[np.csingle]:
        """Please see help(Ptycho) for more info."""
        return self.propagation.fwd(
            self.diffraction.fwd(
                psi=psi,
                scan=scan,
                probe=probe[..., 0, :, :, :],
            ),
            overwrite=True,
        )[..., None, :, :, :]

    def adj(
        self,
        farplane: npt.NDArray[np.csingle],
        probe: npt.NDArray[np.csingle],
        scan: npt.NDArray[np.single],
        psi: npt.NDArray[np.csingle],
        overwrite: bool = False,
        **kwargs,
    ) -> npt.NDArray[np.csingle]:
        """Please see help(Ptycho) for more info."""
        psi_adj, probe_adj = self.diffraction.adj(
            nearplane=self.propagation.adj(
                farplane,
                overwrite=overwrite,
            )[..., 0, :, :, :],
            probe=probe[..., 0, :, :, :],
            scan=scan,
            overwrite=True,
            psi=psi,
        )
        return psi_adj, probe_adj[..., None, :, :, :]

    def _compute_intensity(
        self,
        data: npt.NDArray,
        psi: npt.NDArray[np.csingle],
        scan: npt.NDArray[np.single],
        probe: npt.NDArray[np.csingle],
    ) -> npt.NDArray[np.single]:
        """Compute detector intensities replacing the nth probe mode"""
        farplane = self.fwd(
            psi=psi,
            scan=scan,
            probe=probe,
        )
        return _intensity_from_farplane(farplane), farplane

    def cost(
        self,
        data: npt.NDArray,
        psi: npt.NDArray[np.csingle],
        scan: npt.NDArray[np.single],
        probe: npt.NDArray[np.csingle],
        *,
        model: str,
    ) -> float:
        """Please see help(Ptycho) for more info."""
        intensity, _ = self._compute_intensity(data, psi, scan, probe)
        return getattr(objective, model)(data, intensity)
