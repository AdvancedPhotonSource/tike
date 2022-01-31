"""Defines a ptychography operator based on the CuPy FFT module."""

__author__ = "Daniel Ching, Viktor Nikitin"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."

from .operator import Operator
from .propagation import Propagation
from .convolution import Convolution


class Ptycho(Operator):
    """A Ptychography operator.

    Compose a diffraction and propagation operator to simulate the interaction
    of an illumination wavefront with an object followed by the propagation of
    the wavefront to a detector plane.


    Parameters
    ----------
    detector_shape : int
        The pixel width and height of the (square) detector grid.
    nz, n : int
        The pixel width and height of the reconstructed grid.
    probe_shape : int
        The pixel width and height of the (square) probe illumination.
    propagation : :py:class:`Operator`
        The wave propagation operator being used.
    diffraction : :py:class:`Operator`
        The object probe interaction operator being used.
    model : string
        The type of noise model to use for the cost functions.

    data : (..., FRAME, WIDE, HIGH) float32
        The intensity (square of the absolute value) of the propagated
        wavefront; i.e. what the detector records.
    farplane: (..., POSI, 1, SHARED, detector_shape, detector_shape) complex64
        The wavefronts hitting the detector respectively.
    probe : {(..., 1, 1, SHARED, WIDE, HIGH), (..., POSI, 1, SHARED, WIDE, HIGH)} complex64
        The complex illumination function.
    psi : (..., WIDE, HIGH) complex64
        The wavefront modulation coefficients of the object.
    scan : (..., POSI, 2) float32
        Coordinates of the minimum corner of the probe grid for each
        measurement in the coordinate system of psi. Coordinate order
        consistent with WIDE, HIGH order.

    """

    def __init__(
        self,
        detector_shape,
        probe_shape,
        nz,
        n,
        ntheta=1,
        model='gaussian',
        propagation=Propagation,
        diffraction=Convolution,
        **kwargs,
    ):
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

    def fwd(self, probe, scan, psi, **kwargs):
        """Please see help(Ptycho) for more info."""
        return self.propagation.fwd(
            self.diffraction.fwd(
                psi=psi,
                scan=scan,
                probe=probe[..., 0, :, :, :],
            ),
            overwrite=True,
        )[..., None, :, :, :]

    def adj(self, farplane, probe, scan, psi=None, overwrite=False, **kwargs):
        """Please see help(Ptycho) for more info."""
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
        """Please see help(Ptycho) for more info."""
        return self.diffraction.adj_probe(
            psi=psi,
            scan=scan,
            nearplane=self.propagation.adj(
                farplane=farplane,
                overwrite=overwrite,
            )[..., 0, :, :, :],
            overwrite=True,
        )[..., None, :, :, :]

    def _compute_intensity(self, data, psi, scan, probe):
        """Compute detector intensities replacing the nth probe mode"""
        farplane = self.fwd(
            psi=psi,
            scan=scan,
            probe=probe,
        )
        return self.xp.sum(
            (farplane * farplane.conj()).real,
            axis=tuple(range(1, farplane.ndim - 2)),
        ), farplane

    def cost(self, data, psi, scan, probe) -> float:
        """Please see help(Ptycho) for more info."""
        intensity, _ = self._compute_intensity(data, psi, scan, probe)
        return self.propagation.cost(data, intensity)

    def grad_psi(self, data, psi, scan, probe):
        """Please see help(Ptycho) for more info."""
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

    def grad_probe(self, data, psi, scan, probe, mode=None):
        """Compute the gradient with respect to the probe(s).

        Parameters
        ----------
        mode : list(int)
            Only return the gradient with resepect to these probes.

        """
        mode = list(range(probe.shape[-3])) if mode is None else mode
        intensity, farplane = self._compute_intensity(data, psi, scan, probe)
        # Use the average gradient for all probe positions
        return self.xp.mean(
            self.adj_probe(
                farplane=self.propagation.grad(
                    data,
                    farplane[..., mode, :, :],
                    intensity,
                ),
                psi=psi,
                scan=scan,
                overwrite=True,
            ),
            axis=0,
            keepdims=True,
        )

    def adj_all(self, farplane, probe, scan, psi, overwrite=False, rpie=False):
        """Please see help(Ptycho) for more info."""
        result = self.diffraction.adj_all(
            nearplane=self.propagation.adj(
                farplane,
                overwrite=overwrite,
            )[..., 0, :, :, :],
            probe=probe[..., 0, :, :, :],
            scan=scan,
            overwrite=True,
            psi=psi,
            rpie=rpie,
        )
        return (result[0], result[1][..., None, :, :, :], *result[2:])
