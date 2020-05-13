"""Defines a free-space propagation operator based on the NumPy FFT module."""

import numpy as np
from numpy.fft import fftn, ifftn

from .operator import Operator


class Propagation(Operator):
    """A base class for Fourier-based free-space propagation.

    Take an (..., N, N) array and pads the last two dimensions with zeros until
    it is size (..., M, M). Then apply the Fourier transform to the last two
    dimensions.

    Attributes
    ----------
    probe_shape : int
        The pixel width and height of the nearplane waves.
    detector_shape : int
        The pixel width and height of the farplane waves.
    nwaves : int
        The number of waves to propagate from the nearplane to farplane.
    model : string
        The type of noise model to use for the cost functions.
    cost : (data-like, farplane-like) -> float
        The function to be minimized when solving a problem.
    grad : (data-like, farplane-like) -> farplane-like
        The gradient of cost.

    Parameters
    ----------
    nearplane: (..., probe_shape, probe_shape) complex64
        The wavefronts after exiting the object.
    farplane: (..., detector_shape, detector_shape) complex64
        The wavefronts hitting the detector respectively.
    data : (ntheta, nscan, detector_shape, detector_shape) complex64
        data is the square of the absolute value of `farplane`. `data` is the
        intensity of the `farplane`.

    """

    def __init__(self, nwaves, detector_shape, probe_shape, model='gaussian',
                 **kwargs):  # yapf: disable
        self.nwaves = nwaves
        self.probe_shape = probe_shape
        self.detector_shape = detector_shape
        self.cost = getattr(self, f'_{model}_cost')
        self.grad = getattr(self, f'_{model}_grad')

    def fwd(self, nearplane, overwrite=False, **kwargs):
        """Forward Fourier-based free-space propagation operator."""
        self._check_shape(nearplane)
        if not overwrite:
            nearplane = np.copy(nearplane)
        shape = nearplane.shape
        return fftn(
            nearplane.reshape(self.nwaves, self.detector_shape,
                              self.detector_shape),
            norm='ortho',
            axes=(-2, -1),
            # overwrite_x=True,
        ).reshape(shape).astype('complex64')

    def adj(self, farplane, overwrite=False, **kwargs):
        """Adjoint Fourier-based free-space propagation operator."""
        self._check_shape(farplane)
        if not overwrite:
            farplane = np.copy(farplane)
        shape = farplane.shape
        return ifftn(
            farplane.reshape(self.nwaves, self.detector_shape,
                             self.detector_shape),
            norm='ortho',
            axes=(-2, -1),
            # overwrite_x=True,
        ).reshape(shape).astype('complex64')

    def _check_shape(self, x):
        assert type(x) is self.xp.ndarray, type(x)
        shape = (self.nwaves, self.detector_shape, self.detector_shape)
        if (__debug__ and x.shape[-2:] != shape[-2:]
                and np.prod(x.shape[:-2]) != self.nwaves):
            raise ValueError(f'waves must have shape {shape} not {x.shape}.')

    # COST FUNCTIONS AND GRADIENTS --------------------------------------------

    def _gaussian_cost(self, data, intensity):
        return np.linalg.norm(np.ravel(np.sqrt(intensity) - np.sqrt(data)))**2

    def _gaussian_grad(self, data, farplane, intensity, overwrite=False):
        print('tst')
        return farplane * (
            1 - np.sqrt(data) / (np.sqrt(intensity) + 1e-32)
        )[:, :, np.newaxis, np.newaxis]  # yapf:disable

    def _poisson_cost(self, data, intensity):
        return np.sum(intensity - data * np.log(intensity + 1e-32))

    def _poisson_grad(self, data, farplane, intensity, overwrite=False):
        return farplane * (
            1 - data / (intensity + 1e-32)
        )[:, :, np.newaxis, np.newaxis]  # yapf: disable
