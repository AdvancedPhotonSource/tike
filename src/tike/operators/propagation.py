"""Defines a free-space propagation operator based on the NumPy FFT module."""

import numpy as np

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

    def fwd(self, nearplane, **kwargs):
        """Forward Fourier-based free-space propagation operator."""
        assert self.nwaves == np.prod(nearplane.shape[:-2])
        pad = (self.detector_shape - self.probe_shape) // 2
        end = self.probe_shape + pad
        # We copy nearplane into a larger array the shape of the farplane
        # because this is faster than numpy.pad for older versions of NumPy.
        padded_nearplane = np.zeros(
            (*nearplane.shape[:-2], self.detector_shape, self.detector_shape),
            dtype=nearplane.dtype,
        )
        padded_nearplane[..., pad:end, pad:end] = nearplane
        # We must cast the result of np.fft.fft2
        # because this implementation does not preserve type.
        return np.fft.fft2(
            padded_nearplane,
            norm='ortho',
        ).astype(nearplane.dtype)

    def adj(self, farplane, **kwargs):
        """Adjoint Fourier-based free-space propagation operator."""
        assert self.nwaves == np.prod(farplane.shape[:-2])
        pad = (self.detector_shape - self.probe_shape) // 2
        end = self.probe_shape + pad
        return np.fft.ifft2(
            farplane,
            norm='ortho',
        )[..., pad:end, pad:end].astype(farplane.dtype)

    # COST FUNCTIONS AND GRADIENTS --------------------------------------------

    def _gaussian_cost(self, data, farplane):
        modulus = np.linalg.norm(
            farplane.reshape(*data.shape[:2], -1, *data.shape[2:]),
            ord=2,
            axis=2,
        )
        return np.linalg.norm(modulus - np.sqrt(data))**2

    def _gaussian_grad(self, data, farplane):
        modulus = np.linalg.norm(
            farplane.reshape(*data.shape[:2], -1, *data.shape[2:]),
            ord=2,
            axis=2,
        )
        return farplane * (
            1 - np.sqrt(data) / (modulus + 1e-32)
        )[:, :, np.newaxis, np.newaxis]  # yapf:disable

    def _poisson_cost(self, data, farplane):
        intensity = np.square(np.linalg.norm(
            farplane.reshape(*data.shape[:2], -1, *data.shape[2:]),
            ord=2,
            axis=2,
        ))
        return np.sum(intensity - data * np.log(intensity + 1e-32))

    def _poisson_grad(self, data, farplane):
        intensity = np.square(np.linalg.norm(
            farplane.reshape(*data.shape[:2], -1, *data.shape[2:]),
            ord=2,
            axis=2,
        ))
        return farplane * (
            1 - data / (intensity + 1e-32)
        )[:, :, np.newaxis, np.newaxis]  # yapf: disable
