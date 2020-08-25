"""Defines a free-space propagation operator based on the CuPy FFT module."""

__author__ = "Daniel Ching, Viktor Nikitin"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."

from .cache import CachedFFT
from .operator import Operator
from .usfft import eq2us2d, us2eq2d

import numpy as np


class Propagation(CachedFFT, Operator):
    """A Fourier-based free-space propagation using CuPy.

    Take an (..., N, N) array and apply the Fourier transform to the last two
    dimensions.

    Attributes
    ----------
    detector_shape : int
        The pixel width and height of the nearplane and farplane waves.
    model : string
        The type of noise model to use for the cost functions.
    cost : (data-like, farplane-like) -> float
        The function to be minimized when solving a problem.
    grad : (data-like, farplane-like) -> farplane-like
        The gradient of cost.
    x : frequency information

    Parameters
    ----------
    nearplane: (..., detector_shape, detector_shape) complex64
        The wavefronts after exiting the object.
    farplane: (..., detector_shape, detector_shape) complex64
        The wavefronts hitting the detector respectively.
        Shape for cost functions and gradients is
        (ntheta, nscan // fly, fly, 1, detector_shape, detector_shape).
    data, intensity : (ntheta, nscan, detector_shape, detector_shape) complex64
        data is the square of the absolute value of `farplane`. `data` is the
        intensity of the `farplane`.

    """

    def __init__(self, detector_shape, x, model='gaussian', **kwargs):
        self.detector_shape = detector_shape
        self.cost = getattr(self, f'_{model}_cost')
        self.grad = getattr(self, f'_{model}_grad')
        self.x = x

    def fwd(self, nearplane, overwrite=False, **kwargs):
        """Forward Fourier-based free-space propagation operator."""
        self._check_shape(nearplane)
        shape = nearplane.shape
        return self._fft2(
            nearplane.reshape(-1, self.detector_shape, self.detector_shape),
            norm='ortho',
            axes=(-2, -1),
            overwrite=overwrite,
        ).reshape(shape)

    def _fft2(self, a, norm=None, **kwargs):
        xp = self.xp
        M = a.shape[0]
        n = a.shape[1]
        a = xp.fft.fftshift(a, axes=(-1, -2))
        # [_, kv, ku] = xp.mgrid[0:M, -n // 2:n // 2, -n // 2:n // 2] / n
        # ku = xp.fft.fftshift(ku, axes=(-1, -2))
        # kv = xp.fft.fftshift(kv, axes=(-1, -2))
        # ku = ku.reshape(M, -1).astype('float32')
        # kv = kv.reshape(M, -1).astype('float32')
        # x = xp.stack((kv, ku), axis=-1)
        F = eq2us2d(a, self.x, n, 1e-6, xp=xp).reshape(M, n, n)
        if norm == 'ortho':
            F /= n
        return F

    def adj(self, farplane, overwrite=False, **kwargs):
        """Adjoint Fourier-based free-space propagation operator."""
        self._check_shape(farplane)
        shape = farplane.shape
        return self._ifft2(
            farplane.reshape(-1, self.detector_shape, self.detector_shape),
            norm='ortho',
            axes=(-2, -1),
            overwrite=overwrite,
        ).reshape(shape)

    def _ifft2(self, a, norm=None, **kwargs):
        xp = self.xp
        M = a.shape[0]
        n = a.shape[1]
        # [_, kv, ku] = xp.mgrid[0:M, -n // 2:n // 2, -n // 2:n // 2] / n
        # ku = xp.fft.fftshift(ku, axes=(-1, -2))
        # kv = xp.fft.fftshift(kv, axes=(-1, -2))
        # ku = ku.reshape(M, -1).astype('float32')
        # kv = kv.reshape(M, -1).astype('float32')
        # x = xp.stack((kv, ku), axis=-1)
        F = xp.zeros(a.shape, dtype='complex64')
        a = a.reshape(a.shape[0], -1)
        F = us2eq2d(a, -self.x, n, 1e-6, xp=xp)
        if norm == 'ortho':
            F /= n
        else:
            F /= n * n
        F = xp.fft.ifftshift(F, axes=(-1, -2))
        return F

    def _check_shape(self, x):
        assert type(x) is self.xp.ndarray, type(x)
        shape = (-1, self.detector_shape, self.detector_shape)
        if (__debug__ and x.shape[-2:] != shape[-2:]):
            raise ValueError(f'waves must have shape {shape} not {x.shape}.')

    # COST FUNCTIONS AND GRADIENTS --------------------------------------------

    def _gaussian_cost(self, data, intensity):
        return np.linalg.norm(np.ravel(np.sqrt(intensity) - np.sqrt(data)))**2

    def _gaussian_grad(self, data, farplane, intensity, overwrite=False):
        return farplane * (
            1 - np.sqrt(data) / (np.sqrt(intensity) + 1e-32)
        )[:, :, np.newaxis, np.newaxis]  # yapf:disable

    def _poisson_cost(self, data, intensity):
        return np.sum(intensity - data * np.log(intensity + 1e-32))

    def _poisson_grad(self, data, farplane, intensity, overwrite=False):
        return farplane * (
            1 - data / (intensity + 1e-32)
        )[:, :, np.newaxis, np.newaxis]  # yapf: disable
