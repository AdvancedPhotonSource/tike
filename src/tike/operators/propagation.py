"""Defines a free-space propagation operator based on the NumPy FFT module."""

import numpy as np

from .operator import Operator


class Propagation(Operator):

    def __init__(self, nwaves, detector_shape, probe_shape, model='gaussian',
                 **kwargs):
        super(Propagation, self).__init__(**kwargs)
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
            padded_nearplane, norm='ortho',
        ).astype(nearplane.dtype)

    def adj(self, farplane, **kwargs):
        """Adjoint Fourier-based free-space propagation operator."""
        assert self.nwaves == np.prod(farplane.shape[:-2])
        pad = (self.detector_shape - self.probe_shape) // 2
        end = self.probe_shape + pad
        return np.fft.ifft2(
            farplane, norm='ortho',
        )[..., pad:end, pad:end].astype(farplane.dtype)

    # COST FUNCTIONS AND GRADIENTS --------------------------------------------

    def _gaussian_cost(self, data, farplane, mode_axis):
        intensity = np.linalg.norm(farplane, ord=2, axis=mode_axis)
        return np.linalg.norm((np.sqrt(data) - intensity))

    def _gaussian_grad(self, data, farplane, mode_axis):
        intensity = np.linalg.norm(farplane, ord=2, axis=mode_axis)
        return farplane * np.expand_dims(
            np.conj(1 - np.sqrt(data) / (intensity + 1e-32)),
            axis=mode_axis,
        )

    def _poisson_cost(self, data, farplane, mode_axis):
        intensity = np.sum(np.square(np.abs(farplane)), axis=mode_axis)
        return np.sum(intensity - data * np.log(intensity + 1e-32))

    def _poisson_grad(self, data, farplane, mode_axis):
        intensity = np.sum(np.square(np.abs(farplane)), axis=mode_axis)
        return farplane * np.conj(
                1 - np.expand_dims(data / (intensity + 1e-32), axis=mode_axis)
            )
