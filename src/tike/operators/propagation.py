"""Defines a free-space propagation operator based on the NumPy FFT module."""

from .operator import Operator


class Propagation(Operator):

    def __init__(self, detector_shape, probe_shape, **kwargs):
        super(Propagation, self).__init__(**kwargs)
        self.probe_shape = probe_shape
        self.detector_shape = detector_shape

    def fwd(self, nearplane, **kwargs):
        """Forward Fourier-based free-space propagation operator."""
        xp = self.array_module
        pad = (self.detector_shape - self.probe_shape) // 2
        end = self.probe_shape + pad
        # We copy nearplane into a larger array the shape of the farplane
        # because this is faster than numpy.pad for older versions of NumPy.
        padded_nearplane = xp.zeros(
            (*nearplane.shape[:-2], self.detector_shape, self.detector_shape),
            dtype=nearplane.dtype,
        )
        padded_nearplane[..., pad:end, pad:end] = nearplane
        # We must cast the result of xp.fft.fft2
        # because this implementation does not preserve type.
        return xp.fft.fft2(
            padded_nearplane, norm='ortho',
        ).astype(nearplane.dtype)

    def adj(self, farplane, **kwargs):
        """Adjoint Fourier-based free-space propagation operator."""
        xp = self.array_module
        pad = (self.detector_shape - self.probe_shape) // 2
        end = self.probe_shape + pad
        return xp.fft.ifft2(
            farplane, norm='ortho',
        )[..., pad:end, pad:end].astype(farplane.dtype)
