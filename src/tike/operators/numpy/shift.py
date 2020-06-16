__author__ = "Daniel Ching, Viktor Nikitin"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."

import numpy as np
from numpy.fft import fft2, ifft2

from .operator import Operator


class Shift(Operator):
    """Shift last two dimensions of an array using Fourier method."""

    def _fft2(self, *args, overwrite=False, **kwargs):
        """Reimplement this wrapper to switch out the FFT library"""
        return fft2(*args, **kwargs).astype('complex64')

    def _ifft2(self, *args, overwrite=False, **kwargs):
        """Reimplement this wrapper to switch out the FFT library"""
        return ifft2(*args, **kwargs).astype('complex64')

    def fwd(self, a, shift, overwrite=False):
        """Apply shifts along last two dimensions of a.

        Parameters
        ----------
        array (..., H, W) float32
            The array to be shifted.
        shift (..., 2) float32
            The the shifts to be applied along the last two axes.

        """
        shape = a.shape
        a = a.reshape(-1, *shape[-2:])
        pz, pn = a.shape[-2] // 2, a.shape[-1] // 2
        padded = np.pad(a, ((0, 0), (pz, pz), (pn, pn)))
        [x, y] = self.xp.meshgrid(
            self.xp.fft.fftfreq(2 * pn + a.shape[-1]).astype('float32'),
            self.xp.fft.fftfreq(2 * pz + a.shape[-2]).astype('float32'),
        )
        shift = np.exp(
            -2j * np.pi *
            (x * shift[..., 1, None, None] + y * shift[..., 0, None, None]))
        padded = self._fft2(padded, axes=(-2, -1), overwrite=overwrite)
        padded *= shift
        padded = self._ifft2(padded, axes=(-2, -1), overwrite=overwrite)
        return padded[..., pz:-pz, pn:-pn].reshape(shape)

    def adj(self, a, shift):
        """Apply inverse shifts along last two dimensions of a.

        seealso: fwd
        """
        return self.fwd(a, -shift)
