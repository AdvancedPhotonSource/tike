__author__ = "Daniel Ching, Viktor Nikitin"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."

from .cache import CachedFFT
from .operator import Operator


class Shift(CachedFFT, Operator):
    """Shift last two dimensions of an array using Fourier method."""

    def fwd(self, a, shift, overwrite=False, cval=None):
        """Apply shifts along last two dimensions of a.

        Parameters
        ----------
        array (..., H, W) float32
            The array to be shifted.
        shift (..., 2) float32
            The the shifts to be applied along the last two axes.

        """
        if shift is None:
            return a
        shape = a.shape
        padded = a.reshape(-1, *shape[-2:])
        padded = self._fft2(
            padded,
            axes=(-2, -1),
            overwrite_x=overwrite,
        )
        x, y = self.xp.meshgrid(
            self.xp.fft.fftfreq(padded.shape[-1]).astype('float32'),
            self.xp.fft.fftfreq(padded.shape[-2]).astype('float32'),
        )
        padded *= self.xp.exp(
            -2j * self.xp.pi *
            (x * shift[..., 1, None, None] + y * shift[..., 0, None, None]))
        padded = self._ifft2(padded, axes=(-2, -1), overwrite_x=True)
        return padded.reshape(*shape)

    def adj(self, a, shift, overwrite=False, cval=None):
        if shift is None:
            return a
        return self.fwd(a, -shift, overwrite=overwrite, cval=cval)

    inv = adj
