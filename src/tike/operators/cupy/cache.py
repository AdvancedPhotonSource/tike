__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."

import cupy.fft.config
from cupyx.scipy.fft import fftn, ifftn
from cupyx.scipy.fftpack import get_fft_plan
from numpy.lib.utils import deprecate

# NOTE: Keep this setting change even if class below is removed
cupy.fft.config.enable_nd_planning = True


class CachedFFT():
    """Provides a multi-plan cache for CuPy FFT.

    A class which inherits from this class gains the _fft2, _fftn, and _ifft2
    methods which provide automatic plan caching for the CuPy FFTs.
    """

    def __enter__(self):
        self.plan_cache = {}
        return self

    def __exit__(self, type, value, traceback):
        self.plan_cache.clear()
        del self.plan_cache

    def _get_fft_plan(self, a, axes=None, **kwargs):
        """Cache multiple FFT plans at the same time."""
        axes = tuple(range(a.ndim)) if axes is None else axes
        key = (*a.shape, *axes)
        if key in self.plan_cache:
            plan = self.plan_cache[key]
        else:
            plan = get_fft_plan(a, axes=axes)
            self.plan_cache[key] = plan
        return plan

    @deprecate(
        message='cupy>=8.0 ships an automatic plan cache enabled by default. '
        'Use CuPy FFT functions directly.')
    def _fft2(self, a, *args, overwrite=False, **kwargs):
        with self._get_fft_plan(a, **kwargs):
            return fftn(a, *args, overwrite_x=overwrite, **kwargs)

    @deprecate(
        message='cupy>=8.0 ships an automatic plan cache enabled by default. '
        'Use CuPy FFT functions directly.')
    def _ifft2(self, a, *args, overwrite=False, **kwargs):
        with self._get_fft_plan(a, **kwargs):
            return ifftn(a, *args, overwrite_x=overwrite, **kwargs)

    @deprecate(
        message='cupy>=8.0 ships an automatic plan cache enabled by default. '
        'Use CuPy FFT functions directly.')
    def _fftn(self, a, *args, overwrite=False, **kwargs):
        with self._get_fft_plan(a, **kwargs):
            return fftn(a, *args, overwrite_x=overwrite, **kwargs)
