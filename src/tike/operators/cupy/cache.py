__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."

from cupyx.scipy.fft import fftn, ifftn, get_fft_plan
import cupy.cuda.runtime


class CachedFFT():
    """Provides a multi-plan per-device cache for CuPy FFT.

    A class which inherits from this class gains the _fft2, _fftn, and _ifft2
    methods which provide automatic plan caching for the CuPy FFTs.

    This plan cache differs from the cache included in CuPy>=8 because it is
    NOT per-thread. This allows us to use threadpool.map() and allows us to
    destroy the cache manually.
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
        key = (*a.shape, *axes, cupy.cuda.runtime.getDevice())
        if key in self.plan_cache:
            plan = self.plan_cache[key]
        else:
            plan = get_fft_plan(a, axes=axes)
            self.plan_cache[key] = plan
        return plan

    def _fft2(self, a, *args, axes=(-2, -1), **kwargs):
        return self._fftn(a, *args, axes=axes, **kwargs)

    def _ifft2(self, a, *args, axes=(-2, -1), **kwargs):
        return self._ifftn(a, *args, axes=axes, **kwargs)

    def _ifftn(self, a, *args, **kwargs):
        with self._get_fft_plan(a, **kwargs):
            return ifftn(a, *args, **kwargs)

    def _fftn(self, a, *args, **kwargs):
        with self._get_fft_plan(a, **kwargs):
            return fftn(a, *args, **kwargs)
