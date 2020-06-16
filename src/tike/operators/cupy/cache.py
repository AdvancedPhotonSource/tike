import cupy as cp
from cupyx.scipy.fft import fftn, ifftn
from cupyx.scipy.fftpack import get_fft_plan


class CachedFFT():
    """A multi-plan cache for CuPy FFT."""

    def __enter__(self):
        self.plan_cache = {}
        return self

    def __exit__(self, type, value, traceback):
        self.plan_cache.clear()
        del self.plan_cache

    def _get_fft_plan(self, a, axes, **kwargs):
        """Cache multiple FFT plans at the same time."""
        key = (*a.shape, *axes)
        if key in self.plan_cache:
            plan = self.plan_cache[key]
        else:
            plan = get_fft_plan(a, axes=axes)
            self.plan_cache[key] = plan
        return plan

    def _fft2(self, a, *args, overwrite=False, **kwargs):
        with self._get_fft_plan(a, **kwargs):
            return fftn(a, *args, overwrite_x=overwrite, **kwargs)

    def _ifft2(self, a, *args, overwrite=False, **kwargs):
        with self._get_fft_plan(a, **kwargs):
            return ifftn(a, *args, overwrite_x=overwrite, **kwargs)
