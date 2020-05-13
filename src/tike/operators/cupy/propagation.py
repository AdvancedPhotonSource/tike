import cupy as cp
from cupyx.scipy.fft import fftn, ifftn
from cupyx.scipy.fftpack import get_fft_plan

from .. import numpy
from .operator import Operator


# TODO: Check that in-place FFTs for view of contiguous arrays are fixed
# for cupy>=7.4. See cupy#3079

class Propagation(Operator, numpy.Propagation):
    """A Fourier-based free-space propagation using CuPy."""
    def __enter__(self):
        farplane = cp.empty(
            (self.nwaves, self.detector_shape, self.detector_shape),
            dtype='complex64')
        self.plan = get_fft_plan(farplane, axes=(-2, -1))
        del farplane
        return self

    def __exit__(self, type, value, traceback):
        del self.plan
        pass

    def fwd(self, nearplane, overwrite=False, **kwargs):
        self._check_shape(nearplane)
        if not overwrite:
            nearplane = cp.copy(nearplane)
        shape = nearplane.shape
        with self.plan:
            return fftn(
                nearplane.reshape(self.nwaves, self.detector_shape,
                                  self.detector_shape),
                norm='ortho',
                axes=(-2, -1),
                overwrite_x=True,
            ).reshape(shape)

    def fwd_multi(self, gpu_id, nearplane, overwrite=False, **kwargs):
        self._check_shape(nearplane)
        with cp.cuda.Device(gpu_id):
            if not overwrite:
                nearplane = cp.copy(nearplane)
            shape = nearplane.shape
            with self.plan:
                return fftn(
                    nearplane.reshape(self.nwaves, self.detector_shape,
                                      self.detector_shape),
                    norm='ortho',
                    axes=(-2, -1),
                    overwrite_x=True,
                ).reshape(shape)

    def adj(self, farplane, overwrite=False, **kwargs):
        self._check_shape(farplane)
        if not overwrite:
            farplane = cp.copy(farplane)
        shape = farplane.shape
        with self.plan:
            return ifftn(
                farplane.reshape(self.nwaves, self.detector_shape,
                                 self.detector_shape),
                norm='ortho',
                axes=(-2, -1),
                overwrite_x=True,
            ).reshape(shape)

    def _check_shape(self, x):
        assert type(x) is cp.ndarray, type(x)
        shape = (self.nwaves, self.detector_shape, self.detector_shape)
        if (__debug__ and x.shape[-2:] != shape[-2:]
                and cp.prod(x.shape[:-2]) != self.nwaves):
            raise ValueError(f'waves must have shape {shape} not {x.shape}.')
