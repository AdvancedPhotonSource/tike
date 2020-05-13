from importlib_resources import files
import concurrent.futures as cf
import threading
from functools import partial

import cupy as cp

from .. import numpy
from .operator import Operator

_cu_source = files('tike.operators.cupy').joinpath('convolution.cu').read_text()
_patch_kernel = cp.RawKernel(_cu_source, "patch")


class Convolution(Operator, numpy.Convolution):
    def __enter__(self):
        max_thread = min(self.probe_shape**2,
                         _patch_kernel.attributes['max_threads_per_block'])
        self.blocks = (max_thread, )
        self.grids = (
            -(-self.probe_shape**2 // max_thread),  # ceil division
            self.nscan,
            self.ntheta,
        )
        self.pad = (self.detector_shape - self.probe_shape) // 2
        self.end = self.probe_shape + self.pad
        return self

    def __exit__(self, type, value, traceback):
        pass

    def fwd(self, psi, scan, probe):
        """Extract probe shaped patches from the psi at each scan position.

        The patches within the bounds of psi are linearly interpolated, and
        indices outside the bounds of psi are not allowed.
        """
        psi = psi.reshape(self.ntheta, self.nz, self.n)
        self._check_shape_probe(probe)
        patches = cp.zeros(
            (self.ntheta, self.nscan // self.fly, self.fly, 1,
             self.detector_shape, self.detector_shape),
            dtype='complex64',
        )

        _patch_kernel = cp.RawKernel(_cu_source, "patch")
        _patch_kernel(
            self.grids,
            self.blocks,
            (psi, patches, scan, self.ntheta, self.nz, self.n, self.nscan,
             self.probe_shape, self.detector_shape, True),
        )
        patches[..., self.pad:self.end, self.pad:self.end] *= probe
        return patches

    def fwd_multi(self, gpu_id, psi, scan, probe):
        """Extract probe shaped patches from the psi at each scan position.

        The patches within the bounds of psi are linearly interpolated, and
        indices outside the bounds of psi are not allowed.
        """
        test_shape = self.probe_shape
        test_shape1 = self.detector_shape
        gpu_id = 2
        patches = [None]*gpu_id
        for gpu_i in range(gpu_id):
        #def multiGPU_init(gpu_i, psi, scan, probe):
            with cp.cuda.Device(gpu_i):
                print('testconv2:',gpu_i, self.probe_shape, self.detector_shape)
                psi[gpu_i] = psi[gpu_i].reshape(self.ntheta, self.nz, self.n)
                self._check_shape_probe(probe[gpu_i])
                patches[gpu_i] = cp.zeros(
                    (self.ntheta, self.nscan // self.fly, self.fly, 1,
                     self.detector_shape, self.detector_shape),
                    dtype='complex64',
                )
                _patch_kernel = cp.RawKernel(_cu_source, "patch")
                _patch_kernel(
                    self.grids,
                    self.blocks,
                    (),
                )
                #_patch_kernel(
                #    self.grids,
                #    self.blocks,
                #    (psi[gpu_i], patches[gpu_i], scan[gpu_i], self.ntheta, self.nz, self.n, self.nscan,
                #     self.probe_shape, self.detector_shape, True),
                #)
                print('testconv:',gpu_i, scan[gpu_i][:,:,1])
                print('testconv1:',patches[gpu_i].shape)
                patches[gpu_i][..., self.pad:self.end, self.pad:self.end] *= probe[gpu_i]
                #return patches

        #gpu_count = gpu_id
        #gpu_list = range(gpu_count)
        #with cf.ThreadPoolExecutor(max_workers=gpu_count) as executor:
        #    intensity = executor.map(multiGPU_init, gpu_list, psi, scan, probe)
        exit()

    def adj(self, nearplane, scan, probe, obj=None, overwrite=False):
        """Combine probe shaped patches into a psi shaped grid by addition."""
        self._check_shape_nearplane(nearplane)
        self._check_shape_probe(probe)
        if not overwrite:
            nearplane = nearplane.copy()
        nearplane[..., self.pad:self.end, self.pad:self.end] *= cp.conj(probe)
        if obj is None:
            obj = cp.zeros((self.ntheta, self.nz, self.n), dtype='complex64')
        _patch_kernel = cp.RawKernel(_cu_source, "patch")
        _patch_kernel(
            self.grids,
            self.blocks,
            (obj, nearplane, scan, self.ntheta, self.nz, self.n, self.nscan,
             self.probe_shape, self.detector_shape, False),
        )
        return obj

    def adj_probe(self, nearplane, scan, psi, overwrite=False):
        """Combine probe shaped patches into a probe."""
        self._check_shape_nearplane(nearplane)
        patches = cp.empty(
            (self.ntheta, self.nscan // self.fly, self.fly, 1,
             self.probe_shape, self.probe_shape),
            dtype='complex64',
        )
        _patch_kernel(
            self.grids,
            self.blocks,
            (psi, patches, scan, self.ntheta, self.nz, self.n, self.nscan,
             self.probe_shape, self.probe_shape, True),
        )
        return (nearplane[..., self.pad:self.end, self.pad:self.end] *
                cp.conj(patches))
