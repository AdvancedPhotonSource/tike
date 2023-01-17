import bz2
import os

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import pytest
import unittest

import tike.ptycho
from tike.ptycho.solvers.options import (
    _resize_fft,
    _resize_spline,
    _resize_cubic,
    _resize_lanczos,
    _resize_linear,
)

from .templates import _mpi_size
from .io import result_dir, data_dir
from .test_ptycho import PtychoRecon

output_folder = os.path.join(result_dir, 'multigrid')


@pytest.mark.parametrize("function", [
    _resize_fft,
    _resize_spline,
    _resize_linear,
    _resize_cubic,
    _resize_lanczos,
])
def test_resample(function, filename='siemens-star-small.npz.bz2'):

    os.makedirs(output_folder, exist_ok=True)

    dataset_file = os.path.join(data_dir, filename)
    with bz2.open(dataset_file, 'rb') as f:
        archive = np.load(f)
        probe = archive['probe'][0]

    for i in [0.25, 0.50, 1.0, 2.0, 4.0]:
        p1 = function(probe, i)
        flattened = np.concatenate(
            p1.reshape((-1, *p1.shape[-2:])),
            axis=1,
        )
        plt.imsave(
            f'{output_folder}/{function.__name__}-probe-ampli-{i}.png',
            np.abs(flattened),
        )
        plt.imsave(
            f'{output_folder}/{function.__name__}-probe-phase-{i}.png',
            np.angle(flattened),
        )


@unittest.skipIf(
    _mpi_size > 1,
    reason="MPI not implemented for multi-grid.",
)
class ReconMultiGrid():
    """Test ptychography multi-grid reconstruction method."""

    def interp(self, x, f):
        pass

    def template_consistent_algorithm(self, *, data, params):
        """Check ptycho.solver.algorithm for consistency."""
        if _mpi_size > 1:
            raise NotImplementedError()

        with cp.cuda.Device(self.gpu_indices[0]):
            parameters = tike.ptycho.reconstruct_multigrid(
                parameters=params,
                data=self.data,
                num_gpu=self.gpu_indices,
                use_mpi=self.mpi_size > 1,
                num_levels=2,
                interp=self.interp,
            )

        print()
        print('\n'.join(
            f'{c[0]:1.3e}' for c in parameters.algorithm_options.costs))
        return parameters


class TestPtychoReconMultiGridFFT(
        ReconMultiGrid,
        PtychoRecon,
        unittest.TestCase,
):

    post_name = '-multigrid-fft'

    def interp(self, x, f):
        return _resize_fft(x, f)


if False:
    # Don't need to run these tests on CI every time.

    class TestPtychoReconMultiGridLinear(PtychoReconMultiGrid, TestPtychoRecon,
                                         unittest.TestCase):

        post_name = '-multigrid-linear'

        def interp(self, x, f):
            return _resize_linear(x, f)

    class TestPtychoReconMultiGridCubic(PtychoReconMultiGrid, TestPtychoRecon,
                                        unittest.TestCase):

        post_name = '-multigrid-cubic'

        def interp(self, x, f):
            return _resize_cubic(x, f)

    class TestPtychoReconMultiGridLanczos(PtychoReconMultiGrid, TestPtychoRecon,
                                          unittest.TestCase):

        post_name = '-multigrid-lanczos'

        def interp(self, x, f):
            return _resize_lanczos(x, f)

    class TestPtychoReconMultiGridSpline(PtychoReconMultiGrid, TestPtychoRecon,
                                         unittest.TestCase):

        post_name = '-multigrid-spline'

        def interp(self, x, f):
            return _resize_spline(x, f)
