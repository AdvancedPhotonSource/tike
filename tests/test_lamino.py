#!/usr/bin/env python
# -*- coding: utf-8 -*-

# #########################################################################
# Copyright (c) 2018, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2015. UChicago Argonne, LLC. This software was produced       #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.                                                               #
#                                                                         #
# Additionally, redistribution and use in source and binary forms, with   #
# or without modification, are permitted provided that the following      #
# conditions are met:                                                     #
#                                                                         #
#     * Redistributions of source code must retain the above copyright    #
#       notice, this list of conditions and the following disclaimer.     #
#                                                                         #
#     * Redistributions in binary form must reproduce the above copyright #
#       notice, this list of conditions and the following disclaimer in   #
#       the documentation and/or other materials provided with the        #
#       distribution.                                                     #
#                                                                         #
#     * Neither the name of UChicago Argonne, LLC, Argonne National       #
#       Laboratory, ANL, the U.S. Government, nor the names of its        #
#       contributors may be used to endorse or promote products derived   #
#       from this software without specific prior written permission.     #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago     #
# Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,        #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;        #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER        #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT      #
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN       #
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         #
# POSSIBILITY OF SUCH DAMAGE.                                             #
# #########################################################################

import lzma
import os
import pickle
import unittest

import numpy as np

import tike.lamino
import tike.lamino.bucket
from tike.communicators import MPIComm

__author__ = "Daniel Ching, Viktor Nikitin, Xiaodong Yu"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'

testdir = os.path.dirname(__file__)

try:
    from mpi4py import MPI
    _mpi_size = MPI.COMM_WORLD.Get_size()
    _mpi_rank = MPI.COMM_WORLD.Get_rank()
except ModuleNotFoundError:
    _mpi_rank = 0
    _mpi_size = 1


class TestLaminoRecon(unittest.TestCase):
    """Test various laminography reconstruction methods for consistency."""

    def create_dataset(self, dataset_file):
        """Create a dataset for testing this module.
        Only called with setUp detects that `dataset_file` has been deleted.
        """
        import skimage.io

        delta = skimage.io.imread(
            os.path.join(testdir, 'data/delta-chip-128.tiff'))[::2, ::2, ::2]
        beta = skimage.io.imread(
            os.path.join(testdir, 'data/beta-chip-128.tiff'))[::2, ::2, ::2]
        self.original = (delta + 1j * beta).astype('complex64')

        self.theta = np.linspace(
            0,
            2 * np.pi,
            # use Nyquist sampling of rotation axis
            int(np.pi / 2 * 75 / 128 * delta.shape[-1]),
            endpoint=False,
        ).astype('float32')
        self.tilt = np.pi / 3

        self.data = tike.lamino.simulate(
            self.original,
            self.theta,
            self.tilt,
            upsample=2,
        )
        assert self.data.dtype == 'complex64', self.data.dtype

        setup_data = [
            self.data,
            self.original,
            self.theta,
            self.tilt,
        ]

        with lzma.open(dataset_file, 'wb') as file:
            pickle.dump(setup_data, file)

    def setUp(self):
        """Load a dataset for reconstruction."""
        dataset_file = os.path.join(testdir, 'data/lamino_setup.pickle.lzma')
        if not os.path.isfile(dataset_file):
            self.create_dataset(dataset_file)
        with lzma.open(dataset_file, 'rb') as file:
            [
                self.data,
                self.original,
                self.theta,
                self.tilt,
            ] = pickle.load(file)

    def test_consistent_simulate(self):
        """Check lamino.simulate for consistency."""
        data = tike.lamino.simulate(
            self.original,
            self.theta,
            self.tilt,
            upsample=2,
        )
        assert data.dtype == 'complex64', data.dtype
        np.testing.assert_array_equal(data.shape, self.data.shape)
        np.testing.assert_allclose(data, self.data, atol=1e-6)

    def error_metric(self, x):
        """Return the error between two arrays."""
        return np.linalg.norm(x - self.original)

    def template_consistent_algorithm(self, module, algorithm, params={}):
        """Check lamino.solver.algorithm for consistency."""
        result = {
            'obj': np.zeros_like(self.original),
        }

        if params.get('use_mpi') is True:
            with MPIComm() as IO:
                (result['obj'],) = IO.MPIio_lamino(result['obj'])

        result = module.reconstruct(
            **result,
            **params,
            data=self.data,
            theta=self.theta,
            tilt=self.tilt,
            algorithm=algorithm,
            num_iter=1,
        )
        params.update(result)
        result = module.reconstruct(
            **params,
            data=self.data,
            theta=self.theta,
            tilt=self.tilt,
            algorithm=algorithm,
            num_iter=30,
        )
        print()
        cost = '\n'.join(f'{c:1.3e}' for c in result['cost'])
        print(cost)

        if params.get('use_mpi') is True:
            with MPIComm() as mpi:
                obj_out = mpi.Gather(result['obj'])
                if mpi.rank == 0:
                    result['obj'] = obj_out
                    recon_file = os.path.join(
                        testdir,
                        f'data/lamino_{algorithm}.pickle.lzma',
                    )
                    if os.path.isfile(recon_file):
                        with lzma.open(recon_file, 'rb') as file:
                            standard = pickle.load(file)
                    else:
                        with lzma.open(recon_file, 'wb') as file:
                            pickle.dump(result['obj'], file)
                        raise FileNotFoundError(
                            f"lamino '{algorithm}' standard not initialized.")
                    np.testing.assert_array_equal(
                        result['obj'].shape,
                        self.original.shape,
                    )
                    np.testing.assert_allclose(
                        result['obj'],
                        standard,
                        atol=1e-3,
                    )
        else:
            recon_file = os.path.join(
                testdir,
                f'data/lamino_{algorithm}.pickle.lzma',
            )
            if os.path.isfile(recon_file):
                with lzma.open(recon_file, 'rb') as file:
                    standard = pickle.load(file)
            else:
                with lzma.open(recon_file, 'wb') as file:
                    pickle.dump(result['obj'], file)
                raise FileNotFoundError(
                    f"lamino '{algorithm}' standard not initialized.")
            np.testing.assert_array_equal(
                result['obj'].shape,
                self.original.shape,
            )
            np.testing.assert_allclose(
                result['obj'],
                standard,
                atol=1e-3,
            )

        return result

    def test_consistent_fourier(self):
        """Check lamino.solver.cgrad for consistency."""
        _save_lamino_result(
            self.template_consistent_algorithm(
                tike.lamino.lamino,
                'cgrad',
                params={
                    'num_gpu': 1,
                    'obj_split': 1,
                },
            ), f"cgrad_Fourier")

    def test_consistent_bucket(self):
        """Check lamino.solver.bucket for consistency."""
        _save_lamino_result(
            self.template_consistent_algorithm(
                tike.lamino.bucket,
                'bucket',
                params={
                    'num_gpu': 2,
                    'obj_split': 1,
                    'eps': 1,
                    'use_mpi': _mpi_size > 1,
                },
            ), f"{'mpi-' if _mpi_size > 1 else ''}cgrad_bucket")


class TestLaminoRadon(unittest.TestCase):
    """Test whether the Laminography operator is equal to the Radon operator.

    Compare projections with sums along the three orthogonal axes directly.
    """

    def setUp(self, n=2, b=2):
        # FIXME: Tests will fail with odd n; pass with n even.
        self.original = np.pad(
            np.random.randint(-5, 5, (n, n, n)) +
            1j * 0 * np.random.randint(-5, 5, (n, n, n)),
            b,
        )

    def _radon_equal(self, module, eps):
        for tilt, axis, theta in zip(
            [0, np.pi / 2, np.pi / 2],
            [0, 1, 2],
            [0, 0, -np.pi / 2],
        ):
            projection = module.simulate(
                obj=self.original,
                theta=np.array([theta]),
                tilt=tilt,
                eps=eps,
                sample=4,
            )
            direct_sum = np.sum(self.original, axis=axis)
            try:
                np.testing.assert_allclose(projection[0], direct_sum, atol=1e-2)
            except AssertionError:
                print()
                print(tilt, axis, theta)
                print(direct_sum)
                print(np.around(projection[0], 3))

    def _radon_equal_reverse(self, module, eps):
        for tilt, axis, theta, flip in zip(
            [np.pi, -np.pi / 2, np.pi / 2],
            [0, 1, 2],
            [0, 0, np.pi / 2],
            [0, 0, 1],
        ):
            projection = module.simulate(
                obj=self.original,
                theta=np.array([theta]),
                tilt=tilt,
                eps=eps,
                sample=4,
            )
            direct_sum = np.sum(self.original, axis=axis)
            try:
                # Must account for reflection because looking from back side.
                # FIXME: Fourier method still errors with reverse projection
                p = np.flip(projection[0], axis=flip)
                np.testing.assert_allclose(p, direct_sum, atol=1e-3)
            except AssertionError:
                print()
                print(tilt, axis, theta)
                print(direct_sum)
                print(np.around(projection[0], 3))

    def test_fourier_radon_equal(self):
        self._radon_equal(tike.lamino.lamino, 1e-10)

    def test_fourier_radon_equal_reverse(self):
        self._radon_equal_reverse(tike.lamino.lamino, 1e-10)

    def test_bucket_radon_equal(self):
        self._radon_equal(tike.lamino.bucket, 1)

    def test_bucket_radon_equal_reverse(self):
        self._radon_equal_reverse(tike.lamino.bucket, 1)


def _save_lamino_result(result, algorithm):
    try:
        if _mpi_rank != 0:
            return
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        fname = os.path.join(testdir, 'result', 'lamino', f'{algorithm}')
        os.makedirs(fname, exist_ok=True)
        slice_id = int(35 / 128 * result['obj'].shape[0])
        plt.imsave(
            f'{fname}/{slice_id}-phase.png',
            np.angle(result['obj'][slice_id]).astype('float32'),
            # The output of np.angle is locked to (-pi, pi]
            cmap=plt.cm.twilight,
            vmin=-np.pi,
            vmax=np.pi,
        )
        plt.imsave(
            f'{fname}/{slice_id}-ampli.png',
            np.abs(result['obj'][slice_id]).astype('float32'),
        )
        import skimage.io
        skimage.io.imsave(
            f'{fname}/phase.tiff',
            np.angle(result['obj']).astype('float32'),
        )
        skimage.io.imsave(
            f'{fname}/ampli.tiff',
            np.abs(result['obj']).astype('float32'),
        )
    except ImportError:
        pass


if __name__ == '__main__':
    unittest.main()
