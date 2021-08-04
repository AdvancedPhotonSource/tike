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

import bz2
import lzma
import os
import pickle
import unittest

import numpy as np
from mpi4py import MPI

import tike.ptycho
from tike.ptycho.probe import ProbeOptions
from tike.ptycho.position import PositionOptions
from tike.ptycho.object import ObjectOptions
from tike.communicators import Comm, MPIComm
import tike.random

__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2018, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'

testdir = os.path.dirname(__file__)
_mpi_size = MPI.COMM_WORLD.Get_size()


class TestPtychoUtils(unittest.TestCase):
    """Test various utility functions for correctness."""

    def test_gaussian(self):
        """Check ptycho.gaussian for correctness."""
        fname = os.path.join(testdir, 'data/ptycho_gaussian.pickle.lzma')
        weights = tike.ptycho.probe.gaussian(15, rin=0.8, rout=1.0)
        if os.path.isfile(fname):
            with lzma.open(fname, 'rb') as file:
                truth = pickle.load(file)
        else:
            with lzma.open(fname, 'wb') as file:
                truth = pickle.dump(weights, file)
        np.testing.assert_array_equal(weights, truth)

    def test_check_allowed_positions(self):
        psi = np.empty((7, 4, 9))
        probe = np.empty((7, 1, 1, 8, 2, 2))
        scan = np.array([[1, 1], [1, 6.9], [1.1, 1], [1.9, 5.5]])
        tike.ptycho.check_allowed_positions(scan, psi, probe.shape)

        for scan in np.array([[1, 7], [1, 0.9], [0.9, 1], [1, 0]]):
            with self.assertRaises(ValueError):
                tike.ptycho.check_allowed_positions(scan, psi, probe.shape)

    def test_split_by_scan(self):
        scan = np.mgrid[0:3, 0:3].reshape(2, 1, -1)
        scan = np.moveaxis(scan, 0, -1)

        ind = tike.ptycho.ptycho.split_by_scan_stripes(scan, 3, axis=0)
        split = [scan[:, i] for i in ind]
        solution = [
            [[[0, 0], [0, 1], [0, 2]]],
            [[[1, 0], [1, 1], [1, 2]]],
            [[[2, 0], [2, 1], [2, 2]]],
        ]
        np.testing.assert_equal(split, solution)

        ind = tike.ptycho.ptycho.split_by_scan_stripes(scan, 3, axis=1)
        split = [scan[:, i] for i in ind]
        solution = [
            [[[0, 0], [1, 0], [2, 0]]],
            [[[0, 1], [1, 1], [2, 1]]],
            [[[0, 2], [1, 2], [2, 2]]],
        ]
        np.testing.assert_equal(split, solution)


class TestPtychoSimulate(unittest.TestCase):

    def create_dataset(
        self,
        dataset_file,
        pw=16,
        eigen=1,
        width=128,
    ):
        """Create a dataset for testing this module.

        Only called with setUp detects that `dataset_file` has been deleted.
        """
        import libimage
        # Create a stack of phase-only images
        phase = np.stack(
            [libimage.load('satyre', width),
             libimage.load('satyre', width)],
            axis=0,
        )
        amplitude = np.stack(
            [
                1 - 0 * libimage.load('coins', width),
                1 - libimage.load('coins', width)
            ],
            axis=0,
        )
        original = amplitude * np.exp(1j * phase * np.pi)
        self.original = original.astype('complex64')
        leading = self.original.shape[:-2]

        # Create a multi-probe with gaussian amplitude decreasing as 1/N
        phase = np.stack(
            [
                1 - libimage.load('cryptomeria', pw),
                1 - libimage.load('bombus', pw)
            ],
            axis=0,
        )
        weights = 1.0 / np.arange(1, len(phase) + 1)[:, None, None]
        weights = weights * tike.ptycho.probe.gaussian(pw, rin=0.8, rout=1.0)
        probe = weights * np.exp(1j * phase * np.pi)
        self.probe = np.tile(
            probe.astype('complex64'),
            (*leading, 1, eigen, 1, 1, 1),
        )

        pad = 2
        v, h = np.meshgrid(
            np.linspace(pad, original.shape[-2] - pw - pad, 13, endpoint=True),
            np.linspace(pad, original.shape[-1] - pw - pad, 13, endpoint=True),
            indexing='ij',
        )
        scan = np.stack((np.ravel(v), np.ravel(h)), axis=1)
        self.scan = np.tile(
            scan.astype('float32'),
            (*leading, 1, 1),
        )

        self.data = tike.ptycho.simulate(
            detector_shape=pw * 2,
            probe=self.probe,
            scan=self.scan,
            psi=self.original,
        )

        assert self.data.shape == (*leading, 13 * 13, pw * 2, pw * 2)
        assert self.data.dtype == 'float32', self.data.dtype

        setup_data = [
            self.data,
            self.scan,
            self.probe,
            self.original,
        ]
        with lzma.open(dataset_file, 'wb') as file:
            pickle.dump(setup_data, file)

    def setUp(self):
        """Load a dataset for reconstruction."""
        dataset_file = os.path.join(testdir, 'data/ptycho_setup.pickle.lzma')
        if not os.path.isfile(dataset_file):
            self.create_dataset(dataset_file)
        with lzma.open(dataset_file, 'rb') as file:
            [
                self.data,
                self.scan,
                self.probe,
                self.original,
            ] = pickle.load(file)

    def test_consistent_simulate(self):
        """Check ptycho.simulate for consistency."""
        data = tike.ptycho.simulate(
            detector_shape=self.data.shape[-1],
            probe=self.probe,
            scan=self.scan,
            psi=self.original,
            fly=self.scan.shape[-2] // self.data.shape[-3],
        )
        assert data.dtype == 'float32', data.dtype
        assert self.data.dtype == 'float32', self.data.dtype
        np.testing.assert_array_equal(data.shape, self.data.shape)
        np.testing.assert_allclose(np.sqrt(data), np.sqrt(self.data), atol=1e-6)


class TestPtychoRecon(unittest.TestCase):
    """Test various ptychography reconstruction methods for consistency."""

    def setUp(self, filename='data/siemens-star-small.npz.bz2'):
        """Load a dataset for reconstruction."""
        dataset_file = os.path.join(testdir, filename)
        with bz2.open(dataset_file, 'rb') as f:
            archive = np.load(f)
            self.scan = archive['scan']
            self.data = archive['data']
            self.probe = archive['probe']

    def template_consistent_algorithm(self, algorithm, params={}):
        """Check ptycho.solver.algorithm for consistency."""

        result = {
            'probe': self.probe * np.random.rand(*self.probe.shape),
        }

        if params.get('use_mpi') is True:
            with MPIComm() as IO:
                result['probe'] = IO.Bcast(result['probe'])
                weights = params.get('eigen_weights')
                if weights is not None:
                    self.scan, self.data, params['eigen_weights'] = IO.MPIio(
                        self.scan,
                        self.data,
                        weights,
                    )
                else:
                    self.scan, self.data = IO.MPIio(self.scan, self.data)

        result['scan'] = self.scan

        result = tike.ptycho.reconstruct(
            **result,
            **params,
            data=self.data,
            algorithm=algorithm,
            num_iter=1,
        )
        params.update(result)
        result = tike.ptycho.reconstruct(
            **params,
            data=self.data,
            algorithm=algorithm,
            num_iter=32,
            # Only works when probe recovery is false because scaling
        )
        print()
        cost = '\n'.join(f'{c:1.3e}' for c in result['cost'])
        print(cost)
        return result

    def test_consistent_cgrad(self):
        """Check ptycho.solver.cgrad for consistency."""
        _save_ptycho_result(
            self.template_consistent_algorithm(
                'cgrad',
                params={
                    'subset_is_random': True,
                    'batch_size': int(self.data.shape[1] / 3),
                    'num_gpu': 2,
                    'probe_options': ProbeOptions(),
                    'object_options': ObjectOptions(),
                    'use_mpi': True,
                },
            ), f"{'mpi-' if _mpi_size > 1 else ''}cgrad")

    def test_consistent_lstsq_grad(self):
        """Check ptycho.solver.lstsq_grad for consistency."""
        _save_ptycho_result(
            self.template_consistent_algorithm(
                'lstsq_grad',
                params={
                    'subset_is_random':
                        True,
                    'batch_size':
                        int(self.data.shape[1] / 3),
                    'num_gpu':
                        2,
                    'probe_options':
                        ProbeOptions(),
                    'object_options':
                        ObjectOptions(),
                    'use_mpi':
                        True,
                    'position_options':
                        PositionOptions(
                            self.scan.shape[0:-1],
                            use_adaptive_moment=True,
                        ),
                },
            ), f"{'mpi-' if _mpi_size > 1 else ''}lstsq_grad")

    def test_consistent_lstsq_grad_variable_probe(self):
        """Check ptycho.solver.lstsq_grad for consistency."""

        eigen_probe, weights = tike.ptycho.probe.init_varying_probe(
            self.scan, self.probe, 3)

        _save_ptycho_result(
            self.template_consistent_algorithm(
                'lstsq_grad',
                params={
                    'subset_is_random':
                        True,
                    'batch_size':
                        int(self.data.shape[1] / 3),
                    'num_gpu':
                        2,
                    'probe_options':
                        ProbeOptions(),
                    'object_options':
                        ObjectOptions(),
                    'use_mpi':
                        True,
                    'eigen_probe':
                        eigen_probe,
                    'eigen_weights':
                        weights,
                    'position_options':
                        PositionOptions(
                            self.scan.shape[0:-1],
                            use_adaptive_moment=True,
                        ),
                },
            ), f"{'mpi-' if _mpi_size > 1 else ''}lstsq_grad-variable-probe")

    def test_invaid_algorithm_name(self):
        """Check that wrong names are handled gracefully."""
        with self.assertRaises(ValueError):
            self.template_consistent_algorithm('divided')


class TestProbe(unittest.TestCase):

    def test_eigen_probe(self):

        leading = (2,)
        wide = 18
        high = 21
        posi = 53
        eigen = 1
        comm = Comm(2, None)

        R = comm.pool.bcast(np.random.rand(*leading, posi, 1, 1, wide, high))
        eigen_probe = comm.pool.bcast(
            np.random.rand(*leading, 1, eigen, 1, wide, high))
        weights = np.random.rand(*leading, posi)
        weights -= np.mean(weights)
        weights = comm.pool.bcast(weights)
        patches = comm.pool.bcast(
            np.random.rand(*leading, posi, 1, 1, wide, high))
        diff = comm.pool.bcast(np.random.rand(*leading, posi, 1, 1, wide, high))

        new_probe, new_weights = tike.ptycho.probe.update_eigen_probe(
            comm=comm,
            R=R,
            eigen_probe=eigen_probe,
            weights=weights,
            patches=patches,
            diff=diff,
        )

        assert eigen_probe[0].shape == new_probe[0].shape


def _save_ptycho_result(result, algorithm):
    try:
        import matplotlib.pyplot as plt
        fname = os.path.join(testdir, 'result', f'{algorithm}')
        os.makedirs(fname, exist_ok=True)
        for i in range(len(result['psi'])):
            plt.imsave(
                f'{fname}/{i}-phase.png',
                np.angle(result['psi'][i]).astype('float32'),
            )
            plt.imsave(
                f'{fname}/{i}-ampli.png',
                np.abs(result['psi'][i]).astype('float32'),
            )
        for i in range(result['probe'].shape[-3]):
            plt.imsave(
                f'{fname}/{i}-probe-phase.png',
                np.angle(result['probe'][0, 0, 0, i]),
            )
            plt.imsave(
                f'{fname}/{i}-probe-ampli.png',
                np.abs(result['probe'][0, 0, 0, i]),
            )
    except ImportError:
        pass


if __name__ == '__main__':
    unittest.main()
