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

__author__ = "Daniel Ching, Viktor Nikitin"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'

testdir = os.path.dirname(__file__)


class TestLaminoRecon(unittest.TestCase):
    """Test various laminography reconstruction methods for consistency."""

    def create_dataset(self, dataset_file):
        """Create a dataset for testing this module.
        Only called with setUp detects that `dataset_file` has been deleted.
        """
        import dxchange

        delta = dxchange.read_tiff(
            os.path.join(testdir, 'data/delta-chip-128.tiff'))[::4, ::4, ::4]
        beta = dxchange.read_tiff(
            os.path.join(testdir, 'data/beta-chip-128.tiff'))[::4, ::4, ::4]
        self.original = (delta + 1j * beta).astype('complex64')

        self.theta = np.linspace(0, 2 * np.pi, 16,
                                 endpoint=False).astype('float32')
        self.tilt = np.pi / 3

        self.data = tike.lamino.simulate(self.original, self.theta, self.tilt)
        assert self.data.shape == (16, 32, 32)
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
        data = tike.lamino.simulate(self.original, self.theta, self.tilt)
        assert data.dtype == 'complex64', data.dtype
        np.testing.assert_array_equal(data.shape, self.data.shape)
        np.testing.assert_allclose(data, self.data, atol=1e-6)

    def error_metric(self, x):
        """Return the error between two arrays."""
        return np.linalg.norm(x - self.original)

    def template_consistent_algorithm(self, algorithm):
        """Check lamino.solver.algorithm for consistency."""
        result = {
            'obj': np.zeros_like(self.original),
        }
        for _ in range(5):
            result = tike.lamino.reconstruct(
                **result,
                data=self.data,
                theta=self.theta,
                tilt=self.tilt,
                algorithm=algorithm,
                num_iter=1,
            )

        recon_file = os.path.join(testdir,
                                  f'data/lamino_{algorithm}.pickle.lzma')
        if os.path.isfile(recon_file):
            with lzma.open(recon_file, 'rb') as file:
                standard = pickle.load(file)
        else:
            print(result['obj'].shape)
            with lzma.open(recon_file, 'wb') as file:
                pickle.dump(result['obj'], file)
            raise FileNotFoundError(
                f"lamino '{algorithm}' standard not initialized.")
        np.testing.assert_array_equal(result['obj'].shape, self.original.shape)
        np.testing.assert_allclose(result['obj'], standard, atol=1e-3)

    def test_consistent_combined(self):
        """Check lamino.solver.cgrad for consistency."""
        self.template_consistent_algorithm('cgrad')


if __name__ == '__main__':
    unittest.main()
