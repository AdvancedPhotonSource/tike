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
import numpy as np
import os
import pickle
# import tike.tomo
import unittest

__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2018, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'

testdir = os.path.dirname(__file__)


@unittest.skip(reason="The tomo module is broken/disabled.")
class TestTomoRecon(unittest.TestCase):
    """Test various tomography reconstruction methods for consistency."""

    def create_dataset(self, dataset_file):
        """Create a dataset for testing this module.

        Only called with setUp detects that `dataset_file` has been deleted.
        """
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        # Create object
        delta = plt.imread(
            os.path.join(testdir, "data/Cryptomeria_japonica-0128.tif"))
        beta = plt.imread(
            os.path.join(testdir, "data/Bombus_terrestris-0128.tif"))
        original = np.empty(delta.shape, dtype='complex64')
        original.real = delta / 2550
        original.imag = beta / 2550
        self.original = np.tile(original, (1, 1, 1)).astype('complex64')
        # Define views
        self.theta = np.linspace(0, np.pi, 201,
                                 endpoint=False).astype('float32')
        # Simulate data
        self.data = tike.tomo.simulate(obj=self.original, theta=self.theta)
        setup_data = [
            self.data.astype('complex64'),
            self.theta.astype('float32'),
            self.original.astype('complex64'),
        ]
        with lzma.open(dataset_file, 'wb') as file:
            pickle.dump(setup_data, file)

    def setUp(self):
        """Load a dataset for reconstruction."""
        dataset_file = os.path.join(testdir, 'data/tomo_setup.pickle.lzma')
        if not os.path.isfile(dataset_file):
            self.create_dataset(dataset_file)
        with lzma.open(dataset_file, 'rb') as file:
            [
                self.data,
                self.theta,
                self.original,
            ] = pickle.load(file)

    def test_adjoint(self):
        """Check that the tomo operators meet adjoint definition."""
        from tike.tomo import TomoBackend
        xp = TomoBackend.array_module
        theta = xp.array(self.theta)
        original = xp.array(self.original)
        with TomoBackend(
                ntheta=theta.size,
                nz=original.shape[0],
                n=original.shape[1],
                center=original.shape[1] / 2,
        ) as slv:
            data = slv.fwd(original, theta)
            u1 = slv.adj(data, theta)
            t1 = np.sum(data * xp.conj(data))
            t2 = np.sum(original * xp.conj(u1))
            print()
            print(f"<> = {t1.real.item():06f}{t1.imag.item():+06f}j\n"
                  f"<> = {t2.real.item():06f}{t2.imag.item():+06f}j")
            xp.testing.assert_allclose(t1, t2, rtol=1e-3)

    def test_consistent_simulate(self):
        """Check tomo.forward for consistency."""
        data = tike.tomo.simulate(obj=self.original, theta=self.theta)
        np.testing.assert_allclose(data, self.data, rtol=1e-3)

    def test_simple_integrals(self):
        """Check that the fwd tomo operator sums correctly at 0 and PI/2.

        When we project at angles 0 and PI/2, the foward operator should be the
        same as taking the sum over the object array along each axis.
        """
        theta = np.array([0, np.pi / 2, np.pi, -np.pi / 2], dtype='float32')
        size = 11
        original = np.zeros((1, size, size), dtype='complex64')
        original[0, size // 2, :] += 1
        original[0, :, size // 2] += 1j
        data = tike.tomo.simulate(original, theta)
        data1 = np.sum(original, axis=1)
        data2 = np.sum(original, axis=2)

        np.testing.assert_allclose(data[0], data1, atol=1e-6)
        np.testing.assert_allclose(data[1], data2, atol=1e-6)
        np.testing.assert_allclose(data[2], data1, atol=1e-6)
        np.testing.assert_allclose(data[3], data2, atol=1e-6)

    # def test_consistent_grad(self):
    #     """Check tomo.grad for consistency."""
    #     recon = np.zeros(self.original.shape, dtype=np.complex64)
    #     recon = tike.tomo.reconstruct(
    #         obj=recon,
    #         theta=self.theta,
    #         line_integrals=self.data,
    #         algorithm='grad',
    #         reg_par=-1,
    #         num_iter=10,
    #     )
    #     recon_file = os.path.join(testdir, 'data/tomo_grad.pickle.lzma')
    #     try:
    #         with lzma.open(recon_file, 'rb') as file:
    #             standard = pickle.load(file)
    #     except FileNotFoundError as e:
    #         with lzma.open(recon_file, 'wb') as file:
    #             pickle.dump(recon.astype(np.complex64), file)
    #         raise e
    #     np.testing.assert_allclose(recon, standard, rtol=1e-3)


if __name__ == '__main__':
    unittest.main()
