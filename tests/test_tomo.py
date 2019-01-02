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

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import lzma
import numpy as np
import os
import pickle
import tike.tomo
import unittest

__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2018, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'


class TestPtychoRecon(unittest.TestCase):
    """Test various ptychography reconstruction methods for consistency."""

    def create_dataset(self, dataset_file):
        """Create a dataset for testing this module.

        Only called with setUp detects that `dataset_file` has been deleted.
        """
        import matplotlib.pyplot as plt
        # Create object
        delta = plt.imread("./tests/data/Cryptomeria_japonica-0128.tif")
        beta = plt.imread("./tests/data/Bombus_terrestris-0128.tif")
        original = np.empty(delta.shape, dtype=np.complex64)
        original.real = delta / 2550
        original.imag = beta / 2550
        self.original = np.tile(original, (1, 1, 1))
        # Define views
        self.theta = np.linspace(0, np.pi, 201, endpoint=False)
        # Simulate data
        self.data = tike.tomo.forward(obj=self.original, theta=self.theta)
        setup_data = [
            self.data.astype(np.complex64),
            self.theta.astype(np.float32),
            self.original.astype(np.complex64),
            ]
        with lzma.open(dataset_file, 'wb') as file:
            pickle.dump(setup_data, file)

    def setUp(self):
        """Load a dataset for reconstruction."""
        dataset_file = './tests/data/tomo_setup.pickle.lzma'
        if not os.path.isfile(dataset_file):
            self.create_dataset(dataset_file)
        with lzma.open(dataset_file, 'rb') as file:
            [
                self.data,
                self.theta,
                self.original,
            ] = pickle.load(file)

    def test_consistent_simulate(self):
        """Check tomo.forward for consistency."""
        data = tike.tomo.forward(obj=self.original, theta=self.theta)
        np.testing.assert_allclose(data, self.data, rtol=1e-3)

    def test_consistent_grad(self):
        """Check tomo.grad for consistency."""
        recon = np.zeros(self.original.shape, dtype=np.complex64)
        recon = tike.tomo.reconstruct(
            obj=recon,
            theta=self.theta,
            line_integrals=self.data,
            algorithm='grad',
            reg_par=-1,
            num_iter=10,
        )
        recon_file = './tests/data/tomo_grad.pickle.lzma'
        try:
            with lzma.open(recon_file, 'rb') as file:
                standard = pickle.load(file)
        except FileNotFoundError as e:
            with lzma.open(recon_file, 'wb') as file:
                pickle.dump(recon.astype(np.complex64), file)
            raise e
        np.testing.assert_allclose(recon, standard, rtol=1e-3)


# def test_forward_project_hv_quadrants1():
#     gmin = [0, 0, 0]
#     obj = np.zeros((2, 3, 2), dtype=complex)
#     obj[0, 1, 0] = 1
#     theta, h, v = [0, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 1]
#     pgrid = np.ones((1, 1))
#     integral = forward(obj, gmin,
#                        pgrid, theta, v, h)
#     truth = np.array([1, 0, 0, 0]).reshape(4, 1, 1)
#     np.testing.assert_equal(truth, integral)
#
#
# def test_forward_project_hv_quadrants2():
#     gsize = np.array([3, 2, 2])
#     gmin = -gsize / 2.0
#     obj = np.zeros((3, 2, 2), dtype=complex)
#     obj[2, 0, 1] = 1
#     theta, h, v = [0, 0], [-1, 0], [0.5, 0.5]
#     pgrid = np.ones((1, 1))
#     integral = forward(obj, gmin,
#                        pgrid, theta, v, h)
#     truth = np.array([0, 1]).reshape(2, 1, 1)
#     np.testing.assert_equal(truth, integral)
#
#
# def test_forward_project_hv_quadrants4():
#     gsize = np.array([1, 2, 2])
#     gmin = -gsize / 2.0
#     obj = np.zeros((1, 2, 2), dtype=complex)
#     obj[0, :, 1] = 1
#     theta, h, v = [0], [-0.5], [-0.5]
#     pgrid = np.ones((1, 1))
#     psize = (1, 1)
#     integral = forward(obj, gmin,
#                        pgrid, theta, v, h)
#     truth = np.array([2]).reshape(1, 1, 1)
#     np.testing.assert_equal(truth, integral)
