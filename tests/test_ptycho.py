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
import tike.ptycho
import unittest

__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2018, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'


class TestPtychoUtils(unittest.TestCase):
    """Test various utility functions for correctness."""

    def test_gaussian(self):
        """Check ptycho.gaussian for correctness."""
        fname = './tests/data/ptycho_gaussian.pickle.lzma'
        weights = tike.ptycho.gaussian(15, rin=0.8, rout=1.0)
        if os.path.isfile(fname):
            with lzma.open(fname, 'rb') as file:
                truth = pickle.load(file)
        else:
            with lzma.open(fname, 'wb') as file:
                truth = pickle.dump(weights, file)
        np.testing.assert_array_equal(weights, truth)


class TestPtychoRecon(unittest.TestCase):
    """Test various ptychography reconstruction methods for consistency."""

    def create_dataset(self, dataset_file):
        """Create a dataset for testing this module.

        Only called with setUp detects that `dataset_file` has been deleted.
        """
        import matplotlib.pyplot as plt
        amplitude = plt.imread("./tests/data/Cryptomeria_japonica-0128.tif")
        phase = plt.imread("./tests/data/Bombus_terrestris-0128.tif")
        self.original = amplitude / 255 * np.exp(1j * phase / 255 * np.pi)

        pw = 15  # probe width
        weights = tike.ptycho.gaussian(pw, rin=0.8, rout=1.0)
        self.probe = weights * np.exp(1j * weights * 0.2)

        self.v, self.h = np.meshgrid(
            np.linspace(0, amplitude.shape[0]-pw, 13, endpoint=False),
            np.linspace(0, amplitude.shape[0]-pw, 13, endpoint=False),
            indexing='ij'
            )

        self.data_shape = np.ones(2, dtype=int) * pw * 2

        self.data = tike.ptycho.simulate(
            data_shape=self.data_shape,
            probe=self.probe,
            v=self.v,
            h=self.h,
            psi=self.original
            )

        setup_data = [
            self.data,
            self.data_shape,
            self.v,
            self.h,
            self.probe,
            self.original,
            ]

        with lzma.open(dataset_file, 'wb') as file:
            pickle.dump(setup_data, file)

    def setUp(self):
        """Load a dataset for reconstruction."""
        dataset_file = './tests/data/ptycho_setup.pickle.lzma'
        if not os.path.isfile(dataset_file):
            self.create_dataset(dataset_file)
        with lzma.open(dataset_file, 'rb') as file:
            [
                self.data,
                self.data_shape,
                self.v,
                self.h,
                self.probe,
                self.original,
            ] = pickle.load(file)

    def test_consistent_simulate(self):
        """Check ptycho.simulate for consistency."""
        data = tike.ptycho.simulate(
            data_shape=self.data_shape,
            probe=self.probe,
            v=self.v,
            h=self.h,
            psi=self.original
            )
        np.testing.assert_allclose(data, self.data, rtol=1e-3)

    def test_consistent_grad(self):
        """Check ptycho.grad for consistency."""
        new_psi = tike.ptycho.reconstruct(
            data=self.data,
            probe=self.probe,
            v=self.v,
            h=self.h,
            psi=np.ones_like(self.original),
            algorithm='grad',
            niter=10,
            rho=0.5,
            gamma=0.25
            )
        recon_file = './tests/data/ptycho_grad.pickle.lzma'
        try:
            with lzma.open(recon_file, 'rb') as file:
                standard = pickle.load(file)
        except FileNotFoundError as e:
            with lzma.open(recon_file, 'wb') as file:
                pickle.dump(new_psi, file)
            raise e
        np.testing.assert_allclose(new_psi, standard, rtol=1e-3)
