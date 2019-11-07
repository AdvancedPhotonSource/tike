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
import os
import pickle
import unittest

import numpy as np

import tike.ptycho
from tike.ptycho import PtychoBackend
xp = PtychoBackend.array_module

__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2018, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'

testdir = os.path.dirname(__file__)


class TestPtychoUtils(unittest.TestCase):
    """Test various utility functions for correctness."""

    def test_gaussian(self):
        """Check ptycho.gaussian for correctness."""
        fname = os.path.join(testdir, 'data/ptycho_gaussian.pickle.lzma')
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
        amplitude = plt.imread(
            os.path.join(testdir, "data/Cryptomeria_japonica-0128.tif")
        )
        phase = plt.imread(
            os.path.join(testdir, "data/Bombus_terrestris-0128.tif")
        )
        original = amplitude / 255 * np.exp(1j * phase / 255 * np.pi)
        self.original = np.expand_dims(original, axis=0)

        pw = 15  # probe width
        weights = tike.ptycho.gaussian(pw, rin=0.8, rout=1.0)
        probe = weights * np.exp(1j * weights * 0.2)
        self.probe = np.expand_dims(probe, axis=0)

        v, h = np.meshgrid(
            np.linspace(0, amplitude.shape[0]-pw, 13, endpoint=True),
            np.linspace(0, amplitude.shape[0]-pw, 13, endpoint=True),
            indexing='ij'
            )
        scan = np.stack((np.ravel(v), np.ravel(h)), axis=1)
        self.scan = np.expand_dims(scan, axis=0)

        self.data = tike.ptycho.simulate(
            detector_shape=pw * 2,
            probe=self.probe,
            scan=self.scan,
            psi=self.original
            )

        assert self.data.shape == (1, 13 * 13, pw * 2, pw * 2)

        setup_data = [
            self.data.astype('float32'),
            self.scan.astype('float32'),
            self.probe.astype('complex64'),
            self.original.astype('complex64'),
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
            ] = [xp.array(x) for x in pickle.load(file)]

    def test_adjoint_operators(self):
        """Check that the adjoint operator is correct."""
        # Class gpu solver
        with PtychoBackend(
                nscan=self.scan.shape[-2],
                probe_shape=self.probe.shape[-1],
                detector_shape=self.data.shape[-1],
                nz=self.original.shape[-2],
                n=self.original.shape[-1],
                ntheta=1,
        ) as slv:
            t1 = slv.fwd(
                probe=self.probe,
                scan=self.scan,
                psi=self.original,
            )
            t2 = slv.adj(
                farplane=t1,
                probe=self.probe,
                scan=self.scan,
            )
            t3 = slv.adj_probe(
                farplane=t1,
                scan=self.scan,
                psi=self.original,
            )
            a = xp.sum(self.original * xp.conj(t2))
            b = xp.sum(t1 * xp.conj(t1))
            c = xp.sum(self.probe * xp.conj(t3))
            print()
            print('<FQP,     FQP> = {:.6f}{:+.6f}j'.format(a.real.item(), a.imag.item()))
            print('<P  , Q*F*FQP> = {:.6f}{:+.6f}j'.format(b.real.item(), b.imag.item()))
            print('<Q  , P*F*FPQ> = {:.6f}{:+.6f}j'.format(c.real.item(), c.imag.item()))
            # print('<FQP,FQP> - <P,Q*F*FQP> = ', a-b)
            # print('<FQP,FQP> - <Q,P*F*FPQ> = ', a-c)
            # Test whether Adjoint fixed probe operator is correct
            xp.testing.assert_allclose(a, b)
            xp.testing.assert_allclose(a, c)

    def test_consistent_simulate(self):
        """Check ptycho.simulate for consistency."""
        data = tike.ptycho.simulate(
            detector_shape=self.data.shape[-1],
            probe=self.probe,
            scan=self.scan,
            psi=self.original,
            )
        xp.testing.assert_array_equal(data.shape, self.data.shape)
        xp.testing.assert_allclose(data, self.data, rtol=1e-3)

    def test_consistent_cgrad(self):
        """Check ptycho.cgrad for consistency."""
        result = tike.ptycho.reconstruct(
            data=self.data,
            probe=self.probe,
            scan=self.scan,
            psi=xp.ones_like(self.original),
            algorithm='cgrad',
            num_iter=10,
            rho=0.5,
            gamma=0.25,
            reg=1+0j
            )
        new_psi = result['psi']
        recon_file = os.path.join(testdir, 'data/ptycho_cgrad.pickle.lzma')
        try:
            with lzma.open(recon_file, 'rb') as file:
                standard = pickle.load(file)
        except FileNotFoundError as e:
            with lzma.open(recon_file, 'wb') as file:
                pickle.dump(new_psi, file)
            raise e
        xp.testing.assert_array_equal(new_psi.shape, self.original.shape)
        xp.testing.assert_allclose(new_psi, standard)

if __name__ == '__main__':
  unittest.main()
