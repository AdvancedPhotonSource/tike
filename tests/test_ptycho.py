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

import tike.ptycho

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
            os.path.join(testdir, "data/Cryptomeria_japonica-0128.png"))
        phase = plt.imread(
            os.path.join(testdir, "data/Bombus_terrestris-0128.png"))
        original = amplitude / 255 * np.exp(1j * phase / 255 * np.pi)
        self.original = np.expand_dims(original, axis=0).astype('complex64')

        pw = 15  # probe width
        weights = tike.ptycho.gaussian(pw, rin=0.8, rout=1.0)
        probe = weights * np.exp(1j * weights * 0.2)
        self.probe = np.expand_dims(probe, axis=0).astype('complex64')

        v, h = np.meshgrid(
            np.linspace(0, amplitude.shape[0]-pw, 13, endpoint=True),
            np.linspace(0, amplitude.shape[0]-pw, 13, endpoint=True),
            indexing='ij'
            )
        scan = np.stack((np.ravel(v), np.ravel(h)), axis=1)
        self.scan = np.expand_dims(scan, axis=0).astype('float32')

        self.data = tike.ptycho.simulate(
            detector_shape=pw * 2,
            probe=self.probe,
            scan=self.scan,
            psi=self.original
            )

        assert self.data.shape == (1, 13 * 13, pw * 2, pw * 2)
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

    def test_adjoint_operators(self):
        """Check that the adjoint operator is correct."""
        from tike.ptycho import PtychoBackend
        with PtychoBackend(
                nscan=self.scan.shape[-2],
                probe_shape=self.probe.shape[-1],
                detector_shape=self.data.shape[-1],
                nz=self.original.shape[-2],
                n=self.original.shape[-1],
                ntheta=1,
        ) as slv:
            np.random.seed(0)
            xp = slv.array_module
            scan = xp.array(self.scan)
            probe = xp.random.rand(*self.probe.shape, 2).astype('float32') - 0.5
            original = xp.random.rand(*self.original.shape, 2).astype('float32') - 0.5
            data = xp.random.rand(*self.data.shape, 2).astype('float32') - 0.5
            probe = probe.view('complex64')[..., 0]
            original = original.view('complex64')[..., 0]
            data = data.view('complex64')[..., 0]

            d = slv.fwd(
                probe=probe,
                scan=scan,
                psi=original,
            )
            o = slv.adj(
                farplane=data,
                probe=probe,
                scan=scan,
            )
            p = slv.adj_probe(
                farplane=data,
                scan=scan,
                psi=original,
            )
            a = xp.sum(original * xp.conj(o))
            b = xp.sum(data * xp.conj(d))
            c = xp.sum(probe * xp.conj(p))
            print()
            print('<FQP,     FQP> = {:.6f}{:+.6f}j'.format(
                a.real.item(), a.imag.item()))
            print('<P  , Q*F*FQP> = {:.6f}{:+.6f}j'.format(
                b.real.item(), b.imag.item()))
            print('<Q  , P*F*FPQ> = {:.6f}{:+.6f}j'.format(
                c.real.item(), c.imag.item()))
            # Test whether Adjoint fixed probe operator is correct
            xp.testing.assert_allclose(a.real, b.real, rtol=1e-5)
            # xp.testing.assert_allclose(a.imag, b.imag, rtol=1e-5)
            xp.testing.assert_allclose(a.real, c.real, rtol=1e-5)
            # xp.testing.assert_allclose(a.imag, c.imag, rtol=1e-5)

    def test_consistent_simulate(self):
        """Check ptycho.simulate for consistency."""
        data = tike.ptycho.simulate(
            detector_shape=self.data.shape[-1],
            probe=self.probe,
            scan=self.scan,
            psi=self.original,
        )
        assert data.dtype == 'float32', data.dtype
        assert self.data.dtype == 'float32', self.data.dtype
        np.testing.assert_array_equal(data.shape, self.data.shape)
        np.testing.assert_allclose(np.sqrt(data), np.sqrt(self.data), atol=1e-6)

    # def test_consistent_cgrad(self):
    #     """Check ptycho.cgrad for consistency."""
    #     result = tike.ptycho.reconstruct(
    #         data=self.data,
    #         probe=self.probe,
    #         scan=self.scan,
    #         psi=np.ones_like(self.original),
    #         algorithm='cgrad',
    #         num_iter=10,
    #         rho=0.5,
    #         gamma=0.25,
    #         reg=1+0j
    #         )
    #     recon_file = os.path.join(testdir, 'data/ptycho_cgrad.pickle.lzma')
    #     if os.path.isfile(recon_file):
    #         with lzma.open(recon_file, 'rb') as file:
    #             standard = pickle.load(file)
    #     else:
    #         with lzma.open(recon_file, 'wb') as file:
    #             pickle.dump(result['psi'], file)
    #         raise FileNotFoundError("ptycho.cgrad standard not initialized.")
    #     np.testing.assert_array_equal(result['psi'].shape, self.original.shape)
    #     np.testing.assert_allclose(result['psi'], standard, atol=1e-6)

    def test_adjoint_convolution(self):
        """Check that the diffraction adjoint operator is correct."""
        from tike.ptycho import PtychoBackend
        with PtychoBackend(
                nscan=self.scan.shape[-2],
                probe_shape=self.probe.shape[-1],
                detector_shape=self.data.shape[-1],
                nz=self.original.shape[-2],
                n=self.original.shape[-1],
                ntheta=1,
        ) as slv:
            xp = np
            np.random.seed(0)
            original = np.random.rand(*self.original.shape) + 1j * np.random.rand(*self.original.shape)
            data = np.random.rand(1, self.scan.shape[-2], self.probe.shape[-1], self.probe.shape[-1],) + 1j * np.random.rand(1, self.scan.shape[-2], self.probe.shape[-1], self.probe.shape[-1])

            original = original.astype('float32')
            data = data.astype('float32')

            d = slv.diffraction.fwd(
                scan=self.scan,
                psi=original,
            )
            o = slv.diffraction.adj(
                nearplane=data,
                scan=self.scan,
            )
            a = xp.sum(original * xp.conj(o))
            b = xp.sum(data * xp.conj(d))

            print()
            print('<Q, P*Q> = {:.6f}{:+.6f}j'.format(
                a.real.item(), a.imag.item()))
            print('<N,  QP> = {:.6f}{:+.6f}j'.format(
                b.real.item(), b.imag.item()))
            # Test whether Adjoint fixed probe operator is correct
            xp.testing.assert_allclose(a.real, b.real, rtol=1e-5)
            xp.testing.assert_allclose(a.imag, b.imag, rtol=1e-5)

    def test_adjoint_propagation(self):
        """Check that the adjoint operator is correct."""
        from tike.ptycho import PtychoBackend
        with PtychoBackend(
                nscan=self.scan.shape[-2],
                probe_shape=self.probe.shape[-1],
                detector_shape=self.data.shape[-1],
                nz=self.original.shape[-2],
                n=self.original.shape[-1],
                ntheta=1,
        ) as slv:
            np.random.seed(0)
            xp = slv.array_module

            nearplane = xp.random.rand(*self.data.shape[:-2], slv.probe_shape, slv.probe_shape, 2).astype('float32') - 0.5
            nearplane = nearplane.view('complex64')[..., 0]
            data = xp.random.rand(*self.data.shape, 2).astype('float32') - 0.5
            data = data.view('complex64')[..., 0]

            f = slv.propagation.fwd(
                nearplane=nearplane,
                farplane=data,
            )
            n = slv.propagation.adj(
                nearplane=nearplane,
                farplane=data,
            )

            a = xp.sum(nearplane * xp.conj(n))
            b = xp.sum(data * xp.conj(f))
            print()
            print('<N, F*D> = {:.6f}{:+.6f}j'.format(
                a.real.item(), a.imag.item()))
            print('<D,  FN> = {:.6f}{:+.6f}j'.format(
                b.real.item(), b.imag.item()))
            # Test whether Adjoint fixed probe operator is correct
            xp.testing.assert_allclose(a.real, b.real, rtol=1e-5)
            # xp.testing.assert_allclose(a.imag, b.imag, rtol=1e-5)

if __name__ == '__main__':
    unittest.main()
