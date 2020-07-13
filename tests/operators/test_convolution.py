#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import unittest

import numpy as np

from .util import random_complex, inner_complex
from tike.operators import Convolution

__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'


class TestConvolution(unittest.TestCase):
    """Test the Convolution operator."""

    def setUp(self):
        """Load a dataset for reconstruction."""
        self.ntheta = 3
        self.nscan = 27
        self.original_shape = (self.ntheta, 128, 128)
        self.probe_shape = 15
        self.detector_shape = self.probe_shape * 3
        self.fly = 9

    def test_adjoint(self):
        """Check that the diffraction adjoint operator is correct."""
        np.random.seed(0)
        scan = np.random.rand(self.ntheta, self.nscan, 2) * (127 - 15 - 1)
        original = random_complex(*self.original_shape)
        nearplane = random_complex(self.ntheta, self.nscan // self.fly,
                                   self.fly, 1, self.detector_shape,
                                   self.detector_shape)
        kernel = random_complex(self.ntheta, self.nscan // self.fly, self.fly,
                                1, self.probe_shape, self.probe_shape)

        with Convolution(
                ntheta=self.ntheta,
                nscan=self.nscan,
                nz=self.original_shape[-2],
                n=self.original_shape[-1],
                probe_shape=self.probe_shape,
                detector_shape=self.detector_shape,
                fly=self.fly,
        ) as op:
            scan = op.asarray(scan.astype('float32'))
            original = op.asarray(original.astype('complex64'))
            nearplane = op.asarray(nearplane.astype('complex64'))
            kernel = op.asarray(kernel.astype('complex64'))

            start = time.perf_counter()
            d = op.fwd(scan=scan, psi=original, probe=kernel)
            fwd_time = time.perf_counter() - start
            assert nearplane.shape == d.shape
            start = time.perf_counter()
            o = op.adj(
                nearplane=nearplane,
                scan=scan,
                probe=kernel,
            )
            adj_time = time.perf_counter() - start
            assert original.shape == o.shape
            start = time.perf_counter()
            k = op.adj_probe(
                scan=scan,
                psi=original,
                nearplane=nearplane,
            )
            adj_prb_time = time.perf_counter() - start
            assert kernel.shape == k.shape
            a = inner_complex(original, o)
            b = inner_complex(d, nearplane)
            c = inner_complex(kernel, k)
            print()
            print(Convolution)
            print(f"{fwd_time:1.3e} seconds for fwd")
            print(f"{adj_time:1.3e} seconds for adj")
            print(f"{adj_prb_time:1.3e} seconds for adj_prb")
            print('<Q , P*ψ> = {:.6f}{:+.6f}j'.format(a.real.item(),
                                                      a.imag.item()))
            print('<QP,   ψ> = {:.6f}{:+.6f}j'.format(b.real.item(),
                                                      b.imag.item()))
            print('<P , Q*ψ> = {:.6f}{:+.6f}j'.format(c.real.item(),
                                                      c.imag.item()))
            # Test whether Adjoint fixed probe operator is correct
            op.xp.testing.assert_allclose(a.real, b.real, rtol=1e-5)
            op.xp.testing.assert_allclose(a.imag, b.imag, rtol=1e-5)
            op.xp.testing.assert_allclose(a.real, c.real, rtol=1e-5)
            op.xp.testing.assert_allclose(a.imag, c.imag, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
