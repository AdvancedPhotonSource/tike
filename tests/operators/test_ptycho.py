#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import numpy as np

from .util import random_complex, inner_complex
from tike.ptycho import PtychoBackend

__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'


class TestPtycho(unittest.TestCase):
    """Test the ptychography operator."""

    def setUp(self, ntheta=3, pw=15, nscan=27, fly=9, nmode=5):
        """Load a dataset for reconstruction."""
        self.nscan = nscan
        self.ntheta = ntheta
        self.probe_shape = (ntheta, nscan // fly, fly, nmode, pw, pw)
        self.detector_shape = (pw * 3, pw * 3)
        self.original_shape = (ntheta, 128, 128)
        self.scan_shape = (ntheta, nscan, 2)
        self.fly = fly
        self.nmode = nmode
        print(PtychoBackend)

    def test_adjoint(self):
        """Check that the adjoint operator is correct."""
        np.random.seed(0)
        scan = np.random.rand(*self.scan_shape).astype('float32') * 127 - 15
        probe = random_complex(*self.probe_shape)
        original = random_complex(*self.original_shape)
        farplane = random_complex(*self.probe_shape[:-2], *self.detector_shape)

        probe = probe.astype('complex64')
        original = original.astype('complex64')
        farplane = farplane.astype('complex64')

        with PtychoBackend(
                nscan=self.scan_shape[-2],
                probe_shape=self.probe_shape[-1],
                detector_shape=self.detector_shape[-1],
                nz=self.original_shape[-2],
                n=self.original_shape[-1],
                ntheta=self.ntheta,
                fly=self.fly,
                nmode=self.nmode,
        ) as op:
            d = op.fwd(
                probe=probe,
                scan=scan,
                psi=original,
            )
            assert d.shape == farplane.shape
            o = op.adj(
                farplane=farplane,
                probe=probe,
                scan=scan,
            )
            assert original.shape == o.shape
            p = op.adj_probe(
                farplane=farplane,
                scan=scan,
                psi=original,
            )
            assert probe.shape == p.shape
            a = inner_complex(d, farplane)
            b = inner_complex(probe, p)
            c = inner_complex(original, o)
            print()
            print('<FQP,     Ψ> = {:.6f}{:+.6f}j'.format(
                a.real.item(), a.imag.item()))
            print('<P  , Q*F*Ψ> = {:.6f}{:+.6f}j'.format(
                b.real.item(), b.imag.item()))
            print('<Q  , P*F*Ψ> = {:.6f}{:+.6f}j'.format(
                c.real.item(), c.imag.item()))
            # Test whether Adjoint fixed probe operator is correct
            np.testing.assert_allclose(a.real, b.real, rtol=1e-5)
            np.testing.assert_allclose(a.imag, b.imag, rtol=1e-5)
            np.testing.assert_allclose(a.real, c.real, rtol=1e-5)
            np.testing.assert_allclose(a.imag, c.imag, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
