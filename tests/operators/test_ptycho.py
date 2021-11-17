#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import unittest

import numpy as np
from tike.operators import Ptycho

from .util import random_complex, inner_complex, OperatorTests

__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'


class TestPtycho(unittest.TestCase, OperatorTests):
    """Test the ptychography operator."""

    def setUp(self, ntheta=3, pw=15, nscan=27):
        """Load a dataset for reconstruction."""
        self.nscan = nscan
        self.ntheta = ntheta
        self.nprobe = 3
        self.probe_shape = (ntheta, nscan, 1, self.nprobe, pw, pw)
        self.detector_shape = (pw * 3, pw * 3)
        self.original_shape = (ntheta, 128, 128)
        self.scan_shape = (ntheta, nscan, 2)
        print(Ptycho)

        np.random.seed(0)
        scan = np.random.rand(*self.scan_shape).astype('float32') * (127 - 16)
        probe = random_complex(*self.probe_shape)
        original = random_complex(*self.original_shape)
        farplane = random_complex(*self.probe_shape[:-2], *self.detector_shape)

        self.operator = Ptycho(
            nscan=self.scan_shape[-2],
            probe_shape=self.probe_shape[-1],
            detector_shape=self.detector_shape[-1],
            nz=self.original_shape[-2],
            n=self.original_shape[-1],
            ntheta=self.ntheta,
        )
        self.operator.__enter__()
        self.xp = self.operator.xp

        probe = self.xp.asarray(probe.astype('complex64'))
        original = self.xp.asarray(original.astype('complex64'))
        farplane = self.xp.asarray(farplane.astype('complex64'))
        scan = self.xp.asarray(scan.astype('float32'))

        self.m = self.xp.asarray(original, dtype='complex64')
        self.m_name = 'psi'
        self.kwargs = {
            'scan': self.xp.asarray(scan, dtype='float32'),
            'probe': self.xp.asarray(probe, dtype='complex64')
        }

        self.m1 = self.xp.asarray(probe, dtype='complex64')
        self.m1_name = 'probe'
        self.kwargs1 = {
            'scan': self.xp.asarray(scan, dtype='float32'),
            'psi': self.xp.asarray(original, dtype='complex64')
        }
        self.kwargs2 = {
            'scan': self.xp.asarray(scan, dtype='float32'),
        }

        self.d = self.xp.asarray(farplane, dtype='complex64')
        self.d_name = 'farplane'

    def test_adjoint_probe(self):
        """Check that the adjoint operator is correct."""
        d = self.operator.fwd(**{self.m1_name: self.m1}, **self.kwargs1)
        assert d.shape == self.d.shape
        m = self.operator.adj_probe(**{self.d_name: self.d}, **self.kwargs1)
        assert m.shape == self.m1.shape
        a = inner_complex(d, self.d)
        b = inner_complex(self.m1, m)
        print()
        print('<Fm,   m> = {:.5g}{:+.5g}j'.format(a.real.item(), a.imag.item()))
        print('< d, F*d> = {:.5g}{:+.5g}j'.format(b.real.item(), b.imag.item()))
        self.xp.testing.assert_allclose(a.real, b.real, rtol=1e-5, atol=0)
        self.xp.testing.assert_allclose(a.imag, b.imag, rtol=1e-5, atol=0)

    def test_adj_probe_time(self):
        """Time the adjoint operation."""
        start = time.perf_counter()
        m = self.operator.adj_probe(**{self.d_name: self.d}, **self.kwargs1)
        elapsed = time.perf_counter() - start
        print(f"\n{elapsed:1.3e} seconds")

    @unittest.skip('FIXME: This operator is not scaled.')
    def test_scaled(self):
        pass

    def test_adjoint_all(self):
        """Check that the adjoint operator is correct."""
        d = self.operator.fwd(
            **{
                self.m_name: self.m,
                self.m1_name: self.m1
            },
            **self.kwargs2,
        )
        assert d.shape == self.d.shape
        m, m1 = self.operator.adj_all(
            **{
                self.d_name: self.d,
                self.m_name: self.m,
                self.m1_name: self.m1
            },
            **self.kwargs2,
        )
        assert m.shape == self.m.shape
        assert m1.shape == self.m1.shape
        a = inner_complex(d, self.d)
        b = inner_complex(self.m, m)
        c = inner_complex(self.m1, m1)
        print()
        print('< Fm,    m> = {:.5g}{:+.5g}j'.format(a.real.item(),
                                                    a.imag.item()))
        print('< d0, F*d0> = {:.5g}{:+.5g}j'.format(b.real.item(),
                                                    b.imag.item()))
        print('< d1, F*d1> = {:.5g}{:+.5g}j'.format(c.real.item(),
                                                    c.imag.item()))
        self.xp.testing.assert_allclose(a.real, b.real, rtol=1e-5, atol=0)
        self.xp.testing.assert_allclose(a.imag, b.imag, rtol=1e-5, atol=0)
        self.xp.testing.assert_allclose(a.real, c.real, rtol=1e-5, atol=0)
        self.xp.testing.assert_allclose(a.imag, c.imag, rtol=1e-5, atol=0)


if __name__ == '__main__':
    unittest.main()
