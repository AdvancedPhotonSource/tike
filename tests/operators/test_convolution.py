#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import unittest

import numpy as np
from tike.operators import Convolution

from .util import random_complex, inner_complex, OperatorTests

__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'


class TestConvolution(unittest.TestCase, OperatorTests):
    """Test the Convolution operator."""

    def setUp(self):
        """Load a dataset for reconstruction."""

        self.ntheta = 3
        self.nscan = 27
        self.nprobe = 3
        self.original_shape = (self.ntheta, 128, 128)
        self.probe_shape = 15
        self.detector_shape = self.probe_shape * 3

        self.operator = Convolution(
            ntheta=self.ntheta,
            nscan=self.nscan,
            nz=self.original_shape[-2],
            n=self.original_shape[-1],
            probe_shape=self.probe_shape,
            detector_shape=self.detector_shape,
        )
        self.operator.__enter__()
        self.xp = self.operator.xp

        np.random.seed(0)
        scan = np.random.rand(self.ntheta, self.nscan, 2) * (127 - 15 - 1)
        original = random_complex(*self.original_shape)
        nearplane = random_complex(self.ntheta, self.nscan, self.nprobe,
                                   self.detector_shape, self.detector_shape)
        kernel = random_complex(self.ntheta, self.nscan, self.nprobe,
                                self.probe_shape, self.probe_shape)

        self.m = self.xp.asarray(original, dtype='complex64')
        self.m_name = 'psi'
        self.kwargs = {
            'scan': self.xp.asarray(scan, dtype='float32'),
            'probe': self.xp.asarray(kernel, dtype='complex64')
        }

        self.m1 = self.xp.asarray(kernel, dtype='complex64')
        self.m1_name = 'probe'
        self.kwargs1 = {
            'scan': self.xp.asarray(scan, dtype='float32'),
            'psi': self.xp.asarray(original, dtype='complex64')
        }
        self.kwargs2 = {
            'scan': self.xp.asarray(scan, dtype='float32'),
        }

        self.d = self.xp.asarray(nearplane, dtype='complex64')
        self.d_name = 'nearplane'

        print(self.operator)

    def test_adjoint_probe(self):
        """Check that the adjoint operator is correct."""
        d = self.operator.fwd(**{self.m1_name: self.m1}, **self.kwargs1)
        assert d.shape == self.d.shape
        m = self.operator.adj_probe(**{self.d_name: self.d}, **self.kwargs1)
        assert m.shape == self.m1.shape
        a = inner_complex(d, self.d)
        b = inner_complex(self.m1, m)
        print()
        print('<Fm,   m> = {:.6f}{:+.6f}j'.format(a.real.item(), a.imag.item()))
        print('< d, F*d> = {:.6f}{:+.6f}j'.format(b.real.item(), b.imag.item()))
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
        print('< Fm,    m> = {:.6f}{:+.6f}j'.format(a.real.item(),
                                                    a.imag.item()))
        print('< d0, F*d0> = {:.6f}{:+.6f}j'.format(b.real.item(),
                                                    b.imag.item()))
        print('< d1, F*d1> = {:.6f}{:+.6f}j'.format(c.real.item(),
                                                    c.imag.item()))
        self.xp.testing.assert_allclose(a.real, b.real, rtol=1e-5, atol=0)
        self.xp.testing.assert_allclose(a.imag, b.imag, rtol=1e-5, atol=0)
        self.xp.testing.assert_allclose(a.real, c.real, rtol=1e-5, atol=0)
        self.xp.testing.assert_allclose(a.imag, c.imag, rtol=1e-5, atol=0)


if __name__ == '__main__':
    unittest.main()
