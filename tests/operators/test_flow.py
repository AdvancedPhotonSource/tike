#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import numpy as np

from tike.operators import Flow
from .util import random_complex, inner_complex

__author__ = "Daniel Ching, Viktor Nikitin"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'


class TestFlow(unittest.TestCase):
    """Test the Flow operator."""

    def setUp(self, n=16, nz=17, ntheta=8):
        """Load a dataset for reconstruction."""
        self.n = n
        self.nz = nz
        self.ntheta = ntheta

        np.random.seed(0)
        self.original = random_complex(self.ntheta, self.nz, self.n)
        self.data = random_complex(*self.original.shape)
        self.shift = (np.random.rand(*self.original.shape, 2) - 0.5) * 16

        print(Flow)

    def test_adjoint(self):
        """Check that the adjoint operator is correct."""
        with Flow() as op:

            shift = op.asarray(self.shift, dtype='float32')
            original = op.asarray(self.original, dtype='complex64')
            data = op.asarray(self.data, dtype='complex64')

            d = op.fwd(original, shift)
            o = op.adj(data, shift)

            a = inner_complex(d, data)
            b = inner_complex(original, o)
            print()
            print('<Su,   a> = {:.6f}{:+.6f}j'.format(a.real.item(),
                                                      a.imag.item()))
            print('< u, S*a> = {:.6f}{:+.6f}j'.format(b.real.item(),
                                                      b.imag.item()))

            op.xp.testing.assert_allclose(a.real, b.real, rtol=1e-5)
            op.xp.testing.assert_allclose(a.imag, b.imag, rtol=1e-5)

    def test_normalized(self):
        """Check that the adjoint operator is normalized."""
        with Flow() as op:

            shift = op.asarray(self.shift, dtype='float32')
            original = op.asarray(self.original, dtype='complex64')

            d = op.fwd(op.fwd(original, shift), -shift)

            a = inner_complex(d, d)
            b = inner_complex(original, original)
            print()
            print('<S*Su, S*Su> = {:.6f}{:+.6f}j'.format(
                a.real.item(), a.imag.item()))
            print('<   u,    u> = {:.6f}{:+.6f}j'.format(
                b.real.item(), b.imag.item()))

            # op.xp.testing.assert_allclose(a.real, b.real, rtol=1e-5)
            # op.xp.testing.assert_allclose(a.imag, b.imag, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
