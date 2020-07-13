#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import unittest

import numpy as np

from tike.operators import Shift
from .util import random_complex, inner_complex

__author__ = "Daniel Ching, Viktor Nikitin"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'


class TestShift(unittest.TestCase):
    """Test the Shift operator."""

    def setUp(self, n=16, nz=17, ntheta=8):
        """Load a dataset for reconstruction."""
        self.n = n
        self.nz = nz
        self.ntheta = ntheta

    def test_adjoint(self):
        """Check that the adjoint operator is correct."""
        np.random.seed(0)
        shift = np.random.random([self.ntheta, 2])
        original = random_complex(self.ntheta, self.nz, self.n)
        data = random_complex(*original.shape)

        with Shift() as op:

            shift = op.asarray(shift, dtype='float32')
            original = op.asarray(original, dtype='complex64')
            data = op.asarray(data, dtype='complex64')

            start = time.perf_counter()
            d = op.fwd(original, shift)
            fwd_time = time.perf_counter() - start
            o = op.fwd(data, -shift)
            original1 = op.fwd(d, -shift)

            a = inner_complex(d, data)
            b = inner_complex(original, o)
            e = np.linalg.norm(original - original1) / np.linalg.norm(original)
            print()
            print(Shift)
            print(f"{fwd_time:1.3e} seconds for fwd")
            print('<Su,   a> = {:.6f}{:+.6f}j'.format(a.real.item(),
                                                      a.imag.item()))
            print('< u, S*a> = {:.6f}{:+.6f}j'.format(b.real.item(),
                                                      b.imag.item()))
            print(f'|u - S*Su| / |u|= {e.item():.6f}')

            # Test whether Adjoint fixed probe operator is correct
            op.xp.testing.assert_allclose(a.real, b.real, rtol=1e-5)
            op.xp.testing.assert_allclose(a.imag, b.imag, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
