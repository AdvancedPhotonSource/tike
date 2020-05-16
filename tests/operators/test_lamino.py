#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import numpy as np

from .util import random_complex, inner_complex
from tike.operators import Lamino

__author__ = "Daniel Ching, Viktor Nikitin"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'


class TestLamino(unittest.TestCase):
    """Test the Laminography operator."""

    def setUp(self, n=16, ntheta=8, tilt=np.pi/3, eps=1e-3):
        """Load a dataset for reconstruction."""
        self.n = n
        self.ntheta = ntheta
        self.theta = np.linspace(0, 2*np.pi, ntheta)
        self.tilt = tilt
        self.eps = eps
        print(Lamino)

    def test_adjoint(self):
        """Check that the adjoint operator is correct."""
        np.random.seed(0)
        obj = random_complex(self.n, self.n, self.n)
        data = random_complex(self.ntheta, self.n, self.n)

        with Lamino(
                n=self.n,
                theta=self.theta,
                tilt=self.tilt,
                eps=self.eps
        ) as op:

            obj = op.asarray(obj.astype('complex64'))
            data = op.asarray(data.astype('complex64'))

            d = op.fwd(obj)
            assert d.shape == data.shape
            o = op.adj(data)
            assert obj.shape == o.shape
            a = inner_complex(d, data)
            b = inner_complex(obj, o)
            print('<Lobj,   data> = {:.6f}{:+.6f}j'.format(
                a.real.item(), a.imag.item()))
            print('<obj  , L*data> = {:.6f}{:+.6f}j'.format(
                b.real.item(), b.imag.item()))
            # Test whether Adjoint fixed probe operator is correct
            op.xp.testing.assert_allclose(a.real, b.real, rtol=1e-2)
            op.xp.testing.assert_allclose(a.imag, b.imag, rtol=1e-2)


if __name__ == '__main__':
    unittest.main()
