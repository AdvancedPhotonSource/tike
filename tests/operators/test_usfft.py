#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import numpy as np
from tike.operators.cupy.usfft import (eq2us, us2eq, vector_gather,
                                       vector_scatter)
from tike.operators import Operator

from .util import random_complex, OperatorTests

__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'


class Interp(Operator):

    def __init__(self, eps):
        self.eps = eps
        self.m = 7
        self.mu = 4.42341

    def fwd(self, f, x, n):
        return vector_gather(self.xp, f, x, n, self.m, self.mu)

    def adj(self, F, x, n):
        return vector_scatter(self.xp, F, x, n, self.m, self.mu)


class TestInterp(unittest.TestCase, OperatorTests):
    """Test the Interp operator."""

    def setUp(self, n=16, ntheta=32, eps=1e-6):
        self.operator = Interp(eps)
        self.operator.__enter__()
        self.xp = self.operator.xp
        np.random.seed(0)
        self.m = self.xp.asarray(random_complex(n, n, n), dtype='complex64')
        self.m_name = 'f'
        self.d = self.xp.asarray(random_complex(ntheta), dtype='complex64')
        self.d_name = 'F'
        self.kwargs = {
            'x':
                self.xp.asarray(np.random.rand(ntheta, 3) -
                                0.5).astype('float32'),
            'n':
                n,
        }
        print(self.operator)

    @unittest.skip('FIXME: This operator is not scaled.')
    def test_scaled(self):
        pass


class USFFT(Operator):

    def __init__(self, eps):
        self.eps = eps

    def fwd(self, f, x, n):
        return eq2us(f, x, n, self.eps, self.xp)

    def adj(self, F, x, n):
        return us2eq(F, -x, n, self.eps, self.xp)


class TestUSFFT(unittest.TestCase, OperatorTests):
    """Test the USFFT operator."""

    def setUp(self, n=16, ntheta=8, eps=1e-6):
        self.operator = USFFT(eps)
        self.operator.__enter__()
        self.xp = self.operator.xp
        np.random.seed(1)
        self.m = self.xp.asarray(random_complex(n, n, n), dtype='complex64')
        self.m_name = 'f'
        self.d = self.xp.asarray(random_complex(ntheta), dtype='complex64')
        self.d_name = 'F'
        self.kwargs = {
            'x':
                self.xp.asarray(np.random.rand(ntheta, 3) -
                                0.5).astype('float32'),
            'n':
                n,
        }
        print(self.operator)

    @unittest.skip('FIXME: This operator is not scaled.')
    def test_scaled(self):
        pass

    @unittest.skip('For debugging only.')
    def test_image(self, s=32, ntheta=16 * 16 * 16):
        import libimage
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt

        f = libimage.load('satyre', s)
        f = np.tile(f, (s, 1, 1))
        f = self.xp.asarray(f, dtype='complex64')

        x = [
            g.ravel() for g in np.meshgrid(
                np.linspace(-0.5, 0.5, s),
                np.linspace(-0.5, 0.5, s),
                np.linspace(-0.5, 0.5, s),
            )
        ]

        x = np.stack(x, -1)

        print(x.shape)

        x = self.xp.asarray(x, dtype='float32')

        d = self.operator.fwd(f, x, s)
        m = self.operator.adj(d, x, s)

        plt.figure()
        plt.imshow(m[s // 2].real.get())
        plt.figure()
        plt.imshow(f[s // 2].real.get())
        plt.show()


if __name__ == '__main__':
    unittest.main()
