#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import numpy as np
from tike.operators.cupy.usfft import eq2us, us2eq, vector_gather, vector_scatter
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


if __name__ == '__main__':
    unittest.main()
