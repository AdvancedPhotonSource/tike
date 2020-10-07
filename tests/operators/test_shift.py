#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import numpy as np
from tike.operators import Shift

from .util import random_complex, OperatorTests

__author__ = "Daniel Ching, Viktor Nikitin"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'


class TestShift(unittest.TestCase, OperatorTests):
    """Test the Shift operator."""

    def setUp(self, n=16, nz=17, ntheta=8):
        self.operator = Shift()
        self.operator.__enter__()
        self.xp = self.operator.xp
        np.random.seed(0)
        self.m = self.xp.asarray(random_complex(ntheta, nz, n),
                                 dtype='complex64')
        self.m_name = 'a'
        self.d = self.xp.asarray(random_complex(*self.m.shape),
                                 dtype='complex64')
        self.d_name = 'a'
        self.kwargs = {
            'shift':
                self.xp.asarray((np.random.random([ntheta, 2]) - 0.5) * 7,
                                dtype='float32')
        }
        print(self.operator)


if __name__ == '__main__':
    unittest.main()
