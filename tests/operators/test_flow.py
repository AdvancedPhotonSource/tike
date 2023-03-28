#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import numpy as np
from tike.operators import Flow

from .util import random_complex, random_floating, OperatorTests

__author__ = "Daniel Ching, Viktor Nikitin"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'


class TestFlow(unittest.TestCase, OperatorTests):
    """Test the Flow operator."""

    def setUp(self, n=16, nz=17, ntheta=8):
        """Load a dataset for reconstruction."""

        self.operator = Flow()
        self.operator.__enter__()
        self.xp = self.operator.xp

        np.random.seed(0)
        self.m = self.xp.asarray(random_complex(ntheta, nz, n))
        self.m_name = 'f'
        self.d = self.xp.asarray(random_complex(*self.m.shape))
        self.d_name = 'g'
        self.kwargs = {
            'flow': self.xp.asarray(random_floating(*self.m.shape, 2) * 16),
        }
        print(self.operator)

    @unittest.skip('FIXME: This operator is not scaled.')
    def test_scaled(self):
        pass


if __name__ == '__main__':
    unittest.main()
