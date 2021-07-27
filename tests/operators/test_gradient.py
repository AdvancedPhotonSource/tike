#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import numpy as np
from tike.operators import Gradient
import tike.random

from .util import random_complex, OperatorTests

__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2021, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'


class TestGradient(unittest.TestCase, OperatorTests):
    """Test the Gradient operator."""

    def setUp(self, shape=(8, 19, 5)):
        """Load a dataset for reconstruction."""

        self.operator = Gradient()
        self.operator.__enter__()
        self.xp = self.operator.xp

        np.random.seed(0)
        self.m = tike.random.cupy_complex(*shape)
        self.m_name = 'u'
        self.d = tike.random.cupy_complex(3, *shape)
        self.d_name = 'g'
        self.kwargs = {        }
        print(self.operator)

    @unittest.skip('FIXME: This operator is not scaled.')
    def test_scaled(self):
        pass


if __name__ == '__main__':
    unittest.main()
