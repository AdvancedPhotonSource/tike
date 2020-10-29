#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import numpy as np
from tike.operators import Pad

from .util import random_complex, OperatorTests

__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'


class TestPad(unittest.TestCase, OperatorTests):
    """Test the Pad operator."""

    def setUp(self, shape=(7, 5, 5)):
        """Load a dataset for reconstruction."""

        self.operator = Pad()
        self.operator.__enter__()
        self.xp = self.operator.xp

        padded_shape = shape + np.asarray((0, 41, 32))
        corner = self.xp.asarray(np.random.randint(0, 32, size=(shape[0], 2)))

        np.random.seed(0)
        self.m = self.xp.asarray(random_complex(*shape), dtype='complex64')
        self.m_name = 'unpadded'
        self.d = self.xp.asarray(random_complex(*padded_shape),
                                 dtype='complex64')
        self.d_name = 'padded'
        self.kwargs = {
            'corner': corner,
            'padded_shape': padded_shape,
            'unpadded_shape': shape,
        }
        print(self.operator)


class TestPadDefaults(unittest.TestCase, OperatorTests):
    """Test the Pad operator."""

    def setUp(self, shape=(7, 5, 5)):
        """Load a dataset for reconstruction."""

        self.operator = Pad()
        self.operator.__enter__()
        self.xp = self.operator.xp

        padded_shape = shape

        np.random.seed(0)
        self.m = self.xp.asarray(random_complex(*shape), dtype='complex64')
        self.m_name = 'unpadded'
        self.d = self.xp.asarray(random_complex(*padded_shape),
                                 dtype='complex64')
        self.d_name = 'padded'
        self.kwargs = {
            'corner': None,
            'padded_shape': None,
            'unpadded_shape': None,
        }
        print(self.operator)


if __name__ == '__main__':
    unittest.main()
