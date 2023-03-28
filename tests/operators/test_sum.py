#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import numpy as np
from tike.operators import Operator

from .util import random_complex, OperatorTests

__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2023, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'


class Sum(Operator):

    def fwd(self, unsummed, shape):
        """Perform the forward operator."""
        return self.xp.sum(unsummed, keepdims=True)

    def adj(self, summed, shape):
        """Perform the adjoint operator."""
        return self.xp.ones_like(summed, shape=shape) * summed


class TestSum(unittest.TestCase, OperatorTests):
    """Test the Pad operator."""

    def setUp(self, shape=(7, 5, 5)):
        """Load a dataset for reconstruction."""

        self.operator = Sum()
        self.operator.__enter__()
        self.xp = self.operator.xp

        np.random.seed(0)
        self.m = self.xp.asarray(random_complex(*shape))
        self.m_name = 'unsummed'
        self.d = self.xp.asarray(random_complex(*(1, 1, 1)))
        self.d_name = 'summed'
        self.kwargs = {
            'shape': shape,
        }
        print(self.operator)

    @unittest.skip('FIXME: This operator is not scaled.')
    def test_scaled(self):
        pass


if __name__ == '__main__':
    unittest.main()
