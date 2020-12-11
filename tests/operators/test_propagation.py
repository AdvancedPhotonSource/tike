#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import numpy as np
from tike.operators import Propagation

from .util import random_complex, OperatorTests

__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'


class TestPropagation(unittest.TestCase, OperatorTests):
    """Test the Propagation operator."""

    def setUp(self, nwaves=13, probe_shape=127):
        """Load a dataset for reconstruction."""
        self.operator = Propagation(
            nwaves=nwaves,
            detector_shape=probe_shape,
            probe_shape=probe_shape,
        )
        self.operator.__enter__()
        self.xp = self.operator.xp
        np.random.seed(0)
        self.m = self.xp.asarray(random_complex(nwaves, probe_shape,
                                                probe_shape),
                                 dtype='complex64')
        self.m_name = 'nearplane'
        self.d = self.xp.asarray(random_complex(nwaves, probe_shape,
                                                probe_shape),
                                 dtype='complex64')
        self.d_name = 'farplane'
        self.kwargs = {}
        print(self.operator)


if __name__ == '__main__':
    unittest.main()
