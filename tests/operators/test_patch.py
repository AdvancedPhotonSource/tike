#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import unittest

import numpy as np
from tike.operators import Patch

from .util import random_complex, inner_complex, OperatorTests

__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'


class TestPatch(unittest.TestCase, OperatorTests):
    """Test the Convolution operator."""

    def setUp(self):
        """Load a dataset for reconstruction."""

        self.ntheta = 10
        self.nscan = 73
        self.original_shape = (self.ntheta, 256, 256)
        self.probe_shape = 15
        self.detector_shape = self.probe_shape

        self.operator = Patch()
        self.operator.__enter__()
        self.xp = self.operator.xp

        # np.random.seed(0)
        scan = np.random.rand(self.ntheta, self.nscan, 2) * (
            self.original_shape[-1] - self.probe_shape - 2)
        original = random_complex(*self.original_shape)
        nearplane = random_complex(self.ntheta, self.nscan, self.detector_shape,
                                   self.detector_shape)

        # original /= np.linalg.norm(original)
        # nearplane /= np.linalg.norm(nearplane)

        self.m = self.xp.asarray(original, dtype='complex64')
        self.m_name = 'images'
        self.kwargs = {
            'positions': self.xp.asarray(scan, dtype='float32'),
            'patch_width': self.probe_shape,
            'height': self.original_shape[-2],
            'width': self.original_shape[-1],
        }

        self.d = self.xp.asarray(nearplane, dtype='complex64')
        self.d_name = 'patches'

        print(self.operator)

    @unittest.skip('FIXME: This operator is not scaled.')
    def test_scaled(self):
        pass


if __name__ == '__main__':
    unittest.main()
