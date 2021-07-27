#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import numpy as np
from tike.operators import Rotate, Flow

from .util import random_complex, OperatorTests

__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'


class TestRotate(unittest.TestCase, OperatorTests):
    """Test the Rotate operator."""

    def setUp(self, shape=(7, 25, 53)):
        """Load a dataset for reconstruction."""

        self.operator = Rotate()
        self.operator.__enter__()
        self.xp = self.operator.xp

        np.random.seed(0)
        self.m = self.xp.asarray(random_complex(*shape), dtype='complex64')
        self.m_name = 'unrotated'
        self.d = self.xp.asarray(random_complex(*shape), dtype='complex64')
        self.d_name = 'rotated'
        self.kwargs = {
            'angle': np.random.rand() * 2 * np.pi,
        }
        print(self.operator)

    def debug_show(self, angle=19 * np.pi / 6):
        import libimage
        import matplotlib.pyplot as plt
        x = self.xp.asarray(libimage.load('coins', 256), dtype='complex64')
        y = self.operator.fwd(x[None], angle)
        flow = self.operator._make_flow(unrotated=x, angle=angle)
        y1 = Flow().fwd(x, flow)

        print(x.shape, y.shape, y1.shape)

        plt.figure()
        plt.imshow(x.real.get())

        plt.figure()
        plt.imshow(y[0].real.get())

        plt.figure()
        plt.imshow(y1.real.get())
        plt.show()

    @unittest.skip('FIXME: This operator is not scaled.')
    def test_scaled(self):
        pass


if __name__ == '__main__':
    unittest.main()
