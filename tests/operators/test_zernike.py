#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import numpy as np
from tike.operators import Zernike
import tike.precision
import tike.linalg

from .util import random_complex, OperatorTests

__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2024, UChicago Argonne, LLC."
__docformat__ = "restructuredtext en"


class TestZernike(unittest.TestCase, OperatorTests):
    """Test the Zernike operator."""

    def setUp(self):
        self.nscan = 21
        self.nprobe = 11
        self.nbasis = 128
        self.size = 16
        self.degree_max, self.nbasis = tike.zernike.degree_from_num_coeffients(
            self.nbasis
        )

        basis = tike.zernike.basis(size=3, degree_min=0, degree_max=self.degree_max)
        assert basis.shape == (self.nbasis, 3, 3), basis.size

        self.operator = Zernike()
        self.operator.__enter__()
        self.xp = self.operator.xp

        np.random.seed(0)
        images = random_complex(self.nscan, self.nprobe, self.size, self.size)
        weights = random_complex(self.nscan, self.nprobe, self.nbasis)

        self.m = self.xp.asarray(weights)
        self.m_name = "weights"
        self.kwargs = {
            "size": self.size,
            "degree_max": self.degree_max,
        }

        self.d = self.xp.asarray(images)
        self.d_name = "images"

        print(self.operator)

    @unittest.skip('FIXME: This operator is not scaled.')
    def test_scaled(self):
        pass
