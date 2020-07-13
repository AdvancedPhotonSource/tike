#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import unittest

import numpy as np

from .util import random_complex, inner_complex
from tike.operators import Propagation

__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'


class TestPropagation(unittest.TestCase):
    """Test the Propagation operator."""

    def setUp(self):
        """Load a dataset for reconstruction."""
        self.nwaves = 13
        self.probe_shape = 127
        self.detector_shape = self.probe_shape

    def test_adjoint(self):
        """Check that the adjoint operator is correct."""
        np.random.seed(0)
        nearplane = random_complex(self.nwaves, self.probe_shape,
                                   self.probe_shape)
        farplane = random_complex(self.nwaves, self.detector_shape,
                                  self.detector_shape)

        with Propagation(
                nwaves=self.nwaves,
                detector_shape=self.detector_shape,
                probe_shape=self.probe_shape,
        ) as op:
            nearplane = op.asarray(nearplane, dtype='complex64')
            farplane = op.asarray(farplane, dtype='complex64')

            start = time.perf_counter()
            f = op.fwd(nearplane=nearplane,)
            fwd_time = time.perf_counter() - start
            assert f.shape == farplane.shape
            start = time.perf_counter()
            n = op.adj(farplane=farplane,)
            adj_time = time.perf_counter() - start
            assert nearplane.shape == n.shape
            a = inner_complex(nearplane, n)
            b = inner_complex(f, farplane)
            print()
            print(Propagation)
            print(f"{fwd_time:1.3e} seconds for fwd")
            print(f"{adj_time:1.3e} seconds for adj")
            print('<ψ , F*Ψ> = {:.6f}{:+.6f}j'.format(a.real.item(),
                                                      a.imag.item()))
            print('<Fψ,   Ψ> = {:.6f}{:+.6f}j'.format(b.real.item(),
                                                      b.imag.item()))
            # Test whether Adjoint fixed probe operator is correct
            op.xp.testing.assert_allclose(a.real, b.real, rtol=1e-5)
            op.xp.testing.assert_allclose(a.imag, b.imag, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
