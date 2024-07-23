#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import numpy as np
from tike.operators import Ptycho
import tike.precision
import tike.linalg

from .util import random_complex

__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."
__docformat__ = "restructuredtext en"


class TestPtycho(unittest.TestCase):
    """Test the ptychography operator."""

    def setUp(self, depth=1, pw=15, nscan=27):
        """Load a dataset for reconstruction."""
        self.nscan = nscan
        self.nprobe = 3
        self.probe_shape = (nscan, 1, self.nprobe, pw, pw)
        self.detector_shape = (pw * 3, pw * 3)
        self.original_shape = (depth, 128, 128)
        self.scan_shape = (nscan, 2)
        print(Ptycho)

        np.random.seed(0)
        scan = np.random.rand(*self.scan_shape).astype(tike.precision.floating) * (
            127 - 16
        )
        probe = random_complex(*self.probe_shape)
        original = random_complex(*self.original_shape)
        farplane = random_complex(*self.probe_shape[:-2], *self.detector_shape)

        self.operator = Ptycho(
            nscan=self.scan_shape[-2],
            probe_shape=self.probe_shape[-1],
            detector_shape=self.detector_shape[-1],
            nz=self.original_shape[-2],
            n=self.original_shape[-1],
        )
        self.operator.__enter__()
        self.xp = self.operator.xp

        self.mkwargs = {
            "scan": self.xp.asarray(scan),
            "probe": self.xp.asarray(probe),
            "psi": self.xp.asarray(original),
        }
        self.dkwargs = {
            "farplane": self.xp.asarray(farplane),
        }

    def test_adjoint(self):
        """Check that the adjoint operator is correct."""
        d = self.operator.fwd(**self.mkwargs)
        assert d.shape == self.dkwargs["farplane"].shape
        m0, m1 = self.operator.adj(**self.dkwargs, **self.mkwargs)
        assert m0.shape == self.mkwargs["psi"].shape
        assert m1.shape == self.mkwargs["probe"].shape
        a = tike.linalg.inner(d, self.dkwargs["farplane"])
        b = tike.linalg.inner(self.mkwargs["psi"], m0)
        c = tike.linalg.inner(self.mkwargs["probe"], m1)
        print()
        print("<Fm,    d> = {:.5g}{:+.5g}j".format(a.real.item(), a.imag.item()))
        print("< m0, F*d> = {:.5g}{:+.5g}j".format(b.real.item(), b.imag.item()))
        print("< m1, F*d> = {:.5g}{:+.5g}j".format(c.real.item(), c.imag.item()))
        self.xp.testing.assert_allclose(a.real, b.real, rtol=1e-3, atol=0)
        self.xp.testing.assert_allclose(a.imag, b.imag, rtol=1e-3, atol=0)
        self.xp.testing.assert_allclose(a.real, c.real, rtol=1e-3, atol=0)
        self.xp.testing.assert_allclose(a.imag, c.imag, rtol=1e-3, atol=0)

    @unittest.skip("FIXME: This operator is not scaled.")
    def test_scaled(self):
        pass


if __name__ == "__main__":
    unittest.main()
