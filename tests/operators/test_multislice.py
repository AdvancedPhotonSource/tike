#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import unittest

import numpy as np
from tike.operators import Multislice
from tike.operators.cupy.multislice import SingleSlice
import tike.precision
import tike.linalg

from .util import random_complex, OperatorTests

__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'


class TestMultislice(unittest.TestCase):
    """Test the ptychography operator."""

    def setUp(self, depth=7, pw=15, nscan=27):
        """Load a dataset for reconstruction."""
        self.nscan = nscan
        self.nprobe = 3
        self.probe_shape = (nscan, self.nprobe, pw, pw)
        self.detector_shape = (pw, pw)
        self.original_shape = (depth, 128, 128)
        self.scan_shape = (nscan, 2)
        print(Multislice)

        np.random.seed(0)
        scan = np.random.rand(*self.scan_shape).astype(
            tike.precision.floating) * (127 - 16)
        probe = random_complex(*self.probe_shape)
        original = random_complex(*self.original_shape)
        farplane = random_complex(*self.probe_shape[:-2], *self.detector_shape)

        self.operator = Multislice(
            nscan=self.scan_shape[-2],
            probe_shape=self.probe_shape[-1],
            detector_shape=self.detector_shape[-1],
            nz=self.original_shape[-2],
            n=self.original_shape[-1],
            nslices=depth,
        )
        self.operator.__enter__()
        self.xp = self.operator.xp

        self.mkwargs = {
            'probe': self.xp.asarray(probe),
            'psi': self.xp.asarray(original),
            'scan': self.xp.asarray(scan),
        }
        self.dkwargs = {
            'nearplane': self.xp.asarray(farplane),
            'probe': self.xp.asarray(probe),
            'psi': self.xp.asarray(original),
            'scan': self.xp.asarray(scan),
        }

    def test_adjoint(self):
        """Check that the adjoint operator is correct."""
        d = self.operator.fwd(**self.mkwargs)
        assert d.shape == self.dkwargs['nearplane'].shape
        m0, m1 = self.operator.adj(**self.dkwargs)
        assert m0.shape == self.mkwargs['psi'].shape
        assert m1.shape == self.mkwargs['probe'].shape
        a = tike.linalg.inner(d, self.dkwargs['nearplane'])
        b = tike.linalg.inner(self.mkwargs['psi'], m0)
        c = tike.linalg.inner(self.mkwargs['probe'], m1)
        print()
        print('<Fm,    d> = {:.5g}{:+.5g}j'.format(a.real.item(), a.imag.item()))
        print('< m0, F*d> = {:.5g}{:+.5g}j'.format(b.real.item(), b.imag.item()))
        print('< m1, F*d> = {:.5g}{:+.5g}j'.format(c.real.item(), c.imag.item()))
        # self.xp.testing.assert_allclose(a.real, b.real, rtol=1e-3, atol=0)
        # self.xp.testing.assert_allclose(a.imag, b.imag, rtol=1e-3, atol=0)

    @unittest.skip('FIXME: This operator is not scaled.')
    def test_scaled(self):
        pass


class TestSingleslice(unittest.TestCase):
    """Test the ptychography operator."""

    def setUp(self, depth=1, pw=15, nscan=27):
        """Load a dataset for reconstruction."""
        self.nscan = nscan
        self.nprobe = 3
        self.probe_shape = (nscan, self.nprobe, pw, pw)
        self.detector_shape = (pw, pw)
        self.original_shape = (depth, 128, 128)
        self.scan_shape = (nscan, 2)
        print(Multislice)

        np.random.seed(0)
        scan = np.random.rand(*self.scan_shape).astype(
            tike.precision.floating) * (127 - 16)
        probe = random_complex(*self.probe_shape)
        original = random_complex(*self.original_shape)
        farplane = random_complex(*self.probe_shape[:-2], *self.detector_shape)

        self.operator = SingleSlice(
            nscan=self.scan_shape[-2],
            probe_shape=self.probe_shape[-1],
            detector_shape=self.detector_shape[-1],
            nz=self.original_shape[-2],
            n=self.original_shape[-1],
            nslices=depth,
        )
        self.operator.__enter__()
        self.xp = self.operator.xp

        self.mkwargs = {
            'probe': self.xp.asarray(probe),
            'psi': self.xp.asarray(original),
            'scan': self.xp.asarray(scan),
        }
        self.dkwargs = {
            'nearplane': self.xp.asarray(farplane)
        }

    def test_adjoint(self):
        """Check that the adjoint operator is correct."""
        d = self.operator.fwd(**self.mkwargs)
        assert d.shape == self.dkwargs['nearplane'].shape
        m0, m1 = self.operator.adj(**self.mkwargs, **self.dkwargs)
        assert m0.shape == self.mkwargs['psi'].shape
        assert m1.shape == self.mkwargs['probe'].shape
        a = tike.linalg.inner(d, self.dkwargs['nearplane'])
        b = tike.linalg.inner(self.mkwargs['psi'], m0)
        c = tike.linalg.inner(self.mkwargs['probe'], m1)
        print()
        print('<Fm,  m0> = {:.5g}{:+.5g}j'.format(a.real.item(), a.imag.item()))
        print('<Fm,  m1> = {:.5g}{:+.5g}j'.format(c.real.item(), c.imag.item()))
        print('< d, F*d> = {:.5g}{:+.5g}j'.format(b.real.item(), b.imag.item()))
        self.xp.testing.assert_allclose(a.real, b.real, rtol=1e-3, atol=0)
        self.xp.testing.assert_allclose(a.imag, b.imag, rtol=1e-3, atol=0)

    @unittest.skip('FIXME: This operator is not scaled.')
    def test_scaled(self):
        pass


if __name__ == '__main__':
    unittest.main()
