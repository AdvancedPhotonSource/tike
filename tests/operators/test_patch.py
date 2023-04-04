#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import cupy as cp
import numpy as np
from tike.operators import Patch
import tike.precision
import tike.linalg

from .util import random_complex, OperatorTests

__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'


class TestPatch(unittest.TestCase, OperatorTests):
    """Test the Convolution operator."""

    def setUp(self):
        """Load a dataset for reconstruction."""

        self.ntheta = 3
        self.nscan = 7
        self.original_shape = (self.ntheta, 256, 256)
        self.probe_shape = 16
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

        self.m = self.xp.asarray(original)
        self.m_name = 'images'
        self.kwargs = {
            'positions': self.xp.asarray(scan, dtype=tike.precision.floating),
            'patch_width': self.probe_shape,
            'height': self.original_shape[-2],
            'width': self.original_shape[-1],
        }

        self.d = self.xp.asarray(nearplane)
        self.d_name = 'patches'

        print(self.operator)

    @unittest.skip('FIXME: This operator is not scaled.')
    def test_scaled(self):
        pass


def test_patch_correctness(size=256, win=8):

    try:
        import libimage
        import tike.precision
        fov = (libimage.load('coins', size) +
               1j * libimage.load('earring', size))
    except ModuleNotFoundError:
        import tike.random
        fov = tike.random.numpy_complex(
            size,
            size,
        )
    fov = fov.astype(tike.precision.cfloating)

    subpixel = 0.12346789
    positions = np.array([
        [0, 0],
        [0, size - win],
        [size - win, 0],
        [size - win, size - win],
        [size // 2 - win // 2, size // 2 - win // 2],
        [subpixel, 3],
    ],
                         dtype=tike.precision.floating)
    truth = np.stack(
        (
            fov[:win, :win],
            fov[:win, -win:],
            fov[-win:, :win],
            fov[-win:, -win:],
            fov[size // 2 - win // 2:size // 2 - win // 2 + win,
                size // 2 - win // 2:size // 2 - win // 2 + win],
            (1.0 - subpixel) * fov[0:win, 3:3 + win] +
            subpixel * fov[1:1 + win, 3:3 + win],
        ),
        axis=0,
    )
    with Patch() as op:
        patches = op.fwd(
            images=cp.array(fov),
            positions=cp.array(positions),
            patch_width=win,
        ).get()

    try:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        plt.figure()
        plt.imshow(fov.real)
        plt.savefig('fov.png')
        plt.figure()
        for i in range(len(positions)):
            plt.subplot(len(positions), 3, 3 * i + 1)
            plt.imshow(truth[i].real)
            plt.subplot(len(positions), 3, 3 * i + 2)
            plt.imshow(patches[i].real)
            plt.subplot(len(positions), 3, 3 * i + 3)
            plt.imshow(patches[i].real - truth[i].real, cmap=plt.cm.inferno)
            plt.colorbar()
        plt.savefig('patches.png')
    except ModuleNotFoundError:
        pass

    np.testing.assert_allclose(
        patches,
        truth,
        atol=1e-6,
    )


def test_patch_correctness_adjoint(size=8, win=2):

    positions = np.array([
        [0, 0],
        [0, size - win],
        [size - win, 0],
        [size - win, size - win],
        [size // 2 - win // 2, size // 2 - win // 2],
        [0.123, 3],
        [3, 0.123],
        [5.5, 3.5],
    ])
    fov = cp.zeros((size, size), dtype=tike.precision.cfloating)
    fov[:win, :win] += 1
    fov[:win, -win:] += 1
    fov[-win:, :win] += 1
    fov[-win:, -win:] += 1
    fov[size // 2 - win // 2:size // 2 - win // 2 + win,
        size // 2 - win // 2:size // 2 - win // 2 + win] += 1
    fov[0:win, 3:3 + win] += (1 - 0.123)
    fov[1:1 + win, 3:3 + win] += 0.123
    fov[3:3 + win, 0:win,] += (1 - 0.123)
    fov[3:3 + win, 1:1 + win,] += 0.123
    fov[5:5 + win, 3:3 + win] += 0.25
    fov[6:6 + win, 3:3 + win] += 0.25
    fov[5:5 + win, 4:4 + win] += 0.25
    fov[6:6 + win, 4:4 + win] += 0.25

    patches = cp.ones((len(positions), win, win),
                      dtype=tike.precision.cfloating)
    with Patch() as op:
        combined = op.adj(
            patches=patches,
            positions=cp.array(
                positions,
                dtype=tike.precision.floating,
                order='C',
            ),
            patch_width=win,
            images=cp.zeros(
                (size, size),
                dtype=tike.precision.cfloating,
            ),
        )

    fov = fov.get()
    combined = combined.get()
    try:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(fov.real)
        plt.title('true')
        plt.subplot(1, 3, 2)
        plt.imshow(combined.real)
        plt.title('patch operator')
        plt.subplot(1, 3, 3)
        plt.imshow(combined.real - fov.real, cmap=plt.cm.inferno)
        plt.title('error')
        plt.colorbar()
        plt.savefig('patches-adj.png')
    except ModuleNotFoundError:
        pass

    np.testing.assert_allclose(
        fov,
        combined,
        atol=1e-6,
    )


if __name__ == '__main__':
    unittest.main()
