import lzma
import os
import pickle
import unittest

import numpy as np

import tike.align

__author__ = "Daniel Ching, Viktor Nikitin"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'

testdir = os.path.dirname(__file__)


class TestAlignRecon(unittest.TestCase):
    """Test alignment reconstruction methods."""

    def create_dataset(self, dataset_file):
        """Create a dataset for testing this module.

        Only called with setUp detects that `dataset_file` has been deleted.
        """
        import matplotlib.pyplot as plt
        amplitude = plt.imread(
            os.path.join(testdir, "data/Cryptomeria_japonica-0128.png"))
        phase = plt.imread(
            os.path.join(testdir, "data/Bombus_terrestris-0128.png"))
        original = amplitude * np.exp(1j * phase * np.pi)
        self.original = np.expand_dims(original, axis=0).astype('complex64')

        np.random.seed(0)
        self.shift = 5 * (np.random.rand(1, 2) - 0.5)

        self.data = tike.align.simulate(self.original, self.shift)

        setup_data = [
            self.data,
            self.original,
            self.shift,
        ]

        with lzma.open(dataset_file, 'wb') as file:
            pickle.dump(setup_data, file)

    def setUp(self):
        """Load a dataset for reconstruction."""
        dataset_file = os.path.join(testdir, 'data/algin_setup.pickle.lzma')
        if not os.path.isfile(dataset_file):
            self.create_dataset(dataset_file)
        with lzma.open(dataset_file, 'rb') as file:
            [
                self.data,
                self.original,
                self.shift,
            ] = pickle.load(file)

    def test_consistent_simulate(self):
        """Check align.simulate for consistency."""
        data = tike.align.simulate(self.original, self.shift)
        assert data.dtype == 'complex64', data.dtype
        np.testing.assert_array_equal(data.shape, self.data.shape)
        np.testing.assert_allclose(data, self.data, atol=1e-6)

    def test_align_cross_correlation(self):
        """Check that align.solvers.cross_correlation works."""
        result = tike.align.reconstruct(
            self.data,
            self.original,
            algorithm='cross_correlation',
            upsample_factor=1e3,
        )
        shift = result['shift']
        assert shift.dtype == 'float32', shift.dtype
        np.testing.assert_array_equal(shift.shape, self.shift.shape)
        np.testing.assert_allclose(shift, self.shift, atol=1e-3)

    def test_align_farneback(self):
        """Check that align.solvers.farneback works."""
        result = tike.align.reconstruct(
            self.data,
            self.original,
            algorithm='farneback',
        )
        shift = result['shift']
        assert shift.dtype == 'float32', shift.dtype
        np.testing.assert_array_equal(shift.shape, (*self.original.shape, 2))
        np.testing.assert_allclose(shift[:, 1, 1, :], self.shift, atol=1e-3)
