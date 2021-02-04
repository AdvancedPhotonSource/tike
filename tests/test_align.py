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
        import libimage
        amplitude = libimage.load("cryptomeria", 128)
        phase = libimage.load("bombus", 128)
        original = amplitude * np.exp(1j * phase * np.pi)
        self.original = np.expand_dims(original, axis=0).astype('complex64')

        np.random.seed(0)
        self.flow = np.empty((*self.original.shape, 2), dtype='float32')
        self.flow[..., :] = 5 * (np.random.rand(2) - 0.5)

        self.shift = 2 * (np.random.rand(*self.original.shape[:-2], 2) - 0.5)

        self.data = tike.align.simulate(
            original=self.original,
            flow=self.flow,
            shift=self.shift,
            padded_shape=None,
            angle=None,
        )

        setup_data = [
            self.data,
            self.original,
            self.flow,
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
                self.flow,
                self.shift,
            ] = pickle.load(file)

    def test_consistent_simulate(self):
        """Check align.simulate for consistency."""
        data = tike.align.simulate(
            original=self.original,
            flow=self.flow,
            shift=self.shift,
            padded_shape=None,
            angle=None,
        )
        assert data.dtype == 'complex64', data.dtype
        np.testing.assert_array_equal(data.shape, self.data.shape)
        np.testing.assert_allclose(data, self.data, atol=1e-6)

    def test_align_cross_correlation(self):
        """Check that align.solvers.cross_correlation works."""
        result = tike.align.reconstruct(
            unaligned=self.data,
            original=self.original,
            algorithm='cross_correlation',
            upsample_factor=1e3,
        )
        shift = result['shift']
        assert shift.dtype == 'float32', shift.dtype
        # np.testing.assert_array_equal(shift.shape, self.shift.shape)
        np.testing.assert_allclose(shift,
                                   self.flow[:, 0, 0] + self.shift,
                                   atol=1e-1)

    def test_align_farneback(self):
        """Check that align.solvers.farneback works."""
        result = tike.align.solvers.farneback(
            op=None,
            unaligned=np.angle(self.data),
            original=np.angle(self.original),
        )
        shift = result['flow']
        assert shift.dtype == 'float32', shift.dtype
        np.testing.assert_array_equal(shift.shape, (*self.original.shape, 2))
        h, w = shift.shape[1:3]
        np.testing.assert_allclose(shift[:, h // 2, w // 2, :],
                                   self.flow[:, 0, 0] + self.shift,
                                   atol=1e-1)


if __name__ == '__main__':
    unittest.main()
