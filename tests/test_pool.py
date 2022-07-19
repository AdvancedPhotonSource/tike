import unittest

import numpy as np

from tike.communicators import ThreadPool


class TestThreadPool(unittest.TestCase):

    def setUp(self, workers=3):
        self.pool = ThreadPool(workers)
        self.xp = self.pool.xp

    def test_bcast(self):
        if self.pool.device_count < 2:
            return  # skip test if only one device
        a = self.xp.arange(10)
        result = self.pool.bcast([a])
        for i, x in enumerate(result):
            self.xp.testing.assert_array_equal(a, x)
            # should be copies; not the same array
            if self.xp == np:
                assert (x.__array_interface__['data'][0] !=
                        a.__array_interface__['data'][0]) or i == 0
            else:
                assert (x.__cuda_array_interface__['data'][0] !=
                        a.__cuda_array_interface__['data'][0]) or i == 0

    def test_gather(self):
        if self.pool.device_count < 2:
            return  # skip test if only one device
        a = self.xp.arange(10)
        result = self.pool.gather(np.array_split(a, self.pool.num_workers))
        self.xp.testing.assert_array_equal(a, result)

    def test_reduce_cpu(self):
        ones = self.pool.map(self.xp.ones, shape=(11, 1, 3))
        result = self.pool.reduce_cpu(ones)
        np.testing.assert_array_equal(
            result,
            np.ones((11, 1, 3)) * self.pool.num_workers,
        )

    # TODO: Determine what the correct behavior of scatter should be.
    # def test_scatter(self):
    #     a = np.arange(10)
    #     result = self.pool.scatter(a)


class TestSoloThreadPool(TestThreadPool):

    def setUp(self, workers=1):
        self.pool = ThreadPool(workers)
        self.xp = self.pool.xp


if __name__ == "__main__":
    unittest.main()
