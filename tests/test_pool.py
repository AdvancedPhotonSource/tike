import unittest

import numpy as np

from tike.pool import ThreadPool


class TestThreadPool(unittest.TestCase):

    def setUp(self, workers=7):
        self.pool = ThreadPool(workers)

    def test_bcast(self):
        a = np.arange(10)
        result = self.pool.bcast(a)
        for x in result:
            np.testing.assert_array_equal(a, x)
            # should be copies; not the same array
            assert x.__array_interface__['data'][0] != a.__array_interface__[
                'data'][0]

    def test_gather(self):
        a = np.arange(10)
        result = self.pool.gather(np.array_split(a, self.pool.num_workers))
        np.testing.assert_equal(a, result)

    # TODO: Determine what the correct behavior of scatter should be.
    # def test_scatter(self):
    #     a = np.arange(10)
    #     result = self.pool.scatter(a)


if __name__ == "__main__":
    unittest.main()
