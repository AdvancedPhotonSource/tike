import unittest

import numpy as np

from tike.communicators import Comm


class TestComm(unittest.TestCase):

    def setUp(self, workers=4):
        self.comm = Comm(workers)
        self.xp = self.comm.pool.xp

    def test_reduce(self):
        a = self.xp.ones((1,))
        a_list = self.comm.pool.bcast([a])
        a = a * self.comm.pool.num_workers
        result = self.comm.reduce(a_list, 'cpu')
        self.xp.testing.assert_array_equal(a, result)
        result = self.comm.reduce(a_list, 'gpu')
        self.xp.testing.assert_array_equal(a, result[0])

    def test_Allreduce_reduce(self):
        a = self.xp.ones((1,))
        a_list = self.comm.pool.bcast([a])
        a = a * self.comm.pool.num_workers * self.comm.mpi.size
        result = self.comm.Allreduce_reduce(a_list, 'cpu')
        self.xp.testing.assert_array_equal(a, result)
        result = self.comm.Allreduce_reduce(a_list, 'gpu')
        self.xp.testing.assert_array_equal(a, result[0])

    # TODO: Determine what the correct behavior of scatter should be.
    # def test_scatter(self):
    #     a = np.arange(10)
    #     result = self.pool.scatter(a)


if __name__ == "__main__":
    unittest.main()
