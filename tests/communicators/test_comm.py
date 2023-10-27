import unittest

import cupy as cp
import numpy as np

import tike.communicators

try:
    from mpi4py import MPI
    _mpi_size = MPI.COMM_WORLD.Get_size()
    _mpi_rank = MPI.COMM_WORLD.Get_rank()
except ImportError:
    _mpi_size = 1
    _mpi_rank = 0

_gpu_count = cp.cuda.runtime.getDeviceCount()

class TestComm(unittest.TestCase):

    def setUp(self, workers=max(1,  _gpu_count // _mpi_size)):
        cp.cuda.device.Device((workers * _mpi_rank) % _gpu_count).use()
        self.comm = tike.communicators.Comm(
            tuple(i + (workers * _mpi_rank) % _gpu_count for i in range(workers)))
        self.xp = self.comm.pool.xp

    def test_Allreduce_reduce_gpu(self):
        a = cp.array(1)
        a_list = self.comm.pool.bcast([a])
        truth = [cp.array(self.comm.pool.num_workers * self.comm.mpi.size)]
        result = self.comm.Allreduce_reduce_gpu(a_list)
        # print()
        # print(truth, type(truth))
        # print(result, type(result))
        cp.testing.assert_array_equal(result[0], truth[0])

    def test_Allreduce_reduce_cpu(self):
        a = np.array(1)
        a_list = self.comm.pool.bcast([a])
        truth = a * np.array(self.comm.pool.num_workers * self.comm.mpi.size)
        result = self.comm.Allreduce_reduce_cpu(a_list)
        # print()
        # print(truth, type(truth))
        # print(result, type(result))
        np.testing.assert_array_equal(result, truth)

    def test_Allreduce_reduce_mean(self):
        a = cp.array(1.0)
        a_list = self.comm.pool.bcast([a])
        truth = cp.array(1.0)
        result = self.comm.Allreduce_mean(a_list, axis=None)
        # print()
        # print(truth, type(truth))
        # print(result, type(result))
        cp.testing.assert_array_equal(result, truth)

    def test_Allreduce_allreduce(self):
        a = self.xp.arange(10).reshape(2, 5)
        a_list = self.comm.pool.bcast([a])
        result = self.comm.Allreduce(a_list)

        def check_correct(result):
            self.xp.testing.assert_array_equal(
                result,
                self.xp.arange(10).reshape(2, 5) * self.comm.pool.num_workers *
                self.comm.mpi.size,
            )
            print(result)

        self.comm.pool.map(check_correct, result)


if __name__ == "__main__":
    unittest.main()
