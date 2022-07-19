import unittest

import cupy as cp

import tike.communicators

try:
    from mpi4py import MPI
    _mpi_size = MPI.COMM_WORLD.Get_size()
    _mpi_rank = MPI.COMM_WORLD.Get_rank()
except ModuleNotFoundError:
    _mpi_size = 0
    _mpi_rank = 0


class TestComm(unittest.TestCase):

    def setUp(self, workers=cp.cuda.runtime.getDeviceCount() // _mpi_size):
        cp.cuda.device.Device(workers * _mpi_rank).use()
        self.comm = tike.communicators.Comm(
            tuple(i + workers * _mpi_rank for i in range(workers)))
        self.xp = self.comm.pool.xp

    def test_reduce(self):
        a = self.xp.ones((1,))
        a_list = self.comm.pool.bcast([a])
        a = a * self.comm.pool.num_workers
        result = self.comm.reduce(a_list, 'cpu')
        self.xp.testing.assert_array_equal(a, result)
        result = self.comm.reduce(a_list, 'gpu')
        self.xp.testing.assert_array_equal(a, result[0])

    @unittest.skipIf(tike.communicators.MPIComm == None, "MPI is unavailable.")
    def test_Allreduce_reduce(self):
        a = self.xp.ones((1,))
        a_list = self.comm.pool.bcast([a])
        a = a * self.comm.pool.num_workers * self.comm.mpi.size
        result = self.comm.Allreduce_reduce(a_list, 'cpu')
        self.xp.testing.assert_array_equal(a, result)
        result = self.comm.Allreduce_reduce(a_list, 'gpu')
        self.xp.testing.assert_array_equal(a, result[0])

    @unittest.skipIf(tike.communicators.MPIComm == None, "MPI is unavailable.")
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

    # TODO: Determine what the correct behavior of scatter should be.
    # def test_scatter(self):
    #     a = np.arange(10)
    #     result = self.pool.scatter(a)


if __name__ == "__main__":
    unittest.main()
