import unittest

import numpy as np
import cupy as cp

from tike.communicators import MPIComm, combined_shape

_gpu_count = cp.cuda.runtime.getDeviceCount()

class TestMPIComm(unittest.TestCase):

    def setUp(self):
        self.mpi = MPIComm()
        self.xp = np
        # NOTE: MPI GPU awareness requires the following environment variable
        # to be set: `OMPI_MCA_opal_cuda_support=true` conda-forge openmpi is
        # compiled with GPU awareness.

    def test_p2p(self):
        pass

    def test_bcast(self, root=0):
        with cp.cuda.Device(self.mpi.rank % _gpu_count):
            x = self.xp.ones(5) if self.mpi.rank == root else self.xp.zeros(5)
            truth = self.xp.ones(5)
            result = self.mpi.Bcast(x, root=root)
            self.xp.testing.assert_array_equal(result, truth)

    def test_gather(self, root=0):
        # For testing assign each rank 1 GPU of the same index
        with cp.cuda.Device(self.mpi.rank % _gpu_count):
            x = self.xp.ones(5) * self.mpi.rank
            truth = self.xp.ones(
                (self.mpi.size, 5)) * self.xp.arange(self.mpi.size)[..., None]
            result = self.mpi.Gather(x, root=root, axis=None)
            if self.mpi.rank == root:
                self.xp.testing.assert_array_equal(result, truth)
            else:
                assert result is None

    def test_gather_mismatched_shapes(self, root=0):
        # For testing assign each rank 1 GPU of the same index
        with cp.cuda.Device(self.mpi.rank % _gpu_count):
            x = self.xp.ones((2, self.mpi.rank + 1, 1, 3)) * (self.mpi.rank + 1)
            truth = self.xp.ones((
                2,
                sum(i for i in range(1, self.mpi.size + 1)),
                1,
                3,
            )) * self.xp.array(
                np.concatenate([[i] * i for i in range(1, self.mpi.size + 1)
                               ]))[..., None, None]
            result = self.mpi.Gather(x, root=root, axis=1)
            if self.mpi.rank == root:
                print()
                print(result)
                print()
                print(truth)
                self.xp.testing.assert_array_equal(result, truth)
            else:
                assert result is None

    def test_scatter(self):
        pass

    def test_allreduce(self):
        # For testing assign each rank 1 GPU of the same index
        with cp.cuda.Device(self.mpi.rank % _gpu_count):
            x = self.xp.ones(5)
            truth = self.xp.ones(5) * self.mpi.size
            result = self.mpi.Allreduce(x)
            self.xp.testing.assert_array_equal(result, truth)

    def test_allgather(self):
        # For testing assign each rank 1 GPU of the same index
        with cp.cuda.Device(self.mpi.rank % _gpu_count):
            x = self.xp.ones(5) * self.mpi.rank
            truth = self.xp.arange(self.mpi.size)[:, None] * self.xp.ones(
                (1, 5))
            result = self.mpi.Allgather(x, axis=None)
            print(result, truth)
            self.xp.testing.assert_array_equal(result, truth)

    def test_allgather_mismatched_shapes(self):
        # For testing assign each rank 1 GPU of the same index
        with cp.cuda.Device(self.mpi.rank % _gpu_count):
            x = self.xp.ones((2, self.mpi.rank + 1, 1, 3)) * (self.mpi.rank + 1)
            truth = self.xp.ones((
                2,
                sum(i for i in range(1, self.mpi.size + 1)),
                1,
                3,
            )) * self.xp.array(
                np.concatenate([[i] * i for i in range(1, self.mpi.size + 1)
                               ]))[..., None, None]
            result = self.mpi.Allgather(x, axis=1)
            print()
            print(result)
            print()
            print(truth)
            self.xp.testing.assert_array_equal(result, truth)

    def test_combined_shape(self):

        assert combined_shape([(5, 2, 3), (1, 2, 3)], axis=0) == [6, 2, 3]

        # ValueError: All dimensions except for the named `axis` must be equal
        with self.assertRaises(ValueError):
            combined_shape([(5, 2, 7), (1, 2, 3)], axis=0)

        with self.assertRaises(ValueError):
            combined_shape([(5, 2, 7), (1, 2, 3)], axis=None)

        assert combined_shape([(1, 5, 3), (1, 5, 3)], axis=None) == [2, 1, 5, 3]
