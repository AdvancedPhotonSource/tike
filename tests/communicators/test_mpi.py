import unittest

import numpy as np
import cupy as cp

from tike.communicators import MPIComm


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
        with cp.cuda.Device(self.mpi.rank):
            x = self.xp.ones(5) if self.mpi.rank == root else self.xp.zeros(5)
            truth = self.xp.ones(5)
            result = self.mpi.Bcast(x, root=root)
            self.xp.testing.assert_array_equal(result, truth)

    def test_gather(self, root=0):
        # For testing assign each rank 1 GPU of the same index
        with cp.cuda.Device(self.mpi.rank):
            x = self.xp.ones(5) * self.mpi.rank
            truth = self.xp.ones((self.mpi.size, 5)) * self.xp.arange(
                self.mpi.size)[..., None]
            result = self.mpi.Gather(x, root=root, axis=None)
            if self.mpi.rank == root:
                self.xp.testing.assert_array_equal(result, truth)
            else:
                assert result is None

    def test_scatter(self):
        pass

    def test_allreduce(self):
        # For testing assign each rank 1 GPU of the same index
        with cp.cuda.Device(self.mpi.rank):
            x = self.xp.ones(5)
            truth = self.xp.ones(5) * self.mpi.size
            result = self.mpi.Allreduce(x)
            self.xp.testing.assert_array_equal(result, truth)

    def test_allgather(self):
        # For testing assign each rank 1 GPU of the same index
        with cp.cuda.Device(self.mpi.rank):
            x = self.xp.ones(5) * self.mpi.rank
            truth = self.xp.arange(self.mpi.size)[:, None] * self.xp.ones(
                (1, 5))
            result = self.mpi.Allgather(x, axis=None)
            print(result, truth)
            self.xp.testing.assert_array_equal(result, truth)
