"""Defines a communicator for both inter-GPU and inter-node communications."""

__author__ = "Xiaodong Yu"
__copyright__ = "Copyright (c) 2021, UChicago Argonne, LLC."

import cupy as cp

from .mpi import MPIComm
from .pool import ThreadPool


class Comm:
    """A Ptychography communicator.

    Compose the multiprocessing and multithreading communicators to handle
    synchronization and communication among both GPUs and nodes.

    Attributes
    ----------
    gpu_count : int
        The number of GPUs to use per process.
    mpi : class
        The multi-processing communicator.
    pool : class
        The multi-threading communicator.

    """

    def __init__(self, gpu_count,
                 mpi=MPIComm,
                 pool=ThreadPool,
                 **kwargs):
        if mpi is not None:
            self.mpi = mpi()
            self.use_mpi = True
        else:
            self.use_mpi = False
        self.pool = pool(gpu_count)

    def __enter__(self):
        if self.use_mpi is True:
            self.mpi.__enter__()
        self.pool.__enter__()
        return self

    def __exit__(self, type, value, traceback):
        if self.use_mpi is True:
            self.mpi.__exit__(type, value, traceback)
        self.pool.__exit__(type, value, traceback)

    def reduce(self, x, dest, **kwargs):
        """ThreadPool reduce from all GPUs to a GPU or CPU."""

        if dest == 'gpu':
            return self.pool.reduce_gpu(x, **kwargs)
        elif dest == 'cpu':
            return self.pool.reduce_cpu(x, **kwargs)
        else:
            raise ValueError(f'dest must be gpu or cpu.')

    def Allreduce_reduce(self, x, dest, **kwargs):
        """ThreadPool reduce coupled with MPI allreduce."""

        src = self.reduce(x, dest, **kwargs)
        if dest == 'gpu':
            return cp.asarray(self.mpi.Allreduce(cp.asnumpy(src)))
        elif dest == 'cpu':
            return self.mpi.Allreduce(src)
        else:
            raise ValueError(f'dest must be gpu or cpu.')

    def Allreduce_mean(self, x, **kwargs):
        """ThreadPool mean coupled with MPI allreduce mean."""

        src = self.reduce(x, dest, **kwargs)
        if dest == 'gpu':
            return cp.asarray(self.mpi.Allreduce(cp.asnumpy(src)))
        elif dest == 'cpu':
            return self.mpi.Allreduce(src)
        else:
            raise ValueError(f'dest must be gpu or cpu.')
