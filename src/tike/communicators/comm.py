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
        The number of GPUs to use.
    mpi : class
        The multi-processing communicator.
    pool : class
        The multi-threading communicator.

    """

    def __init__(self, gpu_count,
                 mpi=MPIComm,
                 pool=ThreadPool,
                 **kwargs):
        self.mpi = mpi(gpu_count)
        self.pool = pool(gpu_count)
        self.num_workers = gpu_count

    def __enter__(self):
        self.mpi.__enter__()
        self.pool.__enter__()
        return self

    def __exit__(self, type, value, traceback):
        self.mpi.__exit__(type, value, traceback)
        self.pool.__exit__(type, value, traceback)


    def bcast(self, x: cp.array) -> list:
        """Send a copy of x to all threads in a process."""
        return self.pool.bcast(x)

    def gather(self, x: list, worker=None, axis=0) -> cp.array:
        """Concatenate x on a single thread along the given axis."""
        return self.pool.gather(x, worker, axis)

    def map(self, func, *iterables, **kwargs):
        """Map the given function to all threads."""
        return self.pool.map(func, *iterables, **kwargs)
