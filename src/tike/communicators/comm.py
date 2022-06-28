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

    def __init__(
        self,
        gpu_count,
        mpi=MPIComm,
        pool=ThreadPool,
        **kwargs,
    ):
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

    def reduce(self, x, dest, s=1, **kwargs):
        """ThreadPool reduce from all GPUs to a GPU or CPU.

        Parameters
        ----------
        x : list
            Chunks to be reduced to a device group or the host.
        s : int
            The size of the device group. e.g. s=2 and num_gpu=8, then x[::2]
            will be reduced to workers[0] while x[1::2] will be reduced to
            workers[1].

        """
        if dest == 'gpu':
            return self.pool.reduce_gpu(x, s, **kwargs)
        elif dest == 'cpu':
            return self.pool.reduce_cpu(x, **kwargs)
        else:
            raise ValueError(f'dest must be gpu or cpu.')

    def Allreduce_reduce(self, x, dest, s=1, **kwargs):
        """ThreadPool reduce coupled with MPI allreduce."""
        src = self.reduce(x, dest, s, **kwargs)
        if dest == 'gpu':
            return [cp.asarray(self.mpi.Allreduce(cp.asnumpy(src[0])))]
        elif dest == 'cpu':
            return self.mpi.Allreduce(src).item()
        else:
            raise ValueError(f'dest must be gpu or cpu.')

    def Allreduce_mean(self, x, **kwargs):
        """Multi-process multi-GPU based mean."""
        src = self.pool.reduce_mean(x, **kwargs)
        mean = self.mpi.Allreduce(cp.asnumpy(src)) / self.mpi.size

        return cp.asarray(mean)

    def Allreduce(self, x, s=None, **kwargs):
        """ThreadPool allreduce coupled with MPI allreduce.

        Parameters
        ----------
        x : list
            Chunks to be all-reduced in grouped devices and between processes.
        s : int
            The size of a device group. e.g. s=4 and num_gpu=8, then x[:4] will
            perform all-reduce within workers[:4] while x[4:] will perform
            all-reduce within workers[4:].

        """
        src = self.pool.allreduce(x, s)
        buf = []
        for worker in self.pool.workers:
            with cp.cuda.Device(worker):
                buf.append(
                    cp.asarray(
                        self.mpi.Allreduce(
                            cp.asnumpy(src[self.pool.workers.index(worker)]),)))
        return buf
