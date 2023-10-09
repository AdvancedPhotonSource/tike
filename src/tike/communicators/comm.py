"""Defines a communicator for both inter-GPU and inter-node communications."""

__author__ = "Xiaodong Yu, Daniel Ching"
__copyright__ = "Copyright (c) 2021, UChicago Argonne, LLC."

import warnings
import typing

import cupy as cp
import numpy as np

from .mpi import MPIComm, NoMPIComm
from .pool import ThreadPool


def _init_streams():
    return [cp.cuda.Stream() for _ in range(2)]


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
        mpi: typing.Union[typing.Type[MPIComm],
                          typing.Type[NoMPIComm]] = NoMPIComm,
        pool: typing.Type[ThreadPool] = ThreadPool,
    ):
        if isinstance(mpi, NoMPIComm):
            self.use_mpi = False
        else:
            self.use_mpi = True
        self.mpi = mpi()
        self.pool = pool(gpu_count)
        self.streams = self.pool.map(_init_streams)

    def __enter__(self):
        self.mpi.__enter__()
        self.pool.__enter__()
        return self

    def __exit__(self, type, value, traceback):
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
        warnings.warn(
            "Comm.reduce is deprecated. "
            "Use Comm.pool.reduce_gpu or Comm.pool.reduce_cpu directly.",
            DeprecationWarning)
        if dest == 'gpu':
            return self.pool.reduce_gpu(x, s, **kwargs)
        elif dest == 'cpu':
            return self.pool.reduce_cpu(x, **kwargs)
        else:
            raise ValueError(f'dest must be gpu or cpu.')

    def Allreduce_reduce_gpu(
        self,
        x: typing.List[cp.ndarray],
    ) -> typing.List[cp.ndarray]:
        """ThreadPool reduce followed by MPI Allreduce."""
        # TODO: Support stride/worker params for reduce_gpu
        # pool.map is required to ensure correct device context for x
        return self.pool.map(self.mpi.Allreduce, self.pool.reduce_gpu(x))

    def Allreduce_reduce_cpu(
        self,
        x: typing.List[cp.ndarray],
    ) -> np.ndarray:
        """ThreadPool reduce followed by MPI Allreduce."""
        return self.mpi.Allreduce(self.pool.reduce_cpu(x))

    def Allreduce_mean(
        self,
        x: typing.List[cp.ndarray],
        axis: typing.Union[int, None] = 0,
    ) -> cp.ndarray:
        """Multi-process multi-GPU based mean."""
        with cp.cuda.Device(self.pool.workers[0]):
            counts_local = np.array(
                [1 if x0.ndim == 0 else x0.shape[axis] for x0 in x],
                dtype=x[0].dtype,
            ).sum()
            counts_all = self.mpi.Allgather(counts_local, axis=None).sum()
            weight_local = counts_local / counts_all
            return self.mpi.Allreduce(
                self.pool.reduce_mean(x, axis=axis) * weight_local)

    def Allreduce(
        self,
        x: typing.List[cp.ndarray],
        s: int = None,
        **kwargs,
    ) -> typing.List[cp.ndarray]:
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
                    self.mpi.Allreduce(src[self.pool.workers.index(worker)]))
        return buf
