"""Define a MPI wrapper for inter-node communications."""

__author__ = "Xiaodong Yu, Daniel Ching"
__copyright__ = "Copyright (c) 2021, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'

import os
import typing
import warnings

import numpy as np
import cupy as cp


class MPIio:
    """Implementations for problem specific data loaders"""

    def MPIio_ptycho(self, scan, *args):
        """Read data parts to different processes for ptycho."""

        # Determine the edges of the stripes
        edges = np.linspace(
            scan[..., 0].min(),
            scan[..., 0].max(),
            self.size + 1,
            endpoint=True,
        )

        # Move the outer edges to include all points
        edges[0] -= 1
        edges[-1] += 1

        # Generate the mask
        mask = np.logical_and(edges[self.rank] < scan[:, 0],
                              scan[:, 0] <= edges[self.rank + 1])

        scan = scan[mask]
        split_args = [arg[mask] for arg in args]
        print("size", mask.shape, type((scan, *split_args)))

        return (scan, *split_args)

    def MPIio_lamino(self, *args, axis=0):
        """Read data parts to different processes for lamino."""

        return tuple(
            np.array_split(arg, self.size, axis=axis)[self.rank]
            for arg in args)


class NoMPIComm(MPIio):
    """Placeholder for MPI Communications when no MPI4Py is installed.

        Attributes
        ----------
        rank : int
            The identity of this process.
        size : int
            The total number of MPI processes.
        """

    def __init__(self):
        self.rank = 0
        self.size = 1

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def Bcast(
        self,
        sendbuf: np.ndarray | cp.ndarray,
        root: int = 0,
    ) -> np.ndarray | cp.ndarray:
        """Send data from a root to all processes."""

        if sendbuf is None:
            raise ValueError(f"Broadcast data can't be empty.")

        return sendbuf

    def Gather(
        self,
        sendbuf: np.ndarray | cp.ndarray,
        root: int = 0,
    ) -> typing.List[typing.Union[np.ndarray, cp.ndarray]]:
        """Take data from all processes into one destination."""

        if sendbuf is None:
            raise ValueError(f"Gather data can't be empty.")

        return sendbuf[None, ...]

    def Allreduce(
        self,
        sendbuf: np.ndarray | cp.ndarray,
        op=None,
    ) -> np.ndarray | cp.ndarray:
        """Sum sendbuf from all ranks and return the result to all ranks."""

        if sendbuf is None:
            raise ValueError(f"Allreduce data can't be empty.")

        return sendbuf

    def Allgather(
        self,
        sendbuf: np.ndarray | cp.ndarray,
    ) -> np.ndarray | cp.ndarray:
        """Concatenate sendbuf from all ranks on all ranks."""

        if sendbuf is None:
            raise ValueError("Allgather data can't be None")

        return sendbuf[None, ...]


try:
    from mpi4py import MPI

    class MPIComm(MPIio):
        """A class for python MPI wrapper.

        Many clusters do not support inter-node GPU-GPU
        communications, so we first gather the data into
        main memory then communicate them.

        Attributes
        ----------
        rank : int
            The identity of this process.
        size : int
            The total number of MPI processes.
        """

        def __init__(self):
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
            self._use_opal = os.environ.get(
                'OMPI_MCA_opal_cuda_support',
                default='false',
            ).lower() == 'true'
            if not self._use_opal:
                warnings.warn(
                    'GPU-Aware Open MPI not detected. '
                    'Set OMPI_MCA_opal_cuda_support=true in your environment '
                    'if using GPU-Aware Open MPI.',
                    UserWarning,
                )

        def __enter__(self):
            return self

        def __exit__(self, type, value, traceback):
            pass

        # def p2p(self, sendbuf, src=0, dest=1, tg=0, **kwargs):
        #     """Send data from a source to a designated destination."""

        #     if sendbuf is None:
        #         raise ValueError(f"Sendbuf can't be empty.")
        #     if self.rank == src:
        #         self.comm.Send(sendbuf, dest=dest, tag=tg, **kwargs)
        #     elif self.rank == dest:
        #         info = MPI.Status()
        #         recvbuf = np.empty(sendbuf.shape, sendbuf.dtype)
        #         self.comm.Recv(recvbuf,
        #                        source=src,
        #                        tag=tg,
        #                        status=info,
        #                        **kwargs)
        #         return recvbuf

        def Bcast(
            self,
            sendbuf: np.ndarray | cp.ndarray,
            root: int = 0,
        ) -> np.ndarray | cp.ndarray:
            """Send data from a root to all processes."""

            if sendbuf is None:
                raise ValueError(f"Broadcast data can't be empty.")

            xp = cp.get_array_module(sendbuf)

            if not self._use_opal and xp == cp:
                # Move to host for non-GPU aware MPI
                return cp.asarray(self.Bcast(cp.asnumpy(sendbuf)))

            if self.rank != root:
                sendbuf = xp.empty_like(sendbuf)
            cp.cuda.get_current_stream().synchronize()
            self.comm.Bcast(sendbuf, root=root)
            return sendbuf

        def Gather(
            self,
            sendbuf: np.ndarray | cp.ndarray,
            root: int = 0,
        ) -> typing.List[typing.Union[np.ndarray, cp.ndarray]]:
            """Take data from all processes into one destination."""

            if sendbuf is None:
                raise ValueError(f"Gather data can't be empty.")

            xp = cp.get_array_module(sendbuf)

            if not self._use_opal and xp == cp:
                # Move to host for non-GPU aware MPI
                return cp.asarray(self.Gather(cp.asnumpy(sendbuf)))

            recvbuf = xp.empty_like(
                sendbuf,
                shape=(self.size, *sendbuf.shape),
            ) if self.rank == root else None
            cp.cuda.get_current_stream().synchronize()
            self.comm.Gather(sendbuf, recvbuf, root=root)
            return recvbuf

        # def Scatter(self, sendbuf, src: int = 0):
        #     """Spread data from a source to all processes."""

        #     if sendbuf is None:
        #         raise ValueError(f"Scatter data can't be empty.")
        #     recvbuf = np.empty(sendbuf.shape, sendbuf.dtype)
        #     self.comm.Scatter(sendbuf, recvbuf, src)
        #     return recvbuf

        def Allreduce(
            self,
            sendbuf: np.ndarray | cp.ndarray,
            op=MPI.SUM,
        ) -> np.ndarray | cp.ndarray:
            """Sum sendbuf from all ranks and return the result to all ranks."""

            if sendbuf is None:
                raise ValueError(f"Allreduce data can't be empty.")

            xp = cp.get_array_module(sendbuf)

            if not self._use_opal and xp == cp:
                # Move to host for non-GPU aware MPI
                return cp.asarray(self.Allreduce(cp.asnumpy(sendbuf)))

            recvbuf = xp.empty_like(sendbuf)
            cp.cuda.get_current_stream().synchronize()
            self.comm.Allreduce(sendbuf, recvbuf, op=op)
            return recvbuf

        def Allgather(
            self,
            sendbuf: np.ndarray | cp.ndarray,
        ) -> np.ndarray | cp.ndarray:
            """Concatenate sendbuf from all ranks on all ranks."""

            if sendbuf is None:
                raise ValueError("Allgather data can't be None")

            xp = cp.get_array_module(sendbuf)

            if not self._use_opal and xp == cp:
                # Move to host for non-GPU aware MPI
                return cp.asarray(self.Allgather(cp.asnumpy(sendbuf)))

            recvbuf = xp.empty_like(sendbuf, shape=(self.size, *sendbuf.shape))
            cp.cuda.get_current_stream().synchronize()
            self.comm.Allgather(sendbuf, recvbuf)
            return recvbuf

except ModuleNotFoundError:

    MPIComm = None
    warnings.warn(
        "tike was unable to import mpi4py, "
        "so MPI features are unavailable.", UserWarning)
