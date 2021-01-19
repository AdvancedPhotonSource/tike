"""Define a MPI wrapper for inter-node communications.."""

__author__ = "Xiaodong Yu"
__copyright__ = "Copyright (c) 2021, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'

from mpi4py import MPI

import numpy as np


class MPIComm:
    """A class for python MPI wrapper.

    Many clusters do not support inter-node GPU-GPU
    communications, so we first gather the data into
    main memory then communicate them.

    Attributes
    ----------

    """

    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def p2p(self, sendbuf, src=0, dest=1, tg=0, **kwargs):
        """Send data from a source to a designated destination."""

        if sendbuf is None:
            raise ValueError(f"Sendbuf can't be empty.")
        if self.rank == src:
            self.comm.Send(sendbuf, dest=dest, tag=tg, **kwargs)
        elif self.rank == dest:
            info = MPI.Status()
            recvbuf = np.empty(sendbuf.shape, sendbuf.dtype)
            self.comm.Recv(recvbuf, source=src,
                           tag=tg, status=info, **kwargs)
            return recvbuf

    def Bcast(self, data, root: int = 0):
        """Send data from a root to all processes."""

        if data is None:
            raise ValueError(f"Broadcast data can't be empty.")
        if self.rank == root:
            data = data
        else:
            data = np.empty(data.shape, data.dtype)
        self.comm.Bcast(data, root)
        return data

    def Gather(self, sendbuf, dest: int = 0):
        """Take data from all processes into one destination."""

        if sendbuf is None:
            raise ValueError(f"Gather data can't be empty.")
        if self.rank == dest:
            recvbuf = np.empty(sendbuf.shape, sendbuf.dtype)
        self.comm.Scatter(sendbuf, recvbuf, dest)
        if self.rank == dest:
            return recvbuf

    def Scatter(self, sendbuf, src: int = 0):
        """Spread data from a source to all processes."""

        if sendbuf is None:
            raise ValueError(f"Scatter data can't be empty.")
        recvbuf = np.empty(sendbuf.shape, sendbuf.dtype)
        self.comm.Scatter(sendbuf, recvbuf, src)
        return recvbuf

    def Allreduce(self, sendbuf, op=MPI.SUM):
        """Combines data from all processes and distributes
        the result back to all processes."""

        if sendbuf is None:
            raise ValueError(f"Allreduce data can't be empty.")
        recvbuf = np.empty(sendbuf.shape, sendbuf.dtype)
        self.comm.Allreduce(sendbuf, recvbuf, op=op)

        return recvbuf

    def MPIio(self, scan, data):
        """Read data parts to different processes."""

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
        mask = np.logical_and(
                edges[self.rank] < scan[0, :, 0],
                scan[0, :, 0] <= edges[self.rank + 1])

        scan = scan[:, mask]
        data = data[:, mask]
        return scan, data
