"""Define a MPI wrapper for inter-node communications."""

__author__ = "Xiaodong Yu, Daniel Ching"
__copyright__ = "Copyright (c) 2021, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'

import os
import typing
import warnings

import numpy as np
import cupy as cp


def combined_shape(
    shapes: typing.List[typing.Tuple[int, ...]],
    axis: typing.Union[int, None] = 0,
) -> typing.List[int]:
    """Return the `shape` for `shapes` concatenated along the `axis`.

    >>> combined_shape([(5, 2, 3), (1, 2, 3)], axis=0)
    [6, 2, 3]

    >>> combined_shape([(5, 2, 7), (1, 2, 3)], axis=0)
    Traceback (most recent call last):
        ...
    ValueError: All dimensions except for the named `axis` must be equal

    >>> combined_shape([(1, 5, 3), (1, 5, 3)], axis=None)
    [2, 1, 5, 3]

    """
    first = shapes[0]
    ndim = len(first)
    for shape in shapes[1:]:
        if ndim != len(shape):
            msg = 'All shapes must have the same number of dimensions'
            raise ValueError(msg)
        for dim in range(ndim):
            if dim != axis and first[dim] != shape[dim]:
                msg = 'All dimensions except for the named `axis` must be equal'
                raise ValueError(msg)

    if axis is None:
        return [len(shapes), *first]

    combined = list()

    for dim in range(ndim):
        if dim == axis:
            combined.append(sum(shape[dim] for shape in shapes))
        else:
            combined.append(first[dim])

    return combined


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
        mask = np.logical_and(
            edges[self.rank] < scan[:, 0],
            scan[:, 0] <= edges[self.rank + 1],
        )
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

    def bcast(
        self,
        sendobj: typing.Any,
        root: int = 0,
    ) -> typing.Any:
        """Send a Python object from a root to all processes."""
        return sendobj

    def Bcast(
        self,
        sendbuf: typing.Union[np.ndarray, cp.ndarray],
        root: int = 0,
    ) -> typing.Union[np.ndarray, cp.ndarray]:
        """Send data from a root to all processes."""

        if sendbuf is None:
            raise ValueError(f"Broadcast data can't be empty.")

        return sendbuf

    def Gather(
        self,
        sendbuf: typing.Union[np.ndarray, cp.ndarray],
        axis: typing.Union[int, None] = 0,
        root: int = 0,
    ) -> typing.Union[np.ndarray, cp.ndarray]:
        """Take data from all processes into one destination."""

        if sendbuf is None:
            raise ValueError(f"Gather data can't be empty.")

        if axis is None:
            return sendbuf[None, ...]
        return sendbuf

    def Allreduce(
        self,
        sendbuf: typing.Union[np.ndarray, cp.ndarray],
        op=None,
    ) -> typing.Union[np.ndarray, cp.ndarray]:
        """Sum sendbuf from all ranks and return the result to all ranks."""

        if sendbuf is None:
            raise ValueError(f"Allreduce data can't be empty.")

        return sendbuf

    def Allgather(
        self,
        sendbuf: typing.Union[np.ndarray, cp.ndarray],
        axis: typing.Union[int, None] = 0,
    ) -> typing.Union[np.ndarray, cp.ndarray]:
        """Concatenate sendbuf from all ranks on all ranks."""

        if sendbuf is None:
            raise ValueError("Allgather data can't be None")

        if axis is None:
            return sendbuf[None, ...]
        return sendbuf


def check_opal(func):
    """Move sendbuf to host before the function if opal is not avaiable."""

    def wrapper(self, sendbuf, *args, **kwargs):
        xp = cp.get_array_module(sendbuf)

        if not self._use_opal and xp.__name__ == cp.__name__:
            warnings.warn(
                'GPU-Aware Open MPI not detected, '
                'but GPU arrays are being passed over MPI. '
                'Set OMPI_MCA_opal_cuda_support=true in your environment.',
                UserWarning,
            )
            return cp.asarray(func(self, cp.asnumpy(sendbuf), *args, **kwargs))

        return func(self, sendbuf, *args, **kwargs)

    return wrapper


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

        def bcast(
            self,
            sendobj: typing.Any,
            root: int = 0,
        ) -> typing.Any:
            """Send a Python object from a root to all processes."""
            cp.cuda.get_current_stream().synchronize()
            return self.comm.bcast(sendobj, root=root)

        @check_opal
        def Bcast(
            self,
            sendbuf: typing.Union[np.ndarray, cp.ndarray],
            root: int = 0,
        ) -> typing.Union[np.ndarray, cp.ndarray]:
            """Send data from a root to all processes."""

            if sendbuf is None:
                raise ValueError(f"Broadcast data can't be empty.")

            xp = cp.get_array_module(sendbuf)

            if self.rank != root:
                sendbuf = xp.empty_like(sendbuf)
            cp.cuda.get_current_stream().synchronize()
            self.comm.Bcast(sendbuf, root=root)
            return sendbuf

        @check_opal
        def Gather(
            self,
            sendbuf: typing.Union[np.ndarray, cp.ndarray],
            axis: typing.Union[int, None] = 0,
            root: int = 0,
        ) -> typing.Union[np.ndarray, cp.ndarray, None]:
            """Take data from all processes into one destination.

            Parameters
            ----------
            axis:
                Concatenate the gathered arrays long this existing axis; a new
                leading axis is created if axis is None.
            """

            if sendbuf is None:
                raise ValueError(f"Gather data can't be empty.")

            xp = cp.get_array_module(sendbuf)

            assert axis is None or sendbuf.ndim > 0, "Cannot concatenate zero-dimensional arrays; use `axis=None`"

            # Gather() doesn't support mixed shapes; we have to use Gatherv()
            # and keep track of shapes manually
            shapes = self.comm.gather(sendbuf.shape, root=root)
            if self.rank == root:
                assert shapes is not None
                sizes = [np.prod(shape, dtype=int) for shape in shapes]
                recvbuf = xp.empty_like(
                    sendbuf,
                    shape=sum(sizes),
                )
            else:
                recvbuf = None
                sizes = None
            cp.cuda.get_current_stream().synchronize()
            self.comm.Gatherv(sendbuf, (recvbuf, sizes), root=root)
            if self.rank == root:
                assert recvbuf is not None
                assert sizes is not None
                assert shapes is not None
                restored_arrays = [
                    x.reshape(shape) for x, shape in zip(
                        xp.split(
                            recvbuf,
                            np.cumsum(sizes[:-1]),
                        ),
                        shapes,
                    )
                ]
                if axis is None:
                    merge = xp.stack
                    axis = 0
                else:
                    merge = xp.concatenate
                return merge(restored_arrays, axis=axis)
            return recvbuf

        # def Scatter(self, sendbuf, src: int = 0):
        #     """Spread data from a source to all processes."""

        #     if sendbuf is None:
        #         raise ValueError(f"Scatter data can't be empty.")
        #     recvbuf = np.empty(sendbuf.shape, sendbuf.dtype)
        #     self.comm.Scatter(sendbuf, recvbuf, src)
        #     return recvbuf

        @check_opal
        def Allreduce(
            self,
            sendbuf: typing.Union[np.ndarray, cp.ndarray],
            op=MPI.SUM,
        ) -> typing.Union[np.ndarray, cp.ndarray]:
            """Sum sendbuf from all ranks and return the result to all ranks."""

            if sendbuf is None:
                raise ValueError(f"Allreduce data can't be empty.")

            xp = cp.get_array_module(sendbuf)

            recvbuf = xp.empty_like(sendbuf)
            cp.cuda.get_current_stream().synchronize()
            self.comm.Allreduce(sendbuf, recvbuf, op=op)
            return recvbuf

        @check_opal
        def Allgather(
            self,
            sendbuf: typing.Union[np.ndarray, cp.ndarray],
            axis: typing.Union[int, None] = 0,
        ) -> typing.Union[np.ndarray, cp.ndarray]:
            """Concatenate sendbuf from all ranks on all ranks.

            Parameters
            ----------
            axis:
                Concatenate the gathered arrays long this existing axis; a new
                leading axis is created if axis is None.
            """

            if sendbuf is None:
                raise ValueError("Allgather data can't be None")

            xp = cp.get_array_module(sendbuf)

            assert axis is None or sendbuf.ndim > 0, "Cannot concatenate zero-dimensional arrays; use `axis=None`"

            # Gather() doesn't support mixed shapes; we have to use Gatherv()
            # and keep track of shapes manually
            shapes = self.comm.allgather(sendbuf.shape)
            sizes = [np.prod(shape, dtype=int) for shape in shapes]
            recvbuf = xp.empty_like(
                sendbuf,
                shape=sum(sizes),
            )
            cp.cuda.get_current_stream().synchronize()
            self.comm.Allgatherv(sendbuf, (recvbuf, sizes))
            restored_arrays = [
                x.reshape(shape) for x, shape in zip(
                    xp.split(
                        recvbuf,
                        np.cumsum(sizes[:-1]),
                    ),
                    shapes,
                )
            ]
            if axis is None:
                merge = xp.stack
                axis = 0
            else:
                merge = xp.concatenate
            return merge(restored_arrays, axis=axis)

except ImportError:

    MPIComm = NoMPIComm
    warnings.warn(
        "tike was unable to import mpi4py, "
        "so MPI features are unavailable.", UserWarning)
