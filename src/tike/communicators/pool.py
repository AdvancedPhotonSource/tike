"""Defines a worker Pool for multi-device managerment."""

__author__ = "Daniel Ching, Xiaodong Yu"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'

from concurrent.futures import ThreadPoolExecutor
import os
import typing
import warnings

import cupy as cp
import numpy as np
import numpy.typing as npt


class ThreadPool(ThreadPoolExecutor):
    """Python thread pool plus scatter gather methods.

    A Pool is a context manager which provides access to and communications
    amongst workers.

    Attributes
    ----------
    workers : int, tuple(int)
        The number of GPUs to use or a tuple of the device numbers of the GPUs
        to use. If the number of GPUs is less than the requested number, only
        workers for the available GPUs are allocated.
    device_count : int
        The total number of devices on the host as reported by CUDA runtime.
    num_workers : int
        Returns len(self.workers). For convenience.

    Raises
    ------
    ValueError
        When invalid GPU device ids are provided.
        When the current CUDA device does not match the first GPU id in the
        list of workers.
    """

    def __init__(self, workers):
        self.device_count = cp.cuda.runtime.getDeviceCount()
        if type(workers) is int:
            if workers < 1:
                raise ValueError(f"Provide workers > 0, not {workers}.")
            if workers > self.device_count:
                warnings.warn(
                    "Not enough CUDA devices for workers!"
                    f" Requested {workers} of {self.device_count} devices.")
                workers = min(workers, self.device_count)
            if workers == 1:
                # Respect "with cp.cuda.Device()" blocks for single thread
                workers = (cp.cuda.Device().id,)
            else:
                workers = tuple(range(workers))
        for w in workers:
            if w < 0 or w >= self.device_count:
                raise ValueError(f'{w} is not a valid GPU device number.')
        self.workers = workers
        self.xp = cp
        super().__init__(self.num_workers)

    def __enter__(self):
        if self.workers[0] != cp.cuda.Device().id:
            raise ValueError(
                "The primary worker must be the current device. "
                f"Use `with cupy.cuda.Device({self.workers[0]}):` to set the "
                "current device.")
        return self

    @property
    def num_workers(self):
        return len(self.workers)

    def _copy_to(self, x: typing.Union[cp.array, np.array],
                 worker: int) -> cp.array:
        with cp.cuda.Device(worker):
            return self.xp.asarray(x)

    def _copy_host(self, x: cp.array, worker: int) -> np.array:
        with cp.cuda.Device(worker):
            return self.xp.asnumpy(x)

    def bcast(self, x: npt.ArrayLike, s: int = 1) -> typing.List[cp.array]:
        """Send each x to all device groups.

        Parameters
        ----------
        x : list
            A list of data to be broadcast.
        s : int > 0
            The size of a device group. e.g. s=2 and num_gpu=8, then x[0] will
            be broadcast to workers[::2] while x[1] will go to workers[1::2].

        """

        def f(worker):
            idx = self.workers.index(worker) % s
            return self._copy_to(x[idx], worker)

        return list(self.map(f, self.workers))

    def gather(self, x: list, worker=None, axis=0) -> cp.array:
        """Concatenate x on a single worker along the given axis."""
        if self.num_workers == 1:
            return x[0]
        worker = self.workers[0] if worker is None else worker
        with cp.cuda.Device(worker):
            return self.xp.concatenate(
                [self._copy_to(part, worker) for part in x],
                axis,
            )

    def gather_host(self, x: list, axis=0) -> np.array:
        """Concatenate x on host along the given axis."""
        if self.num_workers == 1:
            return cp.asnumpy(x[0])

        def f(x, worker):
            return self._copy_host(x, worker)

        return np.concatenate(
            self.map(f, x, self.workers),
            axis,
        )

    def all_gather(self, x: list, axis=0) -> list:
        """Concatenate x on all workers along the given axis."""

        def f(worker):
            return self.gather(x, worker, axis)

        return list(self.map(f, self.workers))

    def scatter(self, x: list, s=1) -> list:
        """Send each x to a different device group`.

        Parameters
        ----------
        x : list
            Chunks to be sent to other devices.
        s : int
            The size of a device group. e.g. s=4 and num_gpu=8, then x[0] will
            be scattered to workers[:4] while x[1] will go to workers[4:].

        """

        def f(worker):
            idx = self.workers.index(worker) // s
            return self._copy_to(x[idx], worker)

        return self.map(f, self.workers)

    def scatter_bcast(self, x: list, stride=1):
        """Send x chunks to some workers and then copy to remaining workers.

        Parameters
        ----------
        x : list
            Chunks to be sent and copied.
        stride : int
            The stride length of the scatter. e.g. s=4 and num_gpu=8, then
            x[0] will be broadcast to workers[:4] while x[1] will go to
            workers[4:].

        """

        def s(bworkers, chunk):

            def b(worker):
                return self._copy_to(chunk, worker)

            return list(self.map(b, bworkers, workers=bworkers))

        bworkers = []
        if stride == 1:
            sworkers = self.workers[:len(x)]
            for i in range(len(x)):
                bworkers.append(self.workers[i::len(x)])
        else:
            sworkers = self.workers[::stride]
            for i in sworkers:
                bworkers.append(self.workers[i:(i + stride)])

        a = self.map(s, bworkers, x, workers=sworkers)
        output = [None] * self.num_workers
        i, j = 0, 0
        for si in bworkers:
            for bi in si:
                output[bi] = a[i][j]
                j += 1
            i += 1
            j = 0

        return output

    def reduce_gpu(self, x: list, s=1, workers=None):
        """Reduce x by addition to a device group from all other devices.

        Parameters
        ----------
        x : list
            Chunks to be reduced to a device group.
        s : int
            The size of the device group. e.g. s=2 and num_gpu=8, then x[::2]
            will be reduced to workers[0] while x[1::2] will be reduced to
            workers[1].

        """

        def f(worker):
            i = self.workers.index(worker)
            for part in x[(i % s):i:s]:
                x[i] += self._copy_to(part, worker)
            for part in x[(i + s)::s]:
                x[i] += self._copy_to(part, worker)
            return x[i]

        if self.num_workers == 1:
            return x

        workers = self.workers[:s] if workers is None else workers
        return self.map(f, workers, workers=workers)

    def reduce_cpu(self, x: typing.List[cp.array]) -> npt.NDArray:
        """Reduce x by addition from all GPUs to a CPU buffer."""
        assert len(x) <= self.num_workers, (
            f"{len(x)} work is more than {self.num_workers} workers")
        return np.sum(self.map(self._copy_host, x, self.workers), axis=0)

    def reduce_mean(self, x: list, axis, worker=None) -> cp.array:
        """Reduce x by addition to one GPU from all other GPUs."""
        if self.num_workers == 1:
            return x[0]
        worker = self.workers[0] if worker is None else worker
        return cp.mean(
            self.gather(x, worker=worker, axis=axis),
            keepdims=True,
            axis=axis,
        )

    def allreduce(self, x: list, s=None) -> list:
        """All-reduce x by addition within device groups.

        Parameters
        ----------
        x : list
            Chunks to be all-reduced in grouped devices context.
        s : int
            The size of a device group. e.g. s=4 and num_gpu=8, then x[:4] will
            perform all-reduce within workers[:4] while x[4:] will perform
            all-reduce within workers[4:].

        """

        def f(worker, id, buf, stride, s):
            idx = id // s * s + (id + stride) % s
            return cp.add(buf, self._copy_to(x[idx], worker))

        if self.num_workers == 1:
            return x

        buff = list(x)
        s = len(x) if s is None else s
        for stride in range(1, s):
            buff = self.map(
                f,
                self.workers,
                range(self.num_workers),
                buff,
                stride=stride,
                s=s,
            )

        return buff

    def map(self, func, *iterables, workers=None, **kwargs):
        """ThreadPoolExecutor.map, but wraps call in a cuda.Device context."""

        def f(worker, *args):
            with cp.cuda.Device(worker):
                return func(*args, **kwargs)

        workers = self.workers if workers is None else workers

        return list(super().map(f, workers, *iterables))
