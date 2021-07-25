"""Defines a worker Pool for multi-device managerment."""

__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'

from concurrent.futures import ThreadPoolExecutor
import os
import warnings

import cupy as cp


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
        if workers[0] != cp.cuda.Device().id:
            raise ValueError(
                "The primary worker must be the current device. "
                f"Use `with cupy.cuda.Device({workers[0]}):` to set the "
                "current device.")
        self.workers = workers
        self.num_workers = len(workers)
        self.xp = cp
        super().__init__(self.num_workers)

    def _copy_to(self, x, worker: int) -> cp.array:
        with cp.cuda.Device(worker):
            return self.xp.asarray(x)

    def bcast(self, x: list, s=1) -> list:
        """Send a copy of x to all workers."""

        def f(worker):
            idx = worker % s
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

    def all_gather(self, x: list, axis=0) -> list:
        """Concatenate x on all workers along the given axis."""

        def f(worker):
            return self.gather(x, worker, axis)

        return list(self.map(f, self.workers))

    def scatter(self, x):
        """Split x along 0th dimension and send chunks to workers`."""

        def f(worker, chunk):
            return self._copy_to(chunk, worker)

        return self.map(f, self.workers, x)

    def scatter_bcast(self, x: list, stride=1):
        """Send x chunks to some workers and then copy to remaining workers."""

        def s(bworkers, chunk):

            def b(worker):
                return self._copy_to(chunk, worker)

            return list(self.map(b, bworkers, workers=bworkers))

        bworkers = []
        if stride is 1:
            sworkers = self.workers[:len(x)]
            for i in range(len(x)):
                bworkers.append(self.workers[i::len(x)])
        else:
            sworkers = self.workers[::stride]
            for i in sworkers:
                bworkers.append(self.workers[i:(i+stride)])

        a = self.map(s, bworkers, x, workers=sworkers)
        output = [None] * self.num_workers
        i, j = 0, 0
        for si in bworkers:
            for bi in si:
                output[bi]=a[i][j]
                j += 1
            i += 1
            j = 0

        return output

    def reduce_gpu(self, x: list, s=1, workers=None):
        """Reduce x by addition to a subset of GPUs from all other GPUs."""

        def f(worker):
            for part in x[(worker % s):worker:s]:
                x[worker] += self._copy_to(part, worker)
            for part in x[(worker + s)::s]:
                x[worker] += self._copy_to(part, worker)
            return x[worker]

        if self.num_workers == 1:
            return x

        workers = self.workers[:s] if workers is None else workers
        return self.map(f, workers, workers=workers)

    def reduce_cpu(self, x, buf=None):
        """Reduce x by addition from all GPUs to a CPU buffer."""
        buf = 0 if buf is None else buf
        buf += sum([self.xp.asnumpy(part) for part in x])
        return buf

    def reduce_mean(self, x: list, axis, worker=None) -> cp.array:
        """Reduce x by addition to one GPU from all other GPUs."""
        if self.num_workers == 1:
            return x[0]
        worker = self.workers[0] if worker is None else worker
        return cp.mean(self.gather(x, worker=worker, axis=axis),
                       keepdims=True, axis=axis)

    def grouped_allreduce(self, x: list, s: int):
        """All-reduce x by addition within a subset of GPUs."""

        def f(worker, buf, stride):
            idx = worker // s * s + (worker + stride) % s
            return cp.add(buf, self._copy_to(x[idx], worker))

        if self.num_workers == 1:
            return x

        buff = list(x)
        for stride in range(s):
            buff = self.map(f, self.workers, buff, stride=stride)

        return buff

    def map(self, func, *iterables, **kwargs):
        """ThreadPoolExecutor.map, but wraps call in a cuda.Device context."""

        def f(worker, *args):
            with cp.cuda.Device(worker):
                return func(*args, **kwargs)

        if 'workers' in kwargs:
            workers = kwargs.get("workers")
            kwargs.pop("workers")
        else:
            workers = self.workers

        return list(super().map(f, workers, *iterables))
