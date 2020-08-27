"""Defines a worker Pool for multi-device managerment."""

__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['ThreadPool']

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
    """

    def __init__(self, workers):
        self.device_count = cp.cuda.runtime.getDeviceCount()
        if type(workers) is int and workers > 0:
            if workers > self.device_count:
                warnings.warn(
                    "Not enough CUDA devices for workers!"
                    f" Requested {workers} of {self.device_count} devices.")
                workers = min(workers, self.device_count)
            workers = tuple(range(workers))
        else:
            raise ValueError(f"Provide workers > 0, not {workers}.")
        for w in workers:
            if w < 0 or w >= self.device_count:
                raise ValueError(f'{w} is not a valid GPU device number.')
        self.workers = workers
        self.num_workers = len(workers)
        self.xp = cp
        super().__init__(self.num_workers)

    def _copy_to(self, x, worker: int) -> cp.array:
        with cp.cuda.Device(worker):
            return self.xp.asarray(x)

    def bcast(self, x: cp.array) -> list:
        """Send a copy of x to all workers."""

        def f(worker):
            return self._copy_to(x, worker)

        return list(self.map(f, self.workers))

    def gather(self, x: list, worker=0, axis=0) -> cp.array:
        """Concatenate x on a single worker along the given axis."""
        return self.xp.concatenate(
            [self._copy_to(part, worker) for part in x],
            axis,
        )

    def all_gather(self, x: list, axis=0) -> list:
        """Concatenate x on all worker along the given axis."""

        def f(worker):
            return self.gather(x, worker, axis)

        return list(self.map(f, self.workers))

    def map(self, func, *iterables, **kwargs):
        """ThreadPoolExecutor.map, but wraps call in a cuda.Device context."""

        def f(worker, *args):
            with cp.cuda.Device(worker):
                return func(*args)

        return super().map(f, self.workers, *iterables, **kwargs)
