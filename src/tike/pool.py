"""Defines a worker Pool for multi-device managerment."""

__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['ThreadPool']

from concurrent.futures import ThreadPoolExecutor
import os
import warnings

import cupy as cp
import numpy as np


class NumPyThreadPool(ThreadPoolExecutor):
    """Python thread pool plus scatter gather methods.

    A Pool is a context manager which provides access to and communications
    amongst workers.

    """

    def __init__(self, num_workers: int, device_count=1):
        super().__init__(num_workers)
        self.device_count = device_count
        self.num_workers = num_workers
        self.workers = list(range(num_workers))
        self.xp = np

    def _copy_to(self, x: np.array, worker: int) -> np.array:
        """Copy x to the given worker."""
        return self.xp.array(x, copy=True)

    def bcast(self, x: np.array) -> list:
        """Send a copy of x to all workers."""

        def f(worker):
            return self._copy_to(x, worker)

        return list(self.map(f, self.workers))

    # def scatter(self, x: np.array) -> list:
    #     """Divide x amongst all workers along the 0th dimension."""
    #     return list(self.map(self._copy_to, x, self.workers))

    def gather(self, x: list, worker=0, axis=0) -> np.array:
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


class CuPyThreadPool(NumPyThreadPool):

    def __init__(self, num_workers):
        device_count = cp.cuda.runtime.getDeviceCount()
        if num_workers > device_count:
            warnings.warn("Not enough CUDA devices for workers!")
            num_workers = device_count
        super().__init__(num_workers, device_count)
        self.xp = cp

    def _copy_to(self, x: np.array, worker: int) -> np.array:
        with cp.cuda.Device(worker):
            return self.xp.asarray(x)

    def map(self, func, *iterables, **kwargs):
        """ThreadPoolExecutor.map, but wraps call in a cuda.Device context."""

        def f(worker, *args):
            with cp.cuda.Device(worker):
                return func(*args)

        return super().map(f, self.workers, *iterables, **kwargs)


ThreadPool = CuPyThreadPool
