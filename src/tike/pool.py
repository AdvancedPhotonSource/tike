"""Defines a worker Pool for multi-device managerment."""

__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['ThreadPool']

from concurrent.futures import ThreadPoolExecutor
import os

import numpy as np


class NumPyThreadPool(ThreadPoolExecutor):
    """Python thread pool plus scatter gather methods.

    A Pool is a context manager which provides access to and communications
    amongst workers.

    """

    def __init__(self, num_workers: int):
        super().__init__(num_workers)
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

    def scatter(self, x: np.array) -> list:
        """Divide x amongst all workers along the 0th dimension."""
        return list(self.map(self._copy_to, x, self.workers))

    def gather(self, x: list, worker=0, axis=0) -> np.array:
        """Concatenate x on a single worker along the given axis."""
        return self.xp.concatenate(
            [self._copy_to(part, worker) for part in x],
            axis,
        )

    def all_gather(self, x: list) -> list:
        """Copy a scattered x to all workers."""

        def f(worker):
            return self.gather(x, worker)

        return list(self.map(f, self.workers))


class CuPyThreadPool(NumPyThreadPool):

    def __init__(self, num_workers):
        super().__init__(num_workers)
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


# Provide the correct ThreadPool implementaiton based the environment variable
if f"TIKE_BACKEND" in os.environ and os.environ["TIKE_BACKEND"] == 'cupy':
    ThreadPool = CuPyThreadPool
    import cupy as cp
else:
    ThreadPool = NumPyThreadPool
