import unittest

import numpy as np

from tike.communicators import ThreadPool


class TestThreadPool(unittest.TestCase):

    def setUp(self, workers=4):
        self.pool = ThreadPool(workers)
        self.xp = self.pool.xp

    def test_bcast(self):
        if self.pool.device_count < 2:
            return  # skip test if only one device
        a = self.xp.arange(10)
        result = self.pool.bcast([a])
        for i, x in enumerate(result):
            for j, y in enumerate(result):
                self.xp.testing.assert_array_equal(x, y)
                # should be copies; not the same array
                if self.xp == np:
                    assert np.logical_xor(
                        (x.__array_interface__['data'][0]
                         == y.__array_interface__['data'][0]),
                        i != j,
                    )
                else:
                    assert np.logical_xor(
                        (x.__cuda_array_interface__['data'][0]
                         == y.__cuda_array_interface__['data'][0]),
                        i != j,
                    )

    def test_gather(self):
        snake = self.xp.arange(1 * 2 * 3 * 4).reshape(1, 2, 3, 4)
        truth = self.xp.tile(snake, (1, 1, self.pool.num_workers, 1))
        a = self.pool.bcast([snake])
        result = self.pool.gather_host(a, axis=2)
        # print()
        # print(truth.shape, type(truth))
        # print(result.shape, type(result))
        self.xp.testing.assert_array_equal(result, truth)

    def test_gather_host(self):
        snake = np.arange(1 * 2 * 3 * 4).reshape(1, 2, 3, 4)
        truth = np.tile(snake, (1, 1, self.pool.num_workers, 1))
        a = self.pool.bcast([snake])
        result = self.pool.gather_host(a, axis=2)
        # print()
        # print(truth.shape, type(truth))
        # print(result.shape, type(result))
        np.testing.assert_array_equal(result, truth)

    def test_scatter(self, stride=3):
        stride = min(self.pool.num_workers, stride)
        a = [self.xp.ones(3) * i for i in range(self.pool.num_workers)]
        truth = [
            self.xp.ones(3) * i // stride for i in range(self.pool.num_workers)
        ]
        result = self.pool.scatter(
            x=a,
            stride=stride,
        )
        for (a, b) in zip(result, truth):
            print(a.device)
            self.xp.testing.assert_array_equal(a, b)

    def test_scatter_bcast(self, stride=3):
        stride = min(self.pool.num_workers, stride)
        a = [self.xp.ones(3) * i for i in range(self.pool.num_workers)]
        truth = [
            self.xp.ones(3) * i // stride for i in range(self.pool.num_workers)
        ]
        result = self.pool.scatter_bcast(
            x=a,
            stride=stride,
        )
        for (a, b) in zip(result, truth):
            print(a.device)
            self.xp.testing.assert_array_equal(a, b)

    def test_reduce_gpu(self, stride=3):
        stride = min(self.pool.num_workers, stride)
        a = [self.xp.ones(3) * i for i in range(self.pool.num_workers)]
        if self.pool.num_workers > 1:
            truth = [
                self.xp.sum(self.xp.stack(a[i::stride], axis=0), axis=0)
                for i in range(stride)
            ]
        else:
            truth = a
        result = self.pool.reduce_gpu(
            x=a,
            stride=stride,
        )
        for (a, b) in zip(result, truth):
            print(a.device)
            self.xp.testing.assert_array_equal(a, b)

    def test_reduce_cpu(self):
        ones = self.pool.map(self.xp.ones, shape=(11, 1, 3))
        result = self.pool.reduce_cpu(ones)
        np.testing.assert_array_equal(
            result,
            np.ones((11, 1, 3)) * self.pool.num_workers,
        )

    def test_allreduce(self, stride=3):
        stride = min(self.pool.num_workers, stride)
        a = [self.xp.ones(3) * i for i in range(self.pool.num_workers)]
        if self.pool.num_workers > 1:
            truth = [
                self.xp.sum(
                    self.xp.stack(
                        a[(i // stride * stride):((i // stride + 1) * stride)],
                        axis=0,
                    ),
                    axis=0,
                ) for i in range(int(np.ceil(self.pool.num_workers)))
            ]
        else:
            truth = a
        result = self.pool.allreduce(
            x=a,
            stride=stride,
        )
        for (a, b) in zip(result, truth):
            print(a.device)
            self.xp.testing.assert_array_equal(a, b)

    def test_reduce_mean(self):
        k = 0.5
        a = self.pool.bcast([self.xp.ones((4, 2, 5)) * k])
        truth = self.xp.ones((4, 1, 5)) * k
        result = self.pool.reduce_mean(a, axis=1)
        # print()
        # print(truth.shape, type(truth))
        # print(result.shape, type(truth))
        self.xp.testing.assert_array_equal(result, truth)


class TestSoloThreadPool(TestThreadPool):

    def setUp(self, workers=1):
        self.pool = ThreadPool(workers)
        self.xp = self.pool.xp


if __name__ == "__main__":
    unittest.main()
