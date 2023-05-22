import numpy as np
import cupy as cp
import cupyx

import tike.communicators.stream


def test_stream_reduce_prototype():

    def f(a, b, c):
        return a, b * c, b + c

    x0 = np.array([0, 1, 0, 0])
    x1 = np.array([1, 1, 3, 1])
    x2 = np.array([2, 2, 7, 2])
    args = [x0, x1, x2]

    truth = [
        1,
        2 + 2 + 21 + 2,
        3 + 3 + 10 + 3,
    ]

    result = [np.sum(y, axis=0) for y in zip(*[f(*x) for x in zip(*args)])]

    np.testing.assert_array_equal(truth, result)


def test_stream_reduce(dtype=np.double, num_streams=2):

    def f(a, b, c):
        return a, b * c, b + c

    x0 = cupyx.empty_pinned(shape=(4,), dtype=dtype)
    x0[:] = [0, 1, 0, 0]
    x1 = cupyx.empty_pinned(shape=(4,), dtype=dtype)
    x1[:] = [1, 1, 3, 1]
    x2 = cupyx.empty_pinned(shape=(4,), dtype=dtype)
    x2[:] = [2, 2, 7, 2]
    args = [x0, x1, x2]

    truth = [
        np.array([1], dtype=dtype),
        np.array([2 + 2 + 21 + 2], dtype=dtype),
        np.array([3 + 3 + 10 + 3], dtype=dtype),
    ]

    result = tike.communicators.stream.stream_and_reduce(
        f,
        args,
        y_shapes=[(1,), (1,), (1,)],
        y_dtypes=[dtype, dtype, dtype],
        streams=[cp.cuda.Stream() for _ in range(num_streams)]
    )

    print(result)

    np.testing.assert_array_equal(truth, result)
