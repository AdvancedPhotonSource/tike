import typing

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
        return cp.sum(a), cp.sum(b * c), cp.sum(b + c)

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
        y_shapes=[[1], [1], [1]],
        y_dtypes=[dtype, dtype, dtype],
        streams=[cp.cuda.Stream() for _ in range(num_streams)],
    )
    result = [r.get() for r in result]

    print(result)

    np.testing.assert_array_equal(truth, result)


def test_stream_reduce_benchmark(dtype=np.double, num_streams=2, w=512):

    def f(a):
        return (
            cp.sum(cp.fft.fft2(a).real, axis=0, keepdims=True),
            cp.sum(cp.linalg.norm(a, axis=(-1, -2), keepdims=True),
                   axis=0,
                   keepdims=True),
            cp.sum(a, keepdims=True),
        )

    x0 = cupyx.empty_pinned(shape=(1_000, w, w), dtype=dtype)
    x0[:] = 1
    args = [
        x0,
    ]

    result = tike.communicators.stream.stream_and_reduce(
        f,
        args,
        y_shapes=[(1, w, w), (1, 1, 1), (1, 1, 1)],
        y_dtypes=[dtype, dtype, dtype],
        streams=[cp.cuda.Stream() for _ in range(num_streams)],
        chunk_size=32,
    )
    result = [r.get() for r in result]

    print(result)


def test_stream_modify(dtype=np.double, num_streams=2):

    def f(ind_args, mod_args, _):
        (a, b), (c,) = ind_args, mod_args
        return (np.sum(a * b) + c,)

    x0 = cupyx.empty_pinned(shape=(4,), dtype=dtype)
    x0[:] = [0, 1, 2, 0.0]
    x1 = cupyx.empty_pinned(shape=(4,), dtype=dtype)
    x1[:] = [1, 1, 3, 1.0]
    x2 = 0.0
    ind_args = (x0, x1)
    mod_args = (x2,)

    truth = cp.array(0 * 1 + 1 * 1 + 2 * 3 + 0 * 1.0),

    result = tike.communicators.stream.stream_and_modify(
        f,
        ind_args,
        mod_args,
        streams=[cp.cuda.Stream() for _ in range(num_streams)],
        chunk_size=2,
    )

    for t, r in zip(truth, result):
        print(t, type(t))
        print(r, type(t))
        cp.testing.assert_array_equal(t, r)


def test_stream_modify2(dtype=np.double, num_streams=2):

    x0 = cupyx.empty_pinned(shape=(4,), dtype=dtype)
    x0[:] = [0, 1, 2, 0.0]
    x1 = cupyx.empty_pinned(shape=(4,), dtype=dtype)
    x1[:] = [1, 1, 3, 1.0]
    x2 = cp.array(0.0)

    def f(
        ind_args: typing.List[cp.ndarray],
        lo: int,
        hi: int,
    ) -> None:
        nonlocal x2
        (a, b) = ind_args
        x2[...] = cp.sum(a * b) + x2

    tike.communicators.stream.stream_and_modify2(
        f,
        ind_args=[x0, x1],
        streams=[cp.cuda.Stream() for _ in range(num_streams)],
        chunk_size=2,
    )

    truth = cp.array(0 * 1 + 1 * 1 + 2 * 3 + 0 * 1.0)

    t, r = (truth, x2)
    print(t, type(t))
    print(r, type(t))
    cp.testing.assert_array_equal(t, r)
