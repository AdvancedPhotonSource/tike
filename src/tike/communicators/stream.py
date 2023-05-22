import typing

import cupy as cp
import cupyx
import numpy as np
import numpy.typing as npt


def stream_and_reduce(
    f: typing.Callable[[npt.NDArray], typing.Tuple[npt.NDArray, ...]],
    args: typing.List[npt.NDArray],
    y_shapes: typing.List[typing.List[int]],
    y_dtypes: typing.List[npt.DTypeLike],
    streams: typing.List[cp.cuda.Stream] = [cp.cuda.Stream()],
    indices: typing.Union[None, typing.List[int]] = None,
) -> npt.NDArray:
    """Use multiple CUDA streams to compute sum(f(x), axis=0).

    Equivalent to the following expression:

    .. code-block:: python

        [np.sum(y, axis=0) for y in zip(*[f(*x) for x in zip(*args)])]

    Parameters
    ----------
    f:
        A function that takes the zipped elements of args as a parameter
    args: [(N, ...) array, (N, ...) array, ...]
        A list of pinned arrays that can be sliced along the 0-th dimension for
        work. If you have constant args that are not sliced, Use a wrapper
        function
    y_shapes:
        The shape of the output of f(args)
    y_dtypes:
        The dtypes of the outputs of f(args)
    streams:
        A list of CUDA streams to use for streaming
    indices:
        A list of indices to use instead of range(0, N) for slices of args

    Example
    -------
    .. code-block:: python
        :linenos:

        import numpy as np

        def f(a, b, c):
            return a, b*c, b+c

        x0 = np.array([0, 1, 0, 0]) x1 = np.array([1, 1, 3, 1]) x2 =
        np.array([2, 2, 7, 2]) args = [x0, x1, x2]

        truth = [
            1, 2 + 2 + 21 + 2, 3 + 3 + 10 + 3,
        ]

        result = [np.sum(y, axis=0) for y in zip(*[f(*x) for x in zip(*args)])]

    """
    if indices is None:
        num_slices = len(args[0])
        indices = range(num_slices)
    else:
        num_slices = len(indices)
    num_streams = len(streams)
    num_buffer = min(num_streams, num_slices)

    args_gpu = [cp.empty_like(x[0:num_buffer]) for x in args]
    y_sums = [
        cp.zeros(dtype=d, shape=(num_buffer, *s))
        for d, s in zip(y_dtypes, y_shapes)
    ]

    for s, i in enumerate(indices):
        stream_index = s % num_streams
        with streams[stream_index]:
            for x_gpu, x in zip(args_gpu, args):
                # Use a range because set() needs an array always; never scalar
                if isinstance(x, cp.ndarray):
                    x_gpu[stream_index:stream_index + 1] = x[i:i + 1]
                else: # np.ndarray
                    x_gpu[stream_index:stream_index + 1].set(x[i:i + 1])

            results = f(
                *(x_gpu[stream_index:stream_index + 1] for x_gpu in args_gpu))

            for y_sum, y in zip(y_sums, results):
                y_sum[stream_index] += y

    y_sums_pinned = [
        cp.empty(dtype=d, shape=s)
        for d, s in zip(y_dtypes, y_shapes)
    ]

    [stream.synchronize() for stream in streams]

    for y_sum, y_sum_pinned, d in zip(y_sums, y_sums_pinned, y_dtypes):
        # y_sum.sum(axis=0, dtype=d).get(out=y_sum_pinned)
        y_sum_pinned[:] = y_sum.sum(axis=0, dtype=d)

    return y_sums_pinned
