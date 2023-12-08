import typing
import math

import cupy as cp
import numpy.typing as npt


def stream_and_reduce(
    f: typing.Callable[[npt.NDArray], typing.Tuple[npt.NDArray, ...]],
    args: typing.List[npt.NDArray],
    y_shapes: typing.List[typing.List[int]],
    y_dtypes: typing.List[npt.DTypeLike],
    streams: typing.List[cp.cuda.Stream] = [],
    indices: typing.Union[None, typing.List[int]] = None,
    *,
    chunk_size: int = 64,
) -> typing.List[npt.NDArray]:
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
        A list of two CUDA streams to use for streaming
    indices:
        A list of indices to use instead of range(0, N) for slices of args
    chunk_size:
        The number of slices out of N to process at one time

    Example
    -------
    .. code-block:: python
        :linenos:

        import numpy as np

        def f(a, b, c):
            return a, b*c, b+c

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

    """
    if not streams:
        streams = [cp.cuda.Stream(), cp.cuda.Stream()]
    if indices is None:
        N = len(args[0])
        indices = list(range(N))
    else:
        N = len(indices)
    chunk_size = min(chunk_size, N)
    num_streams = 2

    args_gpu = [
        cp.empty_like(
            x,
            shape=(num_streams * chunk_size, *x.shape[1:]),
        ) for x in args
    ]
    y_sums = [
        cp.zeros(dtype=d, shape=(num_streams, *s))
        for d, s in zip(y_dtypes, y_shapes)
    ]

    for s, i in enumerate(range(0, N, chunk_size)):
        buffer_index = s % num_streams

        indices_chunk = indices[i:i + chunk_size]
        buflo = buffer_index * chunk_size
        bufhi = buflo + len(indices_chunk)

        with streams[0]:
            for x_gpu, x in zip(args_gpu, args):
                # Use a range because set() needs an array always; never scalar
                if isinstance(x, cp.ndarray):
                    x_gpu[buflo:bufhi] = x[indices_chunk]
                else:  # x is a pinned np.ndarray
                    x_gpu[buflo:bufhi].set(x[indices_chunk])

        streams[0].synchronize()
        streams[1].synchronize()

        with streams[1]:
            results = f(*(x_gpu[buflo:bufhi] for x_gpu in args_gpu))

            for y_sum, y in zip(y_sums, results):
                y_sum[buffer_index] += y

    streams[1].synchronize()

    return [y_sum.sum(axis=0, dtype=d) for y_sum, d in zip(y_sums, y_dtypes)]


def stream_and_modify(
    f: typing.Callable[
        [typing.List[npt.NDArray], typing.Tuple, typing.List[int]],
        typing.Tuple
    ],
    ind_args: typing.List[npt.NDArray],
    mod_args: typing.Tuple,
    streams: typing.List[cp.cuda.Stream] = [],
    indices: typing.Union[None, typing.List[int]] = None,
    *,
    chunk_size: int = 64,
) -> typing.Tuple:
    """Use multiple CUDA streams to load data for a function.

    The following calls to f should be equivalent.

    mod_args = f(ind_args, mod_args, ind)

    for i in range(N):
        mod_args = f([x[i:i+1] for x in ind_args], mod_args, [i,])

    Parameters
    ----------
    f:
        A function that takes the ind_args and mod_args as parameters
    ind_args: [(N, ...) array, (N, ...) array, ...]
        A list of pinned arrays that can be sliced along the 0-th dimension for
        work. If you have constant args that are not sliced, Use a wrapper
        function
    mod_args:
        A tuple of args that are modified across calls to f
    streams:
        A list of two CUDA streams to use for streaming
    indices:
        A list of indices to use instead of range(0, N) for slices of args
    chunk_size:
        The number of slices out of N to process at one time

    Example
    -------
    .. code-block:: python
        :linenos:

        import numpy as np

        def f(ind_args, mod_args, _):
            (a, b), (c,) = ind_args, mod_args
            return (np.sum(a * b) + c,)

        x0 = np.array([0, 1, 2, 0])
        x1 = np.array([1, 1, 3, 1])
        x2 = 0
        ind_args = (x0, x1)
        mod_args = (x2,)

        truth = (0*1 + 1*1 + 2*3 + 0*1,)

        for i in range(N):
            mod_args = f(
                [x[i:i+1] for x in ind_args],
                mod_args,
                [i],
            )

        assert mod_args == truth

    """
    if not streams:
        streams = [cp.cuda.Stream(), cp.cuda.Stream()]
    if indices is None:
        N = len(ind_args[0])
        indices = list(range(N))
    else:
        N = len(indices)
    chunk_size = min(chunk_size, N)
    num_streams = 2

    ind_args_gpu = [
        cp.empty_like(
            x,
            shape=(num_streams * chunk_size, *x.shape[1:]),
        ) for x in ind_args
    ]

    for s, i in enumerate(range(0, N, chunk_size)):
        buffer_index = s % num_streams

        indices_chunk = indices[i:i + chunk_size]
        buflo: int = buffer_index * chunk_size
        bufhi: int = buflo + len(indices_chunk)

        with streams[0]:
            for x_gpu, x in zip(ind_args_gpu, ind_args):
                # Use a range because set() needs an array always; never scalar
                if isinstance(x, cp.ndarray):
                    x_gpu[buflo:bufhi] = x[indices_chunk]
                else:  # x is a pinned np.ndarray
                    x_gpu[buflo:bufhi].set(x[indices_chunk])

        streams[0].synchronize()
        streams[1].synchronize()

        with streams[1]:
            mod_args = f(
                [x_gpu[buflo:bufhi] for x_gpu in ind_args_gpu],
                mod_args,
                indices_chunk,
            )

    streams[1].synchronize()

    return mod_args


def stream_and_modify_debug(
    f: typing.Callable[
        [typing.List[npt.NDArray], typing.Tuple, typing.List[int]],
        typing.Tuple
    ],
    ind_args: typing.List[npt.NDArray],
    mod_args: typing.Tuple,
    streams: typing.List[cp.cuda.Stream] = [],
    indices: typing.Union[None, typing.List[int]] = None,
    *,
    chunk_size: int = 64,
) -> typing.Tuple:
    """Same as stream_and_modify but without CUDA streams

    Parameters
    ----------
    f:
        A function that takes the ind_args and mod_args as parameters
    ind_args: [(N, ...) array, (N, ...) array, ...]
        A list of pinned arrays that can be sliced along the 0-th dimension for
        work. If you have constant args that are not sliced, Use a wrapper
        function
    mod_args:
        A tuple of args that are modified across calls to f
    streams:
        A list of two CUDA streams to use for streaming
    indices:
        A list of indices to use instead of range(0, N) for slices of args
    chunk_size:
        The number of slices out of N to process at one time

    """
    if not streams:
        streams = [cp.cuda.Stream(), cp.cuda.Stream()]
    if indices is None:
        N = len(ind_args[0])
        indices = list(range(N))
    else:
        N = len(indices)
    chunk_size = min(chunk_size, N)

    ind_args_gpu = [
        cp.asarray(
            x[indices],
        ) for x in ind_args
    ]

    for s, i in enumerate(range(0, N, chunk_size)):
        indices_chunk = indices[i:i + chunk_size]

        mod_args = f(
            [x_gpu[i:i + chunk_size] for x_gpu in ind_args_gpu],
            mod_args,
            indices_chunk,
        )

    return mod_args
