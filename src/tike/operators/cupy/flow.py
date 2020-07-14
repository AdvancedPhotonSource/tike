# Copyright (c) 2015 Preferred Infrastructure, Inc.
# Copyright (c) 2015 Preferred Networks, Inc.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2015 Preferred Networks, Inc."

import itertools
import warnings

import cupy
import six

from tike.operators import numpy
from .operator import Operator

# This code is taken verbatim from the CuPy repository because it contains a
# bug fix which will not be released until CuPy>=8
# https://github.com/cupy/cupy/pull/2813


def _get_output(output, input, shape=None):
    if shape is None:
        shape = input.shape
    if isinstance(output, cupy.ndarray):
        if output.shape != tuple(shape):
            raise ValueError('output shape is not correct')
    else:
        dtype = output
        if dtype is None:
            dtype = input.dtype
        output = cupy.zeros(shape, dtype)
    return output


def _check_parameter(func_name, order, mode):
    if order is None:
        warnings.warn('In the current feature the default order of {} is 1. '
                      'It is different from scipy.ndimage and can change in '
                      'the future.'.format(func_name))
    elif order < 0 or 5 < order:
        raise ValueError('spline order is not supported')
    elif 1 < order:
        # SciPy supports order 0-5, but CuPy supports only order 0 and 1. Other
        # orders will be implemented, therefore it raises NotImplementedError
        # instead of ValueError.
        raise NotImplementedError('spline order is not supported')

    if mode in ('reflect', 'wrap'):
        raise NotImplementedError(
            '\'{}\' mode is not supported. See '
            'https://github.com/scipy/scipy/issues/8465'.format(mode))
    elif mode not in ('constant', 'nearest', 'mirror', 'opencv',
                      '_opencv_edge'):
        raise ValueError('boundary mode is not supported')


def map_coordinates(input,
                    coordinates,
                    output=None,
                    order=None,
                    mode='constant',
                    cval=0.0,
                    prefilter=True):
    """Map the input array to new coordinates by interpolation.
    The array of coordinates is used to find, for each point in the output, the
    corresponding coordinates in the input. The value of the input at those
    coordinates is determined by spline interpolation of the requested order.
    The shape of the output is derived from that of the coordinate array by
    dropping the first axis. The values of the array along the first axis are
    the coordinates in the input array at which the output value is found.
    Args:
        input (cupy.ndarray): The input array.
        coordinates (array_like): The coordinates at which ``input`` is
            evaluated.
        output (cupy.ndarray or ~cupy.dtype): The array in which to place the
            output, or the dtype of the returned array.
        order (int): The order of the spline interpolation. If it is not given,
            order 1 is used. It is different from :mod:`scipy.ndimage` and can
            change in the future. Currently it supports only order 0 and 1.
        mode (str): Points outside the boundaries of the input are filled
            according to the given mode (``'constant'``, ``'nearest'``,
            ``'mirror'`` or ``'opencv'``). Default is ``'constant'``.
        cval (scalar): Value used for points outside the boundaries of
            the input if ``mode='constant'`` or ``mode='opencv'``. Default is
            0.0
        prefilter (bool): It is not used yet. It just exists for compatibility
            with :mod:`scipy.ndimage`.
    Returns:
        cupy.ndarray:
            The result of transforming the input. The shape of the output is
            derived from that of ``coordinates`` by dropping the first axis.
    .. seealso:: :func:`scipy.ndimage.map_coordinates`
    """

    _check_parameter('map_coordinates', order, mode)

    if mode == 'opencv' or mode == '_opencv_edge':
        input = cupy.pad(input, [(1, 1)] * input.ndim,
                         'constant',
                         constant_values=cval)
        coordinates = cupy.add(coordinates, 1)
        mode = 'constant'

    ret = _get_output(output, input, coordinates.shape[1:])

    if mode == 'nearest':
        for i in six.moves.range(input.ndim):
            coordinates[i] = coordinates[i].clip(0, input.shape[i] - 1)
    elif mode == 'mirror':
        for i in six.moves.range(input.ndim):
            length = input.shape[i] - 1
            if length == 0:
                coordinates[i] = 0
            else:
                coordinates[i] = cupy.remainder(coordinates[i], 2 * length)
                coordinates[i] = 2 * cupy.minimum(coordinates[i],
                                                  length) - coordinates[i]

    if input.dtype.kind in 'iu':
        input = input.astype(cupy.float32)

    if order == 0:
        out = input[tuple(cupy.rint(coordinates).astype(cupy.int32))]
    else:
        coordinates_floor = cupy.floor(coordinates).astype(cupy.int32)
        coordinates_ceil = coordinates_floor + 1

        sides = []
        for i in six.moves.range(input.ndim):
            # TODO(mizuno): Use array_equal after it is implemented
            if cupy.all(coordinates[i] == coordinates_floor[i]):
                sides.append([0])
            else:
                sides.append([0, 1])

        out = cupy.zeros(coordinates.shape[1:], dtype=input.dtype)
        if input.dtype in (cupy.float64, cupy.complex128):
            weight = cupy.empty(coordinates.shape[1:], dtype=cupy.float64)
        else:
            weight = cupy.empty(coordinates.shape[1:], dtype=cupy.float32)
        for side in itertools.product(*sides):
            weight.fill(1)
            ind = []
            for i in six.moves.range(input.ndim):
                if side[i] == 0:
                    ind.append(coordinates_floor[i])
                    weight *= coordinates_ceil[i] - coordinates[i]
                else:
                    ind.append(coordinates_ceil[i])
                    weight *= coordinates[i] - coordinates_floor[i]
            out += input[ind] * weight
        del weight

    if mode == 'constant':
        mask = cupy.zeros(coordinates.shape[1:], dtype=cupy.bool_)
        for i in six.moves.range(input.ndim):
            mask += coordinates[i] < 0
            mask += coordinates[i] > input.shape[i] - 1
        out[mask] = cval
        del mask

    if ret.dtype.kind in 'iu':
        out = cupy.rint(out)
    ret[:] = out
    return ret


class Flow(Operator, numpy.Flow):

    @classmethod
    def _map_coordinates(cls, *args, **kwargs):
        return map_coordinates(*args, **kwargs)
