"""Linear algebra routines with broadcasting and complex value support.

This module exists because support for broadcasting and complex values is
spotty in the NumPy and CuPy libraries.
"""

import numpy as np


def projection(a, b, axis=None):
    """Return complex vector projection of a onto b for along given axis."""
    # NOTE: Using np.linalg norm here is not correct because of complex values
    bh = b / inner(b, b, axis=axis, keepdims=True)
    return inner(b, a, axis=axis, keepdims=True) * bh


def inner(x, y, axis=None, keepdims=False):
    """Return the complex inner product; the order of the operands matters."""
    return (x.conj() * y).sum(axis=axis, keepdims=keepdims)


def lstsq(a, b):
    """Return the least-squares solution for a @ x = b.

    This implementation, unlike cp.linalg.lstsq, allows a stack of matricies to
    be processed simultaneously. The input sizes of the matricies are as
    follows:
        a (..., M, N)
        b (..., M)
        x (..., N)

    ...seealso:: https://github.com/numpy/numpy/issues/8720
                 https://github.com/cupy/cupy/issues/3062
    """
    # TODO: Using 'out' parameter of cp.matmul() may reduce memory footprint
    assert a.shape[:-1] == b.shape, (f"Leading dims of a {a.shape}"
                                     f"and b {b.shape} must be same!")
    aT = a.swapaxes(-2, -1)
    x = np.linalg.inv(aT @ a) @ aT @ b[..., None]
    return x[..., 0]


def orthogonalize_gs(x, axis=-1):
    """Gram-schmidt orthogonalization for complex arrays.

    Parameters
    ----------
    x : (..., D) array_like
        Array containing dimensions to be orthogonalized.
    axis : int or tuple(int)
        The axis/axes to be orthogonalized. By default only the last axis
        is orthogonalized. If axis is a tuple, then the number of
        orthogonal vectors is the length of the last dimension not included in
        axis. The other dimensions are broadcast.
    """
    # Find N, the last dimension not included in axis; we iterate over N
    # vectors in the Gram-schmidt algorithm. Dimensions that are not N or
    # included in axis are leading dimensions for broadcasting.
    try:
        axis = [a % x.ndim for a in axis]
    except TypeError:
        axis = [axis % x.ndim]
    N = x.ndim - 1
    while N in axis:
        N -= 1
    if N < 0:
        raise ValueError("Cannot orthogonalize a single vector.")
    # Move axis N to the front for convenience
    x = np.moveaxis(x, N, 0)
    u = x.copy()
    for i in range(1, len(x)):
        u[i:] -= projection(x[i:], u[i - 1:i], axis=axis)
    if __debug__:
        # Test each pair of vectors for orthogonality
        for i in range(len(u)):
            for j in range(i):
                error = abs(inner(u[i:i + 1], u[j:j + 1], axis=axis))
                if np.any(error > 1e-12):
                    raise RuntimeWarning(
                        'Some vectors were not orthogonalized.', error)
    return np.moveaxis(u, 0, N)


def orthogonalize_eig(x):
    """Orthogonalize modes of x using eigenvectors of the pairwise dot product.

    Parameters
    ----------
    x : (..., nmodes, :, :) array_like complex64
        An array of the probe modes vectorized

    References
    ----------
    M. Odstrcil, P. Baksh, S. A. Boden, R. Card, J. E. Chad, J. G. Frey, W. S.
    Brocklesby, "Ptychographic coherent diffractive imaging with orthogonal
    probe relaxation." Opt. Express 24, 8360 (2016). doi: 10.1364/OE.24.008360
    """
    nmodes = x.shape[-3]
    # 'A' holds the dot product of all possible mode pairs. We only fill the
    # lower half of `A` because it is conjugate-symmetric
    A = cp.empty((*x.shape[:-3], nmodes, nmodes), dtype='complex64')
    for i in range(nmodes):
        for j in range(i + 1):
            A[..., i, j] = cp.sum(cp.conj(x[..., i, :, :]) * x[..., j, :, :],
                                  axis=(-1, -2))

    _, vectors = cp.linalg.eigh(A, UPLO='L')
    # np.linalg.eigh guarantees that the eigen values are returned in ascending
    # order, so we just reverse the order of modes to have them sorted in
    # descending order.

    # TODO: Optimize this double-loop
    x_new = cp.zeros_like(x)
    for i in range(nmodes):
        for j in range(nmodes):
            # Sort new modes by eigen value in decending order.
            x_new[..., nmodes - 1 -
                  j, :, :] += vectors[..., i, j, None, None] * x[..., i, :, :]
    assert x_new.shape == x.shape, [x_new.shape, x.shape]

    return x_new
