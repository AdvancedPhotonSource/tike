"""Linear algebra routines with broadcasting and complex value support.

This module exists because support for broadcasting and complex values is
spotty in the NumPy and CuPy libraries.
"""

import numpy as np


def mnorm(x, axis=None, keepdims=False):
    """Return the vector 2-norm of x but replace sum with mean."""
    return np.sqrt(np.mean((x * x.conj()).real, axis=axis, keepdims=keepdims))


def norm(x, axis=None, keepdims=False):
    """Return the vector 2-norm of x along given axis."""
    return np.sqrt(np.sum((x * x.conj()).real, axis=axis, keepdims=keepdims))


def projection(a, b, axis=None):
    """Return complex vector projection of a onto b for along given axis."""
    bh = b / inner(b, b, axis=axis, keepdims=True)
    return inner(a, b, axis=axis, keepdims=True) * bh


def inner(x, y, axis=None, keepdims=False):
    """Return the complex inner product; the order of the operands matters."""
    return (x * y.conj()).sum(axis=axis, keepdims=keepdims)


def lstsq(a, b, weights=None):
    """Return the least-squares solution for a @ x = b.

    This implementation, unlike cp.linalg.lstsq, allows a stack of matricies to
    be processed simultaneously. The input sizes of the matricies are as
    follows:
        a (..., M, N)
        b (..., M, K)
        x (..., N, K)

    Optionally include weights (..., M) for weighted-least-squares if the
    errors are uncorrelated.

    ...seealso:: https://github.com/numpy/numpy/issues/8720
                 https://github.com/cupy/cupy/issues/3062
    """
    # TODO: Using 'out' parameter of cp.matmul() may reduce memory footprint
    assert a.shape[:-1] == b.shape[:-1], (f"Leading dims of a {a.shape}"
                                          f"and b {b.shape} must be same!")
    if weights is not None:
        assert weights.shape == a.shape[:-1]
        a = a * np.sqrt(weights[..., None])
        b = b * np.sqrt(weights[..., None])
    aT = hermitian(a)
    x = np.linalg.inv(aT @ a) @ aT @ b
    return x


def orthogonalize_gs(x, axis=-1, N=None):
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
    N : int
        The axis along which to orthogonalize. Other dimensions are broadcast.
    """
    # Find N, the last dimension not included in axis; we iterate over N
    # vectors in the Gram-schmidt algorithm. Dimensions that are not N or
    # included in axis are leading dimensions for broadcasting.
    try:
        axis = tuple(a % x.ndim for a in axis)
    except TypeError:
        axis = (axis % x.ndim,)
    if N is None:
        N = x.ndim - 1
        while N in axis:
            N -= 1
    N = N % x.ndim
    if N in axis:
        raise ValueError("Cannot orthogonalize a single vector.")
    # Move axis N to the front for convenience
    x = np.moveaxis(x, N, 0)
    u = x.copy()
    for i in range(1, len(x)):
        u[i:] -= projection(x[i:], u[i - 1:i], axis=axis)
    return np.moveaxis(u, 0, N)


def hermitian(x):
    """Compute the conjugate transpose of x along last two dimensions."""
    return x.conj().swapaxes(-1, -2)


def cov(x):
    """Compute the covariance of x with observations along axis -2."""
    x0 = x - np.mean(x, axis=-2, keepdims=True)
    return hermitian(x0) @ x0


def pca_eig(data, k):
    """Return k principal components via Eigen decomposition.

    Parameters
    ----------
    data (..., N, D)
        Array of N observations of a D dimensional space.

    Returns
    -------
    S (..., k)
        The singular values corresponding to the current principal components
        sorted largest to smallest.
    U (..., D, k)
        The current best principal components of the population.
    """
    S, U = np.linalg.eigh(cov(data))
    # eigh() API states that values returned in acending order. i.e.
    # the best vectors are last.
    U = U[..., -1:-(k + 1):-1]
    S = S[..., -1:-(k + 1):-1]
    return S, U
