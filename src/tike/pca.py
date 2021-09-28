import unittest
from unittest.case import skip

import numpy as np

from tike.linalg import hermitian as _hermitian
from tike.linalg import pca_eig


def pca_incremental(data, k, S=None, U=None):
    """Principal component analysis with method IIIB from Arora et al (2012).

    This method iteratively updates a current guess for the principal
    components of a population.

    Parameters
    ----------
    data (..., N, D)
        Array of N observations of a D dimensional space.
    k : int
        The desired number of principal components

    Returns
    -------
    S (..., k)
        The singular values corresponding to the current principal components
        sorted largest to smallest.
    U (..., D, k)
        The current best principal components of the population.

    References
    ----------
    Arora, R., Cotter, A., Livescu, K., & Srebro, N. (2012). Stochastic
    optimization for PCA and PLS. 2012 50th Annual Allerton Conference on
    Communication, Control, and Computing (Allerton), 863.
    https://doi.org/10.1109/Allerton.2012.6483308

    """
    ndim = data.shape[-1]
    nsamples = data.shape[-2]
    lead = data.shape[:-2]
    if S is None or U is None:
        # TODO: Better inital guess for one sample?
        S = np.ones((*lead, k), dtype=data.dtype)
        U = np.zeros((*lead, ndim, k), dtype=data.dtype)
        U[..., list(range(k)), list(range(k))] = 1
    if S.shape != (*lead, k):
        raise ValueError('S is the wrong shape', S.shape)
    if U.shape != (*lead, ndim, k):
        raise ValueError('U is the wrong shape', U.shape)

    for m in range(nsamples):

        x = data[..., m, :, None]

        # (k, d) x (d, 1)
        xy = _hermitian(U) @ x

        # (d, 1) - (d, d) x (d, 1)
        xp = x - U @ _hermitian(U) @ x

        # (..., 1, 1)
        norm_xp = np.linalg.norm(xp, axis=-2, keepdims=True)

        # [
        #   (k, k), (k, 1)
        #   (1, k), (1, 1)
        # ]
        Q = np.empty(shape=(*lead, k + 1, k + 1), dtype=x.dtype)
        Q[..., :-1, :-1] = xy @ _hermitian(xy)
        Q[..., -1:, -1:] = norm_xp * norm_xp
        for i in range(k):
            Q[..., i, i] = S[..., i]
        Q[..., :-1, -1:] = norm_xp * xy
        # Skip one assignment because matrix is conjugate symmetric
        # Q[..., -1:, :-1] = norm_xp * _hermitian(xy)
        S1, U1 = np.linalg.eigh(Q, UPLO='U')

        # [(d, k), (d, 1)] x (k + 1, k + 1)
        Utilde = np.concatenate([U, xp / norm_xp], axis=-1) @ U1
        Stilde = S1

        # Skip sorting because eigh() guarantees vectors already sorted
        # order = np.argsort(Stilde, axis=-1)
        # Stilde = np.take_along_axis(Stilde, order, axis=-1)
        # Utilde = np.take_along_axis(Utilde, order[..., None, :], axis=-1)
        S, U = Stilde[..., -1:-(k + 1):-1], Utilde[..., -1:-(k + 1):-1]

    return S, U


def pca_svd(data, k):
    """Return k principal components via singular value decomposition.

    Parameters
    ----------
    data (..., N, D)
        Array of N observations of a D dimensional space.

    Returns
    -------
    W (..., N, k)
        The weights projecting the original observations onto k-fold subspace
    C (..., k, D)
        The k principal components sorted largest to smallest.

    """
    U, S, Vh = np.linalg.svd(data, full_matrices=False, compute_uv=True)
    assert data.shape == ((U * S[..., None, :]) @ Vh).shape
    # svd API states that values returned in descending order. i.e.
    # the best vectors are first.
    U = U[..., :k]
    S = S[..., None, :k]
    Vh = Vh[..., :k, :]
    return U * S, Vh


class TestPrincipalComponentAnalysis(unittest.TestCase):

    def setUp(self, batch=2, sample=100, dimensions=4):
        # generates some random uncentered data that is strongly biased towards
        # having principal components. The first batch is flipped so the
        # principal components start at the last dimension.
        np.random.seed(0)
        self.data = np.random.normal(
            np.random.rand(dimensions),
            10 / (np.arange(dimensions) + 1),
            size=[batch, sample, dimensions],
        )
        self.data[0] = self.data[0, ..., ::-1]

    def print_metrics(self, W, C, k):
        I = C @ _hermitian(C)
        np.testing.assert_allclose(
            I,
            np.tile(np.eye(k), (C.shape[0], 1, 1)),
            atol=1e-12,
        )
        print(
            'reconstruction error: ',
            np.linalg.norm(W @ C - self.data, axis=(1, 2)),
        )
    @unittest.skip("Broken due to tsting API change.")
    def test_numpy_eig(self, k=2):
        S, U = pca_eig(self.data, k)
        print('EIG COV principal components\n', U)
        self.print_metrics(U, k)

    def test_numpy_svd(self, k=2):
        W, C = pca_svd(self.data, k)
        print('SVD principal components\n', C)
        self.print_metrics(W, C, k)

    @unittest.skip("Broken due to tsting API change.")
    def test_incremental_pca(self, k=2):
        S, U = pca_incremental(self.data, k=k)

        print('INCREMENTAL principal components\n', U)
        self.print_metrics(U, k)


if __name__ == "__main__":
    unittest.main()
