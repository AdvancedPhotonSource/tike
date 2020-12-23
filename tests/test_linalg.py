import unittest

import cupy as cp

import tike.ptycho
import tike.linalg
import tike.random


def test_norm():
    # Complex inner product is equal to square of complex norm
    a = tike.random.cupy_complex(5)
    assert 1.0 == cp.linalg.norm(a / cp.linalg.norm(a))
    cp.testing.assert_allclose(
        cp.sqrt(tike.linalg.inner(a, a)),
        cp.linalg.norm(a),
    )


def test_lstsq():
    a = tike.random.cupy_complex(5, 1, 4, 3, 3)
    x = tike.random.cupy_complex(5, 1, 4, 3, 1)
    b = a @ x
    x1 = tike.linalg.lstsq(a, b[..., 0])
    cp.testing.assert_allclose(x1, x[..., 0])


def test_projection():
    # Tests that we can make an orthogonal vector with this projection operator
    a = tike.random.cupy_complex(5)
    b = tike.random.cupy_complex(5)
    pab = tike.linalg.projection(a, b)
    pba = tike.linalg.projection(b, a)
    assert abs(tike.linalg.inner(a - pab, b)) < 1e-12
    assert abs(tike.linalg.inner(a, b - pba)) < 1e-12


class Orthogonal(unittest.TestCase):

    def setUp(self):
        self.x = tike.random.cupy_complex(1, 4, 3, 3)

    def test_gram_schmidt_single_vector(self):
        with self.assertRaises(ValueError):
            y = tike.linalg.orthogonalize_gs(self.x, axis=(0, 1, 2, 3))

    def test_gram_schmidt_single_axis(self):
        y = tike.linalg.orthogonalize_gs(self.x)
        assert self.x.shape == y.shape

    def test_gram_schmidt_multi_axis(self):
        y = tike.linalg.orthogonalize_gs(self.x, axis=(1, -1))
        assert self.x.shape == y.shape
