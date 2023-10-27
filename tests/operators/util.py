import time

import numpy as np

__author__ = "Daniel Ching, Viktor Nikitin"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'

import tike.linalg
import tike.random
import tike.precision

random_complex = tike.random.numpy_complex


def random_floating(*shape):
    return tike.random.randomizer_np.random(
        size=shape,
        dtype=tike.precision.floating,
    ) - 0.5


class OperatorTests():
    """Provide operator tests for correct adjoint and normalization."""

    def setUp(self):
        self.operator = None
        self.operator.__enter__()
        self.xp = self.operator.xp
        np.random.seed(0)
        self.m = None
        self.m_name = ''
        self.d = None
        self.d_name = ''
        self.kwargs = {}
        print(self.operator)
        raise NotImplementedError()

    def tearDown(self):
        self.operator.__exit__(None, None, None)

    def test_adjoint(self):
        """Check that the adjoint operator is correct."""
        d = self.operator.fwd(**{self.m_name: self.m}, **self.kwargs)
        assert d.shape == self.d.shape, (d.shape, self.d.shape)
        m = self.operator.adj(**{self.d_name: self.d}, **self.kwargs)
        assert m.shape == self.m.shape, (m.shape, self.m.shape)
        a = tike.linalg.inner(d, self.d)
        b = tike.linalg.inner(self.m, m)
        print()
        print('<Fm,   m> = {:.5g}{:+.5g}j'.format(a.real.item(), a.imag.item()))
        print('< d, F*d> = {:.5g}{:+.5g}j'.format(b.real.item(), b.imag.item()))
        self.xp.testing.assert_allclose(a.real, b.real, rtol=1e-3, atol=0)
        self.xp.testing.assert_allclose(a.imag, b.imag, rtol=1e-3, atol=0)

    def test_scaled(self):
        """Check that the adjoint operator is scaled."""
        # NOTE: For a linear operator to be considered 'normal', the input and
        # output spaces must be the same. That requirement is too strict for
        # all of our operators. Here we only test whether |F*Fm| = |m|.
        d = self.operator.fwd(**{self.m_name: self.m}, **self.kwargs)
        m = self.operator.adj(**{self.d_name: d}, **self.kwargs)
        a = tike.linalg.inner(m, m)
        b = tike.linalg.inner(self.m, self.m)
        print()
        # NOTE: Inner product with self is real-only magnitude of self
        print('<F*Fm, F*Fm> = {:.5g}{:+.5g}j'.format(a.real.item(), 0))
        print('<   m,    m> = {:.5g}{:+.5g}j'.format(b.real.item(), 0))
        self.xp.testing.assert_allclose(a.real, b.real, rtol=1e-3, atol=0)

    def test_fwd_time(self):
        """Time the forward operation."""
        start = time.perf_counter()
        d = self.operator.fwd(**{self.m_name: self.m}, **self.kwargs)
        elapsed = time.perf_counter() - start
        print(f"\n{elapsed:1.3e} seconds")

    def test_adj_time(self):
        """Time the adjoint operation."""
        start = time.perf_counter()
        m = self.operator.adj(**{self.d_name: self.d}, **self.kwargs)
        elapsed = time.perf_counter() - start
        print(f"\n{elapsed:1.3e} seconds")
