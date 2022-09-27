import cupy as cp
import tike.random

def complex_multiply(x, y):
    return x.real * y.real - x.imag * y.imag + 1j * (x.real * y.imag + x.imag * y.real)

def test_complex_multiply():

    x = tike.random.cupy_complex(33).astype('complex64') * 100000
    y = tike.random.cupy_complex(33).astype('complex64') * 100000

    z = x * y
    z1 = complex_multiply(x, y)

    cp.testing.assert_allclose(z, z1, rtol=1e-4)
