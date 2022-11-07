import cupy as cp
import tike.random
import tike.view
import matplotlib.pyplot as plt

def complex_multiply_naive(x, y):
    return x.real * y.real - x.imag * y.imag + 1j * (x.real * y.imag + x.imag * y.real)

def complex_multiply3(x, y):
    ac = x.real * y.real
    bd = x.imag * y.imag
    ab = x.real + x.imag
    cd = y.real + y.imag
    return (ac - bd) + 1j * ( ab * cd - ac - bd)

def test_complex_multiply():

    x = tike.random.cupy_complex(64, 64).astype('complex128')
    y = tike.random.cupy_complex(64, 64).astype('complex128')
    z0 = complex_multiply_naive(x, y)
    z1 = x * y
    z2 = complex_multiply3(x, y)

    print('native ', cp.mean(cp.abs((z1-z0)/z0)))
    print('optimal ' , cp.mean(cp.abs((z2-z0)/z0)))

    x = x.astype('complex64')
    y = y.astype('complex64')
    z0 = complex_multiply_naive(x, y)
    z1 = x * y
    z2 = complex_multiply3(x, y)

    print('native ', cp.mean(cp.abs((z1-z0)/z0)))
    print('optimal ' , cp.mean(cp.abs((z2-z0)/z0)))
