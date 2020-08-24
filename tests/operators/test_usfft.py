import lzma
import os
import pickle

import cupy as cp
import matplotlib.pyplot as plt

import tike.view
from tike.operators.cupy.usfft import eq2us2d, us2eq2d, eq2us

testdir = os.path.dirname(os.path.dirname(__file__))


def fft3(a):
    assert a.shape[0] == a.shape[1] == a.shape[2]
    assert a.ndim == 3
    # We have to shift the input so the zero frequency is at the center because
    # that is what the usfft functions assume. Otherwise we get a
    # checkerboarding effect because input is padded incorrectly.
    a = cp.fft.fftshift(a)
    n = a.shape[0]
    # Shift grid coordinates to corner. FIXME: Not correct for odd n.
    grid = cp.roll(cp.arange(-n // 2, n // 2) / n, n // 2)
    frequencies = cp.meshgrid(grid, grid, grid, indexing='ij')
    x = cp.stack(
        [cp.ravel(s).astype('float32') for s in frequencies],
        axis=-1,
    )
    A = eq2us(a, x, n, eps=1e-6, xp=cp)
    return A.reshape(n, n, n)


def fft2(xp, a, norm=None):
    n = a.shape[1]
    a = xp.fft.fftshift(a, axes=(-1, -2))
    [kv, ku] = xp.mgrid[-n // 2:n // 2, -n // 2:n // 2] / n
    ku = xp.fft.fftshift(ku)
    kv = xp.fft.fftshift(kv)
    ku = ku.ravel().astype('float32')
    kv = kv.ravel().astype('float32')
    x = xp.stack((kv, ku), axis=-1)
    F = xp.zeros(a.shape, dtype='complex64')
    for k in range(a.shape[0]):
        F[k] = eq2us2d(a[k], x, a.shape[1], 1e-6, xp=xp).reshape(n, n)
    if norm == 'ortho':
        F /= n
    return F


def ifft2(xp, a, norm=None):
    n = a.shape[1]
    [kv, ku] = xp.mgrid[-n // 2:n // 2, -n // 2:n // 2] / n
    ku = xp.fft.ifftshift(ku)
    kv = xp.fft.ifftshift(kv)
    ku = ku.ravel().astype('float32')
    kv = kv.ravel().astype('float32')
    x = xp.stack((kv, ku), axis=-1)
    F = xp.zeros(a.shape, dtype='complex64')
    a = a.reshape(a.shape[0], -1)
    for k in range(a.shape[0]):
        F[k] = us2eq2d(a[k], -x, n, 1e-6, xp=xp)
    if norm == 'ortho':
        F /= n
    else:
        F /= n * n
    F = xp.fft.ifftshift(F, axes=(-1, -2))
    return F


def test_2DFFT_forward(
    N=64,
    dataset_file=os.path.join(testdir, 'data/nalm256.pickle.lzma'),
    norm='ortho',
    show=False,
):
    """Test whether EQUSFFT is equivalent to FFT for regular frequencies."""
    with lzma.open(dataset_file, 'rb') as file:
        a = pickle.load(file)
    a = cp.asarray(a[::4, ::4, ::4], dtype='complex64')

    esfft = cp.fft.fft2(a, norm=norm).astype('complex64')
    usfft = fft2(cp, a, norm=norm)

    assert esfft.shape == usfft.shape
    assert esfft.dtype == usfft.dtype

    if show:
        plt.figure()
        tike.view.plot_complex(cp.fft.ifft2(esfft[N // 2], norm=norm).get())
        plt.title("Equally spaced")
        plt.figure()
        tike.view.plot_complex(cp.fft.ifft2(usfft[N // 2], norm=norm).get())
        plt.title("Unequally spaced")
        plt.figure()
        tike.view.plot_complex((cp.fft.ifft2(usfft[N // 2] - esfft[N // 2],
                                             norm=norm)).get())
        plt.title("Difference")
        plt.show()
    cp.testing.assert_allclose(esfft, usfft, atol=1e-3)


def test_2DFFT_inverse(
    N=64,
    dataset_file=os.path.join(testdir, 'data/nalm256.pickle.lzma'),
    norm='ortho',
    show=False,
):
    """Test whether USEQFFT is equivalent to FFT for regular frequencies."""
    with lzma.open(dataset_file, 'rb') as file:
        a = pickle.load(file)
    a = cp.asarray(a[::4, ::4, ::4], dtype='complex64')

    esfft = cp.fft.ifft2(a, norm=norm).astype('complex64')
    usfft = ifft2(cp, a, norm=norm)

    assert esfft.shape == usfft.shape
    assert esfft.dtype == usfft.dtype

    if show:
        plt.figure()
        tike.view.plot_complex(cp.fft.fft2(esfft[N // 2], norm=norm).get())
        plt.title("Equally spaced")
        plt.figure()
        tike.view.plot_complex(cp.fft.fft2(usfft[N // 2], norm=norm).get())
        plt.title("Unequally spaced")
        plt.figure()
        tike.view.plot_complex((cp.fft.fft2(usfft[N // 2] - esfft[N // 2],
                                            norm=norm)).get())
        plt.title("Difference")
        plt.show()
    cp.testing.assert_allclose(esfft, usfft, atol=1e-6)


if __name__ == "__main__":
    test_2DFFT_forward(norm='ortho')
    test_2DFFT_inverse(norm='ortho')
