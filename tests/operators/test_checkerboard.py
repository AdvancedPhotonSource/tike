"""Check whether the checkerboard algorithm is equivalent to fftshift."""
import numpy as xp

from tike.operators.cupy.usfft import checkerboard


def shifted_fft_two(a, xp):
    return checkerboard(
        xp,
        xp.fft.fftn(
            checkerboard(xp, a),
            norm='ortho',
        ),
        inverse=True,
    )


def shifted_fft_ref(a, xp):
    return xp.fft.ifftshift(xp.fft.fftn(
        xp.fft.fftshift(a),
        norm='ortho',
    ))


def test_checkerboard_correctness():
    shape = xp.random.randint(1, 32, 3) * 2
    a = xp.random.rand(*shape) + 1j * xp.random.rand(*shape)
    a = a.astype('complex64')
    b = a.copy()
    a = shifted_fft_ref(a, xp)
    b = shifted_fft_two(b, xp)
    xp.testing.assert_array_almost_equal(a, b, decimal=5)
