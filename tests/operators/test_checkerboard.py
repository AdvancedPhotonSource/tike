"""Check whether the checkerboard algorithm is equivalent to fftshift."""
import numpy as np
import cupy as cp
from cupyx.scipy.fft import fftn, ifftn
from cupyx.scipy.fftpack import get_fft_plan
from cupy.cuda import memory_hooks

from tike.operators.numpy.usfft import checkerboard


def fftshift_two(a):
    return checkerboard(
        cp,
        fftn(
            checkerboard(cp, a),
            norm='ortho',
            overwrite_x=True,
        ),
        inverse=True,
    )


def shifted_fft_ref(a):
    return np.fft.ifftshift(np.fft.fftn(
        np.fft.fftshift(a),
        norm='ortho',
    ))


if __name__ == "__main__":
    shape = np.random.randint(1, 32, 3) * 2
    a = cp.random.rand(*shape) + 1j * cp.random.rand(*shape)
    a = a.astype('complex64')
    b = a.copy()
    plan = get_fft_plan(a)

    mempool = cp.get_default_memory_pool()

    mempool.free_all_blocks()
    hook = memory_hooks.LineProfileHook()
    with hook, plan:
        b = fftshift_two(b)
    hook.print_report()

    print()

    mempool.free_all_blocks()
    hook = memory_hooks.LineProfileHook()
    with hook, plan:
        a = shifted_fft_ref(a)
    hook.print_report()

    cp.testing.assert_array_almost_equal(a, b, decimal=5)
