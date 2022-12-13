"""Compare image gradient implementations from various sources."""

import cupy as cp
import cupyx.scipy.ndimage
import libimage
import matplotlib.pyplot as plt
import numpy as np


def _image_grad(x):
    """Return the gradient of the x for each of the last two dimesions."""
    # FIXME: Use different gradient approximation that does not use FFT. Because
    # FFT caches are per-thread and per-device, using FFT is inefficient.
    ramp = 2j * cp.pi * cp.linspace(
        -0.5,
        0.5,
        x.shape[-1],
        dtype='float32',
        endpoint=False,
    )
    grad_x = np.fft.ifftn(
        ramp[:, None] * np.fft.fftn(x, axes=(-2,)),
        axes=(-2,),
    )
    grad_y = np.fft.ifftn(
        ramp * np.fft.fftn(x, axes=(-1,)),
        axes=(-1,),
    )
    return grad_x, grad_y


def _image_grad_sobel(x):
    return (
        -cupyx.scipy.ndimage.sobel(x, axis=-2, mode='nearest'),
        -cupyx.scipy.ndimage.sobel(x, axis=-1, mode='nearest'),
    )


def _image_grad_gradient(x):
    return cp.gradient(
        -x,
        axis=(-2, -1),
    )


def _image_grad_gaussian(x, s=1.0):
    """Return the gradient of the x for each of the last two dimesions."""
    return (
        -cupyx.scipy.ndimage.gaussian_filter1d(
            x, s, order=1, axis=-2, mode='nearest'),
        -cupyx.scipy.ndimage.gaussian_filter1d(
            x, s, order=1, axis=-1, mode='nearest'),
    )


def _diff(x):
    return (
        a - b for a, b in zip(_image_grad_gradient(x), _image_grad_gaussian(x)))


def test_image_grads(w=512):
    x = (libimage.load('earring', w) + np.random.normal(size=(w, w)) + 1j *
         (libimage.load('satyre', w) + np.random.normal(size=(w, w))))
    x = cp.asarray(x)

    for grad in [
            _image_grad,
            _image_grad_gradient,
            _image_grad_sobel,
            _image_grad_gaussian,
            _diff,
    ]:

        dx, dy = grad(x)
        dx = dx.get()
        dy = dy.get()

        f = plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(dx.imag)
        plt.colorbar()
        plt.subplot(2, 2, 2)
        plt.imshow(dy.imag)
        plt.colorbar()
        plt.subplot(2, 2, 3)
        plt.imshow(dx.real)
        plt.colorbar()
        plt.subplot(2, 2, 4)
        plt.imshow(dy.real)
        plt.colorbar()
        plt.savefig(f'{grad.__name__}.png')
