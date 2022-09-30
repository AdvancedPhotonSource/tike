import os

import cupy as cp
import h5py
import matplotlib.pyplot as plt
import numpy as np
import tike.random
import tike.view

_dir = os.path.dirname(__file__)
_test = os.path.join(os.path.dirname(_dir), 'result', 'matlab')

def test_consistent_fft():

    with h5py.File(os.path.join(_dir, 'fft.mat'), 'r') as inputs:
        x = inputs['x0_double'][...].view('complex128')
        print(x.shape, x.dtype)

        y = inputs['y_double'][...].view('complex128')
        print(y.shape, y.dtype)

    y1 = cp.fft.ifft2(cp.fft.fft2(cp.asarray(x, order='C'))).get()

    plt.figure()
    tike.view.plot_complex(np.fft.ifftshift(y1[0]))
    plt.savefig(os.path.join(_test, 'fft-00.png'))

    plt.figure()
    tike.view.plot_complex(np.fft.ifftshift(y1[0] - y[0]))
    plt.suptitle('FFT Error between MATLAB and CUPY for Double Precision.')
    plt.savefig(os.path.join(_test, 'fft-01.svg'))

    np.testing.assert_array_equal(y1, y)

    with h5py.File(os.path.join(_dir, 'fft.mat'), 'r') as inputs:
        x0 = inputs['x0_single'][...].view('complex64')
        print(x0.shape, x0.dtype)

        x = inputs['x_single'][...].view('complex64')
        print(x.shape, x.dtype)

        y = inputs['y_single'][...].view('complex64')
        print(y.shape, y.dtype)

    y1 = cp.fft.fft2(cp.asarray(x0, order='C'), norm='backward')
    x1 = cp.fft.ifft2(y1, norm='backward')

    x1 = x1.get()
    y1 = y1.get()

    plt.figure()
    tike.view.plot_complex(np.fft.ifftshift(y1[0]))
    plt.savefig(os.path.join(_test, 'fft-02.png'))

    plt.figure()
    plt.suptitle('FFT Error between MATLAB and CUPY for Single Precision.')
    tike.view.plot_complex(np.fft.ifftshift(y1[0] - y[0]))
    plt.savefig(os.path.join(_test, 'fft-03.svg'))

    np.testing.assert_array_equal(y1, y)
    np.testing.assert_array_equal(x1, x)

def test_repeated_fft():

    with h5py.File(os.path.join(_dir, 'fft.mat'), 'r') as inputs:
        x = inputs['x0_double'][...].view('complex128')
    x = x.astype('complex128')

    x = cp.asarray(x, order='C')
    x0 = x.copy()
    error = list()

    for _ in range(100):
        x = cp.fft.fft2( x, norm='ortho')
        x = cp.fft.ifft2(x, norm='ortho')
        error.append(cp.mean(cp.abs((x-x0)/x0)).get())

    plt.figure()
    plt.plot(error)
    plt.title('Mean Absolute Relative Error for Double Precision on GPU')
    plt.xlabel('Calls to FFT/iFFT')
    plt.tight_layout()
    plt.savefig(os.path.join(_test, 'fft-04.svg'))

    x = cp.asarray(x, order='C').astype('complex64')
    x0 = x.copy()
    error = list()

    for _ in range(100):
        x = cp.fft.fft2( x, norm='ortho')
        x = cp.fft.ifft2(x, norm='ortho')
        error.append(cp.mean(cp.abs((x-x0)/x0)).get())

    plt.figure()
    plt.plot(error)
    plt.title('Mean Absolute Relative Error for Single Precision on GPU')
    plt.xlabel('Calls to FFT/iFFT')
    plt.tight_layout()
    plt.savefig(os.path.join(_test, 'fft-05.svg'))

    plt.figure()
    tike.view.plot_complex(x[0].get() - x0[0].get())
    plt.savefig(os.path.join(_test, 'fft-06.png'))
