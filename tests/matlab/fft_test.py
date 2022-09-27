import os

import cupy as cp
import h5py
import matplotlib.pyplot as plt
import numpy as np
import tike.view

_dir = os.path.dirname(__file__)


def test_consistent_fft():

    with h5py.File(os.path.join(_dir, 'fft.mat'), 'r') as inputs:
        x = inputs['x'][...].view('complex64')
        print(x.shape, x.dtype)

        y = inputs['y'][...].view('complex64')
        print(y.shape, y.dtype)

        yi = inputs['yi'][...].view('complex64')
        print(yi.shape, yi.dtype)

    y1 = cp.fft.fft2(cp.asarray(x, order='C'), norm='backward').get()

    yi1 = cp.fft.ifft2(cp.asarray(x, order='C'), norm='backward').get()

    plt.figure()
    tike.view.plot_complex(np.fft.ifftshift(y[0]))
    plt.savefig('fft-00.png')

    plt.figure()
    tike.view.plot_complex(np.fft.ifftshift(y1[0]))
    plt.savefig('fft-01.png')

    plt.figure()
    tike.view.plot_complex(np.fft.ifftshift(y1[0] - y[0]))
    plt.savefig('fft-02.png')

    np.testing.assert_allclose(y, y1, rtol=1e-3)
    np.testing.assert_allclose(yi, yi1, rtol=1e-3)
