import os

import cupy as cp
import h5py
import matplotlib.pyplot as plt
import numpy as np
import tike.ptycho.probe
import tike.view

_dir = os.path.dirname(__file__)


def test_ortho():

    with h5py.File(os.path.join(_dir, 'ortho-in.mat'),
                   'r') as input, h5py.File(os.path.join(_dir, 'ortho-out.mat'),
                                            'r') as output:
        print(input.keys())
        print(output.keys())
        x0 = input['x0'][...].view('complex64')
        final0 = output['x1'][...].view('complex64')

    assert x0.shape == (5, 32, 32)
    assert final0.shape == (5, 32, 32)

    final1 = cp.asnumpy(tike.ptycho.probe.orthogonalize_eig(cp.asarray(x0)))

    for i in range(len(final0)):

        plt.figure()
        tike.view.plot_complex(final0[i] - final1[i])
        plt.suptitle('error')

        plt.figure()
        tike.view.plot_complex(final0[i] + final1[i])
        plt.suptitle('error opposite')

        plt.figure()
        tike.view.plot_complex(final0[i])
        plt.suptitle('original')

        plt.figure()
        tike.view.plot_complex(final1[i])
        plt.suptitle('tike')

        # Each mode is either the same or opposite magnitude (which is
        # equivalent orthogonalization)
        assert (np.allclose(final0[i], final1[i], atol=1e-4)
                or np.allclose(final0[i], -final1[i], atol=1e-4))

    # plt.show()
