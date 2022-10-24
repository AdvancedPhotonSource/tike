
import os.path
import bz2
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pytest

from tike.ptycho.solvers.options import _resize_fft, _resize_spline, _resize_cubic, _resize_lanczos, _resize_linear

testdir = os.path.dirname(os.path.dirname(__file__))
output_folder = os.path.join(testdir, 'result', 'ptycho', 'multigrid')

@pytest.mark.parametrize(
    "function",[
        _resize_fft,
        _resize_spline,
        _resize_linear,
        _resize_cubic,
        _resize_lanczos,
    ]
)
def test_resample(function, filename='data/siemens-star-small.npz.bz2'):

    os.makedirs(output_folder, exist_ok=True)

    dataset_file = os.path.join(testdir, filename)
    with bz2.open(dataset_file, 'rb') as f:
        archive = np.load(f)
        probe = archive['probe'][0]

    for i in [0.25, 0.50, 1.0, 2.0, 4.0]:
        p1 = function(probe, i)
        flattened = np.concatenate(
            p1.reshape((-1, *p1.shape[-2:])),
            axis=1,
        )
        plt.imsave(
            f'{output_folder}/{function.__name__}-probe-ampli-{i}.png',
            np.abs(flattened),
        )
        plt.imsave(
            f'{output_folder}/{function.__name__}-probe-phase-{i}.png',
            np.angle(flattened),
        )
