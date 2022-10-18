
import os.path
import bz2
import numpy as np
import scipy.ndimage
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def resize_probe(x, f):
    return scipy.ndimage.zoom(
        x,
        zoom=[1] * (x.ndim - 2) + [f, f],
        grid_mode=True,
        prefilter=False,
    )


testdir = os.path.dirname(os.path.dirname(__file__))
output_folder = os.path.join(testdir, 'result', 'ptycho', 'multigrid')

def test_resample(filename='data/siemens-star-small.npz.bz2'):

    os.makedirs(output_folder, exist_ok=True)

    dataset_file = os.path.join(testdir, filename)
    with bz2.open(dataset_file, 'rb') as f:
        archive = np.load(f)
        probe = archive['probe'][0]

    for i in [0.25, 0.50, 1.0, 2.0, 4.0]:
        p1 = resize_probe(probe, i)
        flattened = np.concatenate(
            p1.reshape((-1, *p1.shape[-2:])),
            axis=1,
        )
        plt.imsave(
            f'{output_folder}/probe-ampli-{i}.png',
            np.abs(flattened),
        )

