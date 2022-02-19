import unittest
import os

import tike.zernike
import tike.view
import matplotlib.pyplot as plt

testdir = os.path.dirname(__file__)


class TestZernike(unittest.TestCase):

    def test_zernike(self):

        fname = os.path.join(testdir, 'result', 'zernike')
        os.makedirs(fname, exist_ok=True)
        for i, Z in enumerate(tike.zernike.mode(256, 3)):
            plt.figure()
            tike.view.plot_complex(Z)
            plt.savefig(os.path.join(fname, f"zernike-{i}.png"))
            plt.close()
