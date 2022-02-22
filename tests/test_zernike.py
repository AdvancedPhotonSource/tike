import unittest
import os

import tike.zernike
import tike.view
import matplotlib.pyplot as plt
import numpy as np

testdir = os.path.dirname(__file__)


class TestZernike(unittest.TestCase):

    def test_zernike(self):

        fname = os.path.join(testdir, 'result', 'zernike')
        os.makedirs(fname, exist_ok=True)
        for i, Z in enumerate(tike.zernike.zernike_basis(256, degree=10)):
            plt.figure()
            tike.view.plot_complex(Z, rmin=-1, rmax=1)
            plt.savefig(os.path.join(fname, f"zernike-{i:02d}.png"))
            plt.close()

    def _radial_template(self, m=0):
        fname = os.path.join(testdir, 'result', 'zernike')
        os.makedirs(fname, exist_ok=True)

        radius = np.linspace(0, 1, 200)

        plt.figure()
        labels = []
        for n in range(0, 9):
            if (n + m) % 2 == 0:
                v = tike.zernike.R(m, n, radius)
                plt.plot(
                    radius,
                    v,
                )
                labels.append(n)
        plt.legend(labels)
        plt.savefig(os.path.join(fname, f"radial-function-{m}.png"))
        plt.close()

    def test_radial(self):
        self._radial_template(0)

    def test_radial_1(self):
        self._radial_template(1)

    def test_radial_2(self):
        self._radial_template(2)
