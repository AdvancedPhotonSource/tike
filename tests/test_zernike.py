import unittest
import os

import tike.zernike
import tike.linalg
import tike.view
import matplotlib.pyplot as plt
import numpy as np

testdir = os.path.dirname(__file__)


class TestZernike(unittest.TestCase):

    def test_zernike(self):

        fname = os.path.join(testdir, 'result', 'zernike')
        os.makedirs(fname, exist_ok=True)
        for i, Z in enumerate(tike.zernike.basis(256, degree=10)):
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
        plt.ylim([-1, 1])
        plt.savefig(os.path.join(fname, f"radial-function-{m}.png"))
        plt.close()

    def test_radial(self):
        self._radial_template(0)

    def test_radial_1(self):
        self._radial_template(1)

    def test_radial_2(self):
        self._radial_template(2)

    def test_transform(self):

        fname = os.path.join(testdir, 'result', 'zernike')
        os.makedirs(fname, exist_ok=True)

        # import libimage

        # f0 = libimage.load('cryptomeria', 256)
        f0 = plt.imread(os.path.join(testdir, 'data', 'probe.png'))
        size = f0.shape[-1]
        plt.imsave(os.path.join(fname, 'basis-0.png'), f0)

        f0 = f0.reshape(size * size, 1)

        _basis = []

        for d in range(0, 100):
            _basis.append(
                tike.zernike.basis(
                    size,
                    degree_min=d,
                    degree_max=d + 1,
                ))

            basis = np.concatenate(_basis, axis=0)
            print(f'degree {d} - {len(basis)}')
            basis = np.moveaxis(basis, 0, -1)
            basis = basis.reshape(size * size, -1)
            # weight only pixels inside basis
            w = (basis[..., 0] > 0).astype('float32')

            # x = tike.linalg.lstsq(basis, f0, weights=w)
            x, _, _, _ = np.linalg.lstsq(basis, f0, rcond=1e-4)

            f1 = basis @ x
            f1 = f1.reshape(size, size)
            plt.imsave(os.path.join(fname, f'basis-{d:02d}.png'), f1)

            plt.figure()
            plt.title(f"basis weights for {d} degree polynomials")
            plt.bar(list(range(x.size)), x.flatten())
            plt.savefig(os.path.join(fname, f'basis-w-{d:02d}.png'))
            plt.close()
