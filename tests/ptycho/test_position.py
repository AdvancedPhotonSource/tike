import numpy as np
import tike.ptycho
import tike.linalg
import matplotlib.pyplot as plt
import os.path
import unittest

fname = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'result')


def test_position_join(N=245, num_batch=11):

    scan = np.random.rand(N, 2)
    assert scan.shape == (N, 2)
    indices = np.arange(N)
    assert np.amin(indices) == 0
    assert np.amax(indices) == N - 1
    np.random.shuffle(indices)
    batches = np.array_split(indices, num_batch)

    opts = tike.ptycho.PositionOptions(
        scan,
        use_adaptive_moment=True,
    )

    optsb = [opts.split(b) for b in batches]

    # Copies non-array params into new object
    new_opts = optsb[0].split([])

    for b, i in zip(optsb, batches):
        new_opts = new_opts.join(b, i)

    np.testing.assert_array_equal(
        new_opts.initial_scan,
        opts.initial_scan,
    )

    np.testing.assert_array_equal(
        new_opts._momentum,
        opts._momentum,
    )


def test_affine_translate():
    T = tike.ptycho.AffineTransform(t0=11, t1=-5)
    positions1 = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [-1, -1],
    ])
    np.testing.assert_equal(
        T(positions1),
        [
            [11, -5],
            [11, -4],
            [12, -5],
            [10, -6],
        ],
    )


def test_affine_scale():
    T = tike.ptycho.AffineTransform(scale0=11, scale1=0.5)
    positions1 = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [-1, -1],
    ])
    np.testing.assert_equal(
        T(positions1),
        [
            [0, 0],
            [0, 0.5],
            [11, 0],
            [-11, -0.5],
        ],
    )


class TestAffineEstimation(unittest.TestCase):

    def setUp(self, N=213) -> None:
        truth = [3.4567, 5.4321, 0.9876, 1.2345, 2.3456, -4.5678]
        T = tike.ptycho.AffineTransform(*truth)
        error = np.random.normal(size=(N, 2), scale=0.1)
        positions0 = (np.random.rand(*(N, 2)) - 0.5)
        positions1 = T(positions0) + error
        weights = (1 / (1 + np.square(error).sum(axis=-1)))

        self.truth = truth
        self.error = error
        self.positions0 = positions0
        self.positions1 = positions1
        self.weights = weights

    def test_fit_linear(self):
        """Fit a linear operator instead of a composed affine matrix."""

        T = tike.linalg.lstsq(
            a=np.pad(self.positions0, ((0, 0), (0, 1)), constant_values=1),
            b=self.positions1,
        )

        result = tike.ptycho.AffineTransform.fromarray(T)

        print()
        print(T)
        print(result.asarray3())

        f = plt.figure(dpi=600)
        plt.title('weighted')
        plt.scatter(
            self.positions0[..., 0],
            self.positions0[..., 1],
            marker='o',
        )
        plt.scatter(
            self.positions1[..., 0],
            self.positions1[..., 1],
            marker='o',
            color='red',
            facecolor='None',
        )
        plt.scatter(
            result(self.positions0)[..., 0],
            result(self.positions0)[..., 1],
            marker='x',
        )
        plt.axis('equal')
        plt.legend(['initial', 'final', 'estimated'])
        plt.savefig(os.path.join(fname, 'fit-weighted-linear.svg'))
        plt.close(f)

        print(self.truth)
        print(result.astuple())
