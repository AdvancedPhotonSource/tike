import os
import unittest

import cupy as cp
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import scipy.io
import tike.ptycho.probe
from tike.communicators import Comm, MPIComm

resultdir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'result',
                         'ptycho', 'probe')


class TestProbe(unittest.TestCase):

    def test_eigen_probe(self):

        leading = (2,)
        wide = 18
        high = 21
        posi = 53
        eigen = 1
        comm = Comm(2, None)

        R = comm.pool.bcast([np.random.rand(*leading, posi, 1, 1, wide, high)])
        eigen_probe = comm.pool.bcast(
            [np.random.rand(*leading, 1, eigen, 1, wide, high)])
        weights = np.random.rand(*leading, posi, eigen + 1, 1)
        weights -= np.mean(weights, axis=-3, keepdims=True)
        weights = comm.pool.bcast([weights])
        patches = comm.pool.bcast(
            [np.random.rand(*leading, posi, 1, 1, wide, high)])
        diff = comm.pool.bcast(
            [np.random.rand(*leading, posi, 1, 1, wide, high)])

        new_probe, new_weights = tike.ptycho.probe.update_eigen_probe(
            comm=comm,
            R=R,
            eigen_probe=eigen_probe,
            weights=weights,
            patches=patches,
            diff=diff,
            c=1,
            m=0,
        )

        assert eigen_probe[0].shape == new_probe[0].shape

    def template_get_varying_probe(
        self,
        p=31,
        e=0,
        s=2,
        w=16,
        vary=False,
    ):

        unique = tike.ptycho.probe.get_varying_probe(
            shared_probe=np.random.rand(1, 1, s, w, w),
            eigen_probe=np.random.rand(1, e, s, w, w) if e > 0 else None,
            weights=np.ones((p, e + 1, s)) if vary else None,
        )
        assert unique.shape == (p if vary else 1, 1, s, w, w)

    def test_get_varying_probe_mono_probe(self):
        self.template_get_varying_probe(0, 0, 1, 16)

    def test_get_varying_probe_multi_probe(self):
        self.template_get_varying_probe(0, 0, 7, 16)

    def test_get_varying_probe_mono_probe_varying(self):
        self.template_get_varying_probe(31, 0, 1, 16, vary=True)

    def test_get_varying_probe_multi_probe_varying(self):
        self.template_get_varying_probe(31, 0, 7, 16, vary=True)

    def test_get_varying_probe_mono_probe_varying_eigen(self):
        self.template_get_varying_probe(31, 3, 1, 16, vary=True)

    def test_get_varying_probe_multi_probe_varying_eigen(self):
        self.template_get_varying_probe(31, 3, 7, 16, vary=True)

    def template_init_varing_probe(self, p=31, e=0, s=2, w=16, v=1):

        eigen_probe, weights = tike.ptycho.probe.init_varying_probe(
            scan=np.random.rand(p, 2),
            shared_probe=np.random.rand(1, 1, s, w, w),
            num_eigen_probes=e,
            probes_with_modes=v,
        )
        if e < 2:
            assert eigen_probe is None
        else:
            assert eigen_probe.shape == (1, e - 1, v, w, w)
        if e < 1:
            assert weights is None
        else:
            assert weights.shape == (p, e, s)

    def test_init_no_varying_probe(self):
        self.template_init_varing_probe(31, 0, 2, 16, 0)
        self.template_init_varing_probe(31, 0, 2, 16, 1)

    def test_init_1_varying_probe(self):
        self.template_init_varing_probe(31, 1, 2, 16, 1)

    def test_init_many_varying_probe(self):
        self.template_init_varing_probe(31, 1, 3, 16, 3)

    def test_init_many_varying_probe_with_multiple_basis(self):
        self.template_init_varing_probe(31, 7, 3, 16, 3)

    def test_init_1_varying_probe_with_multiple_basis(self):
        self.template_init_varing_probe(31, 7, 3, 16, 1)

    def test_probe_support(self):
        """Finite probe support penalty function is within expected bounds."""
        penalty = tike.ptycho.probe.finite_probe_support(
            probe=cp.zeros((101, 101)),  # must be odd shaped for min to be 0
            radius=0.5 * 0.7,
            degree=2.5,  # must have degree >= 1 for upper bound to be p
            p=2.345,
        )
        try:
            import tifffile
            os.makedirs(resultdir, exist_ok=True)
            tifffile.imwrite(os.path.join(resultdir, 'penalty.tiff'),
                             penalty.astype('float32').get())
        except ImportError:
            pass
        assert cp.around(cp.min(penalty), 3) == 0.000
        assert cp.around(cp.max(penalty), 3) == 2.345


def test_orthogonalize_eig():
    """Test the eigen orthogonalization method is correct."""
    np.set_printoptions(
        linewidth=400,
        precision=4,
        floatmode='fixed',
        suppress=True,
    )

    thisdir = os.path.dirname(__file__)
    input = scipy.io.loadmat(f'{thisdir}/ortho-in.mat')
    output = scipy.io.loadmat(f'{thisdir}/ortho-out.mat')

    probe = np.rollaxis(input['modes'], -1, 0)

    final1 = cp.asnumpy(tike.ptycho.probe.orthogonalize_eig(cp.asarray(probe)))
    final0 = np.rollaxis(output['pr'], -1, 0)

    # The phases are not correct because some are out of sync by 180 degrees
    np.testing.assert_allclose(np.abs(final1), np.abs(final0), rtol=1e-4)


def test_hermite_modes():
    """Test the hermite basis probe generation is correct."""
    thisdir = os.path.dirname(__file__)
    inputs = scipy.io.loadmat(f'{thisdir}/hermite.mat')

    result1 = tike.ptycho.probe.add_modes_cartesian_hermite(
        np.rollaxis(inputs['probes'], -1, 0)[None, None, ...],
        12,
    )

    os.makedirs(resultdir, exist_ok=True)
    flattened = np.concatenate(result1[0], axis=1)
    plt.imsave(os.path.join(resultdir, 'hermite.png'), np.abs(flattened))

    np.testing.assert_allclose(
        result1,
        np.rollaxis(inputs['result'], -1, 0)[None, ...],
    )


if __name__ == '__main__':
    unittest.main()
