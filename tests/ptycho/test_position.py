import bz2
import pickle

import matplotlib.pyplot as plt
import numpy as np
import os.path
import unittest

from tike.ptycho.exitwave import ExitWaveOptions
from tike.ptycho.object import ObjectOptions
from tike.ptycho.position import PositionOptions
from tike.ptycho.probe import ProbeOptions
import tike.communicators
import tike.linalg
import tike.ptycho

from .io import result_dir, data_dir, _save_ptycho_result
from .templates import ReconstructTwice


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
            weights=self.weights,
        )

        result = tike.ptycho.AffineTransform.fromarray(T)

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
        plt.savefig(os.path.join(result_dir, 'fit-weighted-linear.svg'))
        plt.close(f)

        np.testing.assert_almost_equal(result.asarray3(), T, decimal=3)


class CNMPositionSetup():

    def setUp(self, filename='position-error-247.pickle.bz2'):
        """Load a dataset for reconstruction.

        This position correction test dataset was collected by Tao Zhou at the
        Center for Nanoscale Materials Hard X-ray Nanoprobe
        (https://www.anl.gov/cnm).
        """
        dataset_file = os.path.join(data_dir, filename)
        with bz2.open(dataset_file, 'rb') as f:
            [
                self.data,
                self.scan,
                self.scan_truth,
                self.probe,
            ] = pickle.load(f)

        with tike.communicators.Comm(1, mpi=tike.communicators.MPIComm) as comm:
            mask = tike.cluster.by_scan_stripes(
                self.scan,
                n=comm.mpi.size,
                fly=1,
                axis=0,
            )[comm.mpi.rank]
            self.scan = self.scan[mask]
            self.scan_truth = self.scan_truth[mask]
            self.data = self.data[mask]

        self.psi = np.full(
            (600, 600),
            dtype=np.complex64,
            fill_value=np.complex64(0.5 + 0j),
        )


class CNMTruePositionSetup(CNMPositionSetup):

    def setUp(self, filename='position-error-247.pickle.bz2'):
        super().setUp(filename)
        self.scan = self.scan_truth


class PtychoPosition(ReconstructTwice, CNMPositionSetup):
    """Test various ptychography reconstruction methods position correction."""

    post_name = "-position"

    def _save_position_error_variance(self, result, algorithm):
        if result is None or self.mpi_rank > 0:
            return
        try:
            import matplotlib
            matplotlib.use('Agg')
            from matplotlib import pyplot as plt
            import tike.view
            fname = os.path.join(result_dir, f'{algorithm}')
            os.makedirs(fname, exist_ok=True)

            f = plt.figure(dpi=600)
            plt.title(algorithm)
            tike.view.plot_positions_convergence(
                self.scan_truth,
                result.position_options.initial_scan,
                result.scan,
            )
            plt.savefig(os.path.join(fname, 'position-error.svg'))
            plt.close(f)

            f = plt.figure(dpi=600)
            plt.title(algorithm)
            plt.scatter(
                np.linalg.norm(self.scan_truth - result.scan, axis=-1),
                result.position_options.confidence[..., 0],
            )
            plt.xlabel('position error')
            plt.ylabel('position confidence')
            plt.savefig(os.path.join(fname, 'position-confidence.svg'))
            plt.close(f)

            f = plt.figure(dpi=600)
            plt.title(algorithm)
            plt.scatter(
                result.position_options.initial_scan[..., 0],
                result.position_options.initial_scan[..., 1],
                marker='o',
            )
            plt.scatter(
                result.scan[..., 0],
                result.scan[..., 1],
                marker='o',
                color='red',
                facecolor='None',
            )
            plt.scatter(
                result.position_options.transform(
                    result.position_options.initial_scan)[..., 0],
                result.position_options.transform(
                    result.position_options.initial_scan)[..., 1],
                marker='x',
            )
            plt.legend(['initial', 'result', 'global'])
            plt.savefig(os.path.join(fname, 'position-models.svg'))
            plt.close(f)

        except ImportError:
            pass

    def test_consistent_rpie_off(self):
        """Check ptycho.solver.rpie position correction."""
        algorithm = f"mpi{self.mpi_size}-rpie-off{self.post_name}"
        params = tike.ptycho.PtychoParameters(
            psi=self.psi,
            probe=self.probe,
            scan=self.scan,
            algorithm_options=tike.ptycho.RpieOptions(
                num_batch=5,
                num_iter=16,
            ),
            probe_options=ProbeOptions(force_orthogonality=True,),
            object_options=ObjectOptions(),
        )

        result = self.template_consistent_algorithm(
            data=self.data,
            params=params,
        )
        _save_ptycho_result(result, algorithm)

    def test_consistent_rpie(self):
        """Check ptycho.solver.rpie position correction."""
        algorithm = f"mpi{self.mpi_size}-rpie{self.post_name}"
        params = tike.ptycho.PtychoParameters(
            psi=self.psi,
            probe=self.probe,
            scan=self.scan,
            algorithm_options=tike.ptycho.RpieOptions(
                num_batch=5,
                num_iter=16,
            ),
            probe_options=ProbeOptions(force_orthogonality=True,),
            object_options=ObjectOptions(),
            position_options=PositionOptions(
                self.scan,
                use_adaptive_moment=True,
                use_position_regularization=True,
                update_magnitude_limit=5,
            ),
        )

        result = self.template_consistent_algorithm(
            data=self.data,
            params=params,
        )
        _save_ptycho_result(result, algorithm)
        self._save_position_error_variance(result, algorithm)

    def test_consistent_lstsq_grad(self):
        """Check ptycho.solver.lstsq_grad for consistency."""
        algorithm = f"mpi{self.mpi_size}-lstsq_grad{self.post_name}"
        params = tike.ptycho.PtychoParameters(
            psi=self.psi,
            probe=self.probe,
            scan=self.scan,
            algorithm_options=tike.ptycho.LstsqOptions(
                num_batch=5,
                num_iter=16,
            ),
            probe_options=ProbeOptions(force_orthogonality=True,),
            object_options=ObjectOptions(),
            position_options=PositionOptions(
                self.scan,
                use_adaptive_moment=True,
                use_position_regularization=True,
            ),
        )

        result = self.template_consistent_algorithm(
            data=self.data,
            params=params,
        )
        _save_ptycho_result(result, algorithm)
        self._save_position_error_variance(result, algorithm)

    def test_consistent_rpie_unmeasured_detector_regions(self):
        """Check ptycho.solver.rpie position correction."""
        algorithm = f"mpi{self.mpi_size}-rpie_unmeasured_detector_regions{self.post_name}"

        # Define regions where we have missing diffraction measurement data
        unmeasured_pixels = np.zeros(self.probe.shape[-2:], np.bool_)
        unmeasured_pixels[100:105, :] = True
        unmeasured_pixels[:, 100:105] = True
        measured_pixels = np.logical_not(unmeasured_pixels)

        # Zero out these regions on the diffraction measurement data
        self.data = self.data.astype(np.floating)
        self.data[:, unmeasured_pixels] = np.nan

        params = tike.ptycho.PtychoParameters(
            psi=self.psi,
            probe=self.probe,
            scan=self.scan,
            algorithm_options=tike.ptycho.RpieOptions(
                num_batch=5,
                num_iter=16,
            ),
            probe_options=ProbeOptions(),
            object_options=ObjectOptions(),
            exitwave_options=ExitWaveOptions(
                measured_pixels=measured_pixels,
                unmeasured_pixels_scaling=1.05,
            ),
            position_options=PositionOptions(
                self.scan,
                use_adaptive_moment=True,
                use_position_regularization=True,
            ),
        )

        result = self.template_consistent_algorithm(
            data=self.data,
            params=params,
        )
        _save_ptycho_result(result, algorithm)
        self._save_position_error_variance(result, algorithm)

    def test_consistent_lstsq_grad_unmeasured_detector_regions(self):
        """Check ptycho.solver.lstsq_grad for consistency."""
        algorithm = f"mpi{self.mpi_size}-lstsq_grad_unmeasured_detector_regions{self.post_name}"

        # Define regions where we have missing diffraction measurement data
        unmeasured_pixels = np.zeros(self.probe.shape[-2:], np.bool_)
        unmeasured_pixels[100:105, :] = True
        unmeasured_pixels[:, 100:105] = True
        measured_pixels = np.logical_not(unmeasured_pixels)

        # Zero out these regions on the diffraction measurement data
        self.data = self.data.astype(np.floating)
        self.data[:, unmeasured_pixels] = np.nan

        params = tike.ptycho.PtychoParameters(
            psi=self.psi,
            probe=self.probe,
            scan=self.scan,
            algorithm_options=tike.ptycho.LstsqOptions(
                num_batch=5,
                num_iter=16,
            ),
            probe_options=ProbeOptions(),
            object_options=ObjectOptions(),
            exitwave_options=ExitWaveOptions(
                measured_pixels=measured_pixels,
                unmeasured_pixels_scaling=1.05,
            ),
            position_options=PositionOptions(
                self.scan,
                use_adaptive_moment=True,
                use_position_regularization=True,
            ),
        )

        result = self.template_consistent_algorithm(
            data=self.data,
            params=params,
        )
        _save_ptycho_result(result, algorithm)
        self._save_position_error_variance(result, algorithm)


class TestPtychoPosition(
        PtychoPosition,
        unittest.TestCase,
):
    """Test various ptychography reconstruction methods position correction."""
    pass


@unittest.skipIf('TIKE_TEST_CI' in os.environ,
                 reason="Just for user reference; not needed on CI.")
class TestPtychoPositionReference(
        CNMTruePositionSetup,
        PtychoPosition,
        unittest.TestCase,
):
    """Test various ptychography reconstruction methods position correction."""

    post_name = '-position-ref'
