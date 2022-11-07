#!/usr/bin/env python
# -*- coding: utf-8 -*-

# #########################################################################
# Copyright (c) 2018, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2015. UChicago Argonne, LLC. This software was produced       #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.                                                               #
#                                                                         #
# Additionally, redistribution and use in source and binary forms, with   #
# or without modification, are permitted provided that the following      #
# conditions are met:                                                     #
#                                                                         #
#     * Redistributions of source code must retain the above copyright    #
#       notice, this list of conditions and the following disclaimer.     #
#                                                                         #
#     * Redistributions in binary form must reproduce the above copyright #
#       notice, this list of conditions and the following disclaimer in   #
#       the documentation and/or other materials provided with the        #
#       distribution.                                                     #
#                                                                         #
#     * Neither the name of UChicago Argonne, LLC, Argonne National       #
#       Laboratory, ANL, the U.S. Government, nor the names of its        #
#       contributors may be used to endorse or promote products derived   #
#       from this software without specific prior written permission.     #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago     #
# Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,        #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;        #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER        #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT      #
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN       #
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         #
# POSSIBILITY OF SUCH DAMAGE.                                             #
# #########################################################################

import bz2
import lzma
import os
import pickle
import unittest
import warnings

import numpy as np
import cupy as cp

import tike.ptycho
from tike.ptycho.probe import ProbeOptions
from tike.ptycho.position import PositionOptions
from tike.ptycho.object import ObjectOptions
from tike.communicators import MPIComm
from tike.ptycho.solvers.options import (
    _resize_fft,
    _resize_spline,
    _resize_cubic,
    _resize_lanczos,
    _resize_linear,
)
import tike.random

__author__ = "Daniel Ching, Xiaodong Yu"
__copyright__ = "Copyright (c) 2018, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'

testdir = os.path.dirname(os.path.dirname(__file__))

try:
    from mpi4py import MPI
    _mpi_size = MPI.COMM_WORLD.Get_size()
    _mpi_rank = MPI.COMM_WORLD.Get_rank()
except ModuleNotFoundError:
    _mpi_size = 1
    _mpi_rank = 0


class TestPtychoUtils(unittest.TestCase):
    """Test various utility functions for correctness."""

    def test_gaussian(self):
        """Check ptycho.gaussian for correctness."""
        fname = os.path.join(testdir, 'data/ptycho_gaussian.pickle.lzma')
        weights = tike.ptycho.probe.gaussian(15, rin=0.8, rout=1.0)
        if os.path.isfile(fname):
            with lzma.open(fname, 'rb') as file:
                truth = pickle.load(file)
        else:
            with lzma.open(fname, 'wb') as file:
                truth = pickle.dump(weights, file)
        np.testing.assert_array_equal(weights, truth)

    def test_check_allowed_positions(self):
        psi = np.empty((4, 9))
        probe = np.empty((8, 2, 2))
        scan = np.array([[1, 1], [1, 6.9], [1.1, 1], [1.9, 5.5]])
        tike.ptycho.check_allowed_positions(scan, psi, probe.shape)

        for scan in np.array([[1, 7], [1, 0.9], [0.9, 1], [1, 0]]):
            with self.assertRaises(ValueError):
                tike.ptycho.check_allowed_positions(scan, psi, probe.shape)

    def test_split_by_scan(self):
        scan = np.mgrid[0:3, 0:3].reshape(2, -1)
        scan = np.moveaxis(scan, 0, -1)

        ind = tike.ptycho.ptycho.split_by_scan_stripes(scan, 3, axis=0)
        split = [scan[i] for i in ind]

        solution = [
            [[0, 0], [0, 1], [0, 2]],
            [[1, 0], [1, 1], [1, 2]],
            [[2, 0], [2, 1], [2, 2]],
        ]
        np.testing.assert_equal(split, solution)

        ind = tike.ptycho.ptycho.split_by_scan_stripes(scan, 3, axis=1)
        split = [scan[i] for i in ind]
        solution = [
            [[0, 0], [1, 0], [2, 0]],
            [[0, 1], [1, 1], [2, 1]],
            [[0, 2], [1, 2], [2, 2]],
        ]
        np.testing.assert_equal(split, solution)


class TestPtychoSimulate(unittest.TestCase):

    def create_dataset(
        self,
        dataset_file,
        pw=16,
        eigen=1,
        width=128,
    ):
        """Create a dataset for testing this module.

        Only called with setUp detects that `dataset_file` has been deleted.
        """
        import libimage
        # Create a stack of phase-only images
        phase = libimage.load('satyre', width)
        amplitude = 1 - libimage.load('coins', width)
        original = amplitude * np.exp(1j * phase * np.pi)
        self.original = original.astype('complex64')
        leading = ()

        # Create a multi-probe with gaussian amplitude decreasing as 1/N
        phase = np.stack(
            [
                1 - libimage.load('cryptomeria', pw),
                1 - libimage.load('bombus', pw)
            ],
            axis=0,
        )
        weights = 1.0 / np.arange(1, len(phase) + 1)[:, None, None]
        weights = weights * tike.ptycho.probe.gaussian(pw, rin=0.8, rout=1.0)
        probe = weights * np.exp(1j * phase * np.pi)
        self.probe = np.tile(
            probe.astype('complex64'),
            (*leading, 1, eigen, 1, 1, 1),
        )

        pad = 2
        v, h = np.meshgrid(
            np.linspace(pad, original.shape[-2] - pw - pad, 13, endpoint=True),
            np.linspace(pad, original.shape[-1] - pw - pad, 13, endpoint=True),
            indexing='ij',
        )
        scan = np.stack((np.ravel(v), np.ravel(h)), axis=1)
        self.scan = np.tile(
            scan.astype('float32'),
            (*leading, 1, 1),
        )

        self.data = tike.ptycho.simulate(
            detector_shape=pw * 2,
            probe=self.probe,
            scan=self.scan,
            psi=self.original,
        )

        assert self.data.shape == (*leading, 13 * 13, pw * 2, pw * 2)
        assert self.data.dtype == 'float32', self.data.dtype

        setup_data = [
            self.data,
            self.scan,
            self.probe,
            self.original,
        ]
        with lzma.open(dataset_file, 'wb') as file:
            pickle.dump(setup_data, file)

    def setUp(self):
        """Load a dataset for reconstruction."""
        dataset_file = os.path.join(testdir, 'data/ptycho_setup.pickle.lzma')
        if not os.path.isfile(dataset_file):
            self.create_dataset(dataset_file)
        with lzma.open(dataset_file, 'rb') as file:
            [
                self.data,
                self.scan,
                self.probe,
                self.original,
            ] = pickle.load(file)

    def test_consistent_simulate(self):
        """Check ptycho.simulate for consistency."""
        data = tike.ptycho.simulate(
            detector_shape=self.data.shape[-1],
            probe=self.probe,
            scan=self.scan,
            psi=self.original,
            fly=self.scan.shape[-2] // self.data.shape[-3],
        )
        assert data.dtype == 'float32', data.dtype
        assert self.data.dtype == 'float32', self.data.dtype
        np.testing.assert_array_equal(data.shape, self.data.shape)
        np.testing.assert_allclose(np.sqrt(data), np.sqrt(self.data), atol=1e-6)


class TemplatePtychoRecon():

    def setUp(self, filename='data/siemens-star-small.npz.bz2'):
        """Load a dataset for reconstruction."""
        dataset_file = os.path.join(testdir, filename)
        with bz2.open(dataset_file, 'rb') as f:
            archive = np.load(f)
            self.scan = archive['scan'][0]
            self.data = archive['data'][0]
            self.probe = archive['probe'][0]
        self.scan -= np.amin(self.scan, axis=-2) - 20
        self.probe = tike.ptycho.probe.add_modes_cartesian_hermite(
            self.probe, 5)
        self.probe = tike.ptycho.probe.adjust_probe_power(self.probe)
        self.probe = tike.ptycho.probe.orthogonalize_eig(self.probe)

    def init_params(self):
        return tike.ptycho.PtychoParameters(
            psi=np.full(
                (600, 600),
                dtype=np.complex64,
                fill_value=np.complex64(0.5 + 0j),
            ),
            probe=self.probe,
            scan=self.scan,
        )

    def template_consistent_algorithm(self, *, params={}):
        """Check ptycho.solver.algorithm for consistency."""

        if _mpi_size > 1:
            with MPIComm() as IO:
                params['parameters'].probe = IO.Bcast(
                    params['parameters'].probe)
                if params['parameters'].eigen_weights is not None:
                    (
                        params['parameters'].scan,
                        self.data,
                        params['parameters'].eigen_weights,
                    ) = IO.MPIio_ptycho(
                        params['parameters'].scan,
                        self.data,
                        params['parameters'].eigen_weights,
                    )
                else:
                    params['parameters'].scan, self.data = IO.MPIio_ptycho(
                        params['parameters'].scan, self.data)

        device_per_rank = cp.cuda.runtime.getDeviceCount() // _mpi_size
        base_device = device_per_rank * _mpi_rank
        with cp.cuda.Device(base_device):
            # Call twice to check that reconstruction continuation is correct
            for _ in range(2):
                params['parameters'] = tike.ptycho.reconstruct(
                    **params,
                    data=self.data,
                    num_gpu=tuple(
                        i + base_device for i in range(device_per_rank)),
                    use_mpi=_mpi_size > 1,
                )
        print()
        print('\n'.join(f'{c[0]:1.3e}'
                        for c in params['parameters'].algorithm_options.costs))
        return params['parameters']


class TestPtychoAbsorption(TemplatePtychoRecon, unittest.TestCase):
    """Test various ptychography reconstruction methods for consistency."""

    def test_absorption(self):
        """Check ptycho.object.get_absorption_image for consistency."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            from matplotlib import pyplot as plt
            fname = os.path.join(testdir, 'result', 'ptycho', 'absorption')
            os.makedirs(fname, exist_ok=True)
            plt.imsave(
                f'{fname}/{0}-ampli.png',
                tike.ptycho.object.get_absorbtion_image(
                    self.data,
                    self.scan,
                    rescale=1.0,
                ),
            )
        except ImportError:
            pass


class TestPtychoRecon(TemplatePtychoRecon, unittest.TestCase):
    """Test various ptychography reconstruction methods for consistency."""

    post_name = ""

    def test_init(self):
        params = self.init_params()
        params.algorithm_options = tike.ptycho.AdamOptions(
            num_batch=5,
            num_iter=16,
        )
        params.probe_options = ProbeOptions()
        params.object_options = ObjectOptions()
        _save_ptycho_result(
            params, f"{'mpi-' if _mpi_size > 1 else ''}init{self.post_name}")

    def test_consistent_adam_grad(self):
        """Check ptycho.solver.adam_grad for consistency."""
        params = self.init_params()
        params.algorithm_options = tike.ptycho.AdamOptions(
            num_batch=5,
            num_iter=16,
        )
        params.probe_options = ProbeOptions()
        params.object_options = ObjectOptions()
        _save_ptycho_result(
            self.template_consistent_algorithm(params={
                'parameters': params,
            },), f"{'mpi-' if _mpi_size > 1 else ''}adam_grad{self.post_name}")

    def test_consistent_cgrad(self):
        """Check ptycho.solver.cgrad for consistency."""
        params = self.init_params()
        params.algorithm_options = tike.ptycho.CgradOptions(
            num_batch=5,
            num_iter=16,
        )
        params.probe_options = ProbeOptions()
        params.object_options = ObjectOptions()
        _save_ptycho_result(
            self.template_consistent_algorithm(params={
                'parameters': params,
            },), f"{'mpi-' if _mpi_size > 1 else ''}cgrad{self.post_name}")

    def test_consistent_lstsq_grad(self):
        """Check ptycho.solver.lstsq_grad for consistency."""
        params = self.init_params()
        params.algorithm_options = tike.ptycho.LstsqOptions(
            num_batch=5,
            num_iter=16,
        )
        params.probe_options = ProbeOptions(use_adaptive_moment=True,)
        params.object_options = ObjectOptions(use_adaptive_moment=True,)
        _save_ptycho_result(
            self.template_consistent_algorithm(params={
                'parameters': params,
            },), f"{'mpi-' if _mpi_size > 1 else ''}lstsq_grad{self.post_name}")

    def test_consistent_lstsq_grad_compact(self):
        """Check ptycho.solver.lstsq_grad for consistency."""
        params = self.init_params()
        params.algorithm_options = tike.ptycho.LstsqOptions(
            num_batch=5,
            num_iter=16,
            batch_method='cluster_compact',
        )
        params.probe_options = ProbeOptions(use_adaptive_moment=True,)
        params.object_options = ObjectOptions(use_adaptive_moment=True,)
        _save_ptycho_result(
            self.template_consistent_algorithm(params={
                'parameters': params,
            },),
            f"{'mpi-' if _mpi_size > 1 else ''}lstsq_grad-compact{self.post_name}"
        )

    def test_consistent_lstsq_grad_variable_probe(self):
        """Check ptycho.solver.lstsq_grad for consistency."""
        params = self.init_params()
        probes_with_modes = min(3, params.probe.shape[-3])
        params.eigen_probe, params.eigen_weights = tike.ptycho.probe.init_varying_probe(
            params.scan,
            params.probe,
            num_eigen_probes=3,
            probes_with_modes=probes_with_modes,
        )
        params.algorithm_options = tike.ptycho.LstsqOptions(
            num_batch=5,
            num_iter=16,
        )
        params.probe_options = ProbeOptions(use_adaptive_moment=True,)
        params.object_options = ObjectOptions(use_adaptive_moment=True,)
        result = self.template_consistent_algorithm(params={
            'parameters': params,
        },)
        _save_ptycho_result(
            result,
            f"{'mpi-' if _mpi_size > 1 else ''}lstsq_grad-variable-probe{self.post_name}",
        )
        assert np.all(result.eigen_weights[..., 1:, probes_with_modes:] == 0), (
            "These weights should be unused/untouched "
            "and should have been initialized to zero.")

    def test_consistent_rpie(self):
        """Check ptycho.solver.rpie for consistency."""
        params = self.init_params()
        params.algorithm_options = tike.ptycho.RpieOptions(
            num_batch=5,
            num_iter=16,
        )
        params.probe_options = ProbeOptions()
        params.object_options = ObjectOptions()
        _save_ptycho_result(
            self.template_consistent_algorithm(params={
                'parameters': params,
            },), f"{'mpi-' if _mpi_size > 1 else ''}rpie{self.post_name}")

    def test_consistent_dm(self):
        """Check ptycho.solver.dm for consistency."""
        params = self.init_params()
        params.algorithm_options = tike.ptycho.DmOptions(
            num_iter=16,
            num_batch=5,
        )
        params.probe_options = ProbeOptions()
        params.object_options = ObjectOptions()
        _save_ptycho_result(
            self.template_consistent_algorithm(params={
                'parameters': params,
            },), f"{'mpi-' if _mpi_size > 1 else ''}dm{self.post_name}")


class TestPtychoOnline(TestPtychoRecon, unittest.TestCase):
    """Test ptychography reconstruction when data is streaming."""

    post_name = "-online"

    def setUp(self, chunks=16) -> None:
        TestPtychoRecon.setUp(self)
        data = np.array_split(self.data, chunks, axis=0)
        scan = np.array_split(self.scan, chunks, axis=0)
        assert len(data) == chunks
        assert len(scan) == chunks

        self.data = data[0]
        self.scan = scan[0]
        self.data_more = data[1:]
        self.scan_more = scan[1:]

    def template_consistent_algorithm(self, *, params={}):

        with tike.ptycho.Reconstruction(**params, data=self.data) as context:
            context.iterate(2)
            for d, s in zip(self.data_more, self.scan_more):
                context.append_new_data(
                    new_data=d,
                    new_scan=s,
                )
                context.iterate(2)
        result = context.parameters
        print()
        print('\n'.join(f'{c[0]:1.3e}' for c in result.algorithm_options.costs))
        return result


@unittest.skipIf('TIKE_TEST_CI' in os.environ,
                 reason="Just for user reference; not needed on CI.")
class TestPtychoPositionReference(TestPtychoRecon, unittest.TestCase):
    """Test various ptychography reconstruction methods position correction."""

    post_name = '-position-ref'

    def setUp(self, filename='data/position-error-247.pickle.bz2'):
        """Load a dataset for reconstruction.

        This position correction test dataset was collected by Tao Zhou at the
        Center for Nanoscale Materials Hard X-ray Nanoprobe
        (https://www.anl.gov/cnm).
        """
        dataset_file = os.path.join(testdir, filename)
        with bz2.open(dataset_file, 'rb') as f:
            [
                self.data,
                _,
                self.scan,
                self.probe,
            ] = pickle.load(f)


class TestPtychoPosition(TemplatePtychoRecon, unittest.TestCase):
    """Test various ptychography reconstruction methods position correction."""

    def setUp(self, filename='data/position-error-247.pickle.bz2'):
        """Load a dataset for reconstruction.

        This position correction test dataset was collected by Tao Zhou at the
        Center for Nanoscale Materials Hard X-ray Nanoprobe
        (https://www.anl.gov/cnm).
        """
        dataset_file = os.path.join(testdir, filename)
        with bz2.open(dataset_file, 'rb') as f:
            [
                self.data,
                self.scan,
                self.scan_truth,
                self.probe,
            ] = pickle.load(f)

    def _save_position_error_variance(self, result, algorithm):
        try:
            import matplotlib
            matplotlib.use('Agg')
            from matplotlib import pyplot as plt
            import tike.view
            fname = os.path.join(testdir, 'result', 'ptycho', f'{algorithm}')
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
        except ImportError:
            pass

    def test_consistent_rpie_off(self):
        """Check ptycho.solver.rpie position correction."""
        algorithm = f"{'mpi-' if _mpi_size > 1 else ''}rpie-position-off"
        params = self.init_params()
        params.algorithm_options = tike.ptycho.RpieOptions(
            num_batch=5,
            num_iter=16,
        )
        params.probe_options = ProbeOptions()
        params.object_options = ObjectOptions()
        result = self.template_consistent_algorithm(params={
            'parameters': params,
        },)
        _save_ptycho_result(result, algorithm)

    def test_consistent_rpie(self):
        """Check ptycho.solver.rpie position correction."""
        algorithm = f"{'mpi-' if _mpi_size > 1 else ''}rpie-position"
        params = self.init_params()
        params.algorithm_options = tike.ptycho.RpieOptions(
            num_batch=5,
            num_iter=16,
        )
        params.position_options = PositionOptions(
            self.scan,
            use_adaptive_moment=True,
        )
        params.probe_options = ProbeOptions()
        params.object_options = ObjectOptions()
        result = self.template_consistent_algorithm(params={
            'parameters': params,
        },)
        _save_ptycho_result(result, algorithm)
        self._save_position_error_variance(result, algorithm)

    def test_consistent_lstsq_grad(self):
        """Check ptycho.solver.lstsq_grad for consistency."""
        algorithm = f"{'mpi-' if _mpi_size > 1 else ''}lstsq_grad-position"
        params = self.init_params()
        params.algorithm_options = tike.ptycho.LstsqOptions(
            num_batch=5,
            num_iter=16,
        )
        params.position_options = PositionOptions(
            self.scan,
            use_adaptive_moment=True,
        )
        params.probe_options = ProbeOptions()
        params.object_options = ObjectOptions()
        result = self.template_consistent_algorithm(params={
            'parameters': params,
        },)
        _save_ptycho_result(result, algorithm)
        self._save_position_error_variance(result, algorithm)


def _save_eigen_probe(output_folder, eigen_probe):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    flattened = []
    for i in range(eigen_probe.shape[-4]):
        probe = eigen_probe[..., i, :, :, :]
        flattened.append(
            np.concatenate(
                probe.reshape((-1, *probe.shape[-2:])),
                axis=1,
            ))
    flattened = np.concatenate(flattened, axis=0)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        plt.imsave(
            f'{output_folder}/eigen-phase.png',
            np.angle(flattened),
            # The output of np.angle is locked to (-pi, pi]
            cmap=plt.cm.twilight,
            vmin=-np.pi,
            vmax=np.pi,
        )
        plt.imsave(
            f'{output_folder}/eigen-ampli.png',
            np.abs(flattened),
        )


def _save_probe(output_folder, probe, algorithm):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    flattened = np.concatenate(
        probe.reshape((-1, *probe.shape[-2:])),
        axis=1,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        plt.imsave(
            f'{output_folder}/probe-phase.png',
            np.angle(flattened),
            # The output of np.angle is locked to (-pi, pi]
            cmap=plt.cm.twilight,
            vmin=-np.pi,
            vmax=np.pi,
        )
        plt.imsave(
            f'{output_folder}/probe-ampli.png',
            np.abs(flattened),
        )
    f = plt.figure()
    tike.view.plot_probe_power(probe)
    plt.semilogy()
    plt.title(algorithm)
    plt.savefig(f'{output_folder}/probe-power.svg')
    plt.close(f)


def _save_ptycho_result(result, algorithm):
    try:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        import tike.view
        fname = os.path.join(testdir, 'result', 'ptycho', f'{algorithm}')
        os.makedirs(fname, exist_ok=True)

        fig = plt.figure()
        ax1, ax2 = tike.view.plot_cost_convergence(
            result.algorithm_options.costs,
            result.algorithm_options.times,
        )
        ax2.set_xlim(0, 20)
        ax1.set_ylim(10**(-1), 10**2)
        fig.suptitle(algorithm)
        fig.tight_layout()
        plt.savefig(os.path.join(fname, 'convergence.svg'))
        plt.close(fig)
        plt.imsave(
            f'{fname}/{0}-phase.png',
            np.angle(result.psi).astype('float32'),
            # The output of np.angle is locked to (-pi, pi]
            cmap=plt.cm.twilight,
            vmin=-np.pi,
            vmax=np.pi,
        )
        plt.imsave(
            f'{fname}/{0}-ampli.png',
            np.abs(result.psi).astype('float32'),
        )
        import tifffile
        tifffile.imwrite(
            f'{fname}/{0}-ampli.tiff',
            np.abs(result.psi).astype('float32'),
        )
        _save_probe(fname, result.probe, algorithm)
        if result.eigen_weights is not None:
            _save_eigen_weights(fname, result.eigen_weights)
            if result.eigen_weights.shape[-2] > 1:
                _save_eigen_probe(fname, result.eigen_probe)
    except ImportError:
        pass


def _save_eigen_weights(fname, weights):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    plt.figure()
    tike.view.plot_eigen_weights(weights)
    plt.suptitle('weights')
    plt.tight_layout()
    plt.savefig(f'{fname}/weights.svg')


class PtychoReconMultiGrid():
    """Test ptychography multi-grid reconstruction method."""

    def interp(self, x, f):
        pass

    def template_consistent_algorithm(self, *, params={}):
        """Check ptycho.solver.algorithm for consistency."""

        if _mpi_size > 1:
            return 1

        device_per_rank = cp.cuda.runtime.getDeviceCount() // _mpi_size
        base_device = device_per_rank * _mpi_rank
        with cp.cuda.Device(base_device):
            parameters = tike.ptycho.reconstruct_multigrid(
                **params,
                data=self.data,
                num_gpu=tuple(i + base_device for i in range(device_per_rank)),
                use_mpi=_mpi_size > 1,
                num_levels=2,
                interp=self.interp,
            )

        print()
        print('\n'.join(
            f'{c[0]:1.3e}' for c in parameters.algorithm_options.costs))
        return parameters


class TestPtychoReconMultiGridFFT(PtychoReconMultiGrid, TestPtychoRecon,
                                  unittest.TestCase):

    post_name = '-multigrid-fft'

    def interp(self, x, f):
        return _resize_fft(x, f)


if False:
    # Don't need to run these tests on CI every time.

    class TestPtychoReconMultiGridLinear(PtychoReconMultiGrid, TestPtychoRecon,
                                         unittest.TestCase):

        post_name = '-multigrid-linear'

        def interp(self, x, f):
            return _resize_linear(x, f)

    class TestPtychoReconMultiGridCubic(PtychoReconMultiGrid, TestPtychoRecon,
                                        unittest.TestCase):

        post_name = '-multigrid-cubic'

        def interp(self, x, f):
            return _resize_cubic(x, f)

    class TestPtychoReconMultiGridLanczos(PtychoReconMultiGrid, TestPtychoRecon,
                                          unittest.TestCase):

        post_name = '-multigrid-lanczos'

        def interp(self, x, f):
            return _resize_lanczos(x, f)

    class TestPtychoReconMultiGridSpline(PtychoReconMultiGrid, TestPtychoRecon,
                                         unittest.TestCase):

        post_name = '-multigrid-spline'

        def interp(self, x, f):
            return _resize_spline(x, f)


if __name__ == '__main__':
    unittest.main()
