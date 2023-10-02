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

import lzma
import os
import pickle
import unittest

import numpy as np

from tike.ptycho.exitwave import ExitWaveOptions
from tike.ptycho.object import ObjectOptions
from tike.ptycho.probe import ProbeOptions
import tike.ptycho
import tike.random

from .io import (
    result_dir,
    data_dir,
    _save_ptycho_result,
)
from .templates import (
    SiemensStarSetup,
    ReconstructTwice,
)

__author__ = "Daniel Ching, Xiaodong Yu"
__copyright__ = "Copyright (c) 2018, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'


class TestPtychoUtils(unittest.TestCase):
    """Test various utility functions for correctness."""

    def test_gaussian(self):
        """Check ptycho.gaussian for correctness."""
        fname = os.path.join(data_dir, 'ptycho_gaussian.pickle.lzma')
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

    def test_get_padded_object(self):
        probe = np.empty((8, 3, 4))
        scan = (np.random.rand(15, 2) * 100) - 50
        psi, scan = tike.ptycho.object.get_padded_object(scan, probe)
        tike.ptycho.check_allowed_positions(scan, psi, probe_shape=probe.shape)


class TestPtychoSimulate(unittest.TestCase):
    """Test the forward model for consistency."""

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
        dataset_file = os.path.join(data_dir, 'ptycho_setup.pickle.lzma')
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


class TestPtychoAbsorption(SiemensStarSetup, unittest.TestCase):
    """Test various ptychography reconstruction methods for consistency."""

    def test_absorption(self):
        """Check ptycho.object.get_absorption_image for consistency."""
        try:
            from matplotlib import pyplot as plt
            fname = os.path.join(result_dir, 'absorption')
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


class PtychoRecon(
        ReconstructTwice,
        SiemensStarSetup,
):
    """Test various ptychography reconstruction methods for consistency."""

    post_name = ""

    def test_init(self):
        """Just test PtychoParameter initialization."""
        params = tike.ptycho.PtychoParameters(
            psi=self.psi,
            probe=self.probe,
            scan=self.scan,
        )
        params.algorithm_options = tike.ptycho.RpieOptions(
            num_batch=5,
            num_iter=16,
        )
        params.probe_options = ProbeOptions(force_orthogonality=True,)
        params.object_options = ObjectOptions()
        params.exitwave_options = ExitWaveOptions(measured_pixels=np.ones(
            self.probe.shape[-2:],
            dtype=np.bool_,
        ))

        _save_ptycho_result(
            params,
            f"mpi{self.mpi_size}-init{self.post_name}",
        )
        try:
            import matplotlib.pyplot as plt
            plt.imsave(
                os.path.join(
                    result_dir,
                    f"mpi{self.mpi_size}-init{self.post_name}",
                    'diffraction.png',
                ),
                self.data[len(self.data) // 2],
            )
        except ImportError:
            pass

    def test_consistent_lstsq_poisson_steplength_allmodes(self):
        """Check ptycho.solver.lstsq_grad for consistency."""
        params = tike.ptycho.PtychoParameters(
            psi=self.psi,
            probe=self.probe,
            scan=self.scan,
            algorithm_options=tike.ptycho.LstsqOptions(
                num_batch=5,
                num_iter=16,
            ),
            probe_options=ProbeOptions(use_adaptive_moment=True,),
            object_options=ObjectOptions(use_adaptive_moment=True,),
            exitwave_options=ExitWaveOptions(
                measured_pixels=np.ones(self.probe.shape[-2:], dtype=np.bool_),
                noise_model="poisson",
                step_length_usemodes="all_modes",
            ),
        )

        _save_ptycho_result(
            self.template_consistent_algorithm(
                data=self.data,
                params=params,
            ),
            f"mpi{self.mpi_size}-lstsq_poisson_steplength_allmodes{self.post_name}"
        )

    def test_consistent_lstsq_poisson_steplength_dominantmode(self):
        """Check ptycho.solver.lstsq_grad for consistency."""
        params = tike.ptycho.PtychoParameters(
            psi=self.psi,
            probe=self.probe,
            scan=self.scan,
            algorithm_options=tike.ptycho.LstsqOptions(
                num_batch=5,
                num_iter=16,
            ),
            probe_options=ProbeOptions(use_adaptive_moment=True,),
            object_options=ObjectOptions(use_adaptive_moment=True,),
            exitwave_options=ExitWaveOptions(
                measured_pixels=np.ones(self.probe.shape[-2:], dtype=np.bool_),
                noise_model="poisson",
                step_length_usemodes="dominant_mode",
            ),
        )

        _save_ptycho_result(
            self.template_consistent_algorithm(
                data=self.data,
                params=params,
            ),
            f"mpi{self.mpi_size}-lstsq_poisson_steplength_dominantmode{self.post_name}"
        )

    def test_consistent_lstsq_unmeasured_detector_regions(self):
        """Check ptycho.solver.lstsq for consistency."""
        # Define regions where we have missing diffraction measurement data
        unmeasured_pixels = np.zeros(self.probe.shape[-2:], np.bool_)
        unmeasured_pixels[100:105, :] = True
        unmeasured_pixels[:, 100:105] = True
        measured_pixels = np.logical_not(unmeasured_pixels)

        # Zero out these regions on the diffraction measurement data
        # to simulate realistic measurements with umeasured pixels
        self.data[..., unmeasured_pixels] = np.nan

        # import matplotlib.pyplot as plt
        # import matplotlib as mpl
        # mpl.use('TKAgg'); plt.figure(); plt.imshow( np.fft.fftshift( self.data[222, ... ] )); plt.show(block=False)

        params = tike.ptycho.PtychoParameters(
            psi=self.psi,
            probe=self.probe,
            scan=self.scan,
            algorithm_options=tike.ptycho.LstsqOptions(
                num_batch=5,
                num_iter=16,
            ),
            probe_options=ProbeOptions(use_adaptive_moment=True,),
            object_options=ObjectOptions(use_adaptive_moment=True,),
            exitwave_options=ExitWaveOptions(
                measured_pixels=measured_pixels,
                unmeasured_pixels_scaling=0.90,
            ),
        )

        _save_ptycho_result(
            self.template_consistent_algorithm(
                data=self.data,
                params=params,
            ),
            f"mpi{self.mpi_size}-lstsq_unmeasured_detector_regions{self.post_name}",
        )

    def test_consistent_lstsq_grad(self):
        """Check ptycho.solver.lstsq_grad for consistency."""
        params = tike.ptycho.PtychoParameters(
            psi=self.psi,
            probe=self.probe,
            scan=self.scan,
            algorithm_options=tike.ptycho.LstsqOptions(
                num_batch=5,
                num_iter=16,
            ),
            probe_options=ProbeOptions(
                force_orthogonality=True,
                use_adaptive_moment=True,
            ),
            object_options=ObjectOptions(use_adaptive_moment=True,),
        )

        _save_ptycho_result(
            self.template_consistent_algorithm(
                data=self.data,
                params=params,
            ), f"mpi{self.mpi_size}-lstsq_grad{self.post_name}")

    def test_consistent_lstsq_grad_no_probe(self):
        """Check ptycho.solver.lstsq_grad for consistency."""
        params = tike.ptycho.PtychoParameters(
            psi=self.psi,
            probe=self.probe,
            scan=self.scan,
            algorithm_options=tike.ptycho.LstsqOptions(
                num_batch=5,
                num_iter=16,
            ),
            object_options=ObjectOptions(use_adaptive_moment=True,),
        )

        _save_ptycho_result(
            self.template_consistent_algorithm(
                data=self.data,
                params=params,
            ), f"mpi{self.mpi_size}-lstsq_grad-no-probe{self.post_name}")

    def test_consistent_lstsq_grad_compact(self):
        """Check ptycho.solver.lstsq_grad for consistency."""
        params = tike.ptycho.PtychoParameters(
            psi=self.psi,
            probe=self.probe,
            scan=self.scan,
            algorithm_options=tike.ptycho.LstsqOptions(
                num_batch=5,
                num_iter=16,
                batch_method='compact',
            ),
            probe_options=ProbeOptions(
                force_orthogonality=True,
                use_adaptive_moment=True,
            ),
            object_options=ObjectOptions(use_adaptive_moment=True,),
        )

        _save_ptycho_result(
            self.template_consistent_algorithm(
                data=self.data,
                params=params,
            ), f"mpi{self.mpi_size}-lstsq_grad-compact{self.post_name}")

    def test_consistent_lstsq_grad_compact_no_probe(self):
        """Check ptycho.solver.lstsq_grad for consistency."""
        params = tike.ptycho.PtychoParameters(
            psi=self.psi,
            probe=self.probe,
            scan=self.scan,
            algorithm_options=tike.ptycho.LstsqOptions(
                num_batch=5,
                num_iter=16,
                batch_method='compact',
            ),
            object_options=ObjectOptions(use_adaptive_moment=True,),
        )

        _save_ptycho_result(
            self.template_consistent_algorithm(
                data=self.data,
                params=params,
            ),
            f"mpi{self.mpi_size}-lstsq_grad-compact-no-probe{self.post_name}")

    def test_consistent_lstsq_grad_variable_probe(self):
        """Check ptycho.solver.lstsq_grad for consistency."""
        params = tike.ptycho.PtychoParameters(
            psi=self.psi,
            probe=self.probe,
            scan=self.scan,
            algorithm_options=tike.ptycho.LstsqOptions(
                num_batch=5,
                num_iter=16,
            ),
            probe_options=ProbeOptions(
                force_orthogonality=True,
                use_adaptive_moment=True,
            ),
            object_options=ObjectOptions(use_adaptive_moment=True,),
        )

        probes_with_modes = min(3, params.probe.shape[-3])
        params.eigen_probe, params.eigen_weights = tike.ptycho.probe.init_varying_probe(
            params.scan,
            params.probe,
            num_eigen_probes=3,
            probes_with_modes=probes_with_modes,
        )
        result = self.template_consistent_algorithm(
            data=self.data,
            params=params,
        )
        _save_ptycho_result(
            result,
            f"mpi{self.mpi_size}-lstsq_grad-variable-probe{self.post_name}",
        )
        assert np.all(result.eigen_weights[..., 1:, probes_with_modes:] == 0), (
            "These weights should be unused/untouched "
            "and should have been initialized to zero.")

    def test_consistent_rpie_poisson_steplength_dominantmode(self):
        """Check ptycho.solver.rpie for consistency."""
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
                measured_pixels=np.ones(self.probe.shape[-2:], dtype=np.bool_),
                noise_model="poisson",
                step_length_usemodes="dominant_mode",
            ),
        )

        _save_ptycho_result(
            self.template_consistent_algorithm(
                data=self.data,
                params=params,
            ),
            f"mpi{self.mpi_size}-rpie_poisson_steplength_dominantmode{self.post_name}",
        )

    def test_consistent_rpie_poisson_steplength_allmodes(self):
        """Check ptycho.solver.rpie for consistency."""
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
                measured_pixels=np.ones(self.probe.shape[-2:], dtype=np.bool_),
                noise_model="poisson",
                step_length_usemodes="all_modes",
            ),
        )

        _save_ptycho_result(
            self.template_consistent_algorithm(
                data=self.data,
                params=params,
            ),
            f"mpi{self.mpi_size}-rpie_poisson_steplength_allmodes{self.post_name}",
        )

    def test_consistent_rpie_unmeasured_detector_regions(self):
        """Check ptycho.solver.rpie for consistency."""
        # Define regions where we have missing diffraction measurement data
        unmeasured_pixels = np.zeros(self.probe.shape[-2:], np.bool_)
        unmeasured_pixels[100:105, :] = True
        unmeasured_pixels[:, 100:105] = True
        measured_pixels = np.logical_not(unmeasured_pixels)

        # Zero out these regions on the diffraction measurement data
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
                unmeasured_pixels_scaling=0.90,
            ),
        )

        _save_ptycho_result(
            self.template_consistent_algorithm(
                data=self.data,
                params=params,
            ),
            f"mpi{self.mpi_size}-rpie_unmeasured_detector_regions{self.post_name}",
        )

    def test_consistent_rpie(self):
        """Check ptycho.solver.rpie for consistency."""
        params = tike.ptycho.PtychoParameters(
            psi=self.psi,
            probe=self.probe,
            scan=self.scan,
            algorithm_options=tike.ptycho.RpieOptions(
                num_batch=5,
                num_iter=16,
            ),
            probe_options=ProbeOptions(force_orthogonality=True,),
            object_options=ObjectOptions(smoothness_constraint=0.01,),
        )

        _save_ptycho_result(
            self.template_consistent_algorithm(
                data=self.data,
                params=params,
            ),
            f"mpi{self.mpi_size}-rpie{self.post_name}",
        )

    def test_consistent_rpie_no_probe(self):
        """Check ptycho.solver.rpie for consistency."""
        params = tike.ptycho.PtychoParameters(
            psi=self.psi,
            probe=self.probe,
            scan=self.scan,
            algorithm_options=tike.ptycho.RpieOptions(
                num_batch=5,
                num_iter=16,
            ),
            object_options=ObjectOptions(smoothness_constraint=0.01,),
        )

        _save_ptycho_result(
            self.template_consistent_algorithm(
                data=self.data,
                params=params,
            ),
            f"mpi{self.mpi_size}-rpie-no-probe{self.post_name}",
        )

    def test_consistent_rpie_compact(self):
        """Check ptycho.solver.rpie for consistency."""
        params = tike.ptycho.PtychoParameters(
            psi=self.psi,
            probe=self.probe,
            scan=self.scan,
            algorithm_options=tike.ptycho.RpieOptions(
                num_batch=5,
                num_iter=16,
                batch_method='compact',
            ),
            probe_options=ProbeOptions(
                force_orthogonality=True,
                use_adaptive_moment=True,
            ),
            object_options=ObjectOptions(use_adaptive_moment=True,),
        )

        _save_ptycho_result(
            self.template_consistent_algorithm(
                data=self.data,
                params=params,
            ), f"mpi{self.mpi_size}-rpie-compact{self.post_name}")

    def test_consistent_rpie_compact_no_probe(self):
        """Check ptycho.solver.rpie for consistency."""
        params = tike.ptycho.PtychoParameters(
            psi=self.psi,
            probe=self.probe,
            scan=self.scan,
            algorithm_options=tike.ptycho.RpieOptions(
                num_batch=5,
                num_iter=16,
                batch_method='compact',
            ),
            object_options=ObjectOptions(use_adaptive_moment=True,),
        )
        _save_ptycho_result(
            self.template_consistent_algorithm(
                data=self.data,
                params=params,
            ), f"mpi{self.mpi_size}-rpie-compact-no-probe{self.post_name}")

    def test_consistent_rpie_variable_probe(self):
        """Check ptycho.solver.lstsq_grad for consistency."""
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

        probes_with_modes = min(1, params.probe.shape[-3])
        params.eigen_probe, params.eigen_weights = tike.ptycho.probe.init_varying_probe(
            params.scan,
            params.probe,
            num_eigen_probes=1,
            probes_with_modes=probes_with_modes,
        )
        result = self.template_consistent_algorithm(
            data=self.data,
            params=params,
        )
        _save_ptycho_result(
            result,
            f"mpi{self.mpi_size}-rpie-variable-probe{self.post_name}",
        )
        assert np.all(result.eigen_weights[..., 1:, probes_with_modes:] == 0), (
            "These weights should be unused/untouched "
            "and should have been initialized to zero.")

    def test_consistent_dm(self):
        """Check ptycho.solver.dm for consistency."""
        params = tike.ptycho.PtychoParameters(
            psi=self.psi,
            probe=self.probe,
            scan=self.scan,
            algorithm_options=tike.ptycho.DmOptions(
                num_iter=16,
                num_batch=5,
            ),
            probe_options=ProbeOptions(force_orthogonality=True,),
            object_options=ObjectOptions(),
        )

        _save_ptycho_result(
            self.template_consistent_algorithm(
                data=self.data,
                params=params,
            ),
            f"mpi{self.mpi_size}-dm{self.post_name}",
        )

    def test_consistent_dm_no_probe(self):
        """Check ptycho.solver.dm for consistency."""
        params = tike.ptycho.PtychoParameters(
            psi=self.psi,
            probe=self.probe,
            scan=self.scan,
            algorithm_options=tike.ptycho.DmOptions(
                num_iter=16,
                num_batch=5,
            ),
            object_options=ObjectOptions(),
        )

        _save_ptycho_result(
            self.template_consistent_algorithm(
                data=self.data,
                params=params,
            ),
            f"mpi{self.mpi_size}-dm-no-probe{self.post_name}",
        )


class TestPtychoRecon(
        PtychoRecon,
        unittest.TestCase,
):
    """Separate test from implementation so that PtychoRecon can be imported elsewhere."""
    pass


if __name__ == '__main__':
    unittest.main()
