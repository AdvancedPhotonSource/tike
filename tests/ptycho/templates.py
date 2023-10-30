import os
import bz2
import typing

import numpy as np
import cupy as cp

from .io import data_dir

import tike.ptycho
import tike.communicators


class SiemensStarSetup():
    """Implements a setUp function which loads the siemens start dataset."""

    def setUp(self, filename='siemens-star-small.npz.bz2'):
        """Load a dataset for reconstruction."""
        dataset_file = os.path.join(data_dir, filename)
        with bz2.open(dataset_file, 'rb') as f:
            archive = np.load(f)
            self.scan = archive['scan'][0]
            self.data = archive['data'][0]
            self.probe = archive['probe'][0]
        self.scan -= np.amin(self.scan, axis=-2) - 20
        self.probe = tike.ptycho.probe.add_modes_cartesian_hermite(
            self.probe, 5)
        self.probe = tike.ptycho.probe.adjust_probe_power(self.probe)
        self.probe, _ = tike.ptycho.probe.orthogonalize_eig(self.probe)

        with tike.communicators.Comm(1, mpi=tike.communicators.MPIComm) as comm:
            mask = tike.cluster.by_scan_stripes(
                self.scan,
                n=comm.mpi.size,
                fly=1,
                axis=0,
            )[comm.mpi.rank]
            self.scan = self.scan[mask]
            self.data = self.data[mask]

        self.psi = np.full(
            (600, 600),
            dtype=np.complex64,
            fill_value=np.complex64(0.5 + 0j),
        )


try:
    from mpi4py import MPI
    _mpi_size = MPI.COMM_WORLD.Get_size()
    _mpi_rank = MPI.COMM_WORLD.Get_rank()
except ImportError:
    _mpi_size = 1
    _mpi_rank = 0

_device_per_rank = max(1, cp.cuda.runtime.getDeviceCount() // _mpi_size)
_base_device = (_device_per_rank * _mpi_rank) % cp.cuda.runtime.getDeviceCount()
_gpu_indices = tuple((i + _base_device) % cp.cuda.runtime.getDeviceCount()
                     for i in range(_device_per_rank))


class MPIAndGPUInfo():
    """Provides mpi rank and gpu index information."""

    mpi_size: int = _mpi_size
    mpi_rank: int = _mpi_rank
    gpu_indices: typing.Tuple[int] = _gpu_indices


class ReconstructTwice(MPIAndGPUInfo):
    """Call tike.ptycho reconstruct twice in a loop."""

    def template_consistent_algorithm(self, *, data, params):
        """Check ptycho.solver.algorithm for consistency."""
        with cp.cuda.Device(self.gpu_indices[0]):
            # Call twice to check that reconstruction continuation is correct
            for _ in range(2):
                params = tike.ptycho.reconstruct(
                    data=data,
                    parameters=params,
                    num_gpu=self.gpu_indices,
                    use_mpi=self.mpi_size > 1,
                )

        print()
        print('\n'.join(f'{c[0]:1.3e}' for c in params.algorithm_options.costs))
        return params
