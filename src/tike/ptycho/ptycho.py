#!/usr/bin/env python
# -*- coding: utf-8 -*-

# #########################################################################
# Copyright (c) 2018, UChicago Argonne, LLC. All rights reserved.    #
#                                                                         #
# Copyright 2018. UChicago Argonne, LLC. This software was produced       #
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

__author__ = "Doga Gursoy, Daniel Ching, Xiaodong Yu"
__copyright__ = "Copyright (c) 2018, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = [
    "reconstruct",
    "simulate",
    "Reconstruction",
    "reconstruct_multigrid",
]

import copy
import logging
import time
import typing
import warnings

import numpy as np
import numpy.typing as npt
import cupy as cp

import tike.operators
import tike.communicators
import tike.opt
from tike.ptycho import solvers
import tike.cluster
import tike.precision

from .position import (
    PositionOptions,
    check_allowed_positions,
    affine_position_regularization,
)
from .probe import (
    constrain_center_peak,
    constrain_probe_sparsity,
    get_varying_probe,
    apply_median_filter_abs_probe,
)

logger = logging.getLogger(__name__)


def _compute_intensity(
    operator,
    psi,
    scan,
    probe,
    eigen_weights=None,
    eigen_probe=None,
    fly=1,
):
    leading = psi.shape[:-2]
    intensity = 0
    for m in range(probe.shape[-3]):
        farplane = operator.fwd(
            probe=get_varying_probe(
                probe[..., [m], :, :],
                None if eigen_probe is None else eigen_probe[..., [m], :, :],
                None if eigen_weights is None else eigen_weights[..., [m]],
            ),
            scan=scan,
            psi=psi,
        )
        intensity += np.sum(
            np.square(np.abs(farplane)).reshape(
                *leading,
                scan.shape[-2] // fly,
                fly,
                operator.detector_shape,
                operator.detector_shape,
            ),
            axis=-3,
            keepdims=False,
        )
    return intensity


def simulate(
        detector_shape,
        probe, scan,
        psi,
        fly=1,
        eigen_probe=None,
        eigen_weights=None,
        **kwargs
):  # yapf: disable
    """Return real-valued detector counts of simulated ptychography data.

    Parameters
    ----------
    detector_shape : int
        The pixel width of the detector.
    probe : (..., 1, 1, SHARED, WIDE, HIGH) complex64
        The shared complex illumination function amongst all positions.
    scan : (..., POSI, 2) float32
        Coordinates of the minimum corner of the probe grid for each
        measurement in the coordinate system of psi.
    psi : (..., WIDE, HIGH) complex64
        The complex wavefront modulation of the object.
    fly : int
        The number of scan positions which combine for one detector frame.
    eigen_probe : (..., 1, EIGEN, SHARED, WIDE, HIGH) complex64
        The eigen probes for all positions.
    eigen_weights : (..., POSI, EIGEN, SHARED) float32
        The relative intensity of the eigen probes at each position.

    Returns
    -------
    data : (..., FRAME, WIDE, HIGH) float32
        The simulated intensity on the detector.

    """
    check_allowed_positions(scan, psi, probe.shape)
    with tike.operators.Ptycho(
            probe_shape=probe.shape[-1],
            detector_shape=int(detector_shape),
            nz=psi.shape[-2],
            n=psi.shape[-1],
            **kwargs,
    ) as operator:
        scan = operator.asarray(scan, dtype=tike.precision.floating)
        psi = operator.asarray(psi, dtype=tike.precision.cfloating)
        probe = operator.asarray(probe, dtype=tike.precision.cfloating)
        if eigen_weights is not None:
            eigen_weights = operator.asarray(eigen_weights,
                                             dtype=tike.precision.floating)
        data = _compute_intensity(operator, psi, scan, probe, eigen_weights,
                                  eigen_probe, fly)
        return operator.asnumpy(data.real)


def reconstruct(
    data: npt.NDArray,
    parameters: solvers.PtychoParameters,
    num_gpu: typing.Union[int, typing.Tuple[int, ...]] = 1,
    use_mpi: bool = False,
) -> solvers.PtychoParameters:
    """Solve the ptychography problem.

    This functional interface is the simplest to use, but deallocates GPU
    memory when it returns. Use the context manager API for the ability to get
    live results without ending the reconsturction.

    Parameters
    ----------
    data : (FRAME, WIDE, HIGH) uint16
        The intensity (square of the absolute value) of the propagated
        wavefront; i.e. what the detector records. FFT-shifted so the
        diffraction peak is at the corners.
    parameters: :py:class:`tike.ptycho.solvers.PtychoParameters`
        A class containing reconstruction parameters.
    num_gpu : int, tuple(int)
        The number of GPUs to use or a tuple of the device numbers of the GPUs
        to use. If the number of GPUs is less than the requested number, only
        workers for the available GPUs are allocated.
    use_mpi : bool
        Whether to use MPI or not.


    .. versionchanged:: 0.25.0 Removed the `model` parameter. Use
        :py:class:`tike.ptycho.exitwave.ExitwaveOptions` instead.

    Raises
    ------
        ValueError
            When shapes are incorrect for various input parameters.

    Returns
    -------
    result : :py:class:`tike.ptycho.solvers.PtychoParameters`
        A class containing reconstruction parameters.


    .. seealso:: :py:func:`tike.ptycho.ptycho.reconstruct_multigrid`, :py:func:`tike.ptycho.ptycho.Reconstruction`
    """
    with Reconstruction(
            data,
            parameters,
            num_gpu,
            use_mpi,
    ) as context:
        context.iterate(parameters.algorithm_options.num_iter)
    return context.parameters


def _clip_magnitude(x, a_max):
    """Clips a complex array's magnitude without changing the phase."""
    magnitude = np.abs(x)
    out_of_range = magnitude > a_max
    x[out_of_range] = a_max * x[out_of_range] / magnitude[out_of_range]
    return x


class Reconstruction():
    """Context manager for online ptychography reconstruction.

    .. versionadded:: 0.22.0

    Uses same parameters as the functional reconstruct API. Using a context
    manager allows for getting the current result or adding additional data
    without deallocating memory on the GPU.

    Example
    -------
    Data structure remain allocated on the GPU as long as the context is
    active. This allows quickly resuming a reconstruction.

    .. code-block:: python

        with tike.ptycho.Reconstruction(
            data,
            parameters,
            num_gpu,
            use_mpi,
        ) as context:
            # Start reconstructing
            context.iterate(num_iter)
            # Check the current solution for the object
            early_result = context.get_psi()
            # Continue reconstructing
            context.iterate()
        # All datastructures are transferred off the GPU at context close
        final_result = context.parameters

    .. seealso:: :py:func:`tike.ptycho.ptycho.reconstruct`
    """

    def __init__(
        self,
        data: npt.NDArray,
        parameters: solvers.PtychoParameters,
        num_gpu: typing.Union[int, typing.Tuple[int, ...]] = 1,
        use_mpi: bool = False,
    ):
        if (np.any(np.asarray(data.shape) < 1) or data.ndim != 3
                or data.shape[-2] != data.shape[-1]):
            raise ValueError(
                f"data shape {data.shape} is incorrect. "
                "It should be (N, W, H), "
                "where N >= 1 is the number of square diffraction patterns.")
        if data.shape[0] != parameters.scan.shape[0]:
            raise ValueError(
                f"data shape {data.shape} and scan shape {parameters.scan.shape} "
                "are incompatible. They should have the same leading dimension."
            )
        if np.any(
                np.asarray(parameters.probe.shape[-2:]) > np.asarray(
                    data.shape[-2:])):
            raise ValueError(f"probe shape {parameters.probe.shape} "
                             f"and data shape {data.shape} are incompatible. "
                             "The probe width/height must be "
                             f"<= the data width/height .")
        logger.info("{} on {:,d} - {:,d} by {:,d} frames for at most {:,d} "
                    "epochs.".format(
                        parameters.algorithm_options.name,
                        *data.shape[-3:],
                        parameters.algorithm_options.num_iter,
                    ))

        if use_mpi:
            mpi = tike.communicators.MPIComm
            if parameters.psi is None:
                raise ValueError(
                    "When MPI is enabled, initial object guess cannot be None; "
                    "automatic psi initialization is not synchronized "
                    "across processes.")
        else:
            mpi = tike.communicators.NoMPIComm

        self.data = data
        self.parameters = copy.deepcopy(parameters)
        self.device = cp.cuda.Device(
            num_gpu[0] if isinstance(num_gpu, tuple) else None)
        self.operator = tike.operators.Ptycho(
            probe_shape=parameters.probe.shape[-1],
            detector_shape=data.shape[-1],
            nz=parameters.psi.shape[-2],
            n=parameters.psi.shape[-1],
            norm=parameters.exitwave_options.propagation_normalization,
        )
        self.comm = tike.communicators.Comm(num_gpu, mpi)

    def __enter__(self):
        self.device.__enter__()
        self.operator.__enter__()
        self.comm.__enter__()

        # Divide the inputs into regions
        if (not np.all(np.isfinite(self.data)) or np.any(self.data < 0)):
            warnings.warn(
                "Diffraction patterns contain invalid data. "
                "All data should be non-negative and finite.", UserWarning)

        (
            self.comm.order,
            self.batches,
            self.parameters.scan,
            self.data,
            self.parameters.eigen_weights,
        ) = tike.cluster.by_scan_stripes_contiguous(
            self.data,
            self.parameters.eigen_weights,
            scan=self.parameters.scan,
            pool=self.comm.pool,
            shape=(self.comm.pool.num_workers, 1),
            dtype=(
                tike.precision.floating,
                tike.precision.floating
                if self.data.itemsize > 2 else self.data.dtype,
                tike.precision.floating,
            ),
            destination=('gpu', 'pinned', 'gpu'),
            batch_method=self.parameters.algorithm_options.batch_method,
            num_batch=self.parameters.algorithm_options.num_batch,
        )

        self.parameters.psi = self.comm.pool.bcast(
            [self.parameters.psi.astype(tike.precision.cfloating)])

        self.parameters.probe = self.comm.pool.bcast(
            [self.parameters.probe.astype(tike.precision.cfloating)])

        if self.parameters.probe_options is not None:
            self.parameters.probe_options = self.parameters.probe_options.copy_to_device(
                self.comm,)

        if self.parameters.object_options is not None:
            self.parameters.object_options = self.parameters.object_options.copy_to_device(
                self.comm,)

        if self.parameters.exitwave_options is not None:
            self.parameters.exitwave_options = self.parameters.exitwave_options.copy_to_device(
                self.comm,)

        if self.parameters.eigen_probe is not None:
            self.parameters.eigen_probe = self.comm.pool.bcast(
                [self.parameters.eigen_probe.astype(tike.precision.cfloating)])

        if self.parameters.position_options is not None:
            # TODO: Consider combining put/split, get/join operations?
            self.parameters.position_options = self.comm.pool.map(
                PositionOptions.copy_to_device,
                (self.parameters.position_options.split(x)
                 for x in self.comm.order),
            )

        if self.parameters.probe_options is not None:

            if self.parameters.probe_options.init_rescale_from_measurements:
                self.parameters.probe = _rescale_probe(
                    self.operator,
                    self.comm,
                    self.data,
                    self.parameters.exitwave_options,
                    self.parameters.psi,
                    self.parameters.scan,
                    self.parameters.probe,
                    num_batch=self.parameters.algorithm_options.num_batch,
                )

            if np.isnan(self.parameters.probe_options.probe_photons):
                self.parameters.probe_options.probe_photons = np.sum(
                    np.abs(self.parameters.probe[0].get())**2)

        return self

    def iterate(self, num_iter: int) -> None:
        """Advance the reconstruction by num_iter epochs."""
        start = time.perf_counter()
        psi_previous = self.parameters.psi[0].copy()
        for i in range(num_iter):

            if (
                np.sum(self.parameters.algorithm_options.times)
                > self.parameters.algorithm_options.time_limit
            ):
                logger.info("Maximum reconstruction time exceeded.")
                break

            logger.info(f"{self.parameters.algorithm_options.name} epoch "
                        f"{len(self.parameters.algorithm_options.times):,d}")

            total_epochs = len(self.parameters.algorithm_options.times)

            if self.parameters.probe_options is not None:
                self.parameters.probe_options.recover_probe = (
                    total_epochs >= self.parameters.probe_options.update_start
                    and (total_epochs % self.parameters.probe_options.update_period) == 0
                )  # yapf: disable

            if self.parameters.probe_options is not None:
                if self.parameters.probe_options.recover_probe:

                    if self.parameters.probe_options.median_filter_abs_probe:
                        self.parameters.probe = self.comm.pool.map(
                            apply_median_filter_abs_probe,
                            self.parameters.probe,
                            med_filt_px = self.parameters.probe_options.median_filter_abs_probe_px
                        )

                    if self.parameters.probe_options.force_centered_intensity:
                        self.parameters.probe = self.comm.pool.map(
                            constrain_center_peak,
                            self.parameters.probe,
                        )

                    if self.parameters.probe_options.force_sparsity < 1:
                        self.parameters.probe = self.comm.pool.map(
                            constrain_probe_sparsity,
                            self.parameters.probe,
                            f=self.parameters.probe_options.force_sparsity,
                        )

                    if self.parameters.probe_options.force_orthogonality:
                        (
                            self.parameters.probe,
                            power,
                        ) = (list(a) for a in zip(*self.comm.pool.map(
                            tike.ptycho.probe.orthogonalize_eig,
                            self.parameters.probe,
                        )))
                    else:
                        power = self.comm.pool.map(
                            tike.ptycho.probe.power,
                            self.parameters.probe,
                        )

                    self.parameters.probe_options.power.append(
                        power[0].get())

            (
                self.parameters.object_options,
                self.parameters.probe_options,
            ) = solvers.update_preconditioners(
                comm=self.comm,
                operator=self.operator,
                scan=self.parameters.scan,
                probe=self.parameters.probe,
                psi=self.parameters.psi,
                object_options=self.parameters.object_options,
                probe_options=self.parameters.probe_options,
            )

            self.parameters = getattr(
                solvers,
                self.parameters.algorithm_options.name,
            )(
                self.operator,
                self.comm,
                data=self.data,
                batches=self.batches,
                parameters=self.parameters,
                epoch=len(self.parameters.algorithm_options.times),
            )

            if self.parameters.object_options.positivity_constraint:
                self.parameters.psi = self.comm.pool.map(
                    tike.ptycho.object.positivity_constraint,
                    self.parameters.psi,
                    r=self.parameters.object_options.positivity_constraint,
                )

            if self.parameters.object_options.smoothness_constraint:
                self.parameters.psi = self.comm.pool.map(
                    tike.ptycho.object.smoothness_constraint,
                    self.parameters.psi,
                    a=self.parameters.object_options.smoothness_constraint,
                )

            if self.parameters.object_options.clip_magnitude:
                self.parameters.psi = self.comm.pool.map(
                    _clip_magnitude,
                    self.parameters.psi,
                    a_max=1.0,
                )

            if (
                self.parameters.algorithm_options.name != 'dm'
                and self.parameters.algorithm_options.rescale_method == 'mean_of_abs_object'
                and self.parameters.object_options.preconditioner is not None
                and len(self.parameters.algorithm_options.costs) % self.parameters.algorithm_options.rescale_period == 0
            ):  # yapf: disable
                (
                    self.parameters.psi,
                    self.parameters.probe,
                ) = (list(a) for a in zip(*self.comm.pool.map(
                    tike.ptycho.object.remove_object_ambiguity,
                    self.parameters.psi,
                    self.parameters.probe,
                    self.parameters.object_options.preconditioner,
                )))

            elif self.parameters.probe_options is not None:
                if (
                    self.parameters.probe_options.recover_probe
                    and self.parameters.algorithm_options.rescale_method == 'constant_probe_photons'
                    and len(self.parameters.algorithm_options.costs) % self.parameters.algorithm_options.rescale_period == 0
                ):  # yapf: disable

                    self.parameters.probe = self.comm.pool.map(
                        tike.ptycho.probe
                        .rescale_probe_using_fixed_intensity_photons,
                        self.parameters.probe,
                        Nphotons=self.parameters.probe_options.probe_photons,
                        probe_power_fraction=None,
                    )

            if (
                self.parameters.probe_options is not None
                and self.parameters.eigen_probe is not None
                and self.parameters.probe_options.recover_probe
            ):  #yapf: disable
                (
                    self.parameters.eigen_probe,
                    self.parameters.eigen_weights,
                ) = tike.ptycho.probe.constrain_variable_probe(
                    self.comm,
                    self.parameters.eigen_probe,
                    self.parameters.eigen_weights,
                )

            if self.parameters.position_options:
                (
                    self.parameters.scan,
                    self.parameters.position_options,
                ) = affine_position_regularization(
                    self.comm,
                    updated=self.parameters.scan,
                    position_options=self.parameters.position_options,
                    regularization_enabled=self.parameters.position_options[
                        0
                    ].use_position_regularization,
                )

            self.parameters.algorithm_options.times.append(time.perf_counter() -
                                                           start)
            start = time.perf_counter()

            update_norm = tike.linalg.mnorm(self.parameters.psi[0] -
                                            psi_previous)

            self.parameters.object_options.update_mnorm.append(
                update_norm.get())

            logger.info(f"The object update mean-norm is {update_norm:.3e}")

            if (np.mean(self.parameters.object_options.update_mnorm[-5:])
                    < self.parameters.object_options.convergence_tolerance):
                logger.info(
                    f"The object seems converged. {update_norm:.3e} < "
                    f"{self.parameters.object_options.convergence_tolerance:.3e}"
                )
                break

            logger.info(
                '%10s cost is %+1.3e',
                self.parameters.exitwave_options.noise_model,
                np.mean(self.parameters.algorithm_options.costs[-1]),
            )

    def get_scan(self):
        reorder = np.argsort(np.concatenate(self.comm.order))
        return self.comm.pool.gather_host(
            self.parameters.scan,
            axis=-2,
        )[reorder]

    def get_result(self):
        """Return the current parameter estimates."""
        reorder = np.argsort(np.concatenate(self.comm.order))
        parameters = solvers.PtychoParameters(
            probe=self.parameters.probe[0].get(),
            psi=self.parameters.psi[0].get(),
            scan=self.comm.pool.gather_host(
                self.parameters.scan,
                axis=-2,
            )[reorder],
            algorithm_options=self.parameters.algorithm_options,
        )

        if self.parameters.eigen_probe is not None:
            parameters.eigen_probe = self.parameters.eigen_probe[0].get()

        if self.parameters.eigen_weights is not None:
            parameters.eigen_weights = self.comm.pool.gather(
                self.parameters.eigen_weights,
                axis=-3,
            )[reorder].get()

        if self.parameters.probe_options is not None:
            parameters.probe_options = self.parameters.probe_options.copy_to_host(
            )

        if self.parameters.object_options is not None:
            parameters.object_options = self.parameters.object_options.copy_to_host(
            )

        if self.parameters.exitwave_options is not None:
            parameters.exitwave_options = self.parameters.exitwave_options.copy_to_host(
            )

        if self.parameters.position_options is not None:
            host_position_options = self.parameters.position_options[0].empty()
            for x, o in zip(
                    self.comm.pool.map(
                        PositionOptions.copy_to_host,
                        self.parameters.position_options,
                    ),
                    self.comm.order,
            ):
                host_position_options = host_position_options.join(x, o)
            parameters.position_options = host_position_options

        return parameters

    def __exit__(self, type, value, traceback):
        self.parameters = self.get_result()
        self.comm.__exit__(type, value, traceback)
        self.operator.__exit__(type, value, traceback)
        self.device.__exit__(type, value, traceback)
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        pinned_mempool.free_all_blocks()

    def get_convergence(
        self
    ) -> typing.Tuple[typing.List[typing.List[float]], typing.List[float]]:
        """Return the cost function values and times as a tuple."""
        return (
            self.parameters.algorithm_options.costs,
            self.parameters.algorithm_options.times,
        )

    def get_psi(self) -> np.array:
        """Return the current object estimate as a numpy array."""
        return self.parameters.psi[0].get()

    def get_probe(self) -> typing.Tuple[np.array, np.array, np.array]:
        """Return the current probe, eigen_probe, weights as numpy arrays."""
        reorder = np.argsort(np.concatenate(self.comm.order))
        if self.parameters.eigen_probe is None:
            eigen_probe = None
        else:
            eigen_probe = self.parameters.eigen_probe[0].get()
        if self.parameters.eigen_weights is None:
            eigen_weights = None
        else:
            eigen_weights = self.comm.pool.gather(
                self.parameters.eigen_weights,
                axis=-3,
            )[reorder].get()
        probe = self.parameters.probe[0].get()
        return probe, eigen_probe, eigen_weights

    def peek(self) -> typing.Tuple[np.array, np.array, np.array, np.array]:
        """Return the curent values of object and probe as numpy arrays.

        Parameters returned in a tuple of object, probe, eigen_probe,
        eigen_weights.
        """
        psi = self.get_psi()
        probe, eigen_probe, eigen_weights = self.get_probe()
        return psi, probe, eigen_probe, eigen_weights

    def append_new_data(
        self,
        new_data: npt.NDArray,
        new_scan: npt.NDArray,
    ) -> None:
        """Append new diffraction patterns and positions to existing result."""
        msg = "Adding data on-the-fly is disabled until further notice."
        raise NotImplementedError(msg)
        # Assign positions and data to correct devices.
        if (not np.all(np.isfinite(new_data)) or np.any(new_data < 0)):
            warnings.warn(
                "New diffraction patterns contain invalid data. "
                "All data should be non-negative and finite.", UserWarning)
        odd_pool = self.comm.pool.num_workers % 2
        (
            order,
            new_scan,
            new_data,
        ) = tike.cluster.by_scan_grid(
            new_data,
            scan=new_scan,
            pool=self.comm.pool,
            shape=(
                self.comm.pool.num_workers
                if odd_pool else self.comm.pool.num_workers // 2,
                1 if odd_pool else 2,
            ),
            dtype=(self.parameters.scan[0].dtype, self.data[0].dtype),
            destination=('gpu', 'pinned'),
        )
        # TODO: Perform sqrt of data here if gaussian model.
        # FIXME: Append makes a copy of each array!
        self.data = self.comm.pool.map(
            cp.append,
            self.data,
            new_data,
            axis=0,
        )
        self.parameters.scan = self.comm.pool.map(
            cp.append,
            self.parameters.scan,
            new_scan,
            axis=0,
        )
        self.comm.order = self.comm.pool.map(
            _order_join,
            self.comm.order,
            order,
        )

        # Rebatch on each device
        self.batches = self.comm.pool.map(
            getattr(tike.cluster,
                    self.parameters.algorithm_options.batch_method),
            self.parameters.scan,
            num_cluster=self.parameters.algorithm_options.num_batch,
        )

        if self.parameters.eigen_weights is not None:
            self.parameters.eigen_weights = self.comm.pool.map(
                cp.pad,
                self.parameters.eigen_weights,
                pad_width=(
                    (0, len(new_scan)),  # position
                    (0, 0),  # eigen
                    (0, 0),  # shared
                ),
                mode='mean',
            )

        if self.parameters.position_options is not None:
            self.parameters.position_options = self.comm.pool.map(
                PositionOptions.append,
                self.parameters.position_options,
                new_scan,
            )


def _order_join(a, b):
    return np.append(a, b + len(a))


def _get_rescale(
    data,
    measured_pixels,
    psi,
    scan,
    probe,
    streams,
    *,
    operator: tike.operators.Ptycho,
):

    sums = cp.zeros((2,), dtype=cp.double)

    def make_certain_args_constant(
        ind_args,
        lo,
        hi,
    ):

        (
            data,
        ) = ind_args
        nonlocal sums, scan

        intensity, _ = operator._compute_intensity(
            None,
            psi,
            scan[lo:hi],
            probe,
        )

        sums[0] += cp.sum(data[:, measured_pixels], dtype=np.double)
        sums[1] += cp.sum(intensity[:, measured_pixels], dtype=np.double)

    tike.communicators.stream.stream_and_modify2(
        f=make_certain_args_constant,
        ind_args=[
            data,
        ],
        streams=streams,
        lo=0,
        hi=len(data),
    )

    return sums


def _rescale_probe(operator, comm, data, exitwave_options, psi, scan, probe,
                   num_batch):
    """Rescale probe so model and measured intensity are similar magnitude.

    Rescales the probe so that the sum of modeled intensity at the detector is
    approximately equal to the measure intensity at the detector.
    """
    try:
        n = comm.pool.map(
            _get_rescale,
            data,
            exitwave_options.measured_pixels,
            psi,
            scan,
            probe,
            comm.streams,
            operator=operator,
        )
    except cp.cuda.memory.OutOfMemoryError:
        raise ValueError(
            "tike.ptycho.reconstruct ran out of memory! "
            "Increase num_batch to process your data in smaller chunks "
            "or use CuPy to switch to the Unified Memory Pool.")

    n = np.sqrt(comm.Allreduce_reduce_cpu(n))

    rescale = cp.asarray(n[0] / n[1])

    logger.info("Probe rescaled by %f", rescale)

    probe[0] *= rescale

    return comm.pool.bcast([probe[0]])


def reconstruct_multigrid(
    data: npt.NDArray,
    parameters: solvers.PtychoParameters,
    num_gpu: typing.Union[int, typing.Tuple[int, ...]] = 1,
    use_mpi: bool = False,
    num_levels: int = 3,
    interp: typing.Callable = solvers.options._resize_fft,
) -> solvers.PtychoParameters:
    """Solve the ptychography problem using a multi-grid method.

    .. versionadded:: 0.23.2

    Uses the same parameters as the functional reconstruct API. This function
    applies a multi-grid approach to the problem by downsampling the real-space
    input parameters and cropping the diffraction patterns to reduce the
    computational cost of early iterations.

    Parameters
    ----------
    num_levels : int > 0
        The number of times to reduce the problem by a factor of two.


    .. seealso:: :py:func:`tike.ptycho.ptycho.reconstruct`
    """
    if (data.shape[-1] * 0.5**(num_levels - 1)) < 64:
        warnings.warn('Cropping diffraction patterns to less than 64 pixels '
                      'wide is not recommended because the full doughnut'
                      ' may be visible.')

    # Downsample PtychoParameters to smallest size
    resampled_parameters = parameters.resample(0.5**(num_levels - 1), interp)

    for level in range((num_levels - 1), -1, -1):

        # Create a new reconstruction context for each level
        with tike.ptycho.Reconstruction(
                data=data if level == 0 else solvers.crop_fourier_space(
                    data,
                    data.shape[-1] // (2**level),
                ),
                parameters=resampled_parameters,
                num_gpu=num_gpu,
                use_mpi=use_mpi,
        ) as context:
            context.iterate(resampled_parameters.algorithm_options.num_iter)

        if level == 0:
            return context.parameters

        # Upsample result to next grid
        resampled_parameters = context.parameters.resample(2.0, interp)

    raise RuntimeError('This should not happen.')
