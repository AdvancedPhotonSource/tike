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
    AffineTransform,
)
from .probe import (
    constrain_center_peak,
    constrain_probe_sparsity,
    get_varying_probe,
    apply_median_filter_abs_probe,
    orthogonalize_eig,
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
        result = context.get_result()

    if (
        logger.getEffectiveLevel() <= logging.INFO
    ) and result.position_options:
        mean_scaling = 0.5 * (
            result.position_options.transform.scale0
            + result.position_options.transform.scale1
        )
        logger.info(
            f"Global scaling of {mean_scaling:.3e} detected from position correction."
            " Probably your estimate of photon energy and/or sample to detector "
            "distance is off by that amount."
        )
        t = result.position_options.transform.asarray()
        logger.info(f"""Affine transform parameters:

{t[0,0]: .3e}, {t[0,1]: .3e}
{t[1,0]: .3e}, {t[1,1]: .3e}
""")

    return result


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

        self.data: typing.List[npt.ArrayLike] = [data]
        self.parameters: typing.List[solvers.PtychoParameters] = [
            copy.deepcopy(parameters)
        ]
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
        if not np.all(np.isfinite(self.data[0])) or np.any(self.data[0] < 0):
            warnings.warn(
                "Diffraction patterns contain invalid data. "
                "All data should be non-negative and finite.", UserWarning)

        (
            self.comm.order,
            self.batches,
            self.comm.stripe_start,
        ) = tike.cluster.by_scan_stripes_contiguous(
            scan=self.parameters[0].scan,
            pool=self.comm.pool,
            shape=(self.comm.pool.num_workers, 1),
            batch_method=self.parameters[0].algorithm_options.batch_method,
            num_batch=self.parameters[0].algorithm_options.num_batch,
        )

        self.data = self.comm.pool.map(
            tike.cluster._split_pinned,
            self.comm.order,
            x=self.data[0],
            dtype=tike.precision.floating
            if self.data[0].itemsize > 2
            else self.data[0].dtype,
        )

        self.parameters = self.comm.pool.map(
            solvers.PtychoParameters.split,
            self.comm.order,
            x=self.parameters[0],
        )
        assert len(self.parameters) == self.comm.pool.num_workers, (
            len(self.parameters),
            self.comm.pool.num_workers,
        )
        assert self.parameters[0].psi.dtype == tike.precision.cfloating, self.parameters[0].psi.dtype
        assert self.parameters[0].probe.dtype == tike.precision.cfloating, self.parameters[0].probe.dtype
        assert self.parameters[0].scan.dtype == tike.precision.floating, self.parameters[0].probe.dtype

        self.parameters = self.comm.pool.map(
            solvers.PtychoParameters.copy_to_device,
            self.parameters,
        )
        assert len(self.parameters) == self.comm.pool.num_workers, (
            len(self.parameters),
            self.comm.pool.num_workers,
        )
        assert self.parameters[0].psi.dtype == tike.precision.cfloating, self.parameters[0].psi.dtype
        assert self.parameters[0].probe.dtype == tike.precision.cfloating, self.parameters[0].probe.dtype
        assert self.parameters[0].scan.dtype == tike.precision.floating, self.parameters[0].probe.dtype

        if self.parameters[0].probe_options is not None:
            if self.parameters[0].probe_options.init_rescale_from_measurements:
                self.parameters = _rescale_probe(
                    self.operator,
                    self.comm,
                    self.data,
                    self.parameters,
                )
        assert self.parameters[0].psi.dtype == tike.precision.cfloating, self.parameters[0].psi.dtype
        assert self.parameters[0].probe.dtype == tike.precision.cfloating, self.parameters[0].probe.dtype
        assert self.parameters[0].scan.dtype == tike.precision.floating, self.parameters[0].probe.dtype

        return self

    def iterate(self, num_iter: int) -> None:
        """Advance the reconstruction by num_iter epochs."""
        start = time.perf_counter()
        # psi_previous = self.parameters[0].psi.copy()
        for i in range(num_iter):

            if (
                np.sum(self.parameters[0].algorithm_options.times)
                > self.parameters[0].algorithm_options.time_limit
            ):
                logger.info("Maximum reconstruction time exceeded.")
                break

            logger.info(
                f"{self.parameters[0].algorithm_options.name} epoch "
                f"{len(self.parameters[0].algorithm_options.times):,d}"
            )

            total_epochs = len(self.parameters[0].algorithm_options.times)

            self.parameters = self.comm.pool.map(
                _apply_probe_constraints,
                self.parameters,
                epoch=total_epochs
            )

            self.parameters = solvers.update_preconditioners(
                comm=self.comm,
                parameters=self.parameters,
                operator=self.operator,
            )

            self.parameters = self.comm.pool.map(
                getattr(solvers, self.parameters[0].algorithm_options.name),
                self.parameters,
                self.data,
                self.batches,
                self.comm.streams,
                op=self.operator,
                epoch=len(self.parameters[0].algorithm_options.times),
            )

            self.parameters = self.comm.pool.map(
                _apply_object_constraints,
                self.parameters,
            )

            self.parameters = self.comm.pool.map(
                _apply_position_constraints,
                self.parameters,
            )

            for i, reduced_probe in enumerate(
                self.comm.Allreduce_mean(
                    [e.probe[None, ...] for e in self.parameters],
                    axis=0,
                )
            ):
                self.parameters[i].probe = reduced_probe

            if self.parameters[0].eigen_probe is not None:
                for i, reduced_probe in enumerate(
                    self.comm.Allreduce_mean(
                        [e.eigen_probe[None, ...] for e in self.parameters],
                        axis=0,
                    )
                ):
                    self.parameters[i].eigen_probe = reduced_probe

            pw = self.parameters[0].probe.shape[-2]
            for swapped, parameters in zip(
                # TODO: Try blending edges during swap instead of replacing
                self.comm.swap_edges(
                    [e.psi for e in self.parameters],
                    # reduce overlap to stay away from edge noise
                    overlap=pw-1,
                    # The actual edge is centered on the probe
                    edges=self.comm.stripe_start,
                ),
                self.parameters,
            ):
                parameters.psi = swapped

            if self.parameters[0].position_options is not None:
                # FIXME: Synchronize across nodes
                reduced_transform = np.mean(
                    [e.position_options.transform.asbuffer() for e in self.parameters],
                    axis=0,
                )
                for i in range(len(self.parameters)):
                    self.parameters[
                        i
                    ].position_options.transform = AffineTransform.frombuffer(
                        reduced_transform
                    )

            reduced_cost = np.mean(
                [e.algorithm_options.costs[-1] for e in self.parameters],
            )
            for i in range(len(self.parameters)):
                self.parameters[i].algorithm_options.costs[-1] = [reduced_cost]

            self.parameters[0].algorithm_options.times.append(
                time.perf_counter() - start
            )
            start = time.perf_counter()

            # update_norm = tike.linalg.mnorm(self.parameters.psi[0] -
            #                                 psi_previous)

            # self.parameters.object_options.update_mnorm.append(
            #     update_norm.get())

            # logger.info(f"The object update mean-norm is {update_norm:.3e}")

            # if (np.mean(self.parameters.object_options.update_mnorm[-5:])
            #         < self.parameters.object_options.convergence_tolerance):
            #     logger.info(
            #         f"The object seems converged. {update_norm:.3e} < "
            #         f"{self.parameters.object_options.convergence_tolerance:.3e}"
            #     )
            #     break

            logger.info(
                "%10s cost is %+1.3e",
                self.parameters[0].exitwave_options.noise_model,
                np.mean(self.parameters[0].algorithm_options.costs[-1]),
            )

    def get_scan(self) -> npt.NDArray:
        reorder = np.argsort(np.concatenate(self.comm.order))
        return np.concatenate(
            [cp.asnumpy(e.scan) for e in self.parameters],
            axis=0,
        )[reorder]

    def get_result(self) -> solvers.PtychoParameters:
        """Return the current parameter estimates."""
        reorder = np.argsort(np.concatenate(self.comm.order))

        assert len(self.parameters) == self.comm.pool.num_workers, (
            len(self.parameters),
            self.comm.pool.num_workers,
        )

        # Use plain map here instead of threaded map so this method can be
        # called when the context is closed.
        parameters = list(
            map(
                solvers.PtychoParameters.copy_to_host,
                self.parameters,
            )
        )

        parameters = solvers.PtychoParameters.join(
            parameters,
            reorder,
            stripe_start=self.comm.stripe_start,
        )

        return parameters

    def __exit__(self, type, value, traceback):
        self.parameters = self.comm.pool.map(
            solvers.PtychoParameters.copy_to_host,
            self.parameters,
        )
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
            self.parameters[0].algorithm_options.costs,
            self.parameters[0].algorithm_options.times,
        )

    def get_psi(self) -> np.array:
        """Return the current object estimate as a numpy array."""
        return ObjectOptions.join_psi(
            [cp.asnumpy(e.psi) for e in self.parameters],
            probe_width=self.parameters[0].probe.shape[-2],
            stripe_start=self.comm.stripe_start,
        )

    def get_probe(self) -> typing.Tuple[np.array, np.array, np.array]:
        """Return the current probe, eigen_probe, weights as numpy arrays."""
        if self.parameters[0].eigen_probe is None:
            eigen_probe = None
        else:
            eigen_probe = self.parameters[0].eigen_probe.get()
        if self.parameters.eigen_weights is None:
            eigen_weights = None
        else:
            reorder = np.argsort(np.concatenate(self.comm.order))
            eigen_weights = self.comm.pool.gather(
                self.parameters.eigen_weights,
                axis=-3,
            )[reorder].get()
        probe = self.parameters[0].probe.get()
        return probe, eigen_probe, eigen_weights

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

def _apply_probe_constraints(
    parameters: solvers.PtychoParameters,
    *,
    epoch: int,
) -> solvers.PtychoParameters:
    if parameters.probe_options is not None:
        if parameters.probe_options.recover_probe(epoch):

            if parameters.probe_options.median_filter_abs_probe:
                parameters.probe = apply_median_filter_abs_probe(
                    parameters.probe,
                    med_filt_px=parameters.probe_options.median_filter_abs_probe_px,
                )

            if parameters.probe_options.force_centered_intensity:
                parameters.probe = constrain_center_peak(
                    parameters.probe,
                )

            if parameters.probe_options.force_sparsity < 1:
                parameters.probe = constrain_probe_sparsity(
                    parameters.probe,
                    f=parameters.probe_options.force_sparsity,
                )

            if parameters.probe_options.force_orthogonality:
                (
                    parameters.probe,
                    power,
                ) = orthogonalize_eig(
                    parameters.probe,
                )
            else:
                power = tike.ptycho.probe.power(
                    parameters.probe,
                )

            parameters.probe_options.power.append(cp.asnumpy(power))

        if parameters.algorithm_options.rescale_method == "constant_probe_photons" and (
            len(parameters.algorithm_options.costs)
            % parameters.algorithm_options.rescale_period
            == 0
        ):
            parameters.probe = (
                tike.ptycho.probe.rescale_probe_using_fixed_intensity_photons(
                    parameters.probe,
                    Nphotons=parameters.probe_options.probe_photons,
                    probe_power_fraction=None,
                )
            )

        if (
            parameters.eigen_probe is not None
            and parameters.probe_options.recover_probe(epoch)
        ):
            (
                parameters.eigen_probe,
                parameters.eigen_weights,
            ) = tike.ptycho.probe.constrain_variable_probe(
                parameters.eigen_probe,
                parameters.eigen_weights,
            )

    return parameters


def _apply_object_constraints(
    parameters: solvers.PtychoParameters,
) -> solvers.PtychoParameters:
    if parameters.object_options.positivity_constraint:
        parameters.psi = tike.ptycho.object.positivity_constraint(
            parameters.psi,
            r=parameters.object_options.positivity_constraint,
        )

    if parameters.object_options.smoothness_constraint:
        parameters.psi = tike.ptycho.object.smoothness_constraint(
            parameters.psi,
            a=parameters.object_options.smoothness_constraint,
        )

    if parameters.object_options.clip_magnitude:
        parameters.psi = _clip_magnitude(
            parameters.psi,
            a_max=1.0,
        )

    if (
        parameters.algorithm_options.name != "dm"
        and parameters.algorithm_options.rescale_method == "mean_of_abs_object"
        and parameters.object_options.preconditioner is not None
        and (
            len(parameters.algorithm_options.costs)
            % parameters.algorithm_options.rescale_period
            == 0
        )
    ):
        (
            parameters.psi,
            parameters.probe,
        ) = tike.ptycho.object.remove_object_ambiguity(
            parameters.psi,
            parameters.probe,
            parameters.object_options.preconditioner,
        )

    return parameters


def _apply_position_constraints(
    parameters: solvers.PtychoParameters,
) -> solvers.PtychoParameters:
    if parameters.position_options:
        (
            parameters.scan,
            parameters.position_options,
        ) = affine_position_regularization(
            updated=parameters.scan,
            position_options=parameters.position_options,
        )

    return parameters


def _order_join(a, b):
    return np.append(a, b + len(a))


def _get_rescale(
    data: npt.ArrayLike,
    parameters: solvers.PtychoParameters,
    streams: typing.List[cp.cuda.Stream],
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
        nonlocal sums

        intensity, _ = operator._compute_intensity(
            None,
            parameters.psi,
            parameters.scan[lo:hi],
            parameters.probe,
        )

        sums[0] += cp.sum(
            data[:, parameters.exitwave_options.measured_pixels], dtype=np.double
        )
        sums[1] += cp.sum(
            intensity[:, parameters.exitwave_options.measured_pixels], dtype=np.double
        )

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


def _rescale_probe(
    operator: tike.operators.Ptycho,
    comm: tike.communicators.Comm,
    data: typing.List[npt.ArrayLike],
    parameters: typing.List[solvers.PtychoParameters],
):
    """Rescale probe so model and measured intensity are similar magnitude.

    Rescales the probe so that the sum of modeled intensity at the detector is
    approximately equal to the measure intensity at the detector.
    """
    try:
        n = comm.pool.map(
            _get_rescale,
            data,
            parameters,
            comm.streams,
            operator=operator,
        )
    except cp.cuda.memory.OutOfMemoryError:
        raise ValueError(
            "tike.ptycho.reconstruct ran out of memory! "
            "Increase num_batch to process your data in smaller chunks "
            "or use CuPy to switch to the Unified Memory Pool.")

    n = np.sqrt(comm.Allreduce_reduce_cpu(n))

    # Force precision to prevent type promotion downstream
    rescale = cp.asarray(n[0] / n[1], dtype=tike.precision.floating)

    logger.info("Probe rescaled by %f", rescale)

    rescale = comm.pool.bcast([rescale])
    return comm.pool.map(
        _rescale_probe_helper,
        parameters,
        rescale,
    )


def _rescale_probe_helper(
    parameters: solvers.PtychoParameters,
    rescale: float,
) -> solvers.PtychoParameters:
    parameters.probe = parameters.probe * rescale

    if np.isnan(parameters.probe_options.probe_photons):
        parameters.probe_options.probe_photons = cp.sum(
            cp.square(cp.abs(parameters.probe))
        ).get()

    return parameters


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
            result = context.get_result()

        if level == 0:
            if (
                logger.getEffectiveLevel() <= logging.INFO
            ) and result.position_options:
                mean_scaling = 0.5 * (
                    result.position_options.transform.scale0
                    + result.position_options.transform.scale1
                )
                logger.info(
                    f"Global scaling of {mean_scaling:.3e} detected from position correction."
                    " Probably your estimate of photon energy and/or sample to detector "
                    "distance is off by that amount."
                )
                t = result.position_options.transform.asarray()
                logger.info(f"""Affine transform parameters:

{t[0,0]: .3e}, {t[0,1]: .3e}
{t[1,0]: .3e}, {t[1,1]: .3e}
""")
            return result

        # Upsample result to next grid
        resampled_parameters = result.resample(2.0, interp)

    raise RuntimeError('This should not happen.')
