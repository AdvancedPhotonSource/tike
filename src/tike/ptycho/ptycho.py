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
from itertools import product
import logging
import time
import typing
import warnings

import numpy as np
import cupy as cp

from tike.operators import Ptycho
from tike.communicators import Comm, MPIComm
import tike.opt
from tike.ptycho import solvers
import tike.random

from .position import (
    PositionOptions,
    check_allowed_positions,
    affine_position_regularization,
)
from .probe import (
    constrain_center_peak,
    constrain_probe_sparsity,
    get_varying_probe,
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
    with Ptycho(
            probe_shape=probe.shape[-1],
            detector_shape=int(detector_shape),
            nz=psi.shape[-2],
            n=psi.shape[-1],
            **kwargs,
    ) as operator:
        scan = operator.asarray(scan, dtype='float32')
        psi = operator.asarray(psi, dtype='complex64')
        probe = operator.asarray(probe, dtype='complex64')
        if eigen_weights is not None:
            eigen_weights = operator.asarray(eigen_weights, dtype='float32')
        data = _compute_intensity(operator, psi, scan, probe, eigen_weights,
                                  eigen_probe, fly)
        return operator.asnumpy(data.real)


def reconstruct(
    data: np.typing.NDArray,
    parameters: solvers.PtychoParameters,
    model: str = 'gaussian',
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
    model : "gaussian", "poisson"
        The noise model to use for the cost function.
    num_gpu : int, tuple(int)
        The number of GPUs to use or a tuple of the device numbers of the GPUs
        to use. If the number of GPUs is less than the requested number, only
        workers for the available GPUs are allocated.
    use_mpi : bool
        Whether to use MPI or not.

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
    with tike.ptycho.Reconstruction(
            data,
            parameters,
            model,
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
            model,
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
        data: np.typing.NDArray,
        parameters: solvers.PtychoParameters,
        model: str = 'gaussian',
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

        if use_mpi is True:
            mpi = MPIComm
            if parameters.psi is None:
                raise ValueError(
                    "When MPI is enabled, initial object guess cannot be None; "
                    "automatic psi initialization is not synchronized "
                    "across processes.")
        else:
            mpi = None

        self.data = data
        self.parameters = parameters
        self._device_parameters = copy.deepcopy(parameters)
        self.device = cp.cuda.Device(
            num_gpu[0] if isinstance(num_gpu, tuple) else None)
        self.operator = Ptycho(
            probe_shape=parameters.probe.shape[-1],
            detector_shape=data.shape[-1],
            nz=parameters.psi.shape[-2],
            n=parameters.psi.shape[-1],
            model=model,
        )
        self.comm = Comm(num_gpu, mpi)

    def __enter__(self):
        self.device.__enter__()
        self.operator.__enter__()
        self.comm.__enter__()

        # Divide the inputs into regions
        if (not np.all(np.isfinite(self.data)) or np.any(self.data < 0)):
            warnings.warn(
                "Diffraction patterns contain invalid data. "
                "All data should be non-negative and finite.", UserWarning)
        odd_pool = self.comm.pool.num_workers % 2
        (
            self.comm.order,
            self._device_parameters.scan,
            self.data,
            self._device_parameters.eigen_weights,
        ) = split_by_scan_grid(
            self.comm.pool,
            (
                self.comm.pool.num_workers
                if odd_pool else self.comm.pool.num_workers // 2,
                1 if odd_pool else 2,
            ),
            ('float32', 'uint16', 'float32'),
            self._device_parameters.scan,
            self.data,
            self._device_parameters.eigen_weights,
        )

        self._device_parameters.psi = self.comm.pool.bcast(
            [self._device_parameters.psi.astype('complex64')])

        self._device_parameters.probe = self.comm.pool.bcast(
            [self._device_parameters.probe.astype('complex64')])

        if self._device_parameters.probe_options is not None:
            self._device_parameters.probe_options = self._device_parameters.probe_options.copy_to_device(
            )

        if self._device_parameters.object_options is not None:
            self._device_parameters.object_options = self._device_parameters.object_options.copy_to_device(
                self.comm,)

        if self._device_parameters.eigen_probe is not None:
            self._device_parameters.eigen_probe = self.comm.pool.bcast(
                [self._device_parameters.eigen_probe.astype('complex64')])

        if self._device_parameters.position_options is not None:
            # TODO: Consider combining put/split, get/join operations?
            self._device_parameters.position_options = self.comm.pool.map(
                PositionOptions.copy_to_device,
                (self._device_parameters.position_options.split(x)
                 for x in self.comm.order),
            )

        # Unique batch for each device
        self.batches = self.comm.pool.map(
            getattr(tike.random,
                    self._device_parameters.algorithm_options.batch_method),
            self._device_parameters.scan,
            num_cluster=self._device_parameters.algorithm_options.num_batch,
        )

        self._device_parameters.probe = _rescale_probe(
            self.operator,
            self.comm,
            self.data,
            self._device_parameters.psi,
            self._device_parameters.scan,
            self._device_parameters.probe,
            num_batch=self._device_parameters.algorithm_options.num_batch,
        )

        return self

    def iterate(self, num_iter: int) -> None:
        """Advance the reconstruction by num_iter epochs."""
        start = time.perf_counter()
        for i in range(num_iter):

            logger.info(
                f"{self._device_parameters.algorithm_options.name} epoch "
                f"{len(self._device_parameters.algorithm_options.times):,d}")

            if self._device_parameters.probe_options is not None:
                if self._device_parameters.probe_options.centered_intensity_constraint:
                    self._device_parameters.probe = self.comm.pool.map(
                        constrain_center_peak,
                        self._device_parameters.probe,
                    )
                if self._device_parameters.probe_options.sparsity_constraint < 1:
                    self._device_parameters.probe = self.comm.pool.map(
                        constrain_probe_sparsity,
                        self._device_parameters.probe,
                        f=self._device_parameters.probe_options
                        .sparsity_constraint,
                    )

            self._device_parameters = getattr(
                solvers,
                self._device_parameters.algorithm_options.name,
            )(
                self.operator,
                self.comm,
                data=self.data,
                batches=self.batches,
                parameters=self._device_parameters,
            )

            if self._device_parameters.object_options.clip_magnitude:
                self._device_parameters.psi = self.comm.pool.map(
                    _clip_magnitude,
                    self._device_parameters.psi,
                    a_max=1.0,
                )

            if (self._device_parameters.position_options
                    and self._device_parameters.position_options[0]
                    .use_position_regularization):

                # TODO: Regularize on all GPUs
                self._device_parameters.scan[
                    0], _ = affine_position_regularization(
                        self.operator,
                        self._device_parameters.psi[0],
                        self._device_parameters.probe[0],
                        self._device_parameters.position_options
                        .initial_scan[0],
                        self._device_parameters.scan[0],
                    )

            self._device_parameters.algorithm_options.times.append(
                time.perf_counter() - start)
            start = time.perf_counter()

            if tike.opt.is_converged(self._device_parameters.algorithm_options):
                break

    def _get_result(self):
        """Return the current parameter estimates."""
        self.parameters.probe = self._device_parameters.probe[0].get()

        self.parameters.psi = self._device_parameters.psi[0].get()

        reorder = np.argsort(np.concatenate(self.comm.order))
        self.parameters.scan = self.comm.pool.gather_host(
            self._device_parameters.scan,
            axis=-2,
        )[reorder]

        if self._device_parameters.eigen_probe is not None:
            self.parameters.eigen_probe = self._device_parameters.eigen_probe[
                0].get()

        if self._device_parameters.eigen_weights is not None:
            self.parameters.eigen_weights = self.comm.pool.gather(
                self._device_parameters.eigen_weights,
                axis=-3,
            )[reorder].get()

        self.parameters.algorithm_options = self._device_parameters.algorithm_options

        if self._device_parameters.probe_options is not None:
            self.parameters.probe_options = self._device_parameters.probe_options.copy_to_host(
            )

        if self._device_parameters.object_options is not None:
            self.parameters.object_options = self._device_parameters.object_options.copy_to_host(
            )

        if self._device_parameters.position_options is not None:
            host_position_options = self._device_parameters.position_options[
                0].empty()
            for x, o in zip(
                    self.comm.pool.map(
                        PositionOptions.copy_to_host,
                        self._device_parameters.position_options,
                    ),
                    self.comm.order,
            ):
                host_position_options = host_position_options.join(x, o)
            self.parameters.position_options = host_position_options

    def __exit__(self, type, value, traceback):
        self._get_result()
        self.comm.__exit__(type, value, traceback)
        self.operator.__exit__(type, value, traceback)
        self.device.__exit__(type, value, traceback)

    def get_convergence(
        self
    ) -> typing.Tuple[typing.List[typing.List[float]], typing.List[float]]:
        """Return the cost function values and times as a tuple."""
        return (
            self._device_parameters.algorithm_options.costs,
            self._device_parameters.algorithm_options.times,
        )

    def get_psi(self) -> np.array:
        """Return the current object estimate as a numpy array."""
        return self._device_parameters.psi[0].get()

    def get_probe(self) -> typing.Tuple[np.array, np.array, np.array]:
        """Return the current probe, eigen_probe, weights as numpy arrays."""
        reorder = np.argsort(np.concatenate(self.comm.order))
        if self._device_parameters.eigen_probe is None:
            eigen_probe = None
        else:
            eigen_probe = self._device_parameters.eigen_probe[0].get()
        if self._device_parameters.eigen_weights is None:
            eigen_weights = None
        else:
            eigen_weights = self.comm.pool.gather(
                self._device_parameters.eigen_weights,
                axis=-3,
            )[reorder].get()
        probe = self._device_parameters.probe[0].get()
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
        new_data: np.typing.NDArray,
        new_scan: np.typing.NDArray,
    ) -> None:
        """Append new diffraction patterns and positions to existing result."""
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
        ) = split_by_scan_grid(
            self.comm.pool,
            (
                self.comm.pool.num_workers
                if odd_pool else self.comm.pool.num_workers // 2,
                1 if odd_pool else 2,
            ),
            ('float32', 'uint16'),
            new_scan,
            new_data,
        )
        # TODO: Perform sqrt of data here if gaussian model.
        # FIXME: Append makes a copy of each array!
        self.data = self.comm.pool.map(
            cp.append,
            self.data,
            new_data,
            axis=0,
        )
        self._device_parameters.scan = self.comm.pool.map(
            cp.append,
            self._device_parameters.scan,
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
            getattr(tike.random,
                    self._device_parameters.algorithm_options.batch_method),
            self._device_parameters.scan,
            num_cluster=self._device_parameters.algorithm_options.num_batch,
        )

        if self._device_parameters.eigen_weights is not None:
            self._device_parameters.eigen_weights = self.comm.pool.map(
                cp.pad,
                self._device_parameters.eigen_weights,
                pad_width=(
                    (0, len(new_scan)),  # position
                    (0, 0),  # eigen
                    (0, 0),  # shared
                ),
                mode='mean',
            )

        if self._device_parameters.position_options is not None:
            self._device_parameters.position_options = self.comm.pool.map(
                PositionOptions.append,
                self._device_parameters.position_options,
                new_scan,
            )


def _order_join(a, b):
    return np.append(a, b + len(a))


def _get_rescale(data, psi, scan, probe, num_batch, operator):

    n1 = 0.0
    n2 = 0.0

    for b in tike.opt.batch_indicies(data.shape[-3],
                                     num_batch,
                                     use_random=False):

        intensity, _ = operator._compute_intensity(
            data[..., b, :, :],
            psi,
            scan[..., b, :],
            probe,
        )

        n1 += np.sum(data[..., b, :, :])
        n2 += np.sum(intensity)

    return n1, n2


def _rescale_probe(operator, comm, data, psi, scan, probe, num_batch):
    """Rescale probe so model and measured intensity are similar magnitude.

    Rescales the probe so that the sum of modeled intensity at the detector is
    approximately equal to the measure intensity at the detector.
    """
    try:
        n1, n2 = zip(*comm.pool.map(
            _get_rescale,
            data,
            psi,
            scan,
            probe,
            num_batch=num_batch,
            operator=operator,
        ))
    except cp.cuda.memory.OutOfMemoryError:
        raise ValueError(
            "tike.ptycho.reconstruct ran out of memory! "
            "Increase num_batch to process your data in smaller chunks "
            "or use CuPy to switch to the Unified Memory Pool.")

    if comm.use_mpi:
        n1 = np.sqrt(comm.Allreduce_reduce(n1, 'cpu'))
        n2 = np.sqrt(comm.Allreduce_reduce(n2, 'cpu'))
    else:
        n1 = np.sqrt(comm.reduce(n1, 'cpu'))
        n2 = np.sqrt(comm.reduce(n2, 'cpu'))

    rescale = n1 / n2

    logger.info("Probe rescaled by %f", rescale)

    probe[0] *= rescale

    return comm.pool.bcast([probe[0]])


def _split(m, x, dtype):
    return cp.asarray(x[m], dtype=dtype)


def split_by_scan_grid(pool, shape, dtype, scan, *args, fly=1):
    """Split the field of view into a 2D grid.

    Mask divide the data into a 2D grid of spatially contiguous regions.

    Parameters
    ----------
    shape : tuple of int
        The number of grid divisions along each dimension.
    dtype : List[str]
        The datatypes of the args after splitting.
    scan : (nscan, 2) float32
        The 2D coordinates of the scan positions.
    args : (nscan, ...) float32 or None
        The arrays to be split by scan position.
    fly : int
        The number of scan positions per frame.

    Returns
    -------
    order : List[array[int]]
        The locations of the inputs in the original arrays.
    scan : List[array[float32]]
        The divided 2D coordinates of the scan positions.
    args : List[array[float32]] or None
        Each input divided into regions or None if arg was None.
    """
    if len(shape) != 2:
        raise ValueError('The grid shape must have two dimensions.')
    vstripes = split_by_scan_stripes(scan, shape[0], axis=0, fly=fly)
    hstripes = split_by_scan_stripes(scan, shape[1], axis=1, fly=fly)
    mask = [np.logical_and(*pair) for pair in product(vstripes, hstripes)]

    order = np.arange(scan.shape[-2])
    order = [order[m] for m in mask]

    split_args = []
    for arg, t in zip([scan, *args], dtype):
        if arg is None:
            split_args.append(None)
        else:
            split_args.append(pool.map(_split, mask, x=arg, dtype=t))

    return (order, *split_args)


def split_by_scan_stripes(scan, n, fly=1, axis=0):
    """Return `n` boolean masks that split the field of view into stripes.

    Mask divide the data into spatially contiguous regions along the position
    axis.

    Split scan into three stripes:
    >>> [scan[s] for s in split_by_scan_stripes(scan, 3)]

    FIXME: Only uses the first view to divide the positions. Assumes the
    positions on all angles are distributed similarly.

    Parameters
    ----------
    scan : (nscan, 2) float32
        The 2D coordinates of the scan positions.
    n : int
        The number of stripes.
    fly : int
        The number of scan positions per frame.
    axis : int (0 or 1)
        Which spatial dimension to divide along. i.e. horizontal or vertical.

    Returns
    -------
    mask : list of (nscan, ) boolean
        A list of boolean arrays which divide the scan positions into `n`
        stripes.

    """
    if scan.ndim != 2:
        raise ValueError('scan must have two dimensions.')
    if n < 1:
        raise ValueError('The number of stripes must be > 0.')

    nscan, _ = scan.shape
    if (nscan // fly) * fly != nscan:
        raise ValueError('The number of scan positions must be an '
                         'integer multiple of the number of fly positions.')

    # Reshape scan so positions in the same fly scan are not separated
    scan = scan.reshape(nscan // fly, fly, 2)

    # Determine the edges of the horizontal stripes
    edges = np.linspace(
        scan[..., axis].min(),
        scan[..., axis].max(),
        n + 1,
        endpoint=True,
    )

    # Move the outer edges to include all points
    edges[0] -= 1
    edges[-1] += 1

    # Generate masks which put points into stripes
    return [
        np.logical_and(
            edges[i] < scan[:, 0, axis],
            scan[:, 0, axis] <= edges[i + 1],
        ).repeat(fly) for i in range(n)
    ]


def reconstruct_multigrid(
    data: np.typing.NDArray,
    parameters: solvers.PtychoParameters,
    model: str = 'gaussian',
    num_gpu: typing.Union[int, typing.Tuple[int, ...]] = 1,
    use_mpi: bool = False,
    num_levels: int = 3,
    interp = None,
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
                model=model,
                use_mpi=use_mpi,
        ) as context:
            context.iterate(resampled_parameters.algorithm_options.num_iter)

        if level == 0:
            return context.parameters

        # Upsample result to next grid
        resampled_parameters = context.parameters.resample(2.0, interp)

    raise RuntimeError('This should not happen.')
