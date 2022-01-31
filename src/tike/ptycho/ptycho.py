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
]

from itertools import product, chain
import logging
import time

import numpy as np
import cupy as cp

from tike.operators import Ptycho
from tike.communicators import Comm, MPIComm
from tike.opt import batch_indicies
from tike.ptycho import solvers
from tike.random import cluster_wobbly_center

from .object import get_padded_object
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
    data,
    probe,
    scan,
    algorithm_options=solvers.RpieOptions(),
    eigen_probe=None,
    eigen_weights=None,
    model='gaussian',
    num_gpu=1,
    object_options=None,
    position_options=None,
    probe_options=None,
    psi=None,
    use_mpi=False,
):
    """Solve the ptychography problem using the given `algorithm`.

    Parameters
    ----------
    data : (FRAME, WIDE, HIGH) float32
        The intensity (square of the absolute value) of the propagated
        wavefront; i.e. what the detector records. FFT-shifted so the
        diffraction peak is at the corners.
    probe : (1, 1, SHARED, WIDE, HIGH) complex64
        The shared complex illumination function amongst all positions.
    scan : (POSI, 2) float32
        Coordinates of the minimum corner of the probe grid for each
        measurement in the coordinate system of psi. Coordinate order
        consistent with WIDE, HIGH order.
    algorithm_options : :py:class:`tike.ptycho.solvers.IterativeOptions`
        A class containing algorithm specific parameters
    eigen_probe : (EIGEN, SHARED, WIDE, HIGH) complex64
        The eigen probes for all positions.
    eigen_weights : (POSI, EIGEN, SHARED) float32
        The relative intensity of the eigen probes at each position.
    model : "gaussian", "poisson"
        The noise model to use for the cost function.
    num_gpu : int, tuple(int)
        The number of GPUs to use or a tuple of the device numbers of the GPUs
        to use. If the number of GPUs is less than the requested number, only
        workers for the available GPUs are allocated.
    object_options : :py:class:`tike.ptycho.ObjectOptions`
        A class containing settings related to object updates.
    position_options : :py:class:`tike.ptycho.PositionOptions`
        A class containing settings related to position correction.
    probe_options : :py:class:`tike.ptycho.ProbeOptions`
        A class containing settings related to probe updates.
    psi : (WIDE, HIGH) complex64
        The wavefront modulation coefficients of the object.
    use_mpi : bool
        Whether to use MPI or not.

    Returns
    -------
    result : dict
        A dictionary of the above parameters that may be passed to this
        function to resume reconstruction from the previous state.

    """
    logger.info("{} for {:,d} - {:,d} by {:,d} frames for {:,d} "
                "iterations.".format(
                    algorithm_options.name,
                    *data.shape[-3:],
                    algorithm_options.num_iter,
                ))

    if use_mpi is True:
        mpi = MPIComm
        if psi is None:
            raise ValueError(
                "When MPI is enabled, initial object guess cannot be None; "
                "automatic psi initialization is not synchronized "
                "across processes.")
    else:
        mpi = None
    (psi, scan) = get_padded_object(scan, probe) if psi is None else (psi, scan)
    check_allowed_positions(scan, psi, probe.shape)
    with cp.cuda.Device(num_gpu[0] if isinstance(num_gpu, tuple) else None):
        operator = Ptycho(
            probe_shape=probe.shape[-1],
            detector_shape=data.shape[-1],
            nz=psi.shape[-2],
            n=psi.shape[-1],
            model=model,
        )
        comm = Comm(num_gpu, mpi)

        (
            batches,
            data,
            result,
            scan,
        ) = _setup(
            algorithm_options,
            comm,
            data,
            eigen_probe,
            eigen_weights,
            object_options,
            operator,
            probe,
            psi,
            position_options,
            probe_options,
            scan,
        )

        start = time.perf_counter()
        for i in range(algorithm_options.num_iter):

            logger.info(f"{algorithm_options.name} epoch {i:,d}")

            # TODO: Append new information to everything that emits from _setup.

            result = _iterate(
                algorithm_options,
                batches,
                comm,
                data,
                operator,
                position_options,
                probe_options,
                result,
            )

            # TODO: Grab intermediate psi/probe from GPU.

            algorithm_options.times.append(time.perf_counter() - start)
            start = time.perf_counter()

        return _teardown(
            algorithm_options,
            comm,
            eigen_probe,
            eigen_weights,
            object_options,
            position_options,
            probe_options,
            result,
            scan,
        )


def _setup(
    algorithm_options,
    comm,
    data,
    eigen_probe,
    eigen_weights,
    object_options,
    operator,
    probe,
    psi,
    position_options,
    probe_options,
    scan,
):
    # Divide the inputs into regions
    odd_pool = comm.pool.num_workers % 2
    (
        comm.order,
        scan,
        data,
        eigen_weights,
        initial_scan,
    ) = split_by_scan_grid(
        comm.pool,
        (
            comm.pool.num_workers if odd_pool else comm.pool.num_workers // 2,
            1 if odd_pool else 2,
        ),
        scan,
        data,
        eigen_weights,
        None if position_options is None else position_options.initial_scan,
    )
    result = dict(
        psi=comm.pool.bcast([psi.astype('complex64')]),
        probe=comm.pool.bcast([probe.astype('complex64')]),
        scan=scan,
        probe_options=probe_options.copy_to_device()
        if probe_options is not None else None,
        object_options=object_options.copy_to_device()
        if object_options is not None else None,
        algorithm_options=algorithm_options,
    )
    if eigen_probe is not None:
        result.update({
            'eigen_probe':
                comm.pool.bcast([eigen_probe.astype('complex64')])
                if eigen_probe is not None else None,
            'eigen_weights':
                eigen_weights,
        })
    if position_options:
        # TODO: Consider combining put/split, get/join operations?
        result['position_options'] = comm.pool.map(
            PositionOptions.copy_to_device,
            (position_options.split(x) for x in comm.order),
        )
        if initial_scan is None:
            position_options.initial_scan = comm.pool.map(cp.copy, scan)
        else:
            position_options.initial_scan = initial_scan

    # Unique batch for each device
    batches = comm.pool.map(
        cluster_wobbly_center,
        scan,
        num_cluster=algorithm_options.num_batch,
    )

    result['probe'] = _rescale_obj_probe(
        operator,
        comm,
        data,
        result['psi'],
        scan,
        result['probe'],
        num_batch=algorithm_options.num_batch,
    )

    return (
        batches,
        data,
        result,
        scan,
    )


def _iterate(
    algorithm_options,
    batches,
    comm,
    data,
    operator,
    position_options,
    probe_options,
    result,
):
    if probe_options is not None:
        if probe_options.centered_intensity_constraint:
            result['probe'] = comm.pool.map(
                constrain_center_peak,
                result['probe'],
            )
        if probe_options.sparsity_constraint < 1:
            result['probe'] = comm.pool.map(
                constrain_probe_sparsity,
                result['probe'],
                f=probe_options.sparsity_constraint,
            )

    result = getattr(solvers, algorithm_options.name)(
        operator,
        comm,
        data=data,
        batches=batches,
        **result,
    )

    if (position_options and position_options.use_position_regularization):

        # TODO: Regularize on all GPUs
        result['scan'][0], _ = affine_position_regularization(
            operator,
            result['psi'][0],
            result['probe'][0],
            position_options.initial_scan[0],
            result['scan'][0],
        )

    return result


def _teardown(
    algorithm_options,
    comm,
    eigen_probe,
    eigen_weights,
    object_options,
    position_options,
    probe_options,
    result,
    scan,
):
    reorder = np.argsort(np.concatenate(comm.order))
    if position_options is not None:
        for x, o in zip(
                comm.pool.map(
                    PositionOptions.copy_to_host,
                    result['position_options'],
                ),
                comm.order,
        ):
            position_options.join(x, o)
        position_options.initial_scan = comm.pool.gather(
            position_options.initial_scan,
            axis=-2,
        )[reorder].get()

    return dict(
        algorithm_options=algorithm_options,
        eigen_probe=result['eigen_probe'][0].get()
        if eigen_probe is not None else None,
        eigen_weights=comm.pool.gather(
            result['eigen_weights'],
            axis=-3,
        )[reorder].get() if eigen_weights is not None else None,
        object_options=result['object_options'].copy_to_host()
        if object_options is not None else None,
        position_options=position_options,
        probe=result['probe'][0].get(),
        probe_options=result['probe_options'].copy_to_host()
        if probe_options is not None else None,
        psi=result['psi'][0].get(),
        scan=comm.pool.gather(scan, axis=-2)[reorder].get(),
    )


def _rescale_obj_probe(operator, comm, data, psi, scan, probe, num_batch):
    """Keep the object amplitude around 1 by scaling probe by a constant."""

    def _get_rescale(data, psi, scan, probe, num_batch, operator):
        i = batch_indicies(data.shape[-3], num_batch, use_random=True)[0]

        intensity, _ = operator._compute_intensity(data[..., i, :, :], psi,
                                                   scan[..., i, :], probe)

        n1 = np.linalg.norm(np.ravel(np.sqrt(data[..., i, :, :])))**2
        n2 = np.linalg.norm(np.ravel(np.sqrt(intensity)))**2

        return n1, n2

    n1, n2 = zip(*comm.pool.map(
        _get_rescale,
        data,
        psi,
        scan,
        probe,
        num_batch=num_batch,
        operator=operator,
    ))

    if comm.use_mpi:
        n1 = np.sqrt(comm.Allreduce_reduce(n1, 'cpu'))
        n2 = np.sqrt(comm.Allreduce_reduce(n2, 'cpu'))
    else:
        n1 = np.sqrt(comm.reduce(n1, 'cpu'))
        n2 = np.sqrt(comm.reduce(n2, 'cpu'))

    rescale = n1 / n2

    logger.info("object and probe rescaled by %f", rescale)

    probe[0] *= rescale

    return comm.pool.bcast([probe[0]])


def split_by_scan_grid(pool, shape, scan, *args, fly=1):
    """Split the field of view into a 2D grid.

    Mask divide the data into a 2D grid of spatially contiguous regions.

    Parameters
    ----------
    shape : tuple of int
        The number of grid divisions along each dimension.
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

    def split(m, x):
        return cp.asarray(x[m], dtype='float32')

    split_args = []
    for arg in [scan, *args]:
        if arg is None:
            split_args.append(None)
        else:
            split_args.append(pool.map(split, mask, x=arg))

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
