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
    "gaussian",
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
from tike.opt import randomizer
from tike.ptycho import solvers
from .position import check_allowed_positions, get_padded_object
from .probe import get_varying_probe

logger = logging.getLogger(__name__)


def gaussian(size, rin=0.8, rout=1.0):
    """Return a complex gaussian probe distribution.

    Illumination probe represented on a 2D regular grid.

    A finite-extent circular shaped probe is represented as
    a complex wave. The intensity of the probe is maximum at
    the center and damps to zero at the borders of the frame.

    Parameters
    ----------
    size : int
        The side length of the distribution
    rin : float [0, 1) < rout
        The inner radius of the distribution where the dampening of the
        intensity will start.
    rout : float (0, 1] > rin
        The outer radius of the distribution where the intensity will reach
        zero.

    """
    r, c = np.mgrid[:size, :size] + 0.5
    rs = np.sqrt((r - size / 2)**2 + (c - size / 2)**2)
    rmax = np.sqrt(2) * 0.5 * rout * rs.max() + 1.0
    rmin = np.sqrt(2) * 0.5 * rin * rs.max()
    img = np.zeros((size, size), dtype='float32')
    img[rs < rmin] = 1.0
    img[rs > rmax] = 0.0
    zone = np.logical_and(rs > rmin, rs < rmax)
    img[zone] = np.divide(rmax - rs[zone], rmax - rmin)
    return img


def compute_intensity(
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
            probe=get_varying_probe(probe, eigen_probe, eigen_weights, m=m),
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
    probe : (..., POSI, EIGEN, SHARED, WIDE, HIGH) complex64
        The shared probes.
    scan : (..., POSI, 2) float32
        Coordinates of the minimum corner of the probe grid for each
        measurement in the coordinate system of psi.
    psi : (..., WIDE, HIGH) complex64
        The complex wavefront modulation of the object.
    eigen_weights : (..., POSI, EIGEN, SHARED) float32
        Eigen probe weights for each position.
    fly : int
        The number of scan positions which combine for one detector frame.

    Returns
    -------
    data : (..., FRAME, WIDE, HIGH) float32
        The simulated intensity on the detector.

    """
    check_allowed_positions(scan, psi, probe)
    with Ptycho(
            probe_shape=probe.shape[-1],
            detector_shape=int(detector_shape),
            nz=psi.shape[-2],
            n=psi.shape[-1],
            ntheta=scan.shape[0],
            **kwargs,
    ) as operator:
        scan = operator.asarray(scan, dtype='float32')
        psi = operator.asarray(psi, dtype='complex64')
        probe = operator.asarray(probe, dtype='complex64')
        if eigen_weights is not None:
            eigen_weights = operator.asarray(eigen_weights, dtype='float32')
        data = compute_intensity(operator, psi, scan, probe, eigen_weights,
                                 eigen_probe, fly)
        return operator.asnumpy(data.real)


def reconstruct(
        data,
        probe, scan,
        algorithm,
        psi=None, num_gpu=1, num_iter=1, rtol=-1,
        model='gaussian', use_mpi=False, cost=None, times=None,
        eigen_probe=None, eigen_weights=None,
        **kwargs
):  # yapf: disable
    """Solve the ptychography problem using the given `algorithm`.

    Parameters
    ----------
    algorithm : string
        The name of one algorithms from :py:mod:`.ptycho.solvers`.
    rtol : float
        Terminate early if the relative decrease of the cost function is
        less than this amount.
    split : 'grid' or 'stripe'
        The method to use for splitting the scan positions among GPUS.
    """
    (psi, scan) = get_padded_object(scan, probe) if psi is None else (psi, scan)
    check_allowed_positions(scan, psi, probe)
    if use_mpi is True:
        mpi = MPIComm
    else:
        mpi = None
    if algorithm in solvers.__all__:
        # Initialize an operator.
        with Ptycho(
                probe_shape=probe.shape[-1],
                detector_shape=data.shape[-1],
                nz=psi.shape[-2],
                n=psi.shape[-1],
                ntheta=scan.shape[0],
                model=model,
        ) as operator, Comm(num_gpu, mpi) as comm:
            logger.info("{} for {:,d} - {:,d} by {:,d} frames for {:,d} "
                        "iterations.".format(algorithm, *data.shape[-3:],
                                             num_iter))
            # Divide the inputs into regions
            odd_pool = comm.pool.num_workers % 2
            order, scan, data, eigen_weights = split_by_scan_grid(
                comm.pool,
                (
                    comm.pool.num_workers
                    if odd_pool else comm.pool.num_workers // 2,
                    1 if odd_pool else 2,
                ),
                scan,
                data,
                eigen_weights,
            )
            result = {
                'psi':
                    comm.pool.bcast(psi.astype('complex64')),
                'probe':
                    comm.pool.bcast(probe.astype('complex64')),
                'eigen_probe':
                    comm.pool.bcast(eigen_probe.astype('complex64'))
                    if eigen_probe is not None else None,
                'scan':
                    scan,
                'eigen_weights':
                    eigen_weights,
            }
            for key, value in kwargs.items():
                if np.ndim(value) > 0:
                    kwargs[key] = comm.pool.bcast(value)

            result['probe'] = comm.pool.bcast(
                _rescale_obj_probe(
                    operator,
                    comm,
                    data[0],
                    result['psi'][0],
                    scan[0],
                    result['probe'][0],
                ))

            costs = []
            times = []
            start = time.perf_counter()
            for i in range(num_iter):

                logger.info(f"{algorithm} epoch {i:,d}")

                kwargs.update(result)
                result = getattr(solvers, algorithm)(
                    operator,
                    comm,
                    data=data,
                    **kwargs,
                )
                if result['cost'] is not None:
                    costs.append(result['cost'])

                times.append(time.perf_counter() - start)
                start = time.perf_counter()

                # Check for early termination
                if i > 0 and abs((costs[-1] - costs[-2]) / costs[-2]) < rtol:
                    logger.info(
                        "Cost function rtol < %g reached at %d "
                        "iterations.", rtol, i)
                    break

            reorder = np.argsort(np.concatenate(order))
            result['scan'] = comm.pool.gather(scan, axis=1)[:, reorder]
            if 'eigen_weights' in result:
                result['eigen_weights'] = comm.pool.gather(
                    eigen_weights,
                    axis=1,
                )[:, reorder]
                result['eigen_probe'] = result['eigen_probe'][0]
            result['probe'] = result['probe'][0]
            result['cost'] = operator.asarray(costs)
            result['times'] = operator.asarray(times)
            for k, v in result.items():
                if isinstance(v, list):
                    result[k] = v[0]
        return {k: operator.asnumpy(v) for k, v in result.items()}
    else:
        raise ValueError(f"The '{algorithm}' algorithm is not an option.\n"
                         f"\tAvailable algorithms are : {solvers.__all__}")


def _rescale_obj_probe(operator, comm, data, psi, scan, probe):
    """Keep the object amplitude around 1 by scaling probe by a constant."""

    intensity, _ = operator._compute_intensity(data, psi, scan, probe)

    rescale = (np.linalg.norm(np.ravel(np.sqrt(data))) /
               np.linalg.norm(np.ravel(np.sqrt(intensity))))

    logger.info("object and probe rescaled by %f", rescale)

    probe *= rescale

    return probe


def split_by_scan_grid(pool, shape, scan, *args, fly=1):
    """Split the field of view into a 2D grid.

    Mask divide the data into a 2D grid of spatially contiguous regions.

    Parameters
    ----------
    shape : tuple of int
        The number of grid divisions along each dimension.
    scan : (ntheta, nscan, 2) float32
        The 2D coordinates of the scan positions.
    args : (ntheta, nscan, ...) float32
        The arrays to be split by scan position.
    fly : int
        The number of scan positions per frame.

    Returns
    -------
    order : List[array[int]]
        The locations of the inputs in the original arrays.
    scan : List[array[float32]]
        The divided 2D coordinates of the scan positions.
    args : List[array[float32]]
        Each input divided into regions.
    """
    if len(shape) != 2:
        raise ValueError('The grid shape must have two dimensions.')
    vstripes = split_by_scan_stripes(scan, shape[0], axis=0, fly=fly)
    hstripes = split_by_scan_stripes(scan, shape[1], axis=1, fly=fly)
    mask = [np.logical_and(*pair) for pair in product(vstripes, hstripes)]

    order = np.arange(scan.shape[1])
    order = [order[m] for m in mask]

    def split(m, x):
        return None if x is None else cp.asarray(x[:, m], dtype='float32')

    split_args = [list(pool.map(split, mask, x=arg)) for arg in [scan, *args]]

    return (order, *split_args)


def split_by_scan_stripes(scan, n, fly=1, axis=0):
    """Return `n` boolean masks that split the field of view into stripes.

    Mask divide the data into spatially contiguous regions along the position
    axis.

    Split scan into three stripes:
    >>> [scan[:, s] for s in split_by_scan_stripes(scan, 3)]

    FIXME: Only uses the first view to divide the positions. Assumes the
    positions on all angles are distributed similarly.

    Parameters
    ----------
    scan : (ntheta, nscan, 2) float32
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
    if scan.ndim != 3:
        raise ValueError('scan must have three dimensions.')
    if n < 1:
        raise ValueError('The number of stripes must be > 0.')

    ntheta, nscan, _ = scan.shape
    if (nscan // fly) * fly != nscan:
        raise ValueError('The number of scan positions must be an '
                         'integer multiple of the number of fly positions.')

    # Reshape scan so positions in the same fly scan are not separated
    scan = scan.reshape(ntheta, nscan // fly, fly, 2)

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
            edges[i] < scan[0, :, 0, axis],
            scan[0, :, 0, axis] <= edges[i + 1],
        ).repeat(fly) for i in range(n)
    ]
