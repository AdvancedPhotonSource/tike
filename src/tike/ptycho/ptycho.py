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

__author__ = "Doga Gursoy, Daniel Ching"
__copyright__ = "Copyright (c) 2018, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = [
    "gaussian",
    "reconstruct",
    "simulate",
]

import logging
import time

import numpy as np

from tike.operators import Ptycho
from tike.pool import ThreadPool
from tike.ptycho import solvers
from .position import check_allowed_positions, get_padded_object

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


def simulate(
        detector_shape,
        probe, scan,
        psi,
        **kwargs
):  # yapf: disable
    """Return real-valued detector counts of simulated ptychography data."""
    assert scan.ndim == 3
    assert psi.ndim == 3
    check_allowed_positions(scan, psi, probe)
    with Ptycho(
            probe_shape=probe.shape[-1],
            detector_shape=int(detector_shape),
            nz=psi.shape[-2],
            n=psi.shape[-1],
            ntheta=scan.shape[0],
            **kwargs,
    ) as operator:
        data = 0
        for mode in np.split(probe, probe.shape[-3], axis=-3):
            farplane = operator.fwd(
                probe=operator.asarray(mode, dtype='complex64'),
                scan=operator.asarray(scan, dtype='float32'),
                psi=operator.asarray(psi, dtype='complex64'),
                **kwargs,
            )
            data += np.square(
                np.linalg.norm(
                    farplane.reshape(operator.ntheta,
                                     scan.shape[-2] // operator.fly, -1,
                                     detector_shape, detector_shape),
                    ord=2,
                    axis=2,
                ))
        return operator.asnumpy(data.real)

def reconstruct(
        data,
        probe, scan,
        algorithm,
        psi=None, num_gpu=1, num_iter=1, rtol=-1, split=None,
        model='gaussian', cost=None, times=None,
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
    if algorithm in solvers.__all__:
        # Initialize an operator.
        with Ptycho(
                probe_shape=probe.shape[-1],
                detector_shape=data.shape[-1],
                nz=psi.shape[-2],
                n=psi.shape[-1],
                ntheta=scan.shape[0],
                model=model,
        ) as operator, ThreadPool(num_gpu) as pool:
            logger.info("{} for {:,d} - {:,d} by {:,d} frames for {:,d} "
                        "iterations.".format(algorithm, *data.shape[1:],
                                             num_iter))
            if split == 'grid':
                scan, data = split_by_scan_grid(
                    operator,
                    pool.num_workers,
                    scan,
                    data,
                )
            else:
                scan, data = split_by_scan_stripes(pool, scan, data,
                                                   operator.fly)
            result = {
                'psi': pool.bcast(psi.astype('complex64')),
                'probe': pool.bcast(probe.astype('complex64')),
                'scan': scan,
            }
            for key, value in kwargs.items():
                if np.ndim(value) > 0:
                    kwargs[key] = pool.bcast(value)

            result['probe'] = _rescale_obj_probe(operator, pool, data,
                                                 result['psi'], result['scan'],
                                                 result['probe'])

            costs = []
            times = []
            start = time.perf_counter()
            for i in range(num_iter):
                kwargs.update(result)
                result = getattr(solvers, algorithm)(
                    operator,
                    pool,
                    data=data,
                    **kwargs,
                )
                costs.append(result['cost'])
                times.append(time.perf_counter() - start)
                start = time.perf_counter()

                # Check for early termination
                if i > 0 and abs((costs[-1] - costs[-2]) / costs[-2]) < rtol:
                    logger.info(
                        "Cost function rtol < %g reached at %d "
                        "iterations.", rtol, i)
                    break

            result['scan'] = pool.gather(result['scan'], axis=1)
            result['cost'] = operator.asarray(costs)
            result['times'] = operator.asarray(times)
            for k, v in result.items():
                if isinstance(v, list):
                    result[k] = v[0]
        return {k: operator.asnumpy(v) for k, v in result.items()}
    else:
        raise ValueError(
            "The '{}' algorithm is not an available.".format(algorithm))


def _rescale_obj_probe(operator, pool, data, psi, scan, probe):
    """Keep the object amplitude around 1 by scaling probe by a constant."""
    # TODO: add multi-GPU support
    scan = pool.gather(scan, axis=1)
    data = pool.gather(data, axis=1)
    psi = psi[0]
    probe = probe[0]

    intensity = operator._compute_intensity(data, psi, scan, probe)

    rescale = (np.linalg.norm(np.ravel(np.sqrt(data))) /
               np.linalg.norm(np.ravel(np.sqrt(intensity))))

    logger.info("object and probe rescaled by %f", rescale)

    probe *= rescale

    probe = pool.bcast(probe)
    del scan
    del data

    return probe


def split_by_scan_grid(op, gpu_count, scan_cpu, data_cpu, *args, **kwargs):
    """Split scan and data and distribute to multiple GPUs.

    Instead of spliting the arrays based on the scanning order, we split
    them in accordance with the scan positions corresponding to the object
    sub-images. For example, if we divide a square object image into four
    sub-images, then the scan positions on the top-left sub-image and their
    corresponding diffraction patterns will be grouped into the first chunk
    of scan and data.

    """
    if gpu_count == 1:
        return ([op.asarray(scan_cpu, dtype='float32')],
                [op.asarray(data_cpu, dtype='float32')])

    scanmlist = [None] * gpu_count
    datamlist = [None] * gpu_count
    nscan = scan_cpu.shape[1]
    tmplist = [0] * nscan
    counter = [0] * gpu_count
    xmax = np.amax(scan_cpu[:, :, 0])
    ymax = np.amax(scan_cpu[:, :, 1])
    for e in range(nscan):
        xgpuid = scan_cpu[0, e, 0] // (xmax / (gpu_count // 2)) - int(
            scan_cpu[0, e, 0] != 0 and scan_cpu[0, e, 0] %
            (xmax / (gpu_count // 2)) == 0)
        ygpuid = scan_cpu[0, e, 1] // (ymax / 2) - int(
            scan_cpu[0, e, 1] != 0 and scan_cpu[0, e, 1] % (ymax / 2) == 0)
        idx = int(xgpuid * 2 + ygpuid)
        tmplist[e] = idx
        counter[idx] += 1
    for i in range(gpu_count):
        tmpscan = np.zeros(
            [scan_cpu.shape[0], counter[i], scan_cpu.shape[2]],
            dtype=scan_cpu.dtype,
        )
        tmpdata = np.zeros(
            [
                data_cpu.shape[0], counter[i], data_cpu.shape[2],
                data_cpu.shape[3]
            ],
            dtype=data_cpu.dtype,
        )
        c = 0
        for e in range(nscan):
            if tmplist[e] == i:
                tmpscan[:, c, :] = scan_cpu[:, e, :]
                tmpdata[:, c] = data_cpu[:, e]
                c += 1
            scanmlist[i] = op.asarray(tmpscan, device=i)
            datamlist[i] = op.asarray(tmpdata, device=i)
        del tmpscan
        del tmpdata
    return scanmlist, datamlist


def split_by_scan_stripes(pool, scan, data, fly=1):
    """Split scan and data and distribute to multiple GPUs.

    Divide the work amongst multiple GPUS by splitting the field of view
    along the vertical axis. i.e. each GPU gets a subset of the data that
    correspond to a horizontal stripe of the field of view.

    This type of division does not minimze the volume of data transferred
    between GPUs. However, it does minimizes the number of neighbors that
    each GPU must communicate with, supports odd numbers of GPUs, and makes
    resizing the subsets easy if the scan positions are not evenly
    distributed across the field of view.

    FIXME: Only uses the first angle to divide the positions. Assumes the
    positions on all angles are distributed similarly.

    Parameters
    ----------
    pool : ThreadPool
    scan (ntheta, nscan, 2) float32
        Scan positions.
    data (ntheta, nscan // fly, D, D) float32
        Captured frames from the detector.
    fly : int
        The number of scan positions per data frame

    Returns
    -------
    scan_list, data_list : list, list
        Scan positions and data split into chunks and scattered to GPUs
    """
    # Reshape scan so positions in the same fly scan are not separated
    ntheta, nscan, _ = scan.shape
    scan = scan.reshape(ntheta, nscan // fly, fly, 2)
    # Determine the edges of the horizontal stripes
    edges = np.linspace(
        0,
        scan[..., 0].max(),
        pool.num_workers + 1,
        endpoint=True,
    )
    # Split the scan positions and data amongst the stripes
    scan_list, data_list = [], []
    for i in range(pool.num_workers):
        keep = np.logical_and(
            edges[i] < scan[0, :, 0, 0],
            scan[0, :, 0, 0] <= edges[i + 1],
        )
        scan_list.append(scan[:, keep].reshape(ntheta, -1, 2).astype('float32'))
        data_list.append(data[:, keep].astype('float32'))
    scan_list = list(pool.scatter(scan_list))
    data_list = list(pool.scatter(data_list))
    return scan_list, data_list
