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
"""This module provides wrappers for the functions from ptycho.solvers."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Doga Gursoy, Daniel Ching"
__copyright__ = "Copyright (c) 2018, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = [
    "gaussian",
    "reconstruct",
    "simulate",
]

import logging
import numpy as np

from tike.ptycho import PtychoBackend
import tike.ptycho.solvers as solvers

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
        nmode=1,
        **kwargs
):  # yapf: disable
    """Propagate the wavefront to the detector.

    Return real-valued intensities measured by the detector.
    """
    assert scan.ndim == 3
    assert psi.ndim == 3
    probe = probe.reshape(scan.shape[0], -1, nmode, *probe.shape[-2:])
    with PtychoBackend(
        nscan=scan.shape[-2],
        probe_shape=probe.shape[-1],
        detector_shape=int(detector_shape),
        nz=psi.shape[-2],
        n=psi.shape[-1],
        ntheta=scan.shape[0],
        **kwargs,
    ) as solver:
        data = 0
        for i in range(nmode):
            data += np.square(np.abs(
                solver.fwd(
                    probe=probe[:, :, i],
                    scan=scan,
                    psi=psi,
                    **kwargs,
                )
            ))
        return solver.asnumpy(data)


def reconstruct(
        data,
        probe, scan,
        psi,
        algorithm, num_iter=1, num_batches=1, rtol=1e-3, **kwargs
):  # yapf: disable
    """Reconstruct the `psi` and `probe` using the given `algorithm`.

    Solve the ptychography problem using mini-batch gradient descent.
    When num_batches equals nscan then this is stochasitc gradient descent
    When num_batches equals 1, it is batch gradient descrent.

    Parameters
    ----------
    algorithm : string
        The name of one algorithms to use for reconstructing.
    """
    if algorithm in solvers.__all__:
        # Divide the work into equal batches small enough for available memory.
        batches = np.arange(scan.shape[-2])
        if num_batches > 1:  # Don't shuffle a single batch
            np.random.shuffle(batches)
        batches = np.split(batches, num_batches)
        # Initialize an operator.
        with PtychoBackend(
                nscan=len(batches[0]),
                probe_shape=probe.shape[-1],
                detector_shape=data.shape[-1],
                nz=psi.shape[-2],
                n=psi.shape[-1],
                ntheta=scan.shape[0],
                **kwargs,
        ) as operator:

            result = {
                'psi': psi,
                'probe': probe,
            }

            scan = scan.copy()

            logger.info("{} for {:,d} - {:,d} by {:,d} frames for {:,d} "
                        "iterations in {:,d} batches.".format(
                            algorithm, *data.shape[1:], num_iter, num_batches))

            cost = 0
            for i in range(num_iter):
                for batch in batches:
                    result['scan'] = scan[:, batch]
                    kwargs.update(result)
                    result = getattr(solvers, algorithm)(
                        operator,
                        data=data[:, batch],
                        num_iter=num_iter,
                        **kwargs,
                    )  # yapf: disable
                    scan[:, batch] = result['scan']
                # Check for early termination
                cost1 = result['cost']
                if i > 0 and abs((cost1 - cost) / cost) < rtol:
                    logger.info(
                        "Cost function rtol < %g reached at %d "
                        "iterations.", rtol, i)
                    break
                cost = cost1
        return result
    else:
        raise ValueError(
            "The {} algorithm is not an available.".format(algorithm))
