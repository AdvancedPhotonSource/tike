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
import numpy as np

from tike.operators import Ptycho
from tike.ptycho import solvers

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
    with Ptycho(
            nscan=scan.shape[-2],
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
                                     operator.nscan // operator.fly, -1,
                                     detector_shape, detector_shape),
                    ord=2,
                    axis=2,
                ))
        return operator.asnumpy(data)

def reconstruct(
        data,
        probe, scan,
        psi,
        algorithm, num_iter=1, rtol=-1, **kwargs
):  # yapf: disable
    """Solve the ptychography problem using the given `algorithm`.

    Parameters
    ----------
    algorithm : string
        The name of one algorithms from :py:mod:`.ptycho.solvers`.
    rtol : float
        Terminate early if the relative decrease of the cost function is
        less than this amount.

    """
    if algorithm in solvers.__all__:
        # Initialize an operator.
        with Ptycho(
                nscan=scan.shape[1],
                probe_shape=probe.shape[-1],
                detector_shape=data.shape[-1],
                nz=psi.shape[-2],
                n=psi.shape[-1],
                ntheta=scan.shape[0],
                **kwargs,
        ) as operator:
            # send any array-likes to device
            data = operator.asarray(data, dtype='float32')
            result = {
                'psi': operator.asarray(psi, dtype='complex64'),
                'probe': operator.asarray(probe, dtype='complex64'),
                'scan': operator.asarray(scan, dtype='float32'),
            }
            for key, value in kwargs.items():
                if np.ndim(value) > 0:
                    kwargs[key] = operator.asarray(value)

            logger.info("{} for {:,d} - {:,d} by {:,d} frames for {:,d} "
                        "iterations.".format(algorithm, *data.shape[1:],
                                             num_iter))

            cost = 0
            for i in range(num_iter):
                kwargs.update(result)
                result = getattr(solvers, algorithm)(
                    operator,
                    data=data,
                    **kwargs,
                )
                # Check for early termination
                if i > 0 and abs((result['cost'] - cost) / cost) < rtol:
                    logger.info(
                        "Cost function rtol < %g reached at %d "
                        "iterations.", rtol, i)
                    break
                cost = result['cost']

                result['probe'] = _rescale_obj_probe(operator, data,
                                                     result['psi'],
                                                     result['scan'],
                                                     result['probe'])

        return {k: operator.asnumpy(v) for k, v in result.items()}
    else:
        raise ValueError(
            "The '{}' algorithm is not an available.".format(algorithm))


def _rescale_obj_probe(operator, data, psi, scan, probe):
    """Keep the object amplitude around 1 by scaling probe by a constant."""
    intensity = operator._compute_intensity(data, psi, scan, probe)

    rescale = (np.linalg.norm(np.ravel(np.sqrt(data))) /
               np.linalg.norm(np.ravel(np.sqrt(intensity))))

    logger.info("object and probe rescaled by %f", rescale)

    probe *= rescale

    return probe
