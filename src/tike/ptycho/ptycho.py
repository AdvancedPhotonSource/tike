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
"""Solve the phase-retrieval problem.

Coordinate Systems
==================
`v, h` are the horizontal and vertical directions perpendicular
to the probe direction where positive directions are to the right and up.

Functions
=========
Each function in this module should have the following interface:

Parameters
----------
data : (M, V, H) :py:class:`numpy.array` float
    An array of detector intensities for each of the `M` probes. The
    grid of each detector is `H` pixels wide (the horizontal
    direction) and `V` pixels tall (the vertical direction).
probe : (V, H) :py:class:`numpy.array` complex
    The single illumination function of the `M` probes.
psi : (V, H) :py:class:`numpy.array` complex
    The object transmission function (for the current view).
foo_corner : (2, ) float [p]
    The min corner (v, h) of `foo` in the global coordinate system. `foo`
    could be `data`, `psi`, etc.
kwargs
    Keyword arguments specific to this function. `**kwargs` should always be
    included so that extra parameters are ignored instead of raising an error.

"""
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
from tike.ptycho.solvers import available_solvers

logging.basicConfig(level=logging.INFO)
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
        data_shape,
        probe, v, h,
        psi,
        **kwargs
):  # yapf: disable
    """Propagate the wavefront to the detector.

    Return real-valued intensities measured by the detector.
    """
    with PtychoBackend(
            nscan=v.size,
            probe_shape=probe.shape[0],
            detector_shape=data_shape[0],
            nz=1,
            n=1,
    ) as solver:
        return np.square(
            np.abs(solver.fwd(probe, v, h, psi, **kwargs)))


def reconstruct(
        data,
        probe, v, h,
        psi,
        algorithm=None, num_iter=1, **kwargs
):  # yapf: disable
    """Reconstruct the `psi` and `probe` using the given `algorithm`.

    Parameters
    ----------
    probe : (V, H, P) :py:class:`numpy.array` float
        The initial guess for the illumnination function of each measurement.
    psi : (V, H, P) :py:class:`numpy.array` float
        The inital guess of the object transmission function at each angle.
    algorithm : string
        The name of one of the following algorithms to use for reconstructing:
            * cgrad : conjugate gradient descent

    Returns
    -------
    new_psi : (V, H, P) :py:class:`numpy.array` float
        The updated obect transmission function at each angle.

    """
    assert len(data) == v.size == h.size, \
        "The size of v, h must be the same as the number of data."
    # Send data to c function
    logger.info("{} on {:,d} - {:,d} by {:,d} grids for {:,d} "
                "iterations".format(algorithm, len(data), *data.shape[1:],
                                    num_iter))
    if algorithm in available_solvers:
        solver = available_solvers[algorithm](
            nscan=v.size,
            probe_shape=probe.shape[0],
            detector_shape=data.shape[1],
            nz=psi.shape[0], n=psi.shape[1],
        )
        new_psi = solver.run(
            data=data,
            probe=probe, v=v, h=h,
            psi=psi,
            num_iter=num_iter,
            **kwargs
        )  # yapf: disable
    else:
        raise ValueError(
            "The {} algorithm is not an available.".format(algorithm))
    return new_psi
