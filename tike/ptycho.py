#!/usr/bin/env python
# -*- coding: utf-8 -*-

# #########################################################################
# Copyright (c) 2017-2018, UChicago Argonne, LLC. All rights reserved.    #
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
psi : (T, V, H) :py:class:`numpy.array` complex
    The object transmission function for each of the `T` views.
foo_min : (2, ) float [p]
    The min corner (v, h) of `foo` in the global coordinate system. `foo`
    could be `data`, `psi`, etc.
kwargs
    Keyword arguments specific to this function. `**kwargs` should always be
    included so that extra parameters are ignored instead of raising an error.

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import logging
import scipy.ndimage.interpolation as sni

__author__ = "Doga Gursoy, Daniel Ching"
__copyright__ = "Copyright (c) 2018, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ["reconstruct",
           "simulate",
           ]


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _ptycho_interface(data, data_min,
                      probe, theta, v, h,
                      psi, psi_min, **kwargs):
    """Define and interface that all functions in this module match.

    This function also sets default values for functions in this module.
    """
    if data is None:
        raise ValueError()
    if data_min is None:
        data_min = (-0.5, -0.5)
    if probe is None:
        raise ValueError()
    if v is None:
            raise ValueError()
    if h is None:
        raise ValueError()
    if theta is None:
        pass
    if psi is None:
        raise ValueError()
    if psi_min is None:
        psi_min = (0, 0)
    assert data.shape[0] % v.size == data.shape[0] % h.size == 0, \
        "The size of v, h must be a factor of the number of data."
    # logger.info(" _ptycho_interface says {}".format("Hello, World!"))
    return (data, data_min,
            probe, theta, v, h,
            psi, psi_min)


def locate_pad(pshape, ushape):
    """Return the min and max indices of the range u in range p."""
    xmin = (pshape - ushape) // 2
    xmax = xmin + ushape
    return xmin, xmax


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
    rs = np.sqrt((r - size/2)**2 + (c - size/2)**2)
    rmax = np.sqrt(2) * 0.5 * rout * rs.max() + 1.0
    rmin = np.sqrt(2) * 0.5 * rin * rs.max()
    img = np.zeros((size, size), dtype='float32')
    img[rs < rmin] = 1.0
    img[rs > rmax] = 0.0
    zone = np.logical_and(rs > rmin, rs < rmax)
    img[zone] = np.divide(rmax - rs[zone], rmax - rmin)
    return img


def fast_pad(unpadded_grid, npadv, npadh):
    """Pad symmetrically with zeros along the last two dimensions.

    The `unpadded_grid` keeps its own data type. The padded_shape is the same
    in all dimensions except for the last two. The unpadded_grid is extracted
    or placed into the an array that is the size of the padded_shape.

    Notes
    -----
    Issue (numpy/numpy #11126, 21 May 2018): Copying into preallocated array
    is faster because of the way that np.pad uses concatenate.

    """
    padded_shape = list(unpadded_grid.shape)
    padded_shape[-2] += 2 * npadv
    padded_shape[-1] += 2 * npadh
    padded_grid = np.zeros(padded_shape, dtype=unpadded_grid.dtype)
    padded_grid[..., npadv:-npadv, npadh:-npadh] = unpadded_grid
    return padded_grid


def combine_grids(grids, v, h,
                  combined_shape, combined_min):
    """Combine some grids by summation.

    The grids are interpolated onto the combined grid using bilinear
    interpolation.

    Parameters
    ----------
    grids : (N, V, H) :py:class:`numpy.array`
        The values on the grids
    v, h : (M, ) :py:class:`numpy.array` float [m]
        The coordinates of the minimum corner of the M grids
    combined_shape : (N+2, ) int
        The last two are numbers of the tuple are the number of indices along
        the h and v directions of the combined grid.
    combined_min : (2, ) float [m]
        The coordinates of the minimum corner of the combined grids

    Return
    ------
    combined : (T, V, H) :py:class:`numpy.array`
        The combined grid

    """
    m_shape, v_shape, h_shape = grids.shape

    # SHARED
    # Find the float coordinates of the grids on the combined grid
    vshift = v - combined_min[-2]
    hshift = h - combined_min[-1]
    assert np.all(vshift >= 0)
    assert np.all(hshift >= 0)
    # Find integer indices (floor) of the min corner of each grid on the
    # combined grid
    V = np.floor(vshift).astype(int)
    V1 = (V + (v_shape + 2)).astype(int)
    assert np.all(V >= 0), "{} is off the combined grid!".format(np.min(V))
    H = np.floor(hshift).astype(int)
    H1 = (H + (h_shape + 2)).astype(int)
    assert np.all(H >= 0), "{} is off the combined grid!".format(np.min(H))
    # Convert to shift less than 1
    vshift -= vshift.astype(int)
    hshift -= hshift.astype(int)
    # END SHARED

    # Create a summed_grids large enough to hold all of the grids
    # plus some padding to cancel out the padding added for shifting
    grids = fast_pad(grids, 1, 1)
    combined = np.zeros([combined_shape[0],
                         combined_shape[1] + 2,
                         combined_shape[2] + 2], dtype=grids.dtype)
    # Add each of the grids to the appropriate place on the summed_grids
    nviews = combined_shape[0]
    nprobes = h.size
    assert grids.shape[0] == nviews * nprobes, ("Wrong number of grids to "
                                                "combine!")
    grids = grids.view(float).reshape(*grids.shape, 2)
    combined = combined.view(float).reshape(*combined.shape, 2)
    for M in range(nviews):
        for N in range(nprobes):
            combined[M,
                     V[N]:V1[N],
                     H[N]:H1[N],
                     ...] += sni.shift(grids[M * nprobes + N],
                                       [vshift[N], hshift[N], 0],
                                       order=1)
    combined = combined.view(complex)
    return combined[:, 1:-1, 1:-1, 0]


def uncombine_grids(grids_shape, v, h,
                    combined, combined_min):
    """Extract a series of grids from a single grid.

    The grids are interpolated onto the combined grid using bilinear
    interpolation.

    Parameters
    ----------
    grids_shape : (N+2, ) int
        The last two are numbers of the tuple are the number of indices along
        the h and v directions of the grids.
    T : (M, ) :py:class:`numpy.array` int
        An index to separate the grids.
    v, h : (M, ) :py:class:`numpy.array` float [m]
        The real coordinates of the minimum corner of the M grids.

    combined : (T, V, H) :py:class:`numpy.array`
        The combined grids.
    combined_min : (2, ) float [m]
        The real coordinates of the minimum corner of the combined grids

    Return
    ------
    grids : (M, V, H) :py:class:`numpy.array`
        The decombined grids

    """
    v_shape, h_shape = grids_shape[-2:]

    # SHARED
    # Find the float coordinates of the grids on the combined grid
    vshift = v - combined_min[-2]
    hshift = h - combined_min[-1]
    assert np.all(vshift >= 0)
    assert np.all(hshift >= 0)
    # Find integer indices (floor) of the min corner of each grid on the
    # combined grid
    V = np.floor(vshift).astype(int)
    V1 = (V + (v_shape + 2)).astype(int)
    assert np.all(V >= 0), "{} is off the combined grid!".format(np.min(V))
    H = np.floor(hshift).astype(int)
    H1 = (H + (h_shape + 2)).astype(int)
    assert np.all(H >= 0), "{} is off the combined grid!".format(np.min(H))
    # Convert to shift less than 1
    vshift -= vshift.astype(int)
    hshift -= hshift.astype(int)
    # END SHARED

    # Create a grids large enough to hold all of the grids
    # plus some padding to cancel out the padding added for shifting
    combined = fast_pad(combined, 1, 1)
    grids = np.empty(grids_shape, dtype=combined.dtype)
    # Retrive the updated values of each of the grids
    nviews = combined.shape[0]
    nprobes = h.size
    assert grids_shape[0] == nviews * nprobes, ("Wrong number of grids to"
                                                " uncombine!")
    grids = grids.view(float).reshape(*grids.shape, 2)
    combined = combined.view(float).reshape(*combined.shape, 2)
    for M in range(nviews):
        for N in range(nprobes):
            grids[M * nprobes + N] = sni.shift(combined[M,
                                                        V[N]:V1[N],
                                                        H[N]:H1[N],
                                                        ...],
                                               [-vshift[N], -hshift[N], 0],
                                               order=1)[1:-1, 1:-1, ...]

    return grids.view(complex)[..., 0]


def grad(data=None, data_min=None,
         probe=None, theta=None, v=None, h=None,
         psi=None, psi_min=None,
         reg=(1+0j), niter=1, rho=0.5, gamma=0.25, lamda=0j, epsilon=1e-8,
         **kwargs):
    """Use gradient descent to estimate `psi`.

    Parameters
    ----------
    reg : (T, V, H, P) :py:class:`numpy.array` complex
        The regularizer for psi.
    rho : float
        The positive penalty parameter. It should be less than 1.
    gamma : float
        The ptychography gradient descent step size.
    lamda : float
        The dual variable. TODO:@Selin Create better description
    epsilon : float
        Primal residual absolute termination criterion.
        TODO:@Selin Create better description

    """
    if not (np.iscomplexobj(psi) and np.iscomplexobj(probe)
            and np.iscomplexobj(reg)):
        raise TypeError("psi, probe, and reg must be complex.")
    # Compute padding between probe and detector size
    npadv = (data.shape[1] - probe.shape[0]) // 2
    npadh = (data.shape[2] - probe.shape[1]) // 2
    # Compute probe inverse
    # TODO: Update the probe too
    probe_inverse = np.conj(probe)
    wavefront_shape = [psi.shape[0] * h.size, probe.shape[0], probe.shape[1]]
    for i in range(niter):
        # combine all wavefronts into one array
        wavefronts = uncombine_grids(grids_shape=wavefront_shape, v=v, h=h,
                                     combined=psi, combined_min=psi_min)
        # Compute near-plane wavefront
        nearplane = probe * wavefronts
        # Pad before FFT
        nearplane_pad = fast_pad(nearplane, npadv, npadh)
        # Go far-plane
        farplane = np.fft.fft2(nearplane_pad)
        # Replace the amplitude with the measured amplitude.
        farplane = (np.sqrt(data) * (farplane.real + 1j * farplane.imag)
                    / np.sqrt(farplane.imag * farplane.imag
                              + farplane.real * farplane.real))
        # Back to near-plane.
        new_nearplane = np.fft.ifft2(farplane)[...,
                                               npadv:npadv+probe.shape[0],
                                               npadh:npadh+probe.shape[1]]
        # Update measurement patch.
        upd_m = probe_inverse * (new_nearplane - nearplane)
        # Combine measurement with other updates
        upd_psi = combine_grids(grids=upd_m, v=v, h=h,
                                combined_shape=psi.shape, combined_min=psi_min)
        # Update psi
        psi = ((1 - gamma * rho) * psi
               + gamma * rho * (reg - lamda / rho)
               + (gamma / 2) * upd_psi)
    return psi


def exitwave(probe, psi, v, h, psi_min=None):
    """Compute the wavefront from probe function and stack of `psi`."""
    wave_shape = [psi.shape[0] * h.size, probe.shape[0], probe.shape[1]]
    wave = uncombine_grids(grids_shape=wave_shape, v=v, h=h,
                           combined=psi, combined_min=psi_min)
    return probe * wave


def simulate(data_shape=None, data_min=None,
             probe=None, theta=None, v=None, h=None,
             psi=None, psi_min=(0, 0),
             **kwargs):
    """Propagate the wavefront to the detector."""
    wavefront = exitwave(probe, psi, v, h, psi_min=psi_min)
    npadx = (data_shape[0] - wavefront.shape[1]) // 2
    npady = (data_shape[1] - wavefront.shape[2]) // 2
    padded_wave = fast_pad(wavefront, npadx, npady)
    intensity = np.square(np.abs(np.fft.fft2(padded_wave)))
    return intensity.astype('float')


def reconstruct(data=None, data_min=None,
                probe=None, theta=None, v=None, h=None,
                psi=None, psi_min=None,
                algorithm=None, niter=1, **kwargs):
    """Reconstruct the `psi` and `probe` using the given `algorithm`.

    Parameters
    ----------
    probe : (V, H, P) :py:class:`numpy.array` float
        The initial guess for the illumnination function of each measurement.
    psi : (T, V, H, P) :py:class:`numpy.array` float
        The inital guess of the object transmission function at each angle.
    algorithm : string
        The name of one of the following algorithms to use for reconstructing:

            * grad : gradient descent

    Returns
    -------
    new_probe : (M, V, H, P) :py:class:`numpy.array` float
        The updated illumination function of each measurement.
    new_psi : (T, V, H, P) :py:class:`numpy.array` float
        The updated obect transmission function at each angle.

    """
    data, data_min, probe, theta, v, h, psi, psi_min = \
        _ptycho_interface(data, data_min,
                          probe, theta, v, h,
                          psi, psi_min, **kwargs)
    # Send data to c function
    logger.info("{} on {:,d} - {:,d} by {:,d} grids for {:,d} "
                "iterations".format(algorithm, *data.shape, niter))
    # Add new algorithms here
    # TODO: The size of this function may be reduced further if all recon clibs
    #   have a standard interface. Perhaps pass unique params to a generic
    #   struct or array.
    if algorithm is "grad":
        new_psi = grad(data=data, data_min=data_min,
                       probe=probe, theta=theta, v=v, h=h,
                       psi=psi, psi_min=psi_min,
                       niter=niter, **kwargs)
    else:
        raise ValueError("The {} algorithm is not an available.".format(
            algorithm))
    return new_psi
