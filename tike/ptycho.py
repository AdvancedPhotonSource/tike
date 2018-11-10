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
psi : (T, V, H) :py:class:`numpy.array` complex
    The object transmission function for each of the `T` views.
foo_corner : (2, ) float [p]
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


def _ptycho_interface(
    data,
    probe, v, h,
    psi, psi_corner, **kwargs
):
    """Define an interface that all functions in this module match.

    This function also sets default values for functions in this module.
    """
    if data is None:
        raise ValueError()
    if probe is None:
        raise ValueError()
    if v is None:
        raise ValueError()
    if h is None:
        raise ValueError()
    if psi is None:
        raise ValueError()
    psi_corner = (0, 0) if psi_corner is None else psi_corner
    assert len(data) == v.size == h.size, \
        "The size of v, h must be the same as the number of data."
    return (data,
            probe, v, h,
            psi, psi_corner)


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
    if npadv < 0 or npadh < 0:
        raise ValueError("Pads must be non-negative.")
    if npadv > 0 or npadv > 0:
        padded_shape = list(unpadded_grid.shape)
        padded_shape[-2] += 2 * npadv
        padded_shape[-1] += 2 * npadh
        padded_grid = np.zeros(padded_shape, dtype=unpadded_grid.dtype)
        padded_grid[...,
                    npadv:padded_shape[-2] - npadv,
                    npadh:padded_shape[-1] - npadh] = unpadded_grid
        return padded_grid
    return unpadded_grid


def shift_coords(r_min, r_shape, combined_min, combined_shape):
    """Find the positions of some 1D ranges in a new 1D coordinate system.

    Pad the new range coordinates with one on each side.

    Parameters
    ----------
    r_min, r_shape : :py:class:`numpy.array` float, int
        1D min and range to be transformed.
    combined_min, combined_shape : :py:class:`numpy.array` float, int
        The min and range of the new coordinate system.

    Return
    ------
    r_shift : :py:class:`numpy.array` float
        The shifted coordinate remainder.
    r_lo, r_hi : :py:class:`numpy.array` int
        New range integer starts and ends.

    """
    # Find the float coordinates of each range on the combined grid
    r_shift = (r_min - combined_min).flatten()
    # Find integer indices (floor) of each range on the combined grid
    r_lo = np.floor(r_shift).astype(int)
    r_hi = (r_lo + (r_shape + 2)).astype(int)
    if np.any(r_lo < 0) or np.any(r_hi > combined_min + combined_shape + 2):
        raise ValueError("Index {} or {} is off the grid!".format(
                         np.min(r_lo), np.max(r_hi)))
    # Find the remainder shift less than 1
    r_shift -= r_shift.astype(int)
    return r_shift, r_lo, r_hi


def combine_grids(
    grids, v, h,
    combined_shape, combined_corner
):
    """Combine grids by summation.

    Multiple grids are interpolated onto a single combined grid using
    bilinear interpolation.

    Parameters
    ----------
    grids : (N, V, H) :py:class:`numpy.array`
        The values on the grids
    v, h : (M, ) :py:class:`numpy.array` float [m]
        The coordinates of the minimum corner of the M grids
    combined_shape : (2, ) int
        The last two are numbers of the tuple are the number of indices along
        the h and v directions of the combined grid.
    combined_corner : (2, ) float [m]
        The coordinates of the minimum corner of the combined grids

    Return
    ------
    combined : (T, V, H) :py:class:`numpy.array`
        The combined grid

    """
    m_shape, v_shape, h_shape = grids.shape
    vshift, V, V1 = shift_coords(v, v_shape,
                                 combined_corner[-2], combined_shape[-2])
    hshift, H, H1 = shift_coords(h, h_shape,
                                 combined_corner[-1], combined_shape[-1])
    # Create a summed_grids large enough to hold all of the grids
    # plus some padding to cancel out the padding added for shifting
    grids = fast_pad(grids, 1, 1)
    combined = np.zeros([combined_shape[-2] + 2, combined_shape[-1] + 2],
                        dtype=grids.dtype)
    # Add each of the grids to the appropriate place on the summed_grids
    nprobes = h.size
    grids = grids.view(float).reshape(*grids.shape, 2)
    combined = combined.view(float).reshape(*combined.shape, 2)
    for N in range(nprobes):
        combined[V[N]:V1[N], H[N]:H1[N],
                 ...] += sni.shift(grids[N],
                                   [vshift[N], hshift[N], 0],
                                   order=1)
    combined = combined.view(complex)
    return combined[1:-1, 1:-1, 0]


def uncombine_grids(
    grids_shape, v, h,
    combined, combined_corner
):
    """Extract a series of grids from a single grid.

    The grids are interpolated onto the combined grid using bilinear
    interpolation.

    Parameters
    ----------
    grids_shape : (2, ) int
        The last two are numbers of the tuple are the number of indices along
        the h and v directions of the grids.
    v, h : (M, ) :py:class:`numpy.array` float [m]
        The real coordinates of the minimum corner of the M grids.

    combined : (V, H) :py:class:`numpy.array`
        The combined grids.
    combined_corner : (2, ) float [m]
        The real coordinates of the minimum corner of the combined grids

    Return
    ------
    grids : (M, V, H) :py:class:`numpy.array`
        The decombined grids

    """
    v_shape, h_shape = grids_shape[-2:]
    vshift, V, V1 = shift_coords(v, v_shape,
                                 combined_corner[-2], combined.shape[-2])
    hshift, H, H1 = shift_coords(h, h_shape,
                                 combined_corner[-1], combined.shape[-1])
    # Create a grids large enough to hold all of the grids
    # plus some padding to cancel out the padding added for shifting
    combined = fast_pad(combined, 1, 1)
    grids = np.empty(grids_shape, dtype=combined.dtype)
    # Retrive the updated values of each of the grids
    nprobes = h.size
    grids = grids.view(float).reshape(*grids.shape, 2)
    combined = combined.view(float).reshape(*combined.shape, 2)
    for N in range(nprobes):
        grids[N] = sni.shift(combined[V[N]:V1[N], H[N]:H1[N], ...],
                             [-vshift[N], -hshift[N], 0],
                             order=1,
                             )[1:-1, 1:-1, ...]
    return grids.view(complex)[..., 0]


def grad(
    data,
    probe, v, h,
    psi, psi_corner,
    reg=(1+0j), niter=1, rho=0, gamma=0.25, lamda=0j, epsilon=1e-8,
    **kwargs
):
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
    wavefront_shape = [h.size, probe.shape[0], probe.shape[1]]
    for i in range(niter):
        # combine all wavefronts into one array
        wavefronts = uncombine_grids(grids_shape=wavefront_shape, v=v, h=h,
                                     combined=psi, combined_corner=psi_corner)
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
                                combined_shape=psi.shape,
                                combined_corner=psi_corner)
        # Update psi
        psi = ((1 - gamma * rho) * psi
               + gamma * (reg * rho - lamda)  # refactored to remove division
               + (gamma * 0.5) * upd_psi)
    return psi


def exitwave(probe, v, h, psi, psi_corner=None):
    """Compute the wavefront from probe function and stack of `psi`."""
    wave_shape = [h.size, probe.shape[0], probe.shape[1]]
    wave = uncombine_grids(grids_shape=wave_shape, v=v, h=h,
                           combined=psi, combined_corner=psi_corner)
    return probe * wave


def simulate(
    data_shape,
    probe, v, h,
    psi, psi_corner=(0, 0),
    **kwargs
):
    """Propagate the wavefront to the detector."""
    wavefront = exitwave(probe, v, h,
                         psi, psi_corner=psi_corner)
    npadx = (data_shape[0] - wavefront.shape[-2]) // 2
    npady = (data_shape[1] - wavefront.shape[-1]) // 2
    padded_wave = fast_pad(wavefront, npadx, npady)
    return np.square(np.abs(np.fft.fft2(padded_wave)))


def reconstruct(
    data,
    probe=None, v=None, h=None,
    psi=None, psi_corner=None,
    algorithm=None, niter=1, **kwargs
):
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
    data, probe, v, h, psi, psi_corner = \
        _ptycho_interface(data,
                          probe, v, h,
                          psi, psi_corner, **kwargs)
    # Send data to c function
    logger.info("{} on {:,d} - {:,d} by {:,d} grids for {:,d} "
                "iterations".format(algorithm,
                                    len(data), *data.shape[1:], niter))
    # Add new algorithms here
    # TODO: The size of this function may be reduced further if all recon clibs
    #   have a standard interface. Perhaps pass unique params to a generic
    #   struct or array.
    if algorithm is "grad":
        new_psi = grad(data=data,
                       probe=probe, v=v, h=h,
                       psi=psi, psi_corner=psi_corner,
                       niter=niter, **kwargs
                       )
    else:
        raise ValueError("The {} algorithm is not an available.".format(
            algorithm))
    return new_psi
