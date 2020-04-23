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
"""Define functions for plotting and viewing data of various types."""

__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2018, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = [
    'plot_complex',
    'plot_wave',
    'trajectory',
    'plot_footprint',
    'plot_trajectories',
    'plot_sino_coverage',
]

import logging
import warnings

from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_complex(Z, rmin=None, rmax=None, imin=None, imax=None):
    """Plot real and imaginary parts of a 2D image size by side.

    Takes parameters rmin, rmax, imin, imax to scale the ranges of the real
    and imaginary plots.
    """
    plt.subplot(1, 2, 1)
    plt.imshow(Z.real, vmin=rmin, vmax=rmax)
    cb0 = plt.colorbar(orientation='horizontal')
    plt.subplot(1, 2, 2)
    plt.imshow(Z.imag, vmin=imin, vmax=imax)
    cb1 = plt.colorbar(orientation='horizontal')


def plot_wave(Z, amin=None, amax=None):
    """Plot the amplitude and phase of a 2D image side by side.

    Takes parameters amin, amax to scale the range of the amplitude. The phase
    is scaled to the range -pi to pi.
    """
    amplitude, phase = np.abs(Z), np.angle(Z)
    if np.any(amplitude == 0):
        warnings.warn(
            "This phase plot will be incorrect because "
            "the phase of a zero-amplitude complex number is undefined. "
            "Adding a small constant to the amplitude may help.")
    plt.subplot(1, 2, 1)
    plt.imshow(amplitude, vmin=amin, vmax=amax)
    cb0 = plt.colorbar(orientation='horizontal')
    plt.subplot(1, 2, 2)
    plt.imshow(phase, vmin=-np.pi, vmax=np.pi, cmap=plt.cm.twilight)
    cb1 = plt.colorbar(orientation='horizontal')
    print(np.min(Z), np.max(Z))


def trajectory(x, y, connect=True, frame=None, pause=True, dt=1e-12):
    """Plot a 2D trajectory."""
    if frame is None:
        frame = [np.min(x), np.max(x), np.min(y), np.max(y)]
    ax = fig.add_subplot(111)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid('on')
    ax.axis('image')
    ax.set_xlim(frame[0], frame[1])
    ax.set_ylim(frame[2], frame[3])
    if connect is True:
        line, = ax.plot([], [], '-or')
    else:
        line, = ax.plot([], [], 'or')
    for m in range(x.size):
        line.set_xdata(np.append(line.get_xdata(), x[m]))
        line.set_ydata(np.append(line.get_ydata(), y[m]))
        plt.draw()
        if pause is True:
            plt.pause(dt)


def plot_footprint(theta, v, h):
    """Plot 2D projections of the trajectory for each pair of axes."""
    theta = theta % (np.pi) / np.pi

    ax1a = plt.subplot(1, 3, 2)
    plt.plot(h, v, color='blue', linewidth=1)
    ax1a.axis('equal')
    plt.xlabel("h")
    plt.ylabel("v")

    ax1b = plt.subplot(1, 3, 3)
    plt.scatter(theta, v, color='red', s=1)
    ax1b.axis('equal')
    plt.xticks([0, 1], [0, r'$\pi$'])
    plt.xlabel('theta')
    plt.ylabel("v")

    ax1c = plt.subplot(1, 3, 1)
    plt.scatter(h, theta, color='green', s=1)
    ax1c.axis('equal')
    plt.yticks([0, 1], [0, r'$\pi$'])
    plt.ylabel('theta')
    plt.xlabel("h")


def plot_sino_coverage(
        theta, v, h, dwell=None, bins=[16, 8, 4],
        probe_grid=[[1]], probe_shape=(0, 0)
):  # yapf: disable
    """Plot projections of minimum coverage in the sinogram space."""
    # Wrap theta into [0, pi)
    theta = theta % (np.pi)
    # Set default dwell value
    if dwell is None:
        dwell = np.ones(theta.shape)
    # Make sure probe_grid is array
    probe_grid = np.asarray(probe_grid)
    # Create one ray for each pixel in the probe grid
    dv, dh = np.meshgrid(
        np.linspace(0, probe_shape[0], probe_grid.shape[0],
                    endpoint=False) + probe_shape[0] / probe_grid.shape[0] / 2,
        np.linspace(0, probe_shape[1], probe_grid.shape[1], endpoint=False) +
        probe_shape[1] / probe_grid.shape[1] / 2,
    )

    dv = dv.flatten()
    dh = dh.flatten()
    probe_grid = probe_grid.flatten()
    H = np.zeros(bins)
    for i in range(probe_grid.size):
        if probe_grid[i] > 0:
            # Compute histogram
            sample = np.stack([theta, v + dv[i]], h + dh[i], axis=1)
            dH, edges = np.histogramdd(
                sample,
                bins=bins,
                range=[[0, np.pi], [-.5, .5], [-.5, .5]],
                weights=dwell * probe_grid[i])
            H += dH
    ideal_bin_count = np.sum(dwell) * np.sum(probe_grid) / np.prod(bins)
    H /= ideal_bin_count
    # Plot
    ax1a = plt.subplot(1, 3, 2)
    plt.imshow(
        np.min(H, axis=0).T, vmin=0, vmax=2, origin="lower", cmap=plt.cm.RdBu)
    ax1a.axis('equal')
    plt.xticks(np.array([0, bins[1] / 2, bins[1]]) - 0.5, [-.5, 0, .5])
    plt.yticks(np.array([0, bins[2] / 2, bins[2]]) - 0.5, [-.5, 0, .5])
    plt.xlabel("h")
    plt.ylabel("v")

    ax1b = plt.subplot(1, 3, 3)
    plt.imshow(
        np.min(H, axis=1).T, vmin=0, vmax=2, origin="lower", cmap=plt.cm.RdBu)
    ax1b.axis('equal')
    plt.xlabel('theta')
    plt.ylabel("v")
    plt.xticks(np.array([0, bins[0]]) - 0.5, [0, r'$\pi$'])
    plt.yticks(np.array([0, bins[2] / 2, bins[2]]) - 0.5, [-.5, 0, .5])

    ax1c = plt.subplot(1, 3, 1)
    plt.imshow(
        np.min(H, axis=2), vmin=0, vmax=2, origin="lower", cmap=plt.cm.RdBu)
    ax1c.axis('equal')
    plt.ylabel('theta')
    plt.xlabel("h")
    plt.yticks(np.array([0, bins[0]]) - 0.5, [0, r'$\pi$'])
    plt.xticks(np.array([0, bins[1] / 2, bins[1]]) - 0.5, [-.5, 0, .5])

    return H


def plot_trajectories(theta, v, h, t):
    """Plot each trajectory as a function of time in the current figure.

    Plots two subplots in the current figure. The top one shows horizonal
    and vertical position as a function of time and the bottom shows angular
    position as a function of time.

    Returns
    -------
    ax1, ax1b : axes
        Handles to the two axes

    """
    ax1a = plt.subplot(2, 1, 1)
    plt.plot(t, h, 'c--', t, v, 'm.')
    plt.ylabel('position [cm]')
    plt.legend(['h', 'v'])
    plt.ylim([-.5, .5])
    plt.setp(ax1a.get_xticklabels(), visible=False)
    ax1b = plt.subplot(2, 1, 2, sharex=ax1a)
    plt.plot(t, theta % (2 * np.pi) / (2 * np.pi), 'yellow')
    plt.ylabel(r'theta [$2\pi$]')
    plt.xlabel('time [s]')
    plt.ylim([0, 1.])
    return ax1a, ax1b
