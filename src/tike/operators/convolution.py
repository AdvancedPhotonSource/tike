import itertools

from numba import njit
import numpy as np

from .operator import Operator


class Convolution(Operator):
    """2D Convolution operator with linear interpolation."""

    def __init__(self, probe_shape, nscan, nz, n, ntheta, nmode=1, fly=1, detector_shape=None,
                 **kwargs):  # yapf: disable
        self.nscan = nscan
        self.probe_shape = probe_shape
        self.nz = nz
        self.n = n
        self.ntheta = ntheta
        self.nmode = nmode
        self.fly = fly
        self.detector_shape = probe_shape if detector_shape is None else detector_shape

    def reshape_psi(self, x):
        """Return x reshaped like an object."""
        return x.reshape(self.ntheta, self.nz, self.n)

    def reshape_probe(self, x):
        """Return x reshaped like a probe."""
        x = x.reshape(self.ntheta, -1, self.fly, self.nmode, self.probe_shape,
                      self.probe_shape)
        assert x.shape[1] == 1 or x.shape[1] == self.nscan // self.fly
        return x

    def reshape_nearplane(self, x):
        """Return x reshaped like a nearplane."""
        return x.reshape(self.ntheta, self.nscan // self.fly, self.fly,
                         self.nmode, self.probe_shape, self.probe_shape)

    def reshape_patches(self, x):
        """Return x reshaped like a object patches."""
        return x.reshape(self.ntheta, self.nscan // self.fly, self.fly, 1,
                         self.probe_shape, self.probe_shape)

    def fwd(self, psi, scan, probe):
        """Extract probe shaped patches from the psi at each scan position.

        The patches within the bounds of psi are linearly interpolated, and
        indices outside the bounds of psi are not allowed.
        """
        psi = self.reshape_psi(psi)
        probe = self.reshape_probe(probe)
        patches = np.zeros(
            (self.ntheta, self.nscan, self.probe_shape, self.probe_shape),
            dtype=psi.dtype,
        )

        _patch_iterator(scan, self.probe_shape, psi.shape, _extract_patches,
                        patches, psi)

        return self.reshape_patches(patches) * probe

    def adj(self, nearplane, scan, probe):
        """Combine probe shaped patches into a psi shaped grid by addition."""
        probe = self.reshape_probe(probe)
        nearplane = self.reshape_nearplane(nearplane)
        # If nearplane cannot be reshaped into this shape, then there are not
        # enough scan positions to correctly do this operation.
        nearplane = np.conj(probe) * nearplane
        nearplane = nearplane.reshape(self.ntheta, self.nscan, -1,
                                      self.probe_shape, self.probe_shape)
        nearplane = np.sum(nearplane, axis=2)

        psi = np.zeros((self.ntheta, self.nz, self.n), dtype=nearplane.dtype)

        _patch_iterator(scan, self.probe_shape, psi.shape, _combine_patches,
                        psi, nearplane)

        return psi

    def adj_probe(self, nearplane, scan, psi):
        """Combine probe shaped patches into a psi shaped grid by addition."""
        nearplane = self.reshape_nearplane(nearplane)
        patches = np.zeros(
            (self.ntheta, self.nscan, self.probe_shape, self.probe_shape),
            dtype=psi.dtype,
        )

        _patch_iterator(scan, self.probe_shape, psi.shape, _extract_patches,
                        patches, psi)

        patches = self.reshape_patches(patches)
        return np.conj(patches) * nearplane


@njit(parallel=False, cache=True)
def _combine_patches(psi, nearplane, view_angle, position, i, j, probe_shape,
                     weight):
    """Add patches to psi at given positions."""
    psi[
        view_angle,
        i:i + probe_shape,
        j:j + probe_shape,
    ] += weight * nearplane[view_angle, position]  # yapf: disable
    return psi


@njit(parallel=False, cache=True)
def _extract_patches(patches, psi, view_angle, position, i, j, probe_shape,
                     weight):
    """Extract patches from psi at given positions."""
    patches[view_angle, position] += psi[
        view_angle,
        i:i + probe_shape,
        j:j + probe_shape,
    ] * weight  # yapf: disable
    return patches


@njit(parallel=True)
def _patch_iterator(scan, probe_shape, psi_shape, patch_op, output, input):
    """Apply `patch_op` at all valid scan positions within psi."""
    # For interpolating a pixel to a non-integer position on a grid, we need
    # to divide the area of the pixel between the 4 grid spaces that it
    # overlaps. The weights of each of the adjacent spaces is their area of
    # overlap with the pixel. The side lengths of each of the areas is the
    # remainder from the coordinates of the pixel on the grid.
    for view_angle in range(scan.shape[0]):
        for position in range(scan.shape[1]):
            ind = scan[view_angle, position] // 1
            rem = scan[view_angle, position] % 1
            if (ind[0] < 0 or ind[1] < 0
                    or psi_shape[-2] <= ind[0] + probe_shape
                    or psi_shape[-1] <= ind[1] + probe_shape):
                # skip scans where the probe position overlaps edges
                # print(ind, rem)
                continue
            assert rem[0] >= 0 and rem[1] >= 0
            w = [1 - rem[0], rem[0]]  # lengths of the pixels
            l = [1 - rem[1], rem[1]]
            x = [int(ind[0]), 1 + int(ind[0])]  # coordinates of patch
            y = [int(ind[1]), 1 + int(ind[1])]
            for i in range(2):
                for j in range(2):
                    output = patch_op(output, input, view_angle, position, x[i],
                                      y[j], probe_shape, w[i] * l[j])
