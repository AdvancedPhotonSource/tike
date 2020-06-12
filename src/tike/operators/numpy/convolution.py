import itertools

import numpy as np

from .operator import Operator


class Convolution(Operator):
    """A 2D Convolution operator with linear interpolation.

    Compute the product two arrays at specific relative positions.

    Attributes
    ----------
    nscan : int
        The number of scan positions at each angular view.
    fly : int
        The number of consecutive scan positions that describe a fly scan.
    probe_shape : int
        The pixel width and height of the (square) probe illumination.
    nz, n : int
        The pixel width and height of the reconstructed grid.
    ntheta : int
        The number of angular partitions of the data.

    Parameters
    ----------
    psi : (ntheta, nz, n) complex64
        The complex wavefront modulation of the object.
    probe : complex64
        The (ntheta, nscan // fly, fly, energy, 1, probe_shape, probe_shape)
        complex illumination function.
    nearplane: complex64
        The (ntheta, nscan // fly, fly, energy, 1, probe_shape, probe_shape)
        wavefronts after exiting the object.
    scan : (ntheta, nscan, 2) float32
        Coordinates of the minimum corner of the probe grid for each
        measurement in the coordinate system of psi. Vertical coordinates
        first, horizontal coordinates second.

    """

    def __init__(self, probe_shape, nz, n, ntheta, energy=1, fly=1,
                 detector_shape=None, **kwargs):  # yapf: disable
        self.probe_shape = probe_shape
        self.nz = nz
        self.n = n
        self.ntheta = ntheta
        self.energy = energy
        self.fly = fly
        if detector_shape is None:
            self.detector_shape = probe_shape
        else:
            self.detector_shape = detector_shape
        self.pad = (self.detector_shape - self.probe_shape) // 2
        self.end = self.probe_shape + self.pad

    def fwd(self, psi, scan, probe):
        """Extract probe shaped patches from the psi at each scan position.

        The patches within the bounds of psi are linearly interpolated, and
        indices outside the bounds of psi are not allowed.
        """
        psi = psi.reshape(self.ntheta, self.nz, self.n)
        self._check_shape_probe(probe, scan.shape[-2])
        patches = self.xp.zeros(
            (self.ntheta, scan.shape[-2], self.detector_shape,
             self.detector_shape),
            dtype='complex64',
        )
        patches = self._patch(patches, psi, scan, fwd=True)
        patches = patches.reshape(self.ntheta, scan.shape[-2] // self.fly,
                                  self.fly, 1, 1, self.detector_shape,
                                  self.detector_shape)
        patches[..., self.pad:self.end, self.pad:self.end] *= probe
        return patches

    def adj(self, nearplane, scan, probe, psi=None, overwrite=False):
        """Combine probe shaped patches into a psi shaped grid by addition."""
        self._check_shape_nearplane(nearplane, scan.shape[-2])
        self._check_shape_probe(probe, scan.shape[-2])
        if not overwrite:
            nearplane = nearplane.copy()
        nearplane[..., self.pad:self.end, self.pad:self.end] *= probe.conj()
        nearplane = nearplane.reshape(self.ntheta, scan.shape[-2],
                                      self.detector_shape, self.detector_shape)
        if psi is None:
            psi = self.xp.zeros((self.ntheta, self.nz, self.n),
                                dtype='complex64')
        return self._patch(nearplane, psi, scan, fwd=False)

    def adj_probe(self, nearplane, scan, psi, overwrite=False):
        """Combine probe shaped patches into a probe."""
        self._check_shape_nearplane(nearplane, scan.shape[-2])
        patches = self.xp.zeros(
            (self.ntheta, scan.shape[-2], self.probe_shape, self.probe_shape),
            dtype='complex64',
        )
        patches = self._patch(patches, psi, scan, fwd=True)
        patches = patches.reshape(self.ntheta, scan.shape[-2] // self.fly,
                                  self.fly, 1, 1, self.probe_shape,
                                  self.probe_shape)
        patches = patches.conj()
        patches *= nearplane[..., self.pad:self.end, self.pad:self.end]
        return patches

    def _check_shape_probe(self, x, nscan):
        """Check that the probe is correctly shaped."""
        assert type(x) is self.xp.ndarray, type(x)
        # unique probe for each position
        shape1 = (self.ntheta, nscan // self.fly, self.fly, 1, 1,
                  self.probe_shape, self.probe_shape)
        # one probe for all positions
        shape2 = (self.ntheta, 1, 1, 1, 1, self.probe_shape, self.probe_shape)
        if __debug__ and x.shape != shape2 and x.shape != shape1:
            raise ValueError(
                f"probe must have shape {shape1} or {shape2} not {x.shape}")

    def _check_shape_nearplane(self, x, nscan):
        """Check that nearplane is correctly shaped."""
        assert type(x) is self.xp.ndarray, type(x)
        shape1 = (self.ntheta, nscan // self.fly, self.fly, 1, 1,
                  self.detector_shape, self.detector_shape)
        if __debug__ and x.shape != shape1:
            raise ValueError(
                f"nearplane must have shape {shape1} not {x.shape}")

    def _patch(self, patches, psi, scan, fwd=True):
            """Reimplement this wrapper to switch the patch getting function."""
            pad = (patches.shape[-1] - self.probe_shape) // 2
            end = self.probe_shape + pad
            if fwd:
                patches[..., pad:end, pad:end] = _patch_iterator(
                    scan,
                    self.probe_shape,
                    psi.shape,
                    _extract_patches,
                    patches[..., pad:end, pad:end],
                    psi,
                )
                return patches
            else:
                return _patch_iterator(
                    scan,
                    self.probe_shape,
                    psi.shape,
                    _combine_patches,
                    psi,
                    patches[..., pad:end, pad:end],
                )


def _combine_patches(psi, patches, view_angle, position, i, j, probe_shape,
                     weight):
    """Add patches to psi at given positions."""
    psi[
        view_angle,
        i:i + probe_shape,
        j:j + probe_shape,
    ] += weight * patches[view_angle, position]  # yapf: disable
    return psi


def _extract_patches(patches, psi, view_angle, position, i, j, probe_shape,
                     weight):
    """Extract patches from psi at given positions."""
    patches[view_angle, position, ...] += psi[
        view_angle,
        i:i + probe_shape,
        j:j + probe_shape,
    ] * weight  # yapf: disable
    return patches


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
                raise ValueError(
                    f"Scan position is out of bounds! {ind[0]}, {ind[1]}")
            assert rem[0] >= 0 and rem[1] >= 0
            w = [1 - rem[0], rem[0]]  # lengths of the pixels
            l = [1 - rem[1], rem[1]]
            x = [int(ind[0]), 1 + int(ind[0])]  # coordinates of patch
            y = [int(ind[1]), 1 + int(ind[1])]
            for i in range(2):
                for j in range(2):
                    output = patch_op(output, input, view_angle, position, x[i],
                                      y[j], probe_shape, w[i] * l[j])
    return output
