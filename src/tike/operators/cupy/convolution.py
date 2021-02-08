__author__ = "Daniel Ching, Viktor Nikitin"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."

import cupy as cp

from .operator import Operator
from .patch import Patch


class Convolution(Operator):
    """A 2D Convolution operator with linear interpolation.

    Compute the product two arrays at specific relative positions.

    Attributes
    ----------
    nscan : int
        The number of scan positions at each angular view.
    probe_shape : int
        The pixel width and height of the (square) probe illumination.
    nz, n : int
        The pixel width and height of the reconstructed grid.
    ntheta : int
        The number of angular partitions of the data.

    Parameters
    ----------
    psi : (..., nz, n) complex64
        The complex wavefront modulation of the object.
    probe : complex64
        The (..., nscan, probe_shape, probe_shape) (..., 1, probe_shape,
        probe_shape) complex illumination function.
    nearplane: complex64
        The (...., nscan, probe_shape, probe_shape)
        wavefronts after exiting the object.
    scan : (..., nscan, 2) float32
        Coordinates of the minimum corner of the probe grid for each
        measurement in the coordinate system of psi. Vertical coordinates
        first, horizontal coordinates second.

    """
    def __init__(self, probe_shape, nz, n, ntheta,
                 detector_shape=None, **kwargs):  # yapf: disable
        self.probe_shape = probe_shape
        self.nz = nz
        self.n = n
        if detector_shape is None:
            self.detector_shape = probe_shape
        else:
            self.detector_shape = detector_shape
        self.pad = (self.detector_shape - self.probe_shape) // 2
        self.end = self.probe_shape + self.pad
        self.patch = Patch()

    def fwd(self, psi, scan, probe):
        """Extract probe shaped patches from the psi at each scan position.

        The patches within the bounds of psi are linearly interpolated, and
        indices outside the bounds of psi are not allowed.
        """
        assert psi.shape[:-2] == scan.shape[:-2]
        assert probe.shape[:-3] == scan.shape[:-2]
        assert probe.shape[-3] == 1 or probe.shape[-3] == scan.shape[-2]
        patches = self.xp.zeros(
            (*scan.shape[:-1], self.detector_shape, self.detector_shape),
            dtype='complex64',
        )
        patches = self.patch.fwd(patches=patches,
                                 images=psi,
                                 positions=scan,
                                 patch_width=self.probe_shape)
        patches[..., self.pad:self.end, self.pad:self.end] *= probe
        return patches

    def adj(self, nearplane, scan, probe, psi=None, overwrite=False):
        """Combine probe shaped patches into a psi shaped grid by addition."""
        assert probe.shape[:-3] == scan.shape[:-2]
        assert probe.shape[-3] == 1 or probe.shape[-3] == scan.shape[-2]
        assert nearplane.shape[:-2] == scan.shape[:-1]
        if not overwrite:
            nearplane = nearplane.copy()
        nearplane[..., self.pad:self.end, self.pad:self.end] *= probe.conj()
        if psi is None:
            psi = self.xp.zeros((*scan.shape[:-2], self.nz, self.n),
                                dtype='complex64')
        assert psi.shape[:-2] == scan.shape[:-2]
        return self.patch.adj(patches=nearplane,
                              images=psi,
                              positions=scan,
                              patch_width=self.probe_shape)

    def adj_probe(self, nearplane, scan, psi, overwrite=False):
        """Combine probe shaped patches into a probe."""
        assert nearplane.shape[:-2] == scan.shape[:-1], (nearplane.shape,
                                                         scan.shape)
        assert psi.shape[:-2] == scan.shape[:-2], (psi.shape, scan.shape)
        patches = self.xp.zeros(
            (*scan.shape[:-1], self.probe_shape, self.probe_shape),
            dtype='complex64',
        )
        patches = self.patch.fwd(patches=patches,
                                 images=psi,
                                 positions=scan,
                                 patch_width=self.probe_shape)
        patches = patches.conj()
        patches *= nearplane[..., self.pad:self.end, self.pad:self.end]
        return patches
