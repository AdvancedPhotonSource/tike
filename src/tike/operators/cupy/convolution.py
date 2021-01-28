__author__ = "Daniel Ching, Viktor Nikitin"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."

from importlib_resources import files

import cupy as cp

from .operator import Operator

_cu_source = files('tike.operators.cupy').joinpath('convolution.cu').read_text()
_fwd_patch = cp.RawKernel(_cu_source, "fwd_patch")
_adj_patch = cp.RawKernel(_cu_source, "adj_patch")


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
    psi : (ntheta, nz, n) complex64
        The complex wavefront modulation of the object.
    probe : complex64
        The (ntheta, nscan // 1, 1, 1, probe_shape, probe_shape)
        complex illumination function.
    nearplane: complex64
        The (ntheta, nscan // 1, 1, 1, probe_shape, probe_shape)
        wavefronts after exiting the object.
    scan : (ntheta, nscan, 2) float32
        Coordinates of the minimum corner of the probe grid for each
        measurement in the coordinate system of psi. Vertical coordinates
        first, horizontal coordinates second.

    """
    def __init__(self, probe_shape, nz, n, ntheta,
                 detector_shape=None, **kwargs):  # yapf: disable
        self.probe_shape = probe_shape
        self.nz = nz
        self.n = n
        self.ntheta = ntheta
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
        patches = patches.reshape(self.ntheta, scan.shape[-2], 1, 1,
                                  self.detector_shape, self.detector_shape)
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
        patches = patches.reshape(self.ntheta, scan.shape[-2], 1, 1,
                                  self.probe_shape, self.probe_shape)
        patches = patches.conj()
        patches *= nearplane[..., self.pad:self.end, self.pad:self.end]
        return patches

    def _check_shape_probe(self, x, nscan):
        """Check that the probe is correctly shaped."""
        assert type(x) is self.xp.ndarray, type(x)
        # unique probe for each position
        shape1 = (self.ntheta, nscan, 1, 1, self.probe_shape, self.probe_shape)
        # one probe for all positions
        shape2 = (self.ntheta, 1, 1, 1, self.probe_shape, self.probe_shape)
        if __debug__ and x.shape != shape2 and x.shape != shape1:
            raise ValueError(
                f"probe must have shape {shape1} or {shape2} not {x.shape}")

    def _check_shape_nearplane(self, x, nscan):
        """Check that nearplane is correctly shaped."""
        assert type(x) is self.xp.ndarray, type(x)
        shape1 = (self.ntheta, nscan, 1, 1, self.detector_shape,
                  self.detector_shape)
        if __debug__ and x.shape != shape1:
            raise ValueError(
                f"nearplane must have shape {shape1} not {x.shape}")

    def _patch(self, patches, psi, scan, fwd=True):
        """Extract or combine patches.

        When fwd is True, the regions are copied from psi into patches.
        When fwd is False, patches are added to psi.
        """
        max_thread = min(_next_power_two(self.probe_shape),
                         _fwd_patch.attributes['max_threads_per_block'])
        grids = (
            scan.shape[-2],
            self.ntheta,
            self.probe_shape,
        )
        blocks = (max_thread,)
        if fwd:
            _fwd_patch(
                grids,
                blocks,
                (psi, patches, scan, self.ntheta, self.nz, self.n,
                 scan.shape[-2], self.probe_shape, patches.shape[-1]),
            )
            return patches
        else:
            _adj_patch(
                grids,
                blocks,
                (psi, patches, scan, self.ntheta, self.nz, self.n,
                 scan.shape[-2], self.probe_shape, patches.shape[-1]),
            )
            return psi


def _next_power_two(v):
    """Return the next highest power of 2 of 32-bit v.

    https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
    """
    v -= 1
    v |= v >> 1
    v |= v >> 2
    v |= v >> 4
    v |= v >> 8
    v |= v >> 16
    return v + 1
