__author__ = "Daniel Ching, Viktor Nikitin"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."

import cupy as cp

from .operator import Operator
from .patch import Patch
from .shift import Shift


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
        The (..., nscan, nprobe, probe_shape, probe_shape) or
        (..., 1, nprobe, probe_shape, probe_shape) complex illumination
        function.
    nearplane: complex64
        The (...., nscan, nprobe, probe_shape, probe_shape)
        wavefronts after exiting the object.
    scan : (..., nscan, 2) float32
        Coordinates of the minimum corner of the probe grid for each
        measurement in the coordinate system of psi. Vertical coordinates
        first, horizontal coordinates second.

    """

    def __init__(self, probe_shape, nz, n, ntheta=None,
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
        assert psi.shape[:-2] == scan.shape[:-2], (psi.shape, scan.shape)
        assert probe.shape[:-4] == scan.shape[:-2], (probe.shape, scan.shape)
        assert probe.shape[-4] == 1 or probe.shape[-4] == scan.shape[-2]
        if self.detector_shape == self.probe_shape:
            patches = self.xp.empty_like(
                psi,
                shape=(
                    *scan.shape[:-2],
                    scan.shape[-2] * probe.shape[-3],
                    self.detector_shape,
                    self.detector_shape,
                ),
            )
        else:
            patches = self.xp.zeros_like(
                psi,
                shape=(
                    *scan.shape[:-2],
                    scan.shape[-2] * probe.shape[-3],
                    self.detector_shape,
                    self.detector_shape,
                ),
            )
        patches = self.patch.fwd(
            patches=patches,
            images=psi,
            positions=scan,
            patch_width=self.probe_shape,
            nrepeat=probe.shape[-3],
        )
        patches = patches.reshape((
            *scan.shape[:-1],
            probe.shape[-3],
            self.detector_shape,
            self.detector_shape,
        ))
        patches[..., self.pad:self.end, self.pad:self.end] *= probe
        return patches

    def adj(self, nearplane, scan, probe, psi=None, overwrite=False):
        """Combine probe shaped patches into a psi shaped grid by addition."""
        assert probe.shape[:-4] == scan.shape[:-2], (probe.shape, scan.shape)
        assert probe.shape[-4] == 1 or probe.shape[-4] == scan.shape[-2]
        assert nearplane.shape[:-3] == scan.shape[:-1], (nearplane.shape, scan.shape)
        if not overwrite:
            nearplane = nearplane.copy()
        nearplane[..., self.pad:self.end, self.pad:self.end] *= probe.conj()
        if psi is None:
            psi = self.xp.zeros_like(
                nearplane,
                shape=(*scan.shape[:-2], self.nz, self.n),
            )
        assert psi.shape[:-2] == scan.shape[:-2]
        return self.patch.adj(
            patches=nearplane.reshape((
                *scan.shape[:-2],
                scan.shape[-2] * nearplane.shape[-3],
                *nearplane.shape[-2:],
            )),
            images=psi,
            positions=scan,
            patch_width=self.probe_shape,
            nrepeat=nearplane.shape[-3],
        )

    def adj_probe(self, nearplane, scan, psi, overwrite=False):
        """Combine probe shaped patches into a probe."""
        assert nearplane.shape[:-3] == scan.shape[:-1], (nearplane.shape,
                                                         scan.shape)
        assert psi.shape[:-2] == scan.shape[:-2], (psi.shape, scan.shape)
        patches = self.xp.zeros_like(
            psi,
            shape=(
                *scan.shape[:-2],
                scan.shape[-2] * nearplane.shape[-3],
                self.probe_shape,
                self.probe_shape,
            ),
        )
        patches = self.patch.fwd(
            patches=patches,
            images=psi,
            positions=scan,
            patch_width=self.probe_shape,
            nrepeat=nearplane.shape[-3],
        )
        patches = patches.reshape((*scan.shape[:-1], nearplane.shape[-3],
                                   self.probe_shape, self.probe_shape))
        patches = patches.conj()
        patches *= nearplane[..., self.pad:self.end, self.pad:self.end]
        return patches

class ConvolutionFFT(Operator):
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
        The (..., nscan, nprobe, probe_shape, probe_shape) or
        (..., 1, nprobe, probe_shape, probe_shape) complex illumination
        function.
    nearplane: complex64
        The (...., nscan, nprobe, probe_shape, probe_shape)
        wavefronts after exiting the object.
    scan : (..., nscan, 2) float32
        Coordinates of the minimum corner of the probe grid for each
        measurement in the coordinate system of psi. Vertical coordinates
        first, horizontal coordinates second.

    """

    def __init__(self, probe_shape, nz, n, ntheta=None,
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
        self.shift = Shift()

    def __enter__(self):
        self.shift.__enter__()
        return self

    def __exit__(self, type, value, traceback):
        self.shift.__exit__(type, value, traceback)

    def fwd(self, psi, scan, probe):
        """Extract probe shaped patches from the psi at each scan position.

        The patches within the bounds of psi are linearly interpolated, and
        indices outside the bounds of psi are not allowed.
        """
        assert psi.shape[:-2] == scan.shape[:-2], (psi.shape, scan.shape)
        assert probe.shape[:-4] == scan.shape[:-2], (probe.shape, scan.shape)
        assert probe.shape[-4] == 1 or probe.shape[-4] == scan.shape[-2]
        if self.detector_shape == self.probe_shape:
            patches = self.xp.empty_like(
                psi,
                shape=(
                    *scan.shape[:-2],
                    scan.shape[-2] * probe.shape[-3],
                    self.detector_shape,
                    self.detector_shape,
                ),
            )
        else:
            patches = self.xp.zeros_like(
                psi,
                shape=(
                    *scan.shape[:-2],
                    scan.shape[-2] * probe.shape[-3],
                    self.detector_shape,
                    self.detector_shape,
                ),
            )
        index, shift = self.xp.divmod(scan, 1.0)
        shift = shift.reshape((*scan.shape[:-1], 1, 2))

        patches = self.patch.fwd(
            patches=patches,
            images=psi,
            positions=index,
            patch_width=self.probe_shape,
            nrepeat=probe.shape[-3],
        )

        patches = patches.reshape((
            *scan.shape[:-1],
            probe.shape[-3],
            self.detector_shape,
            self.detector_shape,
        ))
        patches = self.shift.adj(patches, shift, overwrite=False)

        patches[..., self.pad:self.end, self.pad:self.end] *= probe
        return patches

    def adj(self, nearplane, scan, probe, psi=None, overwrite=False):
        """Combine probe shaped patches into a psi shaped grid by addition."""
        assert probe.shape[:-4] == scan.shape[:-2]
        assert probe.shape[-4] == 1 or probe.shape[-4] == scan.shape[-2]
        assert nearplane.shape[:-3] == scan.shape[:-1]
        if not overwrite:
            nearplane = nearplane.copy()
        nearplane[..., self.pad:self.end, self.pad:self.end] *= probe.conj()

        index, shift = self.xp.divmod(scan, 1.0)
        shift = shift.reshape((*scan.shape[:-1], 1, 2))

        nearplane = self.shift.fwd(nearplane, shift, overwrite=True)

        if psi is None:
            psi = self.xp.zeros_like(
                nearplane,
                shape=(*scan.shape[:-2], self.nz, self.n),
            )
        assert psi.shape[:-2] == scan.shape[:-2]
        return self.patch.adj(
            patches=nearplane.reshape((
                *scan.shape[:-2],
                scan.shape[-2] * nearplane.shape[-3],
                *nearplane.shape[-2:],
            )),
            images=psi,
            positions=index,
            patch_width=self.probe_shape,
            nrepeat=nearplane.shape[-3],
        )
