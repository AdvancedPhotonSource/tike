"""This module defines a pytchography core based on the NumPy FFT module."""

import itertools

import numpy as np

from .core import PtychoCore


class PtychoNumPyFFT(PtychoCore):
    """Implement `tike.ptycho.core.PtychoCore` using the NumPy FFT library."""

    array_module = np
    asnumpy = np.asarray

    def fwd(self, probe, scan, psi, **kwargs):  # noqa: D102
        if not (np.iscomplexobj(psi) and np.iscomplexobj(probe)):
            raise TypeError("psi and probe must be complex.")
        probe = probe.astype(np.complex64)
        psi = psi.astype(np.complex64)
        nearplane = self.diffraction_fwd(
            psi=psi,
            scan=scan,
        ) * probe[:, np.newaxis]
        farplane = self.propagation_fwd(nearplane)
        assert farplane.shape == (self.ntheta, self.nscan, self.detector_shape,
                                  self.detector_shape)
        return farplane

    def adj(self, farplane, probe, scan, **kwargs):  # noqa: D102
        nearplane = self.propagation_adj(farplane)
        psi = self.diffraction_adj(
            nearplane=nearplane * np.conj(probe[:, np.newaxis]),
            scan=scan
        )
        assert psi.shape == (self.ntheta, self.nz, self.n)
        return psi

    def adj_probe(self, farplane, scan, psi, **kwargs):  # noqa: D102
        psi_patches = self.diffraction_fwd(psi=psi, scan=scan)
        nearplane = self.propagation_adj(farplane=farplane)
        probe = np.sum(nearplane * np.conj(psi_patches), axis=1)
        assert probe.shape == (self.ntheta, self.probe_shape, self.probe_shape)
        return probe

    def propagation_fwd(self, nearplane):
        pad = (self.detector_shape - self.probe_shape) // 2
        end = self.probe_shape + pad
        padded_nearplane = np.zeros(
            (self.ntheta, self.nscan, self.detector_shape, self.detector_shape),
            dtype=np.complex64,
        )
        padded_nearplane[..., pad:end, pad:end] = nearplane
        return np.fft.fft2(
            padded_nearplane,
            s=(self.detector_shape, self.detector_shape),
            norm='ortho',
        ).astype(np.complex64)

    def propagation_adj(self, farplane):
        pad = (self.detector_shape - self.probe_shape) // 2
        end = self.probe_shape + pad
        return np.fft.ifft2(
            farplane, norm='ortho',
        )[:, :, pad:end, pad:end].astype(np.complex64)

    def diffraction_fwd(self, psi, scan):
        """Extract probe shaped patches from the psi at each scan position.

        The patches within the bounds of psi are linearly interpolated, and
        indices outside the bounds of psi are not allowed.
        """
        # For interpolating a pixel to a non-integer position on a grid, we need
        # to divide the area of the pixel between the 4 grid spaces that it
        # overlaps. The weights of each of the adjacent spaces is their area of
        # overlap with the pixel. The side lengths of each of the areas is the
        # remainder from the coordinates of the pixel on the grid.
        nearplane = np.zeros_like(
            psi,
            shape=(self.ntheta, self.nscan, self.probe_shape, self.probe_shape),
        )
        for view_angle in range(self.ntheta):
            for position in range(self.nscan):
                rem, ind = np.modf(scan[view_angle, position])
                w = (1 - rem[0], rem[0])
                l = (1 - rem[1], rem[1])
                areas = (w * l for w, l in itertools.product(w, l))
                x = (int(ind[0]), 1 + int(ind[0]))
                y = (int(ind[1]), 1 + int(ind[1]))
                corners = ((x, y) for x, y in itertools.product(x, y))
                for weight, (i, j) in zip(areas, corners):
                    if weight > 0:
                        nearplane[view_angle, position, ...] += psi[
                            view_angle,
                            i:i + self.probe_shape,
                            j:j + self.probe_shape,
                        ] * weight
        return nearplane

    def diffraction_adj(self, nearplane, scan):
        """Combine probe shaped patches into a psi shaped grid by addition."""
        # See diffraction_fwd for description of interpolation algorithm.
        psi = np.zeros_like(nearplane, shape=(self.ntheta, self.nz, self.n))
        for view_angle in range(self.ntheta):
            for position in range(self.nscan):
                rem, ind = np.modf(scan[view_angle, position])
                w = (1 - rem[0], rem[0])
                l = (1 - rem[1], rem[1])
                areas = (w * l for w, l in itertools.product(w, l))
                x = (int(ind[0]), 1 + int(ind[0]))
                y = (int(ind[1]), 1 + int(ind[1]))
                corners = ((x, y) for x, y in itertools.product(x, y))
                for weight, (i, j) in zip(areas, corners):
                    if weight > 0:
                        psi[
                            view_angle,
                            i:i + self.probe_shape,
                            j:j + self.probe_shape,
                        ] += weight * nearplane[view_angle, position]
        return psi
