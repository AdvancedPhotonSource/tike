"""This module defines a pytchography core based on the NumPy FFT module."""

import numpy as np

from .core import PtychoCore
from ._shift import _combine_grids, _uncombine_grids


class PtychoNumPyFFT(PtychoCore):
    """Implement the ptychography operators using the NumPy FFT library."""

    array_module = np

    def fwd(self, probe, scan, psi, **kwargs):
        if not (np.iscomplexobj(psi) and np.iscomplexobj(probe)):
            raise TypeError("psi and probe must be complex.")
        probe = probe.astype(np.complex64)
        psi = psi.astype(np.complex64)
        # Grab all of the patches where the probe and psi interact
        wavefront = np.empty(
            (self.ntheta, self.nscan, self.probe_shape, self.probe_shape),
            dtype=np.complex64)
        for i in range(self.ntheta):
            # Multiply the probe and patches of psi
            wavefront[i] = _uncombine_grids(
                grids_shape=(self.nscan, self.probe_shape, self.probe_shape),
                v=np.ravel(scan[i, :, 0]),
                h=np.ravel(scan[i, :, 1]),
                combined=psi[i],
            ) * probe[i]
        # Propagate the wavefronts to the detector
        farplane = np.fft.fft2(
            wavefront,
            s=(self.detector_shape, self.detector_shape),
            norm='ortho',
        )
        assert farplane.shape == (self.ntheta, self.nscan, self.detector_shape,
                                  self.detector_shape)
        return farplane

    def adj(self, farplane, probe, scan, **kwargs):
        nearplane = np.fft.ifft2(
            farplane, norm='ortho',
        )[..., :self.probe_shape, :self.probe_shape]
        psi = np.empty((self.ntheta, self.nz, self.n), dtype=np.complex64)
        for i in range(self.ntheta):
            psi[i] = _combine_grids(
                grids=nearplane[i] * np.conj(probe[i]),
                v=np.ravel(scan[i, :, 0]),
                h=np.ravel(scan[i, :, 1]),
                combined_shape=(self.nz, self.n),
            )
        assert psi.shape == (self.ntheta, self.nz, self.n)
        return psi

    def adj_probe(self, farplane, scan, psi, **kwargs):
        psi_patches = np.empty(
            (self.ntheta, self.nscan, self.probe_shape, self.probe_shape),
            dtype=np.complex64)
        for i in range(self.ntheta):
            psi_patches[i] = _uncombine_grids(
                grids_shape=(self.nscan, self.probe_shape, self.probe_shape),
                v=np.ravel(scan[i, :, 0]),
                h=np.ravel(scan[i, :, 1]),
                combined=psi[i],
            )
        nearplane = np.fft.ifft2(
            farplane,
            norm='ortho',
        )[..., :self.probe_shape, :self.probe_shape]
        probe = np.sum(nearplane * np.conj(psi_patches), axis=1)
        assert probe.shape == (self.ntheta, self.probe_shape, self.probe_shape)
        return probe
