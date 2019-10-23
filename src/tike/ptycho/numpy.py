"""This module defines a pytchography core based on the NumPy FFT module."""

import numpy as np

from tike.templates import PtychoCore
from tike.ptycho.shift import *


class PtychoNumPyFFT(PtychoCore):
    """Implement the ptychography operators using the NumPy FFT library."""

    def fwd(self, probe, v, h, psi, psi_corner=(0, 0)):
        if not (np.iscomplexobj(psi) and np.iscomplexobj(probe)):
            raise TypeError("psi and probe must be complex.")
        probe = probe.astype(np.complex64)
        psi = psi.astype(np.complex64)
        # Grab all of the patches where the probe and psi interact
        wave = uncombine_grids(
            grids_shape=(h.size, probe.shape[0], probe.shape[1]),
            v=v,
            h=h,
            combined=psi,
            combined_corner=psi_corner,
        )
        # Multiply the probe and patches of psi
        wavefront = probe * wave
        # Propagate the wavefronts to the detector
        return np.fft.fft2(
            wavefront,
            s=(self.detector_shape, self.detector_shape),
            # consider adding norm='ortho' here for numpy >= 1.10
        )

    def adj(self,
            farplane,
            probe, v, h,
            psi_shape, psi_corner=(0, 0),
            combined_probe=1,
    ):  # yapf: disable
        """Compute the nearplane complex wavefronts from the farfield and probe.

        Adjoint ptychography transform (Q*F*).

        The inverse ptychography operator. Computes the inverse Fourier transform
        of a series of farplane measurements, the combines these illuminations
        into a single psi using a weighted average.
        """
        nearplane = np.fft.ifft2(
            farplane)[:, :self.probe_shape, :self.probe_shape]
        return combine_grids(
            grids=nearplane,
            v=v,
            h=h,
            combined_shape=psi_shape,
            combined_corner=psi_corner,
        ) / combined_probe
