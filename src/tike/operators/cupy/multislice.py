"""Defines a free-space propagation operator based on the CuPy FFT module."""

__author__ = "Ashish Tripathi"
__copyright__ = "Copyright (c) 2024, UChicago Argonne, LLC."

import typing
import numpy.typing as npt
import numpy as np
import cupy as cp

from .operator import Operator

from .propagation import Propagation, ZeroPropagation
from .convolution import Convolution


class Multislice(Operator):
    def __init__(
        self,
        detector_shape: int,
        probe_shape: int,
        nz: int,
        n: int,
        propagation: typing.Type[Propagation] = ZeroPropagation,
        diffraction: typing.Type[Convolution] = Convolution,
        norm: str = "ortho",
        nslices: int = 1,
        **kwargs,
    ):
        """Please see help(Multislice) for more info."""
        self.diffraction = diffraction(
            probe_shape=probe_shape,
            detector_shape=detector_shape,
            nz=nz,
            n=n,
            **kwargs,
        )
        self.propagation = propagation(
            detector_shape=detector_shape,
        )

        # TODO: Replace these with @property functions
        self.probe_shape = probe_shape
        self.detector_shape = detector_shape
        self.nz = nz
        self.n = n
        self.nslices = nslices

    def __enter__(self):
        self.propagation.__enter__()
        self.diffraction.__enter__()
        return self

    def __exit__(self, type, value, traceback):
        self.propagation.__exit__(type, value, traceback)
        self.diffraction.__exit__(type, value, traceback)

    def fwd(
        self,
        probe: npt.NDArray[np.csingle],
        scan: npt.NDArray[np.single],
        psi: npt.NDArray[np.csingle],
        **kwargs,
    ) -> npt.NDArray[np.csingle]:
        """Please see help(SingleSlice) for more info."""
        assert psi.shape[0] == self.nslices and psi.ndim == 3
        exitwave = probe
        for s in range(self.nslices):
            exitwave = self.diffraction.fwd(
                psi=psi[s],
                scan=scan,
                probe=exitwave,
            )
        return exitwave

    def adj(
        self,
        nearplane: npt.NDArray[np.csingle],
        probe: npt.NDArray[np.csingle],
        scan: npt.NDArray[np.single],
        psi: npt.NDArray[np.csingle],
        overwrite: bool = False,
        **kwargs,
    ) -> npt.NDArray[np.csingle]:
        """Please see help(SingleSlice) for more info."""
        probe_adj = nearplane
        psi_adj = self.xp.zeros_like(psi)
        exitwave = [
            None,
        ] * len(psi)
        exitwave[0] = probe
        for s in range(1, self.nslices):
            exitwave[s] = self.diffraction.fwd(
                psi=psi[s - 1],
                scan=scan,
                probe=exitwave[s - 1],
            )
        for s in range(self.nslices - 1, -1, -1):
            psi_adj[s] = self.diffraction.adj(
                nearplane=probe_adj,
                probe=exitwave[s],
                scan=scan,
                overwrite=False,
            )
            probe_adj = self.diffraction.adj_probe(
                nearplane=probe_adj,
                scan=scan,
                psi=psi[s],
            )
        # FIXME: Why is division by nslices needed here?
        return psi_adj / self.nslices, probe_adj

    @property
    def patch(self):
        return self.diffraction.patch

    @property
    def pad(self):
        return self.diffraction.pad

    @property
    def end(self):
        return self.diffraction.end


class SingleSlice(Multislice):
    """Single slice wavefield propgation"""

    def __init__(
        self,
        detector_shape: int,
        probe_shape: int,
        nz: int,
        n: int,
        propagation: typing.Type[Propagation] = ZeroPropagation,
        diffraction: typing.Type[Convolution] = Convolution,
        norm: str = "ortho",
        **kwargs,
    ):
        """Please see help(SingleSlice) for more info."""
        self.diffraction = diffraction(
            probe_shape=probe_shape,
            detector_shape=detector_shape,
            nz=nz,
            n=n,
            **kwargs,
        )
        self.propagation = propagation(
            detector_shape=detector_shape,
        )

        # TODO: Replace these with @property functions
        self.probe_shape = probe_shape
        self.detector_shape = detector_shape
        self.nz = nz
        self.n = n
        self.nslices = 1

    def fwd(
        self,
        probe: npt.NDArray[np.csingle],
        scan: npt.NDArray[np.single],
        psi: npt.NDArray[np.csingle],
        **kwargs,
    ) -> npt.NDArray[np.csingle]:
        """Please see help(SingleSlice) for more info."""
        assert psi.shape[0] == 1 and psi.ndim == 3
        return self.diffraction.fwd(
            psi=psi[0],
            scan=scan,
            probe=probe,
        )

    def adj(
        self,
        nearplane: npt.NDArray[np.csingle],
        probe: npt.NDArray[np.csingle],
        scan: npt.NDArray[np.single],
        psi: typing.Optional[npt.NDArray[np.csingle]] = None,
        overwrite: bool = False,
        **kwargs,
    ) -> npt.NDArray[np.csingle]:
        """Please see help(SingleSlice) for more info."""
        assert psi is None or (psi.shape[0] == 1 and psi.ndim == 3)
        psi_adj = self.diffraction.adj(
            nearplane=nearplane,
            probe=probe,
            scan=scan,
            overwrite=False,
        )[None, ...]
        probe_adj = self.diffraction.adj_probe(
            nearplane=nearplane,
            scan=scan,
            psi=psi[0],
            overwrite=False,
        )
        return psi_adj, probe_adj
