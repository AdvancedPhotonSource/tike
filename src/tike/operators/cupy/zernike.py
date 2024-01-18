"""Defines an inverse-Zernike transform operator."""

__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2024, UChicago Argonne, LLC."

import numpy.typing as npt
import numpy as np
import tike.zernike
import tike.linalg

from .operator import Operator


class Zernike(Operator):
    """Reconstruct an image from coefficients and zernike basis using CuPy.

    Take an (..., W) array of zernike coefficients and reconstruct an image
    from them.

    Parameters
    ----------
    size : int
        The pixel width and height of the reconstruction.
    weights: (..., W) complex64
        The zernike coefficients

    .. versionadded:: 0.25.5

    """

    def fwd(
        self,
        weights: npt.NDArray[np.csingle],
        size: int,
        degree_max: int,
        **kwargs,
    ) -> npt.NDArray[np.csingle]:
        basis = tike.zernike.basis(
            size=size,
            degree_min=0,
            degree_max=degree_max,
            xp=self.xp,
        )
        # (..., W, 1, 1) * (W, size, size)
        return np.sum(weights[..., None, None] * basis, axis=-3)

    def adj(
        self,
        images: npt.NDArray[np.csingle],
        size: int,
        degree_max: int,
        **kwargs,
    ) -> npt.NDArray[np.csingle]:
        basis = tike.zernike.basis(
            size=size,
            degree_min=0,
            degree_max=degree_max,
            xp=self.xp,
        )
        # (..., 1, size, size) * (W, size, size)
        return np.sum(images[..., None, :, :] * basis, axis=(-2, -1))
