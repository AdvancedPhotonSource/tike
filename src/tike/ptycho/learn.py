"""Implements functions for ptychographic deep learning."""

__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2021, UChicago Argonne, LLC."

from tike.operators import Patch
from .position import check_allowed_positions


def extract_patches(psi, scan, patch_width):
    """Extract patches from the object function.

    Parameters
    ----------
    scan : (..., POSI, 2) float32
        Coordinates of the minimum corner of the patch grid for each
        extracted patch.
    psi : (..., WIDE, HIGH) complex64
        The complex wavefront modulation of the object.
    patch_width : int
        The desired width of the square patches to be extraced.

    Returns
    -------
    patches : (..., POSI, patch_width, patch_width) complex64 numpy-array
        Patches of psi extracted at the given scan positions.

    """
    check_allowed_positions(scan, psi, (patch_width, patch_width))
    with Patch() as operator:
        psi = operator.asarray(psi, dtype='complex64')
        scan = operator.asarray(scan, dtype='float32')
        patches = operator.fwd(
            images=psi,
            positions=scan,
            patch_width=patch_width,
        )
        patches = operator.asnumpy(patches)
    return patches
