"""This module implements patch grabbing functions."""

import numpy as np
import scipy.ndimage.interpolation as sni


def _combine_grids(
        grids, v, h,
        combined_shape, combined_corner=(0, 0),
):  # yapf: disable
    """Combine grids by summation.

    Multiple grids are interpolated onto a single combined grid using
    bilinear interpolation.

    Parameters
    ----------
    grids : (N, V, H) :py:class:`numpy.array` complex64
        The values on the grids.
    v, h : (M, ) :py:class:`numpy.array` float [m]
        The coordinates of the minimum corner of the M grids.
    combined_shape : (2, ) int
        The last two are numbers of the tuple are the number of indices along
        the h and v directions of the combined grid.
    combined_corner : (2, ) float [m]
        The coordinates of the minimum corner of the combined grids.

    Return
    ------
    combined : (T, V, H) :py:class:`numpy.array` complex64
        The combined grid.

    """
    grids = grids.astype(np.complex64, casting='same_kind', copy=False)
    m_shape, v_shape, h_shape = grids.shape
    vshift, V, V1 = _shift_coords(v, v_shape, combined_corner[-2],
                                  combined_shape[-2])
    hshift, H, H1 = _shift_coords(h, h_shape, combined_corner[-1],
                                  combined_shape[-1])
    # Create a summed_grids large enough to hold all of the grids
    # plus some padding to cancel out the padding added for shifting
    grids = _fast_pad(grids, 1, 1)
    combined = np.zeros([combined_shape[-2] + 2, combined_shape[-1] + 2],
                        dtype=grids.dtype)
    # Add each of the grids to the appropriate place on the summed_grids
    nprobes = h.size
    grids = grids.view(np.float32).reshape(*grids.shape, 2)
    combined = combined.view(np.float32).reshape(*combined.shape, 2)
    for N in range(nprobes):
        combined[V[N]:V1[N], H[N]:H1[N], ...] += sni.shift(
            grids[N], [vshift[N], hshift[N], 0], order=1)
    combined = combined.view(np.complex64)
    return combined[1:-1, 1:-1, 0]


def _uncombine_grids(
        grids_shape, v, h,
        combined, combined_corner=(0, 0),
):  # yapf: disable
    """Extract a series of grids from a single grid.

    The grids are interpolated onto the combined grid using bilinear
    interpolation.

    Parameters
    ----------
    grids_shape : (2, ) int
        The last two are numbers of the tuple are the number of indices along
        the h and v directions of the grids.
    v, h : (M, ) :py:class:`numpy.array` float [m]
        The real coordinates of the minimum corner of the M grids.

    combined : (V, H) :py:class:`numpy.array` complex64
        The combined grids.
    combined_corner : (2, ) float [m]
        The real coordinates of the minimum corner of the combined grids

    Return
    ------
    grids : (M, V, H) :py:class:`numpy.array` complex64
        The decombined grids.

    """
    combined = combined.astype(np.complex64, casting='same_kind', copy=False)
    v_shape, h_shape = grids_shape[-2:]
    vshift, V, V1 = _shift_coords(v, v_shape, combined_corner[-2],
                                  combined.shape[-2])
    hshift, H, H1 = _shift_coords(h, h_shape, combined_corner[-1],
                                  combined.shape[-1])
    # Create a grids large enough to hold all of the grids
    # plus some padding to cancel out the padding added for shifting
    combined = _fast_pad(combined, 1, 1)
    grids = np.empty(grids_shape, dtype=combined.dtype)
    # Retrive the updated values of each of the grids
    nprobes = h.size
    grids = grids.view(np.float32).reshape(*grids.shape, 2)
    combined = combined.view(np.float32).reshape(*combined.shape, 2)
    for N in range(nprobes):
        grids[N] = sni.shift(
            combined[V[N]:V1[N], H[N]:H1[N], ...],
            [-vshift[N], -hshift[N], 0],
            order=1,
        )[1:-1, 1:-1, ...]
    return grids.view(np.complex64)[..., 0]


def _fast_pad(unpadded_grid, npadv, npadh):
    """Pad symmetrically with zeros along the last two dimensions.

    The `unpadded_grid` keeps its own data type. The padded_shape is the same
    in all dimensions except for the last two. The unpadded_grid is extracted
    or placed into the an array that is the size of the padded_shape.

    Notes
    -----
    Issue (numpy/numpy #11126, 21 May 2018): Copying into preallocated array
    is faster because of the way that np.pad uses concatenate.

    """
    if npadv < 0 or npadh < 0:
        raise ValueError("Pads must be non-negative.")
    if npadv > 0 or npadv > 0:
        padded_shape = list(unpadded_grid.shape)
        padded_shape[-2] += 2 * npadv
        padded_shape[-1] += 2 * npadh
        padded_grid = np.zeros(padded_shape, dtype=unpadded_grid.dtype)
        padded_grid[
            ...,
            npadv:padded_shape[-2] - npadv,
            npadh:padded_shape[-1] - npadh,
        ] = unpadded_grid  # yapf: disable
        return padded_grid
    return unpadded_grid


def _shift_coords(r_min, r_shape, combined_min, combined_shape):
    """Find the positions of some 1D ranges in a new 1D coordinate system.

    Pad the new range coordinates with one on each side.

    Parameters
    ----------
    r_min, r_shape : :py:class:`numpy.array` float, int
        1D min and range to be transformed.
    combined_min, combined_shape : :py:class:`numpy.array` float, int
        The min and range of the new coordinate system.

    Return
    ------
    r_shift : :py:class:`numpy.array` float
        The shifted coordinate remainder.
    r_lo, r_hi : :py:class:`numpy.array` int
        New range integer starts and ends.

    """
    # Find the float coordinates of each range on the combined grid
    r_shift = (r_min - combined_min).flatten()
    # Find integer indices (floor) of each range on the combined grid
    r_lo = np.floor(r_shift).astype(int)
    r_hi = (r_lo + (r_shape + 2)).astype(int)
    if np.any(r_lo < 0) or np.any(r_hi > combined_min + combined_shape + 2):
        raise ValueError("Index {} or {} is off the grid!".format(
            np.min(r_lo), np.max(r_hi)))
    # Find the remainder shift less than 1
    r_shift -= r_shift.astype(int)
    return r_shift, r_lo, r_hi
