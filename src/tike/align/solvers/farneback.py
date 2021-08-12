"""Implements a 2D alignmnent algorithm by Gunnar Farneback."""

import numpy as np
from cv2 import calcOpticalFlowFarneback


def _rescale_8bit(a, b, hi=None, lo=None):
    """Return a, b rescaled into the same 8-bit range.

    The images are rescaled into the range [lo, hi] if provided; otherwise, the
    range is decided by clipping the histogram of all bins that are less than
    0.5 percent of the fullest bin.

    """

    if hi is None or lo is None:
        h, e = np.histogram(b, 1000)
        stend = np.where(h > np.max(h) * 0.005)
        st = stend[0][0]
        end = stend[0][-1]
        lo = e[st]
        hi = e[end + 1]

    # Force all values into range [0, 255]
    a = (255 * (a - lo) / (hi - lo))
    b = (255 * (b - lo) / (hi - lo))
    a[a < 0] = 0
    a[a > 255] = 255
    b[b < 0] = 0
    b[b > 255] = 255
    assert np.all(a >= 0), np.all(b >= 0)
    assert np.all(a <= 255), np.all(b <= 255)
    return a, b


def farneback(
    op,
    original,
    unaligned,
    pyr_scale=0.5,
    levels=5,
    winsize=19,
    num_iter=16,
    poly_n=5,
    poly_sigma=1.1,
    flow=None,
    hi=None,
    lo=None,
    **kwargs,
):
    """Find the flow from unaligned to original using Farneback's algorithm

    For parameter descriptions see
    https://docs.opencv.org/4.3.0/dc/d6b/group__video__track.html

    Parameters
    ----------
    original, unaligned (L, M, N)
        The images to be aligned.
    flow : (L, M, N, 2) float32
        The inital guess for the displacement field.

    References
    ----------
    Farneback, Gunnar "Two-Frame Motion Estimation Based on Polynomial
    Expansion" 2003.
    """
    shape = original.shape
    assert original.dtype == 'float32', original.dtype
    assert unaligned.dtype == 'float32', unaligned.dtype

    if flow is None:
        flow = np.zeros((*shape, 2), dtype='float32')
    else:
        flow = flow[..., ::-1].copy()

    # NOTE: Passing a reshaped view as any of the parameters breaks OpenCV's
    # Farneback implementation.
    for i in range(len(original)):
        flow[i] = calcOpticalFlowFarneback(
            *_rescale_8bit(
                original[i],
                unaligned[i],
                hi=hi[i] if hi is not None else None,
                lo=lo[i] if lo is not None else None,
            ),
            flow=flow[i],
            pyr_scale=pyr_scale,
            levels=levels,
            winsize=winsize,
            iterations=num_iter,
            poly_n=poly_n,
            poly_sigma=poly_sigma,
            flags=4,
        )
    return {'flow': flow[..., ::-1], 'cost': -1}
