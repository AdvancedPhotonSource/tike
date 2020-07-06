"""Implements a 2D alignmnent algorithm by Gunnar Farneback."""

import numpy as np
from cv2 import calcOpticalFlowFarneback


def _rescale_8bit(a, b):
    """Return a, b rescaled into the same 8-bit range"""
    hi = max(a.max(), b.max())
    lo = min(a.min(), b.min())
    a = (255 * (a - lo) / (hi - lo)).astype('uint8')
    b = (255 * (b - lo) / (hi - lo)).astype('uint8')
    return a, b


def farneback(
    op,
    data,
    unaligned,
    pyr_scale=0.5,
    levels=3,
    winsize=19,
    iterations=16,
    poly_n=5,
    poly_sigma=4,
    flags=4,
    flow=None,
    **kwargs,
):
    """Find the flow from unaligned to data using Farneback's algorithm

    For parameter descriptions see
    https://docs.opencv.org/4.3.0/dc/d6b/group__video__track.html

    Parameters
    ----------
    data, unaligned (..., M, N)
        The images to be aligned.
    flow : (..., M, N, 2) float32
        The inital guess for the displacement field.

    References
    ----------
    Farneback, Gunnar "Two-Frame Motion Estimation Based on Polynomial
    Expansion" 2003.
    """
    # Reshape inputs into stack of 2D slices
    shape = data.shape
    h, w = shape[-2:]
    data = data.reshape(-1, h, w)
    unaligned = data.reshape(-1, h, w)
    if flow is None:
        flow = op.xp.zeros((*shape, 2), dtype='float32')
    flow = flow.reshape(-1, h, w, 2)

    for i in range(len(data)):
        flow[i] = calcOpticalFlowFarneback(
            *_rescale_8bit(np.angle(data[i]), np.angle(unaligned[i])),
            flow=flow[i],
            pyr_scale=pyr_scale,
            levels=levels,
            winsize=winsize,
            iterations=iterations,
            poly_n=poly_n,
            poly_sigma=poly_sigma,
            flags=flags,
        )
    return {'shift': flow.reshape(*shape, 2), 'cost': -1}
