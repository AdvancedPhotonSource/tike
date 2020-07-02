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
    levels=5,
    winsize=13,
    iterations=10,
    poly_n=5,
    poly_sigma=1.1,
    flags=None,
    flow=None,
    **kwargs,
):
    """Find the flow from unaligned to data using Farneback's algorithm

    For parameter descriptions see https://docs.opencv.org/4.3.0/dc/d6b/group__video__track.html#ga5d10ebbd59fe09c5f650289ec0ece5af

    Parameters
    ----------
    data, unaligned (..., M, N)
        The images to be aligned.
    flow : (..., M, N, 2) float
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
            *_rescale_8bit(data[i], unaligned[i]),
            pyr_scale,
            levels,
            winsize,
            iterations,
            poly_n,
            poly_sigma,
            flags,
            flow=flow[i],
        )
    return flow.reshape(*shape, 2)
