"""Implements a 2D alignmnent algorithm by Gunnar Farneback."""

import numpy as np
from cv2 import calcOpticalFlowFarneback


def _rescale_8bit(a, b):
    """Return a, b rescaled into the same 8-bit range"""

    h, e = np.histogram(a, 1000)
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
    data,
    unaligned,
    pyr_scale=0.5,
    levels=5,
    winsize=19,
    iterations=16,
    poly_n=5,
    poly_sigma=1.1,
    flags=4,
    flow=None,
    **kwargs,
):
    """Find the flow from unaligned to data using Farneback's algorithm

    For parameter descriptions see
    https://docs.opencv.org/4.3.0/dc/d6b/group__video__track.html

    Parameters
    ----------
    data, unaligned (L, M, N)
        The images to be aligned.
    flow : (L, M, N, 2) float32
        The inital guess for the displacement field.

    References
    ----------
    Farneback, Gunnar "Two-Frame Motion Estimation Based on Polynomial
    Expansion" 2003.
    """
    shape = data.shape

    if flow is None:
        flow = np.zeros((*shape, 2), dtype='float32')
    else:
        flow = np.copy(np.flip(flow, axis=-1))

    # NOTE: Passing a reshaped view as any of the parameters breaks OpenCV's
    # Farneback implementation.
    for i in range(len(data)):
        flow[i] = calcOpticalFlowFarneback(
            *_rescale_8bit(np.abs(unaligned[i]), np.abs(data[i])),
            flow=flow[i],
            pyr_scale=pyr_scale,
            levels=levels,
            winsize=winsize,
            iterations=iterations,
            poly_n=poly_n,
            poly_sigma=poly_sigma,
            flags=flags,
        )[..., ::-1]
    return {'shift': flow, 'cost': -1}
