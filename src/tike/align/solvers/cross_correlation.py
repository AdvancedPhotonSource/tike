# Copyright (C) 2019, the scikit-image team
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of skimage nor the names of its contributors may be used
#    to endorse or promote products derived from this software without specific
#    prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
# EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""Implements a cross_correlation 2D alignment algorithm based on
phase_cross_correlation from skimage.registration."""

import numpy as np


def cross_correlation(
    op,
    original,
    unaligned,
    upsample_factor=1,
    space="real",
    num_iter=None,
    reg_weight=1e-9,
):
    """Efficient subpixel image translation alignment by cross-correlation.

    This code gives the same precision as the FFT upsampled cross-correlation
    in a fraction of the computation time and with reduced memory requirements.
    It obtains an initial estimate of the cross-correlation peak by an FFT and
    then refines the shift estimation by upsampling the DFT only in a small
    neighborhood of that estimate by means of a matrix-multiply DFT.

    References
    ----------
    Stéfan van der Walt, Johannes L. Schönberger, Juan Nunez-Iglesias,
    François Boulogne, Joshua D. Warner, Neil Yager, Emmanuelle
    Gouillart, Tony Yu and the scikit-image contributors. scikit-image:
    Image processing in Python. PeerJ 2:e453 (2014)
    :doi:`10.7717/peerj.453`

    Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
    "Efficient subpixel image registration algorithms," Optics Letters
    33, 156-158 (2008). :doi:`10.1364/OL.33.000156`

    James R. Fienup, "Invariant error metrics for image reconstruction"
    Optics Letters 36, 8352-8357 (1997). :doi:`10.1364/AO.36.008352`
    """
    # assume complex data is already in Fourier space
    if space.lower() == 'fourier':
        src_freq = unaligned
        target_freq = original
    # real data needs to be fft'd.
    elif space.lower() == 'real':
        src_freq = op.xp.fft.fft2(unaligned)
        target_freq = op.xp.fft.fft2(original)
    else:
        raise ValueError(f"space must be 'fourier' or 'real' not '{space}'.")

    # Whole-pixel shift - Compute cross-correlation by an IFFT
    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()
    cross_correlation = op.xp.fft.ifft2(image_product)

    # Add a small regularization term so that smaller shifts are preferred when
    # the cross_correlation is the same for multiple shifts.
    if reg_weight > 0:
        w = _area_overlap(op, cross_correlation)
        w = op.xp.fft.fftshift(w) * reg_weight
    else:
        w = 0

    A = np.abs(cross_correlation) + w
    maxima = A.reshape(A.shape[0], -1).argmax(1)
    maxima = np.column_stack(np.unravel_index(maxima, A[0, :, :].shape))
    shifts = op.xp.array(maxima, dtype='float32')

    midpoints = [x // 2 for x in shape[1:]]
    ids = np.where(shifts[:, 0] > midpoints[0])
    shifts[ids[0], 0] -= shape[1]
    ids = np.where(shifts[:, 1] > midpoints[1])
    shifts[ids[0], 1] -= shape[2]

    if upsample_factor > 1:
        # Initial shift estimate in upsampled grid
        shifts = np.round(shifts * upsample_factor) / upsample_factor
        upsampled_region_size = np.ceil(upsample_factor * 1.5)
        # Center of output array at dftshift + 1
        dftshift = np.fix(upsampled_region_size / 2.0)

        normalization = (src_freq[0].size * upsample_factor**2)
        # Matrix multiply DFT around the current shift estimate

        sample_region_offset = dftshift - shifts * upsample_factor
        cross_correlation = _upsampled_dft(
            op,
            image_product.conj(),
            upsampled_region_size,
            upsample_factor,
            sample_region_offset,
        ).conj()
        cross_correlation /= normalization
        # Locate maximum and map back to original pixel grid
        A = np.abs(cross_correlation)
        maxima = A.reshape(A.shape[0], -1).argmax(1)
        maxima = np.column_stack(np.unravel_index(maxima, A[0, :, :].shape))
        maxima = maxima - dftshift
        shifts = shifts + maxima / upsample_factor
    return {'shift': shifts.astype('float32'), 'cost': -1}


def _upsampled_dft(op, data, ups, upsample_factor, axis_offsets):
    im2pi = -2j * np.pi
    shape = data.shape
    kernel = ((op.xp.arange(ups) - axis_offsets[:, 1:2])[:, :, None] *
              op.xp.fft.fftfreq(shape[2], upsample_factor))
    kernel = np.exp(im2pi * kernel)
    data = np.einsum('ijk,ipk->ijp', kernel, data)
    kernel = ((op.xp.arange(ups) - axis_offsets[:, 0:1])[:, :, None] *
              op.xp.fft.fftfreq(shape[1], upsample_factor))
    kernel = np.exp(im2pi * kernel)
    return np.einsum('ijk,ipk->ijp', kernel, data)


def _triangle(op, N):
    """Return N samples from the triangle function."""
    x = op.xp.linspace(0, 1, N, endpoint=False) + 0.5 / N
    return 1 - abs(x - 0.5)


def _area_overlap(op, A):
    """Return overlapping area of A with itself.

    Create overlap arrays for higher dimensions using matrix multiplication.

    >>> _area_overlap(np.empty(4))
    array([0.625, 0.875, 0.875, 0.625])
    >>> _area_overlap(np.empty((3, 5)))
    array([[0.4       , 0.53333333, 0.66666667, 0.53333333, 0.4       ],
           [0.6       , 0.8       , 1.        , 0.8       , 0.6       ],
           [0.4       , 0.53333333, 0.66666667, 0.53333333, 0.4       ]])
    """
    for dim, shape in enumerate(A.shape[-2:]):
        if dim == 0:
            w = _triangle(op, shape)
        else:
            w = w[..., np.newaxis] @ _triangle(op, shape)[np.newaxis, ...]
    return w
