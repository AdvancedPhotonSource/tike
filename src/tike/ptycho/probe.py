"""Functions related to creating and manipulating probe arrays.

Ptychographic probes are represented as two separate components: a shared probe
whose values are the same for all positions and the varying component. The
former is required as it provides the shared probe constraint for ptychography
and the later relaxes the former constraint to accomodate real-world
illuminations which may vary with time.

The shared component consist of a single array representing at least one probe
each of which may have an accompanying varying component.

The varying components are stored sparsely as two arrays, and the full
representation of the varying comonents are only combined as needed. The first
array is an array of eigen probes (principal components) spanning the space of
the probe variation of all positions and the second is an array of weights that
map the variation for each position into this space.

Each probe may have its own set of eigen probes. The unique probe at a given
position is reconstructed by adding the shared probe to a weighted sum of the
eigen probes.

.. code-block:: python

    varying_probe = weights[0] * probe + np.sum(weights[1:] * eigen_probes)


Design comments
---------------
In theory, the probe representation could be implemented in as little as two
arrays: one with all of the shared components where the probe becomes the first
eigen probe and and one with the weights. Choosing to keep the eigen probes
separate from the probe as a third array provides backwards compatability and
allows for storing fewer eigen probes in the case when only some probes are
allowed to vary.

"""

import logging

import cupy as cp
import cupyx.scipy.ndimage
import numpy as np

import tike.random

logger = logging.getLogger(__name__)


class ProbeOptions:
    """Manage data and setting related to probe correction.

    Attributes
    ----------
    orthogonality_constraint : bool
        Forces probes to be orthogonal each iteration.
    num_eigen_probes : int
        The number of eigen probes/components.
    """

    def __init__(
        self,
        num_eigen_probes=0,
        orthogonality_constraint=True,
        use_adaptive_moment=False,
        vdecay=0.6,
        mdecay=0.666,
        centered_intensity_constraint=True,
    ):
        self.orthogonality_constraint = orthogonality_constraint
        self._weights = None
        self._eigen_probes = None
        if num_eigen_probes > 0:
            pass
        self.use_adaptive_moment = use_adaptive_moment
        self.vdecay = vdecay
        self.mdecay = mdecay
        self.v = None
        self.m = None
        self.centered_intensity_constraint = centered_intensity_constraint

    @property
    def num_eigen_probes(self):
        return 0 if self._weights is None else self._weights.shape[-2]

    def put(self):
        """Copy to the current GPU memory."""
        if self.v is not None:
            self.v = cp.asarray(self.v)
        if self.m is not None:
            self.m = cp.asarray(self.m)
        return self

    def get(self):
        """Copy to the host CPU memory."""
        if self.v is not None:
            self.v = cp.asnumpy(self.v)
        if self.m is not None:
            self.m = cp.asnumpy(self.m)
        return self


def get_varying_probe(shared_probe, eigen_probe=None, weights=None):
    """Construct the varying probes.

    Parameters
    ----------
    shared_probe : (..., 1,         1, SHARED, WIDE, HIGH) complex64
        The shared probes amongst all positions.
    eigen_probe :  (..., 1,     EIGEN, SHARED, WIDE, HIGH) complex64
        The eigen probes for all positions.
    weights :   (..., POSI, EIGEN + 1, SHARED) float32
        The relative intensity of the eigen probes at each position.

    Returns
    -------
    unique_probes : (..., POSI, 1, 1, WIDE, HIGH)
    """
    if weights is not None:
        # The zeroth eigen_probe is the shared_probe
        unique_probe = weights[..., [0], :, None, None] * shared_probe
        for w in range(1, weights.shape[-2]):
            unique_probe += (
                weights[..., [w], :, None, None]
                * eigen_probe[..., [w - 1], :, :, :]
            )  # yapf: disable
        return unique_probe
    else:
        return shared_probe.copy()


def constrain_variable_probe(variable_probe, weights):
    """Add the following constraints to variable probe weights

    1. Remove outliars from weights
    2. Enforce orthogonality once per epoch

    """
    logger.info('Orthogonalize variable probes')
    variable_probe = tike.linalg.orthogonalize_gs(
        variable_probe,
        axis=(-3, -2, -1),
    )

    logger.info('Remove outliars from variable probe weights')
    aevol = cp.abs(weights)
    weights = cp.minimum(
        aevol,
        1.5 * cp.percentile(
            aevol,
            [95],
            axis=[-3],
            keepdims=True,
        ).astype(weights.dtype),
    ) * cp.sign(weights)

    # TODO: Smooth the weights as a function of the frame index.

    return variable_probe, weights


def update_eigen_probe(comm, R, eigen_probe, weights, patches, diff, β=0.1):
    """Update eigen probes using residual probe updates.

    This update is copied from the source code of ptychoshelves. It is similar
    to, but not the same as, equation (31) described by Odstrcil et al (2018).
    It is also different from updates described in Odstrcil et al (2016).
    However, they all aim to correct for probe variation.

    Parameters
    ----------
    comm : :py:class:`tike.communicators.Comm`
        An object which manages communications between both GPUs and nodes.
    R : (..., POSI, 1, 1, WIDE, HIGH) complex64
        Residual probe updates; what's left after subtracting the shared probe
        update from the varying probe updates for each position
    patches : (..., POSI, 1, 1, WIDE, HIGH) complex64
    diff : (..., POSI, 1, 1, WIDE, HIGH) complex64
    eigen_probe : (..., 1, 1, 1, WIDE, HIGH) complex64
        The eigen probe being updated.
    β : float
        A relaxation constant that controls how quickly the eigen probe modes
        are updated. Recommended to be < 1 for mini-batch updates.
    weights : (..., POSI) float32
        A vector whose elements are sums of the previous optimal updates for
        each posiiton.

    References
    ----------
    M. Odstrcil, P. Baksh, S. A. Boden, R. Card, J. E. Chad, J. G. Frey, W. S.
    Brocklesby, "Ptychographic coherent diffractive imaging with orthogonal
    probe relaxation." Opt. Express 24, 8360 (2016). doi: 10.1364/OE.24.008360

    Michal Odstrcil, Andreas Menzel, and Manuel Guizar-Sicaros. Iterative
    least-squares solver for generalized maximum-likelihood ptychography.
    Optics Express. 2018.
    """
    assert R[0].shape[-3] == R[0].shape[-4] == 1
    assert eigen_probe[0].shape[-3] == 1 == eigen_probe[0].shape[-5]
    assert R[0].shape[:-5] == eigen_probe[0].shape[:-5] == weights[0].shape[:-1]
    assert weights[0].shape[-1] == R[0].shape[-5]
    assert R[0].shape[-2:] == eigen_probe[0].shape[-2:]

    def _get_update(R, eigen_probe, weights):
        # (..., POSI, 1, 1, 1, 1) to match other arrays
        weights = weights[..., None, None, None, None]
        norm_weights = np.linalg.norm(weights, axis=-5, keepdims=True)**2

        if np.all(norm_weights == 0):
            raise ValueError('eigen_probe weights cannot all be zero?')

        # FIXME: What happens when weights is zero!?
        proj = (np.real(R.conj() * eigen_probe) + weights) / norm_weights
        return weights, np.mean(
            R * np.mean(proj, axis=(-2, -1), keepdims=True),
            axis=-5,
            keepdims=True,
        )

    weights, update = (list(a) for a in zip(*comm.pool.map(
        _get_update,
        R,
        eigen_probe,
        weights,
    )))
    if comm.use_mpi:
        update[0] = comm.Allreduce_mean(
            update,
            axis=-5,
        )
        update = comm.pool.bcast([update[0]])
    else:
        update = comm.pool.bcast([comm.pool.reduce_mean(
            update,
            axis=-5,
        )])

    def _get_d(patches, diff, eigen_probe, update, β):
        eigen_probe += β * update / np.linalg.norm(
            update,
            axis=(-2, -1),
            keepdims=True,
        )
        assert np.all(np.isfinite(eigen_probe))

        eigen_probe /= np.linalg.norm(eigen_probe, axis=(-2, -1), keepdims=True)

        # Determine new eigen_weights for the updated eigen probe
        phi = patches * eigen_probe
        n = np.mean(
            np.real(diff * phi.conj()),
            axis=(-1, -2),
            keepdims=True,
        )
        norm_phi = np.square(np.abs(phi))
        d = np.mean(norm_phi, axis=(-1, -2), keepdims=True)
        d_mean = np.mean(d, axis=-5, keepdims=True)
        return n, d, d_mean

    (n, d, d_mean) = (list(a) for a in zip(*comm.pool.map(
        _get_d,
        patches,
        diff,
        eigen_probe,
        update,
        β=β,
    )))

    if comm.use_mpi:
        d_mean[0] = comm.Allreduce_mean(
            d_mean,
            axis=-5,
        )
        d_mean = comm.pool.bcast([d_mean[0]])
    else:
        d_mean = comm.pool.bcast([comm.pool.reduce_mean(
            d_mean,
            axis=-5,
        )])

    def _get_weights_mean(n, d, d_mean, weights):
        d += 0.1 * d_mean

        weight_update = (n / d).reshape(*weights.shape)
        assert np.all(np.isfinite(weight_update))

        # (33) The sum of all previous steps constrained to zero-mean
        weights += weight_update
        return np.mean(
            weights,
            axis=-5,
            keepdims=True,
        )

    weights_mean = list(comm.pool.map(
        _get_weights_mean,
        n,
        d,
        d_mean,
        weights,
    ))
    if comm.use_mpi:
        weights_mean[0] = comm.Allreduce_mean(
            weights_mean,
            axis=-5,
        )
        weights_mean = comm.pool.bcast([weights_mean[0]])
    else:
        weights_mean = comm.pool.bcast(
            [comm.pool.reduce_mean(
                weights_mean,
                axis=-5,
            )])

    def _update_weights(weights, weights_mean):
        weights -= weights_mean
        return weights[..., 0, 0, 0, 0]

    weights = comm.pool.map(
        _update_weights,
        weights,
        weights_mean,
    )

    return eigen_probe, weights


def add_modes_random_phase(probe, nmodes):
    """Initialize additional probe modes by phase shifting the first mode.

    Parameters
    ----------
    probe : (..., M, :, :) array
        A probe with M > 0 incoherent modes.
    nmodes : int
        The number of desired modes.

    References
    ----------
    M. Odstrcil, P. Baksh, S. A. Boden, R. Card, J. E. Chad, J. G. Frey, W. S.
    Brocklesby, "Ptychographic coherent diffractive imaging with orthogonal
    probe relaxation." Opt. Express 24, 8360 (2016). doi: 10.1364/OE.24.008360
    """
    all_modes = np.empty((*probe.shape[:-3], nmodes, *probe.shape[-2:]),
                         dtype='complex64')
    pw = probe.shape[-1]
    for m in range(nmodes):
        if m < probe.shape[-3]:
            # copy existing mode
            all_modes[..., m, :, :] = probe[..., m, :, :]
        else:
            # randomly shift the first mode
            shift = np.exp(-2j * np.pi * (np.random.rand(2, 1) - 0.5) *
                           ((np.arange(0, pw) + 0.5) / pw - 0.5))
            all_modes[..., m, :, :] = (probe[..., 0, :, :] * shift[0][None] *
                                       shift[1][:, None])
    return all_modes


def simulate_varying_weights(scan, eigen_probe):
    """Generate weights for eigen probe that follow random sinusoid.

    The amplitude of the of weights is 1, the phase shift is (0, 2π], and the
    period is at most one full scan.
    """
    N = scan.shape[1]
    x = np.arange(N)[..., :, None, None]
    period = N * np.random.rand(*eigen_probe.shape[:-2])
    phase = 2 * np.pi * np.random.rand(*eigen_probe.shape[:-2])
    return np.sin(2 * np.pi / period * x - phase)


def init_varying_probe(scan, shared_probe, N):
    """Initialize arrays for N eigen modes."""
    if N < 1:
        return None, None

    eigen_probe = tike.random.numpy_complex(
        *shared_probe.shape[:-4],
        N - 1,
        *shared_probe.shape[-3:],
    ).astype('complex64')
    eigen_probe /= np.linalg.norm(eigen_probe, axis=(-2, -1), keepdims=True)

    weights = 1e-6 * np.random.rand(
        *scan.shape[:-1],
        N,
        shared_probe.shape[-3],
    ).astype('float32')
    weights -= np.mean(weights, axis=-3, keepdims=True)
    weights[..., 0, :] = 1.0  # The weight of the first eigen probe is non-zero

    return eigen_probe, weights


def orthogonalize_eig(x):
    """Orthogonalize modes of x using eigenvectors of the pairwise dot product.

    Parameters
    ----------
    x : (..., nmodes, :, :) array_like complex64
        An array of the probe modes vectorized

    References
    ----------
    M. Odstrcil, P. Baksh, S. A. Boden, R. Card, J. E. Chad, J. G. Frey, W. S.
    Brocklesby, "Ptychographic coherent diffractive imaging with orthogonal
    probe relaxation." Opt. Express 24, 8360 (2016). doi: 10.1364/OE.24.008360
    """
    nmodes = x.shape[-3]
    # 'A' holds the dot product of all possible mode pairs. We only fill the
    # lower half of `A` because it is conjugate-symmetric
    A = cp.empty((*x.shape[:-3], nmodes, nmodes), dtype='complex64')
    for i in range(nmodes):
        for j in range(i + 1):
            A[..., i, j] = cp.sum(cp.conj(x[..., i, :, :]) * x[..., j, :, :],
                                  axis=(-1, -2))

    _, vectors = cp.linalg.eigh(A, UPLO='L')
    # np.linalg.eigh guarantees that the eigen values are returned in ascending
    # order, so we just reverse the order of modes to have them sorted in
    # descending order.

    # TODO: Optimize this double-loop
    x_new = cp.zeros_like(x)
    for i in range(nmodes):
        for j in range(nmodes):
            # Sort new modes by eigen value in decending order.
            x_new[..., nmodes - 1 -
                  j, :, :] += vectors[..., i, j, None, None] * x[..., i, :, :]
    assert x_new.shape == x.shape, [x_new.shape, x.shape]

    return x_new


def gaussian(size, rin=0.8, rout=1.0):
    """Return a complex gaussian probe distribution.

    Illumination probe represented on a 2D regular grid.

    A finite-extent circular shaped probe is represented as
    a complex wave. The intensity of the probe is maximum at
    the center and damps to zero at the borders of the frame.

    Parameters
    ----------
    size : int
        The side length of the distribution
    rin : float [0, 1) < rout
        The inner radius of the distribution where the dampening of the
        intensity will start.
    rout : float (0, 1] > rin
        The outer radius of the distribution where the intensity will reach
        zero.

    """
    r, c = np.mgrid[:size, :size] + 0.5
    rs = np.sqrt((r - size / 2)**2 + (c - size / 2)**2)
    rmax = np.sqrt(2) * 0.5 * rout * rs.max() + 1.0
    rmin = np.sqrt(2) * 0.5 * rin * rs.max()
    img = np.zeros((size, size), dtype='float32')
    img[rs < rmin] = 1.0
    img[rs > rmax] = 0.0
    zone = np.logical_and(rs > rmin, rs < rmax)
    img[zone] = np.divide(rmax - rs[zone], rmax - rmin)
    return img


def constrain_center_peak(probe):
    """Force the peak illumination intensity to the center of the probe grid.

    After smoothing the intensity of the combined illumination with a gaussian
    filter with standard deviation sigma, the probe is shifted such that the
    maximum intensity is centered.
    """
    half = probe.shape[-2] // 2, probe.shape[-1] // 2
    logger.info("Constrained probe intensity to center with sigma=%f", half[0])
    # First reshape the probe to 3D so it is a single stack of 2D images.
    stack = probe.reshape((-1, *probe.shape[-2:]))
    intensity = cupyx.scipy.ndimage.gaussian_filter(
        input=np.sum(np.square(np.abs(stack)), axis=0),
        sigma=half,
        mode='wrap',
    )
    # Find the maximum intensity in 2D.
    center = np.argmax(intensity)
    # Find the 2D coordinates of the maximum.
    coords = cp.unravel_index(center, dims=probe.shape[-2:])
    # Shift each of the probes so the max is in the center.
    p = np.roll(stack, half[0] - coords[0], axis=0)
    stack = np.roll(p, half[1] - coords[1], axis=1)
    # Reform to the original shape; make contiguous.
    probe = stack.reshape(probe.shape)
    return probe


if __name__ == "__main__":
    cp.random.seed(0)
    x = (cp.random.rand(7, 1, 9, 3, 3) +
         1j * cp.random.rand(7, 1, 9, 3, 3)).astype('complex64')
    x1 = orthogonalize_eig(x)
    assert x1.shape == x.shape, x1.shape

    p = (cp.random.rand(2, 7, 7) * 100).astype(int)
    print(constrain_center_peak(p))
