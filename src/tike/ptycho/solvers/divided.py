import logging

import cupy as cp
import numpy as np

from tike.opt import conjugate_gradient, line_search, direction_dy
from ..position import update_positions_pd

logger = logging.getLogger(__name__)


def _batch_indicies(n, m, use_random=False):
    """Return list of indices [0...n) as groups of at most m indices.

    >>> _random_subset(10, 4)
    [array([2, 4, 7, 3]), array([1, 8, 9]), array([6, 5, 0])]

    """
    i = np.random.default_rng().permutation(n) if use_random else np.arange(n)
    return np.array_split(i, (n + m - 1) // m)

def divided(
    op, pool,
    data, probe, scan, psi,
    recover_psi=True, recover_probe=False, recover_positions=False,
    cg_iter=4,
    batch_size=40,
    cost=None,
    subset_is_random=True,
):  # yapf: disable
    """Solve near- and farfield- ptychography problems separately.

    References
    ----------
    Michal Odstrcil, Andreas Menzel, and Manuel Guizar-Sicaros. Iteraive
    least-squares solver for generalized maximum-likelihood ptychography.
    Optics Express. 2018.

    """
    xp = op.xp
    probe = probe[0]
    psi = psi[0]

    # Divide the scan positions into smaller batches to be processed
    # sequentially. Otherwise we run out memory processing all of
    # the diffraction patterns at the same time.
    for index in _batch_indicies(data[0].shape[1], batch_size,
                                 subset_is_random):
        data_ = data[0][:, index]
        scan_ = scan[0][:, index]

        # Compute the diffraction patterns for all of the probe modes at once.
        # We need access to all of the modes of a position to solve the phase
        # problem. The Ptycho operator doesn't do this natively, so it's messy.
        patches = cp.zeros(data_.shape, dtype='complex64')
        patches = op.diffraction._patch(
            patches=patches,
            psi=psi,
            scan=scan_,
            fwd=True,
        )
        patches = patches.reshape(op.ntheta, scan_.shape[-2] // op.fly, op.fly,
                                  1, op.detector_shape, op.detector_shape)
        patches = op.xp.tile(patches, reps=(1, 1, 1, probe.shape[-3], 1, 1))

        pad, end = op.diffraction.pad, op.diffraction.end
        nearplane = patches.copy()
        nearplane[..., pad:end, pad:end] *= probe

        # Solve the farplane phase problem
        farplane = op.propagation.fwd(nearplane, overwrite=False)
        farplane, cost = update_phase(op, data_, farplane, num_iter=cg_iter)

        # Use χ (chi) to solve the nearplane problem. We use least-squares to
        # find the update of all the search directions: object, probe,
        # positions, etc that causes the nearplane wavefront to match the that
        # we just found by solving the phase problem.
        chi = op.propagation.adj(farplane, overwrite=True) - nearplane

        lstsq_shape = (*nearplane.shape[:-2],
                       nearplane.shape[-2] * nearplane.shape[-1] * 2)
        updates = []

        logger.info('%10s cost is %+12.5e', 'nearplane',
                    cp.linalg.norm(cp.ravel(chi)))

        if recover_psi:
            # FIXME: Implement conjugate gradient
            grad_psi = chi.copy()
            grad_psi[..., pad:end, pad:end] *= cp.conj(probe)

            # FIXME: What to do when elements of this norm are zero?
            norm_probe = cp.ones_like(psi)
            dir_psi = cp.zeros_like(psi)

            for m in range(probe.shape[-3]):
                # FIXME: Shape changes required for fly scans.
                intensity = cp.ones(
                    (*scan_.shape[:2], 1, 1, 1, 1),
                    dtype='complex64',
                ) * cp.square(cp.abs(probe[..., m:m + 1, :, :]))
                norm_probe = op.diffraction._patch(
                    patches=intensity,
                    psi=norm_probe,
                    scan=scan_,
                    fwd=False,
                )
                dir_psi = op.diffraction._patch(
                    patches=grad_psi[..., m:m + 1, :, :],
                    psi=dir_psi,
                    scan=scan_,
                    fwd=False,
                )

            dir_psi /= norm_probe

            dOP = cp.zeros((*scan_.shape[:2], *data_.shape[-2:]),
                           dtype='complex64')
            dOP = op.diffraction._patch(
                patches=dOP,
                psi=dir_psi,
                scan=scan_,
                fwd=True,
            )
            dOP = dOP.reshape(op.ntheta, scan_.shape[-2] // op.fly, op.fly, 1,
                              op.detector_shape, op.detector_shape)
            dOP = op.xp.tile(dOP, reps=(1, 1, 1, probe.shape[-3], 1, 1))
            dOP[..., pad:end, pad:end] *= probe

            updates.append(dOP.view('float32').reshape(lstsq_shape))

        if recover_probe:
            grad_probe = (chi * xp.conj(patches))[..., pad:end, pad:end]
            dir_probe = cp.sum(
                grad_probe,
                axis=(1, 2),
                keepdims=True,
            ) / cp.sum(
                cp.square(cp.abs(patches[..., pad:end, pad:end])),
                axis=(1, 2),
            )

            dPO = patches.copy()
            dPO[..., pad:end, pad:end] *= dir_probe

            updates.append(dPO.view('float32').reshape(lstsq_shape))

        # Use least-squares to find the optimal step sizes simultaneously for
        # all search directions.
        if updates:
            A = cp.stack(updates, axis=-1)
            b = chi.view('float32').reshape(lstsq_shape)
            steps = _lstsq(A, b)
            num_steps = 0
        d = 0

        # Update each direction
        if recover_psi:
            step = steps[..., num_steps, None, None]
            # logger.info('%10s step is %+12.5e', 'object', step)
            num_steps += 1

            weighted_step = cp.zeros_like(psi)
            for m in range(probe.shape[-3]):
                # FIXME: Shape changes required for fly scans.
                intensity = cp.ones(
                    (*scan_.shape[:2], 1, 1, 1, 1),
                    dtype='complex64',
                ) * cp.square(cp.abs(probe[..., m:m + 1, :, :]))
                weighted_step = op.diffraction._patch(
                    patches=step * intensity,
                    psi=weighted_step,
                    scan=scan_,
                    fwd=False,
                )

            psi += dir_psi * weighted_step / norm_probe
            d += step * dOP

        if recover_probe:
            step = steps[..., num_steps, None, None]
            num_steps += 1

            weighted_step = cp.sum(
                step * cp.square(cp.abs(patches[..., pad:end, pad:end])),
                axis=(1, 2),
            )

            # FIXME: What to do when elements of this norm are zero?
            norm_psi = cp.sum(
                cp.square(cp.abs(patches[..., pad:end, pad:end])),
                axis=(1, 2),
            ) + 1

            probe += dir_probe * weighted_step / norm_psi
            d += step * dPO

    logger.info('%10s cost is %+12.5e', 'nearplane',
                cp.linalg.norm(cp.ravel(chi - d)))

    return {
        'psi': [psi],
        'probe': [probe],
        'cost': cost,
        'scan': scan,
    }


def update_phase(op, data, farplane, num_iter=1):
    """Solve the farplane phase problem."""
    xp = op.xp

    def grad(farplane):
        intensity = xp.sum(xp.square(xp.abs(farplane)), axis=(2, 3))
        return op.propagation.grad(data, farplane, intensity)

    def cost_function(farplane):
        intensity = xp.sum(xp.square(xp.abs(farplane)), axis=(2, 3))
        return op.propagation.cost(data, intensity)

    logger.info('%10s cost is %+12.5e', 'farplane', cost_function(farplane))

    farplane, cost = conjugate_gradient(
        xp,
        x=farplane,
        cost_function=cost_function,
        grad=grad,
        num_iter=num_iter,
        step_length=2,
    )

    def step(intensity, step0):
        """Analytic step size for Poisson noise model"""
        ξ = 1 - data / (intensity + 1e-16)
        return xp.sum(
            intensity - ξ * (data / (1 - step0 * ξ)),
            axis=(-1, -2),
        ) / xp.sum(
            ξ * ξ * intensity,
            axis=(-1, -2),
        )

    # print cost function for sanity check
    logger.info('%10s cost is %+12.5e', 'farplane', cost)
    return farplane, cost


def update_probe(
    op,
    nearplane,
    probe,
    scan,
    psi,
    num_iter=1,
    position=1,
    eps=1e-16,
    pixels=(-1, -2),
):
    """Solve the nearplane single probe recovery problem."""
    xp = op.xp
    pad, end = op.diffraction.pad, op.diffraction.end
    obj_patches = op.diffraction._patch(
        patches=xp.zeros(
            shape=(*scan.shape[:2], 1, 1, *probe.shape[-2:]),
            dtype='complex64',
        ),
        psi=psi,
        scan=scan,
        fwd=True,
    )

    norm_patches = xp.sum(
        xp.square(xp.abs(obj_patches)),
        axis=position,
        keepdims=True,
    ) + eps

    def cost_function(probe):
        return xp.linalg.norm(
            xp.ravel(probe * obj_patches - nearplane[..., pad:end, pad:end]))**2

    def chi(probe):
        return nearplane[..., pad:end, pad:end] - probe * obj_patches

    def grad(probe):
        return chi(probe) * xp.conj(obj_patches)

    def common_dir_(dir_):
        return xp.sum(dir_, axis=position, keepdims=True) / norm_patches

    def step(probe, dir_):
        return xp.sum(
            # overflow here
            xp.real(chi(probe) * xp.conj(dir_ * obj_patches)),
            axis=pixels,
            keepdims=True,
        ) / (xp.sum(
            xp.square(xp.abs(dir_ * obj_patches)),
            axis=pixels,
            keepdims=True,
        ) + eps)

    for i in range(num_iter):
        grad1 = grad(probe)
        if i == 0:
            dir_ = -grad1
        else:
            dir_ = direction_dy(xp, grad0, grad1, dir_)
        grad0 = grad1

        weighted_patches = xp.sum(
            step(probe, dir_) * xp.square(xp.abs(obj_patches)),
            axis=position,
            keepdims=True,
        )

        probe = probe + common_dir_(dir_) * weighted_patches / norm_patches

    cost = cost_function(probe)
    logger.info('%10s cost is %+12.5e', 'probe', cost)
    return probe, cost


def update_object(
    op,
    nearplane,
    probe,
    scan,
    psi,
    num_iter=1,
    position=1,
    eps=1e-16,
    pixels=(-1, -2),
):
    """Solve the nearplane object recovery problem."""
    xp = op.xp
    pad, end = op.diffraction.pad, op.diffraction.end

    def cost_function(psi):
        return xp.linalg.norm(
            xp.ravel(
                op.diffraction.fwd(psi=psi, scan=scan, probe=probe) -
                nearplane))**2

    def chi(psi):
        return (nearplane -
                op.diffraction.fwd(psi=psi, scan=scan, probe=probe))[...,
                                                                     pad:end,
                                                                     pad:end]

    def grad(psi):
        return chi(psi) * xp.conj(probe)

    norm_probe = op.diffraction._patch(
        patches=xp.square(xp.abs(probe)) * xp.ones(
            (*scan.shape[:2], 1, 1, 1, 1),
            dtype='complex64',
        ),
        psi=xp.zeros_like(psi),
        scan=scan,
        fwd=False,
    ) + eps

    def common_dir_(dir_):
        return op.diffraction._patch(
            patches=dir_,
            scan=scan,
            psi=xp.zeros_like(psi),
            fwd=False,
        ) / norm_probe

    def step(psi, dir_):
        # TODO: Figure out if steps should be complex instead of real
        result = xp.sum(
            xp.real(chi(psi) * xp.conj(dir_ * probe)),
            axis=pixels,
            keepdims=True,
        ) / (xp.sum(
            xp.square(xp.abs(dir_ * probe)),
            axis=pixels,
            keepdims=True,
        ) + eps)
        return result

    for i in range(num_iter):
        grad1 = grad(psi)
        if i == 0:
            dir_ = -grad1
        else:
            dir_ = direction_dy(xp, grad0, grad1, dir_)
        grad0 = grad1

        weight_probe = op.diffraction._patch(
            patches=(step(psi, dir_) *
                     xp.square(xp.abs(probe))).astype('complex64'),
            psi=xp.zeros_like(psi),
            scan=scan,
            fwd=False,
        )
        psi = psi + common_dir_(dir_) * weight_probe / norm_probe

    cost = cost_function(psi)
    logger.info('%10s cost is %+12.5e', 'object', cost)
    return psi, cost


def _lstsq(a, b):
    """Return the least-squares solution for a @ x = b.

    This implementation, unlike cp.linalg.lstsq, allows a stack of matricies to
    be processed simultaneously. The input sizes of the matricies are as
    follows:
        a (..., M, N)
        b (..., M)
        x (...,    N)

    ...seealso:: https://github.com/numpy/numpy/issues/8720
                 https://github.com/cupy/cupy/issues/3062
    """
    assert a.shape[:-1] == b.shape, (f"Leading dims of a {a.shape}"
                                     f"and b {b.shape} must be same!")
    shape = a.shape[:-2]
    a = a.reshape(-1, *a.shape[-2:])
    b = b.reshape(-1, *b.shape[-1:], 1)
    x = cp.empty((a.shape[0], a.shape[-1], 1), dtype=a.dtype)
    for i in range(a.shape[0]):
        x[i], _, _, _ = cp.linalg.lstsq(a[i], b[i])
    return x.reshape(*shape, a.shape[-1])


if __name__ == "__main__":
    N = (3, 4)

    a = cp.random.rand(*N, 5, 2) + 1j * cp.random.rand(*N, 5, 2)
    b = cp.random.rand(*N, 5) + 1j * cp.random.rand(*N, 5)

    x = _lstsq(a.astype('complex64'), b.astype('complex64'))

    assert x.shape == (*N, 2)
