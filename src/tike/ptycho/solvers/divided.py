import logging

import numpy as np

from tike.opt import conjugate_gradient, line_search, direction_dy
from ..position import update_positions_pd

logger = logging.getLogger(__name__)


def random_subset(n, m):
    """Yield indices [0...n) as groups of at most m indices."""
    rng = np.random.default_rng()
    i = np.arange(n)
    rng.shuffle(i)
    for s in np.array_split(i, (n + m - 1) // m):
        yield s


def divided(
    op, pool,
    data, probe, scan, psi,
    recover_psi=True, recover_probe=False, recover_positions=False,
    cg_iter=4,
    batch_size=40,
    **kwargs
):  # yapf: disable
    """Solve near- and farfield- ptychography problems separately.

    References
    ----------
    Michal Odstrcil, Andreas Menzel, and Manuel Guizar-Sicaros. Iteraive
    least-squares solver for generalized maximum-likelihood ptychography.
    Optics Express. 2018.

    """
    probe = probe[0]
    psi = psi[0]

    for lo in range(0, data[0].shape[1], batch_size):
        hi = min(data[0].shape[1], lo + batch_size)

        data_ = data[0][:, lo:hi]
        scan_ = scan[0][:, lo:hi]

        # Compute the diffraction patterns for all of the modes at once.
        # The Ptycho operator doesn't do this natively, so it's messy.
        patches = op.xp.zeros((*scan_.shape[:2], *data_.shape[-2:]),
                          dtype='complex64')
        patches = op.diffraction._patch(patches=patches,
                                        psi=psi,
                                            scan=scan_,
                                        fwd=True)
        patches = patches.reshape(op.ntheta, scan_.shape[-2] // op.fly, op.fly, 1,
                              op.detector_shape, op.detector_shape)
        nearplane = op.xp.tile(patches, reps=(1, 1, 1, probe.shape[-3], 1, 1))
        pad, end = op.diffraction.pad, op.diffraction.end
        nearplane[..., pad:end, pad:end] *= probe
        farplane = op.propagation.fwd(nearplane, overwrite=True)
        farplane, cost = update_phase(op, data_, farplane, num_iter=cg_iter)
        nearplane = op.propagation.adj(farplane, overwrite=True)

        # TODO: Could move this loop over modes into update_object and
        # update_probe to reuse cached values.
        for mode in range(probe.shape[-3]):
            end = mode + 1

            if recover_psi:
                psi, cost = update_object(
                    op,
                    nearplane[..., mode:end, :, :],
                    probe[..., mode:end, :, :],
                    scan_,
                    psi,
                    num_iter=cg_iter,
                )

            if recover_probe:
                probe[..., mode:end, :, :], cost = update_probe(
                    op,
                    nearplane[..., mode:end, :, :],
                    probe[..., mode:end, :, :],
                    scan_,
                    psi,
                    num_iter=cg_iter,
                )

    # if recover_positions:
    #     scan, cost = update_positions_pd(op, data, psi, probe, scan)

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
