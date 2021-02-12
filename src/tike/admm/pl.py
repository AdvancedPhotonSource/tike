import logging

import numpy as np
import cupy as cp

import tike.admm.alignment
import tike.admm.lamino
import tike.admm.ptycho
import tike.align
import tike.communicator

from .admm import print_log_line

logger = logging.getLogger(__name__)


def ptycho_lamino(
    data,
    psi,
    scan,
    probe,
    theta,
    tilt,
    angle,
    w,
    flow=False,
    shift=None,
    niter=1,
    interval=8,
    folder=None,
    cg_iter=4,
    align_method=False,
):
    """Solve the joint ptycho-lamino problem using ADMM."""
    presult = {
        'psi': psi,
        'scan': scan,
        'probe': probe,
    }

    u = np.zeros((w, w, w), dtype='complex64')
    Hu = np.ones_like(psi)
    λ_p = np.zeros_like(psi)
    ρ_p = 0.5

    comm = tike.communicator.MPICommunicator()

    with cp.cuda.Device(comm.rank if comm.size > 1 else None):
        for k in range(1, niter + 1):
            logger.info(f"Start ADMM iteration {k}.")
            save_result = k if k % interval == 0 else False

            presult = tike.admm.ptycho.subproblem(
                comm,
                # constants
                data,
                λ=λ_p,
                ρ=ρ_p,
                Aφ=Hu,
                # updated
                presult=presult,
                # parameters
                num_iter=4,
                cg_iter=cg_iter,
                folder=folder,
                save_result=save_result,
                rescale=(k == 1),
            )

            phi = tike.align.invert(
                presult['psi'],
                angle=angle,
                flow=None,
                shift=shift,
                unpadded_shape=(len(theta), w, w),
                cval=1.0,
            )
            Hu = tike.align.invert(
                Hu,
                angle=angle,
                flow=None,
                shift=shift,
                unpadded_shape=(len(theta), w, w),
                cval=1.0,
            )
            λ_p = tike.align.invert(
                λ_p,
                angle=angle,
                flow=None,
                shift=shift,
                unpadded_shape=(len(theta), w, w),
                cval=1.0,
            )

            (
                u,
                λ_p,
                ρ_p,
                Hu,
                lamino_cost,
            ) = tike.admm.lamino.subproblem(
                # constants
                comm,
                phi,
                theta,
                tilt,
                # updated
                u,
                λ_p,
                ρ_p,
                Hu0=Hu,
                # parameters
                num_iter=4,
                cg_iter=cg_iter,
                folder=folder,
                save_result=save_result,
            )

            Hu = tike.align.simulate(
                Hu,
                angle=angle,
                flow=None,
                shift=shift,
                padded_shape=psi.shape,
                cval=1.0,
            )
            λ_p = tike.align.simulate(
                λ_p,
                angle=angle,
                flow=None,
                shift=shift,
                padded_shape=psi.shape,
                cval=1.0,
            )

            # Record metrics for each subproblem
            ψHu = presult['psi'] - Hu
            lagrangian = (
                [presult['cost']],
                [
                    2 * np.real(λ_p.conj() * ψHu) +
                    ρ_p * np.linalg.norm(ψHu.ravel())**2
                ],
            )
            lagrangian = [comm.gather(x) for x in lagrangian]

            if comm.rank == 0:
                lagrangian = [np.sum(x) for x in lagrangian]
                print_log_line(
                    k=k,
                    ρ_p=ρ_p,
                    Lagrangian=np.sum(lagrangian),
                    dGψ=lagrangian[0],
                    ψHu=lagrangian[1],
                    lamino=float(lamino_cost),
                )
