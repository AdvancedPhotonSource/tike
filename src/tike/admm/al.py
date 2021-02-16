import logging

import numpy as np
import cupy as cp

import tike.admm.alignment
import tike.admm.lamino
import tike.admm.ptycho
import tike.communicator

from .admm import print_log_line

logger = logging.getLogger(__name__)


def ptycho__align_lamino(
    data,
    psi,
    scan,
    probe,
    theta,
    tilt,
    angle,
    w,
    flow=None,
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
    Hu = np.ones((len(theta), w, w), dtype='complex64')
    phi = Hu
    Aφ = np.ones(psi.shape, dtype='complex64')

    λ_l = np.zeros([len(theta), w, w], dtype='complex64')
    ρ_l = 0.5

    comm = tike.communicator.MPICommunicator()

    with cp.cuda.Device(comm.rank if comm.size > 1 else None):

        presult, _ = tike.admm.ptycho.subproblem(
            # constants
            comm=comm,
            data=data,
            λ=None,
            ρ=None,
            Aφ=None,
            # updated
            presult=presult,
            # parameters
            num_iter=4 * niter,
            cg_iter=cg_iter,
            folder=folder,
            save_result=niter + 1,
            rescale=True,
        )

        for k in range(1, niter + 1):
            logger.info(f"Start ADMM iteration {k}.")
            save_result = k if k % interval == 0 else False

            (
                phi,
                _,
                _,
                flow,
                shift,
                Aφ,
                align_cost,
            ) = tike.admm.alignment.subproblem(
                # constants
                comm=comm,
                psi=presult['psi'],
                angle=angle,
                Hu=Hu,
                λ_l=λ_l,
                ρ_l=ρ_l,
                # updated
                phi=phi,
                λ_p=None,
                ρ_p=1,
                flow=flow,
                shift=shift,
                Aφ0=None,
                # parameters
                align_method=align_method,
                cg_iter=cg_iter,
                num_iter=4,
                folder=folder,
                save_result=save_result,
            )

            (
                u,
                λ_l,
                ρ_l,
                Hu,
                lamino_cost,
            ) = tike.admm.lamino.subproblem(
                # constants
                comm=comm,
                phi=phi,
                theta=theta,
                tilt=tilt,
                # updated
                u=u,
                λ_l=λ_l,
                ρ_l=ρ_l,
                Hu0=Hu,
                # parameters
                num_iter=4,
                cg_iter=cg_iter,
                folder=folder,
                save_result=save_result,
            )

            # Record metrics for each subproblem
            ψAφ = presult['psi'] - Aφ
            φHu = phi - Hu
            lagrangian = (
                [np.mean(np.real(ψAφ.conj() * ψAφ))],
                [
                    2 * np.mean(np.real(λ_l.conj() * φHu)) +
                    ρ_l * np.mean(np.real(φHu.conj() * φHu))
                ],
                [align_cost],
            )
            lagrangian = [comm.gather(x) for x in lagrangian]

            if comm.rank == 0:
                lagrangian = [np.sum(x) for x in lagrangian]
                print_log_line(
                    k=k,
                    ρ_l=ρ_l,
                    Lagrangian=np.sum(lagrangian[:2]),
                    ψAφ=lagrangian[0],
                    φHu=lagrangian[1],
                    align=lagrangian[2],
                    lamino=float(lamino_cost),
                )
