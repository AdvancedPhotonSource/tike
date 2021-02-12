import logging

import dxchange
import numpy as np

import tike.lamino
from .admm import update_penalty

logger = logging.getLogger(__name__)


def subproblem(
    # constants
    comm,
    phi,
    theta,
    tilt,
    # updated
    u,
    λ_l,
    ρ_l,
    Hu0=None,
    # parameters
    num_iter=1,
    cg_iter=1,
    folder=None,
    save_result=False,
):
    """Solver the laminography subproblem.

    Parameters
    ----------
    phi
        Exponentiated projections through the object, u.
    u
        Refractive indices of the object.
    theta
        Rotation angle of each projection, phi
    tilt
        The off-rotation axis angle of laminography

    """
    logger.info('Solve the laminography problem.')

    save_result = False if folder is None else save_result

    # Gather all to one process
    λ_l, phi, theta = [comm.gather(x) for x in (λ_l, phi, theta)]

    cost = None
    if comm.rank == 0:
        if save_result:
            # We cannot reorder phi, theta without ruining correspondence
            # with data, psi, etc, but we can reorder the saved array
            order = np.argsort(theta)
            dxchange.write_tiff(
                np.angle(phi[order]),
                f'{folder}/phi-angle-{save_result:03d}.tiff',
                dtype='float32',
            )
            dxchange.write_tiff(
                np.abs(phi[order]),
                f'{folder}/phi-abs-{save_result:03d}.tiff',
                dtype='float32',
            )

        lresult = tike.lamino.reconstruct(
            data=-1j * np.log(phi + λ_l / ρ_l),
            theta=theta,
            tilt=tilt,
            obj=u,
            algorithm='cgrad',
            num_iter=num_iter,
            cg_iter=cg_iter,
            num_gpu=comm.size,
        )
        u = lresult['obj']
        cost = lresult['cost']

    # Separate again to multiple processes
    λ_l, phi, theta = [comm.scatter(x) for x in (λ_l, phi, theta)]
    # u = comm.broadcast(u)  # volume too large to fit in MPI buffer

    Hu = np.exp(1j * tike.lamino.simulate(
        obj=u,
        tilt=tilt,
        theta=theta,
    ))
    φHu = phi - Hu

    logger.info('Update laminography lambdas and rhos.')

    λ_l += ρ_l * φHu

    if Hu0 is not None:
        ρ_l = update_penalty(comm, phi, Hu, Hu0, ρ_l)

    Hu0 = Hu

    if comm.rank == 0 and save_result:
        dxchange.write_tiff(
            u.real,
            f'{folder}/particle-real-{save_result:03d}.tiff',
            dtype='float32',
        )
        dxchange.write_tiff(
            u.imag,
            f'{folder}/particle-imag-{save_result:03d}.tiff',
            dtype='float32',
        )
        dxchange.write_tiff(
            np.angle(Hu),
            f'{folder}/Hu-angle-{save_result:03d}.tiff',
            dtype='float32',
        )
        dxchange.write_tiff(
            np.abs(Hu),
            f'{folder}/Hu-abs-{save_result:03d}.tiff',
            dtype='float32',
        )

    return (
        u,
        λ_l,
        ρ_l,
        Hu0,
        cost,
    )
