import logging

import dxchange
import numpy as np

import tike.lamino
from . import update_penalty

logger = logging.getLogger(__name__)


def lamino(
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

    cost, Hu = None, None
    if comm.rank == 0:
        if save_result:
            # We cannot reorder phi, theta without ruining correspondence
            # with data, psi, etc, but we can reorder the saved array
            order = np.argsort(theta)
            dxchange.write_tiff(
                np.angle(phi[order]),
                f'{folder}/phi-angle-{save_result:03d}.tiff',
                dtype='float32',
                overwrite=True,
            )
            dxchange.write_tiff(
                np.abs(phi[order]),
                f'{folder}/phi-abs-{save_result:03d}.tiff',
                dtype='float32',
                overwrite=True,
            )

        lresult = tike.lamino.reconstruct(
            data=-1j * np.log(phi + λ_l / ρ_l),
            theta=theta,
            tilt=tilt,
            obj=u,
            algorithm='cgrad',
            num_iter=num_iter,
            cg_iter=cg_iter,
            # FIXME: Communications overhead makes 1 GPU faster than 8.
            num_gpu=1,  # comm.size,
        )
        u = lresult['obj']
        cost = lresult['cost'][-1]

        # FIXME: volume becomes too large to fit in MPI buffer.
        # Used to broadcast u, now broadcast only Hu
        # u = comm.broadcast(u)
        Hu = np.exp(1j * tike.lamino.simulate(
            obj=u,
            tilt=tilt,
            theta=theta,
        ))

    # Separate again to multiple processes
    λ_l, phi, theta, Hu = [comm.scatter(x) for x in (λ_l, phi, theta, Hu)]

    logger.info('Update laminography lambdas and rhos.')

    λ_l += ρ_l * (phi - Hu)

    if Hu0 is not None:
        ρ_l = update_penalty(comm, phi, Hu, Hu0, ρ_l)

    Hu0 = Hu

    if comm.rank == 0 and save_result:
        dxchange.write_tiff(
            u.real,
            f'{folder}/particle-real-{save_result:03d}.tiff',
            dtype='float32',
            overwrite=True,
        )
        dxchange.write_tiff(
            u.imag,
            f'{folder}/particle-imag-{save_result:03d}.tiff',
            dtype='float32',
            overwrite=True,
        )
        dxchange.write_tiff(
            np.angle(Hu),
            f'{folder}/Hu-angle-{save_result:03d}.tiff',
            dtype='float32',
            overwrite=True,
        )
        dxchange.write_tiff(
            np.abs(Hu),
            f'{folder}/Hu-abs-{save_result:03d}.tiff',
            dtype='float32',
            overwrite=True,
        )

    return (
        u,
        λ_l,
        ρ_l,
        Hu0,
        cost,
    )
