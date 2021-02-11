import logging

import dxchange
import numpy as np

import tike.ptycho

logger = logging.getLogger(__name__)


def subproblem(
    comm,
    # constants
    data,
    λ,
    ρ,
    Aφ,
    # updated
    presult,
    # parameters
    num_iter=1,
    cg_iter=1,
    folder=None,
    save_result=False,
    rescale=False,
):
    """Solve the ptychography subsproblem.

    Parameters
    ----------

    """
    logger.info("Solve the ptychography problem.")

    presult = tike.ptycho.reconstruct(
        data=data,
        reg=None if λ is None else λ / ρ - Aφ,
        rho=ρ,
        algorithm='cgrad',
        num_iter=num_iter,
        cg_iter=cg_iter,
        recover_psi=True,
        recover_probe=True,
        recover_positions=False,
        model='gaussian',
        rescale=rescale,
        **presult,
    )

    logger.info("No update for ptychography lambdas and rhos")

    if comm.rank == 0 and save_result:
        dxchange.write_tiff(
            np.abs(presult['psi']),
            f'{folder}/psi-abs-{save_result:03d}.tiff',
            dtype='float32',
        )
        dxchange.write_tiff(
            np.angle(presult['psi']),
            f'{folder}/psi-angle-{save_result:03d}.tiff',
            dtype='float32',
        )

    return presult
