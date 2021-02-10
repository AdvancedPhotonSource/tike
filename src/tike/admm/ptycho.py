import logging

import tike.ptycho

logger = logging.getLogger(__name__)


def subproblem(
    # constants
    data,
    λ_p,
    ρ_p,
    Aφ,
    # updated
    presult,
    # parameters
    cg_iter=1,
    folder=None,
    save_result=False,
):
    """Solve the ptychography subsproblem.

    Parameters
    ----------

    """
    logger.info("Solve the ptychography problem.")

    presult = tike.ptycho.reconstruct(
        data=data,
        reg=λ_p / ρ_p - Aφ,
        rho=ρ_p,
        algorithm='cgrad',
        num_iter=1,
        cg_iter=cg_iter,
        recover_psi=True,
        recover_probe=True,
        recover_positions=False,
        model='gaussian',
        **presult,
    )

    logger.info("No update for ptychography lambdas and rhos")

    return presult
