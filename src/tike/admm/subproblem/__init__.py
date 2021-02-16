"""Implements subproblem formulations for ADMM.

Each ADMM subproblem is implemented in a separate function such that the
problems are consistently implemented across different ADMM compositions.

"""


def update_penalty(comm, g, h, h0, rho, diff=4):
    """Increase rho when L2 error between g and h becomes too large.

    If rho is the penalty parameter associated with the constraint norm(y - x),
    then rho is increased when

        norm(g - h) > diff * rho^2 * norm(h - h0)

    and decreased when

        norm(g - h) * diff < rho^2 * norm(h - h0)

    """
    r = np.linalg.norm(g - h)**2
    s = rho * rho * np.linalg.norm(h - h0)**2
    r, s = [np.sum(comm.gather(x)) for x in ([r], [s])]
    if comm.rank == 0:
        if (r > diff * s):
            rho *= 2
        elif (r * diff < s):
            rho *= 0.5
    rho = comm.broadcast(rho)
    logging.info(f"Update penalty parameter ρ = {rho}.")
    return rho


from .align import *
from .lamino import *
from .ptycho import *
