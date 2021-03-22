__author__ = "Viktor Nikitin, Daniel Ching"
__copyright__ = "Copyright (c) 2021, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'

import tike.linalg


def cost(op, x, dual, penalty, alpha):
    """Minimization functional for regularization problem.

    Parameters
    ----------
    op : operators.Gradient
        The gradient operator.
    x : (L, M, N) array-like
        The object being regularized
    dual : float
        ADMM dual variable.
    penalty : float
        ADMM penalty parameter.
    alpha : float
        Some tuning parameter.
    """
    grad = op.fwd(x)
    cost = alpha * tike.linalg.norm1(grad)
    cost += penalty * tike.linalg.norm(grad - reg + dual / penalty)**2
    return cost


def soft_threshold(op, x, dual, penalty, alpha):
    """Soft thresholding operator for solving something.

    Parameters
    ----------
    op : operators.Gradient
        The gradient operator.
    x : (L, M, N) array-like
        The object being regularized
    dual : float
        ADMM dual variable.
    penalty : float
        ADMM penalty parameter.
    alpha : float
        Some tuning parameter.

    Returns
    -------
    x1 : (L, M, N) array-like
        The updated x.

    """
    z = op.fwd(x) + dual / penalty
    za = op.xp.abs(z)
    return z / za * op.xp.maximum(0, za - alpha / penalty)
