"""Generic implementations of optimization routines.

Generic implementations of optimization algorithm such as conjugate gradient and
line search that can be reused between domain specific modules. In, the future,
this module may be replaced by Operator Discretization Library (ODL) solvers
library.

"""

import logging
import warnings

logger = logging.getLogger(__name__)


def line_search(
    f,
    x,
    d,
    num_gpu,
    update_multi=None,
    step_length=1,
    step_shrink=0.5,
):
    """Return a new `step_length` using a backtracking line search.

    Parameters
    ----------
    f : function(x)
        The function being optimized.
    x : vector
        The current position.
    d : vector
        The search direction.
    step_length : float
        The initial step_length.
    step_shrink : float
        Decrease the step_length by this fraction at each iteration.

    Returns
    -------
    step_length : float
        The optimal step length along d.
    cost : float
        The new value of the cost function after stepping along d.

    References
    ----------
    https://en.wikipedia.org/wiki/Backtracking_line_search

    """
    assert step_shrink > 0 and step_shrink < 1
    m = 0  # Some tuning parameter for termination
    fx = f(x)  # Save the result of f(x) instead of computing it many times
    # Decrease the step length while the step increases the cost function
    while True:
        if (num_gpu <= 1):
            fxsd = f(x + step_length * d)
        else:
            fxsd = f(update_multi(x, step_length, d))
        if fxsd <= fx + step_shrink * m:
            break
        step_length *= step_shrink
        if step_length < 1e-32:
            warnings.warn("Line search failed for conjugate gradient.")
            return 0, fx
    return step_length, fxsd


def direction_dy(xp, grad0, grad1, dir):
    """Return the Dai-Yuan search direction.

    Parameters
    ----------
    grad0 : array_like
        The gradient from the previous step.
    grad1 : array_like
        The gradient from this step.
    dir : array_like
        The previous search direction.

    """
    return (
        - grad1
        + dir * xp.linalg.norm(grad1.ravel())**2
        / (xp.sum(dir.conj() * (grad1 - grad0)) + 1e-32)
    )  # yapf: disable


def conjugate_gradient(
    array_module,
    x,
    cost_function,
    grad,
    dir_multi=None,
    update_multi=None,
    num_gpu=1,
    num_iter=1,
):
    """Use conjugate gradient to estimate `x`.

    Parameters
    ----------
    array_module : module
        The Python module that will provide array operations.
    x : array_like
        The object to be recovered.
    cost_function : func(x) -> float
        The function being minimized to recover x.
    grad : func(x) -> array_like
        The gradient of cost_function.
    dir_multi : func(x) -> list_of_array
        The dir in all GPUs.
    update_multi : func(x) -> list_of_array
        The updated subimages in all GPUs.
    num_iter : int
        The number of steps to take.

    """
    for i in range(num_iter):
        grad1 = grad(x)
        if i == 0:
            dir = -grad1
        else:
            dir = direction_dy(array_module, grad0, grad1, dir)
        grad0 = grad1
        if (num_gpu <= 1):
            gamma, cost = line_search(
                f=cost_function,
                x=x,
                d=dir,
                num_gpu=num_gpu,
            )
            x = x + gamma * dir
        else:
            dir_list = dir_multi(dir)

            gamma, cost = line_search(
                f=cost_function,
                x=x,
                d=dir_list,
                num_gpu=num_gpu,
                update_multi=update_multi,
            )
            # update the image
            x = update_multi(x, gamma, dir_list)
        logger.debug("%4d, %.3e, %.7e", (i + 1), gamma, cost)
    return x, cost
