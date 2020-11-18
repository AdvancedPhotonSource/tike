"""Generic implementations of optimization routines.

Generic implementations of optimization algorithm such as conjugate gradient and
line search that can be reused between domain specific modules. In, the future,
this module may be replaced by Operator Discretization Library (ODL) solvers
library.

"""

import logging
import warnings

import numpy as np

logger = logging.getLogger(__name__)
randomizer = np.random.default_rng()


def line_search(
    f,
    x,
    d,
    update_multi,
    step_length=1,
    step_shrink=0.1,
    cost=None,
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
    cost : float
        f(x) if it is already known.

    Returns
    -------
    step_length : float
        The optimal step length along d.
    cost : float
        The new value of the cost function after stepping along d.
    x : float
        The new value of x after stepping along d.

    References
    ----------
    https://en.wikipedia.org/wiki/Backtracking_line_search

    """
    assert step_shrink > 0 and step_shrink < 1
    m = 0  # Some tuning parameter for termination
    # Save the result of f(x) instead of computing it many times
    fx = f(x) if cost is None else cost
    # Decrease the step length while the step increases the cost function
    while True:
        xsd = update_multi(x, step_length, d)
        fxsd = f(xsd)
        if fxsd <= fx + step_shrink * m:
            break
        step_length *= step_shrink
        if step_length < 1e-32:
            warnings.warn("Line search failed for conjugate gradient.")
            return 0, fx, x
    return step_length, fxsd, xsd


def direction_dy(xp, grad0, grad1, dir_):
    """Return the Dai-Yuan search direction.

    Parameters
    ----------
    grad0 : array_like
        The gradient from the previous step.
    grad1 : array_like
        The gradient from this step.
    dir_ : array_like
        The previous search direction.

    """
    return (
        - grad1
        + dir_ * xp.linalg.norm(grad1.ravel())**2
        / (xp.sum(dir_.conj() * (grad1 - grad0)) + 1e-32)
    )  # yapf: disable


def update_single(x, step_length, d):
    return x + step_length * d


def dir_single(x):
    return x


def conjugate_gradient(
    array_module,
    x,
    cost_function,
    grad,
    dir_multi=dir_single,
    update_multi=update_single,
    num_iter=1,
    step_length=1,
    num_search=None,
    cost=None,
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
        The dir_ in all GPUs.
    update_multi : func(x) -> list_of_array
        The updated subimages in all GPUs.
    num_iter : int
        The number of steps to take.
    num_search : int
        The number of during which to perform line search.
    step_length : float
        The initial multiplier of the search direction.
    """
    num_search = num_iter if num_search is None else num_search
    blind_update = False

    for i in range(num_iter):

        grad1 = grad(x)
        if i == 0:
            dir_ = -grad1
        else:
            dir_ = direction_dy(array_module, grad0, grad1, dir_)
        grad0 = grad1

        dir_list = dir_multi(dir_)

        if i < num_search:
            gamma, cost, x = line_search(
                f=cost_function,
                x=x,
                d=dir_list,
                update_multi=update_multi,
                step_length=step_length,
                cost=cost,
            )

            logger.debug("step %d; length %.3e -> %.3e; cost %.6e", i,
                         step_length, gamma, cost)
            step_length = gamma
        else:
            x = update_multi(x, step_length, dir_list)
            blind_update = True

    if __debug__ and blind_update:
        cost = cost_function(x)

    return x, cost, step_length
