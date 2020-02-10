"""Define generic implementations of optimization routines.

This optimization library contains implementations of optimization routies such
as conjusate gradient that can be reused between domain specific modules. In,
the future, this module may be replaced by Operator Discretization Library (ODL)
solvers library.
"""

import warnings


def line_search(f, x, d, step_length=1, step_shrink=0.5):
    """Return a new `step_length` using a backtracking line search.

    Parameters
    ----------
    f : function(x)
        The function being optimized.
    x : vector
        The current position.
    d : vector
        The search direction.

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
        fxsd = f(x + step_length * d)
        if fxsd <= fx + step_shrink * m:
            break
        step_length *= step_shrink
        if step_length < 1e-32:
            warnings.warn("Line search failed for conjugate gradient.")
            return 0, fx
    return step_length, fxsd


def conjugate_gradient(
        array_module,
        x,
        cost_function,
        grad,
        num_iter=1,
        dir_=None,
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
    num_iter : int
        The number of steps to take.
    dir_ : array-like
        The initial search direction.

    """
    xp = array_module

    for i in range(num_iter):
        grad_ = grad(x)
        if dir_ is None:
            dir_ = -grad_
        else:
            dir_ = (
                -grad_
                + dir_ * xp.square(xp.linalg.norm(grad_))
                / (xp.sum(xp.conj(dir_) * (grad_ - grad0))
                   + 1e-32)
            )  # yapf: disable
        grad0 = grad_
        gamma, cost = line_search(
            f=cost_function,
            x=x,
            d=dir_,
        )
        x = x + gamma * dir_
        # check convergence
        if (i + 1) % 8 == 0:
            print("%4d, %.3e, 0, %.7e" % (
                (i + 1), gamma,
                cost,
            ))  # yapf: disable

    return x, cost
