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


def is_converged(algorithm_options):
    """Return True if cost slope is non-negative within the the window.

    Every half-window, look at the slope of the line that fits to the last
    window cost values (average cost values if mini-batch). If this slope is
    non-negative, return True else return False.

    This is a smoothed absolute differences convergence criteria because we are
    considering the difference between consecutive cost values (absolute
    differences) per epoch.
    """
    window = algorithm_options.convergence_window
    if (window >= 2 and len(algorithm_options.costs) >= window
            and len(algorithm_options.costs) % window // 2 == 0):
        m = np.array(algorithm_options.costs[-window:])
        m = np.reshape(m, (len(m), -1))
        m = np.mean(m, axis=1)
        p = np.polyfit(x=range(window), y=m, deg=1, full=False, cov=False)
        if p[0] >= 0:
            logger.info(f"Considering the last {window:d} epochs,"
                        " the cost function seems converged.")
            return True
    return False


def batch_indicies(n, m=1, use_random=True):
    """Return list of indices [0...n) as m groups.

    >>> batch_indicies(10, 3)
    [array([2, 4, 7, 3]), array([1, 8, 9]), array([6, 5, 0])]
    """
    assert 0 < m and m <= n, (m, n)
    i = randomizer.permutation(n) if use_random else np.arange(n)
    return np.array_split(i, m)


def get_batch(x, b, n):
    """Returns x[:, b[n]]; for use with map()."""
    return x[b[n]]


def put_batch(y, x, b, n):
    """Assigns y into x[:, b[n]]; for use with map()."""
    x[b[n]] = y


def momentum(g, v, m, vdecay=None, mdecay=0.9):
    """Add momentum to the gradient direction.

    Parameters
    ----------
    g : vector
        The current gradient.
    m : vector
        The momentum.
    eps : float
        A tiny constant to prevent zero division.
    """
    logger.info("Momentum decay m=%+.3e", mdecay)
    m = 0 if m is None else m
    m = mdecay * m + (1 - mdecay) * g
    return m, None, m


def adagrad(g, v=None, eps=1e-6):
    """Return the adaptive gradient algorithm direction.

    Used to provide a better search direction to stochastic gradient
    descent.

    Parameters
    ----------
    g : vector
        The current gradient.
    v : vector
        The adagrad gradient weights.
    eps : float
        A tiny constant to prevent zero division.

    Returns
    -------
    d : vector
        The new search direction.
    v : vector
        The new gradient weights.

    References
    ----------
    Duchi, John, Elad Hazan, and Yoram Singer. "Adaptive subgradient methods
    for online learning and stochastic optimization." Journal of machine
    learning research 12, no. 7 (2011).
    """
    if v is None:
        return g, (g * g.conj()).real
    v += (g * g.conj()).real
    d = g / np.sqrt(v + eps)
    return d, v


def adadelta(g, d0=None, v=None, m=None, decay=0.9, eps=1e-6):
    """Return the adadelta algorithm direction.

    Used to provide a better search direction to stochastic gradient
    descent.

    Parameters
    ----------
    g : vector
        The current gradient.
    d0: vector
        The previous search direction.
    v : vector
        The adadelta gradient weights.
    m : vector
        The adadelta direction weights.
    eps : float
        A tiny constant to prevent zero division.

    Returns
    -------
    d : vector
        The new search direction.
    v : vector
        The new gradient weights.

    References
    ----------
    Zeiler, Matthew D. "Adadelta: an adaptive learning rate method." arXiv
    preprint arXiv:1212.5701 (2012).
    """
    v = 0 if v is None else v
    m = 0 if m is None else m
    d0 = 0 if d0 is None else d0
    v = v * decay + (1 - decay) * (g * g.conj()).real
    m = m * decay + (1 - decay) * (d0 * d0.conj()).real
    d = np.sqrt((m + eps) / (v + eps)) * g
    return d, v, m


def adam(g, v=None, m=None, vdecay=0.999, mdecay=0.9, eps=1e-8):
    """Return the adaptive moment estimation direction.

    Used to provide a better search direction to stochastic gradient
    descent.

    Parameters
    ----------
    g : vector
        The current search direction.
    v : vector
        Second moment estimate.
    m : vector
        First moment estimate.
    vdecay, mdecay : float [0, 1)
        A factor which determines how quickly information from previous steps
        decays.
    eps : float
        A tiny constant to prevent zero division.

    Returns
    -------
    d : vector
        The new search direction.
    v : vector
        The new gradient weights.
    m : vector
        The new momentum weights.

    References
    ----------
    Kingma, Diederik P., and Jimmy Ba. "Adam: A method for stochastic
    optimization." arXiv preprint arXiv:1412.6980 (2014).
    """
    logger.info("ADAM decay m=%+.3e, v=%+.3e; eps=%+.3e", mdecay, vdecay, eps)
    v = 0 if v is None else v
    m = 0 if m is None else m
    m = mdecay * m + (1 - mdecay) * g
    v = vdecay * v + (1 - vdecay) * (g * g.conj()).real
    m_ = m / (1 - mdecay)
    v_ = np.sqrt(v / (1 - vdecay))
    return m_ / (v_ + eps), v, m


def line_search(
    f,
    x,
    d,
    update_multi,
    step_length=1,
    step_shrink=0.5,
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
    step_count = 0
    first_step = step_length
    while True:
        xsd = update_multi(x, step_length, d)
        fxsd = f(xsd)
        if fxsd <= fx + step_shrink * m:
            break
        step_length *= step_shrink
        if step_length < 1e-32:
            warnings.warn("Line search failed for conjugate gradient.")
            step_length, fxsd, xsd = 0, fx, x
            break
        step_count += 1

    logger.info("line_search: %d backtracks; %.3e -> %.3e; cost %.6e",
                step_count, first_step, step_length, fxsd)

    return step_length, fxsd, xsd


def direction_dy(xp, grad1, grad0=None, dir_=None):
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
    if dir_ is None:
        return [-grad1[0]]

    return [
        - grad1[0]
        + dir_[0] * xp.linalg.norm(grad1[0].ravel())**2
        / (xp.sum(dir_[0].conj() * (grad1[0] - grad0[0])) + 1e-32)
    ]  # yapf: disable


def update_single(x, step_length, d):
    return x + step_length * d


def dir_single(x):
    return x


def conjugate_gradient(
    array_module,
    x,
    cost_function,
    grad,
    direction_dy=direction_dy,
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
        The search direction in all GPUs.
    update_multi : func(x) -> list_of_array
        The updated subimages in all GPUs.
    num_iter : int
        The number of steps to take.
    num_search : int
        The number of during which to perform line search.
    step_length : float
        The initial multiplier of the search direction.
    cost : float
        The current loss function estimate.

    """
    num_search = num_iter if num_search is None else num_search

    for i in range(num_iter):

        grad1 = grad(x)
        if i == 0:
            dir_ = direction_dy(array_module, grad1)
        else:
            dir_ = direction_dy(array_module, grad1, grad0, dir_)
        grad0 = grad1

        dir_list = dir_multi(dir_)

        if i < num_search:
            step_length, cost, x = line_search(
                f=cost_function,
                x=x,
                d=dir_list,
                update_multi=update_multi,
                step_length=step_length,
                cost=cost,
            )
        else:
            x = update_multi(x, step_length, dir_list)
            logger.info("Blind update; length %.3e", step_length)

    if __debug__ and num_search < num_iter:
        cost = cost_function(x)

    return x, cost
