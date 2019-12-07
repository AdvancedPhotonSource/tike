"""This module provides Solver implementations for a variety of algorithms."""

from tike.tomo import TomoBackend

__all__ = [
    "available_solvers",
    "ConjugateGradientTomoSolver",
]


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

    References
    ----------
    https://en.wikipedia.org/wiki/Backtracking_line_search

    """
    assert step_shrink > 0 and step_shrink < 1
    m = 0  # Some tuning parameter for termination
    fx = f(x)  # Save the result of f(x) instead of computing it many times
    # Decrease the step length while the step increases the cost function
    while f(x + step_length * d) > fx + step_shrink * m:
        if step_length < 1e-32:
            warnings.warn("Line search failed for conjugate gradient.")
            return 0
        step_length *= step_shrink
    return step_length


class ConjugateGradientTomoSolver(TomoBackend):
    """Solve the ptychography problem using gradient descent."""

    def run(self, tomo, obj, theta, num_iter, **kwargs):
        """Use conjugate gradient to estimate `x`.

        Parameters
        ----------
        x : array-like
            The object to be recovered.
        num_iter : int
            Number of steps to take.
        dir_ : array-like
            The initial search direction.

        """
        xp = self.array_module

        def cost_function(model):
            return xp.square(xp.linalg.norm(model - tomo))

        def get_grad(model):
            return self.adj(model - tomo, theta) / (self.ntheta * self.n * 0.5)

        for i in range(num_iter):
            model = self.fwd(obj, theta)
            grad = get_grad(model)
            if i == 0:
                dir_ = -grad
            else:
                dir_ = (
                    -grad
                    + dir_ * xp.square(xp.linalg.norm(grad))
                    / (xp.sum(xp.conj(dir_) * (grad - grad0))
                       + 1e-32)
                )  # yapf: disable
            grad0 = grad
            gamma = line_search(
                f=cost_function,
                x=model,
                d=self.fwd(dir_, theta),
            )
            obj = obj + gamma * dir_
            # check convergence
            if (i + 1) % 8 == 0:
                print("%4d, %.3e, 0, %.7e" % (
                    (i + 1), gamma,
                    cost_function(model),
                ))  # yapf: disable

        return {
            'obj': obj
        }

# TODO: Add new algorithms here
available_solvers = {
    "cgrad": ConjugateGradientTomoSolver,
}
