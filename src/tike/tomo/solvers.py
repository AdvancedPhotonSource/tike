"""This module provides Solver implementations for a variety of algorithms."""

from tike.opt import conjugate_gradient
from tike.tomo import TomoBackend

__all__ = [
    "available_solvers",
    "ConjugateGradientTomoSolver",
]


class ConjugateGradientTomoSolver(TomoBackend):
    """Solve the ptychography problem using gradient descent."""

    def run(self, tomo, obj, theta, num_iter, K=1, rho=1, tau=0, **kwargs):
        """Use conjugate gradient to estimate `obj`.

        Parameters
        ----------
        tomo: array-like float32
            Line integrals through the object.
        obj : array-like float32
            The object to be recovered.
        num_iter : int
            Number of steps to take.
        K, rho : complex64
            Some constants

        """
        xp = self.array_module

        def cost_function(obj):
            model = self.fwd(obj=obj, theta=theta)
            return xp.square(xp.linalg.norm(model - tomo))

        def grad(obj):
            model = self.fwd(obj, theta=theta)
            return self.adj(model - tomo, theta=theta)

        obj = conjugate_gradient(
            self.array_module,
            x=obj,
            cost_function=cost_function,
            grad=grad,
            num_iter=num_iter,
        )

        return {
            'obj': obj
        }

# TODO: Add new algorithms here
available_solvers = {
    "cgrad": ConjugateGradientTomoSolver,
}
