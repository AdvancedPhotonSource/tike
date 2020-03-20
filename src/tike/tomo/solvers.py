"""Provides Solver implementations for a variety of algorithms."""

from tike.opt import conjugate_gradient
import tike.reg as tv
from tike.tomo import TomoBackend

__all__ = [
    "available_solvers",
    "ConjugateGradientTomoSolver",
]


class ConjugateGradientTomoSolver(TomoBackend):
    """Solve the ptychography problem using gradient descent."""

    def run(self, tomo, obj, theta, num_iter,
            rho=1.0, tau=0.0, reg=0j, K=1 + 0j, **kwargs
    ):  # yapf: disable
        """Use conjugate gradient to estimate `obj`.

        Parameters
        ----------
        tomo: array-like float32
            Line integrals through the object.
        obj : array-like float32
            The object to be recovered.
        num_iter : int
            Number of steps to take.
        rho, tau : float32
            Weights for data and variation components of the cost function
        reg : complex64
            The regularizer for total variation

        """
        xp = self.array_module
        reg = xp.asarray(reg, dtype='complex64')
        K = xp.asarray(K, dtype='complex64')
        K_conj = xp.conj(K, dtype='complex64')

        def cost_function(obj):
            model = K * self.fwd(obj=obj, theta=theta)
            return (
                + rho * xp.square(xp.linalg.norm(model - tomo))
                + tau * xp.square(xp.linalg.norm(tv.fwd(xp, obj) - reg))
            )

        def grad(obj):
            model = K * self.fwd(obj, theta=theta)
            return (
                + rho * self.adj(K_conj * (model - tomo), theta=theta)
                + tau * tv.adj(xp, tv.fwd(xp, obj) - reg)
            )

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
