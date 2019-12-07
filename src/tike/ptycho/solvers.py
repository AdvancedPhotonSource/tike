"""This module provides Solver implementations for a variety of algorithms."""

from tike.opt import conjugate_gradient
from tike.ptycho import PtychoBackend

__all__ = [
    "available_solvers",
    "ConjugateGradientPtychoSolver",
]

class ConjugateGradientPtychoSolver(PtychoBackend):
    """Solve the ptychography problem using gradient descent."""

    def run(
        self, data, probe, scan, psi,
        reg=0j, num_iter=1, rho=0, dir_psi=None,
        model='poisson', recover_probe=False, dir_probe=None,
        **kwargs
    ):  # yapf: disable
        """Use conjugate gradient to estimate `psi`.

        Parameters
        ----------
        reg : (V, H, P) :py:class:`numpy.array` complex
            The regularizer for psi. (h + lamda / rho)
        rho : float
            The positive penalty parameter. It should be less than 1.
        dir_psi : (V, H) :py:class:`numpy.array` complex
            The search direction.

        """
        xp = self.array_module
        if not (xp.iscomplexobj(psi) and xp.iscomplexobj(probe)
                and xp.iscomplexobj(reg)):
            raise TypeError("psi, probe, and reg must be complex.")
        data = data.astype(xp.float32)
        probe = probe.astype(xp.complex64)
        psi = psi.astype(xp.complex64)
        rho = xp.asarray(rho, 'float32')
        reg = xp.asarray(reg, 'complex64')

        if model is 'poisson':

            def maximum_a_posteriori_probability(farplane):
                simdata = xp.square(xp.abs(farplane))
                return xp.sum(simdata - data * xp.log(simdata + 1e-32))

            def data_diff(farplane):
                return farplane * (
                    1 - data / (xp.square(xp.abs(farplane)) + 1e-32))

        elif model is 'gaussian':

            def maximum_a_posteriori_probability(farplane):
                return xp.square(
                    xp.linalg.norm(xp.abs(farplane) - xp.sqrt(data)))

            def data_diff(farplane):
                return (farplane
                        - xp.sqrt(data) * xp.exp(1j * xp.angle(farplane)))

        else:
            raise ValueError("model must be 'gaussian' or 'poisson.'")

        def grad(farplane):
            grad_psi = self.adj(
                farplane=data_diff(farplane),
                probe=probe, scan=scan,
            )  # yapf: disable
            grad_psi /= xp.max(xp.abs(probe))**2
            grad_psi -= rho * (reg - psi)
            return grad_psi

        psi = conjugate_gradient(
            self.array_module,
            x=psi,
            fwd=lambda x: self.fwd(psi=x, scan=scan, probe=probe),
            cost_function=maximum_a_posteriori_probability,
            grad=grad,
            num_iter=num_iter,
        )

        # def get_grad_probe(farplane):
        #     grad_probe = self.adj_probe(
        #         farplane=data_diff(farplane),
        #         scan=scan,
        #         psi=psi,
        #     )  # yapf: disable
        #     grad_probe /= xp.square(xp.max(xp.abs(psi)))
        #     grad_probe /= self.nscan
        #
        # probe = ConjugateGradient.run(
        #     self,
        #     'probe',
        #     x=probe,
        #     num_iter=num_iter,
        #     cost_function=maximum_a_posteriori_probability,
        #     get_grad=get_grad,
        #     scan=scan,
        #     psi=psi,
        # )

        return {
            'psi': psi,
            'probe': probe,
        }


# TODO: Add new algorithms here
available_solvers = {
    "cgrad": ConjugateGradientPtychoSolver,
}
