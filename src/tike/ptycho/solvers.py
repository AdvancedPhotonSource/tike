"""This module provides Solver implementations for a variety of algorithms."""

import numpy as np

from tike.ptycho import PtychoBackend
# TODO: This module should not need to import from _shift
from tike.ptycho._core._shift import _combine_grids, _uncombine_grids

__all__ = [
    "available_solvers",
    "GradientDescentPtychoSolver",
    "ConjugateGradientPtychoSolver",
]


class GradientDescentPtychoSolver(PtychoBackend):
    """Solve the ptychography problem using gradient descent."""

    def run(self,
            data,
            probe, v, h,
            psi,
            reg=0j, num_iter=1, rho=0, gamma_psi=0.25,
            **kwargs
    ):  # yapf: disable
        """Use gradient descent to estimate `psi`.

        Parameters
        ----------
        reg : (V, H, P) :py:class:`numpy.array` complex
            The regularizer for psi. (h + lamda / rho)
        rho : float
            The positive penalty parameter. It should be less than 1.
        gamma_psi : float
            The ptychography gradient descent step size.

        """
        if not (np.iscomplexobj(psi) and np.iscomplexobj(probe)
                and np.iscomplexobj(reg)):
            raise TypeError("psi, probe, and reg must be complex.")
        data = data.astype(np.float32)
        probe = probe.astype(np.complex64)
        psi = psi.astype(np.complex64)
        for i in range(num_iter):
            farplane = self.fwd(
                probe=probe, v=v, h=h,
                psi=psi,
            )  # yapf: disable
            # Updates for each illumination patch
            grad_psi = self.adj(
                # FIXME: Divide by zero occurs when probe is all zeros?
                farplane * (1 - data / (np.square(np.abs(farplane)) + 1e-32)),
                probe=probe, v=v, h=h,
                psi_shape=psi.shape,
            )  # yapf: disable
            grad_psi /= np.max(np.abs(probe))**2
            grad_psi -= rho * (reg - psi)
            # Update the guess for psi
            psi = psi - gamma_psi * grad_psi
        return psi


class ConjugateGradientPtychoSolver(PtychoBackend):
    """Solve the ptychography problem using gradient descent."""

    @staticmethod
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
        m = 0.5  # Some tuning parameter for termination
        fx = f(x)  # Save the result of f(x) instead of computing it many times
        # Decrease the step length while the step increases the cost function
        while f(x + step_length * d) > fx + step_shrink * m:
            if step_length < 1e-32:
                warnings.warn("Line search failed for conjugate gradient.")
                return 0
            step_length *= step_shrink
        return step_length

    def run(self,
            data,
            probe, v, h,
            psi,
            reg=0j, num_iter=1, rho=0, gamma_psi=0.25, dir_psi=None,
            **kwargs
    ):  # yapf: disable
        """Use conjugate gradient to estimate `psi`.

        Parameters
        ----------
        reg : (V, H, P) :py:class:`numpy.array` complex
            The regularizer for psi. (h + lamda / rho)
        rho : float
            The positive penalty parameter. It should be less than 1.
        gamma_psi : float
            The ptychography gradient descent step size.
        dir_psi : (V, H) :py:class:`numpy.array` complex
            The search direction.

        """
        if not (np.iscomplexobj(psi) and np.iscomplexobj(probe)
                and np.iscomplexobj(reg)):
            raise TypeError("psi, probe, and reg must be complex.")
        data = data.astype(np.float32)
        probe = probe.astype(np.complex64)
        psi = psi.astype(np.complex64)

        # Define the function that we are minimizing
        def maximum_a_posteriori_probability(simdata):
            """Return the probability that psi is correct given the data."""
            return np.nansum(
                np.square(np.abs(simdata)) - 2 * data * np.log(np.abs(simdata)))

        print("# congujate gradient parameters\n"
              "iteration, step size object, step size probe, function min"
             )  # csv column headers
        for i in range(num_iter):
            # Compute the gradient at the current location
            farplane = self.fwd(
                probe=probe, v=v, h=h,
                psi=psi,
            )  # yapf: disable
            # Updates for each illumination patch
            grad_psi = self.adj(
                farplane * (1 - data / (np.square(np.abs(farplane)) + 1e-32)),
                probe=probe, v=v, h=h,
                psi_shape=psi.shape,
            )  # yapf: disable
            # FIXME: Divide by zero occurs when probe is all zeros?
            grad_psi /= np.max(np.abs(probe))**2
            grad_psi -= rho * (reg - psi)
            # Update the search direction, dir_psi.
            # dir_psi and grad_psi are the same shape as psi
            if dir_psi is None:
                dir_psi = -grad_psi
            else:
                dir_psi = (
                    -grad_psi
                    + dir_psi * np.square(np.linalg.norm(grad_psi))
                    / (np.sum(np.conj(dir_psi) * (grad_psi - grad_psi0)) + 1e-32)
                )  # yapf: disable
            grad_psi0 = grad_psi
            # Update the step length, gamma_psi
            gamma_psi = self.line_search(
                f=maximum_a_posteriori_probability,
                x=farplane,
                d=self.fwd(
                    probe=probe, v=v, h=h,
                    psi=dir_psi,
                ),
            )  # yapf: disable
            # Update the guess for psi
            psi = psi + gamma_psi * dir_psi

            gamma_prb = 0

            # check convergence
            if (np.mod(i, 8) == 0):
                print("%4d, %.3e, %.3e, %.7e" % (
                    i, gamma_psi, gamma_prb,
                    maximum_a_posteriori_probability(farplane),
                ))  # yapf: disable

        return psi


# TODO: Add new algorithms here
available_solvers = {
    "grad": GradientDescentPtychoSolver,
    "cgrad": ConjugateGradientPtychoSolver,
}
