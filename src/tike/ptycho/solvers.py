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
            psi, psi_corner,
            reg=0j, num_iter=1, rho=0, gamma=0.25,
            **kwargs
    ):  # yapf: disable
        """Use gradient descent to estimate `psi`.

        Parameters
        ----------
        reg : (V, H, P) :py:class:`numpy.array` complex
            The regularizer for psi. (h + lamda / rho)
        rho : float
            The positive penalty parameter. It should be less than 1.
        gamma : float
            The ptychography gradient descent step size.

        """
        if not (np.iscomplexobj(psi) and np.iscomplexobj(probe)
                and np.iscomplexobj(reg)):
            raise TypeError("psi, probe, and reg must be complex.")
        data = data.astype(np.float32)
        probe = probe.astype(np.complex64)
        psi = psi.astype(np.complex64)
        # Combine the illumination from all positions
        combined_probe =_combine_grids(
            grids=np.tile(probe[np.newaxis, ...], [len(data), 1, 1]),
            v=v, h=h,
            combined_shape=psi.shape,
            combined_corner=psi_corner,
        )  # yapf: disable
        combined_probe[combined_probe == 0] = 1
        detector_shape = data.shape[1:]
        for i in range(num_iter):
            farplane = self.fwd(
                probe=probe, v=v, h=h,
                psi=psi, psi_corner=psi_corner,
            )  # yapf: disable
            # Updates for each illumination patch
            grad = self.adj(
                # FIXME: Divide by zero occurs when probe is all zeros?
                farplane=farplane - data / np.conjugate(farplane),
                probe=probe, v=v, h=h,
                psi_shape=psi.shape, psi_corner=psi_corner,
                combined_probe=combined_probe,
            )  # yapf: disable
            grad -= rho * (reg - psi)
            # Update the guess for psi
            psi = psi - gamma * grad
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
            psi, psi_corner,
            reg=0j, num_iter=1, rho=0, gamma=0.25, eta=None,
            **kwargs
    ):  # yapf: disable
        """Use conjugate gradient to estimate `psi`.

        Parameters
        ----------
        reg : (V, H, P) :py:class:`numpy.array` complex
            The regularizer for psi. (h + lamda / rho)
        rho : float
            The positive penalty parameter. It should be less than 1.
        gamma : float
            The ptychography gradient descent step size.
        eta : (V, H) :py:class:`numpy.array` complex
            The search direction.

        """
        if not (np.iscomplexobj(psi) and np.iscomplexobj(probe)
                and np.iscomplexobj(reg)):
            raise TypeError("psi, probe, and reg must be complex.")
        data = data.astype(np.float32)
        probe = probe.astype(np.complex64)
        psi = psi.astype(np.complex64)
        # Combine the illumination from all positions
        combined_probe =_combine_grids(
            grids=np.tile(np.abs(probe)[np.newaxis, ...], [len(data), 1, 1]),
            v=v, h=h,
            combined_shape=psi.shape,
            combined_corner=psi_corner,
        )  # yapf: disable
        combined_probe[combined_probe == 0] = 1
        detector_shape = data.shape[1:]

        # Define the function that we are minimizing
        def maximum_a_posteriori_probability(psi):
            """Return the probability that psi is correct given the data."""
            simdata = self.fwd(
                probe=probe, v=v, h=h,
                psi=psi, psi_corner=psi_corner,
            )  # yapf: disable
            return np.nansum(
                np.square(np.abs(simdata)) - 2 * data * np.log(np.abs(simdata)))

        for i in range(num_iter):
            # Compute the gradient at the current location
            farplane = self.fwd(
                probe=probe, v=v, h=h,
                psi=psi, psi_corner=psi_corner,
            )  # yapf: disable
            # Updates for each illumination patch
            denominator = np.conjugate(farplane)
            denominator[denominator == 0] = 1
            grad = self.adj(
                # FIXME: Divide by zero occurs when probe is all zeros?
                farplane - data / denominator,
                probe=probe, v=v, h=h,
                psi_shape=psi.shape, psi_corner=psi_corner,
                combined_probe=combined_probe,
            )  # yapf: disable
            grad -= rho * (reg - psi)
            # Update the search direction, eta.
            # eta and grad are the same shape as psi
            if eta is None:
                eta = -grad
            else:
                denominator = np.nansum(np.conjugate(grad - grad0) * eta)
                if denominator != 0:
                    # Use previous eta if previous (grad - grad0) is zero
                    eta = -grad + eta * np.square(
                        np.linalg.norm(grad)) / denominator
            # Update the step length, gamma
            gamma = self.line_search(
                f=maximum_a_posteriori_probability,
                x=psi,
                d=eta,
                step_length=gamma,
            )
            # Update the guess for psi
            psi = psi + gamma * eta
            grad0 = grad
        return psi


# TODO: Add new algorithms here
available_solvers = {
    "grad": GradientDescentPtychoSolver,
    "cgrad": ConjugateGradientPtychoSolver,
}
