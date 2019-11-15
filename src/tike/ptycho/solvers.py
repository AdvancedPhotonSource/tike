"""This module provides Solver implementations for a variety of algorithms."""

from tike.ptycho import PtychoBackend

__all__ = [
    "available_solvers",
    "ConjugateGradientPtychoSolver",
]


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
        m = 0  # Some tuning parameter for termination
        fx = f(x)  # Save the result of f(x) instead of computing it many times
        # Decrease the step length while the step increases the cost function
        while f(x + step_length * d) > fx + step_shrink * m:
            if step_length < 1e-32:
                warnings.warn("Line search failed for conjugate gradient.")
                return 0
            step_length *= step_shrink
        return step_length

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
        gamma_psi : float
            The ptychography gradient descent step size.
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

        print("# congujate gradient parameters\n"
              "iteration, step size object, step size probe, function min"
              )  # csv column headers
        for i in range(num_iter):
            # Compute the gradient at the current location
            farplane = self.fwd(
                probe=probe, scan=scan,
                psi=psi,
            )  # yapf: disable
            # Updates for each illumination patch
            grad_psi = self.adj(
                farplane=data_diff(farplane),
                probe=probe, scan=scan,
            )  # yapf: disable
            # FIXME: Divide by zero occurs when probe is all zeros?
            grad_psi /= xp.max(xp.abs(probe))**2
            grad_psi -= rho * (reg - psi)
            # Update the search direction, dir_psi, using Dai-Yuan direction.
            # dir_psi and grad_psi are the same shape as psi
            if dir_psi is None:
                dir_psi = -grad_psi
            else:
                dir_psi = (
                    -grad_psi
                    + dir_psi * xp.square(xp.linalg.norm(grad_psi))
                    / (xp.sum(xp.conj(dir_psi) * (grad_psi - grad_psi0))
                       + 1e-32)
                )  # yapf: disable
            grad_psi0 = grad_psi
            # Update the step length, gamma_psi
            gamma_psi = self.line_search(
                f=maximum_a_posteriori_probability,
                x=farplane,
                d=self.fwd(
                    probe=probe, scan=scan,
                    psi=dir_psi,
                ),
            )  # yapf: disable
            # Update the guess for psi
            psi = psi + gamma_psi * dir_psi

            if not recover_probe:
                if (i + 1) % 8 == 0:
                    print("%4d, %.3e, 0, %.7e" % (
                        (i + 1), gamma_psi,
                        maximum_a_posteriori_probability(farplane),
                    ))  # yapf: disable
                continue

            farplane = self.fwd(
                probe=probe, scan=scan,
                psi=psi,
            )  # yapf: disable
            # Updates for each probe
            grad_probe = self.adj_probe(
                farplane=data_diff(farplane),
                scan=scan,
                psi=psi,
            )  # yapf: disable
            grad_probe /= xp.square(xp.max(xp.abs(psi)))
            grad_probe /= self.nscan
            # Update the search direction, dir_probe, using Dai-Yuan direction.
            if dir_probe is None:
                dir_probe = -grad_probe
            else:
                dir_probe = (
                    -grad_probe
                    + dir_probe * xp.square(xp.linalg.norm(grad_probe))
                    / (xp.sum(xp.conj(dir_probe) * (grad_probe - grad_probe0))
                       + 1e-32)
                )  # yapf: disable
            grad_probe0 = grad_probe
            # Update the step length, gamma_probe
            gamma_probe = self.line_search(
                f=maximum_a_posteriori_probability,
                x=farplane,
                d=self.fwd(
                    probe=dir_probe, scan=scan,
                    psi=psi,
                ),
            )  # yapf: disable
            # Update the guess for the probes
            probe = probe + gamma_probe * dir_probe

            # check convergence
            if (i + 1) % 8 == 0:
                print("%4d, %.3e, %.3e, %.7e" % (
                    (i + 1), gamma_psi, gamma_probe,
                    maximum_a_posteriori_probability(farplane),
                ))  # yapf: disable

        return {
            'psi': psi,
            'probe': probe,
        }


# TODO: Add new algorithms here
available_solvers = {
    "cgrad": ConjugateGradientPtychoSolver,
}
