"""This module provides Solver implementations for a variety of algorithms."""

import logging

from tike.opt import conjugate_gradient
from tike.ptycho import PtychoBackend

__all__ = [
    "available_solvers",
    "ConjugateGradientPtychoSolver",
    "GradientDescentLeastSquaresSteps",
]

logger = logging.getLogger(__name__)

class ConjugateGradientPtychoSolver(PtychoBackend):
    """Solve the ptychography problem using gradient descent."""

    def run(
        self, data, probe, scan, psi,
        reg=0j, num_iter=1, rho=0.0,
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

        """
        xp = self.array_module
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
                return xp.sum(xp.square(xp.abs(farplane) - xp.sqrt(data)))

            def data_diff(farplane):
                return (farplane
                        - xp.sqrt(data) * xp.exp(1j * xp.angle(farplane)))

        else:
            raise ValueError("model must be 'gaussian' or 'poisson.'")

        def cost_function(psi):
            farplane = self.fwd(psi=psi, scan=scan, probe=probe)
            return (
                + maximum_a_posteriori_probability(farplane)
                + rho * xp.square(xp.linalg.norm(reg - psi))
            )

        def grad(psi):
            farplane = self.fwd(psi=psi, scan=scan, probe=probe)
            grad_psi = self.adj(
                farplane=data_diff(farplane),
                probe=probe, scan=scan,
            )  # yapf: disable
            grad_psi /= xp.max(xp.abs(probe))**2  # this is not in the math
            grad_psi -= rho * (reg - psi)
            return grad_psi

        psi = conjugate_gradient(
            self.array_module,
            x=psi,
            cost_function=cost_function,
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

class GradientDescentLeastSquaresSteps(PtychoBackend):
    """Solve the Ptychography Problem using method from Odstrcil et al (2018).

    References
    ----------
    Michal Odstrcil, Andreas Menzel, and Manuel Guizar-Sicaros. Iteraive
    least-squares solver for generalized maximum-likelihood ptychography. Optics
    Express. 2018.
    """

    def cost_poisson(xp, data, intensity):
        return xp.sum(intensity - data * xp.log(intensity + 1e-32))

    def cost_amplitude(xp, data, intensity):
        return xp.sum(xp.square(xp.sqrt(data) - xp.sqrt(intensity)))

    def grad_poisson(xp, data, farplane, mode_axis):
        intensity = xp.square(xp.abs(farplane))
        return (
            farplane
            * xp.conj(
                1
                - data[:, :, xp.newaxis]
                / (
                    intensity
                    * xp.sum(intensity, axis=mode_axis, keepdims=True)
                    + 1e-32
                )
            )
        )

    def grad_amplitude(xp, data, farplane, mode_axis):
        intensity = xp.sum(xp.square(xp.abs(farplane)), axis=mode_axis)
        return (
            farplane
            * xp.conj(1 - xp.sqrt(data / (intensity + 1e-32)))[:, :, xp.newaxis]
        )

    cost = {
        'poisson': cost_poisson,
        'gaussian': cost_amplitude,
    }

    grad = {
        'poisson': grad_poisson,
        'gaussian': grad_amplitude,
    }

    def update_phase(self, data, farplane, nmodes=1, num_iter=2, model='gaussian'):
        """Solve the farplane phase problem.

        Parameters
        ----------
        nmodes : int
            The number of incoherent farplane waves that hit the detector
            simultaneously; the number of waves to sum incoherently.
        """
        xp = self.array_module
        farplane = farplane.reshape(
            (self.ntheta, -1, nmodes, self.detector_shape, self.detector_shape))
        mode_axis = 2

        def grad(farplane):
            return self.grad[model](xp, data, farplane, mode_axis)

        def cost_function(farplane):
            intensity = xp.sum(xp.square(xp.abs(farplane)),
                               axis=mode_axis,
                               keepdims=False)
            return self.cost[model](xp, data, intensity)

        farplane = conjugate_gradient(
            self.array_module,
            x=farplane,
            cost_function=cost_function,
            grad=grad,
            num_iter=num_iter,
        )

        # print cost function for sanity check
        if logger.isEnabledFor(logging.INFO):
            logger.info(' farplane cost is %+12.5e', cost_function(farplane))

        return farplane.reshape(
            (self.ntheta, -1, nmodes, self.detector_shape, self.detector_shape))

    def update_probe(self, nearplane, probe, scan, psi, nmodes=1, num_iter=1):
        """Solve the nearplane single probe recovery problem."""
        # name the axes
        position_axis, mode_axis = 1, 2
        xp = self.array_module

        probe = probe.reshape(
            (self.ntheta, -1, nmodes, self.probe_shape, self.probe_shape)
        )
        nearplane = nearplane.reshape(
            (self.ntheta, self.nscan, nmodes, self.probe_shape, self.probe_shape)
        )
        obj_patches = xp.expand_dims(
            self.diffraction.fwd(psi=psi, scan=scan),
            axis=mode_axis,
        )

        def cost_function(probe):
            return xp.sum(xp.square(xp.abs(nearplane - probe * obj_patches)))

        def grad(probe):
            return xp.sum(
                xp.conj(-obj_patches) * (nearplane - probe * obj_patches),
                axis=position_axis, keepdims=True,
            ) / self.nscan

        probe = conjugate_gradient(
            self.array_module,
            x=probe,
            cost_function=cost_function,
            grad=grad,
            num_iter=num_iter,
        )

        if logger.isEnabledFor(logging.INFO):
            cost = cost_function(probe)
            logger.info('nearplane cost is             %+12.5e', cost)

        return probe

    def update_object(self, nearplane, probe, scan, psi, nmodes=1, num_iter=1):
        """Solve the nearplane object recovery problem."""
        xp = self.array_module
        mode_axis = 2

        _probe = probe.reshape(
            (self.ntheta, -1, nmodes, self.probe_shape, self.probe_shape)
        )
        _nearplane = nearplane.reshape(
            (self.ntheta, self.nscan, nmodes, self.probe_shape, self.probe_shape)
        )

        for i in range(nmodes):
            nearplane, probe = _nearplane[:, :, i], _probe[:, :, i]

            # net_flux = self.diffraction.adj(
            #     nearplane=xp.zeros_like(nearplane) + xp.square(xp.abs(probe)),
            #     scan=scan,
            # ) + 1e-32

            def cost_function(psi):
                return xp.sum(xp.square(xp.abs(
                    _nearplane
                    - _probe * xp.expand_dims(
                        self.diffraction.fwd(psi=psi, scan=scan),
                        axis=mode_axis,
                    )
                )))

            def grad(psi):
                return self.diffraction.adj(
                    xp.conj(-probe)
                    * (nearplane - probe * self.diffraction.fwd(psi=psi, scan=scan)),
                    scan=scan,
                )

            psi = conjugate_gradient(
                self.array_module,
                x=psi,
                cost_function=cost_function,
                grad=grad,
                num_iter=num_iter,
            )

        if logger.isEnabledFor(logging.INFO):
            cost = cost_function(psi)
            logger.info('nearplane cost is             %+12.5e', cost)

        return psi

    def run(
        self, data, probe, scan, psi,
        model='poisson',
        num_iter=1, recover_probe=False, recover_obj=True,
        nmodes=1,
        **kwargs
    ):  # yapf: disable
        """Estimate `psi` and `probe`."""
        xp = self.array_module
        mode_axis = 2

        probe = probe.reshape(
            (self.ntheta, -1, nmodes, self.probe_shape, self.probe_shape),
        )

        for _ in range(num_iter):

            nearplane = xp.expand_dims(
                self.diffraction.fwd(psi=psi, scan=scan),
                axis=mode_axis,
            ) * probe

            farplane = self.propagation.fwd(nearplane)
            farplane = self.update_phase(data, farplane, nmodes=nmodes)
            nearplane = self.propagation.adj(farplane)

            if recover_obj:
                psi = self.update_object(nearplane, probe, scan, psi, nmodes=nmodes)

            if recover_probe:
                probe = self.update_probe(nearplane, probe, scan, psi, nmodes=nmodes)

        return {
            'psi': psi,
            'probe': probe,
        }

# TODO: Add new algorithms here
available_solvers = {
    "cgrad": ConjugateGradientPtychoSolver,
    "odstrcil": GradientDescentLeastSquaresSteps,
}
