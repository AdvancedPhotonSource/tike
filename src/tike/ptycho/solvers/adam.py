import logging

from tike.opt import get_batch, adam, randomizer

from ..object import positivity_constraint, smoothness_constraint
from ..probe import orthogonalize_eig

logger = logging.getLogger(__name__)


def adam_grad(
    op,
    comm,
    data,
    batches,
    *,
    parameters,
):
    """Solve the ptychography problem using ADAptive Moment gradient descent.

    Parameters
    ----------
    op : :py:class:`tike.operators.Ptycho`
        A ptychography operator.
    comm : :py:class:`tike.communicators.Comm`
        An object which manages communications between GPUs and nodes.
    data : list((FRAME, WIDE, HIGH) float32, ...)
        A list of unique CuPy arrays for each device containing
        the intensity (square of the absolute value) of the propagated
        wavefront; i.e. what the detector records. FFT-shifted so the
        diffraction peak is at the corners.
    batches : list(list((BATCH_SIZE, ) int, ...), ...)
        A list of list of indices along the FRAME axis of `data` for
        each device which define the batches of `data` to process
        simultaneously.
    parameters : :py:class:`tike.ptycho.solvers.PtychoParameters`
        An object which contains reconstruction parameters.

    Returns
    -------
    result : dict
        A dictionary containing the updated keyword-only arguments passed to
        this function.

    .. seealso:: :py:mod:`tike.ptycho`

    """
    probe = parameters.probe
    scan = parameters.scan
    psi = parameters.psi
    algorithm_options = parameters.algorithm_options
    probe_options = parameters.probe_options
    position_options = parameters.position_options
    object_options = parameters.object_options
    batch_cost = []

    for n in randomizer.permutation(len(batches[0])):

        bdata = comm.pool.map(get_batch, data, batches, n=n)
        bscan = comm.pool.map(get_batch, scan, batches, n=n)

        cost, psi, probe = _update_all(
            op,
            comm,
            bdata,
            psi,
            bscan,
            probe,
            object_options,
            probe_options,
            algorithm_options,
        )
        batch_cost.append(cost)

    if probe_options and probe_options.orthogonality_constraint:
        probe = comm.pool.map(orthogonalize_eig, probe)

    if object_options:
        psi = comm.pool.map(positivity_constraint,
                            psi,
                            r=object_options.positivity_constraint)

        psi = comm.pool.map(smoothness_constraint,
                            psi,
                            a=object_options.smoothness_constraint)

    algorithm_options.costs.append(batch_cost)
    parameters.probe = probe
    parameters.psi = psi
    parameters.scan = scan
    parameters.algorithm_options = algorithm_options
    parameters.probe_options = probe_options
    parameters.object_options = object_options
    parameters.position_options = position_options
    return parameters


def _grad_all(data, psi, scan, probe, mode=None, op=None):
    """Compute the gradient with respect to probe(s) and object.

    Parameters
    ----------
    mode : list(int)
        Only return the gradient with resepect to these probes.

    """
    self = op
    mode = list(range(probe.shape[-3])) if mode is None else mode
    intensity, farplane = self._compute_intensity(
        data,
        psi,
        scan,
        probe,
    )
    cost = self.propagation.cost(data, intensity)
    grad_psi, grad_probe, psi_amp, probe_amp = self.adj_all(
        farplane=self.propagation.grad(
            data,
            farplane[..., mode, :, :],
            intensity,
            overwrite=True,
        ),
        probe=probe[..., mode, :, :],
        scan=scan,
        psi=psi,
        overwrite=True,
        rpie=True,
    )
    grad_probe = self.xp.sum(
        grad_probe,
        axis=0,
        keepdims=True,
    )
    return cost, grad_psi, grad_probe, psi_amp, probe_amp


def _update_all(
    op,
    comm,
    data,
    psi,
    scan,
    probe,
    object_options,
    probe_options,
    algorithm_options,
):
    (
        cost,
        grad_psi,
        grad_probe,
        psi_amp,
        probe_amp,
    ) = (list(a) for a in zip(*comm.pool.map(
        _grad_all,
        data,
        psi,
        scan,
        probe,
        op=op,
    )))

    if comm.use_mpi:
        cost = comm.Allreduce_reduce(cost, 'cpu')
    else:
        cost = comm.reduce(cost, 'cpu')
    logger.info('%10s cost is %+12.5e', 'farplane', cost)

    if object_options is not None:
        if comm.use_mpi:
            dpsi = comm.Allreduce_reduce(grad_psi, 'gpu')[0]
            probe_amp = comm.Allreduce_reduce(probe_amp, 'gpu')[0]
        else:
            dpsi = comm.reduce(grad_psi, 'gpu')[0]
            probe_amp = comm.reduce(probe_amp, 'gpu')[0]

        dpsi /= (1 - algorithm_options.alpha
                ) * probe_amp + algorithm_options.alpha * probe_amp.max(
                    keepdims=True,
                    axis=(-1, -2),
                )

        object_options.use_adaptive_moment = True
        (
            dpsi,
            object_options.v,
            object_options.m,
        ) = adam(
            g=dpsi,
            v=object_options.v,
            m=object_options.m,
            vdecay=object_options.vdecay,
            mdecay=object_options.mdecay,
        )
        psi[0] = psi[0] - algorithm_options.step_length * dpsi
        psi = comm.pool.bcast([psi[0]])

    if probe_options is not None:
        if comm.use_mpi:
            dprobe = comm.Allreduce_reduce(grad_probe, 'gpu')[0]
            psi_amp = comm.Allreduce_reduce(psi_amp, 'gpu')[0]

        else:
            dprobe = comm.reduce(grad_probe, 'gpu')[0]
            psi_amp = comm.reduce(psi_amp, 'gpu')[0]

        dprobe /= (1 - algorithm_options.alpha
                  ) * psi_amp + algorithm_options.alpha * psi_amp.max(
                      keepdims=True,
                      axis=(-1, -2),
                  )

        probe_options.use_adaptive_moment = True
        (
            dprobe,
            probe_options.v,
            probe_options.m,
        ) = adam(
            g=dprobe,
            v=probe_options.v,
            m=probe_options.m,
            vdecay=probe_options.vdecay,
            mdecay=probe_options.mdecay,
        )
        probe[0] = probe[0] - algorithm_options.step_length * dprobe
        probe = comm.pool.bcast([probe[0]])

    return cost, psi, probe
