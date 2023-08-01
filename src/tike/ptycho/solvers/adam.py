import logging
import typing

import cupy as cp
import numpy.typing as npt

import tike.communicators
import tike.operators
import tike.opt
import tike.ptycho.object
import tike.ptycho.probe
import tike.random

from .options import *

logger = logging.getLogger(__name__)


def adam_grad(
    op: tike.operators.Ptycho,
    comm: tike.communicators.Comm,
    data: typing.List[npt.NDArray],
    batches: typing.List[npt.NDArray[cp.intc]],
    *,
    parameters: PtychoParameters,
) -> PtychoParameters:
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
    if parameters.algorithm_options.batch_method == 'compact':
        order = range
    else:
        order = tike.random.randomizer_np.permutation

    batch_cost = []
    sum_grad_object = 0
    sum_grad_probe = 0
    sum_amp_psi = 0
    sum_amp_probe = 0
    for n in order(parameters.algorithm_options.num_batch):

        bdata = comm.pool.map(tike.opt.get_batch, data, batches, n=n)
        bscan = comm.pool.map(tike.opt.get_batch, parameters.scan, batches, n=n)

        (
            cost,
            grad_psi,
            grad_probe,
            amp_psi,
            amp_probe,
        ) = (list(a) for a in zip(*comm.pool.map(
            _grad_all,
            bdata,
            parameters.psi,
            bscan,
            parameters.probe,
            op=op,
            model=parameters.exitwave_options.noise_model,
        )))

        cost = comm.Allreduce_mean(cost, axis=None).get()
        logger.info('%10s cost is %+12.5e', 'farplane', cost)
        batch_cost.append(cost)

        sum_grad_object += comm.Allreduce_reduce_gpu(grad_psi)[0]
        sum_amp_probe += comm.Allreduce_reduce_gpu(amp_probe)[0]
        sum_grad_probe += comm.Allreduce_reduce_gpu(grad_probe)[0]
        sum_amp_psi += comm.Allreduce_reduce_gpu(amp_psi)[0]

        if parameters.algorithm_options.batch_method != 'compact':
            (
                parameters.psi,
                parameters.probe,
            ) = _update_all(
                comm,
                parameters.psi,
                parameters.probe,
                sum_grad_object,
                sum_grad_probe,
                sum_amp_psi,
                sum_amp_probe,
                parameters.object_options,
                parameters.probe_options,
                parameters.algorithm_options,
            )
            sum_grad_object = 0
            sum_grad_probe = 0
            sum_amp_psi = 0
            sum_amp_probe = 0

    if parameters.algorithm_options.batch_method == 'compact':
        (
            parameters.psi,
            parameters.probe,
        ) = _update_all(
            comm,
            parameters.psi,
            parameters.probe,
            sum_grad_object,
            sum_grad_probe,
            sum_amp_psi,
            sum_amp_probe,
            parameters.object_options,
            parameters.probe_options,
            parameters.algorithm_options,
        )

    if parameters.object_options:
        parameters.psi = comm.pool.map(
            tike.ptycho.object.positivity_constraint,
            parameters.psi,
            r=parameters.object_options.positivity_constraint,
        )

        parameters.psi = comm.pool.map(
            tike.ptycho.object.smoothness_constraint,
            parameters.psi,
            a=parameters.object_options.smoothness_constraint,
        )

    parameters.algorithm_options.costs.append(batch_cost)
    return parameters


def _grad_all(
    data: npt.NDArray,
    psi: npt.NDArray,
    scan: npt.NDArray,
    probe: npt.NDArray,
    *,
    mode: typing.Union[None, typing.List[int]] = None,
    op: tike.operators.Ptycho,
    model: str,
) -> typing.Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """Compute the gradient with respect to probe(s) and object.

    Parameters
    ----------
    mode : list(int)
        Only return the gradient with resepect to these probes.

    """
    mode = list(range(probe.shape[-3])) if mode is None else mode
    intensity, farplane = op._compute_intensity(
        data,
        psi,
        scan,
        probe,
    )
    cost = getattr(tike.operators, model)(data, intensity)
    grad_psi, grad_probe, amp_psi, amp_probe = op.adj_all(
        farplane=getattr(tike.operators, f'{model}_grad')(
            data,
            farplane[..., mode, :, :],
            intensity,
        ),
        probe=probe[..., mode, :, :],
        scan=scan,
        psi=psi,
        overwrite=True,
        rpie=True,
    )
    grad_probe = op.xp.sum(
        grad_probe,
        axis=0,
        keepdims=True,
    )
    return cost, grad_psi, grad_probe, amp_psi, amp_probe


def _update_all(
    comm: tike.communicators.Comm,
    psi: typing.List[npt.NDArray],
    probe: typing.List[npt.NDArray],
    dpsi: npt.NDArray,
    dprobe: npt.NDArray,
    amp_psi: npt.NDArray,
    amp_probe: npt.NDArray,
    object_options: ObjectOptions,
    probe_options: ProbeOptions,
    algorithm_options: AdamOptions,
) -> typing.Tuple[typing.List[npt.NDArray], typing.List[npt.NDArray]]:
    if object_options:

        deno = (1 - algorithm_options.alpha
               ) * amp_probe + algorithm_options.alpha * amp_probe.max(
                   keepdims=True,
                   axis=(-1, -2),
               )
        dpsi = dpsi / deno

        (
            dpsi,
            object_options.v,
            object_options.m,
        ) = tike.opt.adam(
            g=dpsi,
            v=object_options.v,
            m=object_options.m,
            vdecay=object_options.vdecay,
            mdecay=object_options.mdecay,
        )
        psi[0] = psi[0] - algorithm_options.step_length * dpsi / deno
        psi = comm.pool.bcast([psi[0]])

    if probe_options:

        deno = (1 - algorithm_options.alpha
               ) * amp_psi + algorithm_options.alpha * amp_psi.max(
                   keepdims=True,
                   axis=(-1, -2),
               )
        dprobe = dprobe / deno

        (
            dprobe,
            probe_options.v,
            probe_options.m,
        ) = tike.opt.adam(
            g=dprobe,
            v=probe_options.v,
            m=probe_options.m,
            vdecay=object_options.vdecay,
            mdecay=object_options.mdecay,
        )
        probe[0] = probe[0] - algorithm_options.step_length * dprobe
        probe = comm.pool.bcast([probe[0]])

    return psi, probe
