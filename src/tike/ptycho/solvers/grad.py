import logging
import typing

import cupy as cp
import cupyx
import numpy.typing as npt

import tike.communicators
import tike.communicators.stream
import tike.operators
import tike.opt
import tike.ptycho.object
import tike.ptycho.probe

from .options import *

logger = logging.getLogger(__name__)


def grad(
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
        batches_per_update = parameters.algorithm_options.num_batch
    else:
        order = tike.opt.randomizer.permutation
        batches_per_update = 1

    batch_cost = []
    sum_amp_probe = [0] * comm.pool.num_workers
    sum_amp_psi = [0] * comm.pool.num_workers
    sum_grad_object = [0] * comm.pool.num_workers
    sum_grad_probe = [0] * comm.pool.num_workers
    num_batch_combined = 0
    for n in order(parameters.algorithm_options.num_batch):

        # cost = comm.pool.map(
        #     _cost_function,
        #     data,
        #     parameters.psi,
        #     parameters.scan,
        #     parameters.probe,
        #     batches,
        #     op=op,
        #     n=n,
        #     streams=streams,
        # )

        (
            cost,
            grad_psi,
            grad_probe,
            amp_psi,
            amp_probe,
        ) = (list(a) for a in zip(*comm.pool.map(
            _grad_function,
            data,
            parameters.psi,
            parameters.scan,
            parameters.probe,
            batches,
            op=op,
            n=n,
            streams=comm.streams,
        )))

        cost = comm.Allreduce_mean(cost, axis=0).get()
        logger.info('%10s cost is %+12.5e', 'farplane', cost)
        batch_cost.append(cost[0])

        sum_amp_probe = comm.pool.map(cp.add, sum_amp_probe, amp_probe)
        sum_amp_psi = comm.pool.map(cp.add, sum_amp_psi, amp_psi)
        sum_grad_object = comm.pool.map(cp.add, sum_grad_object, grad_psi)
        sum_grad_probe = comm.pool.map(cp.add, sum_grad_probe, grad_probe)
        num_batch_combined += 1

        if num_batch_combined == batches_per_update:

            sum_amp_probe = comm.Allreduce_reduce_gpu(amp_probe)[0]
            sum_amp_psi = comm.Allreduce_reduce_gpu(amp_psi)[0]
            sum_grad_object = comm.Allreduce_reduce_gpu(grad_psi)[0]
            sum_grad_probe = comm.Allreduce_reduce_gpu(grad_probe)[0]
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
            sum_amp_probe = [0] * comm.pool.num_workers
            sum_amp_psi = [0] * comm.pool.num_workers
            sum_grad_object = [0] * comm.pool.num_workers
            sum_grad_probe = [0] * comm.pool.num_workers
            num_batch_combined = 0

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


def _cost_function(
    data: npt.NDArray[tike.precision.floating],
    psi: npt.NDArray[tike.precision.cfloating],
    scan: npt.NDArray[tike.precision.floating],
    probe: npt.NDArray[tike.precision.cfloating],
    batches: typing.List[typing.List[int]],
    *,
    n: int,
    op: tike.operators.Ptycho,
    streams,
):
    def make_certain_args_constant(
        data,
        scan,
    ):
        return [op.cost(
            data,
            psi,
            scan,
            probe,
        )]

    return tike.communicators.stream.stream_and_reduce(
        f=make_certain_args_constant,
        args=[data, scan],
        y_shapes=[(1,)],
        y_dtypes=[tike.precision.floating],
        indices=batches[n],
        streams=streams,
    )[0]


def _grad_function(
    data: npt.NDArray[tike.precision.floating],
    psi: npt.NDArray[tike.precision.cfloating],
    scan: npt.NDArray[tike.precision.floating],
    probe: npt.NDArray[tike.precision.cfloating],
    batches: typing.List[typing.List[int]],
    *,
    n: int,
    op: tike.operators.Ptycho,
    streams,
) -> typing.Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """Compute the gradient with respect to probe(s) and object."""

    def make_certain_args_constant(
        data,
        scan,
    ):
        intensity, farplane = op._compute_intensity(
            None,
            psi,
            scan,
            probe,
        )
        cost = op.propagation.cost(data, intensity)
        grad_psi, grad_probe, amp_psi, amp_probe = op.adj_all(
            farplane=op.propagation.grad(
                data,
                farplane,
                intensity,
            ),
            probe=probe,
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

    return tike.communicators.stream.stream_and_reduce(
        f=make_certain_args_constant,
        args=[data, scan],
        y_shapes=[(1,), psi.shape, probe.shape, probe.shape, psi.shape],
        y_dtypes=[tike.precision.floating, psi.dtype, probe.dtype, probe.dtype, psi.dtype],
        indices=batches[n],
        streams=streams,
    )

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


def _update_probe(
    op,
    comm,
    data,
    psi,
    scan,
    probe,
    num_iter,
    step_length,
    mode,
    probe_options,
):
    """Solve the probe recovery problem."""

    def cost_function(probe):
        cost_out = comm.pool.map(op.cost, data, psi, scan, probe)
        return comm.Allreduce_reduce_cpu(cost_out)

    def grad(probe):
        grad_list = comm.pool.map(
            op.grad_probe,
            data,
            psi,
            scan,
            probe,
            mode=mode,
        )
        return comm.Allreduce_reduce_gpu(grad_list)

    def dir_multi(dir):
        """Scatter dir to all GPUs"""
        return comm.pool.bcast(dir)

    def update_multi(x, gamma, d):

        def f(x, d):
            return x[..., mode, :, :] + gamma * d

        return comm.pool.map(f, x, d)

    probe, cost = conjugate_gradient(
        op.xp,
        x=probe,
        cost_function=cost_function,
        grad=grad,
        dir_multi=dir_multi,
        update_multi=update_multi,
        num_iter=num_iter,
        step_length=step_length,
    )

    if probe[0].shape[-3] > 1 and probe_options.force_orthogonality:
        probe = comm.pool.map(orthogonalize_gs, probe, axis=(-2, -1))

    logger.info('%10s cost is %+12.5e', 'probe', cost)
    return probe, cost, probe_options


def _update_object(
    op,
    comm,
    data,
    psi,
    scan,
    probe,
    num_iter,
    step_length,
    object_options,
):
    """Solve the object recovery problem."""

    def cost_function_multi(psi, **kwargs):
        cost_out = comm.pool.map(op.cost, data, psi, scan, probe)
        return comm.Allreduce_mean(cost_out, axis=None).get()

    def grad_multi(psi):
        grad_list = comm.pool.map(op.grad_psi, data, psi, scan, probe)
        return comm.Allreduce_reduce_gpu(grad_list)

    def dir_multi(dir):
        """Scatter dir to all GPUs"""
        return comm.pool.bcast(dir)

    def update_multi(psi, gamma, dir):

        def f(psi, dir):
            return psi + gamma * dir

        return list(comm.pool.map(f, psi, dir))

    psi, cost = conjugate_gradient(
        op.xp,
        x=psi,
        cost_function=cost_function_multi,
        grad=grad_multi,
        dir_multi=dir_multi,
        update_multi=update_multi,
        num_iter=num_iter,
        step_length=step_length,
    )

    logger.info('%10s cost is %+12.5e', 'object', cost)
    return psi, cost, object_options


