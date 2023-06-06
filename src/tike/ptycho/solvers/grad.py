import logging
import typing

import cupy as cp
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
    batches: typing.List[typing.List[npt.NDArray[cp.intc]]],
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
    result : :py:class:`tike.ptycho.solvers.PtychoParameters`
        An object which contains the updated reconstruction parameters.

    .. seealso:: :py:mod:`tike.ptycho`

    """
    if parameters.algorithm_options.batch_method == 'compact':
        order = range
    else:
        order = tike.opt.randomizer.permutation

    # The objective function value for each batch
    batch_cost: typing.List[float] = []
    for n in order(parameters.algorithm_options.num_batch):

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
            comm.streams,
            op=op,
            n=n,
        )))

        cost = comm.Allreduce_mean(cost, axis=0).get()[0]
        logger.info('%10s cost is %+12.5e', 'farplane', cost)
        batch_cost.append(cost)

        sum_amp_probe = comm.Allreduce_reduce_gpu(amp_probe)
        sum_amp_psi = comm.Allreduce_reduce_gpu(amp_psi)
        sum_grad_object = comm.Allreduce_reduce_gpu(grad_psi)
        sum_grad_probe = comm.Allreduce_reduce_gpu(grad_probe)
        (
            updated_psi,
            updated_probe,
        ) = _update_all(
            comm,
            parameters.psi[0],
            parameters.probe[0],
            sum_grad_object[0],
            sum_grad_probe[0],
            sum_amp_psi[0],
            sum_amp_probe[0],
            parameters.object_options,
            parameters.probe_options,
            parameters.algorithm_options,
        )
        parameters.psi = comm.pool.bcast([updated_psi])
        parameters.probe = comm.pool.bcast([updated_probe])

    parameters.algorithm_options.costs.append(batch_cost)
    return parameters


def _cost_function(
    data: npt.NDArray[tike.precision.floating],
    psi: npt.NDArray[tike.precision.cfloating],
    scan: npt.NDArray[tike.precision.floating],
    probe: npt.NDArray[tike.precision.cfloating],
    batches: typing.List[typing.List[int]],
    streams,
    *,
    n: int,
    op: tike.operators.Ptycho,
) -> typing.Tuple[npt.NDArray]:

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
        cost = tike.operators.gaussian_each_pattern(data, intensity).sum(axis=0)
        return [
            cost,
        ]

    result = tike.communicators.stream.stream_and_reduce(
        f=make_certain_args_constant,
        args=[data, scan],
        y_shapes=[(1,)],
        y_dtypes=[tike.precision.floating],
        indices=batches[n],
        streams=streams,
    )
    result[0] = result[0] / len(batches[n])
    return result


def _grad_function(
    data: npt.NDArray[tike.precision.floating],
    psi: npt.NDArray[tike.precision.cfloating],
    scan: npt.NDArray[tike.precision.floating],
    probe: npt.NDArray[tike.precision.cfloating],
    batches: typing.List[typing.List[int]],
    streams,
    *,
    n: int,
    op: tike.operators.Ptycho,
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
        cost = tike.operators.gaussian_each_pattern(data, intensity).sum(axis=0)
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

    result = tike.communicators.stream.stream_and_reduce(
        f=make_certain_args_constant,
        args=[data, scan],
        y_shapes=[(1,), psi.shape, probe.shape, probe.shape, psi.shape],
        y_dtypes=[
            tike.precision.floating, psi.dtype, probe.dtype, probe.dtype,
            psi.dtype
        ],
        indices=batches[n],
        streams=streams,
    )
    result[0] = result[0] / len(batches[n])
    return result


def _update_all(
    comm: tike.communicators.Comm,
    psi: npt.NDArray,
    probe: npt.NDArray,
    dpsi: npt.NDArray,
    dprobe: npt.NDArray,
    amp_psi: npt.NDArray,
    amp_probe: npt.NDArray,
    object_options: ObjectOptions,
    probe_options: ProbeOptions,
    algorithm_options: GradOptions,
) -> typing.Tuple[npt.NDArray, npt.NDArray]:
    if object_options:

        deno = (1 - algorithm_options.alpha
               ) * amp_probe + algorithm_options.alpha * amp_probe.max(
                   keepdims=True,
                   axis=(-1, -2),
               )
        dpsi = dpsi / deno

        if object_options.use_adaptive_moment:
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
        psi = psi - algorithm_options.step_length * dpsi

    if probe_options:

        deno = (1 - algorithm_options.alpha
               ) * amp_psi + algorithm_options.alpha * amp_psi.max(
                   keepdims=True,
                   axis=(-1, -2),
               )
        dprobe = dprobe / deno

        if probe_options.use_adaptive_moment:
            (
                dprobe,
                probe_options.v,
                probe_options.m,
            ) = tike.opt.adam(
                g=dprobe,
                v=probe_options.v,
                m=probe_options.m,
                vdecay=probe_options.vdecay,
                mdecay=probe_options.mdecay,
            )
        probe = probe - algorithm_options.step_length * dprobe

    return psi, probe


def _update(arg0, dir0, arg1, dir1, *, step):
    return [
        arg0 + step * dir0,
        arg1 + step * dir1,
    ]


def _cg(
    op: tike.operators.Ptycho,
    comm: tike.communicators.Comm,
    data: typing.List[npt.NDArray],
    batches: typing.List[typing.List[npt.NDArray[cp.intc]]],
    *,
    parameters: PtychoParameters,
    n: int,
) -> typing.Tuple[PtychoParameters, float]:

    def f(args: typing.List[typing.List[npt.NDArray]]) -> float:
        (cost,) = (list(a) for a in zip(*comm.pool.map(
            _cost_function,
            data,
            args[0],
            parameters.scan,
            args[1],
            batches,
            comm.streams,
            op=op,
            n=n,
        )))
        return comm.Allreduce_mean(cost, axis=0).get()[0]

    def g(
        args: typing.List[typing.List[npt.NDArray]]
    ) -> typing.List[typing.List[npt.NDArray]]:
        (
            _,
            grad_psi,
            grad_probe,
            _,
            _,
        ) = (list(a) for a in zip(*comm.pool.map(
            _grad_function,
            data,
            args[0],
            parameters.scan,
            args[1],
            batches,
            comm.streams,
            op=op,
            n=n,
        )))
        return [
            grad_psi,
            grad_probe,
        ]

    def u(
        args: typing.List[typing.List[npt.NDArray]],
        step: float,
        dirs: typing.List[typing.List[npt.NDArray]],
    ) -> typing.List[typing.List[npt.NDArray]]:
        return [
            list(a) for a in zip(*comm.pool.map(
                _update,
                args[0],
                dirs[0],
                args[1],
                dirs[1],
                step=step,
            ))
        ]

    def d(
        grads1: typing.List[typing.List[npt.NDArray]],
        grads0: typing.List[typing.List[npt.NDArray]] = None,
        dirs: typing.List[typing.List[npt.NDArray]] = None,
    ) -> typing.List[typing.List[npt.NDArray]]:

        result = list()

        for i in range(len(grads1)):
            grad1 = comm.Allreduce_reduce_cpu(grads1[i])
            if grads0 is not None:
                grad0 = comm.Allreduce_reduce_cpu(grads0[i])
            else:
                grad0 = None
            if dirs is not None:
                dir0 = comm.Allreduce_reduce_cpu(dirs[i])
            else:
                dir0 = None
            dir1 = tike.opt.direction_dy(np, grad1, grad0, dir0)
            result.append(comm.pool.bcast(dir1))

        return result

    (
        (
            parameters.psi,
            parameters.probe,
        ),
        cost,
    ) = conjugate_gradient(
        f=f,
        u=u,
        g=g,
        d=d,
        args=[parameters.psi, parameters.probe],
        num_iter=4,
    )

    return parameters, cost


def conjugate_gradient(
    f: typing.Callable[[typing.List[npt.NDArray]], float],
    u: typing.Callable[
        [typing.List[npt.NDArray], float, typing.List[npt.NDArray]],
        typing.List[npt.NDArray],
    ],
    g: typing.Callable[
        [typing.List[npt.NDArray]],
        typing.List[npt.NDArray],
    ],
    d: typing.Callable[
        [typing.List[npt.NDArray]],
        typing.List[npt.NDArray],
    ],
    args: typing.List[npt.NDArray],
    step_length: float = 1,
    cost: typing.Union[None, float] = None,
    num_iter: int = 1,
    num_search: typing.Union[None, int] = None,
) -> typing.Tuple[typing.List[npt.NDArray], float]:
    """Use conjugate gradient to estimate `x`.

    Parameters
    ----------
    f :
        The objective function. Takes args, returns float.
    u :
        The update function. Takes args, step_length, direction and returns args.
    g :
        The gradient function. Takes args and returns gradients.
    d :
        The conjugate function. Takes gradients and returns conjugate directions.
    args :
        The initial position in parameter space.
    num_iter :
        The number of steps to take.
    num_search :
        The maximum number of times to perform line search.
    step_length :
        The initial multiplier of the search direction.
    cost :
        The current loss function estimate.

    Returns
    -------
    args :
        The updated values
    cost :
        The new objective value after stepping
    """
    num_search = num_iter if num_search is None else num_search
    assert num_iter > 0, (
        f"A non-positive number of conjugate gradient steps!? {num_iter}")

    for i in range(num_iter):

        grads1 = g(args)
        if i == 0:
            dirs = d(grads1)
        else:
            dirs = d(grads1, grads0, dirs)
        grads0 = grads1

        if i < num_search:
            step_length, cost, args = tike.opt.line_search1(
                f=f,
                u=u,
                args=args,
                dirs=dirs,
                # step_length=step_length,
                cost=cost,
            )
        else:
            args = u(args, step_length, dirs)
            cost = f(args)
            logger.info("Blind update; length %.3e", step_length)

    return args, cost
