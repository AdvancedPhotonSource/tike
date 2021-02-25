import logging

import dxchange
import numpy as np

import tike.align
from . import update_penalty

logger = logging.getLogger(__name__)


def _find_min_max(data):
    mmin = np.zeros(data.shape[0], dtype='float32')
    mmax = np.zeros(data.shape[0], dtype='float32')

    for k in range(data.shape[0]):
        h, e = np.histogram(data[k][:], 1000)
        stend = np.where(h > np.max(h) * 0.005)
        st = stend[0][0]
        end = stend[0][-1]
        mmin[k] = e[st]
        mmax[k] = e[end + 1]

    return mmin, mmax


def _optical_flow_tvl1(unaligned, original, num_iter=16):
    """Wrap scikit-image optical_flow_tvl1 for complex values"""
    from skimage.registration import optical_flow_tvl1
    pflow = [
        optical_flow_tvl1(
            np.angle(original[i]),
            np.angle(unaligned[i]),
            num_iter=num_iter,
        ) for i in range(len(original))
    ]
    # rflow = [
    #     optical_flow_tvl1(
    #         original[i].real,
    #         unaligned[i].real,
    #         num_iter=num_iter,
    #     ) for i in range(len(original))
    # ]
    flow = np.array(pflow, dtype='float32')
    #+ np.array(iflow, dtype='float32')) /2
    flow = np.moveaxis(flow, 1, -1)
    return flow


def _center_of_mass(m, axis=None):
    """Return the center of mass of m along the given axis.

    Parameters
    ----------
    m : array
        Values to find the center of mass from
    axis : tuple(int)
        The axes to find center of mass along.

    Returns
    -------
    center : (..., len(axis)) array[int]
        The shape of center is the shape of m with the dimensions corresponding
        to axis removed plus a new dimension appended whose length is the
        of length of axis in the order of axis.

    """
    centers = []
    for a in range(m.ndim) if axis is None else axis:
        shape = np.ones_like(m.shape)
        shape[a] = m.shape[a]
        x = np.arange(1, m.shape[a] + 1).reshape(*shape).astype(m.dtype)
        centers.append((m * x).sum(axis=axis) / m.sum(axis=axis) - 1)
    return np.stack(centers, axis=-1)


def align(
    # constants
    comm,
    psi,
    angle,
    Hu,
    λ_l,
    ρ_l,
    # updated
    phi,
    λ_p,
    ρ_p,
    flow,
    shift,
    Aφ0=None,
    # parameters
    align_method=False,
    cg_iter=1,
    num_iter=1,
    folder=None,
    save_result=False,
):
    """
    Parameters
    ----------
    psi
        ptychography result. psi = A(phi)
    angle
        alignment rotation angle
    Hu
        Forward model of tomography phi = Hu
    """

    logging.info("Solve alignment subproblem.")

    save_result = False if folder is None else save_result

    aresult = tike.align.reconstruct(
        unaligned=psi if λ_p is None else psi + λ_p / ρ_p,
        original=phi,
        flow=flow,
        shift=shift,
        angle=angle,
        num_iter=cg_iter * num_iter,
        algorithm='cgrad',
        reg=Hu - λ_l / ρ_l,
        rho_p=ρ_p,
        rho_a=ρ_l,
        cval=1.0,
    )
    phi = aresult['original']
    cost = aresult['cost']

    if align_method:

        # TODO: Try combining rotation and flow because they use the same
        # interpolator
        rotated = tike.align.invert(
            psi if λ_p is None else psi + λ_p / ρ_p,
            angle=angle,
            flow=None,
            shift=None,
            unpadded_shape=None,
            cval=1.0,
        )
        padded = tike.align.simulate(
            phi,
            angle=None,
            flow=None,
            shift=None,
            padded_shape=psi.shape,
            cval=1.0,
        )

        if comm.rank == 0 and save_result:
            dxchange.write_tiff(
                np.angle(rotated),
                f'{folder}/rotated-angle-{save_result:03d}.tiff',
                dtype='float32',
            )
            dxchange.write_tiff(
                np.angle(padded),
                f'{folder}/padded-angle-{save_result:03d}.tiff',
                dtype='float32',
            )

        if align_method.lower() == 'flow':
            if shift is not None:
                flow = np.zeros((*rotated.shape, 2), dtype='float32')
                flow[..., :] = shift[..., None, None, :]
                shift = None
            hi, lo = _find_min_max(np.angle(rotated))
            winsize = max(winsize - 1, 32)
            logging.info("Estimate alignment using Farneback.")
            fresult = tike.align.solvers.farneback(
                op=None,
                unaligned=np.angle(rotated),
                original=np.angle(padded),
                flow=flow,
                pyr_scale=0.5,
                levels=4,
                winsize=winsize,
                num_iter=32,
                hi=hi,
                lo=lo,
            )
            flow = fresult['flow']
        elif align_method.lower() == 'tvl1':
            logging.info("Estimate alignment using TV-L1.")
            flow = _optical_flow_tvl1(
                unaligned=rotated,
                original=padded,
                num_iter=cg_iter,
            )
        elif align_method.lower() == 'xcor':
            logging.info("Estimate rigid alignment with cross correlation.")
            sresult = tike.align.reconstruct(
                algorithm='cross_correlation',
                unaligned=rotated,
                original=padded,
                upsample_factor=100,
                reg_weight=0,
            )
            shift = sresult['shift']
        else:
            logging.info("Estimate rigid alignment with center of mass.")
            centers = _center_of_mass(np.abs(np.angle(rotated)), axis=(-2, -1))
            # shift is defined from padded coords to rotated coords
            shift = centers - np.array(rotated.shape[-2:]) / 2

    Aφ = tike.align.simulate(
        phi,
        angle=angle,
        flow=flow,
        shift=shift,
        padded_shape=psi.shape,
        cval=1.0,
    )

    logger.info("Update alignment lambdas and rhos")

    if λ_p is not None:
        λ_p += ρ_p * (psi - Aφ)

    if Aφ0 is not None:
        ρ_p = update_penalty(comm, psi, Aφ, Aφ0, ρ_p)

    Aφ0 = Aφ

    return (
        phi,
        λ_p,
        ρ_p,
        flow,
        shift,
        Aφ0,
        cost,
    )
