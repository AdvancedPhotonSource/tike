import logging

from .admm import update_penalty
import tike.align

logger = logging.getLogger(__name__)


def subproblem(
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
        unaligned=psi + λ_p / ρ_p,
        original=phi,
        flow=flow,
        shift=shift,
        angle=angle,
        num_iter=cg_iter,
        algorithm='cgrad',
        reg=Hu - λ_l / ρ_l,
        rho_p=ρ_p,
        rho_a=ρ_l,
        cval=1.0,
    )
    phi = aresult['original']

    if align_method:

        hi, lo = find_min_max(np.angle(psi + λ_p / ρ_p))

        rotated = tike.align.simulate(
            psi + λ_p / ρ_p,
            angle=-angle,
            flow=None,
            shift=None,
            padded_shape=None,
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
        winsize = max(winsize - 1, 128)
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
        flow = optical_flow_tvl1(
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
        )
        # Limit shift change per iteration
        if shift is None:
            shift = np.clip(sresult['shift'], -16, 16)
        else:
            shift += np.clip(sresult['shift'] - shift, -16, 16)

    Aφ = tike.align.simulate(
        phi,
        angle=angle,
        flow=flow,
        shift=shift,
        padded_shape=psi.shape,
        cval=1.0,
    )
    ψAφ = psi - Aφ

    logger.info("Update alignment lambdas and rhos")

    λ_p += ρ_p * ψAφ

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
    )
