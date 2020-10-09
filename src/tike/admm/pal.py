import logging


import numpy as np
import cupy as cp
import dxchange

import tike.align
from tike.communicator import MPICommunicator
import tike.ptycho
import tike.lamino

from .admm import *

def ptycho_lamino_align(
    data,
    psi,
    scan,
    probe,
    theta,
    tilt,
    u=None,
    flow=None,
    niter=1,
    folder=None,
    fixed_crop=False,
    angle=0,
    w=256 + 64 + 16 + 8,
    cg_iter=4,
    shift=None,
    interval=8,
    winsize=None,
    align_method='',
):
    """Solve the joint ptycho-lamino-alignment problem using ADMM."""
    u = np.zeros((w, w, w), dtype='complex64')
    Hu = np.ones((len(theta), w, w), dtype='complex64')
    phi = Hu
    TDPφ = np.ones(psi.shape, dtype='complex64')
    flow = np.zeros([*psi.shape, 2], dtype='float32')
    winsize = min(*psi.shape[-2:]) if winsize is None else winsize
    presult = {  # ptychography result
            'psi': np.ones(psi.shape, dtype='complex64'),
            'scan': scan,
            'probe': probe,
    }
    λ_p = np.zeros_like(psi)
    ρ_p = 0.5
    λ_l = np.zeros([len(theta), w, w], dtype='complex64')
    ρ_l = 0.5
    comm = MPICommunicator()
    with cp.cuda.Device(comm.rank if comm.size > 1 else None):

        for k in range(niter):
            logging.info(f"Start ADMM iteration {k}.")

            logging.info("Solve the ptychography problem.")
            presult = tike.ptycho.reconstruct(
                data=data,
                reg=λ_p / ρ_p - TDPφ,
                rho=ρ_p,
                algorithm='combined',
                num_iter=1,
                cg_iter=cg_iter,
                recover_psi=True,
                recover_probe=True,
                recover_positions=False,
                model='gaussian',
                **presult,
            )
            psi = presult['psi']

            rotated = tike.align.simulate(
                psi + λ_p / ρ_p,
                angle=-angle,
                flow=None,
                padded_shape=None,
                cval=1.0,
            )
            if (k + 1) == 8:
                dxchange.write_tiff(
                    rotated.real,
                    f'{folder}/{comm.rank}-rotated-real-{(k+1):03d}.tiff',
                    dtype='float32',
                )
                dxchange.write_tiff(
                    rotated.imag,
                    f'{folder}/{comm.rank}-rotated-imag-{(k+1):03d}.tiff',
                    dtype='float32',
                )

            logging.info("Recover aligned projections from unaligned.")
            aresult = tike.align.reconstruct(
                unaligned=psi + λ_p / ρ_p,
                original=phi,
                flow=flow,
                angle=angle,
                num_iter=cg_iter,
                algorithm='cgrad',
                reg=Hu - λ_l / ρ_l,
                rho_p=ρ_p,
                rho_a=ρ_l,
                cval=1.0,
            )
            phi = aresult['original']

            padded = tike.align.simulate(
                phi,
                angle=None,
                flow=None,
                padded_shape=psi.shape,
                cval=1.0,
            )
            if comm.rank == 0 and (k + 1) % interval == 0:
                dxchange.write_tiff(
                    rotated.real,
                    f'{folder}/rotated-real-{(k+1):03d}.tiff',
                    dtype='float32',
                )
                dxchange.write_tiff(
                    rotated.imag,
                    f'{folder}/rotated-imag-{(k+1):03d}.tiff',
                    dtype='float32',
                )
                dxchange.write_tiff(
                    padded.real,
                    f'{folder}/padded-real-{(k+1):03d}.tiff',
                    dtype='float32',
                )
                dxchange.write_tiff(
                    padded.imag,
                    f'{folder}/padded-imag-{(k+1):03d}.tiff',
                    dtype='float32',
                )

            if align_method.lower() == 'flow':
                winsize = max(winsize - 1, 24)
                logging.info("Estimate alignment using Farneback.")
                fresult = tike.align.solvers.farneback(
                    op=None,
                    unaligned=rotated,
                    original=padded,
                    flow=flow,
                    pyr_scale=0.5,
                    levels=4,
                    winsize=winsize,
                    num_iter=32,
                )
                flow = fresult['flow']
            elif align_method.lower() == 'tvl1':
                logging.info("Estimate alignment using TV-L1.")
                flow = optical_flow_tvl1(
                    unaligned=rotated,
                    original=padded,
                    num_iter=8,
                )
            else:
                logging.info("Estimate rigid alignment with cross correlation.")
                sresult = tike.align.reconstruct(
                    algorithm='cross_correlation',
                    unaligned=rotated,
                    original=padded,
                    upsample_factor=100,
                )
                flow[:] = sresult['shift'][:, None, None, :]

            # Gather all to one thread
            λ_l, phi, theta = [comm.gather(x) for x in (λ_l, phi, theta)]

            if comm.rank == 0:
                logging.info('Solve the laminography problem.')
                lresult = tike.lamino.reconstruct(
                    data=-1j * np.log(phi + λ_l / ρ_l),
                    theta=theta,
                    tilt=tilt,
                    obj=u,
                    algorithm='cgrad',
                    num_iter=1,
                    cg_iter=cg_iter,
                )
                u = lresult['obj']

                # We cannot reorder phi, theta without ruining correspondence
                # with data, psi, etc, but we can reorder the saved array
                if (k + 1) % interval == 0:
                    order = np.argsort(theta)
                    dxchange.write_tiff(
                        phi[order].real,
                        f'{folder}/phi-real-{(k+1):03d}.tiff',
                        dtype='float32',
                    )
                    dxchange.write_tiff(
                        phi[order].imag,
                        f'{folder}/phi-imag-{(k+1):03d}.tiff',
                        dtype='float32',
                    )

            # Separate again to multiple threads
            λ_l, phi, theta = [comm.scatter(x) for x in (λ_l, phi, theta)]
            u = comm.broadcast(u)

            logging.info('Update lambdas and rhos.')

            TDPφ = tike.align.simulate(
                phi,
                angle=angle,
                flow=flow,
                padded_shape=psi.shape,
                cval=1.0,
            )
            ψTDPφ = psi - TDPφ
            Hu = np.exp(1j * tike.lamino.simulate(
                obj=u,
                tilt=tilt,
                theta=theta,
            ))
            φHu = phi - Hu
            λ_p += ρ_p * ψTDPφ
            λ_l += ρ_l * φHu

            if k > 0:
                ρ_p = update_penalty(comm, psi, TDPφ, TDPφ0, ρ_p)
                ρ_l = update_penalty(comm, phi, Hu, Hu0, ρ_l)
            Hu0 = Hu
            TDPφ0 = TDPφ

            lagrangian = (
                [presult['cost']],
                [
                    2 * np.real(λ_p.conj() * ψTDPφ) +
                    ρ_p * np.linalg.norm(ψTDPφ.ravel())**2
                ],
                [
                    2 * np.real(λ_l.conj() * φHu) +
                    ρ_l * np.linalg.norm(φHu.ravel())**2
                ],
            )
            lagrangian = [comm.gather(x) for x in lagrangian]
            acost = comm.gather([aresult['cost']])

            if comm.rank == 0:
                lagrangian = [np.sum(x) for x in lagrangian]
                print_log_line(
                    k=k,
                    ρ_p=ρ_p,
                    ρ_l=ρ_l,
                    # shift=np.linalg.norm(shift[:, None, None, :] - flow),
                    winsize=winsize,
                    alignment=np.sum(acost),
                    laminography=float(lresult['cost']),
                    Lagrangian=np.sum(lagrangian),
                    dGψ=lagrangian[0],
                    ψDφ=lagrangian[1],
                    φHu=lagrangian[2],
                )
            if comm.rank == 0 and (k + 1) % interval == 0:
                dxchange.write_tiff(
                    psi.real,
                    f'{folder}/psi-real-{(k+1):03d}.tiff',
                    dtype='float32',
                )
                dxchange.write_tiff(
                    psi.imag,
                    f'{folder}/psi-imag-{(k+1):03d}.tiff',
                    dtype='float32',
                )
                dxchange.write_tiff(
                    u.real,
                    f'{folder}/particle-real-{(k+1):03d}.tiff',
                    dtype='float32',
                )
                dxchange.write_tiff(
                    u.imag,
                    f'{folder}/particle-imag-{(k+1):03d}.tiff',
                    dtype='float32',
                )
                dxchange.write_tiff(
                    Hu.real,
                    f'{folder}/Hu-real-{(k+1):03d}.tiff',
                    dtype='float32',
                )
                dxchange.write_tiff(
                    Hu.imag,
                    f'{folder}/Hu-imag-{(k+1):03d}.tiff',
                    dtype='float32',
                )
                np.save(f"{folder}/flow-{(k+1):03d}", flow)

    return u
