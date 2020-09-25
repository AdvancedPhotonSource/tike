import logging
import multiprocessing

import cupy as cp
import dxchange
import numpy as np
import skimage.transform

import tike.align
from tike.communicator import MPICommunicator
import tike.lamino
import tike.ptycho


def update_penalty(comm, psi, h, h0, rho):
    r = np.linalg.norm(psi - h)**2
    s = np.linalg.norm(rho * (h - h0))**2
    r, s = [comm.gather(x) for x in ([r], [s])]
    if comm.rank == 0:
        r = np.sum(r)
        s = np.sum(s)
        if (r > 10 * s):
            rho *= 2
        elif (s > 10 * r):
            rho *= 0.5
    rho = comm.broadcast(rho)
    logging.info(f"Update penalty parameter ρ = {rho}.")
    return rho


def find_min_max(data):
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

# lamino + align converges (ish) according to Viktor, Doga when looking at lagrangian
def lamino_align(
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
):
    """Solve the joint lamino-alignment problem using ADMM.

    Parameters
    ----------
    data : (ntheta, detector, detector) float32
    tilt : radians float32
        The laminography tilt angle in radians.
    theta : float32
        The rotation angle of each data frame in radians.
    u : (detector, detector, detector) complex64
        An initial guess for the object
    lamd : (ntheta, detector, detector) float32
    flow : (ntheta, detector, detector, 2) float32
        An initial guess for the alignmnt displacement field.
    """
    comm = MPICommunicator()

    all_theta = comm.gather(theta)
    if comm.rank == 0:
        np.save(f"{folder}/alltheta", all_theta)

    with cp.cuda.Device(comm.rank):

        logging.info("Solve the ptychography problem.")

        # presult = {
        #     'psi': psi,
        #     'scan': scan,
        #     'probe': probe,
        # }

        # presult = tike.ptycho.reconstruct(
        #     data=data,
        #     algorithm='combined',
        #     num_iter=niter,
        #     cg_iter=4,
        #     recover_psi=True,
        #     recover_probe=True,
        #     recover_positions=False,
        #     model='gaussian',
        #     **presult,
        # )
        # psi = presult['psi']

        # if comm.rank == 0:
        #     dxchange.write_tiff(
        #         presult['psi'].real,
        #         f'{folder}/psi-real-{(1):03d}.tiff',
        #         dtype='float32',
        #     )
        #     dxchange.write_tiff(
        #         presult['psi'].imag,
        #         f'{folder}/psi-imag-{(1):03d}.tiff',
        #         dtype='float32',
        #     )

        # logging.info("Rotate and crop projections.")

        # _, trimmed, _ = rotate_and_crop(psi.copy())

        # Set preliminary values for ADMM
        w = 256
        flow = np.zeros(
            [len(theta), w, w, 2],
            dtype='float32',
        ) if flow is None else flow
        winsize = w

        u = np.zeros(
            [w, w, w],
            dtype='complex64',
        ) if u is None else u
        phi = np.exp(1j * tike.lamino.simulate(
            obj=u,
            tilt=tilt,
            theta=theta,
        ))
        phi = phi.astype('complex64')
        Hu = phi.copy()
        Hu0 = Hu

        λ_a = np.zeros_like(phi)
        rho = 0.5

        all_trimmed = None
        if comm.rank == 0:
            all_trimmed = dxchange.read_tiff(
                f'{folder}/trimmed-real-{(1):03d}.tiff'
            ) + 1j * dxchange.read_tiff(f'{folder}/trimmed-imag-{(1):03d}.tiff')
            all_trimmed = all_trimmed.astype('complex64')
        trimmed = comm.scatter(all_trimmed)

        # all_trimmed = comm.gather(trimmed)
        # if comm.rank == 0:
        #     dxchange.write_tiff(
        #         all_trimmed.real,
        #         f'{folder}/trimmed-real-{(1):03d}.tiff',
        #         dtype='float32',
        #     )
        #     dxchange.write_tiff(
        #         all_trimmed.imag,
        #         f'{folder}/trimmed-imag-{(1):03d}.tiff',
        #         dtype='float32',
        #     )
        del all_trimmed

        for k in range(niter):

            logging.info("Recover original/aligned projections.")

            aresult = tike.align.reconstruct(
                unaligned=trimmed,
                original=phi,
                flow=flow,
                num_iter=4,
                algorithm='cgrad',
                reg=Hu + λ_a / rho,
                rho=rho,
            )
            phi = aresult['original']

            logging.info("Find flow using farneback.")

            fresult = tike.align.solvers.farneback(
                op=None,
                unaligned=trimmed,
                original=phi,
                flow=flow,
                pyr_scale=0.5,
                levels=1,
                winsize=winsize,
                num_iter=4,
            )
            flow = fresult['shift']

            # Gather all to one thread
            λ_a, phi, theta = [comm.gather(x) for x in (λ_a, phi, theta)]

            if comm.rank == 0:
                logging.info('Solve the laminography problem.')

                result = tike.lamino.reconstruct(
                    data=-1j * np.log(phi - λ_a / rho),
                    theta=theta,
                    tilt=tilt,
                    obj=u,
                    algorithm='cgrad',
                    num_iter=1,
                )
                u = result['obj']

                # We cannot reorder phi, theta without ruining correspondence
                # with data, psi, etc, but we can reorder the saved array
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
            λ_a, phi, theta = [comm.scatter(x) for x in (λ_a, phi, theta)]
            u = comm.broadcast(u)

            logging.info('Update lambda and rho.')

            CψDφ = tike.align.simulate(phi, flow) - trimmed
            Hu = np.exp(1j * tike.lamino.simulate(
                obj=u,
                tilt=tilt,
                theta=theta,
            ))
            φHu = Hu - phi
            λ_a = λ_a + rho * φHu
            rho = update_penalty(phi, Hu, Hu0, rho)
            Hu0 = Hu

            lagrangian = [
                [np.linalg.norm(CψDφ.ravel())**2],
                [2 * np.real(λ_a.conj() * φHu)],
                [rho * np.linalg.norm(φHu.ravel())**2],
            ]
            lagrangian = [comm.gather(x) for x in lagrangian]

            if comm.rank == 0:
                lagrangian = [np.sum(x) for x in lagrangian]
                print(
                    f"k: {k:03d}, ρ: {rho:.3e}, winsize: {winsize:03d}, "
                    "Lagrangian: {:+6.3e} = {:+6.3e} {:+6.3e} {:+6.3e}".format(
                        np.sum(lagrangian), *lagrangian),
                    flush=True,
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
                np.save(f"{folder}/flow-tike-{(k+1):03d}", flow)

            # Limit winsize to larger value. 20?
            if winsize > 20:
                winsize -= 1

    return u


# lamino by itself works converges
def lamino(
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
):
    """Solve the joint lamino-alignment problem using ADMM.

    Parameters
    ----------
    data : (ntheta, detector, detector) float32
    tilt : radians float32
        The laminography tilt angle in radians.
    theta : float32
        The rotation angle of each data frame in radians.
    u : (detector, detector, detector) complex64
        An initial guess for the object
    lamd : (ntheta, detector, detector) float32
    flow : (ntheta, detector, detector, 2) float32
        An initial guess for the alignmnt displacement field.
    """
    comm = MPICommunicator()

    all_theta = comm.gather(theta)
    if comm.rank == 0:
        np.save(f"{folder}/alltheta", all_theta)

    with cp.cuda.Device(comm.rank):

        if comm.rank == 0:
            all_trimmed = dxchange.read_tiff(
                f'{folder}/trimmed-real-{(1):03d}.tiff'
            ) + 1j * dxchange.read_tiff(f'{folder}/trimmed-imag-{(1):03d}.tiff')
            phi = all_trimmed.astype('complex64')

        for k in range(niter):

            if comm.rank == 0:
                logging.info('Solve the laminography problem.')

                lresult = tike.lamino.reconstruct(
                    data=-1j * np.log(phi),
                    theta=all_theta,
                    tilt=tilt,
                    obj=u,
                    algorithm='cgrad',
                    num_iter=1,
                )
                u = lresult['obj']

                Hu = np.exp(1j * tike.lamino.simulate(
                    obj=u,
                    tilt=tilt,
                    theta=all_theta,
                ))

                print(
                    f"k: {k:03d}, "
                    "lamino: {:+6.3e}".format(lresult['cost']),
                    flush=True,
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

    return u


def rotate_and_crop(x, corners=None, radius=128, angle=72.035):
    """Rotate x in two trailing dimensions then crop around center-of-mass.

    Parameters
    ----------
    x : (M, N, O) complex64
        The image to be cropped.
    radius : int
        How much to crop around the center-of-mass along each dimension.
    angle : float
        Rotation angle in degrees.
    corners : (len(x), 2) float32
        Crop at these positions instead of the center-of-mass.
    """
    rotate_params = dict(
        angle=angle,
        clip=False,
        preserve_range=True,
        resize=False,
        mode='edge',
    )
    corners = -np.ones((len(x), 2), dtype=int) if corners is None else corners
    patch = np.zeros((len(x), 2 * radius, 2 * radius), dtype='complex64')
    for i in range(len(x)):
        # Rotate by desired angle (degrees)
        x[i].real = skimage.transform.rotate(x[i].real, **rotate_params)
        x[i].imag = skimage.transform.rotate(x[i].imag, **rotate_params)

        if corners[i][0] < 0:
            # Find the center of mass
            phase = np.angle(x[i])
            phase[phase < 0] = 0
            M = skimage.measure.moments(phase, order=1)
            center = np.array([M[1, 0] / M[0, 0],
                               M[0, 1] / M[0, 0]]).astype('int')

            # Adjust the cropping region so it stays within the image
            lo = np.fmax(0, center - radius)
            hi = lo + 2 * radius
            shift = np.fmin(0, x[i].shape - hi)
            hi += shift
            lo += shift
            assert np.all(lo >= 0), lo
            assert np.all(hi <= x[i].shape), (hi, x[i].shape)
            corners[i] = lo
        else:
            lo = corners[i]
            hi = corners[i] + 2 * radius

        # Crop image
        patch[i] = x[i][lo[0]:hi[0], lo[1]:hi[1]]

    return x, patch, corners


def uncrop_and_rotate(x, patch, lo, radius=128, angle=-72.035):
    lo = np.zeros((len(x), 2), dtype=int) if lo is None else lo
    rotate_params = dict(
        angle=angle,
        clip=False,
        preserve_range=True,
        resize=False,
        mode='edge',
    )
    for i in range(len(x)):
        x[i][lo[i][0]:lo[i][0] + 2 * radius,
             lo[i][1]:lo[i][1] + 2 * radius] = patch[i]
        # Rotate by desired angle (degrees)
        x[i].real = skimage.transform.rotate(x[i].real, **rotate_params)
        x[i].imag = skimage.transform.rotate(x[i].imag, **rotate_params)
    return x


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
):
    """Solve the joint ptycho-lamino-alignment problem using ADMM."""
    comm = MPICommunicator()
    with cp.cuda.Device(comm.rank):
        # Set preliminary values for ADMM
        w = 256 + 32
        flow = np.zeros(
            [len(theta), w, w, 2],
            dtype='float32',
        ) if flow is None else flow
        winsize = w
        corners = None

        u = np.zeros(
            [w, w, w],
            dtype='complex64',
        ) if u is None else u

        phi = np.exp(1j * tike.lamino.simulate(
            obj=u,
            tilt=tilt,
            theta=theta,
        )).astype('complex64')

        Hu = phi.copy()
        Hu0 = phi.copy()
        Dφ0 = phi.copy()

        presult = {  # ptychography result
            'psi': psi,
            'scan': scan,
            'probe': probe,
        }

        λ_p = np.zeros_like(phi)
        ρ_p = 0.5
        λ_a = np.zeros_like(phi)
        ρ_a = 0.5

        for k in range(niter):
            logging.info(f"Start ADMM iteration {k}.")

            logging.info("Solve the ptychography problem.")

            if k > 0:
                reg_p = uncrop_and_rotate(
                    x=psi_rotated,
                    patch=λ_p / ρ_p - Dφ,
                    lo=corners,
                    radius=w // 2,
                )
            else:
                reg_p = None
            presult = tike.ptycho.reconstruct(
                data=data,
                reg=reg_p,
                rho=ρ_p,
                algorithm='combined',
                num_iter=1,
                cg_iter=4,
                recover_psi=True,
                recover_probe=True,
                recover_positions=False,
                model='gaussian',
                **presult,
            )
            psi = presult['psi']

            logging.info("Rotate and crop projections.")
            psi_rotated, trimmed, corners = rotate_and_crop(
                x=psi.copy(),
                corners=corners if fixed_crop else None,
                radius=w // 2,
            )

            logging.info("Recover aligned projections from unaligned.")
            aresult = tike.align.reconstruct(
                unaligned=trimmed + λ_p / ρ_p,
                original=phi,
                flow=flow,
                num_iter=4,
                algorithm='cgrad',
                reg=Hu - λ_a / ρ_a,
                rho_p=ρ_p,
                rho_a=ρ_a,
            )
            phi = aresult['original']

            logging.info("Estimate alignment using Farneback.")
            fresult = tike.align.solvers.farneback(
                op=None,
                unaligned=trimmed + λ_p / ρ_p,
                original=phi,
                flow=flow,
                pyr_scale=0.5,
                levels=1,
                winsize=winsize,
                num_iter=4,
            )
            flow = fresult['shift']

            # Gather all to one thread
            λ_a, phi, theta = [comm.gather(x) for x in (λ_a, phi, theta)]

            if comm.rank == 0:
                logging.info('Solve the laminography problem.')
                lresult = tike.lamino.reconstruct(
                    data=-1j * np.log(phi + λ_a / ρ_a),
                    theta=theta,
                    tilt=tilt,
                    obj=u,
                    algorithm='cgrad',
                    num_iter=1,
                    cg_iter=4,
                )
                u = lresult['obj']

                # We cannot reorder phi, theta without ruining correspondence
                # with data, psi, etc, but we can reorder the saved array
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
            λ_a, phi, theta = [comm.scatter(x) for x in (λ_a, phi, theta)]
            u = comm.broadcast(u)

            logging.info('Update lambdas and rhos.')

            Dφ = tike.align.simulate(phi, flow)
            CψDφ = trimmed - Dφ
            Hu = np.exp(1j * tike.lamino.simulate(
                obj=u,
                tilt=tilt,
                theta=theta,
            ))
            φHu = phi - Hu
            λ_p += ρ_p * CψDφ
            λ_a += ρ_a * φHu

            ρ_p = update_penalty(comm, trimmed, Dφ, Dφ0, ρ_p)
            ρ_a = update_penalty(comm, phi, Hu, Hu0, ρ_a)
            Hu0 = Hu
            Dφ0 = Dφ

            lagrangian = (
                [presult['cost']],
                [
                    2 * np.real(λ_p.conj() * CψDφ) +
                    ρ_p * np.linalg.norm(CψDφ.ravel())**2
                ],
                [
                    2 * np.real(λ_a.conj() * φHu) +
                    ρ_a * np.linalg.norm(φHu.ravel())**2
                ],
            )
            lagrangian = [comm.gather(x) for x in lagrangian]
            acost = comm.gather([aresult['cost']])

            if comm.rank == 0:
                lagrangian = [np.sum(x) for x in lagrangian]
                print(
                    f"k: {k:03d}, ρ_p: {ρ_p:6.3e}, ρ_a: {ρ_a:6.3e}, "
                    f"winsize: {winsize:03d}, "
                    f"alignment: {np.sum(acost):+6.3e} "
                    f"laminography: {lresult['cost']:+6.3e} "
                    'Lagrangian: {:+6.3e} = {:+6.3e} {:+6.3e} {:+6.3e}'.format(
                        np.sum(lagrangian), *lagrangian),
                    flush=True,
                )
                dxchange.write_tiff(
                    trimmed.real,
                    f'{folder}/psi-real-{(k+1):03d}.tiff',
                    dtype='float32',
                )
                dxchange.write_tiff(
                    trimmed.imag,
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
                np.save(f"{folder}/flow-tike-{(k+1):03d}", flow)

            # Limit winsize to larger value. 20?
            if winsize > 20:
                winsize -= 1

    result = presult
    return result

# Does not converge!?
def ptycho_lamino(
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
    fixed_crop=True,
):
    """Solve the joint ptycho-lamino-alignment problem using ADMM."""
    comm = MPICommunicator()
    with cp.cuda.Device(comm.rank):
        # Set preliminary values for ADMM
        w = 256 + 32
        corners = None

        u = np.zeros(
            [w, w, w],
            dtype='complex64',
        ) if u is None else u

        Hu0 = np.exp(1j * tike.lamino.simulate(
            obj=u,
            tilt=tilt,
            theta=theta,
        )).astype('complex64')
        TPHu0 = uncrop_and_rotate(
            x=np.zeros_like(psi) + 1.0,
            patch=Hu0,
            lo=corners,
            radius=w // 2,
        )

        presult = {  # ptychography result
            'psi': psi,
            'scan': scan,
            'probe': probe,
        }

        λ_p = np.zeros_like(psi)
        ρ_p = 0.5

        for k in range(niter):
            logging.info(f"Start ADMM iteration {k}.")

            logging.info("Solve the ptychography problem.")

            if k > 0:
                reg_p = λ_p / ρ_p - TPHu
            else:
                reg_p = None
            presult = tike.ptycho.reconstruct(
                data=data,
                reg=reg_p,
                rho=ρ_p,
                algorithm='combined',
                num_iter=1,
                cg_iter=4,
                recover_psi=True,
                recover_probe=True,
                recover_positions=False,
                model='gaussian',
                **presult,
            )
            psi = presult['psi']

            logging.info("Rotate and crop projections.")
            _, trimmed, corners = rotate_and_crop(
                x=psi + λ_p / ρ_p,
                corners=corners if fixed_crop else None,
                radius=w // 2,
            )

            # Gather all to one thread
            trimmed, theta = [comm.gather(x) for x in (trimmed, theta)]

            if comm.rank == 0:
                logging.info('Solve the laminography problem.')
                lresult = tike.lamino.reconstruct(
                    data=-1j * np.log(trimmed),
                    theta=theta,
                    tilt=tilt,
                    obj=u,
                    algorithm='cgrad',
                    num_iter=1,
                    cg_iter=4,
                )
                u = lresult['obj']

                # We cannot reorder phi, theta without ruining correspondence
                # with data, psi, etc, but we can reorder the saved array
                order = np.argsort(theta)
                dxchange.write_tiff(
                    trimmed[order].real,
                    f'{folder}/phi-real-{(k+1):03d}.tiff',
                    dtype='float32',
                )
                dxchange.write_tiff(
                    trimmed[order].imag,
                    f'{folder}/phi-imag-{(k+1):03d}.tiff',
                    dtype='float32',
                )

            # Separate again to multiple threads
            trimmed, theta = [comm.scatter(x) for x in (trimmed, theta)]
            u = comm.broadcast(u)

            logging.info('Update lambdas and rhos.')

            Hu = np.exp(1j * tike.lamino.simulate(
                obj=u,
                tilt=tilt,
                theta=theta,
            ))
            TPHu = uncrop_and_rotate(
                x=np.zeros_like(psi) + 1.0,
                patch=Hu,
                lo=corners,
                radius=w // 2,
            )
            ψTPHu = psi - TPHu
            λ_p += ρ_p * ψTPHu

            ρ_p = update_penalty(comm, psi, TPHu, TPHu0, ρ_p)
            TPHu0 = TPHu

            lagrangian = (
                [presult['cost']],
                [
                    2 * np.real(λ_p.conj() * ψTPHu) +
                    ρ_p * np.linalg.norm(ψTPHu.ravel())**2
                ],
            )
            lagrangian = [comm.gather(x) for x in lagrangian]

            if comm.rank == 0:
                lagrangian = [np.sum(x) for x in lagrangian]
                print(
                    f"k: {k:03d}, ρ_p: {ρ_p:6.3e}, "
                    f"laminography: {lresult['cost']:+6.3e} "
                    'Lagrangian: {:+6.3e} = {:+6.3e} {:+6.3e}'.format(
                        np.sum(lagrangian), *lagrangian),
                    flush=True,
                )
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
                    TPHu.real,
                    f'{folder}/TPHu-real-{(k+1):03d}.tiff',
                    dtype='float32',
                )
                dxchange.write_tiff(
                    TPHu.imag,
                    f'{folder}/TPHu-imag-{(k+1):03d}.tiff',
                    dtype='float32',
                )
                dxchange.write_tiff(
                    (λ_p / ρ_p).imag,
                    f'{folder}/lamb-imag-{(k+1):03d}.tiff',
                    dtype='float32',
                )
                dxchange.write_tiff(
                    (λ_p / ρ_p).real,
                    f'{folder}/lamb-real-{(k+1):03d}.tiff',
                    dtype='float32',
                )

    result = presult
    return result
