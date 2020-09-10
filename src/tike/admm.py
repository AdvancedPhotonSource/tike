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


def update_penalty(psi, h, h0, rho):
    r = np.linalg.norm(psi - h)**2
    s = np.linalg.norm(rho * (h - h0))**2
    if (r > 10 * s):
        rho *= 2
    elif (s > 10 * r):
        rho *= 0.5
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


def lamino_align(data, tilt, theta, u=None, flow=None, niter=8, rho=0.5):
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
    ntheta, _, det = data.shape
    u = np.zeros([det, det, det], dtype='complex64') if u is None else u
    lamd = np.zeros([ntheta, det, det], dtype='float32')
    flow = np.zeros([ntheta, det, det, 2],
                    dtype='float32') if flow is None else flow

    psi = data.copy()
    h0 = psi.copy()
    # Start with large winsize and decrease each ADMM iteration.
    winsize = min(*data.shape[1:])
    mmin, mmax = find_min_max(data.real)

    for k in range(niter):

        logging.info("Find flow using farneback.")
        result = tike.align.solvers.farneback(
            op=None,
            unaligned=data,
            original=psi,
            flow=flow,
            pyr_scale=0.5,
            levels=1,
            winsize=winsize,
            num_iter=4,
            hi=mmax,
            lo=mmin,
        )
        flow = result['shift']

        logging.info("Recover original/aligned projections.")
        result = tike.align.reconstruct(
            unaligned=data,
            original=psi,
            flow=flow,
            num_iter=4,
            algorithm='cgrad',
            reg=tike.lamino.simulate(
                obj=u,
                tilt=tilt,
                theta=theta,
            ) + lamd / rho,
            rho=rho,
        )
        psi = result['original']

        logging.info('Solve the laminography problem.')
        result = tike.lamino.reconstruct(
            data=psi - lamd / rho,
            theta=theta,
            tilt=tilt,
            obj=u,
            algorithm='cgrad',
            num_iter=1,
        )
        u = result['obj']

        logging.info('Update lambda and rho.')
        h = tike.lamino.simulate(obj=u, theta=theta, tilt=tilt)
        lamd = lamd + rho * (h - psi)
        rho = update_penalty(psi, h, h0, rho)
        h0 = h

        np.save(f"flow-tike-{(k+1):03d}", flow)
        # np.save(f"flow-tike-v-{(k+1):03d}", vflow[..., ::-1])

        # checking intermediate results
        lagr = [
            np.linalg.norm((tike.align.simulate(psi, flow) - data))**2,
            np.sum(np.real(np.conj(lamd) * (h - psi))),
            rho * np.linalg.norm(h - psi)**2,
        ]
        print(
            "k: {:03d}, ρ: {:.3e}, winsize: {:03d}, flow: {:.3e}, "
            " lagrangian: {:.3e}, {:.3e}, {:.3e} = {:.3e}".format(
                k,
                rho,
                winsize,
                np.linalg.norm(flow),
                *lagr,
                np.sum(lagr),
            ),
            flush=True,
        )

        # Limit winsize to larger value. 20?
        if winsize > 20:
            winsize -= 1

        if (k + 1) % 10 == 0:
            dxchange.write_tiff(
                np.imag(u),
                f'particle-i-{(k+1):03d}.tiff',
                dtype='float32',
            )
            dxchange.write_tiff(
                np.real(u),
                f'particle-r-{(k+1):03d}.tiff',
                dtype='float32',
            )

    return u


def rotate_and_crop(x, radius=128, angle=72.035):
    """Rotate x in two trailing dimensions then crop around center-of-mass.

    Parameters
    ----------
    x : (M, N, O) complex64
    radius : int
    angle : float
        Rotation angle in degrees.
    """
    rotate_params = dict(
        angle=angle,
        clip=False,
        preserve_range=True,
        resize=False,
    )
    corner = np.zeros((len(x), 2), dtype=int)
    patch = np.zeros((len(x), 2 * radius, 2 * radius), dtype='complex64')
    for i in range(len(x)):
        # Rotate by desired angle (degrees)
        x[i].real = skimage.transform.rotate(x[i].real, **rotate_params, cval=1)
        x[i].imag = skimage.transform.rotate(x[i].imag, **rotate_params, cval=0)

        # Find the center of mass
        phase = np.angle(x[i])
        phase[phase < 0] = 0
        M = skimage.measure.moments(phase, order=1)
        center = np.array([M[1, 0] / M[0, 0], M[0, 1] / M[0, 0]]).astype('int')

        # Adjust the cropping region so it stays within the image
        lo = np.fmax(0, center - radius)
        hi = lo + 2 * radius
        shift = np.fmin(0, x[i].shape - hi)
        hi += shift
        lo += shift
        assert np.all(lo >= 0), lo
        assert np.all(hi <= x[i].shape), (hi, x[i].shape)
        # Crop image
        patch[i] = x[i][lo[0]:hi[0], lo[1]:hi[1]]
        corner[i] = lo

    return x, patch, corner


def uncrop_and_rotate(x, patch, lo, radius=128, angle=-72.035):
    rotate_params = dict(
        angle=angle,
        clip=False,
        preserve_range=True,
        resize=False,
    )
    for i in range(len(x)):
        x[i][lo[i][0]:lo[i][0] + 2 * radius,
             lo[i][1]:lo[i][1] + 2 * radius] = patch[i]
        # Rotate by desired angle (degrees)
        x[i].real = skimage.transform.rotate(x[i].real, **rotate_params, cval=1)
        x[i].imag = skimage.transform.rotate(x[i].imag, **rotate_params, cval=0)
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
):
    """Solve the joint ptycho-lamino-alignment problem using ADMM."""
    comm = MPICommunicator()
    with cp.cuda.Device(comm.rank):
        # Set initial values for intermediate variables
        w = 256
        u = np.zeros(
            [w, w, w],
            dtype='complex64',
        ) if u is None else u
        winsize = min(*u.shape[:2])

        # data, psi, scan, probe, theta = [
        #     comm.scatter(x) for x in (data, psi, scan, probe, theta)
        # ]

        flow = np.zeros(
            [len(theta), w, w, 2],
            dtype='float32',
        ) if flow is None else flow
        presult = {  # ptychography result
            'psi': psi,
            'scan': scan,
            'probe': probe,
        }
        phi = np.exp(1j * tike.lamino.simulate(obj=u, tilt=tilt, theta=theta))
        phi = phi.astype('complex64')
        λ_p = np.zeros_like(phi)
        λ_a = np.zeros_like(phi)
        reg_p = np.zeros_like(psi)

        for k in range(niter):
            logging.info(f"Start ADMM iteration {k}.")

            logging.info("Solve the ptychography problem.")

            if k > 0:
                # Skip regularization on zeroth iteration because we don't know
                # value of the cropping corner locations
                reg_p = uncrop_and_rotate(
                    psi_rotated,
                    λ_p / 0.5 - tike.align.simulate(phi, flow),
                    corners,
                )
            presult = tike.ptycho.reconstruct(
                data=data,
                reg=reg_p,
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
            psi_rotated, trimmed, corners = rotate_and_crop(psi.copy())

            logging.info("Recover aligned projections from unaligned.")
            aresult = tike.align.reconstruct(
                unaligned=trimmed + λ_p / 0.5,
                original=phi,
                flow=flow,
                num_iter=4,
                algorithm='cgrad',
                reg=np.exp(1j * tike.lamino.simulate(
                    obj=u,
                    tilt=tilt,
                    theta=theta,
                )) - λ_a / 0.5,
            )
            phi = aresult['original']

            logging.info("Estimate alignment using Farneback.")
            aresult = tike.align.solvers.farneback(
                op=None,
                unaligned=trimmed + λ_p / 0.5,
                original=phi,
                flow=flow,
                pyr_scale=0.5,
                levels=1,
                winsize=winsize,
                num_iter=4,
            )
            flow = aresult['shift']

            # Gather all to one thread
            λ_a, phi, theta = [comm.gather(x) for x in (λ_a, phi, theta)]

            if comm.rank == 0:
                logging.info('Solve the laminography problem.')
                lresult = tike.lamino.reconstruct(
                    data=-1j * np.log(phi + λ_a / 0.5),
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

            λ_p += 0.5 * (trimmed - tike.align.simulate(phi, flow))
            λ_a += 0.5 * (phi - np.exp(
                1j * tike.lamino.simulate(obj=u, tilt=tilt, theta=theta)))

            # Limit winsize to larger value. 20?
            if winsize > 20:
                winsize -= 1

            if (k + 1) % 1 == 0 and comm.rank == 0:
                dxchange.write_tiff(
                    skimage.restoration.unwrap_phase(np.angle(
                        presult['psi'])).astype('float32'),
                    f'{folder}/object-phase-{(k+1):03d}.tiff',
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

    result = presult
    return result
