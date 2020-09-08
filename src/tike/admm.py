import logging
import multiprocessing

import cupy as cp
import dxchange
import numpy as np
import skimage.transform

import tike.align
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


def recon_with_device(device, data, psi, scan, probe, reg):
    with cp.cuda.Device(device):
        result = tike.ptycho.reconstruct(
            data=data,
            rho=0.5,
            reg=reg,
            psi=psi,
            scan=scan,
            probe=probe,
            algorithm='combined',
            num_iter=1,
            cg_iter=4,
            recover_psi=True,
            recover_probe=True,
            recover_positions=False,
            model='gaussian',
        )
    return result['psi'], result['scan'], result['probe']


def multi_ptycho(processes, data, probe, scan, psi, reg, num_gpu=8):

    nsplit = len(data) // 6
    data_ = np.array_split(data, nsplit, axis=0)
    psi_ = np.array_split(psi, nsplit, axis=0)
    scan_ = np.array_split(scan, nsplit, axis=0)
    probe_ = np.array_split(probe, nsplit, axis=0)
    reg_ = np.array_split(reg, nsplit, axis=0)
    devices = np.arange(0, len(data_)) % num_gpu

    psi_, scan_, probe_ = zip(*processes.starmap(
        recon_with_device,
        zip(devices, data_, psi_, scan_, probe_, reg_),
    ))
    # psi_, scan_, probe_ = zip(*map(
    #     recon_with_device,
    #     devices,
    #     data_,
    #     psi_,
    #     scan_,
    #     probe_,
    #     reg_,
    # ))

    return {
        'psi': np.concatenate(psi_, axis=0),
        'scan': np.concatenate(scan_, axis=0),
        'probe': np.concatenate(probe_, axis=0),
    }


def rotate_and_crop(psi, radius=128, angle=-72.035):
    # Rotate by desired angle (degrees)
    rotate_params = dict(
        angle=angle,
        clip=False,
        preserve_range=True,
        resize=False,
    )
    psi.real = skimage.transform.rotate(psi.real, **rotate_params)
    psi.imag = skimage.transform.rotate(psi.imag, **rotate_params)
    phase = np.angle(psi)
    phase[phase < 0] = 0

    # Find the center of mass
    M = skimage.measure.moments(phase, order=1)
    center = np.array([M[1, 0] / M[0, 0], M[0, 1] / M[0, 0]]).astype('int')

    # Adjust the cropping region so it stays within the image
    lo = np.fmax(0, center - radius)
    hi = lo + 2 * radius
    shift = np.fmin(0, psi.shape - hi)
    hi += shift
    lo += shift
    assert np.all(lo >= 0), lo
    assert np.all(hi <= psi.shape), (hi, psi.shape)
    # Crop image
    patch = psi[lo[0]:hi[0], lo[1]:hi[1]].astype('complex64')

    return psi, patch, lo


def uncrop_and_rotate(psi, patch, lo, radius=128, angle=72.035):

    psi[lo[0]:lo[0] + 2 * radius, lo[1]:lo[1] + 2 * radius] = patch

    # Rotate by desired angle (degrees)
    rotate_params = dict(
        angle=angle,
        clip=False,
        preserve_range=True,
        resize=False,
    )
    psi.real = skimage.transform.rotate(psi.real, **rotate_params)
    psi.imag = skimage.transform.rotate(psi.imag, **rotate_params)
    return psi


def ptycho_lamino_align(
    processes,
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

    # Set initial values for intermediate variables
    w = 256
    u = np.zeros(
        [w, w, w],
        dtype='complex64',
    ) if u is None else u
    flow = np.zeros(
        [len(theta), w, w, 2],
        dtype='float32',
    ) if flow is None else flow
    winsize = min(*u.shape[:2])

    presult = {  # ptychography result
        'psi': psi,
        'scan': scan,
        'probe': probe,
    }

    phi = np.exp(1j * tike.lamino.simulate(obj=u, tilt=tilt, theta=theta))
    λ_p = np.zeros_like(phi)
    λ_a = np.zeros_like(phi)
    reg_p = np.zeros_like(psi)

    for k in range(niter):
        logging.info(f"Start ADMM iteration {k}.")

        logging.info("Solve the ptychography problem.")

        if k > 0:
            # Skip regularization on zeroth iteration because we don't know
            # value of the cropping corner locations
            reg_p = np.stack(
                list(
                    processes.starmap(
                        uncrop_and_rotate,
                        zip(
                            psi_rotated,
                            tike.align.simulate(phi, flow) + λ_p / 0.5,
                            corners,
                        ),
                    )),
                axis=0,
            )
        presult = multi_ptycho(
            processes,
            data=data,
            reg=reg_p,
            **presult,
        )
        psi = presult['psi']

        logging.info("Rotate and crop projections.")
        psi_rotated, trimmed, corners = zip(
            *processes.map(rotate_and_crop, psi))
        psi_rotated = np.stack(psi_rotated, axis=0)
        trimmed = np.stack(trimmed, axis=0)
        corners = np.stack(corners, axis=0)

        logging.info("Recover aligned projections from unaligned.")
        aresult = tike.align.reconstruct(
            unaligned=trimmed - λ_p / 0.5,
            original=phi,
            flow=flow,
            num_iter=4,
            algorithm='cgrad',
            reg=np.exp(1j * tike.lamino.simulate(obj=u, tilt=tilt, theta=theta))
            + λ_a / 0.5,
            rho=0.5,
        )
        phi = aresult['original']

        logging.info('Solve the laminography problem.')
        lresult = tike.lamino.reconstruct(
            data=np.log(λ_a / 0.5 - phi) / (1j),
            theta=theta,
            tilt=tilt,
            obj=u,
            algorithm='cgrad',
            num_iter=1,
            cg_iter=4,
        )
        u = lresult['obj']

        logging.info("Estimate alignment using Farneback.")
        aresult = tike.align.solvers.farneback(
            op=None,
            unaligned=trimmed,
            original=tike.align.simulate(phi, flow) + λ_p / 0.5,
            flow=flow,
            pyr_scale=0.5,
            levels=1,
            winsize=winsize,
            num_iter=4,
        )
        flow = aresult['shift']

        logging.info('Update lambdas and rhos.')

        λ_p += 0.5 * (-trimmed + tike.align.simulate(phi, flow))
        λ_a += 0.5 * (-phi + np.exp(
            1j * tike.lamino.simulate(obj=u, tilt=tilt, theta=theta)))

        # Limit winsize to larger value. 20?
        if winsize > 20:
            winsize -= 1

        if (k + 1) % 1 == 0:
            dxchange.write_tiff(
                skimage.restoration.unwrap_phase(np.angle(
                    presult['psi'])).astype('float32'),
                f'{folder}/object-phase-{(k+1):03d}.tiff',
            )
            dxchange.write_tiff(
                phi.real,
                f'{folder}/phi-real-{(k+1):03d}.tiff',
                dtype='float32',
            )
            dxchange.write_tiff(
                phi.imag,
                f'{folder}/phi-imag-{(k+1):03d}.tiff',
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

    result = presult
    return result
