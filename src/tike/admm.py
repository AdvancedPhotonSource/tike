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


def recon_with_device(device, data, psi, scan, probe):
    with cp.cuda.Device(device):
        result = tike.ptycho.reconstruct(
            data=data,
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


def multi_ptycho(data, probe, scan, psi, num_gpu=8):

    nsplit = len(data) // 6
    data_ = np.array_split(data, nsplit, axis=0)
    psi_ = np.array_split(psi, nsplit, axis=0)
    scan_ = np.array_split(scan, nsplit, axis=0)
    probe_ = np.array_split(probe, nsplit, axis=0)
    devices = np.arange(0, len(data_)) % num_gpu

    with multiprocessing.Pool(num_gpu) as processes:
        psi_, scan_, probe_ = zip(*processes.starmap(
            recon_with_device,
            zip(devices, data_, psi_, scan_, probe_),
        ))

    return {
        'psi': np.concatenate(psi_, axis=0),
        'scan': np.concatenate(scan_, axis=0),
        'probe': np.concatenate(probe_, axis=0),
    }


def rotate_and_crop(psi, radius, angle):
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


def uncrop_and_rotate(psi, patch, lo, radius, angle):

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


def ptycho_lamino_align(data, psi, scan, probe, theta, tilt, niter=8):
    """Solve the joint ptycho-lamino-alignment problem using ADMM."""

    presult = {  # ptychography result
        'psi': psi,
        'scan': scan,
        'probe': probe,
    }

    flow = 0  # FIXME
    phi = 0  # FIXME
    u = 0  # FIXME

    for k in range(niter):

        logging.info("Solve the ptychography problem.")
        presult = multi_ptycho(data=data, **presult)
        psi = presult['psi']

        logging.info("Rotate and crop projections.")
        trimmed = rotate_and_crop(psi, radius=256, angle=-72.035)

        logging.info("Convert projections to linear space.")
        unaligned = -np.log(trimmed)

        # logging.info("Estimate alignment using Farneback.")
        # aresult = tike.align.solvers.farneback(
        #     op=None,
        #     unaligned=unaligned,
        #     original=phi,
        #     flow=flow,
        #     pyr_scale=0.5,
        #     levels=1,
        #     winsize=winsize,
        #     num_iter=4,
        # )
        # flow = aresult['shift']

        # logging.info("Recover aligned projections from unaligned.")
        # aresult = tike.align.reconstruct(
        #     unaligned=unaligned,
        #     original=phi,
        #     flow=flow,
        #     num_iter=4,
        #     algorithm='cgrad',
        # )
        # phi = aresult['original']

        # logging.info('Solve the laminography problem.')
        # result = tike.lamino.reconstruct(
        #     data=phi,
        #     theta=theta,
        #     tilt=tilt,
        #     obj=u,
        #     algorithm='cgrad',
        #     num_iter=1,
        #     cg_iter=4,
        # )
        # u = result['obj']

        # logging.info('Update lambdas and rhos.')
        # # lambda rho for ptychography

        # # lamda rho for alignment
        # h = tike.lamino.simulate(obj=u, theta=theta, tilt=tilt)
        # lamd += rho * (h - phi)

        # # Limit winsize to larger value. 20?
        # if winsize > 20:
        #     winsize -= 1

        # if (k + 1) % 10 == 0:
        #     np.save(f"flow-tike-{(k+1):03d}", flow)
        #     dxchange.write_tiff(
        #         u.real,
        #         f'particle-{(k+1):03d}.tiff',
        #         dtype='float32',
        #     )

    return u
