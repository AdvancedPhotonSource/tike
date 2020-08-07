import logging

import numpy as np

import tike.align
import tike.lamino

import dxchange

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

        if (k+1) % 10 == 0:
            dxchange.write_tiff(
                u.real,
                f'particle-{(k+1):03d}.tiff',
                dtype='float32',
            )

    return u
