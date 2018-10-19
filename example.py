# !/usr/bin/env python
# -*- coding: utf-8 -*-

import tomopy
import dxchange
import scipy

# Load a 3D object.
beta = dxchange.read_tiff('test-beta-128.tiff')[0:30, :, :]
delta = dxchange.read_tiff('test-delta-128.tiff')[0:30, :, :]

# Downsample.
beta = scipy.ndimage.zoom(beta, zoom=1/2)
delta = scipy.ndimage.zoom(delta, zoom=1/2)
refbeta = beta
refdelta = delta
# Create object.
obj = Object(beta, delta, 1e-7)
dxchange.write_tiff(obj.beta, 'tmp/beta')
dxchange.write_tiff(obj.delta, 'tmp/delta')

# Create probe.
weights = gaussian(15, rin=0.8, rout=1.0)
# weights = uniform(15)
# weights = np.random.rand(15, 15).astype('float32')
prb = Probe(weights, maxint=1)
dxchange.write_tiff(prb.amplitude, 'tmp/probe-amplitude')
dxchange.write_tiff(prb.phase, 'tmp/probe-phase')

# Detector parameters.
det = Detector(63, 63)

# Define scan positions.
theta, h, v = np.meshgrid(np.linspace(0, 2*np.pi, 360, endpoint=False),
                          np.linspace(-0.5, 0.5, beta.shape[0],
                                      endpoint=False),
                          np.linspace(-0.5, 0.5, beta.shape[1],
                                      endpoint=False))

# Project.
tike.tomo.forward()
psis = project(obj, theta, energy=5)
dxchange.write_tiff(np.real(psis), 'tmp/psi-amplitude')
dxchange.write_tiff(np.imag(psis), 'tmp/psi-phase')

# Propagate.
data = propagate(prb, psis, scan, theta, det, noise=False)
# TODO: is this allowed? Is data a number of photons?
# dat = np.random.poisson(dat).astype('float32')
dxchange.write_tiff(np.fft.fftshift(np.log(np.array(data[0]))), 'tmp/data')

# Init.
preq = np.divide(1, (np.multiply(np.conj(prb.complex), prb.complex))+1)
# preq = np.ones(prb.complex.shape)
dxchange.write_tiff(np.abs(preq), 'tmp/preq-amplitude')
dxchange.write_tiff(np.angle(preq), 'tmp/preq-phase')
#hobj0 = np.ones(psis.shape, dtype='complex')
hobj = np.ones(psis.shape, dtype='complex')
psi = np.ones(psis.shape, dtype='complex')
# hobj += 1j * 0.03177
tmp = np.zeros(obj.shape)
#recobj0 = Object(tmp, tmp, 1e-7)
recobj = Object(tmp, tmp, 1e-7)
lamd = np.zeros(psi.shape, dtype='complex')
rho = 1
gamma = 0.5
alpha = 1
res = np.zeros(psis.shape, dtype='complex')
dualres = np.zeros(psis.shape, dtype='complex')
mu = 1.5
tau_dec = 1.5
tau_inc = 1.5

Lagpsi_admm = list()
Lagx_admm = list()
Laglambda_admm = list()
convallpsi_admm = list()
convallx_admm = list()
clall = list()


for m in range(20):

    # Ptychography.
    psi, convallpsi, _convallLagpsi, allfail = invptycho3(data, prb, scan, psi, theta, niter=10000, rho=rho, gamma=gamma, hobj=hobj, lamd=lamd, preq=preq)
    dxchange.write_tiff(np.abs(psi[0]).astype('float32'), 'tmp/psi-amplitude/psi-amplitude')
    dxchange.write_tiff(np.angle(psi[0]).astype('float32'), 'tmp/psi-phase/psi-phase')
    cp = np.sqrt(np.sum(np.power(np.abs(hobj-psi), 2)))


    #Take the mean (or std, min, max) value of _convallpsi for all thetas; so you just have one output for each SD-iteration
#    for n in range(len(convallpsi)):
#        convallpsi_admm.append(np.mean(convallpsi[n]))
#        Lagpsi_admm.append(np.mean(_convallLagpsi[n]))

    # Tomography.
    _recobj, _convx, _convLagx = invtomo3(psi + lamd/rho, theta, obj.voxelsize, energy=5, niter=2000, init=recobj)
    co = np.sqrt(np.sum(np.power(np.abs(recobj.complexform- _recobj.complexform), 2)))
    convallx_admm.extend(_convx)
    Lagx_admm.append(_convLagx)
    dualres= rho * co
    recobj = _recobj
    dxchange.write_tiff(recobj.beta[beta.shape[0] // 2], 'tmp/beta/beta')
    dxchange.write_tiff(recobj.delta[delta.shape[0] // 2], 'tmp/delta/delta')

    # Lambda update.
    hobj = project(recobj, theta, energy=5)
    dxchange.write_tiff(np.abs(hobj[0]).astype('float32'), 'tmp/hobj-amplitude/hobj-amplitude')
    dxchange.write_tiff(np.angle(hobj[0]).astype('float32'), 'tmp/hobj-phase/hobj-phase')
    _res = psi - hobj
    _lamd = lamd + alpha * rho * _res
    res = np.sqrt(np.sum(np.power(np.abs(_res), 2)))
    cl = np.sqrt(np.sum(np.power(np.abs(lamd-_lamd), 2)))
    clall.append(cl)
    lamd = _lamd.copy()
    dxchange.write_tiff(np.abs((hobj - psi)[0]).astype('float32'), 'tmp/lamd-amplitude/lamd-amplitude')
    dxchange.write_tiff(np.angle((hobj - psi)[0]).astype('float32'), 'tmp/lamd-phase/lamd-phase')

    Laglambda_admm.append(np.sqrt(np.sum(np.power(np.abs(psi - hobj), 2))))
     # varying penalty parameter
#    if res > mu * dualres and rho < 3:
#        rho = tau_inc * rho
#    elif dualres > mu * res:
#        rho = rho / tau_dec

    print(m, cp, co, cl, res, dualres)
