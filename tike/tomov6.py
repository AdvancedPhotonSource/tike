
from tike.constants import wavenumber


def invtomo3(data, theta, voxelsize, energy, niter, init):
    _data = 1 / wavenumber(energy) * np.log(data) / voxelsize

    # pb1 = tomopy.recon(-np.real(_data), theta, algorithm='sirt',
    #                    num_iter=niter, init_recon=init.beta.copy())
    # pd1 = tomopy.recon(np.imag(_data), theta, algorithm='sirt',
    #                    num_iter=niter, init_recon=init.delta.copy())

    _pb = [init.beta.copy()]
    _pd = [init.delta.copy()]
    obj0 = init
    convx = list()
    convLagx = list()
    for l in range(niter):
        pb = tomopy.recon(-np.real(_data), theta, algorithm='sirt', num_iter=1,
                          init_recon=_pb[l].copy())
        pd = tomopy.recon(np.imag(_data), theta, algorithm='sirt', num_iter=1,
                          init_recon=_pd[l].copy())
        obj1 = Object(pb, pd, 1e-7)

        _pb.append(pb)
        _pd.append(pd)

        convx.append(np.sqrt(np.sum(np.power(np.abs(obj1.complexform
                                                    - obj0.complexform), 2))))

        obj0 = obj1

#    print("_pb={}\npb1={}".format(_pb[10], pb1))
#    print("_pb[0]={}".format(_pb[0]))
#    np.testing.assert_equal(_pb[niter], pb1)
#    np.testing.assert_equal(_pd[niter], pd1)

    # this part to measure Lagrangian x - ???
    Robb = tomopy.project(_pb[niter-1].copy(), theta, pad=False)
    Robd = tomopy.project(_pd[niter-1].copy(), theta, pad=False)

    Rox = 1j * (Robd + 1j * Robb)

    convLagx.append(np.sqrt(np.sum(np.power(np.abs(Rox - _data), 2))))
    return obj1, convx, convLagx
