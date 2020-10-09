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
    angle=0,  #-72.035 / 180 * np.pi
    w=256 + 64,
):
    """Solve the joint ptycho-lamino problem using ADMM."""
    u = np.zeros((w, w, w), dtype='complex64')
    Hu = np.ones((len(theta), w, w), dtype='complex64')
    presult = {  # ptychography result
            'psi': np.ones(psi.shape, dtype='complex64'),
            'scan': scan,
            'probe': probe,
    }
    λ_p = np.zeros_like(psi)
    ρ_p = 0.5
    comm = MPICommunicator()
    with cp.cuda.Device(comm.rank):
        for k in range(niter):
            logging.info(f"Start ADMM iteration {k}.")

            logging.info("Solve the ptychography problem.")

            logging.info("Solve the ptychography problem.")
            presult = tike.ptycho.reconstruct(
                data=data,
                reg=λ_p / ρ_p - Hu,
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

            # Gather all to one thread
            psi, theta, λ_p = [comm.gather(x) for x in (psi, theta, λ_p)]

            if comm.rank == 0:
                logging.info('Solve the laminography problem.')
                lresult = tike.lamino.reconstruct(
                    data=-1j * np.log(psi + λ_p / ρ_p),
                    theta=theta,
                    tilt=tilt,
                    obj=u,
                    algorithm='cgrad',
                    num_iter=1,
                    cg_iter=4,
                )
                u = lresult['obj']

            # Separate again to multiple threads
            psi, theta, λ_p = [comm.scatter(x) for x in (psi, theta, λ_p)]
            u = comm.broadcast(u)

            logging.info('Update lambdas and rhos.')

            Hu = np.exp(1j * tike.lamino.simulate(
                obj=u,
                tilt=tilt,
                theta=theta,
            ))
            ψHu = psi - Hu
            λ_p += ρ_p * ψHu

            if k > 0:
                ρ_p = update_penalty(comm, psi, Hu, Hu0, ρ_p)
            Hu0 = Hu

            lagrangian = (
                [presult['cost']],
                [
                    2 * np.real(λ_p.conj() * ψHu) +
                    ρ_p * np.linalg.norm(ψHu.ravel())**2
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
                    Hu.real,
                    f'{folder}/TPHu-real-{(k+1):03d}.tiff',
                    dtype='float32',
                )
                dxchange.write_tiff(
                    Hu.imag,
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
