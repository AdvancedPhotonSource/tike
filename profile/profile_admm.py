#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Benchmark ptychotomography reconstruction.

Profile tike.simulate and tike.admm on the function level by running the main
function of this script. Line by line profile hotspots for the file
tike/foo.py can be obtained by using pprofile. As below:

```
$ pprofile --statistic 0.001 --include tike/foo.py profile_admm.py
```
"""
import click
import logging
import os
import pickle
from pyinstrument import Profiler
# These environmental variables must be set before numpy is imported anywhere.
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np  # noqa
import tike  # noqa


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.argument('data-file', type=click.Path(exists=True))
@click.argument('params-file', type=click.Path(exists=True))
@click.option('--recon-file', type=click.Path(file_okay=True, dir_okay=True,
                                              writable=True),
              default=None,
              help='Save reconstruction to this file.')
@click.option('--profile', is_flag=True,
              help='Profile at function level using pyinstrument.',)
@click.option('-A', '--admm-iters', default=1, type=click.INT,
              help='The number of ADMM interations.',)
@click.option('-P', '--ptycho-iters', default=1, type=click.INT,
              help='The number of pytchography iterations.')
@click.option('-T', '--tomo-iters', default=1, type=click.INT,
              help='The number of tomography iterations.')
def admm_profile_workload(
        data_file,
        params_file,
        recon_file,
        profile,
        admm_iters,
        ptycho_iters,
        tomo_iters,
):
    """Run some admm work which may be profiled."""
    comm = tike.MPICommunicator()
    # Load data
    data = None
    if comm.rank == 0:
        with open(data_file, 'rb') as file:
            data = pickle.load(file)
    data = comm.scatter(data)
    # Load acquisition parameters
    (
        obj, voxelsize,
        probe, energy,
        theta, v, h,
        detector_shape,
    ) = comm.load(params_file)
    recon = np.zeros(obj.shape, dtype=np.complex64)

    if comm.rank == 0:
        logger.info("""
        recon shape is {}
        voxelsize is {}
        data shape is {}
        theta shape is {}
        v shape is {}
        energy is {}
        """.format(recon.shape, voxelsize, np.asarray(data).shape,
                   theta.shape, v.shape, energy))

    pkwargs = {
        'algorithm': 'grad',
        'num_iter': ptycho_iters,
    }

    tkwargs = {
        'algorithm': 'grad',
        'num_iter': tomo_iters,
        'ncore': 1,
        'reg_par': -1,
    }

    if profile and comm.rank == 0:
        profiler = Profiler()
        profiler.start()

    recon = tike.admm(obj=recon, voxelsize=voxelsize,
                      data=data,
                      probe=probe, theta=theta, v=v, h=h, energy=energy,
                      num_iter=admm_iters,
                      rho=0.5, gamma=0.25,
                      comm=comm, pkwargs=pkwargs, tkwargs=tkwargs,
                      )

    if profile and comm.rank == 0:
        profiler.stop()
        print(profiler.output_text(unicode=True, color=False))

    # Save result to disk
    logger.info("Rank {} complete.".format(comm.rank))
    if recon_file is not None:
        recon = comm.gather(recon, root=0, axis=0)
        if comm.rank == 0:
            logger.info("Saving the result.")
            os.makedirs(os.path.dirname(recon_file), exist_ok=True)
            with open(recon_file, 'wb') as file:
                pickle.dump(recon, file)
    else:
        if comm.rank == 0:
            logger.info("Not saving the result.")


if __name__ == '__main__':
    admm_profile_workload()
