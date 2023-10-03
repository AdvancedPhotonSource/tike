import os
import typing
import warnings

import numpy as np
import numpy.typing as npt
import tike.view
import tike.ptycho

test_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

result_dir = os.path.join(test_dir, 'result', 'ptycho')
os.makedirs(result_dir, exist_ok=True)

data_dir = os.path.join(test_dir, 'data')


def _save_eigen_probe(output_folder, eigen_probe):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    flattened = []
    for i in range(eigen_probe.shape[-4]):
        probe = eigen_probe[..., i, :, :, :]
        flattened.append(
            np.concatenate(
                probe.reshape((-1, *probe.shape[-2:])),
                axis=1,
            ))
    flattened = np.concatenate(flattened, axis=0)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        plt.imsave(
            f'{output_folder}/eigen-phase.png',
            np.angle(flattened),
            # The output of np.angle is locked to (-pi, pi]
            cmap=plt.cm.twilight,
            vmin=-np.pi,
            vmax=np.pi,
        )
        plt.imsave(
            f'{output_folder}/eigen-ampli.png',
            np.abs(flattened),
        )


def _save_probe(
    output_folder: str,
    probe: npt.NDArray,
    probe_options: typing.Union[None, tike.ptycho.ProbeOptions],
    algorithm: str,
):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    flattened = np.concatenate(
        probe.reshape((-1, *probe.shape[-2:])),
        axis=1,
    )
    flattened /= (np.abs(flattened).max() * 1.001)
    plt.imsave(
        f'{output_folder}/probe.png',
        tike.view.complexHSV_to_RGB(flattened),
    )
    if probe_options is not None and len(probe_options.power) > 0:
        f = plt.figure()
        tike.view.plot_probe_power_series(probe_options.power)
        plt.title(algorithm)
        plt.savefig(f'{output_folder}/probe-power.png')
        plt.close(f)
    nmodes = probe.shape[-3]
    probe_orthogonality_matrix = np.zeros((nmodes, nmodes))
    for i in range(nmodes):
        for j in range(nmodes):
            probe_orthogonality_matrix[i, j] = np.abs(tike.linalg.inner(
                probe[..., i, :, :],
                probe[..., j, :, :]
            ))
    f = plt.figure()
    plt.imshow(probe_orthogonality_matrix, interpolation='nearest')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f'{output_folder}/probe-orthogonality.png')
    plt.close(f)


def _save_ptycho_result(result, algorithm):
    if result is None:
        return
    try:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        import tike.view
        fname = os.path.join(result_dir, f'{algorithm}')
        os.makedirs(fname, exist_ok=True)
        if len(result.algorithm_options.costs) > 1:
            fig = plt.figure()
            ax1, ax2 = tike.view.plot_cost_convergence(
                result.algorithm_options.costs,
                result.algorithm_options.times,
            )
            ax2.set_xlim(0, 60)
            ax1.set_ylim(10**(-1), 10**2)
            fig.suptitle(algorithm)
            fig.tight_layout()
            plt.savefig(os.path.join(fname, 'convergence.png'))
            plt.close(fig)
        plt.imsave(
            f'{fname}/{0}-phase.png',
            np.angle(result.psi).astype('float32'),
            # The output of np.angle is locked to (-pi, pi]
            cmap=plt.cm.twilight,
            vmin=-np.pi,
            vmax=np.pi,
        )
        plt.imsave(
            f'{fname}/{0}-ampli.png',
            np.abs(result.psi).astype('float32'),
            cmap=plt.cm.gray,
        )
        _save_probe(fname, result.probe, result.probe_options, algorithm)
        if result.eigen_weights is not None:
            _save_eigen_weights(fname, result.eigen_weights)
            if result.eigen_weights.shape[-2] > 1:
                _save_eigen_probe(fname, result.eigen_probe)
    except ImportError:
        pass


def _save_eigen_weights(fname, weights):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    plt.figure()
    tike.view.plot_eigen_weights(weights)
    plt.suptitle('weights')
    plt.tight_layout()
    plt.savefig(f'{fname}/weights.png')
