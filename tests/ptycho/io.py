import warnings
import os

import numpy as np
import tike.view

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


def _save_probe(output_folder, probe, algorithm):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    flattened = np.concatenate(
        probe.reshape((-1, *probe.shape[-2:])),
        axis=1,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        plt.imsave(
            f'{output_folder}/probe-phase.png',
            np.angle(flattened),
            # The output of np.angle is locked to (-pi, pi]
            cmap=plt.cm.twilight,
            vmin=-np.pi,
            vmax=np.pi,
        )
        plt.imsave(
            f'{output_folder}/probe-ampli.png',
            np.abs(flattened),
        )
    f = plt.figure()
    tike.view.plot_probe_power(probe)
    plt.semilogy()
    plt.title(algorithm)
    plt.savefig(f'{output_folder}/probe-power.svg')
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

        fig = plt.figure()
        ax1, ax2 = tike.view.plot_cost_convergence(
            result.algorithm_options.costs,
            result.algorithm_options.times,
        )
        ax2.set_xlim(0, 20)
        ax1.set_ylim(10**(-1), 10**2)
        fig.suptitle(algorithm)
        fig.tight_layout()
        plt.savefig(os.path.join(fname, 'convergence.svg'))
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
        )
        import tifffile
        tifffile.imwrite(
            f'{fname}/{0}-ampli.tiff',
            np.abs(result.psi).astype('float32'),
        )
        _save_probe(fname, result.probe, algorithm)
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
    plt.savefig(f'{fname}/weights.svg')
