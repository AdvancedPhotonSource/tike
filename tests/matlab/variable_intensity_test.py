import os.path
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import h5py

_dir = os.path.dirname(__file__)


def test_variable_intensity():
    """Test that the variable intensity coefficient update is consistent."""

    from tike.ptycho.solvers.lstsq import _get_coefs_intensity

    with (h5py.File(os.path.join(_dir, 'variable_intensity_input0.mat'),
                    'r')) as input0, (h5py.File(
                        os.path.join(_dir, 'variable_intensity_input1.mat'),
                        'r')) as input1, (h5py.File(
                            os.path.join(_dir, 'variable_intensity_output.mat'),
                            'r')) as output:

        P = input0['probe'][...][None, None, None].view('complex64')
        O = input0['obj_proj_tmp'][...][:, None, None].view('complex64')
        xi = input1['chi_tmp'][...][:, None, None].view('complex64')
        weights = input1['probe_evolution'][...].transpose()[..., None]
        m = 0

        ref_weights = output['probe_evolution'][...].transpose()[..., None]

    assert weights.shape == (120, 1, 1)
    assert ref_weights.shape == (120, 1, 1)
    assert xi.shape == (120, 1, 1, 236, 236)
    assert P.shape == (1, 1, 1, 236, 236)
    assert O.shape == (120, 1, 1, 236, 236)

    new_weights = cp.asnumpy(
        _get_coefs_intensity(
            cp.asarray(weights),
            cp.asarray(xi),
            cp.asarray(P),
            cp.asarray(O),
            m,
        ))

    plt.figure()
    width = 0.5
    x = np.arange(120)
    plt.bar(x - width, ref_weights[:, 0, 0], width)
    plt.bar(x, new_weights[:, 0, 0], width)
    plt.legend(['ptychoshelves', 'tike'])
    plt.savefig(os.path.join(_dir, 'variable_intensity.svg'))
    plt.close('all')

    np.testing.assert_allclose(
        ref_weights,
        new_weights,
    )
