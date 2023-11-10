import lzma
import os
import pickle
import unittest

import cupy as cp
import numpy as np
import torch
from torch.nn.modules.loss import GaussianNLLLoss

from tike.operators.torch import LaminoFunction, LaminoModule


@unittest.skip('single precision is not enough to pass gradcheck')
def test_lamino_gradcheck(n=16, ntheta=8):

    lamino = LaminoFunction.apply

    # gradcheck takes a tuple of tensors as input, check if your gradient
    # evaluated with these tensors are close enough to numerical
    # approximations and returns True if they all verify this condition.
    input = (
        torch.randn(
            n,
            n,
            n,
            2,
            dtype=torch.float32,
            requires_grad=True,
            device='cpu',
        ),
        cp.pi * torch.randn(
            ntheta,
            dtype=torch.float32,
            requires_grad=False,
            device='cpu',
        ),
    )
    test = torch.autograd.gradcheck(
        lamino,
        input,
        eps=1e-6,
        atol=1e-4,
        nondet_tol=1e-6,
    )
    print(test)


testdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


class L2Loss(torch.nn.Module):

    def forward(self, input, target):
        return torch.mean(torch.square(torch.abs(input - target)))


class TestLaminoModel(unittest.TestCase):

    def setUp(self):
        """Load a dataset for reconstruction."""
        dataset_file = os.path.join(testdir, 'data/lamino_setup.pickle.lzma')
        if not os.path.isfile(dataset_file):
            self.create_dataset(dataset_file)
        with lzma.open(dataset_file, 'rb') as file:
            [
                self.data,
                self.original,
                self.theta,
                self.tilt,
            ] = pickle.load(file)

    def test_lamino_model(self, num_epoch=32, device=0):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        theta = torch.from_numpy(self.theta).type(torch.float32).to(device)
        data = torch.view_as_real(
            torch.from_numpy(self.data).type(torch.complex64)).to(device)
        var = torch.ones(data.shape, dtype=torch.float32,
                         requires_grad=True).to(device)

        model = LaminoModule(data.shape[1]).to(device)
        lossf = GaussianNLLLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters())

        loss_log = []
        for epoch in range(num_epoch):
            pred = model(theta, self.tilt)
            loss = lossf(pred, data, var)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_log.append(loss.item())
            print(f"loss: {loss_log[-1]:.3e}  [{epoch:>5d}/{num_epoch:>5d}]")

        obj = torch.view_as_complex(model.weight.cpu().detach()).numpy()

        _save_lamino_result({'obj': obj, 'costs': loss_log}, 'torch')


def _save_lamino_result(result, algorithm):
    try:
        import matplotlib.pyplot as plt
        fname = os.path.join(testdir, 'result', 'lamino', f'{algorithm}')
        os.makedirs(fname, exist_ok=True)
        plt.figure()
        plt.title(algorithm)
        plt.plot(result['costs'])
        plt.semilogy()
        plt.savefig(os.path.join(fname, 'convergence.svg'))
        slice_id = int(35 / 128 * result['obj'].shape[0])
        plt.imsave(
            f'{fname}/{slice_id}-phase.png',
            np.angle(result['obj'][slice_id]).astype('float32'),
            # The output of np.angle is locked to (-pi, pi]
            cmap=plt.cm.twilight,
            vmin=-np.pi,
            vmax=np.pi,
        )
        plt.imsave(
            f'{fname}/{slice_id}-ampli.png',
            np.abs(result['obj'][slice_id]).astype('float32'),
        )
        import skimage.io
        skimage.io.imsave(
            f'{fname}/phase.tiff',
            np.angle(result['obj']).astype('float32'),
        )
        skimage.io.imsave(
            f'{fname}/ampli.tiff',
            np.abs(result['obj']).astype('float32'),
        )

    except ImportError:
        pass
