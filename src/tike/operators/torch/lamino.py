import cupy as cp
import torch

import tike.operators.cupy


class LaminoFunction(torch.autograd.Function):
    """The forward/adjoint laminography operations.

    Parameters
    ----------
    u : (N, N, N, 2) tensor float32
        A (3 + 1)D tensor where the first dimensions are spatial dimensions and
        the last dimension of len 2 is the real/imaginary components. Pytorch
        doesn't presently have good complex-value support.
    theta : (M, ) tensor float32
        The rotation angles of the projections.
    tilt : float
        The laminography angle
    output : (M, N, N, 2) float32
        Projections through the volume at each rotation angle.

    """

    @staticmethod
    def forward(ctx, u, theta, tilt=cp.pi / 2):
        ctx.n = u.shape[0]
        ctx.tilt = tilt
        ctx.save_for_backward(theta)
        with tike.operators.cupy.Lamino(
                n=ctx.n,
                tilt=ctx.tilt,
                eps=1e-6,
                upsample=2,
        ) as operator:
            output = operator.fwd(
                u=cp.asarray(torch.view_as_complex(u).detach(),
                             dtype='complex64'),
                theta=cp.asarray(theta, dtype='float32'),
            )
        output = torch.view_as_real(torch.as_tensor(output, device=u.device))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        theta, = ctx.saved_tensors
        with tike.operators.cupy.Lamino(
                n=ctx.n,
                tilt=ctx.tilt,
                eps=1e-6,
                upsample=2,
        ) as operator:
            grad_u = operator.adj(
                data=cp.asarray(torch.view_as_complex(grad_output),
                                dtype='complex64'),
                theta=cp.asarray(theta, dtype='float32'),
            ) / grad_output.shape[0]
        grad_u = torch.view_as_real(
            torch.as_tensor(grad_u, device=grad_output.device))
        grad_theta = grad_tilt = None
        return grad_u, grad_theta, grad_tilt


class LaminoModule(torch.nn.Module):

    def __init__(self, width):
        super(LaminoModule, self).__init__()
        self.width = width
        self.weight = torch.nn.Parameter(
            torch.zeros(width, width, width, 2, dtype=torch.float32))

    def forward(self, theta, tilt=cp.pi / 2):
        return LaminoFunction.apply(self.weight, theta, tilt)

    def extra_repr(self):
        return f'width={self.width}'
