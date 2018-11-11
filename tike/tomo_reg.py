import numpy as np
import cupy as cp 
from lprec import lpTransform
import tomopy
import matplotlib.pyplot as plt

def tomo_reg(lp=None, tomo=None, init_recon=None, grad = None, 
             num_iter=1, reg_par=[1,1], gpu=0,
             **kwargs):
    """
    L2 regularization with the conjugate gradient method
    Solving the problem: reg_par[0]*||R(recon)-tomo||^2_2 + reg_par[1]*||nabla(recon)-grad||^2_2 -> min
    """

    # take regularization parameters
    rho = np.float32(reg_par[0])
    tau = np.float32(reg_par[1])

    # choose device
    cp.cuda.Device(gpu).use()

    # Allocating necessary gpu arrays
    recon = cp.array(init_recon)
    tomo = cp.array(tomo)
    grad = cp.array(grad)
    b = recon*0
    f = recon*0
    g = tomo*0

    # right side: rho R^*(tomo)+tau (div(grad)))
    lp.adjp(b, tomo, gpu)
    b = rho*b + tau*div(grad)

    # residual r = b - rho R^*(R(recon)) - tau div(nabla(recon))
    lp.fwdp(g, recon, gpu)
    lp.adjp(f, g, gpu)
    r = b - rho*f - tau*div(nabla(recon))
    p = r.copy()
    rsold = cp.sum(r*r)

    # cg iterations
    for i in range(0, num_iter):
        #rho R^*(R(p)) + tau div(nabla(p))
        lp.fwdp(g, p, gpu)
        lp.adjp(f, g, gpu)
        f = rho*f + tau*div(nabla(p))
        # cg update
        alpha = rsold/cp.sum(p*f)
        recon = recon+alpha*p
        r = r-alpha*f
        rsnew = cp.sum(r*r)
        p = r+(rsnew/rsold)*p
        rsold = rsnew

    return recon.get()

def nabla(f):
    """
    Gradient of function f with respect to x and y variables
    """

    gr = cp.zeros([2,f.shape[0],f.shape[1],f.shape[2]],dtype="float32")
    gr[0,:, :, :-1] = f[:, :, 1:]-f[:, :, :-1]
    gr[1,:, :-1, :] = f[:, 1:, :]-f[:, :-1, :]

    return gr*0.5 # to have nabla(div(gr))~gr

def div(gr):
    """
    Divergence, the adjoint operator to the gradient
    """

    f = cp.zeros([gr.shape[1],gr.shape[2],gr.shape[3]],dtype="float32")
    f[:, :, 1:] = (gr[0, :, :, 1:]-gr[0, :, :, :-1])
    f[:, :, 0] = gr[0, :, :, 0]
    f[:, 1:, :] += (gr[1, :, 1:, :]-gr[1, :, :-1, :])
    f[:, 0, :] += gr[1, :, 0, :]
    f = -f 

    return f*0.5 # to have nabla(div(gr))~gr

if __name__ == '__main__':    
    Ns = 1
    N = 256
    Ntheta = 256*3//2
    gpu = 0

    f = tomopy.misc.phantom.shepp2d(N)

    # gg = nabla(cp.array(f))
    # ff = div(gg)
    # ggg = nabla(ff)
    # scale = cp.sum(gg*ggg)/cp.sum(ggg*ggg)
    # dif = (cp.sum(cp.array(f)*ff)-cp.sum(gg*gg))/cp.sum(gg*gg)
    # print(scale)
    # print(dif)

    # init Lprec handle.
    lp = lpTransform.lpTransform(
        N, Ntheta, Ns, "None", f.shape[2]//2, "cubic")
    lp.precompute(1)
    lp.initcmem(1, gpu)

    # compute projections
    tomo = lp.fwd(f, gpu)
    plt.imshow(tomo[0,:,:],cmap='grey')
    plt.show()


    #reconstruct by CG without regularization (0*||nabla(recon)-grad||^2_2)
    recon0 = np.zeros(f.shape,dtype="float32")+1e-6
    grad = np.zeros([2,f.shape[0],f.shape[1],f.shape[2]],dtype="float32")

    recon0 = tomo_reg(lp, tomo, recon0, grad, reg_par=[1,0], num_iter=32, gpu=gpu)
    plt.subplot(1,2,1)
    plt.imshow(recon0[0,:,:],cmap="gray")

    #reconstruct by CG with regularization (0.1*||nabla(recon)-grad||^2_2)
    recon = np.zeros(f.shape,dtype="float32")+1e-6
    grad = np.zeros([2,f.shape[0],f.shape[1],f.shape[2]],dtype="float32")
    recon = tomo_reg(lp, tomo, recon, grad, reg_par=[1,0.1], num_iter=32, gpu=gpu)
    plt.subplot(1,2,2) 
    plt.imshow(recon[0,:,:],cmap="gray")