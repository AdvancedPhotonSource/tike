// Cannot use complex types because of atomicAdd()
// #include <cupy/complex.cuh>

// n % d, but the sign always matches the divisor (d)
__device__ int
mod(int n, int d) {
  return ((n % d) + d) % d;
}

// power function for integers and exponents >= 0
__device__ int
pow(int b, int e) {
  assert(e >= 0);
  int result = 1;
  for (int i = 0; i < e; i++) {
    result *= b;
  }
  return result;
}

// Convert a 1d coordinate (d) where s is the max 1d coordinate to nd
// coordinates (nd) for a grid with diameter along all dimensions and centered
// on the origin.
__device__ void
_1d_to_nd(int d, int s, int* nd, int ndim, int diameter, int radius) {
  for (int dim = 0; dim < ndim; dim++) {
    s /= diameter;
    nd[dim] = d / s - radius;
    assert(nd[dim] < diameter);
    d = d % s;
  }
}

// Helper function that lets us switch the index variables (si, gi) easily.
__device__ void
_gather_scatter(float2* gather, int gi, const float2* scatter, int si,
                float kernel) {
  atomicAdd(&gather[gi].x, scatter[si].x * kernel);
  atomicAdd(&gather[gi].y, scatter[si].y * kernel);
}

// grid shape (-(-kernel_size // max_threads), 0, nf)
// block shape (min(kernel_size, max_threads), 0, 0)
__device__ void
_loop_over_kernels(bool eq2us, float2* gathered, const float2* scattered,
                   int nf, const float* x, int n, int radius,
                   const float* cons) {
  const int ndim = 3;
  const int diameter = 2 * radius;  // kernel width
  const int nk = pow(diameter, ndim);
  const int gw = 2 * n;  // width of G along each dimension

  // non-uniform frequency index (fi)
  for (int fi = blockIdx.z; fi < nf; fi += gridDim.z) {
    int center[ndim];  // closest ND coord to kernel center
    for (int dim = 0; dim < ndim; dim++) {
      center[dim] = int(floor(2 * n * x[3 * fi + dim]));
    }
    // intra-kernel index (ki)
    // clang-format off
    for (
      int ki = threadIdx.x + blockDim.x * blockIdx.x;
      ki < nk;
      ki += blockDim.x * gridDim.x
    ) {
      // clang-format on
      // Convert linear index to 3D intra-kernel index
      int k[ndim];  // ND kernel coord
      _1d_to_nd(ki, nk, k, ndim, diameter, radius);

      // Compute sum square value for kernel
      float ssdelta = 0;
      float delta;
      for (int dim = 0; dim < ndim; dim++) {
        delta = (float)(center[dim] + k[dim]) / (2 * n)
                - x[3 * fi + dim];
        ssdelta += delta * delta;
      }
      float kernel = cons[0] * exp(cons[1] * ssdelta);

      // clang-format off
      int gi = (  // equally-spaced grid index (gi)
        + mod((n + center[0] + k[0]), gw) * gw * gw
        + mod((n + center[1] + k[1]), gw) * gw
        + mod((n + center[2] + k[2]), gw)
      );
      // clang-format on
      if (eq2us) {
        _gather_scatter(gathered, fi, scattered, gi, kernel);

      } else {
        _gather_scatter(gathered, gi, scattered, fi, kernel);
      }
    }
  }
}

extern "C" __global__ void
gather(float2* F, const float2* Fe, int nf, const float* x, int n, int radius,
       const float* cons) {
  _loop_over_kernels(true, F, Fe, nf, x, n, radius, cons);
}

extern "C" __global__ void
scatter(float2* G, const float2* f, int nf, const float* x, int n, int radius,
        const float* cons) {
  _loop_over_kernels(false, G, f, nf, x, n, radius, cons);
}
