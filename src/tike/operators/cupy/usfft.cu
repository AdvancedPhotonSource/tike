// Cannot use complex types because of atomicAdd()
// #include <cupy/complex.cuh>

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

// convert 1d coordinates (d) to nd coordinates (nd) for grid with diameter
// s is the max 1d coordinate.
__device__ void
_1d_to_nd(int d, int s, int* nd, int ndim, int diameter) {
  for (int dim = 0; dim < ndim; dim++) {
    s /= diameter;
    nd[dim] = d / s;
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
  const int gw = 2 * (n + radius);  // width of G along each dimension

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
      _1d_to_nd(ki, nk, k, ndim, diameter);

      // Compute sum square value for kernel
      float ssdelta = 0;
      float delta;
      for (int dim = 0; dim < ndim; dim++) {
        delta = (float)(center[dim] - radius + k[dim]) / (2 * n)
                - x[3 * fi + dim];
        ssdelta += delta * delta;
      }
      float kernel = cons[0] * exp(cons[1] * ssdelta);

      // clang-format off
      int gi = (  // equally-spaced grid index (gi)
        + (n + center[0] + k[0]) * gw * gw
        + (n + center[1] + k[1]) * gw
        + (n + center[2] + k[2])
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
