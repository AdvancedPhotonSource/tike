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

typedef void
scatterOrGather(float2*, int, const float2*, int, float);

// grid shape (-(-kernel_size // max_threads), 0, nf)
// block shape (min(kernel_size, max_threads), 0, 0)
__device__ void
_loop_over_kernels(scatterOrGather operation, float2* gathered,
                   const float2* scattered, int nf, const float* x, int n,
                   int radius, const float* cons, int ndim) {
  const int diameter = 2 * radius;  // kernel width
  const int nk = pow(diameter, ndim);
  const int half = n / 2;  // shifts frequency coordinates to center
  const int max_dim = 3;
  assert(0 < ndim && ndim <= max_dim);

  // non-uniform frequency index (fi)
  for (int fi = blockIdx.z; fi < nf; fi += gridDim.z) {
    int center[max_dim];  // closest ND coord to kernel center
    for (int dim = 0; dim < ndim; dim++) {
      center[dim] = int(floor(n * x[ndim * fi + dim]));
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
      int k[max_dim];  // ND kernel coord
      _1d_to_nd(ki, nk, k, ndim, diameter, radius);

      // Compute sum square value for kernel and equally-spaced grid index (gi)
      float ssdelta = 0;
      float delta;
      int gi = 0;
      int stride = 1;
      for (int dim = ndim - 1; dim >= 0; dim--) {
        delta = (float)(center[dim] + k[dim]) / n - x[ndim * fi + dim];
        ssdelta += delta * delta;
        gi += mod((half + center[dim] + k[dim]), n) * stride;
        stride *= n;
      }
      const float kernel = cons[0] * exp(cons[1] * ssdelta);

      operation(gathered, fi, scattered, gi, kernel);
    }
  }
}

// Helper functions _gather and _scatter let us switch the index variables
// (si, gi) without an if statement.
__device__ void
_gather(float2* gather, int gi, const float2* scatter, int si, float kernel) {
  atomicAdd(&gather[gi].x, scatter[si].x * kernel);
  atomicAdd(&gather[gi].y, scatter[si].y * kernel);
}

extern "C" __global__ void
gather(float2* F, const float2* Fe, int nf, const float* x, int n, int radius,
       const float* cons) {
  _loop_over_kernels(_gather, F, Fe, nf, x, n, radius, cons, 3);
}

__device__ void
_scatter(float2* gather, int si, const float2* scatter, int gi, float kernel) {
  atomicAdd(&gather[gi].x, scatter[si].x * kernel);
  atomicAdd(&gather[gi].y, scatter[si].y * kernel);
}

extern "C" __global__ void
scatter(float2* G, const float2* f, int nf, const float* x, int n, int radius,
        const float* cons) {
  _loop_over_kernels(_scatter, G, f, nf, x, n, radius, cons, 3);
}
