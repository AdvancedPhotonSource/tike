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
_1d_to_nd(int* nd, int ndim, int d, int s, int diameter, const int* origin) {
  assert(0 <= d && d < s);
  int radius = diameter / 2;
  for (int dim = 0; dim < ndim; dim++) {
    s /= diameter;
    nd[dim] = d / s - radius + origin[dim];
    d = d % s;
  }
}

__device__ void
nearest(int ndim, int* x, const int* limit) {
  for (int dim = 0; dim < ndim; dim++) {
    x[dim] = min(max(0, x[dim]), limit[dim] - 1);
  }
}

__device__ void
wrap(int ndim, int* x, const int* limit) {
  for (int dim = 0; dim < ndim; dim++) {
    x[dim] = mod(x[dim], limit[dim]);
  }
}

// Convert an Nd coordinate (nd) from a grid with given shape a 1d linear
// coordinate.
__device__ int
_nd_to_1d(int ndim, const int* nd, const int* shape) {
  int linear = 0;
  int stride = 1;
  for (int dim = ndim - 1; dim >= 0; dim--) {
    assert(shape[dim] > 0);
    assert(0 <= nd[dim] && nd[dim] < shape[dim]);
    linear += nd[dim] * stride;
    stride *= shape[dim];
  }
  assert(linear >= 0);
  return linear;
}

typedef float
kernel_function(int ndim, const float* center, const int* point);

__device__ float
_lanczos(float x, float nlobes) {
  if (x == 0.0f) {
    return 1.0f;
  } else if (fabsf(x) <= nlobes) {
    // printf("distance: %f\n", x);
    const float pix = x * 3.141592653589793238462643383279502884f;
    return nlobes * sin(pix) * sin(pix / nlobes) / (pix * pix);
  } else {
    return 0.0f;
  }
}

// Return the lanczos kernel weight for the given kernel center and point.
__device__ float
lanczos_kernel(int ndim, const float* center, const int* point) {
  float weight = 1.0f;
  for (int dim = 0; dim < ndim; dim++) {
    weight *= _lanczos(center[dim] - (float)point[dim], 2.0f);
  }
  return weight;
}

typedef void
scatterOrGather(float2*, int, float2*, int, float weight);

// Many uniform grid points are collected to one nonuniform point. This is
// linear interpolation, smoothing, etc.
__device__ void
gather(float2* grid, int gi, float2* points, int pi, float weight) {
  atomicAdd(&points[pi].x, grid[gi].x * weight);
  atomicAdd(&points[pi].y, grid[gi].y * weight);
}

// One nonuniform point is spread to many uniform grid points. This is the
// adjoint operation.
__device__ void
scatter(float2* grid, int gi, float2* points, int pi, float weight) {
  atomicAdd(&grid[gi].x, points[pi].x * weight);
  atomicAdd(&grid[gi].y, points[pi].y * weight);
}

// grid shape (-(-diameter^ndim // max_threads), 0, nf)
// block shape (min(diameter^ndim, max_threads), 0, 0)
__device__ void
_loop_over_kernels(int ndim,  // number of dimensions
                   kernel_function get_weight, scatterOrGather operation,
                   float2* grid,       // values on uniform grid
                   const int* gshape,  // dimensions of uniform grid
                   float2* points,     // values at nonuniform points
                   const float* x,     // coordinates of nonuniform points
                   const int nx,       // the number of nonuniform points
                   const int diameter  // kernel diameter, should be odd?
) {
  assert(grid != NULL);
  assert(gshape != NULL);
  assert(points != NULL);
  assert(x != NULL);
  assert(nx >= 0);
  assert(diameter > 0);
  const int max_dim = 3;
  assert(0 < ndim && ndim <= max_dim);

  const int nk = pow(diameter, ndim);  // number of grid positions in kernel

  // nonuniform position index (xi)
  for (int xi = blockIdx.z; xi < nx; xi += gridDim.z) {
    // closest ND grid coord to point center of kernel
    int center[max_dim];
    for (int dim = 0; dim < ndim; dim++) {
      center[dim] = int(floor(x[ndim * xi + dim]));
    }
    // linear intra-kernel index (ki)
    // clang-format off
    for (
      int ki = threadIdx.x + blockDim.x * blockIdx.x;
      ki < nk;
      ki += blockDim.x * gridDim.x
    ) {
      // clang-format on
      // Convert linear intra-kernel index to ND grid coord (knd)
      int knd[max_dim];
      _1d_to_nd(knd, ndim, ki, nk, diameter, center);

      // Weights are computed from correct distance...
      const float weight = get_weight(ndim, &x[ndim * xi], knd);

      // ... but for values outside the grid we wrap around so that all of the
      // values are valid.
      wrap(ndim, knd, gshape);

      // Convert ND grid coord to linear grid coord
      const int gi = _nd_to_1d(ndim, knd, gshape);

      operation(grid, gi, points, xi, weight);
    }
  }
}

extern "C" __global__ void
fwd_lanczos_interp2D(float2* grid, const int* grid_shape, float2* points,
                     const float* x, int num_points, int diameter

) {
  _loop_over_kernels(2, lanczos_kernel, gather, grid, grid_shape, points, x,
                     num_points, diameter);
}

extern "C" __global__ void
adj_lanczos_interp2D(float2* grid, const int* grid_shape, float2* points,
                     const float* x, int num_points, int diameter) {
  _loop_over_kernels(2, lanczos_kernel, scatter, grid, grid_shape, points, x,
                     num_points, diameter);
}
