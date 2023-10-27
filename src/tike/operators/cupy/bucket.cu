// https://docs.nvidia.com/cuda/archive/12.2.2/cuda-c-programming-guide/index.html#atomic-functions
#if __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

// Rotate the 3D point 'p' around the origin.
template <typename thetaType, typename thetaType3>
__device__ void
forward_rotation(thetaType3& p, thetaType ctilt, thetaType stilt,
                 thetaType ctheta, thetaType stheta) {
  thetaType x = ctilt * p.x - stilt * p.y;
  thetaType y = -ctheta * stilt * p.x + ctheta * ctilt * p.y - stheta * p.z;
  thetaType z = -stheta * stilt * p.x + stheta * ctilt * p.y + ctheta * p.z;
  p.x = x;
  p.y = y;
  p.z = z;
}

// Rotate the 3D point 'p' around the origin in the reverse direction.
template <typename thetaType, typename thetaType3>
__device__ void
reverse_rotation(thetaType3& p, thetaType ctilt, thetaType stilt,
                 thetaType ctheta, thetaType stheta) {
  // x unused
  // float x = ctilt * p.x - ctheta * stilt * p.y - stheta * stilt * p.z;
  thetaType y = stilt * p.x + ctheta * ctilt * p.y + stheta * ctilt * p.z;
  thetaType z = -stheta * p.y + ctheta * p.z;
  // x unused
  // p.x = x;
  p.y = y;
  p.z = z;
}

// Project the point 'p' onto the plane with the given normal
template <typename thetaType, typename thetaType3>
__device__ void
project_point_to_plane(thetaType3& point, const thetaType3& normal) {
  thetaType distance
      = point.x * normal.x + point.y * normal.y + point.z * normal.z;
  point.x = point.x - distance * normal.x;
  point.y = point.y - distance * normal.y;
  point.z = point.z - distance * normal.z;
}

// Get the 2D coordinates of each of the 3D grid points projected onto the
// plane defined by tilt and theta.
// grid shape (ngrid, 0, 0)
// block shape (precision, precision, precision)
template <typename thetaType, typename thetaType3>
__global__ void
coordinates_and_weights(const short3* grid, const int ngrid, const float tilt,
                        const thetaType* theta, const int t,
                        const int precision, short2* plane_coords) {
  // Compute the normal of the projection plane.
  thetaType ctilt = cosf(tilt);
  thetaType stilt = sinf(tilt);
  thetaType ctheta = cosf(theta[t]);
  thetaType stheta = sinf(theta[t]);
  thetaType3 normal = {1.f, 0.f, 0.f};
  forward_rotation<thetaType, thetaType3>(normal, ctilt, stilt, ctheta, stheta);
  // printf("normal is %f, %f, %f\n", normal.x, normal.y, normal.z);

  for (int g = blockIdx.x; g < ngrid; g += gridDim.x) {
    short2* cluster = plane_coords + g * precision * precision * precision;

    // Improve the precision of this method by using a cluster of projections
    // instead of a single point for each grid point.
    for (int i = threadIdx.z; i < precision; i += blockDim.z) {
      for (int j = threadIdx.y; j < precision; j += blockDim.y) {
        for (int k = threadIdx.x; k < precision; k += blockDim.x) {
          thetaType3 point;
          point.x = grid[g].x + (i + 0.5f) / precision;
          point.y = grid[g].y + (j + 0.5f) / precision;
          point.z = grid[g].z + (k + 0.5f) / precision;

          project_point_to_plane<thetaType, thetaType3>(point, normal);
          reverse_rotation<thetaType, thetaType3>(point, ctilt, stilt, ctheta,
                                                  stheta);

          short2* chunk = cluster + k + precision * (j + precision * i);
          chunk->x = floorf(point.y);
          chunk->y = floorf(point.z);
          // printf("point is %lld, %lld\n", chunk->x, chunk->y);
        }
      }
    }
  }
}

template <typename dataType>
__global__ void
fwd(dataType* data, int t, int datashapex, int datashapey, double weight,
    const dataType* u, int ushapex, int ushapey, int ushapez,
    const short2* plane_index, const short3* grid_index, int gridshapex,
    int precision) {
  int nchunk = precision * precision * precision;
  for (int g = blockIdx.x; g < gridshapex; g += gridDim.x) {
    const dataType* ui
        = u + grid_index[g].z
          + ushapez * (grid_index[g].y + ushapey * (grid_index[g].x));
    for (int p = threadIdx.x; p < nchunk; p += blockDim.x) {
      const short2* pi = plane_index + p + g * nchunk;
      dataType* di = data + pi->y + datashapey * (pi->x + datashapex * (t));
      atomicAdd(&(di->x), weight * ui->x);
      atomicAdd(&(di->y), weight * ui->y);
    }
  }
}

template <typename dataType>
__global__ void
adj(const dataType* data, int t, int datashapex, int datashapey, double weight,
    dataType* u, int ushapex, int ushapey, int ushapez,  //
    const short2* plane_index, const short3* grid_index, int gridshapex,
    int precision) {
  int nchunk = precision * precision * precision;
  for (int g = blockIdx.x; g < gridshapex; g += gridDim.x) {
    dataType* ui = u + grid_index[g].z
                   + ushapez * (grid_index[g].y + ushapey * (grid_index[g].x));
    for (int p = threadIdx.x; p < nchunk; p += blockDim.x) {
      const short2* pi = plane_index + p + g * nchunk;
      const dataType* di
          = data + pi->y + datashapey * (pi->x + datashapex * (t));
      atomicAdd(&(ui->x), weight * di->x);
      atomicAdd(&(ui->y), weight * di->y);
    }
  }
}
