
// Rotate the 3D point 'p' around the origin.
__device__ void
forward_rotation(float3& p, float ctilt, float stilt, float ctheta,
                 float stheta) {
  float x = ctilt * p.x - stilt * p.y;
  float y = -ctheta * stilt * p.x + ctheta * ctilt * p.y - stheta * p.z;
  float z = -stheta * stilt * p.x + stheta * ctilt * p.y + ctheta * p.z;
  p.x = x;
  p.y = y;
  p.z = z;
}

// Rotate the 3D point 'p' around the origin in the reverse direction.
__device__ void
reverse_rotation(float3& p, float ctilt, float stilt, float ctheta,
                 float stheta) {
  // x unused
  // float x = ctilt * p.x - ctheta * stilt * p.y - stheta * stilt * p.z;
  float y = stilt * p.x + ctheta * ctilt * p.y + stheta * ctilt * p.z;
  float z = -stheta * p.y + ctheta * p.z;
  // x unused
  // p.x = x;
  p.y = y;
  p.z = z;
}

// Project the point 'p' onto the plane with the given normal
__device__ void
project_point_to_plane(float3& point, const float3& normal) {
  float distance = point.x * normal.x + point.y * normal.y + point.z * normal.z;
  point.x = point.x - distance * normal.x;
  point.y = point.y - distance * normal.y;
  point.z = point.z - distance * normal.z;
}

// Get the 2D coordinates of each of the 3D grid points projected onto the
// plane defined by tilt and theta.
extern "C" __global__ void
coordinates_and_weights(const longlong3* grid, const int ngrid,
                        const float tilt, const float* theta, const int t,
                        const int precision, longlong2* plane_coords) {
  // Compute the normal of the projection plane.
  float ctilt = cosf(tilt);
  float stilt = sinf(tilt);
  float ctheta = cosf(theta[t]);
  float stheta = sinf(theta[t]);
  float3 normal = {1.f, 0.f, 0.f};
  forward_rotation(normal, ctilt, stilt, ctheta, stheta);
  // printf("normal is %f, %f, %f\n", normal.x, normal.y, normal.z);

  for (int g = 0; g < ngrid; g++) {
    longlong2* cluster = plane_coords + g * precision * precision * precision;

    // Improve the precision of this method by using a cluster of projections
    // instead of a single point for each grid point.
    for (int i = 0; i < precision; i++) {
      for (int j = 0; j < precision; j++) {
        for (int k = 0; k < precision; k++) {
          float3 point;
          point.x = grid[g].x + (i + 0.5f) / precision;
          point.y = grid[g].y + (j + 0.5f) / precision;
          point.z = grid[g].z + (k + 0.5f) / precision;

          project_point_to_plane(point, normal);
          reverse_rotation(point, ctilt, stilt, ctheta, stheta);

          longlong2* chunk = cluster + k + precision * (j + precision * i);
          chunk->x = floorf(point.y);
          chunk->y = floorf(point.z);
          // printf("point is %lld, %lld\n", chunk->x, chunk->y);
        }
      }
    }
  }
}
