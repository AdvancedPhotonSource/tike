// Extract padded patches from an image at scan locations OR add padded patches
// to an image at scan locations.

// The forward kernel extracts patches using linear interpolation at each of
// the scan points and includes optional padding. Assumes square patches, but
// rectangular image. The reverse kernel adds the patches to the images.
// Padding areas are untouched and retain whatever values they had before the
// kernel was launched.

// The kernel should be launched with the following maximum shapes:
// grid shape = (patch_size, nscan, nimage)
// block shape = (min(max_thread, patch_size), 1, 1)

// images has shape (nimage, nimagey, nimagex)
// patches has shape (nscan, patch_shape, patch_shape)
// nscan is the number of positions per images
// scan has shape (nimage, nscan)
extern "C" __global__ void
patch(float2 *images, float2 *patches, const float2 *scan, int nimage,
      int nimagey, int nimagex, int nscan, int patch_shape, int padded_shape,
      bool forward) {
  const int pad = (padded_shape - patch_shape) / 2;

  // for each image
  for (int ti = blockIdx.z; ti < nimage; ti += gridDim.z) {
    // for each scan position
    for (int ts = blockIdx.y; ts < nscan; ts += gridDim.y) {
      // x,y scan coordinates in image
      const float sx = floor(scan[ts + ti * nscan].y);
      const float sy = floor(scan[ts + ti * nscan].x);

      if (sx < 0 || nimagex <= sx + patch_shape || sy < 0
          || nimagey <= sy + patch_shape) {
        // printf("%f, %f - %f, %f\n", sx, sy, sxf, syf);
        assert(false);
        return;
      }

      // for x,y coords in patch
      for (int py = blockIdx.x; py < patch_shape; py += gridDim.x) {
        for (int px = threadIdx.x; px < patch_shape; px += blockDim.x) {
          // linear patch index (pi)
          // clang-format off
          const int pi = (
            + pad + px + padded_shape * (pad + py)
            + padded_shape * padded_shape * (ts + nscan * ti)
          );
          // clang-format on

          // image index (ii)
          const int ii = sx + px + nimagex * (sy + py + nimagey * ti);

          const float sxf = scan[ts + ti * nscan].y - sx;
          const float syf = scan[ts + ti * nscan].x - sy;
          assert(1.0f >= sxf && sxf >= 0.0f && 1.0f >= syf && syf >= 0.0f);

          // Linear interpolation
          // clang-format off
          if (forward) {
            patches[pi].x = images[ii              ].x * (1.0f - sxf) * (1.0f - syf)
                          + images[ii + 1          ].x * (       sxf) * (1.0f - syf)
                          + images[ii     + nimagex].x * (1.0f - sxf) * (       syf)
                          + images[ii + 1 + nimagex].x * (       sxf) * (       syf);

            patches[pi].y = images[ii              ].y * (1.0f - sxf) * (1.0f - syf)
                          + images[ii + 1          ].y * (       sxf) * (1.0f - syf)
                          + images[ii     + nimagex].y * (1.0f - sxf) * (       syf)
                          + images[ii + 1 + nimagex].y * (       sxf) * (       syf);
          } else {
            const float2 tmp = patches[pi];
            atomicAdd(&images[ii              ].x, tmp.x * (1.0f - sxf) * (1.0f - syf));
            atomicAdd(&images[ii              ].y, tmp.y * (1.0f - sxf) * (1.0f - syf));
            atomicAdd(&images[ii + 1          ].y, tmp.y * (       sxf) * (1.0f - syf));
            atomicAdd(&images[ii + 1          ].x, tmp.x * (       sxf) * (1.0f - syf));
            atomicAdd(&images[ii     + nimagex].x, tmp.x * (1.0f - sxf) * (       syf));
            atomicAdd(&images[ii     + nimagex].y, tmp.y * (1.0f - sxf) * (       syf));
            atomicAdd(&images[ii + 1 + nimagex].x, tmp.x * (       sxf) * (       syf));
            atomicAdd(&images[ii + 1 + nimagex].y, tmp.y * (       sxf) * (       syf));
          }
        }
      }
    }
  }
}
