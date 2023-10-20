// Extract padded patches from an image at scan locations OR add padded patches
// to an image at scan locations.

// The forward kernel extracts patches using linear interpolation at each of
// the scan points and includes optional padding. Assumes square patches, but
// rectangular image. The reverse kernel adds the patches to the images.
// Padding areas are untouched and retain whatever values they had before the
// kernel was launched.

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

// Consider the point 0.0 in 1 dimension. The weight distribution should be
// 0 [[ w = 1.0 ]] 1 [ w = 0.0 ] 2
// Consider the point 1.2 in 1 dimension. The weight distribution should be
// 0 [ ] 1 [ [w = 1 - 0.2] ] 2 [ [w = 0.2] ] 3
template <typename patchType, typename imageType, typename scanType>
__device__ void
_forward(patchType *patches, imageType *images, int nimagex, int pi, int ii,
         const scanType w[4]) {
  // clang-format off
  patches[pi].x = images[ii              ].x * w[0]
                + images[ii + 1          ].x * w[1]
                + images[ii     + nimagex].x * w[2]
                + images[ii + 1 + nimagex].x * w[3];
  patches[pi].y = images[ii              ].y * w[0]
                + images[ii + 1          ].y * w[1]
                + images[ii     + nimagex].y * w[2]
                + images[ii + 1 + nimagex].y * w[3];
  // clang-format on
}

template <typename patchType, typename imageType, typename scanType>
__device__ void
_adjoint(patchType *patches, imageType *images, int nimagex, int pi, int ii,
         const scanType w[4]) {
  const patchType tmp = patches[pi];
  // clang-format off
  atomicAdd(&images[ii              ].x, tmp.x * w[0]);
  atomicAdd(&images[ii              ].y, tmp.y * w[0]);
  atomicAdd(&images[ii + 1          ].y, tmp.y * w[1]);
  atomicAdd(&images[ii + 1          ].x, tmp.x * w[1]);
  atomicAdd(&images[ii     + nimagex].x, tmp.x * w[2]);
  atomicAdd(&images[ii     + nimagex].y, tmp.y * w[2]);
  atomicAdd(&images[ii + 1 + nimagex].x, tmp.x * w[3]);
  atomicAdd(&images[ii + 1 + nimagex].y, tmp.y * w[3]);
  // clang-format on
}

// NOTE: Uses the same loop structure for forward and adjoint operations
// because inverting the loop structure to loop over image pixels instead of
// patch pixels is much slower. There are many more checks and conditionals in
// order to avoid atomic operations.

// NOTE: nscan is the first grid dimension because it is has a larger maximum
// (2^31) than the later two (6k).

// The kernel should be launched with the following maximum shapes:
// grid shape = (nscan, nimage, patch_size)
// block shape = (min(max_thread, patch_size), 1, 1)
template <typename patchType, typename imageType, typename scanType>
__device__ void
_loop_over_patches(
    void operation(patchType *, imageType *, int, int, int, const scanType[4]),
    imageType *images,     // has shape (nimage, nimagey, nimagex)
    patchType *patches,    // has shape (nscan, patch_shape, patch_shape)
    const scanType *scan,  // has shape (nimage, nscan)
    int nimage, int nimagey, int nimagex,
    int nscan,         // the number of positions per images
    int nrepeat,       // number of times to repeat the patch
    int patch_shape,   // the width of the valid area of the patch
    int padded_shape,  // the width of the patch including padding
    int npatch         // the number of unique patches
) {
  const int pad = (padded_shape - patch_shape) / 2;

  // for each image
  for (int ti = blockIdx.y; ti < nimage; ti += gridDim.y) {
    const int image_offset = padded_shape * padded_shape * nrepeat * nscan * ti;
    // for each scan position
    for (int ts = blockIdx.x; ts < nscan; ts += gridDim.x) {
      // x,y scan coordinates in image
      size_t s_index = 2 * (ts + ti * nscan);
      const scanType sx = floor(scan[s_index + 1]);
      const scanType sy = floor(scan[s_index]);

      const scanType sxf = scan[s_index + 1] - sx;
      const scanType syf = scan[s_index] - sy;
      assert(1.0f >= sxf && sxf >= 0.0f && 1.0f >= syf && syf >= 0.0f);

      // for x,y coords in patch
      for (int py = blockIdx.z; py < patch_shape; py += gridDim.z) {
        if (sy + py < 0 || nimagey <= sy + py) continue;

        const scanType _syf = ((nimagey > sy + py) ? syf : 0.0f);

        // Only x coodrinate is used in thread block because the max number
        // of threads on an SM is too small for one patch
        for (int px = threadIdx.x; px < patch_shape; px += blockDim.x) {
          if (sx + px < 0 || nimagex <= sx + px) continue;

          // linear patch index (pi)
          const int pi = pad + px + padded_shape * (pad + py) + image_offset;
          // image index (ii)
          const int ii = sx + px + nimagex * (sy + py + nimagey * ti);

          // Linear interpolation. Ternary sets trailing pixel weights to
          // zero when leading pixel is at the edge of the grid.
          const scanType _sxf = ((nimagex > sx + px) ? sxf : 0.0f);
          const scanType w[4] = {
              (1.0f - sxf) * (1.0f - syf),
              (sxf) * (1.0f - syf),
              (1.0f - sxf) * (syf),
              (sxf) * (syf),
          };

          for (int r = 0; r < nrepeat; ++r) {
            const int r_offset
                = padded_shape * padded_shape * (r + (nrepeat * ts) % npatch);
            operation(patches, images, nimagex, pi + r_offset, ii, w);
          }
        }
      }
    }
  }
}

template <typename patchType, typename imageType, typename scanType>
__global__ void
fwd_patch(imageType *images, patchType *patches, const scanType *scan,
          int nimage, int nimagey, int nimagex, int nscan, int nrepeat,
          int patch_shape, int padded_shape) {
  _loop_over_patches<patchType, imageType, scanType>(
      _forward<patchType, imageType, scanType>, images, patches, scan, nimage,
      nimagey, nimagex, nscan, nrepeat, patch_shape, padded_shape,
      nscan * nrepeat);
}

template <typename patchType, typename imageType, typename scanType>
__global__ void
adj_patch(imageType *images, patchType *patches, const scanType *scan,
          int nimage, int nimagey, int nimagex, int nscan, int nrepeat,
          int patch_shape, int padded_shape, int npatch) {
  _loop_over_patches<patchType, imageType, scanType>(
      _adjoint<patchType, imageType, scanType>, images, patches, scan, nimage,
      nimagey, nimagex, nscan, nrepeat, patch_shape, padded_shape, npatch);
}
