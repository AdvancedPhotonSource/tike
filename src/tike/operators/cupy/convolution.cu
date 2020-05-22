// Extract padded patches from an image at scan locations OR add padded patches
// to an image at scan locations.

// The forward kernel extracts patches using linear interpolation at each of
// the scan points and includes optional padding. Assumes square patches, but
// rectangular image. The reverse kernel adds the patches to the images.
// Padding areas are untouched and retain whatever values they had before the
// kernel was launched.

// The kernel should be launched with the following config:
// block shape = (min(max_thread, patch_size**2), 0, 0)
// grid shape = (-(-patch_size**2 // max_thread), nscan, nimage)

// images has shape (nimage, nimagey, nimagex)
// patches has shape (nscan, patch_shape, patch_shape)
// nscan is the number of positions per images
// scan has shape (nimage, nscan)

extern "C" __global__
void patch(float2 *images, float2 *patches, const float2 *scan,
           int nimage, int nimagey, int nimagex,
           int nscan, int patch_shape, int padded_shape,
           bool forward) {
  const int tp = threadIdx.x + blockDim.x * (blockIdx.x);  // thread patch
  const int ts = blockIdx.y;  // thread scan
  const int ti = blockIdx.z;  // thread image

  if (tp >= patch_shape * patch_shape || ts >= nscan || ti >= nimage) return;

  // patch index (pi)
  const int px = tp % patch_shape;
  const int py = tp / patch_shape;
  const int pad = (padded_shape - patch_shape) / 2;
  const int pi = (
    + pad + px + padded_shape * (pad + py)
    + padded_shape * padded_shape * (ts + nscan * ti)
  );

  const float sx = floor(scan[ts + ti * nscan].y);
  const float sy = floor(scan[ts + ti * nscan].x);

  if (sx < 0 || nimagex <= sx + patch_shape ||
      sy < 0 || nimagey <= sy + patch_shape){
    // printf("%f, %f - %f, %f\n", sx, sy, sxf, syf);
    // scans where the probe position overlaps edges we fill with zeros
    if (forward){
      patches[pi].x = 0.0f;
      patches[pi].y = 0.0f;
    }
    return;
  }

  // image index (ii)
  const int ii = sx + px + nimagex * (sy + py + nimagey * ti);

  const float sxf = scan[ts + ti * nscan].y - sx;
  const float syf = scan[ts + ti * nscan].x - sy;
  //if (tp==0) printf("kernel:%f\n", scan[ts + ti * nscan].y);
  assert(1.0f >= sxf && sxf >= 0.0f && 1.0f >= syf && syf >= 0.0f);

  // Linear interpolation
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
