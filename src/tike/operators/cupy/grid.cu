
// Compute the planar USFFT frequencies for laminography.
//
// The frequencies are in the range [-0.5, 0.5) and planes at sampled to an (N,
// N) grid. The tilt angle is the same for all planes, but the rotation angle
// is unique for each plane. Radians for all angles. The shape of frequency
// should be (R * N * N, 3) where R is the number of rotations. The shape of
// rotation is (R, ).

// Each thread gets one frequency.
// grid shape (-(-N // max_threads), N, R)
// block shape (min(N, max_threads), 0, 0)
template <typename frequencyType, typename rotationType>
__global__ void
make_grids(frequencyType* frequency, const rotationType* rotation, int R, int N,
           float tilt) {
  frequencyType ctilt = cosf(tilt);
  frequencyType stilt = sinf(tilt);

  for (int p = blockIdx.z; p < R; p += gridDim.z) {
    frequencyType ctheta = cosf(rotation[p]);
    frequencyType stheta = sinf(rotation[p]);
    // NOTE: Use pointer arithmetic to avoid indexing overflows without using
    // size_t.
    frequencyType* plane = 3 * N * N * p + frequency;

    for (int y = blockIdx.y; y < N; y += gridDim.y) {
      frequencyType kv = (frequencyType)(y - N / 2) / N;
      frequencyType* height = 3 * N * y + plane;

      // clang-format off
      for (
        int x = threadIdx.x + blockDim.x * blockIdx.x;
        x < N;
        x += blockDim.x * gridDim.x
      ) {
        // clang-format on
        frequencyType ku = (frequencyType)(x - N / 2) / N;
        frequencyType* f = 3 * x + height;

        f[0] = +kv * stilt;
        f[1] = -ku * stheta + kv * ctheta * ctilt;
        f[2] = +ku * ctheta + kv * stheta * ctilt;
      }
    }
  }
}
