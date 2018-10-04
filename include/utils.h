#ifndef _utils_h
#define _utils_h

#define _USE_MATH_DEFINES
#ifndef M_PI
#define M_PI 3.14159265358979323846264338327
#endif

/* @brief Fill gridx with n+1 floats across the range [xmin, xmin + xsize].
*/
void
make_grid(
    const float xmin,
    const float xsize,
    const int nx,
    float * const gridx);

/* @brief Adds a magnitude to the appropriate angular bin

Given nbins [0, PI), if theta is goes in the i-th bin, then magnitude is
added to the bins + i.

@param bins A pointer to the 0th bin
@param magnitude The value to add to the ith bin
@param theta The angle used to determine the ith bin
@param nbins The number of angular bins
*/
void
bin_angle(
    float *bins, const float magnitude,
    const float theta, const int nbins);

void
calc_back(
    const float *dist,
    int const dist_size,
    float const line_weight,
    float *cov,
    const int *ind_cov);

#endif
