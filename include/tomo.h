#ifndef _tomo_h
#define _tomo_h

/** @brief Computes line integrals across a 3D object.

Computes the sum of the lengths * grid_weights of all intersections with
the lines described by theta, h, and v and the grid.

The coordinates of the grid are (z, x, y). The lines are all perpendicular
to the z direction. Theta is the angle from the x-axis using the right hand
rule. v is parallel to z, and h is parallel to y when theta is zero.

@param obj_weights The weights of the grid being integrated over.
@param min The minimum coordinates of the grid.
@param n The number of grid spaces along the grid.
@param theta, h, v The coordinates of the line for each integral.
@param dsize The size of theta, h, v.
@param data The line integrals.
*/
void
forward_project(
    const float *obj_weights,
    const float zmin, const float xmin, const float ymin,
    const int nz, const int nx, const int ny,
    const float *theta,
    const float *h,
    const float *v,
    const int dsize,
    float *data);

/* @brief Back project lines over a 4D grid where the 4th dimension is an
          an angular bin.

coverage_map is an array of size (nz, nx, ny, nt) where each element of the
array contains the sum of the lengths*line_weights of all intersections with
the lines described by theta, h, and v. The sums are binned by angle of the
line into `nt` bins from 0 to PI.

The coordinates of the grid are (z, x, y). The lines are all perpendicular
to the z direction. Theta is the angle from the x-axis using the right hand
rule. v is parallel to z, and h is parallel to y when theta is zero. The
rotation axis is [0, 0, 1].

@param min The minimum coordinates of the grid.
@param n The number of grid spaces along the grid.
@param theta, h, v The coordinates of the lines.
@param line_weights The weight of each line in the integral.
@param dsize The length of theta, h, v, line_weights.
@param coverage_map The grid to project over.
*/
void
coverage(
    const float zmin, const float xmin, const float ymin,
    const float zsize, const float xsize, const float ysize,
    const int nz, const int nx, const int ny, const int nt,
    const float *theta,
    const float *h,
    const float *v,
    const float *line_weights,
    const int dsize,
    float *coverage_map);

/* @brief Algebraic Reconstruction Technique

@param min The minimum coordinates of the grid.
@param size The width of the grid.
@param n The number of grid spaces along the grid.
@param data The measured line integral of each line.
@param theta, h, v The coordinates of the lines.
@param ndata The length of data, theta, h, and v
@param init The initial guess of the reconstruction
@param niter The number of iterations of ART to complete
*/
void
art(
    const float zmin, const float xmin, const float ymin,
    const int nz, const int nx, const int ny,
    const float * const data,
    const float * const theta,
    const float * const h,
    const float * const v,
    const int ndata,
    float * const init,
    const int niter);

#endif
