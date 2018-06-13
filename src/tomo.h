#ifndef _tomo_h
#define _tomo_h

#define _USE_MATH_DEFINES
#ifndef M_PI
#define M_PI 3.14159265358979323846264338327
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <assert.h>
#include "string.h"

enum mode {Forward, Back, Coverage, ART, SIRT};

/**
  Return an array of size (ox, oy, oz) where each element of the array contains
  the sum of the lengths*line_weights of all intersections with the lines
  described by theta, h, and v. The grid defining the array has a minimum
  corner (oxmin, oymin, ozmin).

  The coordinates of the grid are (z, x, y). The lines are all perpendicular
  to the z direction. Theta is the angle from the x-axis using the right hand
  rule. v is parallel to z, and h is parallel to y when theta is zero. The
  rotation axis is [0, 0, 1].
*/
void
coverage(
    const float ozmin, const float oxmin, const float oymin,
    const float zsize, const float xsize, const float ysize,
    const int oz, const int ox, const int oy, const int ot,
    const float *theta,
    const float *h,
    const float *v,
    const float *line_weights,
    const int dsize,
    float *coverage_map);

/**
  Fill gridx and gridy with floats reprsenting the boundaries of the
  n gridlines between min and min + size.
*/
void make_grid(
    const float zmin, const float xmin, const float ymin,
    const float zsize, const float xsize, const float ysize,
    const int nz, const int nx, const int ny,
    float * const gridz, float * const gridx, float * const gridy);

void worker_function(
    float *obj_weights,
    const float ozmin, const float oxmin, const float oymin,
    const float zsize, const float xsize, const float ysize,
    const int oz, const int ox, const int oy, const int ot,
    float *data,
    const float *theta, const float *h, const float *v, const float *weights,
    const int dsize,
    const float *gridx, const float *gridy,
    const enum mode);

/**
  Return 1 for first and third quadrants, 0 otherwise.
*/
int
calc_quadrant(
    const float theta_p);

/**
  Compute the list of intersections of the line (xi, yi) and the grid.
  The intersections are then located in two lists:
  (gridx, coordy) and (coordx, gridy). The length of gridx is ngridx+1.
*/
void
calc_coords(
    const int ngridx, const int ngridy,
    const float xi, const float yi,
    const float sin_p, const float cos_p,
    const float *gridx, const float *gridy,
    float * const coordx, float * const coordy);

/**
  (coordx, gridy) and (gridx, coordy) are sets of points along a line. Remove
  points from these sets that lay outside the boundaries of the grid.
*/
void
trim_coords(
    const int ox, const int oy,
    const float *coordx, const float *coordy,
    const float *gridx, const float *gridy,
    int *asize, float *ax, float *ay,
    int *bsize, float *bx, float *by);

/**
  (ax, ay) and (bx, by) are two sets of ordered points along a line. Combine
  the two sets of points into (coorx, coory). The total number of points is
  asize + bsize = csize.
*/
void
sort_intersections(
    const int ind_condition,
    const int asize, const float *ax, const float *ay,
    const int bsize, const float *bx, const float *by,
    int *csize, float *coorx, float *coory);

/**
  (coorx, coory) describe the ordered points where the line intersects the
  grid. Find the distances between adjacent points and return the midpoints of
  these line segments.
*/
void
calc_dist(
    int const csize, const float *coorx, const float *coory,
    float *midx, float *midy, float *dist);

/** Find the linear index of the pixels containing the points (midx, midy) on
  the grid defined by min corner xmin and size ox.
  */
void
calc_index(
    int const ox, int const oy, int const oz,
    float const oxmin, float const oymin, float const ozmin,
    float const xstep, float const, float const,
    int const msize, const float *midx, const float *midy,
    int const indz, unsigned *indi);

/**
  Multiply the distances by the weights then add them to the coverage map at
  locations defined by index_xyz[i]
*/
void
calc_coverage(
    const float *dist,
    int const dist_size,
    float const line_weight,
    float const theta,
    const int nbins,
    float *cov,
    const unsigned *ind_cov);

void
calc_back(
    const float *dist,
    int const dist_size,
    float const line_weight,
    float *cov,
    const unsigned *ind_cov);
/**
  Multiply the distances by the weights then sum over the line.
*/
void
calc_forward(
    const float *grided_weights,
    const unsigned *ind_grid,
    const float *dist,
    int const dist_size,
    float *data);

void
calc_art(
    float *grided_weights,
    const unsigned *ind_grid,
    const float *dist,
    int const dist_size,
    float *data);

void
calc_sirt(
    const float *grided_weights,
    float *update,
    float *sumdist,
    const unsigned *ind_grid,
    const float *dist,
    int const dist_size,
    float *data);

#endif
