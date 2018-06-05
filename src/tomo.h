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

void
art(
    const float *data,
    const float *x,
    const float *y,
    const float *theta,
    float *recon);

/**
  Return an array of size (dsize, ) where each element of the array contains
  the sum of the lengths*grid_weights*line_weights of all intersections with
  the lines described by theta, h, and v. The grid is centered on the origin,
  and is the same shape as the obj providing the grid_weights.

  The coordinates of the grid are (z, x, y). The lines are all perpendicular
  to the z direction. Theta is the angle from the x-axis using the right hand
  rule. v is parallel to z, and h is parallel to y when theta is zero.
*/
void
project(
    const float *obj_weights,
    float ozmin,
    float oxmin,
    float oymin,
    int oz,
    int ox,
    int oy,
    const float *theta,
    const float *h,
    const float *v,
    int dsize,
    float *data);

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
    float ozmin,
    float oxmin,
    float oymin,
    int oz,
    int ox,
    int oy,
    const float *theta,
    const float *h,
    const float *v,
    const float *weights,
    int dsize,
    float *cov,
    const bool anisotropy);


// Utility functions for data simultation
/**
*/
void worker_function(
    const float *obj_weights,
    const float ozmin, const float oxmin, const float oymin,
    const int ox, const int oy, const int oz,
    float *data,
    const float *theta, const float *h, const float *v, const float *weights,
    const int dsize,
    const float *gridx, const float *gridy,
    enum mode
    );


/**
  Fill gridx and gridy with floats reprsenting the boundaries of the gridlines
  in the x and y directions. Gridlines start at minx and miny and are
  spaced 1.0 apart.
*/
void
preprocessing(
    float minx,
    float miny,
    int ngridx,
    int ngridy,
    float *gridx,
    float *gridy);

/**
  Return 1 for first and third quadrants, 0 otherwise.
*/
int
calc_quadrant(
    float theta_p);

/**
  Compute the list of intersections of the line (xi, yi) and the grid.
  The intersections are then located in two lists:
  (gridx, coordy) and (coordx, gridy). The length of gridx is ngridx+1.
*/
void
calc_coords(
    int ngridx, int ngridy,
    float xi, float yi,
    float sin_p, float cos_p,
    const float *gridx, const float *gridy,
    float *coordx, float *coordy);

/**
  (coordx, gridy) and (gridx, coordy) are sets of points along a line. Remove
  points from these sets that lay outside the boundaries of the grid.
*/
void
trim_coords(
    int ox, int oy,
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
    int ind_condition,
    int asize, const float *ax, const float *ay,
    int bsize, const float *bx, const float *by,
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
    float const sin_p,
    float const cos_p,
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
    float *data,
    int const ind_data);

#endif
