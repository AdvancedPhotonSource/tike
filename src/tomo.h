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

  The coordinates of the grid are (x, y, z). The lines are all perpendicular
  to the z direction. Theta is the angle from the x-axis using the right hand
  rule. v is parallel to z, and h is parallel to y when theta is zero.

  TODO: Convert project to theta, h, v coordinates
*/
void
project(
    const float *obj,
    int ox,
    int oy,
    int oz,
    const float *x,
    const float *y,
    const float *theta,
    int dsize,
    float *data);

/**
  Return an array of size (ox, oy, oz) where each element of the array contains
  the sum of the lengths*line_weights of all intersections with the lines
  described by theta, h, and v. The grid defining the array is centered on the
  origin.

  The coordinates of the grid are (x, y, z). The lines are all perpendicular
  to the z direction. Theta is the angle from the x-axis using the right hand
  rule. v is parallel to z, and h is parallel to y when theta is zero.
*/
void
coverage(
    int ox,
    int oy,
    int oz,
    const float *theta,
    const float *h,
    const float *v,
    const float *weights,
    int dsize,
    float *cov);


// Utility functions for data simultation
/**
  Fill gridx and gridy with floats reprsenting the boundaries of the gridlines
  in the x and y directions. Gridlines are centered on zero and spaced 1.0
  apart.
*/
void
preprocessing(
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
  Put the list of intersections with the grid into coordx coordy.
*/
void
calc_coords(
    int ngridx, int ngridy,
    float xi, float yi,
    float sin_p, float cos_p,
    const float *gridx, const float *gridy,
    float *coordx, float *coordy);
/**
  Merge the (coordx, gridy) and (gridx, coordy)
*/
void
trim_coords(
    int ngridx, int ngridy,
    const float *coordx, const float *coordy,
    const float *gridx, const float *gridy,
    int *asize, float *ax, float *ay,
    int *bsize, float *bx, float *by);

/**
  Sort the array of intersection points (ax, ay) and (bx, by). The new sorted
  intersection points are stored in (coorx, coory). Total number of points are
  csize.
*/
void
sort_intersections(
    int ind_condition,
    int asize, const float *ax, const float *ay,
    int bsize, const float *bx, const float *by,
    int *csize,
    float *coorx, float *coory);

/**
  Calculate the distances (dist) between the intersection points
  (coorx, coory). Find the indices of the pixels on the object grid.
*/
void
calc_dist(
    int ngridx, int ngridy, int ngridz,
    int csize,
    const float *coorx, const float *coory,
    int *indi,
    float *dist);

/**
  Multiply the distances by the weights then add them to the coverage map at
  locations defined by index_xy[i] + index_z
*/
void
calc_coverage(
    int data_size,
    int index_z,
    const int *index_xy,
    const float *distances,
    const float line_weight,
    float *coverage_map);

/**
  Multiply the distances by the weights then sum over the line.
*/
void
calc_simdata(
    const float *grid_weights,
    int data_size,
    int index_z,
    const int *index_xy,
    const float *distances,
    int index_line,
    float *data);

#endif
