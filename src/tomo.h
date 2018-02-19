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

void
coverage(
    int ox,
    int oy,
    int oz, 
    const float *x,
    const float *y,
    const float *theta,
    int dsize, 
    float *cov);


// Utility functions for data simultation

void
preprocessing(
    int ngridx,
    int ngridy,
    float *gridx,
    float *gridy);


int
calc_quadrant(
    float theta_p); 


void
calc_coords(
    int ngridx, int ngridy,
    float xi, float yi,
    float sin_p, float cos_p,
    const float *gridx, const float *gridy,
    float *coordx, float *coordy);


void
trim_coords(
    int ngridx, int ngridy,
    const float *coordx, const float *coordy,
    const float *gridx, const float *gridy,
    int *asize, float *ax, float *ay, 
    int *bsize, float *bx, float *by);


void
sort_intersections(
    int ind_condition, 
    int asize, const float *ax, const float *ay,
    int bsize, const float *bx, const float *by,
    int *csize, 
    float *coorx, float *coory);


void
calc_dist(
    int ngridx, int ngridy, 
    int csize, 
    const float *coorx, const float *coory,
    int *indi, 
    float *dist);


void
calc_coverage(
    int ry, 
    int rz, 
    int csize,
    float slice, 
    const int *indi, 
    const float *dist,
    float *data);


void
calc_simdata(
    const float *obj, 
    int ry, 
    int rz, 
    int csize,
    float slice, 
    const int *indi, 
    const float *dist,
    int ray, 
    float *data);

#endif