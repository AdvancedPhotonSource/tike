#ifndef _raytrace_h
#define _raytrace_h

#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <stdbool.h>


/** @brief Allocates and fills the projection and backprojection sparse
matricies

Given some rays defined by theta and rho, the intersections of the rays and a
rectangular domain of size nx * ny are computed. For each ray, the indices of
pixels that are touched and the length of intersection are computed, and for
each pixel, the indicies of the rays that touch it and the length o
intersection are computed.

@param nray The length of theta and rho.
@param theta The angles of the rays intersecting the matrix.
@param rho The distance of the rays from the origin.
@param nx, ny The number of voxels along the x, y dimension.
@param xmin, ymin The minimum corner of the domain.
@param xsize, ysize The width of the domain in the x, y dimension.
...
@return num_pix[i] The number of pixels touched by the ith ray.
@return pix_start[i] The start of the block in pix and pix_lengths for the ith
        ray.
@return pix The indices the pixels touched by rays.
@return pix_lengths The lengths of intersection between the pixels and rays.
@return num_rays[j] The number of rays touching the jth pixel.
@return ray_start[j] The start of the block in ray and ray_lengths for the jth
        pixel.
@return rays The indices of the rays touching pixels.
@return ray_lengths The lengths of intersection between the pixels and rays.
*/
void get_intersections_and_lengths(
    const int nray, const float * const theta, const float * const rho,
    const int nx, const float xmin, const float xsize,
    const int ny, const float ymin, const float ysize
);

/* @brief Calculates the number of pixels touched by the ray defined by theta
and rho.

TODO

*/
void findnumpix(
    float theta, float rho,
    int *numpix, float *domain,
    float res, int nx, int ny
);

/* @brief TODO

TODO

*/
void placepixels(float theta, float rho, int *indices,
    float *weights,
    int *numpix, float *domain, float res,
    int nx, int ny);

/*@brief Backprojects a sinogram over an object

For the each of the npix pixels that have been calculated to intersect with the
rays, add the ray_lengths * sinogram to object

@param npix The size of pix, num_rays, and ray_start.
@param num_rays[j] The number of rays touching the jth pixel.
@param ray_start[j] The start of the block in ray and ray_lengths for the jth
        pixel.
@param rays The indices of the rays touching pixels.
@param ray_lengths The lengths of intersection between the pixels and rays.
@param object The weights of each pixel
@param sinogram The weights of each ray
*/
void back_project(
    const int npix,
    const int * const num_rays, const int * const ray_start,
    const int * const rays, const float * const ray_lengths,
    float * const object, const float * const sinogram);

/*@brief Compute weighted ray-sums over an object

For the each of the nrays rays that have been calculated to intersect with the
domain, compute a ray-sum over the object.

@param nrays The size of rays, num_pix, and pix_start
@param num_pix[i] The number of pixels touched by the ith ray.
@param pix_start[i] The start of the block in pix and pix_lengths for the ith
        ray.
@param pix The indices the pixels touched by rays.
@param pix_lengths The lengths of intersection between the pixels and rays.
@param object The weights of each pixel
@param sinogram The ray sums of each ray
*/
void forward_project(
    const int nrays,
    const int * const num_pix, const int * const pix_start,
    const int * const pix, const float * const pix_lengths,
    const float * const object, float * const sinogram);

#endif
