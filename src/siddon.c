#include "siddon.h"
#include "limits.h"

/* @brief Use method by Siddon (1984) to compute the intersections of rays with
          the grid lines.

Siddon, R. L. (1984). Fast calculation of the exact radiological path for a
three‐dimensional CT array. Medical Physics, 12(2), 252–255.
https://doi.org/10.1118/1.595715
*/
void
get_pixel_indexes_and_lengths(
    const float ozmin, const float oxmin, const float oymin,
    const float zsize, const float xsize, const float ysize,
    const unsigned oz, const unsigned ox, const unsigned oy,
    const float * const theta, const float * const h, const float * const v,
    const unsigned dsize,
    const float *gridx, const float *gridy,
    unsigned **pixels, unsigned **rays, float **lengths, unsigned *psize)
{
    // Check inputs for valid values
    assert(oz > 0 && ox > 0 && oy > 0
           && "Array dimensions must be larger than zero in all dimensions.");
    assert(zsize > 0 && xsize > 0 && ysize > 0
           && "Bounding box must be larger than zero in all dimensions.");
    assert(dsize >= 0 && "Data size must be a natural number");
    assert(theta != NULL && h != NULL && v != NULL
           && "Input data must not be NULL.");
    // Coordinate pairs of gridx gridy where the line intersects the grid
    float *coordy = malloc(sizeof *coordy * (ox+oy+2));
    float *coordx = malloc(sizeof *coordx * (ox+oy+2));
    // Distances between (gridx, coordy) points
    float *ax = (float *)malloc((ox+oy+2)*sizeof(float));
    float *ay = (float *)malloc((ox+oy+2)*sizeof(float));
    // Distances between (gridy, coordx) points
    float *bx = (float *)malloc((ox+oy+2)*sizeof(float));
    float *by = (float *)malloc((ox+oy+2)*sizeof(float));
    // All intersection points sorted into one array
    float *coorx = coordx;
    float *coory = coordy;
    // Final distances between intersection points
    // float *dist = bx;
    // Midpoints between intersection points used to find indices
    float *midx = ax;
    float *midy = ay;
    assert(coordx != NULL && coordy != NULL &&
        ax != NULL && ay != NULL && by != NULL && bx != NULL &&
        coorx != NULL && coory != NULL &&
        midx != NULL && midy != NULL);

    unsigned ray, quadrant, zi, asize, bsize, csize;
    float ri, hi, theta_p, sin_p, cos_p;

    (*psize) = 0;
    unsigned num_lists = 0;
    unsigned *csize_list = malloc(sizeof *csize_list * dsize);
    unsigned *rays_list = malloc(sizeof *rays_list *dsize);
    unsigned * *pixels_list = malloc(sizeof *pixels_list * dsize);
    float * *lengths_list = malloc(sizeof *lengths_list * dsize);

    for (ray = 0; ray < dsize; ray++)
    {
        zi = floor((v[ray]-ozmin) * oz / zsize);
        // Skip this ray if it is out of the z range
        if ((0 <= zi) && (zi < oz))
        {
            // Index of the grid that the ray passes through.
            unsigned *indi = malloc(sizeof *indi * (ox+oy+1));
            float *dist = malloc(sizeof *dist * (ox+oy+1));
            assert(dist != NULL && indi != NULL);
            // Calculate the sin and cos values of the projection angle and find
            // at which quadrant on the cartesian grid.
            theta_p = fmod(theta[ray], 2*M_PI);
            quadrant = calc_quadrant(theta_p);
            sin_p = sinf(theta_p);
            cos_p = cosf(theta_p);
            ri = abs(oxmin)+abs(oymin)+ox+oy;
            hi = h[ray]+1e-6;
            calc_coords(
                ox, oy, ri, hi, sin_p, cos_p, gridx, gridy, coordx, coordy);
            trim_coords(
                ox, oy, coordx, coordy, gridx, gridy,
                &asize, ax, ay, &bsize, bx, by);
            sort_intersections(
                quadrant, asize, ax, ay, bsize, bx, by, &csize, coorx, coory);
            calc_dist(
                csize, coorx, coory, midx, midy, dist);
            calc_index(
                ox, oy, oz, oxmin, oymin, ozmin, xsize/ox, ysize/oy, zsize/oz,
                csize-1, midx, midy, zi, indi);
            if (csize > 1)
            {
                // indi.size is csize - 1
                *psize += csize-1;
                csize_list[num_lists] = csize-1;
                rays_list[num_lists] = ray;
                pixels_list[num_lists] = indi;
                lengths_list[num_lists] = dist;
                num_lists++;
            }
        }
    }
    free(coordx);
    free(coordy);
    free(ax);
    free(ay);
    free(bx);
    free(by);
    // free(coorx); -> coorx
    // free(coory); -> coorx
    // free(dist); -> bx
    // free(midx); -> ax
    // free(midy); -> ay

    // Copy all of the intersections to one array
    *pixels = malloc(sizeof **pixels * psize[0]);
    *rays = malloc(sizeof **rays * psize[0]);
    *lengths = malloc(sizeof **lengths * psize[0]);
    assert(*pixels != NULL && *rays != NULL && *lengths != NULL);
    unsigned j = 0;
    for(unsigned i=0; i < num_lists; i++)
    {
        // Copy the data from buffers to final array
        memcpy(&(*pixels)[j], pixels_list[i], sizeof **pixels * csize_list[i]);
        free(pixels_list[i]);
        memcpy(&(*lengths)[j], lengths_list[i], sizeof **lengths * csize_list[i]);
        free(lengths_list[i]);
        for(unsigned k=0; k < csize_list[i]; k++)
        {
            (*rays)[j + k] = rays_list[i];
        }
        j += csize_list[i];
    }
    free(csize_list);
    free(rays_list);
    free(pixels_list);
    free(lengths_list);
}


int
calc_quadrant(
    const float theta_p)
{
    int quadrant;
    if ((theta_p >= 0 && theta_p < M_PI/2) ||
        (theta_p >= M_PI && theta_p < 3*M_PI/2) ||
        (theta_p >= -M_PI && theta_p < -M_PI/2) ||
        (theta_p >= -2*M_PI && theta_p < -3*M_PI/2))
    {
        quadrant = 1;
    }
    else
    {
        quadrant = 0;
    }
    return quadrant;
}


void
calc_coords(
    const int ngridx, const int ngridy,
    const float xi, const float yi,
    const float sin_p, const float cos_p,
    const float *gridx, const float *gridy,
    float * const coordx, float * const coordy)
{
    float srcx, srcy, detx, dety;
    float slope, islope;
    int n;

    srcx = xi*cos_p-yi*sin_p;
    srcy = xi*sin_p+yi*cos_p;
    detx = -xi*cos_p-yi*sin_p;
    dety = -xi*sin_p+yi*cos_p;

    slope = (srcy-dety)/(srcx-detx);
    islope = 1/slope;
    for (n=0; n<=ngridx; n++)
    {
        coordy[n] = slope*(gridx[n]-srcx)+srcy;
    }
    for (n=0; n<=ngridy; n++)
    {
        coordx[n] = islope*(gridy[n]-srcy)+srcx;
    }
}


void
trim_coords(
    const int ox, const int oy,
    const float *coordx, const float *coordy,
    const float *gridx, const float* gridy,
    int *asize, float *ax, float *ay,
    int *bsize, float *bx, float *by)
{
    int n;

    *asize = 0;
    // bool ascending = coordx[0] < coordx[1];
    for (n=0; n<=oy; n++)
    {
        if (gridx[0] <= coordx[n] && coordx[n] < gridx[ox])
        {
          // assert(n==0 || (ascending && coordx[n] > coordx[n-1]) || (!ascending && coordx[n] <= coordx[n-1]));
          ax[*asize] = coordx[n];
          ay[*asize] = gridy[n];
          (*asize)++;
        }
    }

    *bsize = 0;
    // ascending = coordy[0] < coordy[1];
    for (n=0; n<=ox; n++)
    {
        if (gridy[0] <= coordy[n] && coordy[n] < gridy[oy])
        {
          // assert(n==0 || (ascending && coordy[n] > coordy[n-1]) || (!ascending && coordy[n] <= coordy[n-1]));
          bx[*bsize] = gridx[n];
          by[*bsize] = coordy[n];
          (*bsize)++;
        }
    }
}


/*
TODO: Check the last element of the first array is less or equal to the first
element of the second array.
TODO: Check the arrays are both ascending or descending
TODO: Use memcpy instead of explicit copy
*/
void
sort_intersections(
    const int ind_condition,
    const int asize, const float *ax, const float *ay,
    const int bsize, const float *bx, const float *by,
    int *csize, float *coorx, float *coory)
{
    int i=0, j=0, k=0;
    int a_ind;
    while (i<asize && j<bsize)
    {
        a_ind = (ind_condition) ? i : (asize-1-i);
        if (ax[a_ind] < bx[j])
        {
            coorx[k] = ax[a_ind];
            coory[k] = ay[a_ind];
            i++;
            k++;
        }
        else
        {
            coorx[k] = bx[j];
            coory[k] = by[j];
            j++;
            k++;
        }
    }
    while (i < asize)
    {
        a_ind = (ind_condition) ? i : (asize-1-i);
        coorx[k] = ax[a_ind];
        coory[k] = ay[a_ind];
        i++;
        k++;
    }
    while (j < bsize)
    {
        coorx[k] = bx[j];
        coory[k] = by[j];
        j++;
        k++;
    }
    *csize = asize+bsize;
}


void
calc_dist(
    int const csize, const float *coorx, const float *coory,
    float *midx, float *midy, float *dist)
{
    int n;
    float diffx, diffy;
    for (n=0; n<csize-1; n++)
    {
        diffx = coorx[n+1]-coorx[n];
        diffy = coory[n+1]-coory[n];
        dist[n] = sqrt(diffx*diffx+diffy*diffy);
        midx[n] = (coorx[n+1]+coorx[n])/2.00;
        midy[n] = (coory[n+1]+coory[n])/2.00;
        // Divide by 2 + eps in order to keep midxy inside the grid when both
        // coords are inside the grid. Numerical instability sometimes pushes
        // the sum of midxy and oxmin outside the grid
    }
}


void
calc_index(
    int const ox, int const oy, int const oz,
    float const oxmin, float const oymin, float const ozmin,
    float const xstep, float const ystep, float const zstep,
    int const msize, const float * const midx, const float * const midy,
    int const indz, unsigned * const indi)
{
    assert(UINT_MAX/ox/oy/oz > 0 && "Array is too large to index.");
    // printf("SHAPE: %d, %d, %d : %f, %f, %f : %f, %f, %f\n",
    //         ox, oy, oz, oxmin, oymin, ozmin, xstep, ystep, zstep);
    unsigned n, indx, indy;
    for (n=0; n<msize; n++)
    {
        // Midpoints assigned to pixels by nearest mincorner
        // Ensure above zero because some rounding error can cause dip below
        indx = floor((midx[n]-oxmin) / xstep);
        assert(indx < (unsigned)ox);
        // indx = fmin(fmax(0, indx), ox-1);
        indy = floor((midy[n]-oymin) / ystep);
        assert(indy < (unsigned)oy);
        // indy = fmin(fmax(0, indy), oy-1);
        // Convert from 3D to linear C-order indexing
        indi[n] = indy + oy * (indx + ox * (indz));
        assert(0 <= indi[n] && indi[n] < (unsigned)ox*oy*oz);
    }
}
