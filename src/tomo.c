#include "tomo.h"
#include "limits.h"
#include <omp.h>

/**
Siddon, R. L. (1984). Fast calculation of the exact radiological path for a
three‐dimensional CT array. Medical Physics, 12(2), 252–255.
https://doi.org/10.1118/1.595715
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
    float *coverage_map)
{
    assert(UINT_MAX/ox/oy/oz/ot > 0 && "Array is too large to index.");
    assert(oz > 0 && ox > 0 && oy > 0 && ot > 0);
    // Initialize the grid on object space.
    float *gridx = malloc(sizeof *gridx * (ox+1));
    float *gridy = malloc(sizeof *gridy * (oy+1));
    assert(gridx != NULL && gridy != NULL);
    make_grid(
        0.0, oxmin, oymin, 0.0, xsize, ysize, -1, ox, oy,
        NULL, gridx, gridy);

    enum mode work_mode = ot > 1 ? Coverage : Back;

    // Divide into chunks along the z direction
    #pragma omp parallel for schedule(static)
    for(int i=0; i < oz; i++)
    {
      float chunk_zmin = ozmin + i * zsize / oz;
      float *chunk_map = coverage_map + i * ox * oy * ot;
      worker_function(
          NULL,
          chunk_zmin, oxmin, oymin,
          zsize / oz, xsize, ysize,
          1, ox, oy, ot,
          chunk_map,
          theta, h, v, line_weights,
          dsize,
          gridx, gridy,
          work_mode);
    }
    free(gridx);
    free(gridy);
}

void make_grid(
    const float zmin, const float xmin, const float ymin,
    const float zsize, const float xsize, const float ysize,
    const int nz, const int nx, const int ny,
    float * const gridz, float * const gridx, float * const gridy)
{
    int i;
    float xstep = xsize / nx;
    float ystep = ysize / ny;
    float zstep = zsize / nz;
    for(i=0; i<=nx; i++)
    {
        gridx[i] = xmin + i * xstep;
    }
    for(i=0; i<=ny; i++)
    {
        gridy[i] = ymin + i * ystep;
    }
    for(i=0; i<=nz; i++)
    {
        gridz[i] = zmin + i * zstep;
    }
}

void worker_function(
    float *obj_weights,
    const float ozmin, const float oxmin, const float oymin,
    const float zsize, const float xsize, const float ysize,
    const int oz, const int ox, const int oy, const int ot,
    float *data,
    const float *theta, const float *h, const float *v, const float *weights,
    const int dsize,
    const float *gridx, const float *gridy,
    const enum mode work_mode)
{
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
    float *dist = bx;
    // Midpoints between intersection points used to find indices
    float *midx = ax;
    float *midy = ay;
    // Index of the grid that the ray passes through.
    unsigned *indi = malloc(sizeof *indi * (ox+oy+1));
    assert(coordx != NULL && coordy != NULL &&
        ax != NULL && ay != NULL && by != NULL && bx != NULL &&
        coorx != NULL && coory != NULL &&
        dist != NULL && indi != NULL && midx != NULL && midy != NULL);

    int ray;
    int quadrant;
    float ri, hi;
    int zi;
    float theta_p, sin_p, cos_p;
    int asize, bsize, csize;
    const unsigned ozxy = (unsigned)oz*ox*oy;

    for (ray=0; ray<dsize; ray++)
    {
        zi = floor((v[ray]-ozmin) * oz / zsize);
        // Skip this ray if it is out of the z range
        //TODO: Presort lines by z coordinate so this check isn't necessary.
        if ((0 <= zi) && (zi < oz))
        {
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
                csize, midx, midy, zi, indi);
            switch (work_mode) {
                default:
                case Forward:
                    calc_forward(obj_weights, indi, dist, csize, &data[ray]);
                    break;
                case Back:
                    calc_back(dist, csize, weights[ray], data, indi);
                    break;
                case Coverage:
                    calc_coverage(dist, csize, weights[ray], theta_p, ot,
                                  data, indi);
                    break;
                case ART:
                    calc_art(obj_weights, indi, dist, csize, &data[ray]);
                    break;
                case SIRT:
                    calc_sirt(obj_weights, data, &data[ozxy],
                              indi, dist, csize, &weights[ray]);
                    break;
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
    free(indi);
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
    assert(UINT_MAX/ox/oy/oz/4 > 0 && "Array is too large to index.");
    // printf("SHAPE: %d, %d, %d : %f, %f, %f : %f, %f, %f\n",
    //         ox, oy, oz, oxmin, oymin, ozmin, xstep, ystep, zstep);
    int n;
    unsigned indx, indy;
    for (n=0; n<msize-1; n++)
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


void
bin_angle(float *bins, const float magnitude,
    const float theta, const int nbins)
{
    assert(bins != NULL);
    assert(nbins > 0);
    int bin = floor(fmod(theta, M_PI) / (M_PI / nbins));
    // Negative angles yield negative bins
    if (bin < 0) bin += nbins;
    assert(bin >= 0 && bin < nbins);
    bins[bin] += magnitude;
}


void
calc_coverage(
    const float *dist,
    int const dist_size,
    float const line_weight,
    float const theta,
    const int nbins,
    float *cov,
    const unsigned *ind_cov
    )
{
    int n;
    for (n=0; n<dist_size-1; n++)
    {
        bin_angle(&cov[nbins*ind_cov[n]], dist[n]*line_weight,
            theta, nbins);
    }
}


void
calc_back(
    const float *dist,
    int const dist_size,
    float const line_weight,
    float *cov,
    const unsigned *ind_cov)
{
    int n;
    for (n=0; n<dist_size-1; n++)
    {
      cov[ind_cov[n]] += dist[n]*line_weight;
    }
}


void
calc_forward(
    const float *grided_weights,
    const unsigned *ind_grid,
    const float *dist,
    int const dist_size,
    float *data)
{
    int n;
    for (n=0; n<dist_size-1; n++)
    {
        data[0] += grided_weights[ind_grid[n]]*dist[n];
    }
}

void
calc_art(
    float *grided_weights,
    const unsigned *ind_grid,
    const float *dist,
    int const dist_size,
    float *data)
{
    float update;
    float forward = 0;
    float dist_dot = 0;
    int n;
    for (n=0; n<dist_size-1; n++)
    {
        forward += grided_weights[ind_grid[n]]*dist[n];
        dist_dot += dist[n]*dist[n];
    }

    if (dist_dot <= 0) return;

    for (n=0; n < dist_size-1; n++)
    {
        update = (data[0] - forward) / dist_dot;
        grided_weights[ind_grid[n]] += dist[n] * update;
    }
}

void
calc_sirt(
    const float *grided_weights,
    float *update,
    float *sumdist,
    const unsigned *ind_grid,
    const float *dist,
    int const dist_size,
    float *data)
{
    float forward = 0;
    float dist_dot = 0;
    int n;
    for (n=0; n<dist_size-1; n++)
    {
        sumdist[ind_grid[n]] += dist[n];
        forward += grided_weights[ind_grid[n]]*dist[n];
        dist_dot += dist[n]*dist[n];
    }

    if (dist_dot <= 0) return;

    for (n=0; n < dist_size-1; n++)
    {
        update[ind_grid[n]] += dist[n] * (data[0] - forward) / dist_dot;
    }
}
