#include "tomo.h"
#include "limits.h"
#include <omp.h>

void
art(
    const float *data,
    const float *x,
    const float *y,
    const float *theta,
    float *recon)
{
    printf("Hey!\n");
}


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
    float *data)
{
    // Initialize the grid on object space.
    float *gridx = (float *)malloc((ox+1)*sizeof(float));
    float *gridy = (float *)malloc((oy+1)*sizeof(float));
    assert(gridx != NULL && gridy != NULL);

    preprocessing(oxmin, oymin, ox, oy, gridx, gridy); // Outputs: gridx, gridy

    worker_function(obj_weights, ozmin, oxmin, oymin, ox, oy, oz,
        data, theta, h, v, NULL, dsize,
        gridx, gridy, calc_simdata);

    free(gridx);
    free(gridy);
}

/**
Siddon, R. L. (1984). Fast calculation of the exact radiological path for a
three‐dimensional CT array. Medical Physics, 12(2), 252–255.
https://doi.org/10.1118/1.595715
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
    const float *line_weights,
    int dsize,
    float *coverage_map)
{
    assert(UINT_MAX/ox/oy/oz > 0 && "Array is too large to index.");

    // Initialize the grid on object space.
    float *gridx = malloc(sizeof *gridx * (ox+1));
    float *gridy = malloc(sizeof *gridy * (oy+1));
    assert(gridx != NULL && gridy != NULL);
    preprocessing(oxmin, oymin, ox, oy, gridx, gridy); // Outputs: gridx, gridy

    // Divide into chunks along the z direction
    unsigned chunk_size_ind = ox*oy;
    int chunk_size_oz = 1;

    #pragma omp parallel for schedule(static)
    for(int i=0; i < oz; i++)
    {
      float chunk_zmin = ozmin + i * chunk_size_oz;
      float *chunk_map = coverage_map + i * chunk_size_ind;
      worker_function(NULL, chunk_zmin, oxmin, oymin, ox, oy, chunk_size_oz,
          chunk_map, theta, h, v, line_weights, dsize,
          gridx, gridy, calc_coverage);
    }

    free(gridx);
    free(gridy);
}


void worker_function(
    const float *obj_weights,
    const float ozmin, const float oxmin, const float oymin,
    const int ox, const int oy, const int oz,
    float *data,
    const float *theta, const float *h, const float *v, const float *weights,
    const int dsize,
    const float *gridx, const float *gridy,
    void (*calc_f)(
        const unsigned *, const float *,
        const float *,
        int const,
        float const,
        int const,
        float *))
{
    assert(calc_f != NULL);
    float *coordy = malloc(sizeof *coordy * (ox+oy+2));
    float *coordx = malloc(sizeof *coordx * (ox+oy+2));
    // Initialize intermediate vectors
    float *ax = (float *)malloc((ox+oy+2)*sizeof(float));
    float *ay = (float *)malloc((ox+oy+2)*sizeof(float));
    float *bx = (float *)malloc((ox+oy+2)*sizeof(float));
    float *by = (float *)malloc((ox+oy+2)*sizeof(float));
    float *coorx = coordx;
    float *coory = coordy;
    // Initialize the distance vector per ray.
    float *dist = bx;
    float *midx = ax;
    float *midy = ay;
    // Initialize the index of the grid that the ray passes through.
    unsigned *indi = malloc(sizeof *indi * (ox+oy+1));
    // Diagnostics for pointers.
    assert(coordx != NULL && coordy != NULL &&
        ax != NULL && ay != NULL && by != NULL && bx != NULL &&
        coorx != NULL && coory != NULL &&
        dist != NULL && indi != NULL && midx != NULL && midy != NULL);

    int ray;
    int quadrant;
    float ri, hi, line_weight;
    int zi;
    float theta_p, sin_p, cos_p;
    int asize, bsize, csize;

    for (ray=0; ray<dsize; ray++)
    {
      zi = floor(v[ray]-ozmin);
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
        // printf("ray=%d, ri=%f, hi=%f, zi=%f\n", ray, ri, hi, zi);
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
            ox, oy, oz, oxmin, oymin, ozmin, csize,
            midx, midy, zi, indi);
        // Calculate coverage
        if (weights != NULL)
            line_weight = weights[ray];
        (*calc_f)(indi, dist, obj_weights, csize, line_weight, ray, data);
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



void
preprocessing(
    float minx,
    float miny,
    int ngridx,
    int ngridy,
    float *gridx,
    float *gridy)
{
    int i;

    for(i=0; i<=ngridx; i++)
    {
        gridx[i] = minx+i;
    }

    for(i=0; i<=ngridy; i++)
    {
        gridy[i] = miny+i;
    }
}


int
calc_quadrant(
    float theta_p)
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
    int ngridx, int ngridy,
    float xi, float yi,
    float sin_p, float cos_p,
    const float *gridx, const float *gridy,
    float *coordx, float *coordy)
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


/** Given the ordered points (grid, coord). Remove points outside the range
  [min_coord, max_coord). Put the points you're keeping inside (agrid, acoord).
*/
void trim_coord_1D(
    const int grid_size, const float *grid, const float *coord,
    int *a_size, float *agrid, float *acoord,
    const float min_coord, const float max_coord)
{
  assert(grid_size >= 2);
  assert(min_coord <= max_coord);
  if isfinite(coord[0])
  {
    // Determine whether coords are sorted ascending or descending
    bool ascending = coord[0] < coord[1];
    // if (coord[grid_size-1] < min_coord || max_coord < coord[0])
    // Convert the range [min_coord, max_coord) to [left, right) where left,
    // right are indices of grid
    int left = 0, right = grid_size;
    int low, high, mid;
    low = left;
    high = right;
    while (low <= high) {
        mid = (low + high) / 2;
        if (mid == grid_size || (ascending && min_coord <= coord[mid])
            || (!ascending && max_coord > coord[mid]))
        {
          if (mid == 0 || (ascending && coord[mid - 1] < min_coord)
              || (!ascending && coord[mid - 1] >= max_coord))
          {
            left = mid;
            break;
          }
          else // edge is to the left of 'mid'
            high = mid - 1;
        }
        else // edge is to the right of 'mid'
            low = mid + 1;
    }
    assert(low <= high);
    low = left;
    high = right;
    while (low <= high) {
        mid = (low + high) / 2;
        if (mid == grid_size || (ascending && max_coord <= coord[mid])
            || (!ascending && min_coord > coord[mid]))
        {
          if (mid == 0 || (ascending && coord[mid - 1] < max_coord)
              || (!ascending && coord[mid - 1] >= min_coord))
          {
            right = mid;
            break;
          }
          else // edge is to the left of 'mid'
            high = mid - 1;
        }
        else // edge is to the right of 'mid'
            low = mid + 1;
    }
    assert(low <= high);

    // Copy the range [left, right)
    *a_size = right-left;
    assert(*a_size >= 0);
    assert(left >= 0 && left + *a_size <= grid_size);
    memcpy(agrid, &grid[left], sizeof *grid * *a_size);
    memcpy(acoord, &coord[left], sizeof *coord * *a_size);
  }
  else
  {
    // All coords are infinite because the line is horizontal or vertical.
    // Trim all because grid intersections are all included in other list
    *a_size = 0;
    return;
  }

}


void
trim_coords(
    int ox, int oy,
    const float *coordx, const float *coordy,
    const float *gridx, const float* gridy,
    int *asize, float *ax, float *ay,
    int *bsize, float *bx, float *by)
{
    // trim_coord_1D(ox+1, gridx, coordy, asize,
    //               ax, ay, gridy[0], gridy[oy]);
    // trim_coord_1D(oy+1, gridy, coordx, bsize,
    //               by, bx, gridx[0], gridx[ox]);
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
    int ind_condition,
    int asize, const float *ax, const float *ay,
    int bsize, const float *bx, const float *by,
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
        midx[n] = (coorx[n+1]+coorx[n])/2.001;
        midy[n] = (coory[n+1]+coory[n])/2.001;
        // Divide by 2 + eps in order to keep midxy inside the grid when both
        // coords are inside the grid. Numerical instability sometimes pushes
        // the sum of midxy and oxmin outside the grid
    }
}

void
calc_index(
    int const ox, int const oy, int const oz,
    float const oxmin, float const oymin, float const ozmin,
    int const msize, const float *midx, const float *midy,
    int const indz, unsigned *indi)
{
    assert(UINT_MAX/ox/oy/oz > 0 && "Array is too large to index.");
    // printf("SHAPE: %d, %d, %d : %f, %f, %f\n",
    //         ox, oy, oz, oxmin, oymin, ozmin);
    int n;
    unsigned indx, indy;
    for (n=0; n<msize-1; n++)
    {
        // Midpoints assigned to pixels by nearest mincorner
        indx = floor(midx[n]-oxmin);
        assert(indx < (unsigned)ox);
        indy = floor(midy[n]-oymin);
        assert(indy < (unsigned)oy);
        assert(((indx*oy*oz)+(indy*oz)+indz) < (unsigned)ox*oy*oz);
        // Convert from 3D to linear C-order indexing
        indi[n] = indy + oy * (indx + ox * (indz));
    }
}

void
calc_coverage(
    const unsigned *indi,
    const float *dist,
    const float *unused_float,
    int const csize,
    float const line_weight,
    int const unused_int,
    float *cov)
{
    int n;
    for (n=0; n<csize-1; n++)
    {
        cov[indi[n]] += dist[n]*line_weight;
    }
}


void
calc_simdata(
    const unsigned *indi,
    const float *dist,
    const float *obj,
    int const csize,
    float const unused_float,
    int const ray,
    float *data)
{
    int n;
    for (n=0; n<csize-1; n++)
    {
        data[ray] += obj[indi[n]]*dist[n];
    }
}
