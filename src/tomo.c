#include "tomo.h"

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
    const float *obj,
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
    float *coordy = (float *)malloc((ox+1)*sizeof(float));
    float *coordx = (float *)malloc((oy+1)*sizeof(float));
    float *gridy = (float *)malloc((oy+1)*sizeof(float));
    // Initialize intermediate vectors
    // TODO: Reduce memory consumption by reusing some of these arrays
    float *ax = (float *)malloc((ox+oy+2)*sizeof(float));
    float *ay = (float *)malloc((ox+oy+2)*sizeof(float));
    float *bx = (float *)malloc((ox+oy+2)*sizeof(float));
    float *by = (float *)malloc((ox+oy+2)*sizeof(float));
    float *coorx = (float *)malloc((ox+oy+2)*sizeof(float));
    float *coory = (float *)malloc((ox+oy+2)*sizeof(float));
    // Initialize the distance vector per ray.
    float *dist = (float *)malloc((ox+oy+1)*sizeof(float));
    float *midx = (float *)malloc((ox+oy+1)*sizeof(float));
    float *midy = (float *)malloc((ox+oy+1)*sizeof(float));
    // Initialize the index of the grid that the ray passes through.
    unsigned *indi = (unsigned *)malloc((ox+oy+1)*sizeof(unsigned));
    // Diagnostics for pointers.
    assert(coordx != NULL && coordy != NULL &&
        ax != NULL && ay != NULL && by != NULL && bx != NULL &&
        coorx != NULL && coory != NULL &&
        dist != NULL && indi != NULL);
    int ray;
    int quadrant;
    float ri, hi;
    int zi;
    float theta_p, sin_p, cos_p;
    int asize, bsize, csize;
    preprocessing(oxmin, oymin, ox, oy, gridx, gridy); // Outputs: gridx, gridy
    for (ray=0; ray<dsize; ray++)
    {
        zi = floor(v[ray]-ozmin);
        if ((oz <= zi) || (zi < 0))
        {
          //TODO: Replace this hard exclusion with a weight over z gridlines
          // Skip this ray if it is out of the z range
          // printf("skipped %d. %f, %f, %f\n", ray, 0.0, zi, oz);
          continue;
        }
        // Calculate the sin and cos values
        // of the projection angle and find
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
        calc_dist(csize, coorx, coory, midx, midy, dist);
        calc_index(ox, oy, oz, oxmin, oymin, ozmin, csize,
                   midx, midy, zi, indi);
        // Calculate projection
        calc_simdata(obj, csize, indi, dist, ray, data);
    }
    free(gridx);
    free(gridy);
    free(coordx);
    free(coordy);
    free(ax);
    free(ay);
    free(bx);
    free(by);
    free(coorx);
    free(coory);
    free(dist);
    free(midx);
    free(midy);
    free(indi);
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
    const float *weights,
    int dsize,
    float *cov)
{
    // Initialize the grid on object space.
    float *gridx = malloc(sizeof *gridx * (ox+1));
    float *coordy = malloc(sizeof *coordy * (ox+1));
    float *coordx = malloc(sizeof *coordx * (oy+1));
    float *gridy = malloc(sizeof *gridy * (oy+1));
    // Initialize intermediate vectors
    // TODO: Reduce memory consumption by reusing some of these arrays
    float *ax = (float *)malloc((ox+oy+2)*sizeof(float));
    float *ay = (float *)malloc((ox+oy+2)*sizeof(float));
    float *bx = (float *)malloc((ox+oy+2)*sizeof(float));
    float *by = (float *)malloc((ox+oy+2)*sizeof(float));
    float *coorx = (float *)malloc((ox+oy+2)*sizeof(float));
    float *coory = (float *)malloc((ox+oy+2)*sizeof(float));
    // Initialize the distance vector per ray.
    float *dist = (float *)malloc((ox+oy+1)*sizeof(float));
    float *midx = (float *)malloc((ox+oy+1)*sizeof(float));
    float *midy = (float *)malloc((ox+oy+1)*sizeof(float));
    // Initialize the index of the grid that the ray passes through.
    unsigned *indi = malloc(sizeof *indi * (ox+oy+1));
    // Diagnostics for pointers.
    assert(coordx != NULL && coordy != NULL &&
        ax != NULL && ay != NULL && by != NULL && bx != NULL &&
        coorx != NULL && coory != NULL &&
        dist != NULL && indi != NULL);
    int ray;
    int quadrant;
    float ri, hi;
    int zi;
    float theta_p, sin_p, cos_p;
    int asize, bsize, csize;
    preprocessing(oxmin, oymin, ox, oy, gridx, gridy); // Outputs: gridx, gridy
    for (ray=0; ray<dsize; ray++)
    {
        zi = floor(v[ray]-ozmin);
        if ((oz <= zi) || (zi < 0))
        {
          //TODO: Replace this hard exclusion with a weight over z gridlines
          // Skip this ray if it is out of the z range
          // printf("skipped %d. %f, %f, %f\n", ray, 0.0, zi, oz);
          continue;
        }
        // Calculate the sin and cos values
        // of the projection angle and find
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
        calc_dist(csize, coorx, coory, midx, midy, dist);
        calc_index(ox, oy, oz, oxmin, oymin, ozmin, csize,
                   midx, midy, zi, indi);
        // Calculate coverage
        calc_coverage(csize, indi, dist, weights[ray], cov);
    }
    free(gridx);
    free(gridy);
    free(coordx);
    free(coordy);
    free(ax);
    free(ay);
    free(bx);
    free(by);
    free(coorx);
    free(coory);
    free(dist);
    free(midx);
    free(midy);
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
  // Convert the range [min_coord, max_coord) to [left, right) where left,
  // right are indices of grid
  // if (coord[grid_size-1] < min_coord || max_coord < coord[0])
  // {
  //   // All points are outside the range
  //   *a_size = 0;
  //   printf("Nothing!\n");
  //   return;
  // }
  // Find the left and right edge of the overlap of ranges
  assert(grid_size >= 2);
  bool ascending = true;
  // Determine whether coords are sorted ascending or descending
  if (coord[0] > coord[1])
    ascending = false;
  int left = 0;
  int right = grid_size;
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


void
trim_coords(
    int ox, int oy,
    const float *coordx, const float *coordy,
    const float *gridx, const float* gridy,
    int *asize, float *ax, float *ay,
    int *bsize, float *bx, float *by)
{
    trim_coord_1D(ox+1, gridx, coordy, asize,
                  ax, ay, gridy[0], gridy[oy]);
    trim_coord_1D(oy+1, gridy, coordx, bsize,
                  by, bx, gridx[0], gridx[ox]);
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
        // printf("%f, %f\n", coorx[n+1], coorx[n]);
        midx[n] = (coorx[n+1]+coorx[n])/2;
        midy[n] = (coory[n+1]+coory[n])/2;
        // printf("midx, midy = %f, %f\n", midx[n], midy[n]);
    }
}

void
calc_index(
    int const ox, int const oy, int const oz,
    int const oxmin, int const oymin, int const ozmin,
    int const msize, const float *midx, const float *midy,
    int const indz, unsigned *indi)
{
    int n;
    unsigned indx, indy;
    for (n=0; n<msize-1; n++)
    {
        // Midpoints assigned to pixels by nearest mincorner
        indx = floor(midx[n]-oxmin);
        // printf("imidx[n]-oxmin %f\n", midx[n]-oxmin );
        // printf("indx, ox, %u, %d\n", indx, ox);
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
    int const csize,
    const unsigned *indi,
    const float *dist,
    float const line_weight,
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
    const float *obj,
    int const csize,
    const unsigned *indi,
    const float *dist,
    int const ray,
    float *data)
{
    int n;
    for (n=0; n<csize-1; n++)
    {
        data[ray] += obj[indi[n]]*dist[n];
    }
}
