#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "utils.h"

void make_grid(
    const float xmin, const float xsize, const int nx, float * const gridx)
{
    assert(xsize > 0 && nx > 0 && "Grid must have non-zero dimensions.");
    float xstep = xsize / nx;
    assert(gridx != NULL);
    for(int i=0; i<=nx; i++)
    {
        gridx[i] = xmin + i * xstep;
    }
    assert(gridx[nx] == xmin + xsize);
}

void bin_angle(
    float *bins, const float magnitude,
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
calc_back(
    const float *dist,
    int const dist_size,
    float const line_weight,
    float *cov,
    const int *ind_cov)
{
    int n;
    for (n=0; n<dist_size-1; n++)
    {
      cov[ind_cov[n]] += dist[n]*line_weight;
    }
}
