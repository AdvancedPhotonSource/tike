#include "tomo.h"
#include "siddon.h"
#include "limits.h"
#include <omp.h>

void
forward_project(
    const float *obj_weights,
    const float ozmin, const float oxmin, const float oymin,
    const float zsize, const float xsize, const float ysize,
    const int oz, const int ox, const int oy,
    const float *theta,
    const float *h,
    const float *v,
    const int dsize,
    float *data)
{
    // Initialize the grid on object space.
    assert(oz > 0 && ox > 0 && oy > 0);
    float *gridx = malloc(sizeof *gridx * (ox+1));
    float *gridy = malloc(sizeof *gridy * (oy+1));
    assert(gridx != NULL && gridy != NULL);
    make_grid(oxmin, xsize, ox, gridx);
    make_grid(oymin, ysize, oy, gridy);
    // Allocate arrays for tracking intersections
    unsigned *pixels, *rays;
    float *lengths;
    unsigned psize;
    get_pixel_indexes_and_lengths(
        ozmin, oxmin, oymin,
        zsize, xsize, ysize,
        oz, ox, oy,
        theta, h, v,
        dsize,
        gridx, gridy,
        &pixels, &rays, &lengths, &psize
    );
    assert(pixels != NULL && rays != NULL && lengths != NULL);
    // Bin the intersections by angle
    assert(UINT_MAX/ox/oy/oz > 0 && "Array is too large to index.");
    assert(dsize >= 0 && "Data size must be a natural number");
    assert(data != NULL && obj_weights != NULL);
    for (unsigned n=0; n < psize; n++)
    {
        // printf("pixel %u touched ray %u for length %f\n",
        //        pixels[n], rays[n], lengths[n]);
        assert(pixels[n] < (unsigned)oz*ox*oy);
        assert(rays[n] < dsize);
        data[rays[n]] += obj_weights[pixels[n]]*lengths[n];
    }
    free(pixels);
    free(rays);
    free(lengths);
    free(gridx);
    free(gridy);
}

void
coverage(
    const float ozmin, const float oxmin, const float oymin,
    const float zsize, const float xsize, const float ysize,
    const int oz, const int ox, const int oy, const int ot,
    const float * const theta,
    const float * const h,
    const float * const v,
    const float * const line_weights,
    const int dsize,
    float *coverage_map)
{
    // Initialize the grid on object space.
    assert(oz > 0 && ox > 0 && oy > 0 && ot > 0);
    float *gridx = malloc(sizeof *gridx * (ox+1));
    float *gridy = malloc(sizeof *gridy * (oy+1));
    assert(gridx != NULL && gridy != NULL);
    make_grid(oxmin, xsize, ox, gridx);
    make_grid(oymin, ysize, oy, gridy);
    // Allocate arrays for tracking intersections
    unsigned *pixels, *rays;
    float *lengths;
    unsigned psize;
    get_pixel_indexes_and_lengths(
        ozmin, oxmin, oymin,
        zsize, xsize, ysize,
        oz, ox, oy,
        theta, h, v,
        dsize,
        gridx, gridy,
        &pixels, &rays, &lengths, &psize
    );
    assert(pixels != NULL && rays != NULL && lengths != NULL);
    // Bin the intersections by angle
    assert(UINT_MAX/ox/oy/oz/ot > 0 && "Array is too large to index.");
    assert(theta != NULL && line_weights != NULL && coverage_map != NULL);
    assert(dsize >= 0);
    for (unsigned n=0; n < psize; n++)
    {
        // printf("pixel %u touched ray %u for length %f\n",
        //        pixels[n], rays[n], lengths[n]);
        assert(pixels[n] < (unsigned)oz*ox*oy);
        assert(rays[n] < dsize);
        bin_angle(
            &coverage_map[pixels[n] * ot],
            lengths[n]*line_weights[rays[n]],
            theta[rays[n]],
            ot);
    }
    free(pixels);
    free(rays);
    free(lengths);
    free(gridx);
    free(gridy);
}

void
art(
    const float zmin, const float xmin, const float ymin,
    const float zsize, const float xsize, const float ysize,
    const int nz, const int nx, const int ny,
    const float * const data,
    const float * const theta, const float * const h, const float * const v,
    const int ndata,
    float * const init,
    const int niter)
{
    // Initialize the grid on object space.
    assert(nz > 0 && nx > 0 && ny > 0);
    float *gridx = malloc(sizeof *gridx * (nx+1));
    float *gridy = malloc(sizeof *gridy * (ny+1));
    assert(gridx != NULL && gridy != NULL);
    make_grid(xmin, xsize, nx, gridx);
    make_grid(ymin, ysize, ny, gridy);
    // Allocate arrays for tracking intersections
    unsigned *pixels, *rays;
    float *lengths;
    unsigned psize;
    get_pixel_indexes_and_lengths(
        zmin, xmin, ymin,
        zsize, xsize, ysize,
        nz, nx, ny,
        theta, h, v,
        ndata,
        gridx, gridy,
        &pixels, &rays, &lengths, &psize
    );
    assert(pixels != NULL && rays != NULL && lengths != NULL);

    assert(ndata >= 0);
    float *sim = malloc(sizeof sim * ndata);
    float *line_update = malloc(sizeof line_update * ndata);
    float *lengths_dot = malloc(sizeof lengths_dot * ndata);
    assert(sim != NULL && line_update != NULL && lengths_dot != NULL);

    memset(lengths_dot, 0, sizeof lengths_dot * ndata);
    for (unsigned j=0; j < psize; j++)
    {
        lengths_dot[rays[j]] += lengths[j] * lengths[j];
    }

    assert(init != NULL && data != NULL);
    for (int i=0; i < niter; i++)
    {
        memset(sim, 0, sizeof sim * ndata);
        memset(line_update, 0, sizeof line_update * ndata);
        // simulate data acquisition by projecting over current model
        for (unsigned j=0; j < psize; j++)
        {
            sim[rays[j]] += init[pixels[j]] * lengths[j];
        }
        // Compute an update value for each line
        for (unsigned k=0; k < ndata; k++)
        {
            if (lengths_dot[k] > 0)
            {
                line_update[k] = (data[k] - sim[k]) / lengths_dot[k];
            }
        }
        // Project the line update back over the current model.
        for (unsigned j=0; j < psize; j++)
        {
            init[pixels[j]] += lengths[j] * line_update[rays[j]];
        }
    }
    free(sim);
    free(line_update);
    free(lengths_dot);
    free(pixels);
    free(rays);
    free(lengths);
    free(gridx);
    free(gridy);
}

void
sirt(
    const float zmin, const float xmin, const float ymin,
    const float zsize, const float xsize, const float ysize,
    const int nz, const int nx, const int ny,
    const float * const data,
    const float * const theta, const float * const h, const float * const v,
    const int ndata,
    float * const init,
    const int niter)
{
    // Initialize the grid on object space.
    assert(nz > 0 && nx > 0 && ny > 0);
    float *gridx = malloc(sizeof *gridx * (nx+1));
    float *gridy = malloc(sizeof *gridy * (ny+1));
    assert(gridx != NULL && gridy != NULL);
    make_grid(xmin, xsize, nx, gridx);
    make_grid(ymin, ysize, ny, gridy);
    // Allocate arrays for tracking intersections
    unsigned *pixels, *rays;
    float *lengths;
    unsigned psize;
    get_pixel_indexes_and_lengths(
        zmin, xmin, ymin,
        zsize, xsize, ysize,
        nz, nx, ny,
        theta, h, v,
        ndata,
        gridx, gridy,
        &pixels, &rays, &lengths, &psize
    );
    assert(pixels != NULL && rays != NULL && lengths != NULL);

    assert(UINT_MAX/nz/nx/ny > 0 && "Array is too large to index.");
    unsigned grid_size = (unsigned)nz*nx*ny;
    float *grid_update = malloc(sizeof grid_update * grid_size);
    int *num_grid_updates = malloc(sizeof num_grid_updates * grid_size);
    assert(ndata >= 0);
    float *sim = malloc(sizeof sim * ndata);
    float *line_update = malloc(sizeof line_update * ndata);
    float *lengths_dot = malloc(sizeof lengths_dot * ndata);
    assert(grid_update != NULL && num_grid_updates != NULL && sim != NULL
           && line_update != NULL && lengths_dot != NULL);

    memset(lengths_dot, 0, sizeof lengths_dot * ndata);
    for (unsigned j=0; j < psize; j++)
    {
        lengths_dot[rays[j]] += lengths[j] * lengths[j];
    }

    assert(init != NULL && data != NULL);
    for (int i=0; i < niter; i++)
    {
        memset(grid_update, 0, sizeof grid_update * grid_size);
        memset(num_grid_updates, 0, sizeof num_grid_updates * grid_size);
        memset(sim, 0, sizeof sim * ndata);
        memset(line_update, 0, sizeof line_update * ndata);
        // simulate data acquisition by projecting over current model
        for (unsigned j=0; j < psize; j++)
        {
            sim[rays[j]] += init[pixels[j]] * lengths[j];
        }
        // Compute an update value for each line
        for (unsigned k=0; k < ndata; k++)
        {
            if (lengths_dot[k] > 0)
            {
                line_update[k] = (data[k] - sim[k]) / lengths_dot[k];
            }
        }
        // Project the line update back over the grid update.
        for (unsigned j=0; j < psize; j++)
        {
            grid_update[pixels[j]] += lengths[j] * line_update[rays[j]];
            num_grid_updates[pixels[j]] += 1;

        }
        // Update the inital guess.
        for (unsigned l=0; l < grid_size; l++)
        {
            if (num_grid_updates[l] > 0)
            {
                init[l] += grid_update[l] / num_grid_updates[l];
            }
        }
    }
    free(grid_update);
    free(num_grid_updates);
    free(sim);
    free(line_update);
    free(lengths_dot);
    free(pixels);
    free(rays);
    free(lengths);
    free(gridx);
    free(gridy);
}

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
    const unsigned *ind_cov)
{
    int n;
    for (n=0; n<dist_size-1; n++)
    {
      cov[ind_cov[n]] += dist[n]*line_weight;
    }
}
