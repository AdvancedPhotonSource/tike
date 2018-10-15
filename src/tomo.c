#include <stdlib.h>
#include <assert.h>
#include <limits.h>

#include "tomo.h"
#include "utils.h"
#include "siddon.h"

void
forward_project(
    const float *obj_weights,
    const float ozmin, const float oxmin, const float oymin,
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
    make_grid(oxmin, ox, ox, gridx);
    make_grid(oymin, oy, oy, gridy);
    // Allocate arrays for tracking intersections
    int *pixels, *rays;
    float *lengths;
    int psize;
    get_pixel_indexes_and_lengths(
        ozmin, oxmin, oymin,
        oz, ox, oy,
        oz, ox, oy,
        theta, h, v,
        dsize,
        gridx, gridy,
        &pixels, &rays, &lengths, &psize
    );
    free(gridx);
    free(gridy);
    // Bin the intersections by angle
    assert(UINT_MAX/ox/oy/oz > 0 && "Array is too large to index.");
    assert(dsize >= 0 && "Data size must be a natural number");
    assert(data != NULL && obj_weights != NULL);
    for (int n=0; n < psize; n++)
    {
        // printf("pixel %u touched ray %u for length %f\n",
        //        pixels[n], rays[n], lengths[n]);
        assert(pixels[n] < (int)oz*ox*oy);
        assert(rays[n] < dsize);
        data[rays[n]] += obj_weights[pixels[n]]*lengths[n];
    }
    free(pixels);
    free(rays);
    free(lengths);
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
    int *pixels, *rays;
    float *lengths;
    int psize;
    get_pixel_indexes_and_lengths(
        ozmin, oxmin, oymin,
        zsize, xsize, ysize,
        oz, ox, oy,
        theta, h, v,
        dsize,
        gridx, gridy,
        &pixels, &rays, &lengths, &psize
    );
    free(gridx);
    free(gridy);
    // Bin the intersections by angle
    assert(UINT_MAX/ox/oy/oz/ot > 0 && "Array is too large to index.");
    assert(theta != NULL && line_weights != NULL && coverage_map != NULL);
    assert(dsize >= 0);
    for (int n=0; n < psize; n++)
    {
        // printf("pixel %u touched ray %u for length %f\n",
        //        pixels[n], rays[n], lengths[n]);
        assert(pixels[n] < (int)oz*ox*oy);
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
}

void
art(
    const float zmin, const float xmin, const float ymin,
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
    make_grid(xmin, nx, nx, gridx);
    make_grid(ymin, ny, ny, gridy);
    // Allocate arrays for tracking intersections
    int *pixels, *rays;
    float *lengths;
    int psize;
    get_pixel_indexes_and_lengths(
        zmin, xmin, ymin,
        nz, nx, ny,
        nz, nx, ny,
        theta, h, v,
        ndata,
        gridx, gridy,
        &pixels, &rays, &lengths, &psize
    );
    free(gridx);
    free(gridy);

    assert(ndata >= 0);
    float *lengths_dot = calloc(ndata, sizeof *lengths_dot);
    assert(lengths_dot != NULL);
    for (int j=0; j < psize; j++)
    {
        lengths_dot[rays[j]] += lengths[j] * lengths[j];
    }

    assert(init != NULL && data != NULL);
    for (int i=0; i < niter; i++)
    {
        float *sim = calloc(ndata, sizeof *sim);
        float *line_update = calloc(ndata, sizeof *line_update );
        assert(sim != NULL && line_update != NULL);
        // simulate data acquisition by projecting over current model
        for (int j=0; j < psize; j++)
        {
            sim[rays[j]] += init[pixels[j]] * lengths[j];
        }
        // Compute an update value for each line
        for (int k=0; k < ndata; k++)
        {
            if (lengths_dot[k] > 0)
            {
                line_update[k] = (data[k] - sim[k]) / lengths_dot[k];
            }
        }
        // Project the line update back over the current model.
        for (int j=0; j < psize; j++)
        {
            init[pixels[j]] += lengths[j] * line_update[rays[j]];
        }
        free(sim);
        free(line_update);
    }
    free(lengths_dot);
    free(pixels);
    free(rays);
    free(lengths);
}

void
sirt(
    const float zmin, const float xmin, const float ymin,
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
    make_grid(xmin, nx, nx, gridx);
    make_grid(ymin, ny, ny, gridy);
    // Allocate arrays for tracking intersections
    int *pixels, *rays;
    float *lengths;
    int psize;
    get_pixel_indexes_and_lengths(
        zmin, xmin, ymin,
        nz, nx, ny,
        nz, nx, ny,
        theta, h, v,
        ndata,
        gridx, gridy,
        &pixels, &rays, &lengths, &psize
    );
    free(gridx);
    free(gridy);

    assert(UINT_MAX/nz/nx/ny > 0 && "Array is too large to index.");
    int grid_size = (int)nz*nx*ny;
    assert(ndata >= 0);
    float *lengths_dot = calloc(ndata, sizeof *lengths_dot);
    assert(lengths_dot != NULL);
    for (int j=0; j < psize; j++)
    {
        lengths_dot[rays[j]] += lengths[j] * lengths[j];
    }

    assert(init != NULL && data != NULL);
    for (int i=0; i < niter; i++)
    {
        float *grid_update = calloc(grid_size, sizeof *grid_update);
        int *num_grid_updates = calloc(grid_size, sizeof *num_grid_updates);
        float *sim = calloc(ndata, sizeof *sim);
        float *line_update = calloc(ndata, sizeof *line_update);
        assert(grid_update != NULL && num_grid_updates != NULL && sim != NULL
            && line_update != NULL);
        // simulate data acquisition by projecting over current model
        for (int j=0; j < psize; j++)
        {
            sim[rays[j]] += init[pixels[j]] * lengths[j];
        }
        // Compute an update value for each line
        for (int k=0; k < ndata; k++)
        {
            if (lengths_dot[k] > 0)
            {
                line_update[k] = (data[k] - sim[k]) / lengths_dot[k];
            }
        }
        // Project the line update back over the grid update.
        for (int j=0; j < psize; j++)
        {
            grid_update[pixels[j]] += lengths[j] * line_update[rays[j]];
            num_grid_updates[pixels[j]] += 1;

        }
        // Update the inital guess.
        for (int l=0; l < grid_size; l++)
        {
            if (num_grid_updates[l] > 0)
            {
                init[l] += grid_update[l] / num_grid_updates[l];
            }
        }
        free(grid_update);
        free(num_grid_updates);
        free(sim);
        free(line_update);
    }
    free(lengths_dot);
    free(pixels);
    free(rays);
    free(lengths);
}
