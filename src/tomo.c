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
    int ox,
    int oy,
    int oz,
    const float *x,
    const float *y,
    const float *theta,
    int dsize,
    float *data)
{
    printf("Hey!\n");

    // Initialize the grid on object space.
    float *gridx = (float *)malloc((ox+1)*sizeof(float));
    float *gridy = (float *)malloc((oz+1)*sizeof(float));

    // Initialize intermediate vectors.
    float *coordx = (float *)malloc((oz+1)*sizeof(float));
    float *coordy = (float *)malloc((ox+1)*sizeof(float));
    float *ax = (float *)malloc((ox+oz+2)*sizeof(float));
    float *ay = (float *)malloc((ox+oz+2)*sizeof(float));
    float *bx = (float *)malloc((ox+oz+2)*sizeof(float));
    float *by = (float *)malloc((ox+oz+2)*sizeof(float));
    float *coorx = (float *)malloc((ox+oz+2)*sizeof(float));
    float *coory = (float *)malloc((ox+oz+2)*sizeof(float));

    // Initialize the distance vector per ray.
    float *dist = (float *)malloc((ox+oz+1)*sizeof(float));

    // Initialize the index of the grid that the ray passes through.
    int *indi = (int *)malloc((ox+oz+1)*sizeof(int));

    // Diagnostics for pointers.
    assert(coordx != NULL && coordy != NULL &&
        ax != NULL && ay != NULL && by != NULL && bx != NULL &&
        coorx != NULL && coory != NULL &&
        dist != NULL && indi != NULL);

    int ray;
    int quadrant;
    int xi, yi, zi;
    float theta_p, sin_p, cos_p;
    int asize, bsize, csize;

    preprocessing(ox, oz,
        gridx, gridy); // Outputs: gridx, gridy

    // For each data point
    for (ray=0; ray<dsize; ray++)
    {
        // Calculate the sin and cos values
        // of the projection angle and find
        // at which quadrant on the cartesian grid.
        theta_p = fmod(theta[ray], 2*M_PI);
        quadrant = calc_quadrant(theta_p);
        sin_p = sinf(theta_p);
        cos_p = cosf(theta_p);

        xi = -ox-oz;
        yi = x[ray]+1e-6;
        zi = y[ray];

        calc_coords(
            ox, oz, xi, yi, sin_p, cos_p, gridx, gridy,
            coordx, coordy);

        // Merge the (coordx, gridy) and (gridx, coordy)
        trim_coords(
            ox, oz, coordx, coordy, gridx, gridy,
            &asize, ax, ay, &bsize, bx, by);

        // Sort the array of intersection points (ax, ay) and
        // (bx, by). The new sorted intersection points are
        // stored in (coorx, coory). Total number of points
        // are csize.
        sort_intersections(
            quadrant, asize, ax, ay, bsize, bx, by,
            &csize, coorx, coory);

        // Calculate the distances (dist) between the
        // intersection points (coorx, coory). Find the
        // indices of the pixels on the object grid.
        calc_dist(
            ox, oz, csize, coorx, coory,
            indi, dist);

        // Calculate simdata
        calc_simdata(obj, ox, oz,
            csize, zi, indi, dist, ray,
            data); // Output: simulated data
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
    free(indi);
}


void
coverage(
    int ox,
    int oy,
    int oz,
    const float *x,
    const float *y,
    const float *theta,
    int dsize,
    float *cov)
{
    printf("Hey!\n");

    // Initialize the grid on object space.
    float *gridx = (float *)malloc((ox+1)*sizeof(float));
    float *gridy = (float *)malloc((oz+1)*sizeof(float));

    // Initialize intermediate vectors.
    float *coordx = (float *)malloc((oz+1)*sizeof(float));
    float *coordy = (float *)malloc((ox+1)*sizeof(float));
    float *ax = (float *)malloc((ox+oz+2)*sizeof(float));
    float *ay = (float *)malloc((ox+oz+2)*sizeof(float));
    float *bx = (float *)malloc((ox+oz+2)*sizeof(float));
    float *by = (float *)malloc((ox+oz+2)*sizeof(float));
    float *coorx = (float *)malloc((ox+oz+2)*sizeof(float));
    float *coory = (float *)malloc((ox+oz+2)*sizeof(float));

    // Initialize the distance vector per ray.
    float *dist = (float *)malloc((ox+oz+1)*sizeof(float));

    // Initialize the index of the grid that the ray passes through.
    int *indi = (int *)malloc((ox+oz+1)*sizeof(int));

    // Diagnostics for pointers.
    assert(coordx != NULL && coordy != NULL &&
        ax != NULL && ay != NULL && by != NULL && bx != NULL &&
        coorx != NULL && coory != NULL &&
        dist != NULL && indi != NULL);

    int ray;
    int quadrant;
    int xi, yi, zi;
    float theta_p, sin_p, cos_p;
    int asize, bsize, csize;

    preprocessing(ox, oz,
        gridx, gridy); // Outputs: gridx, gridy

    // For each data point
    for (ray=0; ray<dsize; ray++)
    {
        // Calculate the sin and cos values
        // of the projection angle and find
        // at which quadrant on the cartesian grid.
        theta_p = fmod(theta[ray], 2*M_PI);
        quadrant = calc_quadrant(theta_p);
        sin_p = sinf(theta_p);
        cos_p = cosf(theta_p);

        xi = -ox-oz;
        yi = x[ray]+1e-6;
        zi = y[ray];
        //printf("ray=%d, xi=%d, yi=%d, zi=%d\n", ray, xi, yi, zi);

        calc_coords(
            ox, oz, xi, yi, sin_p, cos_p, gridx, gridy,
            coordx, coordy);

        // Merge the (coordx, gridy) and (gridx, coordy)
        trim_coords(
            ox, oz, coordx, coordy, gridx, gridy,
            &asize, ax, ay, &bsize, bx, by);

        // Sort the array of intersection points (ax, ay) and
        // (bx, by). The new sorted intersection points are
        // stored in (coorx, coory). Total number of points
        // are csize.
        sort_intersections(
            quadrant, asize, ax, ay, bsize, bx, by,
            &csize, coorx, coory);

        // Calculate the distances (dist) between the
        // intersection points (coorx, coory). Find the
        // indices of the pixels on the object grid.
        calc_dist(
            ox, oz, csize, coorx, coory,
            indi, dist);

        // Calculate simdata
        calc_coverage(ox, oz,
            csize, zi, indi, dist,
            cov); // Output: simulated coverage
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
    free(indi);
}


void
preprocessing(
    int ngridx,
    int ngridy,
    float *gridx,
    float *gridy)
{
    int i;

    for(i=0; i<=ngridx; i++)
    {
        gridx[i] = -ngridx/2.+i;
    }

    for(i=0; i<=ngridy; i++)
    {
        gridy[i] = -ngridy/2.+i;
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
    int ry, int rz,
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
    for (n=0; n<=ry; n++)
    {
        coordy[n] = slope*(gridx[n]-srcx)+srcy;
    }
    for (n=0; n<=rz; n++)
    {
        coordx[n] = islope*(gridy[n]-srcy)+srcx;
    }
}


void
trim_coords(
    int ry, int rz,
    const float *coordx, const float *coordy,
    const float *gridx, const float* gridy,
    int *asize, float *ax, float *ay,
    int *bsize, float *bx, float *by)
{
    int n;

    *asize = 0;
    *bsize = 0;
    for (n=0; n<=rz; n++)
    {
        if (coordx[n] >= gridx[0]+1e-2)
        {
            if (coordx[n] <= gridx[ry]-1e-2)
            {
                ax[*asize] = coordx[n];
                ay[*asize] = gridy[n];
                (*asize)++;
            }
        }
    }
    for (n=0; n<=ry; n++)
    {
        if (coordy[n] >= gridy[0]+1e-2)
        {
            if (coordy[n] <= gridy[rz]-1e-2)
            {
                bx[*bsize] = gridx[n];
                by[*bsize] = coordy[n];
                (*bsize)++;
            }
        }
    }
}


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
    int ry, int rz,
    int csize, const float *coorx, const float *coory,
    int *indi, float *dist)
{
    int n, i1, i2;
    float x1, x2;
    float diffx, diffy, midx, midy;
    int indx, indy;

    for (n=0; n<csize-1; n++)
    {
        diffx = coorx[n+1]-coorx[n];
        diffy = coory[n+1]-coory[n];
        dist[n] = sqrt(diffx*diffx+diffy*diffy);
        midx = (coorx[n+1]+coorx[n])/2;
        midy = (coory[n+1]+coory[n])/2;
        x1 = midx+ry/2.;
        x2 = midy+rz/2.;
        i1 = (int)(midx+ry/2.);
        i2 = (int)(midy+rz/2.);
        indx = i1-(i1>x1);
        indy = i2-(i2>x2);
        indi[n] = indy+(indx*rz);
    }
}


void
calc_coverage(
    int ry,
    int rz,
    int csize,
    float slice,
    const int *indi,
    const float *dist,
    float *cov)
{
    int n;
    int iobj = floor(slice)*ry*rz;
    for (n=0; n<csize-1; n++)
    {
        cov[indi[n]+iobj] += dist[n];
    }
}


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
    float *data)
{
    int n;
    int iobj = floor(slice)*ry*rz;
    for (n=0; n<csize-1; n++)
    {
        data[ray] += obj[indi[n]+iobj]*dist[n];
    }
}
