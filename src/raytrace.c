
#include "raytrace.h"

int *numpix;
int *pixstart;
int *indices;
float *lengths;
int *numrays;
int *raystart;
int *rays;
float *clens;

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
    get_intersections_and_lengths(
        dsize, theta, h,
        ox, oxmin, xsize,
        oy, oymin, ysize);

    back_project(ox*oy, numrays, raystart, rays, lengths,
       coverage_map, line_weights);
}

void get_intersections_and_lengths(const int numprob, const float *theta, const float *rho,
    const int nx, const float oxmin, const float xsize,
    const int ny, const float oymin, const float ysize){
  //PREPROCESSING FIRST PASS
  int *numpix = malloc(sizeof *numpix * numprob);
  int *pixstart = malloc(sizeof *pixstart * numprob);

  float domain[4] = {oxmin, oxmin+xsize, oymin, oymin+ysize};
  float res = xsize / nx;

  #pragma omp parallel for
  for(int k = 0; k < numprob; k++){ //FOR EACH RAY
      numpix[k] = 0;
      findnumpix(theta[k], rho[k], &numpix[k], &domain[0], res, nx, ny);
  }
  pixstart[0] = 0;
  for(int k = 1; k < numprob; k++)pixstart[k] = pixstart[k-1]+numpix[k-1];
  int totpix = pixstart[numprob-1]+numpix[numprob-1];
  printf("totpx %d\n",totpix);
  //PREPROCESSING SECOND PASS
  int *indices = malloc(sizeof *indices * totpix);
  float *lengths = malloc(sizeof *lengths * totpix);
  #pragma omp parallel for
  for(int k = 0; k < numprob; k++){ //FOR EACH RAY
      numpix[k] = 0;
      placepixels(theta[k], rho[k], &indices[pixstart[k]],
          &lengths[pixstart[k]],
          &numpix[k], &domain[0], res, nx, ny);
  }
  //TRANSPOSE DATA STRUCTURES FIRST PASS
  unsigned numunk = nx*ny;
  int *numrays = malloc(sizeof *numrays * numunk);
  int *raystart = malloc(sizeof *raystart * numunk);
  #pragma omp parallel for
  for(int n = 0; n < numunk; n++)numrays[n] = 0;
  for(int k = 0; k < numprob; k++)
    for(int m = 0; m < numpix[k]; m++)
      numrays[indices[pixstart[k]+m]]++;
  raystart[0] = 0;
  for(int n = 1; n < numunk; n++)raystart[n] = raystart[n-1]+numrays[n-1];
  int totrays = raystart[numunk-1]+numrays[numunk-1];
  //TRANSPOSE DATA STRUCTURES SECOND PASS
  int *rays = malloc(sizeof *rays * totrays);
  float *clens = malloc(sizeof *clens * totrays);
  #pragma omp parallel for
  for(int n = 0; n < numunk; n++)numrays[n] = 0;
  for(int k = 0; k < numprob; k++)
    for(int m = 0; m < numpix[k]; m++){
      int n = indices[pixstart[k]+m];
      rays[raystart[n]+numrays[n]] = k;
      clens[raystart[n]+numrays[n]] = lengths[pixstart[k]+m];
      numrays[n]++;
    }
}

void forward_project(
    const int numprob,
    const int * const numpix, const int * const pixstart,
    const int * const indices, const float * const lengths,
    const float * const object, float * const sinogram){
  #pragma omp parallel for
  for(int k = 0; k < numprob; k++){
    int m = pixstart[k];
    for(int t = 0; t < numpix[k]; t++)
      sinogram[k] = sinogram[k] + object[indices[m+t]]*lengths[m+t];
  }
}

void back_project(
    const int numunk,
    const int * const numrays, const int * const raystart,
    const int * const rays, const float * const clens,
    float * const object, const float * const sinogram){
  #pragma omp parallel for
  for(int n = 0; n < numunk; n++){
    int k = raystart[n];
    for(int t = 0; t < numrays[n]; t++)
      object[n] = object[n] + sinogram[rays[k+t]]*clens[k+t];
  }
}

//domain is the [left, right, bottom, top] boundaries of the region
//res is the width of a pixel assumed to be cubic
void findnumpix(float theta, float rho, int *numpix, float *domain, float res,
    int nx, int ny){

  // int numproc;
  // int myid;
  // MPI_Comm_size(MPI_COMM_WORLD,&numproc);
  // MPI_Comm_rank(MPI_COMM_WORLD,&myid);
  float raylength = 1e6;

  //RAY'S VECTOR REPRESENTAION
  float x = rho*cosf(theta)+0.5*raylength*sinf(theta);
  float y = rho*sinf(theta)-0.5*raylength*cosf(theta);
  float dx = -raylength*sinf(theta);
  float dy = +raylength*cosf(theta);

  //TOP LEVEL
  // Check if this ray passes on my rank's output domain region
  float p[4] = {-dx,dx,-dy,dy};
  float q[4] = {x-domain[0],domain[1]-x,y-domain[2],domain[3]-y};
  float u1 = -1*INFINITY;
  float u2 = INFINITY;
  bool pass = true;
  int inid = 0;
  for(int k = 0; k < 4; k++)
    if(p[k] == 0){
      if(q[k] < 0){
        pass = false;
        break;
      }
    }else{
      float t = q[k]/p[k];
      if(p[k] < 0 && u1 < t){
        u1 = t;
        inid = k;
      }
      else if(p[k] > 0 && u2 > t)
        u2 = t;
    }
  if(u1 > u2 || u1 > 1 || u1 < 0) pass = false;
  //IF RAY COLLIDES WITH DOMAIN
  if(pass){
    //FIND THE INITIAL PIXEL
    int init = 0;
    int initx = 0;
    int inity = 0;
    if(inid == 0){ /// left
      initx = 0;
      inity = (int)((y+u1*dy-domain[2])/res);
    }
    if(inid == 1){ /// right
      initx = nx-1;
      inity = (int)((y+u1*dy-domain[2])/res);
    }
    if(inid == 2){ /// bottom
      initx = (int)((x+u1*dx-domain[0])/res);
      inity = 0;
    }
    if(inid == 3){ /// top
      initx = (int)((x+u1*dx-domain[0])/res);
      inity = ny - 1;
    }
    float px = domain[0] + initx*res+res/2;
    float py = domain[2] + inity*res+res/2;
    //TRACE RAY WHILE IT IS IN THE DOMAIN
    while(px > domain[0] && px < domain[1] && py < domain[3] && py > domain[2]){
      int exid = 0;
      q[0] = x-(px-res/2);
      q[1] = (px+res/2)-x;
      q[2] = y-(py-res/2);
      q[3] = (py+res/2)-y;
      u1 = -1*INFINITY;
      u2 = INFINITY;
      for(int k = 0; k < 4; k++){
        float t = q[k]/p[k];
        if(p[k] < 0 && u1 < t)
          u1 = t;
        else if(p[k] > 0 && u2 > t){
          u2 = t;
          exid = k;
        }
      } /// u2-u1 = ray length in first/current pixel (px,py)
        /// exid shows which side this ray exits
      //ADD CONTRIBUTION FROM CURRENT PIXEL
      //int z = unkmap[inity*len+initx];
      //int z = encode(initx,inity);
      //int z = inity*nx+initx;
      *numpix = *numpix + 1;
      //FIND NEXT PIXEL
      if(exid == 0){
        initx = initx-1;
        px = px - res;
      }
      if(exid == 1){
        initx = initx+1;
        px = px + res;
      }
      if(exid == 2){
        inity = inity-1;
        py = py - res;
      }
      if(exid == 3){
        inity = inity+1;
        py = py + res;
      }
    }/// Done with the tracing for this ray/measurement */
  } /// Does this ray pass or not
}

void placepixels(float theta, float rho, int *indices, float *weights,
    int *numpix, float *domain, float res,
    int nx, int ny){

  // int numproc;
  // int myid;
  // MPI_Comm_size(MPI_COMM_WORLD,&numproc);
  // MPI_Comm_rank(MPI_COMM_WORLD,&myid);
  float raylength = 1e6;

  //RAY'S VECTOR REPRESENTAION
  float x = rho*cosf(theta)+0.5*raylength*sinf(theta);
  float y = rho*sinf(theta)-0.5*raylength*cosf(theta);
  float dx = -raylength*sinf(theta);
  float dy = +raylength*cosf(theta);

  //TOP LEVEL
  // Check if this ray passes on my rank's output domain region
  float p[4] = {-dx,dx,-dy,dy};
  float q[4] = {x-domain[0],domain[1]-x,y-domain[2],domain[3]-y};
  float u1 = -1*INFINITY;
  float u2 = INFINITY;
  bool pass = true;
  int inid = 0;
  for(int k = 0; k < 4; k++)
    if(p[k] == 0){
      if(q[k] < 0){
        pass = false;
        break;
      }
    }else{
      float t = q[k]/p[k];
      if(p[k] < 0 && u1 < t){
        u1 = t;
        inid = k;
      }
      else if(p[k] > 0 && u2 > t)
        u2 = t;
    }
  if(u1 > u2 || u1 > 1 || u1 < 0) pass = false;
  //IF RAY COLLIDES WITH DOMAIN
  if(pass){
      //FIND THE INITIAL PIXEL
      int init = 0;
      int initx = 0;
      int inity = 0;
      if(inid == 0){ /// left
        initx = 0;
        inity = (int)((y+u1*dy-domain[2])/res);
      }
      if(inid == 1){ /// right
        initx = nx-1;
        inity = (int)((y+u1*dy-domain[2])/res);
      }
      if(inid == 2){ /// bottom
        initx = (int)((x+u1*dx-domain[0])/res);
        inity = 0;
      }
      if(inid == 3){ /// top
        initx = (int)((x+u1*dx-domain[0])/res);
        inity = ny - 1;
      }
      float px = domain[0] + initx*res+res/2;
      float py = domain[2] + inity*res+res/2;
    //TRACE RAY WHILE IT IS IN THE DOMAIN
    while(px > domain[0] && px < domain[1] && py < domain[3] && py > domain[2]){
      int exid = 0;
      q[0] = x-(px-res/2);
      q[1] = (px+res/2)-x;
      q[2] = y-(py-res/2);
      q[3] = (py+res/2)-y;
      u1 = -1*INFINITY;
      u2 = INFINITY;
      for(int k = 0; k < 4; k++){
        float t = q[k]/p[k];
        if(p[k] < 0 && u1 < t)
          u1 = t;
        else if(p[k] > 0 && u2 > t){
          u2 = t;
          exid = k;
        }
      } /// u2-u1 = ray length in first/current pixel (px,py)
        /// exid shows which side this ray exits
      //ADD CONTRIBUTION FROM CURRENT PIXEL
      //int z = unkmap[inity*len+initx];
      //int z = encode(initx,inity);
      int z = inity*nx+initx;
      indices[*numpix] = z;
      weights[*numpix] = (u2-u1)*raylength;
      *numpix = *numpix + 1;
      //FIND NEXT PIXEL
      if(exid == 0){
        initx = initx-1;
        px = px - res;
      }
      if(exid == 1){
        initx = initx+1;
        px = px + res;
      }
      if(exid == 2){
        inity = inity-1;
        py = py - res;
      }
      if(exid == 3){
        inity = inity+1;
        py = py + res;
      }
    }/// Done with the tracing for this ray/measurement */
  } /// Does this ray pass or not
}
