
//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// ***************************************************
//@HEADER

/*!
 @file ComputeSPMV_ref.cpp

 HPCG routine
 */
#include <iostream>

#include "ComputeSPMV_ref.hpp"

#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#endif

#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif
#include <cassert>
#include <mpi.h>

/*!
  Routine to compute matrix vector product y = Ax where:
  Precondition: First call exchange_externals to get off-processor values of x

  This is the reference SPMV implementation.  It CANNOT be modified for the
  purposes of this benchmark.

  @param[in]  A the known system matrix
  @param[in]  x the known vector
  @param[out] y the On exit contains the result: Ax.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeSPMV
*/


//add openmp later
int ComputeSPMV_ref( const SparseMatrix & A, Vector & x, Vector & y) {

  //Possibly will pass y > cols - but shouldn't change beyond this point
  assert(x.localLength>=A.localNumberOfColumns); // Test vector lengths
  assert(y.localLength>=A.localNumberOfRows);

#ifndef HPCG_NO_MPI
    ExchangeHalo(A,x);
#endif

  const double * const xv = x.values;
  double * const yv = y.values;

  local_int_t ix = 0;
  local_int_t iy = 0;
  local_int_t iz = 0;
  local_int_t nex = 0;
  local_int_t ney = 0;
  local_int_t nez = 0;
  local_int_t nx = A.geom->nx;
  local_int_t ny = A.geom->ny;
  local_int_t nz = A.geom->nz;
  local_int_t nlocal = nx*ny*nz;
//  global_int_t gix0 = A.geom->gix0;
//  global_int_t giy0 = A.geom->giy0;
//  global_int_t giz0 = A.geom->giz0;
//  global_int_t gnx = A.geom->gnx;
//  global_int_t gny = A.geom->gny;
//  global_int_t gnz = A.geom->gnz;
  int npx          = A.geom->npx;
  int npy          = A.geom->npy;
  int npz          = A.geom->npz;
  int ipx          = A.geom->ipx;
  int ipy          = A.geom->ipy;
  int ipz          = A.geom->ipz;

  double diagonal_element = A.matrixDiagonal[0][0];// big assumption?
  int printRank = 13;
  if (A.geom->rank == printRank)
  {
  std::cout << "Printing indices spmv rank "<< printRank << std::endl;
  // old subroutine
  const double * const xv = x.values;
  double * const yv = y.values;
  const local_int_t nrow = A.localNumberOfRows;

  for (local_int_t i=0; i< nrow; i++)  {
    double sum = 0.0;
    const double * const cur_vals = A.matrixValues[i];
    const local_int_t * const cur_inds = A.mtxIndL[i];
    const int cur_nnz = A.nonzerosInRow[i];

    for (int j=0; j< cur_nnz; j++){
//      sum += cur_vals[j]*xv[cur_inds[j]];
        std::cout << i << " x " << j << " " << cur_inds[j] << std::endl;
    //yv[i] = sum;
    }
  }
  }

//  std::cout << ipx << " " << ipy << " " << ipz << " " << A.localNumberOfColumns - nx*ny*nz << std::endl;
//  MPI_Barrier(MPI_COMM_WORLD);
//  return 1000;

//  local_int_t x_min = 0;
//  local_int_t x_max = nx;
  //y z
  // use ipx and npx etc rather than gix0+nx=gnx
//  if (gix0 == 0) x_min = 1;
//  if (gix0 + nx == gnx) x_max = nx-1;

  //yv needs nx original and xv needs different
//  if (npx == 1) nx =

  //zero the vector y
//  for (int i = 0; i< y.localLength ; i++) yv[i] = 0.0;
  for (int i = 0; i< nlocal ; i++) yv[i] = 0.0; //Only specifically zero the part this subroutine changes


  //assert rows = nx*... and cols is new_x*new_y etc

  //local dimensions include halo
  //local_int_t nlx = 0;

  //std::cout << ipx << " " << ipy << " " << ipz << " " << nex << " " << ney << " " << nez << " " << nex*ney*nez << std::endl;
//  local_int_t nlx = nx; nly = ny;local_int_t nlz = nz; //delete this

  assert(nx*ny*nz == A.localNumberOfRows);
//bulk part
for( iz=1; iz< nz-1; iz++){
    for( iy=1; iy< ny-1; iy++){
        for (ix=1; ix < nx-1; ix++){
yv[ix+iy*nx+iz*ny*nx] = diagonal_element * xv[ix+iy*nx+iz*ny*nx]
- xv[ix+-1 + (iy+-1)*nx + (iz+-1)*nx*ny]
- xv[ix+-1 + (iy+-1)*nx + (iz+0)*nx*ny]
- xv[ix+-1 + (iy+-1)*nx + (iz+1)*nx*ny]
- xv[ix+-1 + (iy+0)*nx + (iz+-1)*nx*ny]
- xv[ix+-1 + (iy+0)*nx + (iz+0)*nx*ny]
- xv[ix+-1 + (iy+0)*nx + (iz+1)*nx*ny]
- xv[ix+-1 + (iy+1)*nx + (iz+-1)*nx*ny]
- xv[ix+-1 + (iy+1)*nx + (iz+0)*nx*ny]
- xv[ix+-1 + (iy+1)*nx + (iz+1)*nx*ny]
- xv[ix+0 + (iy+-1)*nx + (iz+-1)*nx*ny]
- xv[ix+0 + (iy+-1)*nx + (iz+0)*nx*ny]
- xv[ix+0 + (iy+-1)*nx + (iz+1)*nx*ny]
- xv[ix+0 + (iy+0)*nx + (iz+-1)*nx*ny]
- xv[ix+0 + (iy+0)*nx + (iz+1)*nx*ny]
- xv[ix+0 + (iy+1)*nx + (iz+-1)*nx*ny]
- xv[ix+0 + (iy+1)*nx + (iz+0)*nx*ny]
- xv[ix+0 + (iy+1)*nx + (iz+1)*nx*ny]
- xv[ix+1 + (iy+-1)*nx + (iz+-1)*nx*ny]
- xv[ix+1 + (iy+-1)*nx + (iz+0)*nx*ny]
- xv[ix+1 + (iy+-1)*nx + (iz+1)*nx*ny]
- xv[ix+1 + (iy+0)*nx + (iz+-1)*nx*ny]
- xv[ix+1 + (iy+0)*nx + (iz+0)*nx*ny]
- xv[ix+1 + (iy+0)*nx + (iz+1)*nx*ny]
- xv[ix+1 + (iy+1)*nx + (iz+-1)*nx*ny]
- xv[ix+1 + (iy+1)*nx + (iz+0)*nx*ny]
- xv[ix+1 + (iy+1)*nx + (iz+1)*nx*ny]
;}
}
}
//sides
iz = 0;
for (ix=1; ix<nx-1;ix++){
    for (iy=1; iy<ny-1;iy++)
{
yv[ix+iy*nx+iz*ny*nx] = diagonal_element * xv[ix+iy*nx+iz*ny*nx]
-xv[(ix+-1)+(iy+-1)*nx+(iz+0)*ny*nx]
-xv[(ix+-1)+(iy+-1)*nx+(iz+1)*ny*nx]
-xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
-xv[(ix+-1)+(iy+0)*nx+(iz+1)*ny*nx]
-xv[(ix+-1)+(iy+1)*nx+(iz+0)*ny*nx]
-xv[(ix+-1)+(iy+1)*nx+(iz+1)*ny*nx]
-xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
-xv[(ix+0)+(iy+-1)*nx+(iz+1)*ny*nx]
-xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
-xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
-xv[(ix+0)+(iy+1)*nx+(iz+1)*ny*nx]
-xv[(ix+1)+(iy+-1)*nx+(iz+0)*ny*nx]
-xv[(ix+1)+(iy+-1)*nx+(iz+1)*ny*nx]
-xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
-xv[(ix+1)+(iy+0)*nx+(iz+1)*ny*nx]
-xv[(ix+1)+(iy+1)*nx+(iz+0)*ny*nx]
-xv[(ix+1)+(iy+1)*nx+(iz+1)*ny*nx]
;}
}
iz = nz-1;
for (ix=1; ix<nx-1;ix++){
    for (iy=1; iy<ny-1;iy++)
{
yv[ix+iy*nx+iz*ny*nx] = diagonal_element * xv[ix+iy*nx+iz*ny*nx]
-xv[(ix+-1)+(iy+-1)*nx+(iz+-1)*ny*nx]
-xv[(ix+-1)+(iy+-1)*nx+(iz+0)*ny*nx]
-xv[(ix+-1)+(iy+0)*nx+(iz+-1)*ny*nx]
-xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
-xv[(ix+-1)+(iy+1)*nx+(iz+-1)*ny*nx]
-xv[(ix+-1)+(iy+1)*nx+(iz+0)*ny*nx]
-xv[(ix+0)+(iy+-1)*nx+(iz+-1)*ny*nx]
-xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
-xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
-xv[(ix+0)+(iy+1)*nx+(iz+-1)*ny*nx]
-xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
-xv[(ix+1)+(iy+-1)*nx+(iz+-1)*ny*nx]
-xv[(ix+1)+(iy+-1)*nx+(iz+0)*ny*nx]
-xv[(ix+1)+(iy+0)*nx+(iz+-1)*ny*nx]
-xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
-xv[(ix+1)+(iy+1)*nx+(iz+-1)*ny*nx]
-xv[(ix+1)+(iy+1)*nx+(iz+0)*ny*nx]
;}
}
iy = 0;
for (ix=1; ix<nx-1;ix++){
    for (iz=1; iz<nz-1;iz++)
{
yv[ix+iy*nx+iz*ny*nx] = diagonal_element * xv[ix+iy*nx+iz*ny*nx]
-xv[(ix+-1)+(iy+0)*nx+(iz+-1)*ny*nx]
-xv[(ix+-1)+(iy+1)*nx+(iz+-1)*ny*nx]
-xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
-xv[(ix+-1)+(iy+1)*nx+(iz+0)*ny*nx]
-xv[(ix+-1)+(iy+0)*nx+(iz+1)*ny*nx]
-xv[(ix+-1)+(iy+1)*nx+(iz+1)*ny*nx]
-xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
-xv[(ix+0)+(iy+1)*nx+(iz+-1)*ny*nx]
-xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
-xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
-xv[(ix+0)+(iy+1)*nx+(iz+1)*ny*nx]
-xv[(ix+1)+(iy+0)*nx+(iz+-1)*ny*nx]
-xv[(ix+1)+(iy+1)*nx+(iz+-1)*ny*nx]
-xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
-xv[(ix+1)+(iy+1)*nx+(iz+0)*ny*nx]
-xv[(ix+1)+(iy+0)*nx+(iz+1)*ny*nx]
-xv[(ix+1)+(iy+1)*nx+(iz+1)*ny*nx]
;}
}
iy = ny-1;
for (ix=1; ix<nx-1;ix++){
    for (iz=1; iz<nz-1;iz++)
{
yv[ix+iy*nx+iz*ny*nx] = diagonal_element * xv[ix+iy*nx+iz*ny*nx]
-xv[(ix+-1)+(iy+-1)*nx+(iz+-1)*ny*nx]
-xv[(ix+-1)+(iy+0)*nx+(iz+-1)*ny*nx]
-xv[(ix+-1)+(iy+-1)*nx+(iz+0)*ny*nx]
-xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
-xv[(ix+-1)+(iy+-1)*nx+(iz+1)*ny*nx]
-xv[(ix+-1)+(iy+0)*nx+(iz+1)*ny*nx]
-xv[(ix+0)+(iy+-1)*nx+(iz+-1)*ny*nx]
-xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
-xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
-xv[(ix+0)+(iy+-1)*nx+(iz+1)*ny*nx]
-xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
-xv[(ix+1)+(iy+-1)*nx+(iz+-1)*ny*nx]
-xv[(ix+1)+(iy+0)*nx+(iz+-1)*ny*nx]
-xv[(ix+1)+(iy+-1)*nx+(iz+0)*ny*nx]
-xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
-xv[(ix+1)+(iy+-1)*nx+(iz+1)*ny*nx]
-xv[(ix+1)+(iy+0)*nx+(iz+1)*ny*nx]
;}
}
ix = 0;
for (iy=1; iy<ny-1;iy++){
    for (iz=1; iz<nz-1;iz++)
{
yv[ix+iy*nx+iz*ny*nx] = diagonal_element * xv[ix+iy*nx+iz*ny*nx]
-xv[(ix+0)+(iy+-1)*nx+(iz+-1)*ny*nx]
-xv[(ix+1)+(iy+-1)*nx+(iz+-1)*ny*nx]
-xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
-xv[(ix+1)+(iy+-1)*nx+(iz+0)*ny*nx]
-xv[(ix+0)+(iy+-1)*nx+(iz+1)*ny*nx]
-xv[(ix+1)+(iy+-1)*nx+(iz+1)*ny*nx]
-xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
-xv[(ix+1)+(iy+0)*nx+(iz+-1)*ny*nx]
-xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
-xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
-xv[(ix+1)+(iy+0)*nx+(iz+1)*ny*nx]
-xv[(ix+0)+(iy+1)*nx+(iz+-1)*ny*nx]
-xv[(ix+1)+(iy+1)*nx+(iz+-1)*ny*nx]
-xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
-xv[(ix+1)+(iy+1)*nx+(iz+0)*ny*nx]
-xv[(ix+0)+(iy+1)*nx+(iz+1)*ny*nx]
-xv[(ix+1)+(iy+1)*nx+(iz+1)*ny*nx]
;}
}
ix = nx-1;
for (iy=1; iy<ny-1;iy++){
    for (iz=1; iz<nz-1;iz++)
{
yv[ix+iy*nx+iz*ny*nx] = diagonal_element * xv[ix+iy*nx+iz*ny*nx]
-xv[(ix+-1)+(iy+-1)*nx+(iz+-1)*ny*nx]
-xv[(ix+0)+(iy+-1)*nx+(iz+-1)*ny*nx]
-xv[(ix+-1)+(iy+-1)*nx+(iz+0)*ny*nx]
-xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
-xv[(ix+-1)+(iy+-1)*nx+(iz+1)*ny*nx]
-xv[(ix+0)+(iy+-1)*nx+(iz+1)*ny*nx]
-xv[(ix+-1)+(iy+0)*nx+(iz+-1)*ny*nx]
-xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
-xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
-xv[(ix+-1)+(iy+0)*nx+(iz+1)*ny*nx]
-xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
-xv[(ix+-1)+(iy+1)*nx+(iz+-1)*ny*nx]
-xv[(ix+0)+(iy+1)*nx+(iz+-1)*ny*nx]
-xv[(ix+-1)+(iy+1)*nx+(iz+0)*ny*nx]
-xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
-xv[(ix+-1)+(iy+1)*nx+(iz+1)*ny*nx]
-xv[(ix+0)+(iy+1)*nx+(iz+1)*ny*nx]
;}
}

//edges
ix = 0;
iy = 0;
for (iz=1; iz<nz-1;iz++){
yv[ix+iy*nx+iz*ny*nx] = diagonal_element * xv[ix+iy*nx+iz*ny*nx]
-xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
-xv[(ix+0)+(iy+1)*nx+(iz+-1)*ny*nx]
-xv[(ix+1)+(iy+0)*nx+(iz+-1)*ny*nx]
-xv[(ix+1)+(iy+1)*nx+(iz+-1)*ny*nx]
-xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
-xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
-xv[(ix+1)+(iy+1)*nx+(iz+0)*ny*nx]
-xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
-xv[(ix+0)+(iy+1)*nx+(iz+1)*ny*nx]
-xv[(ix+1)+(iy+0)*nx+(iz+1)*ny*nx]
-xv[(ix+1)+(iy+1)*nx+(iz+1)*ny*nx]
;}
ix = 0;
iy = ny-1;
for (iz=1; iz<nz-1;iz++){
yv[ix+iy*nx+iz*ny*nx] = diagonal_element * xv[ix+iy*nx+iz*ny*nx]
-xv[(ix+0)+(iy+-1)*nx+(iz+-1)*ny*nx]
-xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
-xv[(ix+1)+(iy+-1)*nx+(iz+-1)*ny*nx]
-xv[(ix+1)+(iy+0)*nx+(iz+-1)*ny*nx]
-xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
-xv[(ix+1)+(iy+-1)*nx+(iz+0)*ny*nx]
-xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
-xv[(ix+0)+(iy+-1)*nx+(iz+1)*ny*nx]
-xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
-xv[(ix+1)+(iy+-1)*nx+(iz+1)*ny*nx]
-xv[(ix+1)+(iy+0)*nx+(iz+1)*ny*nx]
;}
ix = nx-1;
iy = 0;
for (iz=1; iz<nz-1;iz++){
yv[ix+iy*nx+iz*ny*nx] = diagonal_element * xv[ix+iy*nx+iz*ny*nx]
-xv[(ix+-1)+(iy+0)*nx+(iz+-1)*ny*nx]
-xv[(ix+-1)+(iy+1)*nx+(iz+-1)*ny*nx]
-xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
-xv[(ix+0)+(iy+1)*nx+(iz+-1)*ny*nx]
-xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
-xv[(ix+-1)+(iy+1)*nx+(iz+0)*ny*nx]
-xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
-xv[(ix+-1)+(iy+0)*nx+(iz+1)*ny*nx]
-xv[(ix+-1)+(iy+1)*nx+(iz+1)*ny*nx]
-xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
-xv[(ix+0)+(iy+1)*nx+(iz+1)*ny*nx]
;}
ix = nx-1;
iy = ny-1;
for (iz=1; iz<nz-1;iz++){
yv[ix+iy*nx+iz*ny*nx] = diagonal_element * xv[ix+iy*nx+iz*ny*nx]
-xv[(ix+-1)+(iy+-1)*nx+(iz+-1)*ny*nx]
-xv[(ix+-1)+(iy+0)*nx+(iz+-1)*ny*nx]
-xv[(ix+0)+(iy+-1)*nx+(iz+-1)*ny*nx]
-xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
-xv[(ix+-1)+(iy+-1)*nx+(iz+0)*ny*nx]
-xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
-xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
-xv[(ix+-1)+(iy+-1)*nx+(iz+1)*ny*nx]
-xv[(ix+-1)+(iy+0)*nx+(iz+1)*ny*nx]
-xv[(ix+0)+(iy+-1)*nx+(iz+1)*ny*nx]
-xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
;}
ix = 0;
iz = 0;
for (iy=1; iy<ny-1;iy++){
yv[ix+iy*nx+iz*ny*nx] = diagonal_element * xv[ix+iy*nx+iz*ny*nx]
-xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
-xv[(ix+0)+(iy+-1)*nx+(iz+1)*ny*nx]
-xv[(ix+1)+(iy+-1)*nx+(iz+0)*ny*nx]
-xv[(ix+1)+(iy+-1)*nx+(iz+1)*ny*nx]
-xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
-xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
-xv[(ix+1)+(iy+0)*nx+(iz+1)*ny*nx]
-xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
-xv[(ix+0)+(iy+1)*nx+(iz+1)*ny*nx]
-xv[(ix+1)+(iy+1)*nx+(iz+0)*ny*nx]
-xv[(ix+1)+(iy+1)*nx+(iz+1)*ny*nx]
;}
ix = 0;
iz = nz-1;
for (iy=1; iy<ny-1;iy++){
yv[ix+iy*nx+iz*ny*nx] = diagonal_element * xv[ix+iy*nx+iz*ny*nx]
-xv[(ix+0)+(iy+-1)*nx+(iz+-1)*ny*nx]
-xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
-xv[(ix+1)+(iy+-1)*nx+(iz+-1)*ny*nx]
-xv[(ix+1)+(iy+-1)*nx+(iz+0)*ny*nx]
-xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
-xv[(ix+1)+(iy+0)*nx+(iz+-1)*ny*nx]
-xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
-xv[(ix+0)+(iy+1)*nx+(iz+-1)*ny*nx]
-xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
-xv[(ix+1)+(iy+1)*nx+(iz+-1)*ny*nx]
-xv[(ix+1)+(iy+1)*nx+(iz+0)*ny*nx]
;}
ix = nx-1;
iz = 0;
for (iy=1; iy<ny-1;iy++){
yv[ix+iy*nx+iz*ny*nx] = diagonal_element * xv[ix+iy*nx+iz*ny*nx]
-xv[(ix+-1)+(iy+-1)*nx+(iz+0)*ny*nx]
-xv[(ix+-1)+(iy+-1)*nx+(iz+1)*ny*nx]
-xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
-xv[(ix+0)+(iy+-1)*nx+(iz+1)*ny*nx]
-xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
-xv[(ix+-1)+(iy+0)*nx+(iz+1)*ny*nx]
-xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
-xv[(ix+-1)+(iy+1)*nx+(iz+0)*ny*nx]
-xv[(ix+-1)+(iy+1)*nx+(iz+1)*ny*nx]
-xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
-xv[(ix+0)+(iy+1)*nx+(iz+1)*ny*nx]
;}
ix = nx-1;
iz = nz-1;
for (iy=1; iy<ny-1;iy++){
yv[ix+iy*nx+iz*ny*nx] = diagonal_element * xv[ix+iy*nx+iz*ny*nx]
-xv[(ix+-1)+(iy+-1)*nx+(iz+-1)*ny*nx]
-xv[(ix+-1)+(iy+-1)*nx+(iz+0)*ny*nx]
-xv[(ix+0)+(iy+-1)*nx+(iz+-1)*ny*nx]
-xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
-xv[(ix+-1)+(iy+0)*nx+(iz+-1)*ny*nx]
-xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
-xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
-xv[(ix+-1)+(iy+1)*nx+(iz+-1)*ny*nx]
-xv[(ix+-1)+(iy+1)*nx+(iz+0)*ny*nx]
-xv[(ix+0)+(iy+1)*nx+(iz+-1)*ny*nx]
-xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
;}
iy = 0;
iz = 0;
for (ix=1; ix<nx-1;ix++){
yv[ix+iy*nx+iz*ny*nx] = diagonal_element * xv[ix+iy*nx+iz*ny*nx]
-xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
-xv[(ix+-1)+(iy+0)*nx+(iz+1)*ny*nx]
-xv[(ix+-1)+(iy+1)*nx+(iz+0)*ny*nx]
-xv[(ix+-1)+(iy+1)*nx+(iz+1)*ny*nx]
-xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
-xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
-xv[(ix+0)+(iy+1)*nx+(iz+1)*ny*nx]
-xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
-xv[(ix+1)+(iy+0)*nx+(iz+1)*ny*nx]
-xv[(ix+1)+(iy+1)*nx+(iz+0)*ny*nx]
-xv[(ix+1)+(iy+1)*nx+(iz+1)*ny*nx]
;}
iy = 0;
iz = nz-1;
for (ix=1; ix<nx-1;ix++){
yv[ix+iy*nx+iz*ny*nx] = diagonal_element * xv[ix+iy*nx+iz*ny*nx]
-xv[(ix+-1)+(iy+0)*nx+(iz+-1)*ny*nx]
-xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
-xv[(ix+-1)+(iy+1)*nx+(iz+-1)*ny*nx]
-xv[(ix+-1)+(iy+1)*nx+(iz+0)*ny*nx]
-xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
-xv[(ix+0)+(iy+1)*nx+(iz+-1)*ny*nx]
-xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
-xv[(ix+1)+(iy+0)*nx+(iz+-1)*ny*nx]
-xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
-xv[(ix+1)+(iy+1)*nx+(iz+-1)*ny*nx]
-xv[(ix+1)+(iy+1)*nx+(iz+0)*ny*nx]
;}
iy = ny-1;
iz = 0;
for (ix=1; ix<nx-1;ix++){
yv[ix+iy*nx+iz*ny*nx] = diagonal_element * xv[ix+iy*nx+iz*ny*nx]
-xv[(ix+-1)+(iy+-1)*nx+(iz+0)*ny*nx]
-xv[(ix+-1)+(iy+-1)*nx+(iz+1)*ny*nx]
-xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
-xv[(ix+-1)+(iy+0)*nx+(iz+1)*ny*nx]
-xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
-xv[(ix+0)+(iy+-1)*nx+(iz+1)*ny*nx]
-xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
-xv[(ix+1)+(iy+-1)*nx+(iz+0)*ny*nx]
-xv[(ix+1)+(iy+-1)*nx+(iz+1)*ny*nx]
-xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
-xv[(ix+1)+(iy+0)*nx+(iz+1)*ny*nx]
;}
iy = ny-1;
iz = nz-1;
for (ix=1; ix<nx-1;ix++){
yv[ix+iy*nx+iz*ny*nx] = diagonal_element * xv[ix+iy*nx+iz*ny*nx]
-xv[(ix+-1)+(iy+-1)*nx+(iz+-1)*ny*nx]
-xv[(ix+-1)+(iy+-1)*nx+(iz+0)*ny*nx]
-xv[(ix+-1)+(iy+0)*nx+(iz+-1)*ny*nx]
-xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
-xv[(ix+0)+(iy+-1)*nx+(iz+-1)*ny*nx]
-xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
-xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
-xv[(ix+1)+(iy+-1)*nx+(iz+-1)*ny*nx]
-xv[(ix+1)+(iy+-1)*nx+(iz+0)*ny*nx]
-xv[(ix+1)+(iy+0)*nx+(iz+-1)*ny*nx]
-xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
;}

//corners
ix = 0;
iy = 0;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] = diagonal_element * xv[ix+iy*nx+iz*ny*nx]
-xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
-xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
-xv[(ix+0)+(iy+1)*nx+(iz+1)*ny*nx]
-xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
-xv[(ix+1)+(iy+0)*nx+(iz+1)*ny*nx]
-xv[(ix+1)+(iy+1)*nx+(iz+0)*ny*nx]
-xv[(ix+1)+(iy+1)*nx+(iz+1)*ny*nx]
;
ix = 0;
iy = 0;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] = diagonal_element * xv[ix+iy*nx+iz*ny*nx]
-xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
-xv[(ix+0)+(iy+1)*nx+(iz+-1)*ny*nx]
-xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
-xv[(ix+1)+(iy+0)*nx+(iz+-1)*ny*nx]
-xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
-xv[(ix+1)+(iy+1)*nx+(iz+-1)*ny*nx]
-xv[(ix+1)+(iy+1)*nx+(iz+0)*ny*nx]
;
ix = 0;
iy = ny-1;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] = diagonal_element * xv[ix+iy*nx+iz*ny*nx]
-xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
-xv[(ix+0)+(iy+-1)*nx+(iz+1)*ny*nx]
-xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
-xv[(ix+1)+(iy+-1)*nx+(iz+0)*ny*nx]
-xv[(ix+1)+(iy+-1)*nx+(iz+1)*ny*nx]
-xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
-xv[(ix+1)+(iy+0)*nx+(iz+1)*ny*nx]
;
ix = 0;
iy = ny-1;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] = diagonal_element * xv[ix+iy*nx+iz*ny*nx]
-xv[(ix+0)+(iy+-1)*nx+(iz+-1)*ny*nx]
-xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
-xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
-xv[(ix+1)+(iy+-1)*nx+(iz+-1)*ny*nx]
-xv[(ix+1)+(iy+-1)*nx+(iz+0)*ny*nx]
-xv[(ix+1)+(iy+0)*nx+(iz+-1)*ny*nx]
-xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
;
ix = nx-1;
iy = 0;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] = diagonal_element * xv[ix+iy*nx+iz*ny*nx]
-xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
-xv[(ix+-1)+(iy+0)*nx+(iz+1)*ny*nx]
-xv[(ix+-1)+(iy+1)*nx+(iz+0)*ny*nx]
-xv[(ix+-1)+(iy+1)*nx+(iz+1)*ny*nx]
-xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
-xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
-xv[(ix+0)+(iy+1)*nx+(iz+1)*ny*nx]
;
ix = nx-1;
iy = 0;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] = diagonal_element * xv[ix+iy*nx+iz*ny*nx]
-xv[(ix+-1)+(iy+0)*nx+(iz+-1)*ny*nx]
-xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
-xv[(ix+-1)+(iy+1)*nx+(iz+-1)*ny*nx]
-xv[(ix+-1)+(iy+1)*nx+(iz+0)*ny*nx]
-xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
-xv[(ix+0)+(iy+1)*nx+(iz+-1)*ny*nx]
-xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
;
ix = nx-1;
iy = ny-1;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] = diagonal_element * xv[ix+iy*nx+iz*ny*nx]
-xv[(ix+-1)+(iy+-1)*nx+(iz+0)*ny*nx]
-xv[(ix+-1)+(iy+-1)*nx+(iz+1)*ny*nx]
-xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
-xv[(ix+-1)+(iy+0)*nx+(iz+1)*ny*nx]
-xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
-xv[(ix+0)+(iy+-1)*nx+(iz+1)*ny*nx]
-xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
;
ix = nx-1;
iy = ny-1;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] = diagonal_element * xv[ix+iy*nx+iz*ny*nx]
-xv[(ix+-1)+(iy+-1)*nx+(iz+-1)*ny*nx]
-xv[(ix+-1)+(iy+-1)*nx+(iz+0)*ny*nx]
-xv[(ix+-1)+(iy+0)*nx+(iz+-1)*ny*nx]
-xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
-xv[(ix+0)+(iy+-1)*nx+(iz+-1)*ny*nx]
-xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
-xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
;


  if (nex*ney*nez == 0) return 0;



if(ipx > 0)
{
  if(ipx < npx - 1)
{
    if(ipy > 0)
{
      if(ipy < npy - 1)
{
        if(ipz > 0)
{
            if(ipz < npz - 1)
{
ix = 0;
iy = 0;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+1+1]
;
ix = 0;
iy = 0;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+1]
;
ix = 0;
iy = ny-1;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+1]
;
ix = 0;
iy = ny-1;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+ny+1+1]
;
ix = nx-1;
iy = 0;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx-2]
-xv[nlocal+0+1+nx-1]
-xv[nlocal+0+1+nx]
;
ix = nx-1;
iy = 0;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx]
;
ix = nx-1;
iy = ny-1;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx]
;
ix = nx-1;
iy = ny-1;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+ny+1+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+ny+1+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+ny+1+nx]
;
iy=0;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+ix-1]
-xv[nlocal+0+1+ix]
-xv[nlocal+0+1+ix+1]
;}
iy=0;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx+ix-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx+ix]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix+(nz-2)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix+(nz-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix+(nz-1)*nx+1]
;}
iy=0;
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+ix-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+ix]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+ix+1]
;}
iz=0;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+ix-1]
-xv[nlocal+0+1+nx+1+ny+ix]
-xv[nlocal+0+1+nx+1+ny+ix+1]
-xv[nlocal+0+1+nx+1+ny+nx+ix-1]
-xv[nlocal+0+1+nx+1+ny+nx+ix]
-xv[nlocal+0+1+nx+1+ny+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+1+nx+1+ny+ix+(ny-2)*nx]
-xv[nlocal+0+1+nx+1+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+1+nx+1+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+1+nx+1+ny+ix+(ny-1)*nx]
-xv[nlocal+0+1+nx+1+ny+ix+(ny-1)*nx+1]
;}
iz=nz-1;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx+ix-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx+ix]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix+(ny-2)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix+(ny-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix+(ny-1)*nx+1]
;}
iy=ny-1;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+ix-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+ix]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+ix+1]
;}
iy=ny-1;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx+ix-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx+ix]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(nz-2)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(nz-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(nz-1)*nx+1]
;}
iy=ny-1;
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+ny+1+ix-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+ny+1+ix]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+ny+1+ix+1]
;}
;
ix=0;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+iy-1]
-xv[nlocal+0+1+nx+1+iy]
-xv[nlocal+0+1+nx+1+iy+1]
;}
ix=0;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny+iy-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny+iy]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy+(nz-2)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy+(nz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy+(nz-1)*ny+1]
;}
ix=0;
iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+iy-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+iy]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+iy+1]
;}
iz=0;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+(iy-1)*nx]
-xv[nlocal+0+1+nx+1+ny+(iy-1)*nx+1]
-xv[nlocal+0+1+nx+1+ny+(iy)*nx]
-xv[nlocal+0+1+nx+1+ny+(iy-1)*nx+1]
-xv[nlocal+0+1+nx+1+ny+(iy+1)*nx]
-xv[nlocal+0+1+nx+1+ny+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+1+nx+1+ny+(iy)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+(iy)*nx+nx-1]
-xv[nlocal+0+1+nx+1+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+(iy+1)*nx+nx-1]
;}
iz=nz-1;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+(iy-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+(iy-1)*nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+(iy)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+(iy-1)*nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+(iy+1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+(iy)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+(iy)*nx+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+(iy+1)*nx+nx-1]
;}
ix=nx-1;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+iy-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+iy]
-xv[nlocal+0+1+nx+1+ny+nx*ny+iy+1]
;}
ix=nx-1;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny+iy-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny+iy]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy+(nz-2)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy+(nz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy+(nz-1)*ny+1]
;}
ix=nx-1;
iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+iy-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+iy]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+iy+1]
;}
;
ix=0;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+iz-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+iz]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+iz+1]
;}
ix=0;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+(iz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+(iz-1)*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+(iz)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+(iz-1)*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+(iz+1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+(iz)*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+(iz)*ny+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+(iz+1)*ny+ny-1]
;}
ix=0;
iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+iz-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+iz]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+iz+1]
;}
iy=0;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+(iz-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+(iz-1)*nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+(iz)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+(iz-1)*nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+(iz+1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+(iz)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+(iz)*nx+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+(iz+1)*nx+nx-1]
;}
iy=ny-1;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz-1)*nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz-1)*nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz+1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz)*nx+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz+1)*nx+nx-1]
;}
ix=nx-1;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+iz-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+iz]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+iz+1]
;}
ix=nx-1;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+(iz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+(iz-1)*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+(iz)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+(iz-1)*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+(iz+1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+(iz)*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+(iz)*ny+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+(iz+1)*ny+ny-1]
;}
ix=nx-1;
iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+iz-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+iz]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+iz+1]
;}
;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy-1+(iz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy+(iz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy+1+(iz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy-1+(iz)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy+iz*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy+1+iz*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy-1+(iz+1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy+(iz+1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy+1+(iz+1)*ny]
;}
}
ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy-1+(iz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy+(iz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy+1+(iz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy-1+(iz)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy+iz*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy+1+iz*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy-1+(iz+1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy+(iz+1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy+1+(iz+1)*ny]
;}
}
iy=0;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix-1+(iz-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix+(iz-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix+1+(iz-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix-1+(iz)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix+iz*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix+1+iz*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix-1+(iz+1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix+(iz+1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix+1+(iz+1)*nx]
;}
}
iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix-1+(iz-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(iz-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+1+(iz-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix-1+(iz)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+iz*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+1+iz*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix-1+(iz+1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(iz+1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+1+(iz+1)*nx]
;}
}
iz=0;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+ix-1+(iy-1)*nx]
-xv[nlocal+0+1+nx+1+ny+ix+(iy-1)*nx]
-xv[nlocal+0+1+nx+1+ny+ix+1+(iy-1)*nx]
-xv[nlocal+0+1+nx+1+ny+ix-1+(iy)*nx]
-xv[nlocal+0+1+nx+1+ny+ix+iy*nx]
-xv[nlocal+0+1+nx+1+ny+ix+1+iy*nx]
-xv[nlocal+0+1+nx+1+ny+ix-1+(iy+1)*nx]
-xv[nlocal+0+1+nx+1+ny+ix+(iy+1)*nx]
-xv[nlocal+0+1+nx+1+ny+ix+1+(iy+1)*nx]
;}
}
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix-1+(iy-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix+(iy-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix+1+(iy-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix-1+(iy)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix+iy*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix+1+iy*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix-1+(iy+1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix+(iy+1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix+1+(iy+1)*nx]
;}
}
}//ipz < npz - 1
            else
{
ix = 0;
iy = 0;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+1+1]
;
ix = 0;
iy = ny-1;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+1]
;
ix = nx-1;
iy = 0;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx-2]
-xv[nlocal+0+1+nx-1]
-xv[nlocal+0+1+nx]
;
ix = nx-1;
iy = ny-1;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx]
;
iy=0;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+ix-1]
-xv[nlocal+0+1+ix]
-xv[nlocal+0+1+ix+1]
;}
iy=0;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx+ix-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx+ix]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix+(nz-2)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix+(nz-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix+(nz-1)*nx+1]
;}
iz=0;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+ix-1]
-xv[nlocal+0+1+nx+1+ny+ix]
-xv[nlocal+0+1+nx+1+ny+ix+1]
-xv[nlocal+0+1+nx+1+ny+nx+ix-1]
-xv[nlocal+0+1+nx+1+ny+nx+ix]
-xv[nlocal+0+1+nx+1+ny+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+1+nx+1+ny+ix+(ny-2)*nx]
-xv[nlocal+0+1+nx+1+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+1+nx+1+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+1+nx+1+ny+ix+(ny-1)*nx]
-xv[nlocal+0+1+nx+1+ny+ix+(ny-1)*nx+1]
;}
iy=ny-1;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+ix-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+ix]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+ix+1]
;}
iy=ny-1;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx+ix-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx+ix]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(nz-2)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(nz-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(nz-1)*nx+1]
;}
;
ix=0;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+iy-1]
-xv[nlocal+0+1+nx+1+iy]
-xv[nlocal+0+1+nx+1+iy+1]
;}
ix=0;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny+iy-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny+iy]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy+(nz-2)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy+(nz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy+(nz-1)*ny+1]
;}
iz=0;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+(iy-1)*nx]
-xv[nlocal+0+1+nx+1+ny+(iy-1)*nx+1]
-xv[nlocal+0+1+nx+1+ny+(iy)*nx]
-xv[nlocal+0+1+nx+1+ny+(iy-1)*nx+1]
-xv[nlocal+0+1+nx+1+ny+(iy+1)*nx]
-xv[nlocal+0+1+nx+1+ny+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+1+nx+1+ny+(iy)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+(iy)*nx+nx-1]
-xv[nlocal+0+1+nx+1+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+(iy+1)*nx+nx-1]
;}
ix=nx-1;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+iy-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+iy]
-xv[nlocal+0+1+nx+1+ny+nx*ny+iy+1]
;}
ix=nx-1;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny+iy-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny+iy]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy+(nz-2)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy+(nz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy+(nz-1)*ny+1]
;}
;
ix=0;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+iz-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+iz]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+iz+1]
;}
ix=0;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+(iz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+(iz-1)*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+(iz)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+(iz-1)*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+(iz+1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+(iz)*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+(iz)*ny+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+(iz+1)*ny+ny-1]
;}
ix=0;
iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+iz-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+iz]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+iz+1]
;}
iy=0;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+(iz-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+(iz-1)*nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+(iz)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+(iz-1)*nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+(iz+1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+(iz)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+(iz)*nx+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+(iz+1)*nx+nx-1]
;}
iy=ny-1;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz-1)*nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz-1)*nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz+1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz)*nx+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz+1)*nx+nx-1]
;}
ix=nx-1;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+iz-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+iz]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+iz+1]
;}
ix=nx-1;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+(iz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+(iz-1)*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+(iz)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+(iz-1)*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+(iz+1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+(iz)*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+(iz)*ny+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+(iz+1)*ny+ny-1]
;}
ix=nx-1;
iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+iz-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+iz]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+iz+1]
;}
;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy-1+(iz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy+(iz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy+1+(iz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy-1+(iz)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy+iz*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy+1+iz*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy-1+(iz+1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy+(iz+1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy+1+(iz+1)*ny]
;}
}
ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy-1+(iz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy+(iz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy+1+(iz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy-1+(iz)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy+iz*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy+1+iz*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy-1+(iz+1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy+(iz+1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy+1+(iz+1)*ny]
;}
}
iy=0;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix-1+(iz-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix+(iz-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix+1+(iz-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix-1+(iz)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix+iz*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix+1+iz*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix-1+(iz+1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix+(iz+1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix+1+(iz+1)*nx]
;}
}
iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix-1+(iz-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(iz-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+1+(iz-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix-1+(iz)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+iz*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+1+iz*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix-1+(iz+1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(iz+1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+1+(iz+1)*nx]
;}
}
iz=0;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+ix-1+(iy-1)*nx]
-xv[nlocal+0+1+nx+1+ny+ix+(iy-1)*nx]
-xv[nlocal+0+1+nx+1+ny+ix+1+(iy-1)*nx]
-xv[nlocal+0+1+nx+1+ny+ix-1+(iy)*nx]
-xv[nlocal+0+1+nx+1+ny+ix+iy*nx]
-xv[nlocal+0+1+nx+1+ny+ix+1+iy*nx]
-xv[nlocal+0+1+nx+1+ny+ix-1+(iy+1)*nx]
-xv[nlocal+0+1+nx+1+ny+ix+(iy+1)*nx]
-xv[nlocal+0+1+nx+1+ny+ix+1+(iy+1)*nx]
;}
}
}//ipz < npz - 1
          }//ipz > 0 
        else
{
            if(ipz < npz - 1)
{
ix = 0;
iy = 0;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+1]
;
ix = 0;
iy = ny-1;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+ny+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+ny+1+1]
;
ix = nx-1;
iy = 0;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx]
;
ix = nx-1;
iy = ny-1;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+ny+1+nx-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+ny+1+nx-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+ny+1+nx]
;
iy=0;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+ix-1]
-xv[nlocal+0+nz+ix]
-xv[nlocal+0+nz+ix+1]
-xv[nlocal+0+nz+nx+ix-1]
-xv[nlocal+0+nz+nx+ix]
-xv[nlocal+0+nz+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+nz+ix+(nz-2)*nx]
-xv[nlocal+0+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+nz+ix+(nz-1)*nx]
-xv[nlocal+0+nz+ix+(nz-1)*nx+1]
;}
iy=0;
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+ix-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+ix]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+ix+1]
;}
iz=nz-1;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx+ix-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx+ix]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix+(ny-2)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix+(ny-1)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix+(ny-1)*nx+1]
;}
iy=ny-1;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx+ix-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx+ix]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(nz-2)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(nz-1)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(nz-1)*nx+1]
;}
iy=ny-1;
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+ny+1+ix-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+ny+1+ix]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+ny+1+ix+1]
;}
;
ix=0;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+iy-1]
-xv[nlocal+0+nz+nx*nz+nz+iy]
-xv[nlocal+0+nz+nx*nz+nz+iy+1]
-xv[nlocal+0+nz+nx*nz+nz+ny+iy-1]
-xv[nlocal+0+nz+nx*nz+nz+ny+iy]
-xv[nlocal+0+nz+nx*nz+nz+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+nz+nx*nz+nz+iy+(nz-2)*ny]
-xv[nlocal+0+nz+nx*nz+nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+nz+nx*nz+nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+nz+nx*nz+nz+iy+(nz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+iy+(nz-1)*ny+1]
;}
ix=0;
iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+iy-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+iy]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+iy+1]
;}
iz=nz-1;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+(iy-1)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+(iy-1)*nx+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+(iy)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+(iy-1)*nx+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+(iy+1)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+(iy)*nx+nx-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+(iy)*nx+nx-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+(iy+1)*nx+nx-1]
;}
ix=nx-1;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny+iy-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny+iy]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(nz-2)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(nz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(nz-1)*ny+1]
;}
ix=nx-1;
iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+iy-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+iy]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+iy+1]
;}
;
ix=0;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iz-1]
-xv[nlocal+0+iz]
-xv[nlocal+0+iz+1]
;}
ix=0;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+(iz-1)*ny+1]
-xv[nlocal+0+nz+nx*nz+nz+(iz)*ny]
-xv[nlocal+0+nz+nx*nz+nz+(iz-1)*ny+1]
-xv[nlocal+0+nz+nx*nz+nz+(iz+1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+nz+nx*nz+nz+(iz)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+(iz)*ny+ny-1]
-xv[nlocal+0+nz+nx*nz+nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+(iz+1)*ny+ny-1]
;}
ix=0;
iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+iz-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+iz]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+iz+1]
;}
iy=0;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+(iz-1)*nx]
-xv[nlocal+0+nz+(iz-1)*nx+1]
-xv[nlocal+0+nz+(iz)*nx]
-xv[nlocal+0+nz+(iz-1)*nx+1]
-xv[nlocal+0+nz+(iz+1)*nx]
-xv[nlocal+0+nz+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+nz+(iz)*nx+nx-2]
-xv[nlocal+0+nz+(iz)*nx+nx-1]
-xv[nlocal+0+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+nz+(iz+1)*nx+nx-1]
;}
iy=ny-1;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz-1)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz-1)*nx+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz-1)*nx+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz+1)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz)*nx+nx-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz)*nx+nx-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz+1)*nx+nx-1]
;}
ix=nx-1;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+iz-1]
-xv[nlocal+0+nz+nx*nz+iz]
-xv[nlocal+0+nz+nx*nz+iz+1]
;}
ix=nx-1;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz-1)*ny+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz-1)*ny+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz+1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz)*ny+ny-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz+1)*ny+ny-1]
;}
ix=nx-1;
iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+iz-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+iz]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+iz+1]
;}
;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+iy-1+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+iy+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+iy+1+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+iy-1+(iz)*ny]
-xv[nlocal+0+nz+nx*nz+nz+iy+iz*ny]
-xv[nlocal+0+nz+nx*nz+nz+iy+1+iz*ny]
-xv[nlocal+0+nz+nx*nz+nz+iy-1+(iz+1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+iy+(iz+1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+iy+1+(iz+1)*ny]
;}
}
ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy-1+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+1+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy-1+(iz)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+iz*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+1+iz*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy-1+(iz+1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(iz+1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+1+(iz+1)*ny]
;}
}
iy=0;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+ix-1+(iz-1)*nx]
-xv[nlocal+0+nz+ix+(iz-1)*nx]
-xv[nlocal+0+nz+ix+1+(iz-1)*nx]
-xv[nlocal+0+nz+ix-1+(iz)*nx]
-xv[nlocal+0+nz+ix+iz*nx]
-xv[nlocal+0+nz+ix+1+iz*nx]
-xv[nlocal+0+nz+ix-1+(iz+1)*nx]
-xv[nlocal+0+nz+ix+(iz+1)*nx]
-xv[nlocal+0+nz+ix+1+(iz+1)*nx]
;}
}
iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix-1+(iz-1)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(iz-1)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+1+(iz-1)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix-1+(iz)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+iz*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+1+iz*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix-1+(iz+1)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(iz+1)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+1+(iz+1)*nx]
;}
}
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix-1+(iy-1)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix+(iy-1)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix+1+(iy-1)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix-1+(iy)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix+iy*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix+1+iy*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix-1+(iy+1)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix+(iy+1)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix+1+(iy+1)*nx]
;}
}
}//ipz < npz - 1
            else
{
iy=0;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+ix-1]
-xv[nlocal+0+nz+ix]
-xv[nlocal+0+nz+ix+1]
-xv[nlocal+0+nz+nx+ix-1]
-xv[nlocal+0+nz+nx+ix]
-xv[nlocal+0+nz+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+nz+ix+(nz-2)*nx]
-xv[nlocal+0+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+nz+ix+(nz-1)*nx]
-xv[nlocal+0+nz+ix+(nz-1)*nx+1]
;}
iy=ny-1;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx+ix-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx+ix]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(nz-2)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(nz-1)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(nz-1)*nx+1]
;}
;
ix=0;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+iy-1]
-xv[nlocal+0+nz+nx*nz+nz+iy]
-xv[nlocal+0+nz+nx*nz+nz+iy+1]
-xv[nlocal+0+nz+nx*nz+nz+ny+iy-1]
-xv[nlocal+0+nz+nx*nz+nz+ny+iy]
-xv[nlocal+0+nz+nx*nz+nz+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+nz+nx*nz+nz+iy+(nz-2)*ny]
-xv[nlocal+0+nz+nx*nz+nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+nz+nx*nz+nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+nz+nx*nz+nz+iy+(nz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+iy+(nz-1)*ny+1]
;}
ix=nx-1;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny+iy-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny+iy]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(nz-2)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(nz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(nz-1)*ny+1]
;}
;
ix=0;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iz-1]
-xv[nlocal+0+iz]
-xv[nlocal+0+iz+1]
;}
ix=0;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+(iz-1)*ny+1]
-xv[nlocal+0+nz+nx*nz+nz+(iz)*ny]
-xv[nlocal+0+nz+nx*nz+nz+(iz-1)*ny+1]
-xv[nlocal+0+nz+nx*nz+nz+(iz+1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+nz+nx*nz+nz+(iz)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+(iz)*ny+ny-1]
-xv[nlocal+0+nz+nx*nz+nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+(iz+1)*ny+ny-1]
;}
ix=0;
iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+iz-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+iz]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+iz+1]
;}
iy=0;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+(iz-1)*nx]
-xv[nlocal+0+nz+(iz-1)*nx+1]
-xv[nlocal+0+nz+(iz)*nx]
-xv[nlocal+0+nz+(iz-1)*nx+1]
-xv[nlocal+0+nz+(iz+1)*nx]
-xv[nlocal+0+nz+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+nz+(iz)*nx+nx-2]
-xv[nlocal+0+nz+(iz)*nx+nx-1]
-xv[nlocal+0+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+nz+(iz+1)*nx+nx-1]
;}
iy=ny-1;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz-1)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz-1)*nx+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz-1)*nx+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz+1)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz)*nx+nx-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz)*nx+nx-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz+1)*nx+nx-1]
;}
ix=nx-1;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+iz-1]
-xv[nlocal+0+nz+nx*nz+iz]
-xv[nlocal+0+nz+nx*nz+iz+1]
;}
ix=nx-1;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz-1)*ny+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz-1)*ny+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz+1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz)*ny+ny-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz+1)*ny+ny-1]
;}
ix=nx-1;
iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+iz-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+iz]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+iz+1]
;}
;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+iy-1+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+iy+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+iy+1+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+iy-1+(iz)*ny]
-xv[nlocal+0+nz+nx*nz+nz+iy+iz*ny]
-xv[nlocal+0+nz+nx*nz+nz+iy+1+iz*ny]
-xv[nlocal+0+nz+nx*nz+nz+iy-1+(iz+1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+iy+(iz+1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+iy+1+(iz+1)*ny]
;}
}
ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy-1+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+1+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy-1+(iz)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+iz*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+1+iz*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy-1+(iz+1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(iz+1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+1+(iz+1)*ny]
;}
}
iy=0;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+ix-1+(iz-1)*nx]
-xv[nlocal+0+nz+ix+(iz-1)*nx]
-xv[nlocal+0+nz+ix+1+(iz-1)*nx]
-xv[nlocal+0+nz+ix-1+(iz)*nx]
-xv[nlocal+0+nz+ix+iz*nx]
-xv[nlocal+0+nz+ix+1+iz*nx]
-xv[nlocal+0+nz+ix-1+(iz+1)*nx]
-xv[nlocal+0+nz+ix+(iz+1)*nx]
-xv[nlocal+0+nz+ix+1+(iz+1)*nx]
;}
}
iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix-1+(iz-1)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(iz-1)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+1+(iz-1)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix-1+(iz)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+iz*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+1+iz*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix-1+(iz+1)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(iz+1)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+1+(iz+1)*nx]
;}
}
}//ipz < npz - 1
          }//ipz > 0 
        }//ipy < npy - 1 
      else
{
        if(ipz > 0)
{
            if(ipz < npz - 1)
{
ix = 0;
iy = 0;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+1+1]
;
ix = 0;
iy = 0;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+1]
;
ix = nx-1;
iy = 0;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx-2]
-xv[nlocal+0+1+nx-1]
-xv[nlocal+0+1+nx]
;
ix = nx-1;
iy = 0;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx]
;
iy=0;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+ix-1]
-xv[nlocal+0+1+ix]
-xv[nlocal+0+1+ix+1]
;}
iy=0;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx+ix-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx+ix]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix+(nz-2)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix+(nz-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix+(nz-1)*nx+1]
;}
iy=0;
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+ix-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+ix]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+ix+1]
;}
iz=0;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+ix-1]
-xv[nlocal+0+1+nx+1+ny+ix]
-xv[nlocal+0+1+nx+1+ny+ix+1]
-xv[nlocal+0+1+nx+1+ny+nx+ix-1]
-xv[nlocal+0+1+nx+1+ny+nx+ix]
-xv[nlocal+0+1+nx+1+ny+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+1+nx+1+ny+ix+(ny-2)*nx]
-xv[nlocal+0+1+nx+1+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+1+nx+1+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+1+nx+1+ny+ix+(ny-1)*nx]
-xv[nlocal+0+1+nx+1+ny+ix+(ny-1)*nx+1]
;}
iz=nz-1;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx+ix-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx+ix]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix+(ny-2)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix+(ny-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix+(ny-1)*nx+1]
;}
;
ix=0;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+iy-1]
-xv[nlocal+0+1+nx+1+iy]
-xv[nlocal+0+1+nx+1+iy+1]
;}
ix=0;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny+iy-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny+iy]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy+(nz-2)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy+(nz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy+(nz-1)*ny+1]
;}
ix=0;
iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+iy-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+iy]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+iy+1]
;}
iz=0;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+(iy-1)*nx]
-xv[nlocal+0+1+nx+1+ny+(iy-1)*nx+1]
-xv[nlocal+0+1+nx+1+ny+(iy)*nx]
-xv[nlocal+0+1+nx+1+ny+(iy-1)*nx+1]
-xv[nlocal+0+1+nx+1+ny+(iy+1)*nx]
-xv[nlocal+0+1+nx+1+ny+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+1+nx+1+ny+(iy)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+(iy)*nx+nx-1]
-xv[nlocal+0+1+nx+1+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+(iy+1)*nx+nx-1]
;}
iz=nz-1;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+(iy-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+(iy-1)*nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+(iy)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+(iy-1)*nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+(iy+1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+(iy)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+(iy)*nx+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+(iy+1)*nx+nx-1]
;}
ix=nx-1;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+iy-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+iy]
-xv[nlocal+0+1+nx+1+ny+nx*ny+iy+1]
;}
ix=nx-1;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny+iy-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny+iy]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy+(nz-2)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy+(nz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy+(nz-1)*ny+1]
;}
ix=nx-1;
iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx*ny+iy-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx*ny+iy]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx*ny+iy+1]
;}
;
ix=0;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+iz-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+iz]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+iz+1]
;}
ix=0;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+(iz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+(iz-1)*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+(iz)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+(iz-1)*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+(iz+1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+(iz)*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+(iz)*ny+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+(iz+1)*ny+ny-1]
;}
iy=0;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+(iz-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+(iz-1)*nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+(iz)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+(iz-1)*nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+(iz+1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+(iz)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+(iz)*nx+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+(iz+1)*nx+nx-1]
;}
ix=nx-1;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+iz-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+iz]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+iz+1]
;}
ix=nx-1;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+(iz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+(iz-1)*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+(iz)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+(iz-1)*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+(iz+1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+(iz)*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+(iz)*ny+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+(iz+1)*ny+ny-1]
;}
;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy-1+(iz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy+(iz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy+1+(iz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy-1+(iz)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy+iz*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy+1+iz*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy-1+(iz+1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy+(iz+1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy+1+(iz+1)*ny]
;}
}
ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy-1+(iz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy+(iz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy+1+(iz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy-1+(iz)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy+iz*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy+1+iz*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy-1+(iz+1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy+(iz+1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy+1+(iz+1)*ny]
;}
}
iy=0;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix-1+(iz-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix+(iz-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix+1+(iz-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix-1+(iz)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix+iz*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix+1+iz*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix-1+(iz+1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix+(iz+1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix+1+(iz+1)*nx]
;}
}
iz=0;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+ix-1+(iy-1)*nx]
-xv[nlocal+0+1+nx+1+ny+ix+(iy-1)*nx]
-xv[nlocal+0+1+nx+1+ny+ix+1+(iy-1)*nx]
-xv[nlocal+0+1+nx+1+ny+ix-1+(iy)*nx]
-xv[nlocal+0+1+nx+1+ny+ix+iy*nx]
-xv[nlocal+0+1+nx+1+ny+ix+1+iy*nx]
-xv[nlocal+0+1+nx+1+ny+ix-1+(iy+1)*nx]
-xv[nlocal+0+1+nx+1+ny+ix+(iy+1)*nx]
-xv[nlocal+0+1+nx+1+ny+ix+1+(iy+1)*nx]
;}
}
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix-1+(iy-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix+(iy-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix+1+(iy-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix-1+(iy)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix+iy*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix+1+iy*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix-1+(iy+1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix+(iy+1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix+1+(iy+1)*nx]
;}
}
}//ipz < npz - 1
            else
{
ix = 0;
iy = 0;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+1+1]
;
ix = nx-1;
iy = 0;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx-2]
-xv[nlocal+0+1+nx-1]
-xv[nlocal+0+1+nx]
;
iy=0;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+ix-1]
-xv[nlocal+0+1+ix]
-xv[nlocal+0+1+ix+1]
;}
iy=0;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx+ix-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx+ix]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix+(nz-2)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix+(nz-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix+(nz-1)*nx+1]
;}
iz=0;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+ix-1]
-xv[nlocal+0+1+nx+1+ny+ix]
-xv[nlocal+0+1+nx+1+ny+ix+1]
-xv[nlocal+0+1+nx+1+ny+nx+ix-1]
-xv[nlocal+0+1+nx+1+ny+nx+ix]
-xv[nlocal+0+1+nx+1+ny+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+1+nx+1+ny+ix+(ny-2)*nx]
-xv[nlocal+0+1+nx+1+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+1+nx+1+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+1+nx+1+ny+ix+(ny-1)*nx]
-xv[nlocal+0+1+nx+1+ny+ix+(ny-1)*nx+1]
;}
;
ix=0;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+iy-1]
-xv[nlocal+0+1+nx+1+iy]
-xv[nlocal+0+1+nx+1+iy+1]
;}
ix=0;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny+iy-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny+iy]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy+(nz-2)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy+(nz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy+(nz-1)*ny+1]
;}
iz=0;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+(iy-1)*nx]
-xv[nlocal+0+1+nx+1+ny+(iy-1)*nx+1]
-xv[nlocal+0+1+nx+1+ny+(iy)*nx]
-xv[nlocal+0+1+nx+1+ny+(iy-1)*nx+1]
-xv[nlocal+0+1+nx+1+ny+(iy+1)*nx]
-xv[nlocal+0+1+nx+1+ny+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+1+nx+1+ny+(iy)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+(iy)*nx+nx-1]
-xv[nlocal+0+1+nx+1+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+(iy+1)*nx+nx-1]
;}
ix=nx-1;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+iy-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+iy]
-xv[nlocal+0+1+nx+1+ny+nx*ny+iy+1]
;}
ix=nx-1;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny+iy-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny+iy]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy+(nz-2)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy+(nz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy+(nz-1)*ny+1]
;}
;
ix=0;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+iz-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+iz]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+iz+1]
;}
ix=0;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+(iz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+(iz-1)*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+(iz)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+(iz-1)*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+(iz+1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+(iz)*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+(iz)*ny+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+(iz+1)*ny+ny-1]
;}
iy=0;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+(iz-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+(iz-1)*nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+(iz)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+(iz-1)*nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+(iz+1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+(iz)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+(iz)*nx+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+(iz+1)*nx+nx-1]
;}
ix=nx-1;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+iz-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+iz]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+iz+1]
;}
ix=nx-1;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+(iz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+(iz-1)*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+(iz)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+(iz-1)*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+(iz+1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+(iz)*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+(iz)*ny+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+(iz+1)*ny+ny-1]
;}
;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy-1+(iz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy+(iz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy+1+(iz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy-1+(iz)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy+iz*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy+1+iz*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy-1+(iz+1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy+(iz+1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy+1+(iz+1)*ny]
;}
}
ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy-1+(iz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy+(iz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy+1+(iz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy-1+(iz)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy+iz*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy+1+iz*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy-1+(iz+1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy+(iz+1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy+1+(iz+1)*ny]
;}
}
iy=0;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix-1+(iz-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix+(iz-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix+1+(iz-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix-1+(iz)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix+iz*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix+1+iz*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix-1+(iz+1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix+(iz+1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix+1+(iz+1)*nx]
;}
}
iz=0;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+ix-1+(iy-1)*nx]
-xv[nlocal+0+1+nx+1+ny+ix+(iy-1)*nx]
-xv[nlocal+0+1+nx+1+ny+ix+1+(iy-1)*nx]
-xv[nlocal+0+1+nx+1+ny+ix-1+(iy)*nx]
-xv[nlocal+0+1+nx+1+ny+ix+iy*nx]
-xv[nlocal+0+1+nx+1+ny+ix+1+iy*nx]
-xv[nlocal+0+1+nx+1+ny+ix-1+(iy+1)*nx]
-xv[nlocal+0+1+nx+1+ny+ix+(iy+1)*nx]
-xv[nlocal+0+1+nx+1+ny+ix+1+(iy+1)*nx]
;}
}
}//ipz < npz - 1
          }//ipz > 0 
        else
{
            if(ipz < npz - 1)
{
ix = 0;
iy = 0;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+1]
;
ix = nx-1;
iy = 0;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx]
;
iy=0;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+ix-1]
-xv[nlocal+0+nz+ix]
-xv[nlocal+0+nz+ix+1]
-xv[nlocal+0+nz+nx+ix-1]
-xv[nlocal+0+nz+nx+ix]
-xv[nlocal+0+nz+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+nz+ix+(nz-2)*nx]
-xv[nlocal+0+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+nz+ix+(nz-1)*nx]
-xv[nlocal+0+nz+ix+(nz-1)*nx+1]
;}
iy=0;
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+ix-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+ix]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+ix+1]
;}
iz=nz-1;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx+ix-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx+ix]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix+(ny-2)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix+(ny-1)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix+(ny-1)*nx+1]
;}
;
ix=0;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+iy-1]
-xv[nlocal+0+nz+nx*nz+nz+iy]
-xv[nlocal+0+nz+nx*nz+nz+iy+1]
-xv[nlocal+0+nz+nx*nz+nz+ny+iy-1]
-xv[nlocal+0+nz+nx*nz+nz+ny+iy]
-xv[nlocal+0+nz+nx*nz+nz+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+nz+nx*nz+nz+iy+(nz-2)*ny]
-xv[nlocal+0+nz+nx*nz+nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+nz+nx*nz+nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+nz+nx*nz+nz+iy+(nz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+iy+(nz-1)*ny+1]
;}
ix=0;
iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+iy-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+iy]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+iy+1]
;}
iz=nz-1;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+(iy-1)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+(iy-1)*nx+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+(iy)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+(iy-1)*nx+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+(iy+1)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+(iy)*nx+nx-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+(iy)*nx+nx-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+(iy+1)*nx+nx-1]
;}
ix=nx-1;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny+iy-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny+iy]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(nz-2)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(nz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(nz-1)*ny+1]
;}
ix=nx-1;
iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx*ny+iy-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx*ny+iy]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx*ny+iy+1]
;}
;
ix=0;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iz-1]
-xv[nlocal+0+iz]
-xv[nlocal+0+iz+1]
;}
ix=0;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+(iz-1)*ny+1]
-xv[nlocal+0+nz+nx*nz+nz+(iz)*ny]
-xv[nlocal+0+nz+nx*nz+nz+(iz-1)*ny+1]
-xv[nlocal+0+nz+nx*nz+nz+(iz+1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+nz+nx*nz+nz+(iz)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+(iz)*ny+ny-1]
-xv[nlocal+0+nz+nx*nz+nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+(iz+1)*ny+ny-1]
;}
iy=0;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+(iz-1)*nx]
-xv[nlocal+0+nz+(iz-1)*nx+1]
-xv[nlocal+0+nz+(iz)*nx]
-xv[nlocal+0+nz+(iz-1)*nx+1]
-xv[nlocal+0+nz+(iz+1)*nx]
-xv[nlocal+0+nz+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+nz+(iz)*nx+nx-2]
-xv[nlocal+0+nz+(iz)*nx+nx-1]
-xv[nlocal+0+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+nz+(iz+1)*nx+nx-1]
;}
ix=nx-1;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+iz-1]
-xv[nlocal+0+nz+nx*nz+iz]
-xv[nlocal+0+nz+nx*nz+iz+1]
;}
ix=nx-1;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz-1)*ny+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz-1)*ny+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz+1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz)*ny+ny-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz+1)*ny+ny-1]
;}
;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+iy-1+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+iy+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+iy+1+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+iy-1+(iz)*ny]
-xv[nlocal+0+nz+nx*nz+nz+iy+iz*ny]
-xv[nlocal+0+nz+nx*nz+nz+iy+1+iz*ny]
-xv[nlocal+0+nz+nx*nz+nz+iy-1+(iz+1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+iy+(iz+1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+iy+1+(iz+1)*ny]
;}
}
ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy-1+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+1+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy-1+(iz)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+iz*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+1+iz*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy-1+(iz+1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(iz+1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+1+(iz+1)*ny]
;}
}
iy=0;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+ix-1+(iz-1)*nx]
-xv[nlocal+0+nz+ix+(iz-1)*nx]
-xv[nlocal+0+nz+ix+1+(iz-1)*nx]
-xv[nlocal+0+nz+ix-1+(iz)*nx]
-xv[nlocal+0+nz+ix+iz*nx]
-xv[nlocal+0+nz+ix+1+iz*nx]
-xv[nlocal+0+nz+ix-1+(iz+1)*nx]
-xv[nlocal+0+nz+ix+(iz+1)*nx]
-xv[nlocal+0+nz+ix+1+(iz+1)*nx]
;}
}
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix-1+(iy-1)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix+(iy-1)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix+1+(iy-1)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix-1+(iy)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix+iy*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix+1+iy*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix-1+(iy+1)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix+(iy+1)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix+1+(iy+1)*nx]
;}
}
}//ipz < npz - 1
            else
{
iy=0;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+ix-1]
-xv[nlocal+0+nz+ix]
-xv[nlocal+0+nz+ix+1]
-xv[nlocal+0+nz+nx+ix-1]
-xv[nlocal+0+nz+nx+ix]
-xv[nlocal+0+nz+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+nz+ix+(nz-2)*nx]
-xv[nlocal+0+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+nz+ix+(nz-1)*nx]
-xv[nlocal+0+nz+ix+(nz-1)*nx+1]
;}
;
ix=0;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+iy-1]
-xv[nlocal+0+nz+nx*nz+nz+iy]
-xv[nlocal+0+nz+nx*nz+nz+iy+1]
-xv[nlocal+0+nz+nx*nz+nz+ny+iy-1]
-xv[nlocal+0+nz+nx*nz+nz+ny+iy]
-xv[nlocal+0+nz+nx*nz+nz+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+nz+nx*nz+nz+iy+(nz-2)*ny]
-xv[nlocal+0+nz+nx*nz+nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+nz+nx*nz+nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+nz+nx*nz+nz+iy+(nz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+iy+(nz-1)*ny+1]
;}
ix=nx-1;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny+iy-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny+iy]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(nz-2)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(nz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(nz-1)*ny+1]
;}
;
ix=0;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iz-1]
-xv[nlocal+0+iz]
-xv[nlocal+0+iz+1]
;}
ix=0;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+(iz-1)*ny+1]
-xv[nlocal+0+nz+nx*nz+nz+(iz)*ny]
-xv[nlocal+0+nz+nx*nz+nz+(iz-1)*ny+1]
-xv[nlocal+0+nz+nx*nz+nz+(iz+1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+nz+nx*nz+nz+(iz)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+(iz)*ny+ny-1]
-xv[nlocal+0+nz+nx*nz+nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+(iz+1)*ny+ny-1]
;}
iy=0;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+(iz-1)*nx]
-xv[nlocal+0+nz+(iz-1)*nx+1]
-xv[nlocal+0+nz+(iz)*nx]
-xv[nlocal+0+nz+(iz-1)*nx+1]
-xv[nlocal+0+nz+(iz+1)*nx]
-xv[nlocal+0+nz+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+nz+(iz)*nx+nx-2]
-xv[nlocal+0+nz+(iz)*nx+nx-1]
-xv[nlocal+0+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+nz+(iz+1)*nx+nx-1]
;}
ix=nx-1;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+iz-1]
-xv[nlocal+0+nz+nx*nz+iz]
-xv[nlocal+0+nz+nx*nz+iz+1]
;}
ix=nx-1;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz-1)*ny+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz-1)*ny+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz+1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz)*ny+ny-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz+1)*ny+ny-1]
;}
;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+iy-1+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+iy+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+iy+1+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+iy-1+(iz)*ny]
-xv[nlocal+0+nz+nx*nz+nz+iy+iz*ny]
-xv[nlocal+0+nz+nx*nz+nz+iy+1+iz*ny]
-xv[nlocal+0+nz+nx*nz+nz+iy-1+(iz+1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+iy+(iz+1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+iy+1+(iz+1)*ny]
;}
}
ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy-1+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+1+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy-1+(iz)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+iz*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+1+iz*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy-1+(iz+1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(iz+1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+1+(iz+1)*ny]
;}
}
iy=0;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+ix-1+(iz-1)*nx]
-xv[nlocal+0+nz+ix+(iz-1)*nx]
-xv[nlocal+0+nz+ix+1+(iz-1)*nx]
-xv[nlocal+0+nz+ix-1+(iz)*nx]
-xv[nlocal+0+nz+ix+iz*nx]
-xv[nlocal+0+nz+ix+1+iz*nx]
-xv[nlocal+0+nz+ix-1+(iz+1)*nx]
-xv[nlocal+0+nz+ix+(iz+1)*nx]
-xv[nlocal+0+nz+ix+1+(iz+1)*nx]
;}
}
}//ipz < npz - 1
          }//ipz > 0 
        }//ipy < npy - 1 
      }//ipy > 0 
    else
{
      if(ipy < npy - 1)
{
        if(ipz > 0)
{
            if(ipz < npz - 1)
{
ix = 0;
iy = ny-1;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx*ny+ny]
-xv[nlocal+0+ny+nx*ny+ny+1]
-xv[nlocal+0+ny+nx*ny+ny+1+1]
;
ix = 0;
iy = ny-1;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+ny+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+ny+1+1]
;
ix = nx-1;
iy = ny-1;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx*ny+ny+1+nx-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx]
;
ix = nx-1;
iy = ny-1;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+ny+1+nx-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+ny+1+nx-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+ny+1+nx]
;
iz=0;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+ix-1]
-xv[nlocal+0+ny+ix]
-xv[nlocal+0+ny+ix+1]
-xv[nlocal+0+ny+nx+ix-1]
-xv[nlocal+0+ny+nx+ix]
-xv[nlocal+0+ny+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+ny+ix+(ny-2)*nx]
-xv[nlocal+0+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+ny+ix+(ny-1)*nx]
-xv[nlocal+0+ny+ix+(ny-1)*nx+1]
;}
iz=nz-1;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx+ix-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx+ix]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix+(ny-2)*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix+(ny-1)*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix+(ny-1)*nx+1]
;}
iy=ny-1;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+ix-1]
-xv[nlocal+0+ny+nx*ny+ny+1+ix]
-xv[nlocal+0+ny+nx*ny+ny+1+ix+1]
;}
iy=ny-1;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx+ix-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx+ix]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix+(nz-2)*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix+(nz-1)*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix+(nz-1)*nx+1]
;}
iy=ny-1;
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+ny+1+ix-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+ny+1+ix]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+ny+1+ix+1]
;}
;
ix=0;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1]
-xv[nlocal+0+iy]
-xv[nlocal+0+iy+1]
;}
ix=0;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny+iy-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny+iy]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy+(nz-2)*ny-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy+(nz-2)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy+(nz-2)*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy+(nz-1)*ny-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy+(nz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy+(nz-1)*ny+1]
;}
ix=0;
iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+iy-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+iy]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+iy+1]
;}
iz=0;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+(iy-1)*nx]
-xv[nlocal+0+ny+(iy-1)*nx+1]
-xv[nlocal+0+ny+(iy)*nx]
-xv[nlocal+0+ny+(iy-1)*nx+1]
-xv[nlocal+0+ny+(iy+1)*nx]
-xv[nlocal+0+ny+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+ny+(iy)*nx+nx-2]
-xv[nlocal+0+ny+(iy)*nx+nx-1]
-xv[nlocal+0+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+ny+(iy+1)*nx+nx-1]
;}
iz=nz-1;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+(iy-1)*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+(iy-1)*nx+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+(iy)*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+(iy-1)*nx+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+(iy+1)*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+(iy)*nx+nx-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+(iy)*nx+nx-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+(iy+1)*nx+nx-1]
;}
ix=nx-1;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+iy-1]
-xv[nlocal+0+ny+nx*ny+iy]
-xv[nlocal+0+ny+nx*ny+iy+1]
;}
ix=nx-1;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny+iy-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny+iy]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy+(nz-2)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy+(nz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy+(nz-1)*ny+1]
;}
ix=nx-1;
iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+iy-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+iy]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+iy+1]
;}
;
ix=0;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+(iz-1)*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+(iz)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+(iz-1)*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+(iz+1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+(iz-1)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+(iz-1)*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+(iz)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+(iz)*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+(iz+1)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+(iz+1)*ny+ny-1]
;}
ix=0;
iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+iz-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+iz]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+iz+1]
;}
iy=ny-1;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+(iz-1)*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+(iz-1)*nx+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+(iz)*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+(iz-1)*nx+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+(iz+1)*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+(iz)*nx+nx-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+(iz)*nx+nx-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+(iz+1)*nx+nx-1]
;}
ix=nx-1;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+(iz-1)*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+(iz)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+(iz-1)*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+(iz+1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+(iz)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+(iz)*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+(iz+1)*ny+ny-1]
;}
ix=nx-1;
iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+iz-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+iz]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+iz+1]
;}
;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy-1+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy+1+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy-1+(iz)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy+iz*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy+1+iz*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy-1+(iz+1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy+(iz+1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy+1+(iz+1)*ny]
;}
}
ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy-1+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy+1+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy-1+(iz)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy+iz*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy+1+iz*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy-1+(iz+1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy+(iz+1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy+1+(iz+1)*ny]
;}
}
iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix-1+(iz-1)*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix+(iz-1)*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix+1+(iz-1)*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix-1+(iz)*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix+iz*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix+1+iz*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix-1+(iz+1)*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix+(iz+1)*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix+1+(iz+1)*nx]
;}
}
iz=0;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+ix-1+(iy-1)*nx]
-xv[nlocal+0+ny+ix+(iy-1)*nx]
-xv[nlocal+0+ny+ix+1+(iy-1)*nx]
-xv[nlocal+0+ny+ix-1+(iy)*nx]
-xv[nlocal+0+ny+ix+iy*nx]
-xv[nlocal+0+ny+ix+1+iy*nx]
-xv[nlocal+0+ny+ix-1+(iy+1)*nx]
-xv[nlocal+0+ny+ix+(iy+1)*nx]
-xv[nlocal+0+ny+ix+1+(iy+1)*nx]
;}
}
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix-1+(iy-1)*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix+(iy-1)*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix+1+(iy-1)*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix-1+(iy)*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix+iy*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix+1+iy*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix-1+(iy+1)*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix+(iy+1)*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix+1+(iy+1)*nx]
;}
}
}//ipz < npz - 1
            else
{
ix = 0;
iy = ny-1;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx*ny+ny]
-xv[nlocal+0+ny+nx*ny+ny+1]
-xv[nlocal+0+ny+nx*ny+ny+1+1]
;
ix = nx-1;
iy = ny-1;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx*ny+ny+1+nx-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx]
;
iz=0;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+ix-1]
-xv[nlocal+0+ny+ix]
-xv[nlocal+0+ny+ix+1]
-xv[nlocal+0+ny+nx+ix-1]
-xv[nlocal+0+ny+nx+ix]
-xv[nlocal+0+ny+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+ny+ix+(ny-2)*nx]
-xv[nlocal+0+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+ny+ix+(ny-1)*nx]
-xv[nlocal+0+ny+ix+(ny-1)*nx+1]
;}
iy=ny-1;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+ix-1]
-xv[nlocal+0+ny+nx*ny+ny+1+ix]
-xv[nlocal+0+ny+nx*ny+ny+1+ix+1]
;}
iy=ny-1;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx+ix-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx+ix]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix+(nz-2)*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix+(nz-1)*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix+(nz-1)*nx+1]
;}
;
ix=0;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1]
-xv[nlocal+0+iy]
-xv[nlocal+0+iy+1]
;}
ix=0;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny+iy-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny+iy]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy+(nz-2)*ny-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy+(nz-2)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy+(nz-2)*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy+(nz-1)*ny-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy+(nz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy+(nz-1)*ny+1]
;}
iz=0;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+(iy-1)*nx]
-xv[nlocal+0+ny+(iy-1)*nx+1]
-xv[nlocal+0+ny+(iy)*nx]
-xv[nlocal+0+ny+(iy-1)*nx+1]
-xv[nlocal+0+ny+(iy+1)*nx]
-xv[nlocal+0+ny+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+ny+(iy)*nx+nx-2]
-xv[nlocal+0+ny+(iy)*nx+nx-1]
-xv[nlocal+0+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+ny+(iy+1)*nx+nx-1]
;}
ix=nx-1;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+iy-1]
-xv[nlocal+0+ny+nx*ny+iy]
-xv[nlocal+0+ny+nx*ny+iy+1]
;}
ix=nx-1;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny+iy-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny+iy]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy+(nz-2)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy+(nz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy+(nz-1)*ny+1]
;}
;
ix=0;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+(iz-1)*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+(iz)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+(iz-1)*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+(iz+1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+(iz-1)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+(iz-1)*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+(iz)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+(iz)*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+(iz+1)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+(iz+1)*ny+ny-1]
;}
ix=0;
iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+iz-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+iz]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+iz+1]
;}
iy=ny-1;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+(iz-1)*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+(iz-1)*nx+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+(iz)*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+(iz-1)*nx+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+(iz+1)*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+(iz)*nx+nx-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+(iz)*nx+nx-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+(iz+1)*nx+nx-1]
;}
ix=nx-1;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+(iz-1)*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+(iz)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+(iz-1)*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+(iz+1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+(iz)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+(iz)*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+(iz+1)*ny+ny-1]
;}
ix=nx-1;
iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+iz-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+iz]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+iz+1]
;}
;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy-1+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy+1+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy-1+(iz)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy+iz*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy+1+iz*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy-1+(iz+1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy+(iz+1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy+1+(iz+1)*ny]
;}
}
ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy-1+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy+1+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy-1+(iz)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy+iz*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy+1+iz*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy-1+(iz+1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy+(iz+1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy+1+(iz+1)*ny]
;}
}
iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix-1+(iz-1)*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix+(iz-1)*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix+1+(iz-1)*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix-1+(iz)*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix+iz*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix+1+iz*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix-1+(iz+1)*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix+(iz+1)*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix+1+(iz+1)*nx]
;}
}
iz=0;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+ix-1+(iy-1)*nx]
-xv[nlocal+0+ny+ix+(iy-1)*nx]
-xv[nlocal+0+ny+ix+1+(iy-1)*nx]
-xv[nlocal+0+ny+ix-1+(iy)*nx]
-xv[nlocal+0+ny+ix+iy*nx]
-xv[nlocal+0+ny+ix+1+iy*nx]
-xv[nlocal+0+ny+ix-1+(iy+1)*nx]
-xv[nlocal+0+ny+ix+(iy+1)*nx]
-xv[nlocal+0+ny+ix+1+(iy+1)*nx]
;}
}
}//ipz < npz - 1
          }//ipz > 0 
        else
{
            if(ipz < npz - 1)
{
ix = 0;
iy = ny-1;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+ny]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+ny+1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+ny+1+1]
;
ix = nx-1;
iy = ny-1;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+ny+1+nx-2]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+ny+1+nx-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+ny+1+nx]
;
iz=nz-1;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix+1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx+ix-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx+ix]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix+(ny-2)*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix+(ny-1)*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix+(ny-1)*nx+1]
;}
iy=ny-1;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+nz+ix-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+ix]
-xv[nlocal+0+ny*nz+ny*nz+nz+ix+1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx+ix-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx+ix]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+ix+(nz-2)*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+ny*nz+ny*nz+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+ix+(nz-1)*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+ix+(nz-1)*nx+1]
;}
iy=ny-1;
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+ny+1+ix-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+ny+1+ix]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+ny+1+ix+1]
;}
;
ix=0;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1]
-xv[nlocal+0+iy]
-xv[nlocal+0+iy+1]
-xv[nlocal+0+ny+iy-1]
-xv[nlocal+0+ny+iy]
-xv[nlocal+0+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy+(nz-2)*ny-1]
-xv[nlocal+0+iy+(nz-2)*ny]
-xv[nlocal+0+iy+(nz-2)*ny+1]
-xv[nlocal+0+iy+(nz-1)*ny-1]
-xv[nlocal+0+iy+(nz-1)*ny]
-xv[nlocal+0+iy+(nz-1)*ny+1]
;}
ix=0;
iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+iy-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+iy]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+iy+1]
;}
iz=nz-1;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+(iy-1)*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+(iy-1)*nx+1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+(iy)*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+(iy-1)*nx+1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+(iy+1)*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+(iy)*nx+nx-2]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+(iy)*nx+nx-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+(iy+1)*nx+nx-1]
;}
ix=nx-1;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+iy-1]
-xv[nlocal+0+ny*nz+iy]
-xv[nlocal+0+ny*nz+iy+1]
-xv[nlocal+0+ny*nz+ny+iy-1]
-xv[nlocal+0+ny*nz+ny+iy]
-xv[nlocal+0+ny*nz+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+ny*nz+iy+(nz-2)*ny]
-xv[nlocal+0+ny*nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+ny*nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+ny*nz+iy+(nz-1)*ny]
-xv[nlocal+0+ny*nz+iy+(nz-1)*ny+1]
;}
ix=nx-1;
iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+iy-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+iy]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+iy+1]
;}
;
ix=0;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*ny]
-xv[nlocal+0+(iz-1)*ny+1]
-xv[nlocal+0+(iz)*ny]
-xv[nlocal+0+(iz-1)*ny+1]
-xv[nlocal+0+(iz+1)*ny]
-xv[nlocal+0+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*ny+ny-2]
-xv[nlocal+0+(iz-1)*ny+ny-1]
-xv[nlocal+0+(iz)*ny+ny-2]
-xv[nlocal+0+(iz)*ny+ny-1]
-xv[nlocal+0+(iz+1)*ny+ny-2]
-xv[nlocal+0+(iz+1)*ny+ny-1]
;}
ix=0;
iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+iz-1]
-xv[nlocal+0+ny*nz+ny*nz+iz]
-xv[nlocal+0+ny*nz+ny*nz+iz+1]
;}
iy=ny-1;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+nz+(iz-1)*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+(iz-1)*nx+1]
-xv[nlocal+0+ny*nz+ny*nz+nz+(iz)*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+(iz-1)*nx+1]
-xv[nlocal+0+ny*nz+ny*nz+nz+(iz+1)*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+ny*nz+ny*nz+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+(iz)*nx+nx-2]
-xv[nlocal+0+ny*nz+ny*nz+nz+(iz)*nx+nx-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+ny*nz+ny*nz+nz+(iz+1)*nx+nx-1]
;}
ix=nx-1;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+(iz-1)*ny]
-xv[nlocal+0+ny*nz+(iz-1)*ny+1]
-xv[nlocal+0+ny*nz+(iz)*ny]
-xv[nlocal+0+ny*nz+(iz-1)*ny+1]
-xv[nlocal+0+ny*nz+(iz+1)*ny]
-xv[nlocal+0+ny*nz+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+ny*nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+ny*nz+(iz)*ny+ny-2]
-xv[nlocal+0+ny*nz+(iz)*ny+ny-1]
-xv[nlocal+0+ny*nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+ny*nz+(iz+1)*ny+ny-1]
;}
ix=nx-1;
iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+iz-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+iz]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+iz+1]
;}
;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1+(iz-1)*ny]
-xv[nlocal+0+iy+(iz-1)*ny]
-xv[nlocal+0+iy+1+(iz-1)*ny]
-xv[nlocal+0+iy-1+(iz)*ny]
-xv[nlocal+0+iy+iz*ny]
-xv[nlocal+0+iy+1+iz*ny]
-xv[nlocal+0+iy-1+(iz+1)*ny]
-xv[nlocal+0+iy+(iz+1)*ny]
-xv[nlocal+0+iy+1+(iz+1)*ny]
;}
}
ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+iy-1+(iz-1)*ny]
-xv[nlocal+0+ny*nz+iy+(iz-1)*ny]
-xv[nlocal+0+ny*nz+iy+1+(iz-1)*ny]
-xv[nlocal+0+ny*nz+iy-1+(iz)*ny]
-xv[nlocal+0+ny*nz+iy+iz*ny]
-xv[nlocal+0+ny*nz+iy+1+iz*ny]
-xv[nlocal+0+ny*nz+iy-1+(iz+1)*ny]
-xv[nlocal+0+ny*nz+iy+(iz+1)*ny]
-xv[nlocal+0+ny*nz+iy+1+(iz+1)*ny]
;}
}
iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+nz+ix-1+(iz-1)*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+ix+(iz-1)*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+ix+1+(iz-1)*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+ix-1+(iz)*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+ix+iz*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+ix+1+iz*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+ix-1+(iz+1)*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+ix+(iz+1)*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+ix+1+(iz+1)*nx]
;}
}
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix-1+(iy-1)*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix+(iy-1)*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix+1+(iy-1)*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix-1+(iy)*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix+iy*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix+1+iy*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix-1+(iy+1)*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix+(iy+1)*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix+1+(iy+1)*nx]
;}
}
}//ipz < npz - 1
            else
{
iy=ny-1;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+nz+ix-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+ix]
-xv[nlocal+0+ny*nz+ny*nz+nz+ix+1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx+ix-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx+ix]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+ix+(nz-2)*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+ny*nz+ny*nz+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+ix+(nz-1)*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+ix+(nz-1)*nx+1]
;}
;
ix=0;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1]
-xv[nlocal+0+iy]
-xv[nlocal+0+iy+1]
-xv[nlocal+0+ny+iy-1]
-xv[nlocal+0+ny+iy]
-xv[nlocal+0+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy+(nz-2)*ny-1]
-xv[nlocal+0+iy+(nz-2)*ny]
-xv[nlocal+0+iy+(nz-2)*ny+1]
-xv[nlocal+0+iy+(nz-1)*ny-1]
-xv[nlocal+0+iy+(nz-1)*ny]
-xv[nlocal+0+iy+(nz-1)*ny+1]
;}
ix=nx-1;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+iy-1]
-xv[nlocal+0+ny*nz+iy]
-xv[nlocal+0+ny*nz+iy+1]
-xv[nlocal+0+ny*nz+ny+iy-1]
-xv[nlocal+0+ny*nz+ny+iy]
-xv[nlocal+0+ny*nz+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+ny*nz+iy+(nz-2)*ny]
-xv[nlocal+0+ny*nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+ny*nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+ny*nz+iy+(nz-1)*ny]
-xv[nlocal+0+ny*nz+iy+(nz-1)*ny+1]
;}
;
ix=0;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*ny]
-xv[nlocal+0+(iz-1)*ny+1]
-xv[nlocal+0+(iz)*ny]
-xv[nlocal+0+(iz-1)*ny+1]
-xv[nlocal+0+(iz+1)*ny]
-xv[nlocal+0+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*ny+ny-2]
-xv[nlocal+0+(iz-1)*ny+ny-1]
-xv[nlocal+0+(iz)*ny+ny-2]
-xv[nlocal+0+(iz)*ny+ny-1]
-xv[nlocal+0+(iz+1)*ny+ny-2]
-xv[nlocal+0+(iz+1)*ny+ny-1]
;}
ix=0;
iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+iz-1]
-xv[nlocal+0+ny*nz+ny*nz+iz]
-xv[nlocal+0+ny*nz+ny*nz+iz+1]
;}
iy=ny-1;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+nz+(iz-1)*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+(iz-1)*nx+1]
-xv[nlocal+0+ny*nz+ny*nz+nz+(iz)*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+(iz-1)*nx+1]
-xv[nlocal+0+ny*nz+ny*nz+nz+(iz+1)*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+ny*nz+ny*nz+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+(iz)*nx+nx-2]
-xv[nlocal+0+ny*nz+ny*nz+nz+(iz)*nx+nx-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+ny*nz+ny*nz+nz+(iz+1)*nx+nx-1]
;}
ix=nx-1;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+(iz-1)*ny]
-xv[nlocal+0+ny*nz+(iz-1)*ny+1]
-xv[nlocal+0+ny*nz+(iz)*ny]
-xv[nlocal+0+ny*nz+(iz-1)*ny+1]
-xv[nlocal+0+ny*nz+(iz+1)*ny]
-xv[nlocal+0+ny*nz+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+ny*nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+ny*nz+(iz)*ny+ny-2]
-xv[nlocal+0+ny*nz+(iz)*ny+ny-1]
-xv[nlocal+0+ny*nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+ny*nz+(iz+1)*ny+ny-1]
;}
ix=nx-1;
iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+iz-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+iz]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+iz+1]
;}
;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1+(iz-1)*ny]
-xv[nlocal+0+iy+(iz-1)*ny]
-xv[nlocal+0+iy+1+(iz-1)*ny]
-xv[nlocal+0+iy-1+(iz)*ny]
-xv[nlocal+0+iy+iz*ny]
-xv[nlocal+0+iy+1+iz*ny]
-xv[nlocal+0+iy-1+(iz+1)*ny]
-xv[nlocal+0+iy+(iz+1)*ny]
-xv[nlocal+0+iy+1+(iz+1)*ny]
;}
}
ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+iy-1+(iz-1)*ny]
-xv[nlocal+0+ny*nz+iy+(iz-1)*ny]
-xv[nlocal+0+ny*nz+iy+1+(iz-1)*ny]
-xv[nlocal+0+ny*nz+iy-1+(iz)*ny]
-xv[nlocal+0+ny*nz+iy+iz*ny]
-xv[nlocal+0+ny*nz+iy+1+iz*ny]
-xv[nlocal+0+ny*nz+iy-1+(iz+1)*ny]
-xv[nlocal+0+ny*nz+iy+(iz+1)*ny]
-xv[nlocal+0+ny*nz+iy+1+(iz+1)*ny]
;}
}
iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+nz+ix-1+(iz-1)*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+ix+(iz-1)*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+ix+1+(iz-1)*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+ix-1+(iz)*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+ix+iz*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+ix+1+iz*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+ix-1+(iz+1)*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+ix+(iz+1)*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+ix+1+(iz+1)*nx]
;}
}
}//ipz < npz - 1
          }//ipz > 0 
        }//ipy < npy - 1 
      else
{
        if(ipz > 0)
{
            if(ipz < npz - 1)
{
iz=0;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+ix-1]
-xv[nlocal+0+ny+ix]
-xv[nlocal+0+ny+ix+1]
-xv[nlocal+0+ny+nx+ix-1]
-xv[nlocal+0+ny+nx+ix]
-xv[nlocal+0+ny+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+ny+ix+(ny-2)*nx]
-xv[nlocal+0+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+ny+ix+(ny-1)*nx]
-xv[nlocal+0+ny+ix+(ny-1)*nx+1]
;}
iz=nz-1;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+ix-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+ix]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+ix+1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+nx+ix-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+nx+ix]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+ix+(ny-2)*nx]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+ix+(ny-1)*nx]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+ix+(ny-1)*nx+1]
;}
;
ix=0;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1]
-xv[nlocal+0+iy]
-xv[nlocal+0+iy+1]
;}
ix=0;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+iy-1]
-xv[nlocal+0+ny+nx*ny+ny+iy]
-xv[nlocal+0+ny+nx*ny+ny+iy+1]
-xv[nlocal+0+ny+nx*ny+ny+ny+iy-1]
-xv[nlocal+0+ny+nx*ny+ny+ny+iy]
-xv[nlocal+0+ny+nx*ny+ny+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+iy+(nz-2)*ny-1]
-xv[nlocal+0+ny+nx*ny+ny+iy+(nz-2)*ny]
-xv[nlocal+0+ny+nx*ny+ny+iy+(nz-2)*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+iy+(nz-1)*ny-1]
-xv[nlocal+0+ny+nx*ny+ny+iy+(nz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+iy+(nz-1)*ny+1]
;}
ix=0;
iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+iy-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+iy]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+iy+1]
;}
iz=0;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+(iy-1)*nx]
-xv[nlocal+0+ny+(iy-1)*nx+1]
-xv[nlocal+0+ny+(iy)*nx]
-xv[nlocal+0+ny+(iy-1)*nx+1]
-xv[nlocal+0+ny+(iy+1)*nx]
-xv[nlocal+0+ny+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+ny+(iy)*nx+nx-2]
-xv[nlocal+0+ny+(iy)*nx+nx-1]
-xv[nlocal+0+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+ny+(iy+1)*nx+nx-1]
;}
iz=nz-1;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+(iy-1)*nx]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+(iy-1)*nx+1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+(iy)*nx]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+(iy-1)*nx+1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+(iy+1)*nx]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+(iy)*nx+nx-2]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+(iy)*nx+nx-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+(iy+1)*nx+nx-1]
;}
ix=nx-1;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+iy-1]
-xv[nlocal+0+ny+nx*ny+iy]
-xv[nlocal+0+ny+nx*ny+iy+1]
;}
ix=nx-1;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy+1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny+iy-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny+iy]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy+(nz-2)*ny]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy+(nz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy+(nz-1)*ny+1]
;}
ix=nx-1;
iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+nx*ny+iy-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+nx*ny+iy]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+nx*ny+iy+1]
;}
;
ix=0;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+(iz-1)*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+(iz)*ny]
-xv[nlocal+0+ny+nx*ny+ny+(iz-1)*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+(iz+1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+(iz-1)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+(iz-1)*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+(iz)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+(iz)*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+(iz+1)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+(iz+1)*ny+ny-1]
;}
ix=nx-1;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+(iz-1)*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+(iz)*ny]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+(iz-1)*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+(iz+1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+(iz)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+(iz)*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+(iz+1)*ny+ny-1]
;}
;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+iy-1+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+iy+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+iy+1+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+iy-1+(iz)*ny]
-xv[nlocal+0+ny+nx*ny+ny+iy+iz*ny]
-xv[nlocal+0+ny+nx*ny+ny+iy+1+iz*ny]
-xv[nlocal+0+ny+nx*ny+ny+iy-1+(iz+1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+iy+(iz+1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+iy+1+(iz+1)*ny]
;}
}
ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy-1+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy+1+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy-1+(iz)*ny]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy+iz*ny]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy+1+iz*ny]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy-1+(iz+1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy+(iz+1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy+1+(iz+1)*ny]
;}
}
iz=0;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+ix-1+(iy-1)*nx]
-xv[nlocal+0+ny+ix+(iy-1)*nx]
-xv[nlocal+0+ny+ix+1+(iy-1)*nx]
-xv[nlocal+0+ny+ix-1+(iy)*nx]
-xv[nlocal+0+ny+ix+iy*nx]
-xv[nlocal+0+ny+ix+1+iy*nx]
-xv[nlocal+0+ny+ix-1+(iy+1)*nx]
-xv[nlocal+0+ny+ix+(iy+1)*nx]
-xv[nlocal+0+ny+ix+1+(iy+1)*nx]
;}
}
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+ix-1+(iy-1)*nx]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+ix+(iy-1)*nx]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+ix+1+(iy-1)*nx]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+ix-1+(iy)*nx]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+ix+iy*nx]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+ix+1+iy*nx]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+ix-1+(iy+1)*nx]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+ix+(iy+1)*nx]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+ix+1+(iy+1)*nx]
;}
}
}//ipz < npz - 1
            else
{
iz=0;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+ix-1]
-xv[nlocal+0+ny+ix]
-xv[nlocal+0+ny+ix+1]
-xv[nlocal+0+ny+nx+ix-1]
-xv[nlocal+0+ny+nx+ix]
-xv[nlocal+0+ny+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+ny+ix+(ny-2)*nx]
-xv[nlocal+0+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+ny+ix+(ny-1)*nx]
-xv[nlocal+0+ny+ix+(ny-1)*nx+1]
;}
;
ix=0;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1]
-xv[nlocal+0+iy]
-xv[nlocal+0+iy+1]
;}
ix=0;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+iy-1]
-xv[nlocal+0+ny+nx*ny+ny+iy]
-xv[nlocal+0+ny+nx*ny+ny+iy+1]
-xv[nlocal+0+ny+nx*ny+ny+ny+iy-1]
-xv[nlocal+0+ny+nx*ny+ny+ny+iy]
-xv[nlocal+0+ny+nx*ny+ny+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+iy+(nz-2)*ny-1]
-xv[nlocal+0+ny+nx*ny+ny+iy+(nz-2)*ny]
-xv[nlocal+0+ny+nx*ny+ny+iy+(nz-2)*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+iy+(nz-1)*ny-1]
-xv[nlocal+0+ny+nx*ny+ny+iy+(nz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+iy+(nz-1)*ny+1]
;}
iz=0;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+(iy-1)*nx]
-xv[nlocal+0+ny+(iy-1)*nx+1]
-xv[nlocal+0+ny+(iy)*nx]
-xv[nlocal+0+ny+(iy-1)*nx+1]
-xv[nlocal+0+ny+(iy+1)*nx]
-xv[nlocal+0+ny+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+ny+(iy)*nx+nx-2]
-xv[nlocal+0+ny+(iy)*nx+nx-1]
-xv[nlocal+0+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+ny+(iy+1)*nx+nx-1]
;}
ix=nx-1;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+iy-1]
-xv[nlocal+0+ny+nx*ny+iy]
-xv[nlocal+0+ny+nx*ny+iy+1]
;}
ix=nx-1;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy+1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny+iy-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny+iy]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy+(nz-2)*ny]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy+(nz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy+(nz-1)*ny+1]
;}
;
ix=0;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+(iz-1)*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+(iz)*ny]
-xv[nlocal+0+ny+nx*ny+ny+(iz-1)*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+(iz+1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+(iz-1)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+(iz-1)*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+(iz)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+(iz)*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+(iz+1)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+(iz+1)*ny+ny-1]
;}
ix=nx-1;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+(iz-1)*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+(iz)*ny]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+(iz-1)*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+(iz+1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+(iz)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+(iz)*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+(iz+1)*ny+ny-1]
;}
;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+iy-1+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+iy+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+iy+1+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+iy-1+(iz)*ny]
-xv[nlocal+0+ny+nx*ny+ny+iy+iz*ny]
-xv[nlocal+0+ny+nx*ny+ny+iy+1+iz*ny]
-xv[nlocal+0+ny+nx*ny+ny+iy-1+(iz+1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+iy+(iz+1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+iy+1+(iz+1)*ny]
;}
}
ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy-1+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy+1+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy-1+(iz)*ny]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy+iz*ny]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy+1+iz*ny]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy-1+(iz+1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy+(iz+1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy+1+(iz+1)*ny]
;}
}
iz=0;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+ix-1+(iy-1)*nx]
-xv[nlocal+0+ny+ix+(iy-1)*nx]
-xv[nlocal+0+ny+ix+1+(iy-1)*nx]
-xv[nlocal+0+ny+ix-1+(iy)*nx]
-xv[nlocal+0+ny+ix+iy*nx]
-xv[nlocal+0+ny+ix+1+iy*nx]
-xv[nlocal+0+ny+ix-1+(iy+1)*nx]
-xv[nlocal+0+ny+ix+(iy+1)*nx]
-xv[nlocal+0+ny+ix+1+(iy+1)*nx]
;}
}
}//ipz < npz - 1
          }//ipz > 0 
        else
{
            if(ipz < npz - 1)
{
iz=nz-1;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+ny+ix-1]
-xv[nlocal+0+ny*nz+ny*nz+ny+ix]
-xv[nlocal+0+ny*nz+ny*nz+ny+ix+1]
-xv[nlocal+0+ny*nz+ny*nz+ny+nx+ix-1]
-xv[nlocal+0+ny*nz+ny*nz+ny+nx+ix]
-xv[nlocal+0+ny*nz+ny*nz+ny+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+ny*nz+ny*nz+ny+ix+(ny-2)*nx]
-xv[nlocal+0+ny*nz+ny*nz+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+ny*nz+ny*nz+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+ny*nz+ny*nz+ny+ix+(ny-1)*nx]
-xv[nlocal+0+ny*nz+ny*nz+ny+ix+(ny-1)*nx+1]
;}
;
ix=0;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1]
-xv[nlocal+0+iy]
-xv[nlocal+0+iy+1]
-xv[nlocal+0+ny+iy-1]
-xv[nlocal+0+ny+iy]
-xv[nlocal+0+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy+(nz-2)*ny-1]
-xv[nlocal+0+iy+(nz-2)*ny]
-xv[nlocal+0+iy+(nz-2)*ny+1]
-xv[nlocal+0+iy+(nz-1)*ny-1]
-xv[nlocal+0+iy+(nz-1)*ny]
-xv[nlocal+0+iy+(nz-1)*ny+1]
;}
ix=0;
iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+iy-1]
-xv[nlocal+0+ny*nz+ny*nz+iy]
-xv[nlocal+0+ny*nz+ny*nz+iy+1]
;}
iz=nz-1;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+ny+(iy-1)*nx]
-xv[nlocal+0+ny*nz+ny*nz+ny+(iy-1)*nx+1]
-xv[nlocal+0+ny*nz+ny*nz+ny+(iy)*nx]
-xv[nlocal+0+ny*nz+ny*nz+ny+(iy-1)*nx+1]
-xv[nlocal+0+ny*nz+ny*nz+ny+(iy+1)*nx]
-xv[nlocal+0+ny*nz+ny*nz+ny+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+ny*nz+ny*nz+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+ny*nz+ny*nz+ny+(iy)*nx+nx-2]
-xv[nlocal+0+ny*nz+ny*nz+ny+(iy)*nx+nx-1]
-xv[nlocal+0+ny*nz+ny*nz+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+ny*nz+ny*nz+ny+(iy+1)*nx+nx-1]
;}
ix=nx-1;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+iy-1]
-xv[nlocal+0+ny*nz+iy]
-xv[nlocal+0+ny*nz+iy+1]
-xv[nlocal+0+ny*nz+ny+iy-1]
-xv[nlocal+0+ny*nz+ny+iy]
-xv[nlocal+0+ny*nz+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+ny*nz+iy+(nz-2)*ny]
-xv[nlocal+0+ny*nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+ny*nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+ny*nz+iy+(nz-1)*ny]
-xv[nlocal+0+ny*nz+iy+(nz-1)*ny+1]
;}
ix=nx-1;
iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+ny+nx*ny+iy-1]
-xv[nlocal+0+ny*nz+ny*nz+ny+nx*ny+iy]
-xv[nlocal+0+ny*nz+ny*nz+ny+nx*ny+iy+1]
;}
;
ix=0;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*ny]
-xv[nlocal+0+(iz-1)*ny+1]
-xv[nlocal+0+(iz)*ny]
-xv[nlocal+0+(iz-1)*ny+1]
-xv[nlocal+0+(iz+1)*ny]
-xv[nlocal+0+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*ny+ny-2]
-xv[nlocal+0+(iz-1)*ny+ny-1]
-xv[nlocal+0+(iz)*ny+ny-2]
-xv[nlocal+0+(iz)*ny+ny-1]
-xv[nlocal+0+(iz+1)*ny+ny-2]
-xv[nlocal+0+(iz+1)*ny+ny-1]
;}
ix=nx-1;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+(iz-1)*ny]
-xv[nlocal+0+ny*nz+(iz-1)*ny+1]
-xv[nlocal+0+ny*nz+(iz)*ny]
-xv[nlocal+0+ny*nz+(iz-1)*ny+1]
-xv[nlocal+0+ny*nz+(iz+1)*ny]
-xv[nlocal+0+ny*nz+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+ny*nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+ny*nz+(iz)*ny+ny-2]
-xv[nlocal+0+ny*nz+(iz)*ny+ny-1]
-xv[nlocal+0+ny*nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+ny*nz+(iz+1)*ny+ny-1]
;}
;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1+(iz-1)*ny]
-xv[nlocal+0+iy+(iz-1)*ny]
-xv[nlocal+0+iy+1+(iz-1)*ny]
-xv[nlocal+0+iy-1+(iz)*ny]
-xv[nlocal+0+iy+iz*ny]
-xv[nlocal+0+iy+1+iz*ny]
-xv[nlocal+0+iy-1+(iz+1)*ny]
-xv[nlocal+0+iy+(iz+1)*ny]
-xv[nlocal+0+iy+1+(iz+1)*ny]
;}
}
ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+iy-1+(iz-1)*ny]
-xv[nlocal+0+ny*nz+iy+(iz-1)*ny]
-xv[nlocal+0+ny*nz+iy+1+(iz-1)*ny]
-xv[nlocal+0+ny*nz+iy-1+(iz)*ny]
-xv[nlocal+0+ny*nz+iy+iz*ny]
-xv[nlocal+0+ny*nz+iy+1+iz*ny]
-xv[nlocal+0+ny*nz+iy-1+(iz+1)*ny]
-xv[nlocal+0+ny*nz+iy+(iz+1)*ny]
-xv[nlocal+0+ny*nz+iy+1+(iz+1)*ny]
;}
}
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+ny+ix-1+(iy-1)*nx]
-xv[nlocal+0+ny*nz+ny*nz+ny+ix+(iy-1)*nx]
-xv[nlocal+0+ny*nz+ny*nz+ny+ix+1+(iy-1)*nx]
-xv[nlocal+0+ny*nz+ny*nz+ny+ix-1+(iy)*nx]
-xv[nlocal+0+ny*nz+ny*nz+ny+ix+iy*nx]
-xv[nlocal+0+ny*nz+ny*nz+ny+ix+1+iy*nx]
-xv[nlocal+0+ny*nz+ny*nz+ny+ix-1+(iy+1)*nx]
-xv[nlocal+0+ny*nz+ny*nz+ny+ix+(iy+1)*nx]
-xv[nlocal+0+ny*nz+ny*nz+ny+ix+1+(iy+1)*nx]
;}
}
}//ipz < npz - 1
            else
{
;
ix=0;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1]
-xv[nlocal+0+iy]
-xv[nlocal+0+iy+1]
-xv[nlocal+0+ny+iy-1]
-xv[nlocal+0+ny+iy]
-xv[nlocal+0+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy+(nz-2)*ny-1]
-xv[nlocal+0+iy+(nz-2)*ny]
-xv[nlocal+0+iy+(nz-2)*ny+1]
-xv[nlocal+0+iy+(nz-1)*ny-1]
-xv[nlocal+0+iy+(nz-1)*ny]
-xv[nlocal+0+iy+(nz-1)*ny+1]
;}
ix=nx-1;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+iy-1]
-xv[nlocal+0+ny*nz+iy]
-xv[nlocal+0+ny*nz+iy+1]
-xv[nlocal+0+ny*nz+ny+iy-1]
-xv[nlocal+0+ny*nz+ny+iy]
-xv[nlocal+0+ny*nz+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+ny*nz+iy+(nz-2)*ny]
-xv[nlocal+0+ny*nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+ny*nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+ny*nz+iy+(nz-1)*ny]
-xv[nlocal+0+ny*nz+iy+(nz-1)*ny+1]
;}
;
ix=0;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*ny]
-xv[nlocal+0+(iz-1)*ny+1]
-xv[nlocal+0+(iz)*ny]
-xv[nlocal+0+(iz-1)*ny+1]
-xv[nlocal+0+(iz+1)*ny]
-xv[nlocal+0+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*ny+ny-2]
-xv[nlocal+0+(iz-1)*ny+ny-1]
-xv[nlocal+0+(iz)*ny+ny-2]
-xv[nlocal+0+(iz)*ny+ny-1]
-xv[nlocal+0+(iz+1)*ny+ny-2]
-xv[nlocal+0+(iz+1)*ny+ny-1]
;}
ix=nx-1;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+(iz-1)*ny]
-xv[nlocal+0+ny*nz+(iz-1)*ny+1]
-xv[nlocal+0+ny*nz+(iz)*ny]
-xv[nlocal+0+ny*nz+(iz-1)*ny+1]
-xv[nlocal+0+ny*nz+(iz+1)*ny]
-xv[nlocal+0+ny*nz+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+ny*nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+ny*nz+(iz)*ny+ny-2]
-xv[nlocal+0+ny*nz+(iz)*ny+ny-1]
-xv[nlocal+0+ny*nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+ny*nz+(iz+1)*ny+ny-1]
;}
;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1+(iz-1)*ny]
-xv[nlocal+0+iy+(iz-1)*ny]
-xv[nlocal+0+iy+1+(iz-1)*ny]
-xv[nlocal+0+iy-1+(iz)*ny]
-xv[nlocal+0+iy+iz*ny]
-xv[nlocal+0+iy+1+iz*ny]
-xv[nlocal+0+iy-1+(iz+1)*ny]
-xv[nlocal+0+iy+(iz+1)*ny]
-xv[nlocal+0+iy+1+(iz+1)*ny]
;}
}
ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+iy-1+(iz-1)*ny]
-xv[nlocal+0+ny*nz+iy+(iz-1)*ny]
-xv[nlocal+0+ny*nz+iy+1+(iz-1)*ny]
-xv[nlocal+0+ny*nz+iy-1+(iz)*ny]
-xv[nlocal+0+ny*nz+iy+iz*ny]
-xv[nlocal+0+ny*nz+iy+1+iz*ny]
-xv[nlocal+0+ny*nz+iy-1+(iz+1)*ny]
-xv[nlocal+0+ny*nz+iy+(iz+1)*ny]
-xv[nlocal+0+ny*nz+iy+1+(iz+1)*ny]
;}
}
}//ipz < npz - 1
          }//ipz > 0 
        }//ipy < npy - 1 
      }//ipy > 0 
    }//ipx < npx - 1 
  else
{
    if(ipy > 0)
{
      if(ipy < npy - 1)
{
        if(ipz > 0)
{
            if(ipz < npz - 1)
{
ix = 0;
iy = 0;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+1+1]
;
ix = 0;
iy = 0;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+1]
;
ix = 0;
iy = ny-1;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+ny+nx*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+1]
;
ix = 0;
iy = ny-1;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx*ny+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx*ny+1+1]
;
ix = nx-1;
iy = 0;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx-2]
-xv[nlocal+0+1+nx-1]
;
ix = nx-1;
iy = 0;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx-1]
;
ix = nx-1;
iy = ny-1;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx-1]
;
ix = nx-1;
iy = ny-1;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx*ny+1+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx*ny+1+nx-1]
;
iy=0;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+ix-1]
-xv[nlocal+0+1+ix]
-xv[nlocal+0+1+ix+1]
;}
iy=0;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx+ix-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx+ix]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix+(nz-2)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix+(nz-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix+(nz-1)*nx+1]
;}
iy=0;
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+ix-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+ix]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+ix+1]
;}
iz=0;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+ix-1]
-xv[nlocal+0+1+nx+ny+ix]
-xv[nlocal+0+1+nx+ny+ix+1]
-xv[nlocal+0+1+nx+ny+nx+ix-1]
-xv[nlocal+0+1+nx+ny+nx+ix]
-xv[nlocal+0+1+nx+ny+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+1+nx+ny+ix+(ny-2)*nx]
-xv[nlocal+0+1+nx+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+1+nx+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+1+nx+ny+ix+(ny-1)*nx]
-xv[nlocal+0+1+nx+ny+ix+(ny-1)*nx+1]
;}
iz=nz-1;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx+ix-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx+ix]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix+(ny-2)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix+(ny-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix+(ny-1)*nx+1]
;}
iy=ny-1;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+ix-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+ix]
-xv[nlocal+0+1+nx+ny+nx*ny+1+ix+1]
;}
iy=ny-1;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx+ix-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx+ix]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix+(nz-2)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix+(nz-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix+(nz-1)*nx+1]
;}
iy=ny-1;
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx*ny+1+ix-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx*ny+1+ix]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx*ny+1+ix+1]
;}
;
ix=0;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+iy-1]
-xv[nlocal+0+1+nx+iy]
-xv[nlocal+0+1+nx+iy+1]
;}
ix=0;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny+iy-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny+iy]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy+(nz-2)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy+(nz-1)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy+(nz-1)*ny+1]
;}
ix=0;
iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+iy-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+iy]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+iy+1]
;}
iz=0;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+(iy-1)*nx]
-xv[nlocal+0+1+nx+ny+(iy-1)*nx+1]
-xv[nlocal+0+1+nx+ny+(iy)*nx]
-xv[nlocal+0+1+nx+ny+(iy-1)*nx+1]
-xv[nlocal+0+1+nx+ny+(iy+1)*nx]
-xv[nlocal+0+1+nx+ny+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+1+nx+ny+(iy)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+(iy)*nx+nx-1]
-xv[nlocal+0+1+nx+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+(iy+1)*nx+nx-1]
;}
iz=nz-1;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+(iy-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+(iy-1)*nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+(iy)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+(iy-1)*nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+(iy+1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+(iy)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+(iy)*nx+nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+(iy+1)*nx+nx-1]
;}
;
ix=0;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+iz-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+iz]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+iz+1]
;}
ix=0;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+(iz-1)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+(iz-1)*ny+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+(iz)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+(iz-1)*ny+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+(iz+1)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+(iz)*ny+ny-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+(iz)*ny+ny-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+(iz+1)*ny+ny-1]
;}
ix=0;
iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+iz-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+iz]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+iz+1]
;}
iy=0;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+(iz-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+(iz-1)*nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+(iz)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+(iz-1)*nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+(iz+1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+(iz)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+(iz)*nx+nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+(iz+1)*nx+nx-1]
;}
iy=ny-1;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+(iz-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+(iz-1)*nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+(iz)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+(iz-1)*nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+(iz+1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+(iz)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+(iz)*nx+nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+(iz+1)*nx+nx-1]
;}
;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy-1+(iz-1)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy+(iz-1)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy+1+(iz-1)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy-1+(iz)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy+iz*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy+1+iz*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy-1+(iz+1)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy+(iz+1)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy+1+(iz+1)*ny]
;}
}
iy=0;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix-1+(iz-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix+(iz-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix+1+(iz-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix-1+(iz)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix+iz*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix+1+iz*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix-1+(iz+1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix+(iz+1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix+1+(iz+1)*nx]
;}
}
iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix-1+(iz-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix+(iz-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix+1+(iz-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix-1+(iz)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix+iz*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix+1+iz*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix-1+(iz+1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix+(iz+1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix+1+(iz+1)*nx]
;}
}
iz=0;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+ix-1+(iy-1)*nx]
-xv[nlocal+0+1+nx+ny+ix+(iy-1)*nx]
-xv[nlocal+0+1+nx+ny+ix+1+(iy-1)*nx]
-xv[nlocal+0+1+nx+ny+ix-1+(iy)*nx]
-xv[nlocal+0+1+nx+ny+ix+iy*nx]
-xv[nlocal+0+1+nx+ny+ix+1+iy*nx]
-xv[nlocal+0+1+nx+ny+ix-1+(iy+1)*nx]
-xv[nlocal+0+1+nx+ny+ix+(iy+1)*nx]
-xv[nlocal+0+1+nx+ny+ix+1+(iy+1)*nx]
;}
}
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix-1+(iy-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix+(iy-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix+1+(iy-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix-1+(iy)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix+iy*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix+1+iy*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix-1+(iy+1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix+(iy+1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix+1+(iy+1)*nx]
;}
}
}//ipz < npz - 1
            else
{
ix = 0;
iy = 0;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+1+1]
;
ix = 0;
iy = ny-1;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+ny+nx*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+1]
;
ix = nx-1;
iy = 0;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx-2]
-xv[nlocal+0+1+nx-1]
;
ix = nx-1;
iy = ny-1;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx-1]
;
iy=0;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+ix-1]
-xv[nlocal+0+1+ix]
-xv[nlocal+0+1+ix+1]
;}
iy=0;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx+ix-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx+ix]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix+(nz-2)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix+(nz-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix+(nz-1)*nx+1]
;}
iz=0;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+ix-1]
-xv[nlocal+0+1+nx+ny+ix]
-xv[nlocal+0+1+nx+ny+ix+1]
-xv[nlocal+0+1+nx+ny+nx+ix-1]
-xv[nlocal+0+1+nx+ny+nx+ix]
-xv[nlocal+0+1+nx+ny+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+1+nx+ny+ix+(ny-2)*nx]
-xv[nlocal+0+1+nx+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+1+nx+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+1+nx+ny+ix+(ny-1)*nx]
-xv[nlocal+0+1+nx+ny+ix+(ny-1)*nx+1]
;}
iy=ny-1;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+ix-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+ix]
-xv[nlocal+0+1+nx+ny+nx*ny+1+ix+1]
;}
iy=ny-1;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx+ix-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx+ix]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix+(nz-2)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix+(nz-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix+(nz-1)*nx+1]
;}
;
ix=0;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+iy-1]
-xv[nlocal+0+1+nx+iy]
-xv[nlocal+0+1+nx+iy+1]
;}
ix=0;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny+iy-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny+iy]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy+(nz-2)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy+(nz-1)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy+(nz-1)*ny+1]
;}
iz=0;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+(iy-1)*nx]
-xv[nlocal+0+1+nx+ny+(iy-1)*nx+1]
-xv[nlocal+0+1+nx+ny+(iy)*nx]
-xv[nlocal+0+1+nx+ny+(iy-1)*nx+1]
-xv[nlocal+0+1+nx+ny+(iy+1)*nx]
-xv[nlocal+0+1+nx+ny+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+1+nx+ny+(iy)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+(iy)*nx+nx-1]
-xv[nlocal+0+1+nx+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+(iy+1)*nx+nx-1]
;}
;
ix=0;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+iz-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+iz]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+iz+1]
;}
ix=0;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+(iz-1)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+(iz-1)*ny+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+(iz)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+(iz-1)*ny+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+(iz+1)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+(iz)*ny+ny-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+(iz)*ny+ny-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+(iz+1)*ny+ny-1]
;}
ix=0;
iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+iz-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+iz]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+iz+1]
;}
iy=0;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+(iz-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+(iz-1)*nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+(iz)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+(iz-1)*nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+(iz+1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+(iz)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+(iz)*nx+nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+(iz+1)*nx+nx-1]
;}
iy=ny-1;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+(iz-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+(iz-1)*nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+(iz)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+(iz-1)*nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+(iz+1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+(iz)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+(iz)*nx+nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+(iz+1)*nx+nx-1]
;}
;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy-1+(iz-1)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy+(iz-1)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy+1+(iz-1)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy-1+(iz)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy+iz*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy+1+iz*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy-1+(iz+1)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy+(iz+1)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy+1+(iz+1)*ny]
;}
}
iy=0;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix-1+(iz-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix+(iz-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix+1+(iz-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix-1+(iz)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix+iz*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix+1+iz*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix-1+(iz+1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix+(iz+1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix+1+(iz+1)*nx]
;}
}
iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix-1+(iz-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix+(iz-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix+1+(iz-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix-1+(iz)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix+iz*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix+1+iz*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix-1+(iz+1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix+(iz+1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix+1+(iz+1)*nx]
;}
}
iz=0;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+ix-1+(iy-1)*nx]
-xv[nlocal+0+1+nx+ny+ix+(iy-1)*nx]
-xv[nlocal+0+1+nx+ny+ix+1+(iy-1)*nx]
-xv[nlocal+0+1+nx+ny+ix-1+(iy)*nx]
-xv[nlocal+0+1+nx+ny+ix+iy*nx]
-xv[nlocal+0+1+nx+ny+ix+1+iy*nx]
-xv[nlocal+0+1+nx+ny+ix-1+(iy+1)*nx]
-xv[nlocal+0+1+nx+ny+ix+(iy+1)*nx]
-xv[nlocal+0+1+nx+ny+ix+1+(iy+1)*nx]
;}
}
}//ipz < npz - 1
          }//ipz > 0 
        else
{
            if(ipz < npz - 1)
{
ix = 0;
iy = 0;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+1]
;
ix = 0;
iy = ny-1;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx*ny]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx*ny+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx*ny+1+1]
;
ix = nx-1;
iy = 0;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx-2]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx-1]
;
ix = nx-1;
iy = ny-1;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx*ny+1+nx-2]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx*ny+1+nx-1]
;
iy=0;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+ix-1]
-xv[nlocal+0+nz+ix]
-xv[nlocal+0+nz+ix+1]
-xv[nlocal+0+nz+nx+ix-1]
-xv[nlocal+0+nz+nx+ix]
-xv[nlocal+0+nz+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+nz+ix+(nz-2)*nx]
-xv[nlocal+0+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+nz+ix+(nz-1)*nx]
-xv[nlocal+0+nz+ix+(nz-1)*nx+1]
;}
iy=0;
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+ix-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+ix]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+ix+1]
;}
iz=nz-1;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx+ix-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx+ix]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix+(ny-2)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix+(ny-1)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix+(ny-1)*nx+1]
;}
iy=ny-1;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx+ix-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx+ix]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix+(nz-2)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix+(nz-1)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix+(nz-1)*nx+1]
;}
iy=ny-1;
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx*ny+1+ix-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx*ny+1+ix]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx*ny+1+ix+1]
;}
;
ix=0;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+iy-1]
-xv[nlocal+0+nz+nx*nz+iy]
-xv[nlocal+0+nz+nx*nz+iy+1]
-xv[nlocal+0+nz+nx*nz+ny+iy-1]
-xv[nlocal+0+nz+nx*nz+ny+iy]
-xv[nlocal+0+nz+nx*nz+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+nz+nx*nz+iy+(nz-2)*ny]
-xv[nlocal+0+nz+nx*nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+nz+nx*nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+nz+nx*nz+iy+(nz-1)*ny]
-xv[nlocal+0+nz+nx*nz+iy+(nz-1)*ny+1]
;}
ix=0;
iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+iy-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+iy]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+iy+1]
;}
iz=nz-1;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+(iy-1)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+(iy-1)*nx+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+(iy)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+(iy-1)*nx+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+(iy+1)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+(iy)*nx+nx-2]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+(iy)*nx+nx-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+(iy+1)*nx+nx-1]
;}
;
ix=0;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iz-1]
-xv[nlocal+0+iz]
-xv[nlocal+0+iz+1]
;}
ix=0;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+(iz-1)*ny+1]
-xv[nlocal+0+nz+nx*nz+(iz)*ny]
-xv[nlocal+0+nz+nx*nz+(iz-1)*ny+1]
-xv[nlocal+0+nz+nx*nz+(iz+1)*ny]
-xv[nlocal+0+nz+nx*nz+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+nz+nx*nz+(iz)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+(iz)*ny+ny-1]
-xv[nlocal+0+nz+nx*nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+(iz+1)*ny+ny-1]
;}
ix=0;
iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+ny*nz+iz-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+iz]
-xv[nlocal+0+nz+nx*nz+ny*nz+iz+1]
;}
iy=0;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+(iz-1)*nx]
-xv[nlocal+0+nz+(iz-1)*nx+1]
-xv[nlocal+0+nz+(iz)*nx]
-xv[nlocal+0+nz+(iz-1)*nx+1]
-xv[nlocal+0+nz+(iz+1)*nx]
-xv[nlocal+0+nz+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+nz+(iz)*nx+nx-2]
-xv[nlocal+0+nz+(iz)*nx+nx-1]
-xv[nlocal+0+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+nz+(iz+1)*nx+nx-1]
;}
iy=ny-1;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+(iz-1)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+(iz-1)*nx+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+(iz)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+(iz-1)*nx+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+(iz+1)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+(iz)*nx+nx-2]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+(iz)*nx+nx-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+(iz+1)*nx+nx-1]
;}
;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+iy-1+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+iy+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+iy+1+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+iy-1+(iz)*ny]
-xv[nlocal+0+nz+nx*nz+iy+iz*ny]
-xv[nlocal+0+nz+nx*nz+iy+1+iz*ny]
-xv[nlocal+0+nz+nx*nz+iy-1+(iz+1)*ny]
-xv[nlocal+0+nz+nx*nz+iy+(iz+1)*ny]
-xv[nlocal+0+nz+nx*nz+iy+1+(iz+1)*ny]
;}
}
iy=0;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+ix-1+(iz-1)*nx]
-xv[nlocal+0+nz+ix+(iz-1)*nx]
-xv[nlocal+0+nz+ix+1+(iz-1)*nx]
-xv[nlocal+0+nz+ix-1+(iz)*nx]
-xv[nlocal+0+nz+ix+iz*nx]
-xv[nlocal+0+nz+ix+1+iz*nx]
-xv[nlocal+0+nz+ix-1+(iz+1)*nx]
-xv[nlocal+0+nz+ix+(iz+1)*nx]
-xv[nlocal+0+nz+ix+1+(iz+1)*nx]
;}
}
iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix-1+(iz-1)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix+(iz-1)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix+1+(iz-1)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix-1+(iz)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix+iz*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix+1+iz*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix-1+(iz+1)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix+(iz+1)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix+1+(iz+1)*nx]
;}
}
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix-1+(iy-1)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix+(iy-1)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix+1+(iy-1)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix-1+(iy)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix+iy*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix+1+iy*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix-1+(iy+1)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix+(iy+1)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix+1+(iy+1)*nx]
;}
}
}//ipz < npz - 1
            else
{
iy=0;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+ix-1]
-xv[nlocal+0+nz+ix]
-xv[nlocal+0+nz+ix+1]
-xv[nlocal+0+nz+nx+ix-1]
-xv[nlocal+0+nz+nx+ix]
-xv[nlocal+0+nz+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+nz+ix+(nz-2)*nx]
-xv[nlocal+0+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+nz+ix+(nz-1)*nx]
-xv[nlocal+0+nz+ix+(nz-1)*nx+1]
;}
iy=ny-1;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx+ix-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx+ix]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix+(nz-2)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix+(nz-1)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix+(nz-1)*nx+1]
;}
;
ix=0;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+iy-1]
-xv[nlocal+0+nz+nx*nz+iy]
-xv[nlocal+0+nz+nx*nz+iy+1]
-xv[nlocal+0+nz+nx*nz+ny+iy-1]
-xv[nlocal+0+nz+nx*nz+ny+iy]
-xv[nlocal+0+nz+nx*nz+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+nz+nx*nz+iy+(nz-2)*ny]
-xv[nlocal+0+nz+nx*nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+nz+nx*nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+nz+nx*nz+iy+(nz-1)*ny]
-xv[nlocal+0+nz+nx*nz+iy+(nz-1)*ny+1]
;}
;
ix=0;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iz-1]
-xv[nlocal+0+iz]
-xv[nlocal+0+iz+1]
;}
ix=0;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+(iz-1)*ny+1]
-xv[nlocal+0+nz+nx*nz+(iz)*ny]
-xv[nlocal+0+nz+nx*nz+(iz-1)*ny+1]
-xv[nlocal+0+nz+nx*nz+(iz+1)*ny]
-xv[nlocal+0+nz+nx*nz+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+nz+nx*nz+(iz)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+(iz)*ny+ny-1]
-xv[nlocal+0+nz+nx*nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+(iz+1)*ny+ny-1]
;}
ix=0;
iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+ny*nz+iz-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+iz]
-xv[nlocal+0+nz+nx*nz+ny*nz+iz+1]
;}
iy=0;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+(iz-1)*nx]
-xv[nlocal+0+nz+(iz-1)*nx+1]
-xv[nlocal+0+nz+(iz)*nx]
-xv[nlocal+0+nz+(iz-1)*nx+1]
-xv[nlocal+0+nz+(iz+1)*nx]
-xv[nlocal+0+nz+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+nz+(iz)*nx+nx-2]
-xv[nlocal+0+nz+(iz)*nx+nx-1]
-xv[nlocal+0+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+nz+(iz+1)*nx+nx-1]
;}
iy=ny-1;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+(iz-1)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+(iz-1)*nx+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+(iz)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+(iz-1)*nx+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+(iz+1)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+(iz)*nx+nx-2]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+(iz)*nx+nx-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+(iz+1)*nx+nx-1]
;}
;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+iy-1+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+iy+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+iy+1+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+iy-1+(iz)*ny]
-xv[nlocal+0+nz+nx*nz+iy+iz*ny]
-xv[nlocal+0+nz+nx*nz+iy+1+iz*ny]
-xv[nlocal+0+nz+nx*nz+iy-1+(iz+1)*ny]
-xv[nlocal+0+nz+nx*nz+iy+(iz+1)*ny]
-xv[nlocal+0+nz+nx*nz+iy+1+(iz+1)*ny]
;}
}
iy=0;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+ix-1+(iz-1)*nx]
-xv[nlocal+0+nz+ix+(iz-1)*nx]
-xv[nlocal+0+nz+ix+1+(iz-1)*nx]
-xv[nlocal+0+nz+ix-1+(iz)*nx]
-xv[nlocal+0+nz+ix+iz*nx]
-xv[nlocal+0+nz+ix+1+iz*nx]
-xv[nlocal+0+nz+ix-1+(iz+1)*nx]
-xv[nlocal+0+nz+ix+(iz+1)*nx]
-xv[nlocal+0+nz+ix+1+(iz+1)*nx]
;}
}
iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix-1+(iz-1)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix+(iz-1)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix+1+(iz-1)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix-1+(iz)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix+iz*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix+1+iz*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix-1+(iz+1)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix+(iz+1)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix+1+(iz+1)*nx]
;}
}
}//ipz < npz - 1
          }//ipz > 0 
        }//ipy < npy - 1 
      else
{
        if(ipz > 0)
{
            if(ipz < npz - 1)
{
ix = 0;
iy = 0;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+1+1]
;
ix = 0;
iy = 0;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+1]
;
ix = nx-1;
iy = 0;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx-2]
-xv[nlocal+0+1+nx-1]
;
ix = nx-1;
iy = 0;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx-1]
;
iy=0;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+ix-1]
-xv[nlocal+0+1+ix]
-xv[nlocal+0+1+ix+1]
;}
iy=0;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx+ix-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx+ix]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix+(nz-2)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix+(nz-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix+(nz-1)*nx+1]
;}
iy=0;
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+ix-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+ix]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+ix+1]
;}
iz=0;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+ix-1]
-xv[nlocal+0+1+nx+ny+ix]
-xv[nlocal+0+1+nx+ny+ix+1]
-xv[nlocal+0+1+nx+ny+nx+ix-1]
-xv[nlocal+0+1+nx+ny+nx+ix]
-xv[nlocal+0+1+nx+ny+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+1+nx+ny+ix+(ny-2)*nx]
-xv[nlocal+0+1+nx+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+1+nx+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+1+nx+ny+ix+(ny-1)*nx]
-xv[nlocal+0+1+nx+ny+ix+(ny-1)*nx+1]
;}
iz=nz-1;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+ix-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+ix]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+ix+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+nx+ix-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+nx+ix]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+ix+(ny-2)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+ix+(ny-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+ix+(ny-1)*nx+1]
;}
;
ix=0;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+iy-1]
-xv[nlocal+0+1+nx+iy]
-xv[nlocal+0+1+nx+iy+1]
;}
ix=0;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny+iy-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny+iy]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy+(nz-2)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy+(nz-1)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy+(nz-1)*ny+1]
;}
ix=0;
iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+iy-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+iy]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+iy+1]
;}
iz=0;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+(iy-1)*nx]
-xv[nlocal+0+1+nx+ny+(iy-1)*nx+1]
-xv[nlocal+0+1+nx+ny+(iy)*nx]
-xv[nlocal+0+1+nx+ny+(iy-1)*nx+1]
-xv[nlocal+0+1+nx+ny+(iy+1)*nx]
-xv[nlocal+0+1+nx+ny+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+1+nx+ny+(iy)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+(iy)*nx+nx-1]
-xv[nlocal+0+1+nx+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+(iy+1)*nx+nx-1]
;}
iz=nz-1;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+(iy-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+(iy-1)*nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+(iy)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+(iy-1)*nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+(iy+1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+(iy)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+(iy)*nx+nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+(iy+1)*nx+nx-1]
;}
;
ix=0;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+iz-1]
-xv[nlocal+0+1+nx+ny+nx*ny+iz]
-xv[nlocal+0+1+nx+ny+nx*ny+iz+1]
;}
ix=0;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+(iz-1)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+(iz-1)*ny+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+(iz)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+(iz-1)*ny+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+(iz+1)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+(iz)*ny+ny-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+(iz)*ny+ny-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+(iz+1)*ny+ny-1]
;}
iy=0;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+nz+(iz-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+(iz-1)*nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+(iz)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+(iz-1)*nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+(iz+1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+(iz)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+(iz)*nx+nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+(iz+1)*nx+nx-1]
;}
;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy-1+(iz-1)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy+(iz-1)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy+1+(iz-1)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy-1+(iz)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy+iz*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy+1+iz*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy-1+(iz+1)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy+(iz+1)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy+1+(iz+1)*ny]
;}
}
iy=0;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix-1+(iz-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix+(iz-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix+1+(iz-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix-1+(iz)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix+iz*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix+1+iz*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix-1+(iz+1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix+(iz+1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix+1+(iz+1)*nx]
;}
}
iz=0;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+ix-1+(iy-1)*nx]
-xv[nlocal+0+1+nx+ny+ix+(iy-1)*nx]
-xv[nlocal+0+1+nx+ny+ix+1+(iy-1)*nx]
-xv[nlocal+0+1+nx+ny+ix-1+(iy)*nx]
-xv[nlocal+0+1+nx+ny+ix+iy*nx]
-xv[nlocal+0+1+nx+ny+ix+1+iy*nx]
-xv[nlocal+0+1+nx+ny+ix-1+(iy+1)*nx]
-xv[nlocal+0+1+nx+ny+ix+(iy+1)*nx]
-xv[nlocal+0+1+nx+ny+ix+1+(iy+1)*nx]
;}
}
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+ix-1+(iy-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+ix+(iy-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+ix+1+(iy-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+ix-1+(iy)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+ix+iy*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+ix+1+iy*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+ix-1+(iy+1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+ix+(iy+1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+ix+1+(iy+1)*nx]
;}
}
}//ipz < npz - 1
            else
{
ix = 0;
iy = 0;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+1+1]
;
ix = nx-1;
iy = 0;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx-2]
-xv[nlocal+0+1+nx-1]
;
iy=0;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+ix-1]
-xv[nlocal+0+1+ix]
-xv[nlocal+0+1+ix+1]
;}
iy=0;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx+ix-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx+ix]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix+(nz-2)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix+(nz-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix+(nz-1)*nx+1]
;}
iz=0;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+ix-1]
-xv[nlocal+0+1+nx+ny+ix]
-xv[nlocal+0+1+nx+ny+ix+1]
-xv[nlocal+0+1+nx+ny+nx+ix-1]
-xv[nlocal+0+1+nx+ny+nx+ix]
-xv[nlocal+0+1+nx+ny+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+1+nx+ny+ix+(ny-2)*nx]
-xv[nlocal+0+1+nx+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+1+nx+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+1+nx+ny+ix+(ny-1)*nx]
-xv[nlocal+0+1+nx+ny+ix+(ny-1)*nx+1]
;}
;
ix=0;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+iy-1]
-xv[nlocal+0+1+nx+iy]
-xv[nlocal+0+1+nx+iy+1]
;}
ix=0;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny+iy-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny+iy]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy+(nz-2)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy+(nz-1)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy+(nz-1)*ny+1]
;}
iz=0;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+(iy-1)*nx]
-xv[nlocal+0+1+nx+ny+(iy-1)*nx+1]
-xv[nlocal+0+1+nx+ny+(iy)*nx]
-xv[nlocal+0+1+nx+ny+(iy-1)*nx+1]
-xv[nlocal+0+1+nx+ny+(iy+1)*nx]
-xv[nlocal+0+1+nx+ny+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+1+nx+ny+(iy)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+(iy)*nx+nx-1]
-xv[nlocal+0+1+nx+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+(iy+1)*nx+nx-1]
;}
;
ix=0;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+iz-1]
-xv[nlocal+0+1+nx+ny+nx*ny+iz]
-xv[nlocal+0+1+nx+ny+nx*ny+iz+1]
;}
ix=0;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+(iz-1)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+(iz-1)*ny+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+(iz)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+(iz-1)*ny+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+(iz+1)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+(iz)*ny+ny-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+(iz)*ny+ny-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+(iz+1)*ny+ny-1]
;}
iy=0;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+nz+(iz-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+(iz-1)*nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+(iz)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+(iz-1)*nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+(iz+1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+(iz)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+(iz)*nx+nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+(iz+1)*nx+nx-1]
;}
;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy-1+(iz-1)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy+(iz-1)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy+1+(iz-1)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy-1+(iz)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy+iz*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy+1+iz*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy-1+(iz+1)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy+(iz+1)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy+1+(iz+1)*ny]
;}
}
iy=0;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix-1+(iz-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix+(iz-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix+1+(iz-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix-1+(iz)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix+iz*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix+1+iz*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix-1+(iz+1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix+(iz+1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix+1+(iz+1)*nx]
;}
}
iz=0;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+ix-1+(iy-1)*nx]
-xv[nlocal+0+1+nx+ny+ix+(iy-1)*nx]
-xv[nlocal+0+1+nx+ny+ix+1+(iy-1)*nx]
-xv[nlocal+0+1+nx+ny+ix-1+(iy)*nx]
-xv[nlocal+0+1+nx+ny+ix+iy*nx]
-xv[nlocal+0+1+nx+ny+ix+1+iy*nx]
-xv[nlocal+0+1+nx+ny+ix-1+(iy+1)*nx]
-xv[nlocal+0+1+nx+ny+ix+(iy+1)*nx]
-xv[nlocal+0+1+nx+ny+ix+1+(iy+1)*nx]
;}
}
}//ipz < npz - 1
          }//ipz > 0 
        else
{
            if(ipz < npz - 1)
{
ix = 0;
iy = 0;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx*nz+ny*nz]
-xv[nlocal+0+nz+nx*nz+ny*nz+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+1]
;
ix = nx-1;
iy = 0;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx-2]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx-1]
;
iy=0;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+ix-1]
-xv[nlocal+0+nz+ix]
-xv[nlocal+0+nz+ix+1]
-xv[nlocal+0+nz+nx+ix-1]
-xv[nlocal+0+nz+nx+ix]
-xv[nlocal+0+nz+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+nz+ix+(nz-2)*nx]
-xv[nlocal+0+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+nz+ix+(nz-1)*nx]
-xv[nlocal+0+nz+ix+(nz-1)*nx+1]
;}
iy=0;
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+ny*nz+1+ix-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+ix]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+ix+1]
;}
iz=nz-1;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+ix-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+ix]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+ix+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+nx+ix-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+nx+ix]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+ix+(ny-2)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+ix+(ny-1)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+ix+(ny-1)*nx+1]
;}
;
ix=0;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+iy-1]
-xv[nlocal+0+nz+nx*nz+iy]
-xv[nlocal+0+nz+nx*nz+iy+1]
-xv[nlocal+0+nz+nx*nz+ny+iy-1]
-xv[nlocal+0+nz+nx*nz+ny+iy]
-xv[nlocal+0+nz+nx*nz+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+nz+nx*nz+iy+(nz-2)*ny]
-xv[nlocal+0+nz+nx*nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+nz+nx*nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+nz+nx*nz+iy+(nz-1)*ny]
-xv[nlocal+0+nz+nx*nz+iy+(nz-1)*ny+1]
;}
ix=0;
iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+iy-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+iy]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+iy+1]
;}
iz=nz-1;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+(iy-1)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+(iy-1)*nx+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+(iy)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+(iy-1)*nx+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+(iy+1)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+(iy)*nx+nx-2]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+(iy)*nx+nx-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+(iy+1)*nx+nx-1]
;}
;
ix=0;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iz-1]
-xv[nlocal+0+iz]
-xv[nlocal+0+iz+1]
;}
ix=0;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+(iz-1)*ny+1]
-xv[nlocal+0+nz+nx*nz+(iz)*ny]
-xv[nlocal+0+nz+nx*nz+(iz-1)*ny+1]
-xv[nlocal+0+nz+nx*nz+(iz+1)*ny]
-xv[nlocal+0+nz+nx*nz+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+nz+nx*nz+(iz)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+(iz)*ny+ny-1]
-xv[nlocal+0+nz+nx*nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+(iz+1)*ny+ny-1]
;}
iy=0;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+(iz-1)*nx]
-xv[nlocal+0+nz+(iz-1)*nx+1]
-xv[nlocal+0+nz+(iz)*nx]
-xv[nlocal+0+nz+(iz-1)*nx+1]
-xv[nlocal+0+nz+(iz+1)*nx]
-xv[nlocal+0+nz+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+nz+(iz)*nx+nx-2]
-xv[nlocal+0+nz+(iz)*nx+nx-1]
-xv[nlocal+0+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+nz+(iz+1)*nx+nx-1]
;}
;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+iy-1+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+iy+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+iy+1+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+iy-1+(iz)*ny]
-xv[nlocal+0+nz+nx*nz+iy+iz*ny]
-xv[nlocal+0+nz+nx*nz+iy+1+iz*ny]
-xv[nlocal+0+nz+nx*nz+iy-1+(iz+1)*ny]
-xv[nlocal+0+nz+nx*nz+iy+(iz+1)*ny]
-xv[nlocal+0+nz+nx*nz+iy+1+(iz+1)*ny]
;}
}
iy=0;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+ix-1+(iz-1)*nx]
-xv[nlocal+0+nz+ix+(iz-1)*nx]
-xv[nlocal+0+nz+ix+1+(iz-1)*nx]
-xv[nlocal+0+nz+ix-1+(iz)*nx]
-xv[nlocal+0+nz+ix+iz*nx]
-xv[nlocal+0+nz+ix+1+iz*nx]
-xv[nlocal+0+nz+ix-1+(iz+1)*nx]
-xv[nlocal+0+nz+ix+(iz+1)*nx]
-xv[nlocal+0+nz+ix+1+(iz+1)*nx]
;}
}
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+ix-1+(iy-1)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+ix+(iy-1)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+ix+1+(iy-1)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+ix-1+(iy)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+ix+iy*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+ix+1+iy*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+ix-1+(iy+1)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+ix+(iy+1)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+ix+1+(iy+1)*nx]
;}
}
}//ipz < npz - 1
            else
{
iy=0;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+ix-1]
-xv[nlocal+0+nz+ix]
-xv[nlocal+0+nz+ix+1]
-xv[nlocal+0+nz+nx+ix-1]
-xv[nlocal+0+nz+nx+ix]
-xv[nlocal+0+nz+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+nz+ix+(nz-2)*nx]
-xv[nlocal+0+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+nz+ix+(nz-1)*nx]
-xv[nlocal+0+nz+ix+(nz-1)*nx+1]
;}
;
ix=0;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+iy-1]
-xv[nlocal+0+nz+nx*nz+iy]
-xv[nlocal+0+nz+nx*nz+iy+1]
-xv[nlocal+0+nz+nx*nz+ny+iy-1]
-xv[nlocal+0+nz+nx*nz+ny+iy]
-xv[nlocal+0+nz+nx*nz+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+nz+nx*nz+iy+(nz-2)*ny]
-xv[nlocal+0+nz+nx*nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+nz+nx*nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+nz+nx*nz+iy+(nz-1)*ny]
-xv[nlocal+0+nz+nx*nz+iy+(nz-1)*ny+1]
;}
;
ix=0;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iz-1]
-xv[nlocal+0+iz]
-xv[nlocal+0+iz+1]
;}
ix=0;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+(iz-1)*ny+1]
-xv[nlocal+0+nz+nx*nz+(iz)*ny]
-xv[nlocal+0+nz+nx*nz+(iz-1)*ny+1]
-xv[nlocal+0+nz+nx*nz+(iz+1)*ny]
-xv[nlocal+0+nz+nx*nz+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+nz+nx*nz+(iz)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+(iz)*ny+ny-1]
-xv[nlocal+0+nz+nx*nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+(iz+1)*ny+ny-1]
;}
iy=0;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+(iz-1)*nx]
-xv[nlocal+0+nz+(iz-1)*nx+1]
-xv[nlocal+0+nz+(iz)*nx]
-xv[nlocal+0+nz+(iz-1)*nx+1]
-xv[nlocal+0+nz+(iz+1)*nx]
-xv[nlocal+0+nz+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+nz+(iz)*nx+nx-2]
-xv[nlocal+0+nz+(iz)*nx+nx-1]
-xv[nlocal+0+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+nz+(iz+1)*nx+nx-1]
;}
;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+iy-1+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+iy+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+iy+1+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+iy-1+(iz)*ny]
-xv[nlocal+0+nz+nx*nz+iy+iz*ny]
-xv[nlocal+0+nz+nx*nz+iy+1+iz*ny]
-xv[nlocal+0+nz+nx*nz+iy-1+(iz+1)*ny]
-xv[nlocal+0+nz+nx*nz+iy+(iz+1)*ny]
-xv[nlocal+0+nz+nx*nz+iy+1+(iz+1)*ny]
;}
}
iy=0;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+ix-1+(iz-1)*nx]
-xv[nlocal+0+nz+ix+(iz-1)*nx]
-xv[nlocal+0+nz+ix+1+(iz-1)*nx]
-xv[nlocal+0+nz+ix-1+(iz)*nx]
-xv[nlocal+0+nz+ix+iz*nx]
-xv[nlocal+0+nz+ix+1+iz*nx]
-xv[nlocal+0+nz+ix-1+(iz+1)*nx]
-xv[nlocal+0+nz+ix+(iz+1)*nx]
-xv[nlocal+0+nz+ix+1+(iz+1)*nx]
;}
}
}//ipz < npz - 1
          }//ipz > 0 
        }//ipy < npy - 1 
      }//ipy > 0 
    else
{
      if(ipy < npy - 1)
{
        if(ipz > 0)
{
            if(ipz < npz - 1)
{
ix = 0;
iy = ny-1;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx*ny]
-xv[nlocal+0+ny+nx*ny+1]
-xv[nlocal+0+ny+nx*ny+1+1]
;
ix = 0;
iy = ny-1;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+nx*ny]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+nx*ny+1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+nx*ny+1+1]
;
ix = nx-1;
iy = ny-1;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx*ny+1+nx-2]
-xv[nlocal+0+ny+nx*ny+1+nx-1]
;
ix = nx-1;
iy = ny-1;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+nx*ny+1+nx-2]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+nx*ny+1+nx-1]
;
iz=0;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+ix-1]
-xv[nlocal+0+ny+ix]
-xv[nlocal+0+ny+ix+1]
-xv[nlocal+0+ny+nx+ix-1]
-xv[nlocal+0+ny+nx+ix]
-xv[nlocal+0+ny+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+ny+ix+(ny-2)*nx]
-xv[nlocal+0+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+ny+ix+(ny-1)*nx]
-xv[nlocal+0+ny+ix+(ny-1)*nx+1]
;}
iz=nz-1;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+ix-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+ix]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+ix+1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+nx+ix-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+nx+ix]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+ix+(ny-2)*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+ix+(ny-1)*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+ix+(ny-1)*nx+1]
;}
iy=ny-1;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+ix-1]
-xv[nlocal+0+ny+nx*ny+1+ix]
-xv[nlocal+0+ny+nx*ny+1+ix+1]
;}
iy=ny-1;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix+1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx+ix-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx+ix]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix+(nz-2)*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix+(nz-1)*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix+(nz-1)*nx+1]
;}
iy=ny-1;
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+nx*ny+1+ix-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+nx*ny+1+ix]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+nx*ny+1+ix+1]
;}
;
ix=0;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1]
-xv[nlocal+0+iy]
-xv[nlocal+0+iy+1]
;}
ix=0;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+iy-1]
-xv[nlocal+0+ny+nx*ny+1+nx+iy]
-xv[nlocal+0+ny+nx*ny+1+nx+iy+1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny+iy-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny+iy]
-xv[nlocal+0+ny+nx*ny+1+nx+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+iy+(nz-2)*ny-1]
-xv[nlocal+0+ny+nx*ny+1+nx+iy+(nz-2)*ny]
-xv[nlocal+0+ny+nx*ny+1+nx+iy+(nz-2)*ny+1]
-xv[nlocal+0+ny+nx*ny+1+nx+iy+(nz-1)*ny-1]
-xv[nlocal+0+ny+nx*ny+1+nx+iy+(nz-1)*ny]
-xv[nlocal+0+ny+nx*ny+1+nx+iy+(nz-1)*ny+1]
;}
ix=0;
iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+iy-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+iy]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+iy+1]
;}
iz=0;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+(iy-1)*nx]
-xv[nlocal+0+ny+(iy-1)*nx+1]
-xv[nlocal+0+ny+(iy)*nx]
-xv[nlocal+0+ny+(iy-1)*nx+1]
-xv[nlocal+0+ny+(iy+1)*nx]
-xv[nlocal+0+ny+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+ny+(iy)*nx+nx-2]
-xv[nlocal+0+ny+(iy)*nx+nx-1]
-xv[nlocal+0+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+ny+(iy+1)*nx+nx-1]
;}
iz=nz-1;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+(iy-1)*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+(iy-1)*nx+1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+(iy)*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+(iy-1)*nx+1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+(iy+1)*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+(iy)*nx+nx-2]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+(iy)*nx+nx-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+(iy+1)*nx+nx-1]
;}
;
ix=0;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+1+nx+(iz-1)*ny+1]
-xv[nlocal+0+ny+nx*ny+1+nx+(iz)*ny]
-xv[nlocal+0+ny+nx*ny+1+nx+(iz-1)*ny+1]
-xv[nlocal+0+ny+nx*ny+1+nx+(iz+1)*ny]
-xv[nlocal+0+ny+nx*ny+1+nx+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+(iz-1)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+1+nx+(iz-1)*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+1+nx+(iz)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+1+nx+(iz)*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+1+nx+(iz+1)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+1+nx+(iz+1)*ny+ny-1]
;}
ix=0;
iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+iz-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+iz]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+iz+1]
;}
iy=ny-1;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+(iz-1)*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+(iz-1)*nx+1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+(iz)*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+(iz-1)*nx+1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+(iz+1)*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+(iz)*nx+nx-2]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+(iz)*nx+nx-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+(iz+1)*nx+nx-1]
;}
;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+iy-1+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+1+nx+iy+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+1+nx+iy+1+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+1+nx+iy-1+(iz)*ny]
-xv[nlocal+0+ny+nx*ny+1+nx+iy+iz*ny]
-xv[nlocal+0+ny+nx*ny+1+nx+iy+1+iz*ny]
-xv[nlocal+0+ny+nx*ny+1+nx+iy-1+(iz+1)*ny]
-xv[nlocal+0+ny+nx*ny+1+nx+iy+(iz+1)*ny]
-xv[nlocal+0+ny+nx*ny+1+nx+iy+1+(iz+1)*ny]
;}
}
iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix-1+(iz-1)*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix+(iz-1)*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix+1+(iz-1)*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix-1+(iz)*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix+iz*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix+1+iz*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix-1+(iz+1)*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix+(iz+1)*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix+1+(iz+1)*nx]
;}
}
iz=0;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+ix-1+(iy-1)*nx]
-xv[nlocal+0+ny+ix+(iy-1)*nx]
-xv[nlocal+0+ny+ix+1+(iy-1)*nx]
-xv[nlocal+0+ny+ix-1+(iy)*nx]
-xv[nlocal+0+ny+ix+iy*nx]
-xv[nlocal+0+ny+ix+1+iy*nx]
-xv[nlocal+0+ny+ix-1+(iy+1)*nx]
-xv[nlocal+0+ny+ix+(iy+1)*nx]
-xv[nlocal+0+ny+ix+1+(iy+1)*nx]
;}
}
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+ix-1+(iy-1)*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+ix+(iy-1)*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+ix+1+(iy-1)*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+ix-1+(iy)*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+ix+iy*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+ix+1+iy*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+ix-1+(iy+1)*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+ix+(iy+1)*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+ix+1+(iy+1)*nx]
;}
}
}//ipz < npz - 1
            else
{
ix = 0;
iy = ny-1;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx*ny]
-xv[nlocal+0+ny+nx*ny+1]
-xv[nlocal+0+ny+nx*ny+1+1]
;
ix = nx-1;
iy = ny-1;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx*ny+1+nx-2]
-xv[nlocal+0+ny+nx*ny+1+nx-1]
;
iz=0;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+ix-1]
-xv[nlocal+0+ny+ix]
-xv[nlocal+0+ny+ix+1]
-xv[nlocal+0+ny+nx+ix-1]
-xv[nlocal+0+ny+nx+ix]
-xv[nlocal+0+ny+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+ny+ix+(ny-2)*nx]
-xv[nlocal+0+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+ny+ix+(ny-1)*nx]
-xv[nlocal+0+ny+ix+(ny-1)*nx+1]
;}
iy=ny-1;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+ix-1]
-xv[nlocal+0+ny+nx*ny+1+ix]
-xv[nlocal+0+ny+nx*ny+1+ix+1]
;}
iy=ny-1;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix+1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx+ix-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx+ix]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix+(nz-2)*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix+(nz-1)*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix+(nz-1)*nx+1]
;}
;
ix=0;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1]
-xv[nlocal+0+iy]
-xv[nlocal+0+iy+1]
;}
ix=0;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+iy-1]
-xv[nlocal+0+ny+nx*ny+1+nx+iy]
-xv[nlocal+0+ny+nx*ny+1+nx+iy+1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny+iy-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny+iy]
-xv[nlocal+0+ny+nx*ny+1+nx+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+iy+(nz-2)*ny-1]
-xv[nlocal+0+ny+nx*ny+1+nx+iy+(nz-2)*ny]
-xv[nlocal+0+ny+nx*ny+1+nx+iy+(nz-2)*ny+1]
-xv[nlocal+0+ny+nx*ny+1+nx+iy+(nz-1)*ny-1]
-xv[nlocal+0+ny+nx*ny+1+nx+iy+(nz-1)*ny]
-xv[nlocal+0+ny+nx*ny+1+nx+iy+(nz-1)*ny+1]
;}
iz=0;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+(iy-1)*nx]
-xv[nlocal+0+ny+(iy-1)*nx+1]
-xv[nlocal+0+ny+(iy)*nx]
-xv[nlocal+0+ny+(iy-1)*nx+1]
-xv[nlocal+0+ny+(iy+1)*nx]
-xv[nlocal+0+ny+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+ny+(iy)*nx+nx-2]
-xv[nlocal+0+ny+(iy)*nx+nx-1]
-xv[nlocal+0+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+ny+(iy+1)*nx+nx-1]
;}
;
ix=0;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+1+nx+(iz-1)*ny+1]
-xv[nlocal+0+ny+nx*ny+1+nx+(iz)*ny]
-xv[nlocal+0+ny+nx*ny+1+nx+(iz-1)*ny+1]
-xv[nlocal+0+ny+nx*ny+1+nx+(iz+1)*ny]
-xv[nlocal+0+ny+nx*ny+1+nx+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+(iz-1)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+1+nx+(iz-1)*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+1+nx+(iz)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+1+nx+(iz)*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+1+nx+(iz+1)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+1+nx+(iz+1)*ny+ny-1]
;}
ix=0;
iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+iz-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+iz]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+iz+1]
;}
iy=ny-1;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+(iz-1)*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+(iz-1)*nx+1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+(iz)*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+(iz-1)*nx+1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+(iz+1)*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+(iz)*nx+nx-2]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+(iz)*nx+nx-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+(iz+1)*nx+nx-1]
;}
;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+iy-1+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+1+nx+iy+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+1+nx+iy+1+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+1+nx+iy-1+(iz)*ny]
-xv[nlocal+0+ny+nx*ny+1+nx+iy+iz*ny]
-xv[nlocal+0+ny+nx*ny+1+nx+iy+1+iz*ny]
-xv[nlocal+0+ny+nx*ny+1+nx+iy-1+(iz+1)*ny]
-xv[nlocal+0+ny+nx*ny+1+nx+iy+(iz+1)*ny]
-xv[nlocal+0+ny+nx*ny+1+nx+iy+1+(iz+1)*ny]
;}
}
iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix-1+(iz-1)*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix+(iz-1)*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix+1+(iz-1)*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix-1+(iz)*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix+iz*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix+1+iz*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix-1+(iz+1)*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix+(iz+1)*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix+1+(iz+1)*nx]
;}
}
iz=0;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+ix-1+(iy-1)*nx]
-xv[nlocal+0+ny+ix+(iy-1)*nx]
-xv[nlocal+0+ny+ix+1+(iy-1)*nx]
-xv[nlocal+0+ny+ix-1+(iy)*nx]
-xv[nlocal+0+ny+ix+iy*nx]
-xv[nlocal+0+ny+ix+1+iy*nx]
-xv[nlocal+0+ny+ix-1+(iy+1)*nx]
-xv[nlocal+0+ny+ix+(iy+1)*nx]
-xv[nlocal+0+ny+ix+1+(iy+1)*nx]
;}
}
}//ipz < npz - 1
          }//ipz > 0 
        else
{
            if(ipz < npz - 1)
{
ix = 0;
iy = ny-1;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+nx*ny]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+nx*ny+1]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+nx*ny+1+1]
;
ix = nx-1;
iy = ny-1;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+nx*ny+1+nx-2]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+nx*ny+1+nx-1]
;
iz=nz-1;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+ix-1]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+ix]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+ix+1]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+nx+ix-1]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+nx+ix]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+ix+(ny-2)*nx]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+ix+(ny-1)*nx]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+ix+(ny-1)*nx+1]
;}
iy=ny-1;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+nz+ix-1]
-xv[nlocal+0+ny*nz+nz+ix]
-xv[nlocal+0+ny*nz+nz+ix+1]
-xv[nlocal+0+ny*nz+nz+nx+ix-1]
-xv[nlocal+0+ny*nz+nz+nx+ix]
-xv[nlocal+0+ny*nz+nz+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+ny*nz+nz+ix+(nz-2)*nx]
-xv[nlocal+0+ny*nz+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+ny*nz+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+ny*nz+nz+ix+(nz-1)*nx]
-xv[nlocal+0+ny*nz+nz+ix+(nz-1)*nx+1]
;}
iy=ny-1;
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+nx*ny+1+ix-1]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+nx*ny+1+ix]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+nx*ny+1+ix+1]
;}
;
ix=0;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1]
-xv[nlocal+0+iy]
-xv[nlocal+0+iy+1]
-xv[nlocal+0+ny+iy-1]
-xv[nlocal+0+ny+iy]
-xv[nlocal+0+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy+(nz-2)*ny-1]
-xv[nlocal+0+iy+(nz-2)*ny]
-xv[nlocal+0+iy+(nz-2)*ny+1]
-xv[nlocal+0+iy+(nz-1)*ny-1]
-xv[nlocal+0+iy+(nz-1)*ny]
-xv[nlocal+0+iy+(nz-1)*ny+1]
;}
ix=0;
iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+nz+nx*nz+iy-1]
-xv[nlocal+0+ny*nz+nz+nx*nz+iy]
-xv[nlocal+0+ny*nz+nz+nx*nz+iy+1]
;}
iz=nz-1;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+(iy-1)*nx]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+(iy-1)*nx+1]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+(iy)*nx]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+(iy-1)*nx+1]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+(iy+1)*nx]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+(iy)*nx+nx-2]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+(iy)*nx+nx-1]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+(iy+1)*nx+nx-1]
;}
;
ix=0;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*ny]
-xv[nlocal+0+(iz-1)*ny+1]
-xv[nlocal+0+(iz)*ny]
-xv[nlocal+0+(iz-1)*ny+1]
-xv[nlocal+0+(iz+1)*ny]
-xv[nlocal+0+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*ny+ny-2]
-xv[nlocal+0+(iz-1)*ny+ny-1]
-xv[nlocal+0+(iz)*ny+ny-2]
-xv[nlocal+0+(iz)*ny+ny-1]
-xv[nlocal+0+(iz+1)*ny+ny-2]
-xv[nlocal+0+(iz+1)*ny+ny-1]
;}
ix=0;
iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+iz-1]
-xv[nlocal+0+ny*nz+iz]
-xv[nlocal+0+ny*nz+iz+1]
;}
iy=ny-1;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+nz+(iz-1)*nx]
-xv[nlocal+0+ny*nz+nz+(iz-1)*nx+1]
-xv[nlocal+0+ny*nz+nz+(iz)*nx]
-xv[nlocal+0+ny*nz+nz+(iz-1)*nx+1]
-xv[nlocal+0+ny*nz+nz+(iz+1)*nx]
-xv[nlocal+0+ny*nz+nz+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+ny*nz+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+ny*nz+nz+(iz)*nx+nx-2]
-xv[nlocal+0+ny*nz+nz+(iz)*nx+nx-1]
-xv[nlocal+0+ny*nz+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+ny*nz+nz+(iz+1)*nx+nx-1]
;}
;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1+(iz-1)*ny]
-xv[nlocal+0+iy+(iz-1)*ny]
-xv[nlocal+0+iy+1+(iz-1)*ny]
-xv[nlocal+0+iy-1+(iz)*ny]
-xv[nlocal+0+iy+iz*ny]
-xv[nlocal+0+iy+1+iz*ny]
-xv[nlocal+0+iy-1+(iz+1)*ny]
-xv[nlocal+0+iy+(iz+1)*ny]
-xv[nlocal+0+iy+1+(iz+1)*ny]
;}
}
iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+nz+ix-1+(iz-1)*nx]
-xv[nlocal+0+ny*nz+nz+ix+(iz-1)*nx]
-xv[nlocal+0+ny*nz+nz+ix+1+(iz-1)*nx]
-xv[nlocal+0+ny*nz+nz+ix-1+(iz)*nx]
-xv[nlocal+0+ny*nz+nz+ix+iz*nx]
-xv[nlocal+0+ny*nz+nz+ix+1+iz*nx]
-xv[nlocal+0+ny*nz+nz+ix-1+(iz+1)*nx]
-xv[nlocal+0+ny*nz+nz+ix+(iz+1)*nx]
-xv[nlocal+0+ny*nz+nz+ix+1+(iz+1)*nx]
;}
}
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+ix-1+(iy-1)*nx]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+ix+(iy-1)*nx]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+ix+1+(iy-1)*nx]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+ix-1+(iy)*nx]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+ix+iy*nx]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+ix+1+iy*nx]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+ix-1+(iy+1)*nx]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+ix+(iy+1)*nx]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+ix+1+(iy+1)*nx]
;}
}
}//ipz < npz - 1
            else
{
iy=ny-1;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+nz+ix-1]
-xv[nlocal+0+ny*nz+nz+ix]
-xv[nlocal+0+ny*nz+nz+ix+1]
-xv[nlocal+0+ny*nz+nz+nx+ix-1]
-xv[nlocal+0+ny*nz+nz+nx+ix]
-xv[nlocal+0+ny*nz+nz+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+ny*nz+nz+ix+(nz-2)*nx]
-xv[nlocal+0+ny*nz+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+ny*nz+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+ny*nz+nz+ix+(nz-1)*nx]
-xv[nlocal+0+ny*nz+nz+ix+(nz-1)*nx+1]
;}
;
ix=0;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1]
-xv[nlocal+0+iy]
-xv[nlocal+0+iy+1]
-xv[nlocal+0+ny+iy-1]
-xv[nlocal+0+ny+iy]
-xv[nlocal+0+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy+(nz-2)*ny-1]
-xv[nlocal+0+iy+(nz-2)*ny]
-xv[nlocal+0+iy+(nz-2)*ny+1]
-xv[nlocal+0+iy+(nz-1)*ny-1]
-xv[nlocal+0+iy+(nz-1)*ny]
-xv[nlocal+0+iy+(nz-1)*ny+1]
;}
;
ix=0;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*ny]
-xv[nlocal+0+(iz-1)*ny+1]
-xv[nlocal+0+(iz)*ny]
-xv[nlocal+0+(iz-1)*ny+1]
-xv[nlocal+0+(iz+1)*ny]
-xv[nlocal+0+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*ny+ny-2]
-xv[nlocal+0+(iz-1)*ny+ny-1]
-xv[nlocal+0+(iz)*ny+ny-2]
-xv[nlocal+0+(iz)*ny+ny-1]
-xv[nlocal+0+(iz+1)*ny+ny-2]
-xv[nlocal+0+(iz+1)*ny+ny-1]
;}
ix=0;
iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+iz-1]
-xv[nlocal+0+ny*nz+iz]
-xv[nlocal+0+ny*nz+iz+1]
;}
iy=ny-1;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+nz+(iz-1)*nx]
-xv[nlocal+0+ny*nz+nz+(iz-1)*nx+1]
-xv[nlocal+0+ny*nz+nz+(iz)*nx]
-xv[nlocal+0+ny*nz+nz+(iz-1)*nx+1]
-xv[nlocal+0+ny*nz+nz+(iz+1)*nx]
-xv[nlocal+0+ny*nz+nz+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+ny*nz+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+ny*nz+nz+(iz)*nx+nx-2]
-xv[nlocal+0+ny*nz+nz+(iz)*nx+nx-1]
-xv[nlocal+0+ny*nz+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+ny*nz+nz+(iz+1)*nx+nx-1]
;}
;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1+(iz-1)*ny]
-xv[nlocal+0+iy+(iz-1)*ny]
-xv[nlocal+0+iy+1+(iz-1)*ny]
-xv[nlocal+0+iy-1+(iz)*ny]
-xv[nlocal+0+iy+iz*ny]
-xv[nlocal+0+iy+1+iz*ny]
-xv[nlocal+0+iy-1+(iz+1)*ny]
-xv[nlocal+0+iy+(iz+1)*ny]
-xv[nlocal+0+iy+1+(iz+1)*ny]
;}
}
iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+nz+ix-1+(iz-1)*nx]
-xv[nlocal+0+ny*nz+nz+ix+(iz-1)*nx]
-xv[nlocal+0+ny*nz+nz+ix+1+(iz-1)*nx]
-xv[nlocal+0+ny*nz+nz+ix-1+(iz)*nx]
-xv[nlocal+0+ny*nz+nz+ix+iz*nx]
-xv[nlocal+0+ny*nz+nz+ix+1+iz*nx]
-xv[nlocal+0+ny*nz+nz+ix-1+(iz+1)*nx]
-xv[nlocal+0+ny*nz+nz+ix+(iz+1)*nx]
-xv[nlocal+0+ny*nz+nz+ix+1+(iz+1)*nx]
;}
}
}//ipz < npz - 1
          }//ipz > 0 
        }//ipy < npy - 1 
      else
{
        if(ipz > 0)
{
            if(ipz < npz - 1)
{
iz=0;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+ix-1]
-xv[nlocal+0+ny+ix]
-xv[nlocal+0+ny+ix+1]
-xv[nlocal+0+ny+nx+ix-1]
-xv[nlocal+0+ny+nx+ix]
-xv[nlocal+0+ny+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+ny+ix+(ny-2)*nx]
-xv[nlocal+0+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+ny+ix+(ny-1)*nx]
-xv[nlocal+0+ny+ix+(ny-1)*nx+1]
;}
iz=nz-1;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+ix-1]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+ix]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+ix+1]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+nx+ix-1]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+nx+ix]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+ix+(ny-2)*nx]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+ix+(ny-1)*nx]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+ix+(ny-1)*nx+1]
;}
;
ix=0;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1]
-xv[nlocal+0+iy]
-xv[nlocal+0+iy+1]
;}
ix=0;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+iy-1]
-xv[nlocal+0+ny+nx*ny+iy]
-xv[nlocal+0+ny+nx*ny+iy+1]
-xv[nlocal+0+ny+nx*ny+ny+iy-1]
-xv[nlocal+0+ny+nx*ny+ny+iy]
-xv[nlocal+0+ny+nx*ny+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+iy+(nz-2)*ny-1]
-xv[nlocal+0+ny+nx*ny+iy+(nz-2)*ny]
-xv[nlocal+0+ny+nx*ny+iy+(nz-2)*ny+1]
-xv[nlocal+0+ny+nx*ny+iy+(nz-1)*ny-1]
-xv[nlocal+0+ny+nx*ny+iy+(nz-1)*ny]
-xv[nlocal+0+ny+nx*ny+iy+(nz-1)*ny+1]
;}
ix=0;
iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny*nz+iy-1]
-xv[nlocal+0+ny+nx*ny+ny*nz+iy]
-xv[nlocal+0+ny+nx*ny+ny*nz+iy+1]
;}
iz=0;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+(iy-1)*nx]
-xv[nlocal+0+ny+(iy-1)*nx+1]
-xv[nlocal+0+ny+(iy)*nx]
-xv[nlocal+0+ny+(iy-1)*nx+1]
-xv[nlocal+0+ny+(iy+1)*nx]
-xv[nlocal+0+ny+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+ny+(iy)*nx+nx-2]
-xv[nlocal+0+ny+(iy)*nx+nx-1]
-xv[nlocal+0+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+ny+(iy+1)*nx+nx-1]
;}
iz=nz-1;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+(iy-1)*nx]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+(iy-1)*nx+1]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+(iy)*nx]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+(iy-1)*nx+1]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+(iy+1)*nx]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+(iy)*nx+nx-2]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+(iy)*nx+nx-1]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+(iy+1)*nx+nx-1]
;}
;
ix=0;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+(iz-1)*ny+1]
-xv[nlocal+0+ny+nx*ny+(iz)*ny]
-xv[nlocal+0+ny+nx*ny+(iz-1)*ny+1]
-xv[nlocal+0+ny+nx*ny+(iz+1)*ny]
-xv[nlocal+0+ny+nx*ny+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+(iz-1)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+(iz-1)*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+(iz)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+(iz)*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+(iz+1)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+(iz+1)*ny+ny-1]
;}
;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+iy-1+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+iy+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+iy+1+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+iy-1+(iz)*ny]
-xv[nlocal+0+ny+nx*ny+iy+iz*ny]
-xv[nlocal+0+ny+nx*ny+iy+1+iz*ny]
-xv[nlocal+0+ny+nx*ny+iy-1+(iz+1)*ny]
-xv[nlocal+0+ny+nx*ny+iy+(iz+1)*ny]
-xv[nlocal+0+ny+nx*ny+iy+1+(iz+1)*ny]
;}
}
iz=0;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+ix-1+(iy-1)*nx]
-xv[nlocal+0+ny+ix+(iy-1)*nx]
-xv[nlocal+0+ny+ix+1+(iy-1)*nx]
-xv[nlocal+0+ny+ix-1+(iy)*nx]
-xv[nlocal+0+ny+ix+iy*nx]
-xv[nlocal+0+ny+ix+1+iy*nx]
-xv[nlocal+0+ny+ix-1+(iy+1)*nx]
-xv[nlocal+0+ny+ix+(iy+1)*nx]
-xv[nlocal+0+ny+ix+1+(iy+1)*nx]
;}
}
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+ix-1+(iy-1)*nx]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+ix+(iy-1)*nx]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+ix+1+(iy-1)*nx]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+ix-1+(iy)*nx]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+ix+iy*nx]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+ix+1+iy*nx]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+ix-1+(iy+1)*nx]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+ix+(iy+1)*nx]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+ix+1+(iy+1)*nx]
;}
}
}//ipz < npz - 1
            else
{
iz=0;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+ix-1]
-xv[nlocal+0+ny+ix]
-xv[nlocal+0+ny+ix+1]
-xv[nlocal+0+ny+nx+ix-1]
-xv[nlocal+0+ny+nx+ix]
-xv[nlocal+0+ny+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+ny+ix+(ny-2)*nx]
-xv[nlocal+0+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+ny+ix+(ny-1)*nx]
-xv[nlocal+0+ny+ix+(ny-1)*nx+1]
;}
;
ix=0;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1]
-xv[nlocal+0+iy]
-xv[nlocal+0+iy+1]
;}
ix=0;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+iy-1]
-xv[nlocal+0+ny+nx*ny+iy]
-xv[nlocal+0+ny+nx*ny+iy+1]
-xv[nlocal+0+ny+nx*ny+ny+iy-1]
-xv[nlocal+0+ny+nx*ny+ny+iy]
-xv[nlocal+0+ny+nx*ny+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+iy+(nz-2)*ny-1]
-xv[nlocal+0+ny+nx*ny+iy+(nz-2)*ny]
-xv[nlocal+0+ny+nx*ny+iy+(nz-2)*ny+1]
-xv[nlocal+0+ny+nx*ny+iy+(nz-1)*ny-1]
-xv[nlocal+0+ny+nx*ny+iy+(nz-1)*ny]
-xv[nlocal+0+ny+nx*ny+iy+(nz-1)*ny+1]
;}
iz=0;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+(iy-1)*nx]
-xv[nlocal+0+ny+(iy-1)*nx+1]
-xv[nlocal+0+ny+(iy)*nx]
-xv[nlocal+0+ny+(iy-1)*nx+1]
-xv[nlocal+0+ny+(iy+1)*nx]
-xv[nlocal+0+ny+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+ny+(iy)*nx+nx-2]
-xv[nlocal+0+ny+(iy)*nx+nx-1]
-xv[nlocal+0+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+ny+(iy+1)*nx+nx-1]
;}
;
ix=0;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+(iz-1)*ny+1]
-xv[nlocal+0+ny+nx*ny+(iz)*ny]
-xv[nlocal+0+ny+nx*ny+(iz-1)*ny+1]
-xv[nlocal+0+ny+nx*ny+(iz+1)*ny]
-xv[nlocal+0+ny+nx*ny+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+(iz-1)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+(iz-1)*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+(iz)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+(iz)*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+(iz+1)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+(iz+1)*ny+ny-1]
;}
;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+iy-1+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+iy+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+iy+1+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+iy-1+(iz)*ny]
-xv[nlocal+0+ny+nx*ny+iy+iz*ny]
-xv[nlocal+0+ny+nx*ny+iy+1+iz*ny]
-xv[nlocal+0+ny+nx*ny+iy-1+(iz+1)*ny]
-xv[nlocal+0+ny+nx*ny+iy+(iz+1)*ny]
-xv[nlocal+0+ny+nx*ny+iy+1+(iz+1)*ny]
;}
}
iz=0;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+ix-1+(iy-1)*nx]
-xv[nlocal+0+ny+ix+(iy-1)*nx]
-xv[nlocal+0+ny+ix+1+(iy-1)*nx]
-xv[nlocal+0+ny+ix-1+(iy)*nx]
-xv[nlocal+0+ny+ix+iy*nx]
-xv[nlocal+0+ny+ix+1+iy*nx]
-xv[nlocal+0+ny+ix-1+(iy+1)*nx]
-xv[nlocal+0+ny+ix+(iy+1)*nx]
-xv[nlocal+0+ny+ix+1+(iy+1)*nx]
;}
}
}//ipz < npz - 1
          }//ipz > 0 
        else
{
            if(ipz < npz - 1)
{
iz=nz-1;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny+ix-1]
-xv[nlocal+0+ny*nz+ny+ix]
-xv[nlocal+0+ny*nz+ny+ix+1]
-xv[nlocal+0+ny*nz+ny+nx+ix-1]
-xv[nlocal+0+ny*nz+ny+nx+ix]
-xv[nlocal+0+ny*nz+ny+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+ny*nz+ny+ix+(ny-2)*nx]
-xv[nlocal+0+ny*nz+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+ny*nz+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+ny*nz+ny+ix+(ny-1)*nx]
-xv[nlocal+0+ny*nz+ny+ix+(ny-1)*nx+1]
;}
;
ix=0;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1]
-xv[nlocal+0+iy]
-xv[nlocal+0+iy+1]
-xv[nlocal+0+ny+iy-1]
-xv[nlocal+0+ny+iy]
-xv[nlocal+0+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy+(nz-2)*ny-1]
-xv[nlocal+0+iy+(nz-2)*ny]
-xv[nlocal+0+iy+(nz-2)*ny+1]
-xv[nlocal+0+iy+(nz-1)*ny-1]
-xv[nlocal+0+iy+(nz-1)*ny]
-xv[nlocal+0+iy+(nz-1)*ny+1]
;}
ix=0;
iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+iy-1]
-xv[nlocal+0+ny*nz+iy]
-xv[nlocal+0+ny*nz+iy+1]
;}
iz=nz-1;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny+(iy-1)*nx]
-xv[nlocal+0+ny*nz+ny+(iy-1)*nx+1]
-xv[nlocal+0+ny*nz+ny+(iy)*nx]
-xv[nlocal+0+ny*nz+ny+(iy-1)*nx+1]
-xv[nlocal+0+ny*nz+ny+(iy+1)*nx]
-xv[nlocal+0+ny*nz+ny+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+ny*nz+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+ny*nz+ny+(iy)*nx+nx-2]
-xv[nlocal+0+ny*nz+ny+(iy)*nx+nx-1]
-xv[nlocal+0+ny*nz+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+ny*nz+ny+(iy+1)*nx+nx-1]
;}
;
ix=0;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*ny]
-xv[nlocal+0+(iz-1)*ny+1]
-xv[nlocal+0+(iz)*ny]
-xv[nlocal+0+(iz-1)*ny+1]
-xv[nlocal+0+(iz+1)*ny]
-xv[nlocal+0+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*ny+ny-2]
-xv[nlocal+0+(iz-1)*ny+ny-1]
-xv[nlocal+0+(iz)*ny+ny-2]
-xv[nlocal+0+(iz)*ny+ny-1]
-xv[nlocal+0+(iz+1)*ny+ny-2]
-xv[nlocal+0+(iz+1)*ny+ny-1]
;}
;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1+(iz-1)*ny]
-xv[nlocal+0+iy+(iz-1)*ny]
-xv[nlocal+0+iy+1+(iz-1)*ny]
-xv[nlocal+0+iy-1+(iz)*ny]
-xv[nlocal+0+iy+iz*ny]
-xv[nlocal+0+iy+1+iz*ny]
-xv[nlocal+0+iy-1+(iz+1)*ny]
-xv[nlocal+0+iy+(iz+1)*ny]
-xv[nlocal+0+iy+1+(iz+1)*ny]
;}
}
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny+ix-1+(iy-1)*nx]
-xv[nlocal+0+ny*nz+ny+ix+(iy-1)*nx]
-xv[nlocal+0+ny*nz+ny+ix+1+(iy-1)*nx]
-xv[nlocal+0+ny*nz+ny+ix-1+(iy)*nx]
-xv[nlocal+0+ny*nz+ny+ix+iy*nx]
-xv[nlocal+0+ny*nz+ny+ix+1+iy*nx]
-xv[nlocal+0+ny*nz+ny+ix-1+(iy+1)*nx]
-xv[nlocal+0+ny*nz+ny+ix+(iy+1)*nx]
-xv[nlocal+0+ny*nz+ny+ix+1+(iy+1)*nx]
;}
}
}//ipz < npz - 1
            else
{
;
ix=0;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1]
-xv[nlocal+0+iy]
-xv[nlocal+0+iy+1]
-xv[nlocal+0+ny+iy-1]
-xv[nlocal+0+ny+iy]
-xv[nlocal+0+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy+(nz-2)*ny-1]
-xv[nlocal+0+iy+(nz-2)*ny]
-xv[nlocal+0+iy+(nz-2)*ny+1]
-xv[nlocal+0+iy+(nz-1)*ny-1]
-xv[nlocal+0+iy+(nz-1)*ny]
-xv[nlocal+0+iy+(nz-1)*ny+1]
;}
;
ix=0;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*ny]
-xv[nlocal+0+(iz-1)*ny+1]
-xv[nlocal+0+(iz)*ny]
-xv[nlocal+0+(iz-1)*ny+1]
-xv[nlocal+0+(iz+1)*ny]
-xv[nlocal+0+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*ny+ny-2]
-xv[nlocal+0+(iz-1)*ny+ny-1]
-xv[nlocal+0+(iz)*ny+ny-2]
-xv[nlocal+0+(iz)*ny+ny-1]
-xv[nlocal+0+(iz+1)*ny+ny-2]
-xv[nlocal+0+(iz+1)*ny+ny-1]
;}
;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1+(iz-1)*ny]
-xv[nlocal+0+iy+(iz-1)*ny]
-xv[nlocal+0+iy+1+(iz-1)*ny]
-xv[nlocal+0+iy-1+(iz)*ny]
-xv[nlocal+0+iy+iz*ny]
-xv[nlocal+0+iy+1+iz*ny]
-xv[nlocal+0+iy-1+(iz+1)*ny]
-xv[nlocal+0+iy+(iz+1)*ny]
-xv[nlocal+0+iy+1+(iz+1)*ny]
;}
}
}//ipz < npz - 1
          }//ipz > 0 
        }//ipy < npy - 1 
      }//ipy > 0 
    }//ipx < npx - 1 
 }//ipx > 0 
else
{
  if(ipx < npx - 1)
{
    if(ipy > 0)
{
      if(ipy < npy - 1)
{
        if(ipz > 0)
{
            if(ipz < npz - 1)
{
ix = 0;
iy = 0;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
;
ix = 0;
iy = 0;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+1]
;
ix = 0;
iy = ny-1;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+1+nx*ny+ny]
-xv[nlocal+0+nx+1+nx*ny+ny+1]
;
ix = 0;
iy = ny-1;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny+ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny+ny+1]
;
ix = nx-1;
iy = 0;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx-2]
-xv[nlocal+0+nx-1]
-xv[nlocal+0+nx]
;
ix = nx-1;
iy = 0;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx]
;
ix = nx-1;
iy = ny-1;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+1+nx*ny+ny+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx]
;
ix = nx-1;
iy = ny-1;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny+ny+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny+ny+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny+ny+nx]
;
iy=0;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
;}
iy=0;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx+ix-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx+ix]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix+(nz-2)*nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix+(nz-2)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix+(nz-2)*nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix+(nz-1)*nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix+(nz-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix+(nz-1)*nx+1]
;}
iy=0;
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+ix-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+ix]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+ix+1]
;}
iz=0;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+ix-1]
-xv[nlocal+0+nx+1+ix]
-xv[nlocal+0+nx+1+ix+1]
-xv[nlocal+0+nx+1+nx+ix-1]
-xv[nlocal+0+nx+1+nx+ix]
-xv[nlocal+0+nx+1+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+ix+(ny-2)*nx-1]
-xv[nlocal+0+nx+1+ix+(ny-2)*nx]
-xv[nlocal+0+nx+1+ix+(ny-2)*nx+1]
-xv[nlocal+0+nx+1+ix+(ny-1)*nx-1]
-xv[nlocal+0+nx+1+ix+(ny-1)*nx]
-xv[nlocal+0+nx+1+ix+(ny-1)*nx+1]
;}
iz=nz-1;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx+ix-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx+ix]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix+(ny-2)*nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix+(ny-2)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix+(ny-2)*nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix+(ny-1)*nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix+(ny-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix+(ny-1)*nx+1]
;}
iy=ny-1;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+ix-1]
-xv[nlocal+0+nx+1+nx*ny+ny+ix]
-xv[nlocal+0+nx+1+nx*ny+ny+ix+1]
;}
iy=ny-1;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx+ix-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx+ix]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix+(nz-2)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix+(nz-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix+(nz-1)*nx+1]
;}
iy=ny-1;
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny+ny+ix-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny+ny+ix]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny+ny+ix+1]
;}
;
iz=0;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+(iy-1)*nx]
-xv[nlocal+0+nx+1+(iy-1)*nx+1]
-xv[nlocal+0+nx+1+(iy)*nx]
-xv[nlocal+0+nx+1+(iy-1)*nx+1]
-xv[nlocal+0+nx+1+(iy+1)*nx]
-xv[nlocal+0+nx+1+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+(iy-1)*nx+nx-2]
-xv[nlocal+0+nx+1+(iy-1)*nx+nx-1]
-xv[nlocal+0+nx+1+(iy)*nx+nx-2]
-xv[nlocal+0+nx+1+(iy)*nx+nx-1]
-xv[nlocal+0+nx+1+(iy+1)*nx+nx-2]
-xv[nlocal+0+nx+1+(iy+1)*nx+nx-1]
;}
iz=nz-1;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+(iy-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+(iy-1)*nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+(iy)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+(iy-1)*nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+(iy+1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+(iy-1)*nx+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+(iy-1)*nx+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+(iy)*nx+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+(iy)*nx+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+(iy+1)*nx+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+(iy+1)*nx+nx-1]
;}
ix=nx-1;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+iy-1]
-xv[nlocal+0+nx+1+nx*ny+iy]
-xv[nlocal+0+nx+1+nx*ny+iy+1]
;}
ix=nx-1;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny+iy-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny+iy]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy+(nz-2)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy+(nz-1)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy+(nz-1)*ny+1]
;}
ix=nx-1;
iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny+iy-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny+iy]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny+iy+1]
;}
;
iy=0;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+(iz-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+(iz-1)*nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+(iz)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+(iz-1)*nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+(iz+1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+(iz-1)*nx+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+(iz-1)*nx+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+(iz)*nx+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+(iz)*nx+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+(iz+1)*nx+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+(iz+1)*nx+nx-1]
;}
iy=ny-1;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+(iz-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+(iz-1)*nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+(iz)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+(iz-1)*nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+(iz+1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+(iz)*nx+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+(iz)*nx+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+(iz+1)*nx+nx-1]
;}
ix=nx-1;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+iz-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+iz]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+iz+1]
;}
ix=nx-1;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+(iz-1)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+(iz-1)*ny+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+(iz)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+(iz-1)*ny+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+(iz+1)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+(iz)*ny+ny-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+(iz)*ny+ny-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+(iz+1)*ny+ny-1]
;}
ix=nx-1;
iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+iz-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+iz]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+iz+1]
;}
;
ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy-1+(iz-1)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy+(iz-1)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy+1+(iz-1)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy-1+(iz)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy+iz*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy+1+iz*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy-1+(iz+1)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy+(iz+1)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy+1+(iz+1)*ny]
;}
}
iy=0;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix-1+(iz-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix+(iz-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix+1+(iz-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix-1+(iz)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix+iz*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix+1+iz*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix-1+(iz+1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix+(iz+1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix+1+(iz+1)*nx]
;}
}
iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix-1+(iz-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix+(iz-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix+1+(iz-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix-1+(iz)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix+iz*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix+1+iz*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix-1+(iz+1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix+(iz+1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix+1+(iz+1)*nx]
;}
}
iz=0;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+ix-1+(iy-1)*nx]
-xv[nlocal+0+nx+1+ix+(iy-1)*nx]
-xv[nlocal+0+nx+1+ix+1+(iy-1)*nx]
-xv[nlocal+0+nx+1+ix-1+(iy)*nx]
-xv[nlocal+0+nx+1+ix+iy*nx]
-xv[nlocal+0+nx+1+ix+1+iy*nx]
-xv[nlocal+0+nx+1+ix-1+(iy+1)*nx]
-xv[nlocal+0+nx+1+ix+(iy+1)*nx]
-xv[nlocal+0+nx+1+ix+1+(iy+1)*nx]
;}
}
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix-1+(iy-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix+(iy-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix+1+(iy-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix-1+(iy)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix+iy*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix+1+iy*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix-1+(iy+1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix+(iy+1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix+1+(iy+1)*nx]
;}
}
}//ipz < npz - 1
            else
{
ix = 0;
iy = 0;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
;
ix = 0;
iy = ny-1;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+1+nx*ny+ny]
-xv[nlocal+0+nx+1+nx*ny+ny+1]
;
ix = nx-1;
iy = 0;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx-2]
-xv[nlocal+0+nx-1]
-xv[nlocal+0+nx]
;
ix = nx-1;
iy = ny-1;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+1+nx*ny+ny+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx]
;
iy=0;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
;}
iy=0;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx+ix-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx+ix]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix+(nz-2)*nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix+(nz-2)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix+(nz-2)*nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix+(nz-1)*nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix+(nz-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix+(nz-1)*nx+1]
;}
iz=0;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+ix-1]
-xv[nlocal+0+nx+1+ix]
-xv[nlocal+0+nx+1+ix+1]
-xv[nlocal+0+nx+1+nx+ix-1]
-xv[nlocal+0+nx+1+nx+ix]
-xv[nlocal+0+nx+1+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+ix+(ny-2)*nx-1]
-xv[nlocal+0+nx+1+ix+(ny-2)*nx]
-xv[nlocal+0+nx+1+ix+(ny-2)*nx+1]
-xv[nlocal+0+nx+1+ix+(ny-1)*nx-1]
-xv[nlocal+0+nx+1+ix+(ny-1)*nx]
-xv[nlocal+0+nx+1+ix+(ny-1)*nx+1]
;}
iy=ny-1;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+ix-1]
-xv[nlocal+0+nx+1+nx*ny+ny+ix]
-xv[nlocal+0+nx+1+nx*ny+ny+ix+1]
;}
iy=ny-1;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx+ix-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx+ix]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix+(nz-2)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix+(nz-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix+(nz-1)*nx+1]
;}
;
iz=0;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+(iy-1)*nx]
-xv[nlocal+0+nx+1+(iy-1)*nx+1]
-xv[nlocal+0+nx+1+(iy)*nx]
-xv[nlocal+0+nx+1+(iy-1)*nx+1]
-xv[nlocal+0+nx+1+(iy+1)*nx]
-xv[nlocal+0+nx+1+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+(iy-1)*nx+nx-2]
-xv[nlocal+0+nx+1+(iy-1)*nx+nx-1]
-xv[nlocal+0+nx+1+(iy)*nx+nx-2]
-xv[nlocal+0+nx+1+(iy)*nx+nx-1]
-xv[nlocal+0+nx+1+(iy+1)*nx+nx-2]
-xv[nlocal+0+nx+1+(iy+1)*nx+nx-1]
;}
ix=nx-1;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+iy-1]
-xv[nlocal+0+nx+1+nx*ny+iy]
-xv[nlocal+0+nx+1+nx*ny+iy+1]
;}
ix=nx-1;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny+iy-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny+iy]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy+(nz-2)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy+(nz-1)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy+(nz-1)*ny+1]
;}
;
iy=0;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+(iz-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+(iz-1)*nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+(iz)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+(iz-1)*nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+(iz+1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+(iz-1)*nx+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+(iz-1)*nx+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+(iz)*nx+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+(iz)*nx+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+(iz+1)*nx+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+(iz+1)*nx+nx-1]
;}
iy=ny-1;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+(iz-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+(iz-1)*nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+(iz)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+(iz-1)*nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+(iz+1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+(iz)*nx+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+(iz)*nx+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+(iz+1)*nx+nx-1]
;}
ix=nx-1;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+iz-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+iz]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+iz+1]
;}
ix=nx-1;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+(iz-1)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+(iz-1)*ny+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+(iz)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+(iz-1)*ny+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+(iz+1)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+(iz)*ny+ny-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+(iz)*ny+ny-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+(iz+1)*ny+ny-1]
;}
ix=nx-1;
iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+iz-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+iz]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+iz+1]
;}
;
ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy-1+(iz-1)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy+(iz-1)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy+1+(iz-1)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy-1+(iz)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy+iz*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy+1+iz*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy-1+(iz+1)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy+(iz+1)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy+1+(iz+1)*ny]
;}
}
iy=0;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix-1+(iz-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix+(iz-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix+1+(iz-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix-1+(iz)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix+iz*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix+1+iz*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix-1+(iz+1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix+(iz+1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix+1+(iz+1)*nx]
;}
}
iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix-1+(iz-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix+(iz-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix+1+(iz-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix-1+(iz)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix+iz*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix+1+iz*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix-1+(iz+1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix+(iz+1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix+1+(iz+1)*nx]
;}
}
iz=0;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+ix-1+(iy-1)*nx]
-xv[nlocal+0+nx+1+ix+(iy-1)*nx]
-xv[nlocal+0+nx+1+ix+1+(iy-1)*nx]
-xv[nlocal+0+nx+1+ix-1+(iy)*nx]
-xv[nlocal+0+nx+1+ix+iy*nx]
-xv[nlocal+0+nx+1+ix+1+iy*nx]
-xv[nlocal+0+nx+1+ix-1+(iy+1)*nx]
-xv[nlocal+0+nx+1+ix+(iy+1)*nx]
-xv[nlocal+0+nx+1+ix+1+(iy+1)*nx]
;}
}
}//ipz < npz - 1
          }//ipz > 0 
        else
{
            if(ipz < npz - 1)
{
ix = 0;
iy = 0;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+1]
;
ix = 0;
iy = ny-1;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny+ny]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny+ny+1]
;
ix = nx-1;
iy = 0;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx-2]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx]
;
ix = nx-1;
iy = ny-1;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny+ny+nx-2]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny+ny+nx-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny+ny+nx]
;
iy=0;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
-xv[nlocal+0+nx+ix-1]
-xv[nlocal+0+nx+ix]
-xv[nlocal+0+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix+(nz-2)*nx-1]
-xv[nlocal+0+ix+(nz-2)*nx]
-xv[nlocal+0+ix+(nz-2)*nx+1]
-xv[nlocal+0+ix+(nz-1)*nx-1]
-xv[nlocal+0+ix+(nz-1)*nx]
-xv[nlocal+0+ix+(nz-1)*nx+1]
;}
iy=0;
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+ix-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+ix]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+ix+1]
;}
iz=nz-1;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix+1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx+ix-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx+ix]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix+(ny-2)*nx-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix+(ny-2)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix+(ny-2)*nx+1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix+(ny-1)*nx-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix+(ny-1)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix+(ny-1)*nx+1]
;}
iy=ny-1;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+ny*nz+ix-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+ix]
-xv[nlocal+0+nx*nz+nz+ny*nz+ix+1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+ix-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+ix]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+ny*nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+ix+(nz-2)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+nx*nz+nz+ny*nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+ix+(nz-1)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+ix+(nz-1)*nx+1]
;}
iy=ny-1;
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny+ny+ix-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny+ny+ix]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny+ny+ix+1]
;}
;
iz=nz-1;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+(iy-1)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+(iy-1)*nx+1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+(iy)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+(iy-1)*nx+1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+(iy+1)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+(iy-1)*nx+nx-2]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+(iy-1)*nx+nx-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+(iy)*nx+nx-2]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+(iy)*nx+nx-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+(iy+1)*nx+nx-2]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+(iy+1)*nx+nx-1]
;}
ix=nx-1;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+iy-1]
-xv[nlocal+0+nx*nz+nz+iy]
-xv[nlocal+0+nx*nz+nz+iy+1]
-xv[nlocal+0+nx*nz+nz+ny+iy-1]
-xv[nlocal+0+nx*nz+nz+ny+iy]
-xv[nlocal+0+nx*nz+nz+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+nx*nz+nz+iy+(nz-2)*ny]
-xv[nlocal+0+nx*nz+nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+nx*nz+nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+nx*nz+nz+iy+(nz-1)*ny]
-xv[nlocal+0+nx*nz+nz+iy+(nz-1)*ny+1]
;}
ix=nx-1;
iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny+iy-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny+iy]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny+iy+1]
;}
;
iy=0;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*nx]
-xv[nlocal+0+(iz-1)*nx+1]
-xv[nlocal+0+(iz)*nx]
-xv[nlocal+0+(iz-1)*nx+1]
-xv[nlocal+0+(iz+1)*nx]
-xv[nlocal+0+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*nx+nx-2]
-xv[nlocal+0+(iz-1)*nx+nx-1]
-xv[nlocal+0+(iz)*nx+nx-2]
-xv[nlocal+0+(iz)*nx+nx-1]
-xv[nlocal+0+(iz+1)*nx+nx-2]
-xv[nlocal+0+(iz+1)*nx+nx-1]
;}
iy=ny-1;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+ny*nz+(iz-1)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+(iz-1)*nx+1]
-xv[nlocal+0+nx*nz+nz+ny*nz+(iz)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+(iz-1)*nx+1]
-xv[nlocal+0+nx*nz+nz+ny*nz+(iz+1)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+ny*nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+nx*nz+nz+ny*nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+(iz)*nx+nx-2]
-xv[nlocal+0+nx*nz+nz+ny*nz+(iz)*nx+nx-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+nx*nz+nz+ny*nz+(iz+1)*nx+nx-1]
;}
ix=nx-1;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+iz-1]
-xv[nlocal+0+nx*nz+iz]
-xv[nlocal+0+nx*nz+iz+1]
;}
ix=nx-1;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+(iz-1)*ny]
-xv[nlocal+0+nx*nz+nz+(iz-1)*ny+1]
-xv[nlocal+0+nx*nz+nz+(iz)*ny]
-xv[nlocal+0+nx*nz+nz+(iz-1)*ny+1]
-xv[nlocal+0+nx*nz+nz+(iz+1)*ny]
-xv[nlocal+0+nx*nz+nz+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+nx*nz+nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+nx*nz+nz+(iz)*ny+ny-2]
-xv[nlocal+0+nx*nz+nz+(iz)*ny+ny-1]
-xv[nlocal+0+nx*nz+nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+nx*nz+nz+(iz+1)*ny+ny-1]
;}
ix=nx-1;
iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+iz-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+iz]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+iz+1]
;}
;
ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+iy-1+(iz-1)*ny]
-xv[nlocal+0+nx*nz+nz+iy+(iz-1)*ny]
-xv[nlocal+0+nx*nz+nz+iy+1+(iz-1)*ny]
-xv[nlocal+0+nx*nz+nz+iy-1+(iz)*ny]
-xv[nlocal+0+nx*nz+nz+iy+iz*ny]
-xv[nlocal+0+nx*nz+nz+iy+1+iz*ny]
-xv[nlocal+0+nx*nz+nz+iy-1+(iz+1)*ny]
-xv[nlocal+0+nx*nz+nz+iy+(iz+1)*ny]
-xv[nlocal+0+nx*nz+nz+iy+1+(iz+1)*ny]
;}
}
iy=0;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1+(iz-1)*nx]
-xv[nlocal+0+ix+(iz-1)*nx]
-xv[nlocal+0+ix+1+(iz-1)*nx]
-xv[nlocal+0+ix-1+(iz)*nx]
-xv[nlocal+0+ix+iz*nx]
-xv[nlocal+0+ix+1+iz*nx]
-xv[nlocal+0+ix-1+(iz+1)*nx]
-xv[nlocal+0+ix+(iz+1)*nx]
-xv[nlocal+0+ix+1+(iz+1)*nx]
;}
}
iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+ny*nz+ix-1+(iz-1)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+ix+(iz-1)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+ix+1+(iz-1)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+ix-1+(iz)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+ix+iz*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+ix+1+iz*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+ix-1+(iz+1)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+ix+(iz+1)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+ix+1+(iz+1)*nx]
;}
}
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix-1+(iy-1)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix+(iy-1)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix+1+(iy-1)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix-1+(iy)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix+iy*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix+1+iy*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix-1+(iy+1)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix+(iy+1)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix+1+(iy+1)*nx]
;}
}
}//ipz < npz - 1
            else
{
iy=0;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
-xv[nlocal+0+nx+ix-1]
-xv[nlocal+0+nx+ix]
-xv[nlocal+0+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix+(nz-2)*nx-1]
-xv[nlocal+0+ix+(nz-2)*nx]
-xv[nlocal+0+ix+(nz-2)*nx+1]
-xv[nlocal+0+ix+(nz-1)*nx-1]
-xv[nlocal+0+ix+(nz-1)*nx]
-xv[nlocal+0+ix+(nz-1)*nx+1]
;}
iy=ny-1;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+ny*nz+ix-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+ix]
-xv[nlocal+0+nx*nz+nz+ny*nz+ix+1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+ix-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+ix]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+ny*nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+ix+(nz-2)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+nx*nz+nz+ny*nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+ix+(nz-1)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+ix+(nz-1)*nx+1]
;}
;
ix=nx-1;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+iy-1]
-xv[nlocal+0+nx*nz+nz+iy]
-xv[nlocal+0+nx*nz+nz+iy+1]
-xv[nlocal+0+nx*nz+nz+ny+iy-1]
-xv[nlocal+0+nx*nz+nz+ny+iy]
-xv[nlocal+0+nx*nz+nz+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+nx*nz+nz+iy+(nz-2)*ny]
-xv[nlocal+0+nx*nz+nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+nx*nz+nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+nx*nz+nz+iy+(nz-1)*ny]
-xv[nlocal+0+nx*nz+nz+iy+(nz-1)*ny+1]
;}
;
iy=0;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*nx]
-xv[nlocal+0+(iz-1)*nx+1]
-xv[nlocal+0+(iz)*nx]
-xv[nlocal+0+(iz-1)*nx+1]
-xv[nlocal+0+(iz+1)*nx]
-xv[nlocal+0+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*nx+nx-2]
-xv[nlocal+0+(iz-1)*nx+nx-1]
-xv[nlocal+0+(iz)*nx+nx-2]
-xv[nlocal+0+(iz)*nx+nx-1]
-xv[nlocal+0+(iz+1)*nx+nx-2]
-xv[nlocal+0+(iz+1)*nx+nx-1]
;}
iy=ny-1;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+ny*nz+(iz-1)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+(iz-1)*nx+1]
-xv[nlocal+0+nx*nz+nz+ny*nz+(iz)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+(iz-1)*nx+1]
-xv[nlocal+0+nx*nz+nz+ny*nz+(iz+1)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+ny*nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+nx*nz+nz+ny*nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+(iz)*nx+nx-2]
-xv[nlocal+0+nx*nz+nz+ny*nz+(iz)*nx+nx-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+nx*nz+nz+ny*nz+(iz+1)*nx+nx-1]
;}
ix=nx-1;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+iz-1]
-xv[nlocal+0+nx*nz+iz]
-xv[nlocal+0+nx*nz+iz+1]
;}
ix=nx-1;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+(iz-1)*ny]
-xv[nlocal+0+nx*nz+nz+(iz-1)*ny+1]
-xv[nlocal+0+nx*nz+nz+(iz)*ny]
-xv[nlocal+0+nx*nz+nz+(iz-1)*ny+1]
-xv[nlocal+0+nx*nz+nz+(iz+1)*ny]
-xv[nlocal+0+nx*nz+nz+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+nx*nz+nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+nx*nz+nz+(iz)*ny+ny-2]
-xv[nlocal+0+nx*nz+nz+(iz)*ny+ny-1]
-xv[nlocal+0+nx*nz+nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+nx*nz+nz+(iz+1)*ny+ny-1]
;}
ix=nx-1;
iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+iz-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+iz]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+iz+1]
;}
;
ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+iy-1+(iz-1)*ny]
-xv[nlocal+0+nx*nz+nz+iy+(iz-1)*ny]
-xv[nlocal+0+nx*nz+nz+iy+1+(iz-1)*ny]
-xv[nlocal+0+nx*nz+nz+iy-1+(iz)*ny]
-xv[nlocal+0+nx*nz+nz+iy+iz*ny]
-xv[nlocal+0+nx*nz+nz+iy+1+iz*ny]
-xv[nlocal+0+nx*nz+nz+iy-1+(iz+1)*ny]
-xv[nlocal+0+nx*nz+nz+iy+(iz+1)*ny]
-xv[nlocal+0+nx*nz+nz+iy+1+(iz+1)*ny]
;}
}
iy=0;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1+(iz-1)*nx]
-xv[nlocal+0+ix+(iz-1)*nx]
-xv[nlocal+0+ix+1+(iz-1)*nx]
-xv[nlocal+0+ix-1+(iz)*nx]
-xv[nlocal+0+ix+iz*nx]
-xv[nlocal+0+ix+1+iz*nx]
-xv[nlocal+0+ix-1+(iz+1)*nx]
-xv[nlocal+0+ix+(iz+1)*nx]
-xv[nlocal+0+ix+1+(iz+1)*nx]
;}
}
iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+ny*nz+ix-1+(iz-1)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+ix+(iz-1)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+ix+1+(iz-1)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+ix-1+(iz)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+ix+iz*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+ix+1+iz*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+ix-1+(iz+1)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+ix+(iz+1)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+ix+1+(iz+1)*nx]
;}
}
}//ipz < npz - 1
          }//ipz > 0 
        }//ipy < npy - 1 
      else
{
        if(ipz > 0)
{
            if(ipz < npz - 1)
{
ix = 0;
iy = 0;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
;
ix = 0;
iy = 0;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+1]
;
ix = nx-1;
iy = 0;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx-2]
-xv[nlocal+0+nx-1]
-xv[nlocal+0+nx]
;
ix = nx-1;
iy = 0;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx]
;
iy=0;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
;}
iy=0;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+ix-1]
-xv[nlocal+0+nx+1+nx*ny+ny+ix]
-xv[nlocal+0+nx+1+nx*ny+ny+ix+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+ix-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+ix]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+ix+(nz-2)*nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+ix+(nz-2)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+ix+(nz-2)*nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+ix+(nz-1)*nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+ix+(nz-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+ix+(nz-1)*nx+1]
;}
iy=0;
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+ix-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+ix]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+ix+1]
;}
iz=0;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+ix-1]
-xv[nlocal+0+nx+1+ix]
-xv[nlocal+0+nx+1+ix+1]
-xv[nlocal+0+nx+1+nx+ix-1]
-xv[nlocal+0+nx+1+nx+ix]
-xv[nlocal+0+nx+1+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+ix+(ny-2)*nx-1]
-xv[nlocal+0+nx+1+ix+(ny-2)*nx]
-xv[nlocal+0+nx+1+ix+(ny-2)*nx+1]
-xv[nlocal+0+nx+1+ix+(ny-1)*nx-1]
-xv[nlocal+0+nx+1+ix+(ny-1)*nx]
-xv[nlocal+0+nx+1+ix+(ny-1)*nx+1]
;}
iz=nz-1;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+ix-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+ix]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+ix+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+nx+ix-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+nx+ix]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+ix+(ny-2)*nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+ix+(ny-2)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+ix+(ny-2)*nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+ix+(ny-1)*nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+ix+(ny-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+ix+(ny-1)*nx+1]
;}
;
iz=0;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+(iy-1)*nx]
-xv[nlocal+0+nx+1+(iy-1)*nx+1]
-xv[nlocal+0+nx+1+(iy)*nx]
-xv[nlocal+0+nx+1+(iy-1)*nx+1]
-xv[nlocal+0+nx+1+(iy+1)*nx]
-xv[nlocal+0+nx+1+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+(iy-1)*nx+nx-2]
-xv[nlocal+0+nx+1+(iy-1)*nx+nx-1]
-xv[nlocal+0+nx+1+(iy)*nx+nx-2]
-xv[nlocal+0+nx+1+(iy)*nx+nx-1]
-xv[nlocal+0+nx+1+(iy+1)*nx+nx-2]
-xv[nlocal+0+nx+1+(iy+1)*nx+nx-1]
;}
iz=nz-1;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+(iy-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+(iy-1)*nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+(iy)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+(iy-1)*nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+(iy+1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+(iy-1)*nx+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+(iy-1)*nx+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+(iy)*nx+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+(iy)*nx+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+(iy+1)*nx+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+(iy+1)*nx+nx-1]
;}
ix=nx-1;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+iy-1]
-xv[nlocal+0+nx+1+nx*ny+iy]
-xv[nlocal+0+nx+1+nx*ny+iy+1]
;}
ix=nx-1;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny+iy-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny+iy]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy+(nz-2)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy+(nz-1)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy+(nz-1)*ny+1]
;}
ix=nx-1;
iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+nx*ny+iy-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+nx*ny+iy]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+nx*ny+iy+1]
;}
;
iy=0;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+(iz-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+(iz-1)*nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+(iz)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+(iz-1)*nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+(iz+1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+(iz-1)*nx+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+(iz-1)*nx+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+(iz)*nx+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+(iz)*nx+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+(iz+1)*nx+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+(iz+1)*nx+nx-1]
;}
ix=nx-1;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+iz-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+iz]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+iz+1]
;}
ix=nx-1;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+(iz-1)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+(iz-1)*ny+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+(iz)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+(iz-1)*ny+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+(iz+1)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+(iz)*ny+ny-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+(iz)*ny+ny-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+(iz+1)*ny+ny-1]
;}
;
ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy-1+(iz-1)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy+(iz-1)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy+1+(iz-1)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy-1+(iz)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy+iz*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy+1+iz*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy-1+(iz+1)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy+(iz+1)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy+1+(iz+1)*ny]
;}
}
iy=0;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+ix-1+(iz-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+ix+(iz-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+ix+1+(iz-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+ix-1+(iz)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+ix+iz*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+ix+1+iz*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+ix-1+(iz+1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+ix+(iz+1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+ix+1+(iz+1)*nx]
;}
}
iz=0;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+ix-1+(iy-1)*nx]
-xv[nlocal+0+nx+1+ix+(iy-1)*nx]
-xv[nlocal+0+nx+1+ix+1+(iy-1)*nx]
-xv[nlocal+0+nx+1+ix-1+(iy)*nx]
-xv[nlocal+0+nx+1+ix+iy*nx]
-xv[nlocal+0+nx+1+ix+1+iy*nx]
-xv[nlocal+0+nx+1+ix-1+(iy+1)*nx]
-xv[nlocal+0+nx+1+ix+(iy+1)*nx]
-xv[nlocal+0+nx+1+ix+1+(iy+1)*nx]
;}
}
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+ix-1+(iy-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+ix+(iy-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+ix+1+(iy-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+ix-1+(iy)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+ix+iy*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+ix+1+iy*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+ix-1+(iy+1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+ix+(iy+1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+ix+1+(iy+1)*nx]
;}
}
}//ipz < npz - 1
            else
{
ix = 0;
iy = 0;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
;
ix = nx-1;
iy = 0;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx-2]
-xv[nlocal+0+nx-1]
-xv[nlocal+0+nx]
;
iy=0;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
;}
iy=0;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+ix-1]
-xv[nlocal+0+nx+1+nx*ny+ny+ix]
-xv[nlocal+0+nx+1+nx*ny+ny+ix+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+ix-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+ix]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+ix+(nz-2)*nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+ix+(nz-2)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+ix+(nz-2)*nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+ix+(nz-1)*nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+ix+(nz-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+ix+(nz-1)*nx+1]
;}
iz=0;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+ix-1]
-xv[nlocal+0+nx+1+ix]
-xv[nlocal+0+nx+1+ix+1]
-xv[nlocal+0+nx+1+nx+ix-1]
-xv[nlocal+0+nx+1+nx+ix]
-xv[nlocal+0+nx+1+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+ix+(ny-2)*nx-1]
-xv[nlocal+0+nx+1+ix+(ny-2)*nx]
-xv[nlocal+0+nx+1+ix+(ny-2)*nx+1]
-xv[nlocal+0+nx+1+ix+(ny-1)*nx-1]
-xv[nlocal+0+nx+1+ix+(ny-1)*nx]
-xv[nlocal+0+nx+1+ix+(ny-1)*nx+1]
;}
;
iz=0;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+(iy-1)*nx]
-xv[nlocal+0+nx+1+(iy-1)*nx+1]
-xv[nlocal+0+nx+1+(iy)*nx]
-xv[nlocal+0+nx+1+(iy-1)*nx+1]
-xv[nlocal+0+nx+1+(iy+1)*nx]
-xv[nlocal+0+nx+1+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+(iy-1)*nx+nx-2]
-xv[nlocal+0+nx+1+(iy-1)*nx+nx-1]
-xv[nlocal+0+nx+1+(iy)*nx+nx-2]
-xv[nlocal+0+nx+1+(iy)*nx+nx-1]
-xv[nlocal+0+nx+1+(iy+1)*nx+nx-2]
-xv[nlocal+0+nx+1+(iy+1)*nx+nx-1]
;}
ix=nx-1;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+iy-1]
-xv[nlocal+0+nx+1+nx*ny+iy]
-xv[nlocal+0+nx+1+nx*ny+iy+1]
;}
ix=nx-1;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny+iy-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny+iy]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy+(nz-2)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy+(nz-1)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy+(nz-1)*ny+1]
;}
;
iy=0;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+(iz-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+(iz-1)*nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+(iz)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+(iz-1)*nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+(iz+1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+(iz-1)*nx+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+(iz-1)*nx+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+(iz)*nx+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+(iz)*nx+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+(iz+1)*nx+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+(iz+1)*nx+nx-1]
;}
ix=nx-1;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+iz-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+iz]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+iz+1]
;}
ix=nx-1;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+(iz-1)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+(iz-1)*ny+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+(iz)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+(iz-1)*ny+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+(iz+1)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+(iz)*ny+ny-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+(iz)*ny+ny-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+(iz+1)*ny+ny-1]
;}
;
ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy-1+(iz-1)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy+(iz-1)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy+1+(iz-1)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy-1+(iz)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy+iz*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy+1+iz*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy-1+(iz+1)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy+(iz+1)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy+1+(iz+1)*ny]
;}
}
iy=0;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+ix-1+(iz-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+ix+(iz-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+ix+1+(iz-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+ix-1+(iz)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+ix+iz*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+ix+1+iz*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+ix-1+(iz+1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+ix+(iz+1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+ix+1+(iz+1)*nx]
;}
}
iz=0;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+ix-1+(iy-1)*nx]
-xv[nlocal+0+nx+1+ix+(iy-1)*nx]
-xv[nlocal+0+nx+1+ix+1+(iy-1)*nx]
-xv[nlocal+0+nx+1+ix-1+(iy)*nx]
-xv[nlocal+0+nx+1+ix+iy*nx]
-xv[nlocal+0+nx+1+ix+1+iy*nx]
-xv[nlocal+0+nx+1+ix-1+(iy+1)*nx]
-xv[nlocal+0+nx+1+ix+(iy+1)*nx]
-xv[nlocal+0+nx+1+ix+1+(iy+1)*nx]
;}
}
}//ipz < npz - 1
          }//ipz > 0 
        else
{
            if(ipz < npz - 1)
{
ix = 0;
iy = 0;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*nz+nz+ny*nz]
-xv[nlocal+0+nx*nz+nz+ny*nz+1]
;
ix = nx-1;
iy = 0;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*nz+nz+ny*nz+nx-2]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx]
;
iy=0;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
-xv[nlocal+0+nx+ix-1]
-xv[nlocal+0+nx+ix]
-xv[nlocal+0+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix+(nz-2)*nx-1]
-xv[nlocal+0+ix+(nz-2)*nx]
-xv[nlocal+0+ix+(nz-2)*nx+1]
-xv[nlocal+0+ix+(nz-1)*nx-1]
-xv[nlocal+0+ix+(nz-1)*nx]
-xv[nlocal+0+ix+(nz-1)*nx+1]
;}
iy=0;
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+ny*nz+ix-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+ix]
-xv[nlocal+0+nx*nz+nz+ny*nz+ix+1]
;}
iz=nz-1;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+ix-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+ix]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+ix+1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+nx+ix-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+nx+ix]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+ix+(ny-2)*nx-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+ix+(ny-2)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+ix+(ny-2)*nx+1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+ix+(ny-1)*nx-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+ix+(ny-1)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+ix+(ny-1)*nx+1]
;}
;
iz=nz-1;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+(iy-1)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+(iy-1)*nx+1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+(iy)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+(iy-1)*nx+1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+(iy+1)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+(iy-1)*nx+nx-2]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+(iy-1)*nx+nx-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+(iy)*nx+nx-2]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+(iy)*nx+nx-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+(iy+1)*nx+nx-2]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+(iy+1)*nx+nx-1]
;}
ix=nx-1;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+iy-1]
-xv[nlocal+0+nx*nz+nz+iy]
-xv[nlocal+0+nx*nz+nz+iy+1]
-xv[nlocal+0+nx*nz+nz+ny+iy-1]
-xv[nlocal+0+nx*nz+nz+ny+iy]
-xv[nlocal+0+nx*nz+nz+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+nx*nz+nz+iy+(nz-2)*ny]
-xv[nlocal+0+nx*nz+nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+nx*nz+nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+nx*nz+nz+iy+(nz-1)*ny]
-xv[nlocal+0+nx*nz+nz+iy+(nz-1)*ny+1]
;}
ix=nx-1;
iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+nx*ny+iy-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+nx*ny+iy]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+nx*ny+iy+1]
;}
;
iy=0;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*nx]
-xv[nlocal+0+(iz-1)*nx+1]
-xv[nlocal+0+(iz)*nx]
-xv[nlocal+0+(iz-1)*nx+1]
-xv[nlocal+0+(iz+1)*nx]
-xv[nlocal+0+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*nx+nx-2]
-xv[nlocal+0+(iz-1)*nx+nx-1]
-xv[nlocal+0+(iz)*nx+nx-2]
-xv[nlocal+0+(iz)*nx+nx-1]
-xv[nlocal+0+(iz+1)*nx+nx-2]
-xv[nlocal+0+(iz+1)*nx+nx-1]
;}
ix=nx-1;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+iz-1]
-xv[nlocal+0+nx*nz+iz]
-xv[nlocal+0+nx*nz+iz+1]
;}
ix=nx-1;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+(iz-1)*ny]
-xv[nlocal+0+nx*nz+nz+(iz-1)*ny+1]
-xv[nlocal+0+nx*nz+nz+(iz)*ny]
-xv[nlocal+0+nx*nz+nz+(iz-1)*ny+1]
-xv[nlocal+0+nx*nz+nz+(iz+1)*ny]
-xv[nlocal+0+nx*nz+nz+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+nx*nz+nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+nx*nz+nz+(iz)*ny+ny-2]
-xv[nlocal+0+nx*nz+nz+(iz)*ny+ny-1]
-xv[nlocal+0+nx*nz+nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+nx*nz+nz+(iz+1)*ny+ny-1]
;}
;
ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+iy-1+(iz-1)*ny]
-xv[nlocal+0+nx*nz+nz+iy+(iz-1)*ny]
-xv[nlocal+0+nx*nz+nz+iy+1+(iz-1)*ny]
-xv[nlocal+0+nx*nz+nz+iy-1+(iz)*ny]
-xv[nlocal+0+nx*nz+nz+iy+iz*ny]
-xv[nlocal+0+nx*nz+nz+iy+1+iz*ny]
-xv[nlocal+0+nx*nz+nz+iy-1+(iz+1)*ny]
-xv[nlocal+0+nx*nz+nz+iy+(iz+1)*ny]
-xv[nlocal+0+nx*nz+nz+iy+1+(iz+1)*ny]
;}
}
iy=0;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1+(iz-1)*nx]
-xv[nlocal+0+ix+(iz-1)*nx]
-xv[nlocal+0+ix+1+(iz-1)*nx]
-xv[nlocal+0+ix-1+(iz)*nx]
-xv[nlocal+0+ix+iz*nx]
-xv[nlocal+0+ix+1+iz*nx]
-xv[nlocal+0+ix-1+(iz+1)*nx]
-xv[nlocal+0+ix+(iz+1)*nx]
-xv[nlocal+0+ix+1+(iz+1)*nx]
;}
}
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+ix-1+(iy-1)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+ix+(iy-1)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+ix+1+(iy-1)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+ix-1+(iy)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+ix+iy*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+ix+1+iy*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+ix-1+(iy+1)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+ix+(iy+1)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+ix+1+(iy+1)*nx]
;}
}
}//ipz < npz - 1
            else
{
iy=0;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
-xv[nlocal+0+nx+ix-1]
-xv[nlocal+0+nx+ix]
-xv[nlocal+0+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix+(nz-2)*nx-1]
-xv[nlocal+0+ix+(nz-2)*nx]
-xv[nlocal+0+ix+(nz-2)*nx+1]
-xv[nlocal+0+ix+(nz-1)*nx-1]
-xv[nlocal+0+ix+(nz-1)*nx]
-xv[nlocal+0+ix+(nz-1)*nx+1]
;}
;
ix=nx-1;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+iy-1]
-xv[nlocal+0+nx*nz+nz+iy]
-xv[nlocal+0+nx*nz+nz+iy+1]
-xv[nlocal+0+nx*nz+nz+ny+iy-1]
-xv[nlocal+0+nx*nz+nz+ny+iy]
-xv[nlocal+0+nx*nz+nz+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+nx*nz+nz+iy+(nz-2)*ny]
-xv[nlocal+0+nx*nz+nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+nx*nz+nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+nx*nz+nz+iy+(nz-1)*ny]
-xv[nlocal+0+nx*nz+nz+iy+(nz-1)*ny+1]
;}
;
iy=0;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*nx]
-xv[nlocal+0+(iz-1)*nx+1]
-xv[nlocal+0+(iz)*nx]
-xv[nlocal+0+(iz-1)*nx+1]
-xv[nlocal+0+(iz+1)*nx]
-xv[nlocal+0+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*nx+nx-2]
-xv[nlocal+0+(iz-1)*nx+nx-1]
-xv[nlocal+0+(iz)*nx+nx-2]
-xv[nlocal+0+(iz)*nx+nx-1]
-xv[nlocal+0+(iz+1)*nx+nx-2]
-xv[nlocal+0+(iz+1)*nx+nx-1]
;}
ix=nx-1;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+iz-1]
-xv[nlocal+0+nx*nz+iz]
-xv[nlocal+0+nx*nz+iz+1]
;}
ix=nx-1;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+(iz-1)*ny]
-xv[nlocal+0+nx*nz+nz+(iz-1)*ny+1]
-xv[nlocal+0+nx*nz+nz+(iz)*ny]
-xv[nlocal+0+nx*nz+nz+(iz-1)*ny+1]
-xv[nlocal+0+nx*nz+nz+(iz+1)*ny]
-xv[nlocal+0+nx*nz+nz+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+nx*nz+nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+nx*nz+nz+(iz)*ny+ny-2]
-xv[nlocal+0+nx*nz+nz+(iz)*ny+ny-1]
-xv[nlocal+0+nx*nz+nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+nx*nz+nz+(iz+1)*ny+ny-1]
;}
;
ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+iy-1+(iz-1)*ny]
-xv[nlocal+0+nx*nz+nz+iy+(iz-1)*ny]
-xv[nlocal+0+nx*nz+nz+iy+1+(iz-1)*ny]
-xv[nlocal+0+nx*nz+nz+iy-1+(iz)*ny]
-xv[nlocal+0+nx*nz+nz+iy+iz*ny]
-xv[nlocal+0+nx*nz+nz+iy+1+iz*ny]
-xv[nlocal+0+nx*nz+nz+iy-1+(iz+1)*ny]
-xv[nlocal+0+nx*nz+nz+iy+(iz+1)*ny]
-xv[nlocal+0+nx*nz+nz+iy+1+(iz+1)*ny]
;}
}
iy=0;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1+(iz-1)*nx]
-xv[nlocal+0+ix+(iz-1)*nx]
-xv[nlocal+0+ix+1+(iz-1)*nx]
-xv[nlocal+0+ix-1+(iz)*nx]
-xv[nlocal+0+ix+iz*nx]
-xv[nlocal+0+ix+1+iz*nx]
-xv[nlocal+0+ix-1+(iz+1)*nx]
-xv[nlocal+0+ix+(iz+1)*nx]
-xv[nlocal+0+ix+1+(iz+1)*nx]
;}
}
}//ipz < npz - 1
          }//ipz > 0 
        }//ipy < npy - 1 
      }//ipy > 0 
    else
{
      if(ipy < npy - 1)
{
        if(ipz > 0)
{
            if(ipz < npz - 1)
{
ix = 0;
iy = ny-1;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*ny+ny]
-xv[nlocal+0+nx*ny+ny+1]
;
ix = 0;
iy = ny-1;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+nx*ny+ny]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+nx*ny+ny+1]
;
ix = nx-1;
iy = ny-1;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*ny+ny+nx-2]
-xv[nlocal+0+nx*ny+ny+nx-1]
-xv[nlocal+0+nx*ny+ny+nx]
;
ix = nx-1;
iy = ny-1;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+nx*ny+ny+nx-2]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+nx*ny+ny+nx-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+nx*ny+ny+nx]
;
iz=0;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
-xv[nlocal+0+nx+ix-1]
-xv[nlocal+0+nx+ix]
-xv[nlocal+0+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix+(ny-2)*nx-1]
-xv[nlocal+0+ix+(ny-2)*nx]
-xv[nlocal+0+ix+(ny-2)*nx+1]
-xv[nlocal+0+ix+(ny-1)*nx-1]
-xv[nlocal+0+ix+(ny-1)*nx]
-xv[nlocal+0+ix+(ny-1)*nx+1]
;}
iz=nz-1;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+ix-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+ix]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+ix+1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+nx+ix-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+nx+ix]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+ix+(ny-2)*nx-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+ix+(ny-2)*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+ix+(ny-2)*nx+1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+ix+(ny-1)*nx-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+ix+(ny-1)*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+ix+(ny-1)*nx+1]
;}
iy=ny-1;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+ix-1]
-xv[nlocal+0+nx*ny+ny+ix]
-xv[nlocal+0+nx*ny+ny+ix+1]
;}
iy=ny-1;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix+1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx+ix-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx+ix]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix+(nz-2)*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix+(nz-1)*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix+(nz-1)*nx+1]
;}
iy=ny-1;
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+nx*ny+ny+ix-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+nx*ny+ny+ix]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+nx*ny+ny+ix+1]
;}
;
iz=0;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iy-1)*nx]
-xv[nlocal+0+(iy-1)*nx+1]
-xv[nlocal+0+(iy)*nx]
-xv[nlocal+0+(iy-1)*nx+1]
-xv[nlocal+0+(iy+1)*nx]
-xv[nlocal+0+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iy-1)*nx+nx-2]
-xv[nlocal+0+(iy-1)*nx+nx-1]
-xv[nlocal+0+(iy)*nx+nx-2]
-xv[nlocal+0+(iy)*nx+nx-1]
-xv[nlocal+0+(iy+1)*nx+nx-2]
-xv[nlocal+0+(iy+1)*nx+nx-1]
;}
iz=nz-1;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+(iy-1)*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+(iy-1)*nx+1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+(iy)*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+(iy-1)*nx+1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+(iy+1)*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+(iy-1)*nx+nx-2]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+(iy-1)*nx+nx-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+(iy)*nx+nx-2]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+(iy)*nx+nx-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+(iy+1)*nx+nx-2]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+(iy+1)*nx+nx-1]
;}
ix=nx-1;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+iy-1]
-xv[nlocal+0+nx*ny+iy]
-xv[nlocal+0+nx*ny+iy+1]
;}
ix=nx-1;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+iy-1]
-xv[nlocal+0+nx*ny+ny+nx+1+iy]
-xv[nlocal+0+nx*ny+ny+nx+1+iy+1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny+iy-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny+iy]
-xv[nlocal+0+nx*ny+ny+nx+1+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+iy+(nz-2)*ny-1]
-xv[nlocal+0+nx*ny+ny+nx+1+iy+(nz-2)*ny]
-xv[nlocal+0+nx*ny+ny+nx+1+iy+(nz-2)*ny+1]
-xv[nlocal+0+nx*ny+ny+nx+1+iy+(nz-1)*ny-1]
-xv[nlocal+0+nx*ny+ny+nx+1+iy+(nz-1)*ny]
-xv[nlocal+0+nx*ny+ny+nx+1+iy+(nz-1)*ny+1]
;}
ix=nx-1;
iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+nx*ny+iy-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+nx*ny+iy]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+nx*ny+iy+1]
;}
;
iy=ny-1;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+(iz-1)*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+(iz-1)*nx+1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+(iz)*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+(iz-1)*nx+1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+(iz+1)*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+(iz)*nx+nx-2]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+(iz)*nx+nx-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+(iz+1)*nx+nx-1]
;}
ix=nx-1;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+(iz-1)*ny]
-xv[nlocal+0+nx*ny+ny+nx+1+(iz-1)*ny+1]
-xv[nlocal+0+nx*ny+ny+nx+1+(iz)*ny]
-xv[nlocal+0+nx*ny+ny+nx+1+(iz-1)*ny+1]
-xv[nlocal+0+nx*ny+ny+nx+1+(iz+1)*ny]
-xv[nlocal+0+nx*ny+ny+nx+1+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+(iz-1)*ny+ny-2]
-xv[nlocal+0+nx*ny+ny+nx+1+(iz-1)*ny+ny-1]
-xv[nlocal+0+nx*ny+ny+nx+1+(iz)*ny+ny-2]
-xv[nlocal+0+nx*ny+ny+nx+1+(iz)*ny+ny-1]
-xv[nlocal+0+nx*ny+ny+nx+1+(iz+1)*ny+ny-2]
-xv[nlocal+0+nx*ny+ny+nx+1+(iz+1)*ny+ny-1]
;}
ix=nx-1;
iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+iz-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+iz]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+iz+1]
;}
;
ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+iy-1+(iz-1)*ny]
-xv[nlocal+0+nx*ny+ny+nx+1+iy+(iz-1)*ny]
-xv[nlocal+0+nx*ny+ny+nx+1+iy+1+(iz-1)*ny]
-xv[nlocal+0+nx*ny+ny+nx+1+iy-1+(iz)*ny]
-xv[nlocal+0+nx*ny+ny+nx+1+iy+iz*ny]
-xv[nlocal+0+nx*ny+ny+nx+1+iy+1+iz*ny]
-xv[nlocal+0+nx*ny+ny+nx+1+iy-1+(iz+1)*ny]
-xv[nlocal+0+nx*ny+ny+nx+1+iy+(iz+1)*ny]
-xv[nlocal+0+nx*ny+ny+nx+1+iy+1+(iz+1)*ny]
;}
}
iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix-1+(iz-1)*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix+(iz-1)*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix+1+(iz-1)*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix-1+(iz)*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix+iz*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix+1+iz*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix-1+(iz+1)*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix+(iz+1)*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix+1+(iz+1)*nx]
;}
}
iz=0;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1+(iy-1)*nx]
-xv[nlocal+0+ix+(iy-1)*nx]
-xv[nlocal+0+ix+1+(iy-1)*nx]
-xv[nlocal+0+ix-1+(iy)*nx]
-xv[nlocal+0+ix+iy*nx]
-xv[nlocal+0+ix+1+iy*nx]
-xv[nlocal+0+ix-1+(iy+1)*nx]
-xv[nlocal+0+ix+(iy+1)*nx]
-xv[nlocal+0+ix+1+(iy+1)*nx]
;}
}
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+ix-1+(iy-1)*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+ix+(iy-1)*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+ix+1+(iy-1)*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+ix-1+(iy)*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+ix+iy*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+ix+1+iy*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+ix-1+(iy+1)*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+ix+(iy+1)*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+ix+1+(iy+1)*nx]
;}
}
}//ipz < npz - 1
            else
{
ix = 0;
iy = ny-1;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*ny+ny]
-xv[nlocal+0+nx*ny+ny+1]
;
ix = nx-1;
iy = ny-1;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*ny+ny+nx-2]
-xv[nlocal+0+nx*ny+ny+nx-1]
-xv[nlocal+0+nx*ny+ny+nx]
;
iz=0;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
-xv[nlocal+0+nx+ix-1]
-xv[nlocal+0+nx+ix]
-xv[nlocal+0+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix+(ny-2)*nx-1]
-xv[nlocal+0+ix+(ny-2)*nx]
-xv[nlocal+0+ix+(ny-2)*nx+1]
-xv[nlocal+0+ix+(ny-1)*nx-1]
-xv[nlocal+0+ix+(ny-1)*nx]
-xv[nlocal+0+ix+(ny-1)*nx+1]
;}
iy=ny-1;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+ix-1]
-xv[nlocal+0+nx*ny+ny+ix]
-xv[nlocal+0+nx*ny+ny+ix+1]
;}
iy=ny-1;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix+1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx+ix-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx+ix]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix+(nz-2)*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix+(nz-1)*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix+(nz-1)*nx+1]
;}
;
iz=0;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iy-1)*nx]
-xv[nlocal+0+(iy-1)*nx+1]
-xv[nlocal+0+(iy)*nx]
-xv[nlocal+0+(iy-1)*nx+1]
-xv[nlocal+0+(iy+1)*nx]
-xv[nlocal+0+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iy-1)*nx+nx-2]
-xv[nlocal+0+(iy-1)*nx+nx-1]
-xv[nlocal+0+(iy)*nx+nx-2]
-xv[nlocal+0+(iy)*nx+nx-1]
-xv[nlocal+0+(iy+1)*nx+nx-2]
-xv[nlocal+0+(iy+1)*nx+nx-1]
;}
ix=nx-1;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+iy-1]
-xv[nlocal+0+nx*ny+iy]
-xv[nlocal+0+nx*ny+iy+1]
;}
ix=nx-1;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+iy-1]
-xv[nlocal+0+nx*ny+ny+nx+1+iy]
-xv[nlocal+0+nx*ny+ny+nx+1+iy+1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny+iy-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny+iy]
-xv[nlocal+0+nx*ny+ny+nx+1+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+iy+(nz-2)*ny-1]
-xv[nlocal+0+nx*ny+ny+nx+1+iy+(nz-2)*ny]
-xv[nlocal+0+nx*ny+ny+nx+1+iy+(nz-2)*ny+1]
-xv[nlocal+0+nx*ny+ny+nx+1+iy+(nz-1)*ny-1]
-xv[nlocal+0+nx*ny+ny+nx+1+iy+(nz-1)*ny]
-xv[nlocal+0+nx*ny+ny+nx+1+iy+(nz-1)*ny+1]
;}
;
iy=ny-1;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+(iz-1)*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+(iz-1)*nx+1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+(iz)*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+(iz-1)*nx+1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+(iz+1)*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+(iz)*nx+nx-2]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+(iz)*nx+nx-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+(iz+1)*nx+nx-1]
;}
ix=nx-1;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+(iz-1)*ny]
-xv[nlocal+0+nx*ny+ny+nx+1+(iz-1)*ny+1]
-xv[nlocal+0+nx*ny+ny+nx+1+(iz)*ny]
-xv[nlocal+0+nx*ny+ny+nx+1+(iz-1)*ny+1]
-xv[nlocal+0+nx*ny+ny+nx+1+(iz+1)*ny]
-xv[nlocal+0+nx*ny+ny+nx+1+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+(iz-1)*ny+ny-2]
-xv[nlocal+0+nx*ny+ny+nx+1+(iz-1)*ny+ny-1]
-xv[nlocal+0+nx*ny+ny+nx+1+(iz)*ny+ny-2]
-xv[nlocal+0+nx*ny+ny+nx+1+(iz)*ny+ny-1]
-xv[nlocal+0+nx*ny+ny+nx+1+(iz+1)*ny+ny-2]
-xv[nlocal+0+nx*ny+ny+nx+1+(iz+1)*ny+ny-1]
;}
ix=nx-1;
iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+iz-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+iz]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+iz+1]
;}
;
ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+iy-1+(iz-1)*ny]
-xv[nlocal+0+nx*ny+ny+nx+1+iy+(iz-1)*ny]
-xv[nlocal+0+nx*ny+ny+nx+1+iy+1+(iz-1)*ny]
-xv[nlocal+0+nx*ny+ny+nx+1+iy-1+(iz)*ny]
-xv[nlocal+0+nx*ny+ny+nx+1+iy+iz*ny]
-xv[nlocal+0+nx*ny+ny+nx+1+iy+1+iz*ny]
-xv[nlocal+0+nx*ny+ny+nx+1+iy-1+(iz+1)*ny]
-xv[nlocal+0+nx*ny+ny+nx+1+iy+(iz+1)*ny]
-xv[nlocal+0+nx*ny+ny+nx+1+iy+1+(iz+1)*ny]
;}
}
iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix-1+(iz-1)*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix+(iz-1)*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix+1+(iz-1)*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix-1+(iz)*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix+iz*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix+1+iz*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix-1+(iz+1)*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix+(iz+1)*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix+1+(iz+1)*nx]
;}
}
iz=0;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1+(iy-1)*nx]
-xv[nlocal+0+ix+(iy-1)*nx]
-xv[nlocal+0+ix+1+(iy-1)*nx]
-xv[nlocal+0+ix-1+(iy)*nx]
-xv[nlocal+0+ix+iy*nx]
-xv[nlocal+0+ix+1+iy*nx]
-xv[nlocal+0+ix-1+(iy+1)*nx]
-xv[nlocal+0+ix+(iy+1)*nx]
-xv[nlocal+0+ix+1+(iy+1)*nx]
;}
}
}//ipz < npz - 1
          }//ipz > 0 
        else
{
            if(ipz < npz - 1)
{
ix = 0;
iy = ny-1;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*nz+nx*nz+nz+nx*ny+ny]
-xv[nlocal+0+ny*nz+nx*nz+nz+nx*ny+ny+1]
;
ix = nx-1;
iy = ny-1;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*nz+nx*nz+nz+nx*ny+ny+nx-2]
-xv[nlocal+0+ny*nz+nx*nz+nz+nx*ny+ny+nx-1]
-xv[nlocal+0+ny*nz+nx*nz+nz+nx*ny+ny+nx]
;
iz=nz-1;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+nx*nz+nz+ix-1]
-xv[nlocal+0+ny*nz+nx*nz+nz+ix]
-xv[nlocal+0+ny*nz+nx*nz+nz+ix+1]
-xv[nlocal+0+ny*nz+nx*nz+nz+nx+ix-1]
-xv[nlocal+0+ny*nz+nx*nz+nz+nx+ix]
-xv[nlocal+0+ny*nz+nx*nz+nz+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+nx*nz+nz+ix+(ny-2)*nx-1]
-xv[nlocal+0+ny*nz+nx*nz+nz+ix+(ny-2)*nx]
-xv[nlocal+0+ny*nz+nx*nz+nz+ix+(ny-2)*nx+1]
-xv[nlocal+0+ny*nz+nx*nz+nz+ix+(ny-1)*nx-1]
-xv[nlocal+0+ny*nz+nx*nz+nz+ix+(ny-1)*nx]
-xv[nlocal+0+ny*nz+nx*nz+nz+ix+(ny-1)*nx+1]
;}
iy=ny-1;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ix-1]
-xv[nlocal+0+ny*nz+ix]
-xv[nlocal+0+ny*nz+ix+1]
-xv[nlocal+0+ny*nz+nx+ix-1]
-xv[nlocal+0+ny*nz+nx+ix]
-xv[nlocal+0+ny*nz+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+ny*nz+ix+(nz-2)*nx]
-xv[nlocal+0+ny*nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+ny*nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+ny*nz+ix+(nz-1)*nx]
-xv[nlocal+0+ny*nz+ix+(nz-1)*nx+1]
;}
iy=ny-1;
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+nx*nz+nz+nx*ny+ny+ix-1]
-xv[nlocal+0+ny*nz+nx*nz+nz+nx*ny+ny+ix]
-xv[nlocal+0+ny*nz+nx*nz+nz+nx*ny+ny+ix+1]
;}
;
iz=nz-1;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+nx*nz+nz+(iy-1)*nx]
-xv[nlocal+0+ny*nz+nx*nz+nz+(iy-1)*nx+1]
-xv[nlocal+0+ny*nz+nx*nz+nz+(iy)*nx]
-xv[nlocal+0+ny*nz+nx*nz+nz+(iy-1)*nx+1]
-xv[nlocal+0+ny*nz+nx*nz+nz+(iy+1)*nx]
-xv[nlocal+0+ny*nz+nx*nz+nz+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+nx*nz+nz+(iy-1)*nx+nx-2]
-xv[nlocal+0+ny*nz+nx*nz+nz+(iy-1)*nx+nx-1]
-xv[nlocal+0+ny*nz+nx*nz+nz+(iy)*nx+nx-2]
-xv[nlocal+0+ny*nz+nx*nz+nz+(iy)*nx+nx-1]
-xv[nlocal+0+ny*nz+nx*nz+nz+(iy+1)*nx+nx-2]
-xv[nlocal+0+ny*nz+nx*nz+nz+(iy+1)*nx+nx-1]
;}
ix=nx-1;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1]
-xv[nlocal+0+iy]
-xv[nlocal+0+iy+1]
-xv[nlocal+0+ny+iy-1]
-xv[nlocal+0+ny+iy]
-xv[nlocal+0+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy+(nz-2)*ny-1]
-xv[nlocal+0+iy+(nz-2)*ny]
-xv[nlocal+0+iy+(nz-2)*ny+1]
-xv[nlocal+0+iy+(nz-1)*ny-1]
-xv[nlocal+0+iy+(nz-1)*ny]
-xv[nlocal+0+iy+(nz-1)*ny+1]
;}
ix=nx-1;
iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+nx*nz+nz+nx*ny+iy-1]
-xv[nlocal+0+ny*nz+nx*nz+nz+nx*ny+iy]
-xv[nlocal+0+ny*nz+nx*nz+nz+nx*ny+iy+1]
;}
;
iy=ny-1;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+(iz-1)*nx]
-xv[nlocal+0+ny*nz+(iz-1)*nx+1]
-xv[nlocal+0+ny*nz+(iz)*nx]
-xv[nlocal+0+ny*nz+(iz-1)*nx+1]
-xv[nlocal+0+ny*nz+(iz+1)*nx]
-xv[nlocal+0+ny*nz+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+ny*nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+ny*nz+(iz)*nx+nx-2]
-xv[nlocal+0+ny*nz+(iz)*nx+nx-1]
-xv[nlocal+0+ny*nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+ny*nz+(iz+1)*nx+nx-1]
;}
ix=nx-1;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*ny]
-xv[nlocal+0+(iz-1)*ny+1]
-xv[nlocal+0+(iz)*ny]
-xv[nlocal+0+(iz-1)*ny+1]
-xv[nlocal+0+(iz+1)*ny]
-xv[nlocal+0+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*ny+ny-2]
-xv[nlocal+0+(iz-1)*ny+ny-1]
-xv[nlocal+0+(iz)*ny+ny-2]
-xv[nlocal+0+(iz)*ny+ny-1]
-xv[nlocal+0+(iz+1)*ny+ny-2]
-xv[nlocal+0+(iz+1)*ny+ny-1]
;}
ix=nx-1;
iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+nx*nz+iz-1]
-xv[nlocal+0+ny*nz+nx*nz+iz]
-xv[nlocal+0+ny*nz+nx*nz+iz+1]
;}
;
ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1+(iz-1)*ny]
-xv[nlocal+0+iy+(iz-1)*ny]
-xv[nlocal+0+iy+1+(iz-1)*ny]
-xv[nlocal+0+iy-1+(iz)*ny]
-xv[nlocal+0+iy+iz*ny]
-xv[nlocal+0+iy+1+iz*ny]
-xv[nlocal+0+iy-1+(iz+1)*ny]
-xv[nlocal+0+iy+(iz+1)*ny]
-xv[nlocal+0+iy+1+(iz+1)*ny]
;}
}
iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ix-1+(iz-1)*nx]
-xv[nlocal+0+ny*nz+ix+(iz-1)*nx]
-xv[nlocal+0+ny*nz+ix+1+(iz-1)*nx]
-xv[nlocal+0+ny*nz+ix-1+(iz)*nx]
-xv[nlocal+0+ny*nz+ix+iz*nx]
-xv[nlocal+0+ny*nz+ix+1+iz*nx]
-xv[nlocal+0+ny*nz+ix-1+(iz+1)*nx]
-xv[nlocal+0+ny*nz+ix+(iz+1)*nx]
-xv[nlocal+0+ny*nz+ix+1+(iz+1)*nx]
;}
}
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+nx*nz+nz+ix-1+(iy-1)*nx]
-xv[nlocal+0+ny*nz+nx*nz+nz+ix+(iy-1)*nx]
-xv[nlocal+0+ny*nz+nx*nz+nz+ix+1+(iy-1)*nx]
-xv[nlocal+0+ny*nz+nx*nz+nz+ix-1+(iy)*nx]
-xv[nlocal+0+ny*nz+nx*nz+nz+ix+iy*nx]
-xv[nlocal+0+ny*nz+nx*nz+nz+ix+1+iy*nx]
-xv[nlocal+0+ny*nz+nx*nz+nz+ix-1+(iy+1)*nx]
-xv[nlocal+0+ny*nz+nx*nz+nz+ix+(iy+1)*nx]
-xv[nlocal+0+ny*nz+nx*nz+nz+ix+1+(iy+1)*nx]
;}
}
}//ipz < npz - 1
            else
{
iy=ny-1;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ix-1]
-xv[nlocal+0+ny*nz+ix]
-xv[nlocal+0+ny*nz+ix+1]
-xv[nlocal+0+ny*nz+nx+ix-1]
-xv[nlocal+0+ny*nz+nx+ix]
-xv[nlocal+0+ny*nz+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+ny*nz+ix+(nz-2)*nx]
-xv[nlocal+0+ny*nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+ny*nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+ny*nz+ix+(nz-1)*nx]
-xv[nlocal+0+ny*nz+ix+(nz-1)*nx+1]
;}
;
ix=nx-1;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1]
-xv[nlocal+0+iy]
-xv[nlocal+0+iy+1]
-xv[nlocal+0+ny+iy-1]
-xv[nlocal+0+ny+iy]
-xv[nlocal+0+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy+(nz-2)*ny-1]
-xv[nlocal+0+iy+(nz-2)*ny]
-xv[nlocal+0+iy+(nz-2)*ny+1]
-xv[nlocal+0+iy+(nz-1)*ny-1]
-xv[nlocal+0+iy+(nz-1)*ny]
-xv[nlocal+0+iy+(nz-1)*ny+1]
;}
;
iy=ny-1;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+(iz-1)*nx]
-xv[nlocal+0+ny*nz+(iz-1)*nx+1]
-xv[nlocal+0+ny*nz+(iz)*nx]
-xv[nlocal+0+ny*nz+(iz-1)*nx+1]
-xv[nlocal+0+ny*nz+(iz+1)*nx]
-xv[nlocal+0+ny*nz+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+ny*nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+ny*nz+(iz)*nx+nx-2]
-xv[nlocal+0+ny*nz+(iz)*nx+nx-1]
-xv[nlocal+0+ny*nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+ny*nz+(iz+1)*nx+nx-1]
;}
ix=nx-1;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*ny]
-xv[nlocal+0+(iz-1)*ny+1]
-xv[nlocal+0+(iz)*ny]
-xv[nlocal+0+(iz-1)*ny+1]
-xv[nlocal+0+(iz+1)*ny]
-xv[nlocal+0+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*ny+ny-2]
-xv[nlocal+0+(iz-1)*ny+ny-1]
-xv[nlocal+0+(iz)*ny+ny-2]
-xv[nlocal+0+(iz)*ny+ny-1]
-xv[nlocal+0+(iz+1)*ny+ny-2]
-xv[nlocal+0+(iz+1)*ny+ny-1]
;}
ix=nx-1;
iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+nx*nz+iz-1]
-xv[nlocal+0+ny*nz+nx*nz+iz]
-xv[nlocal+0+ny*nz+nx*nz+iz+1]
;}
;
ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1+(iz-1)*ny]
-xv[nlocal+0+iy+(iz-1)*ny]
-xv[nlocal+0+iy+1+(iz-1)*ny]
-xv[nlocal+0+iy-1+(iz)*ny]
-xv[nlocal+0+iy+iz*ny]
-xv[nlocal+0+iy+1+iz*ny]
-xv[nlocal+0+iy-1+(iz+1)*ny]
-xv[nlocal+0+iy+(iz+1)*ny]
-xv[nlocal+0+iy+1+(iz+1)*ny]
;}
}
iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ix-1+(iz-1)*nx]
-xv[nlocal+0+ny*nz+ix+(iz-1)*nx]
-xv[nlocal+0+ny*nz+ix+1+(iz-1)*nx]
-xv[nlocal+0+ny*nz+ix-1+(iz)*nx]
-xv[nlocal+0+ny*nz+ix+iz*nx]
-xv[nlocal+0+ny*nz+ix+1+iz*nx]
-xv[nlocal+0+ny*nz+ix-1+(iz+1)*nx]
-xv[nlocal+0+ny*nz+ix+(iz+1)*nx]
-xv[nlocal+0+ny*nz+ix+1+(iz+1)*nx]
;}
}
}//ipz < npz - 1
          }//ipz > 0 
        }//ipy < npy - 1 
      else
{
        if(ipz > 0)
{
            if(ipz < npz - 1)
{
iz=0;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
-xv[nlocal+0+nx+ix-1]
-xv[nlocal+0+nx+ix]
-xv[nlocal+0+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix+(ny-2)*nx-1]
-xv[nlocal+0+ix+(ny-2)*nx]
-xv[nlocal+0+ix+(ny-2)*nx+1]
-xv[nlocal+0+ix+(ny-1)*nx-1]
-xv[nlocal+0+ix+(ny-1)*nx]
-xv[nlocal+0+ix+(ny-1)*nx+1]
;}
iz=nz-1;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+ny*nz+ix-1]
-xv[nlocal+0+nx*ny+ny+ny*nz+ix]
-xv[nlocal+0+nx*ny+ny+ny*nz+ix+1]
-xv[nlocal+0+nx*ny+ny+ny*nz+nx+ix-1]
-xv[nlocal+0+nx*ny+ny+ny*nz+nx+ix]
-xv[nlocal+0+nx*ny+ny+ny*nz+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+ny*nz+ix+(ny-2)*nx-1]
-xv[nlocal+0+nx*ny+ny+ny*nz+ix+(ny-2)*nx]
-xv[nlocal+0+nx*ny+ny+ny*nz+ix+(ny-2)*nx+1]
-xv[nlocal+0+nx*ny+ny+ny*nz+ix+(ny-1)*nx-1]
-xv[nlocal+0+nx*ny+ny+ny*nz+ix+(ny-1)*nx]
-xv[nlocal+0+nx*ny+ny+ny*nz+ix+(ny-1)*nx+1]
;}
;
iz=0;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iy-1)*nx]
-xv[nlocal+0+(iy-1)*nx+1]
-xv[nlocal+0+(iy)*nx]
-xv[nlocal+0+(iy-1)*nx+1]
-xv[nlocal+0+(iy+1)*nx]
-xv[nlocal+0+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iy-1)*nx+nx-2]
-xv[nlocal+0+(iy-1)*nx+nx-1]
-xv[nlocal+0+(iy)*nx+nx-2]
-xv[nlocal+0+(iy)*nx+nx-1]
-xv[nlocal+0+(iy+1)*nx+nx-2]
-xv[nlocal+0+(iy+1)*nx+nx-1]
;}
iz=nz-1;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+ny*nz+(iy-1)*nx]
-xv[nlocal+0+nx*ny+ny+ny*nz+(iy-1)*nx+1]
-xv[nlocal+0+nx*ny+ny+ny*nz+(iy)*nx]
-xv[nlocal+0+nx*ny+ny+ny*nz+(iy-1)*nx+1]
-xv[nlocal+0+nx*ny+ny+ny*nz+(iy+1)*nx]
-xv[nlocal+0+nx*ny+ny+ny*nz+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+ny*nz+(iy-1)*nx+nx-2]
-xv[nlocal+0+nx*ny+ny+ny*nz+(iy-1)*nx+nx-1]
-xv[nlocal+0+nx*ny+ny+ny*nz+(iy)*nx+nx-2]
-xv[nlocal+0+nx*ny+ny+ny*nz+(iy)*nx+nx-1]
-xv[nlocal+0+nx*ny+ny+ny*nz+(iy+1)*nx+nx-2]
-xv[nlocal+0+nx*ny+ny+ny*nz+(iy+1)*nx+nx-1]
;}
ix=nx-1;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+iy-1]
-xv[nlocal+0+nx*ny+iy]
-xv[nlocal+0+nx*ny+iy+1]
;}
ix=nx-1;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+iy-1]
-xv[nlocal+0+nx*ny+ny+iy]
-xv[nlocal+0+nx*ny+ny+iy+1]
-xv[nlocal+0+nx*ny+ny+ny+iy-1]
-xv[nlocal+0+nx*ny+ny+ny+iy]
-xv[nlocal+0+nx*ny+ny+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+iy+(nz-2)*ny-1]
-xv[nlocal+0+nx*ny+ny+iy+(nz-2)*ny]
-xv[nlocal+0+nx*ny+ny+iy+(nz-2)*ny+1]
-xv[nlocal+0+nx*ny+ny+iy+(nz-1)*ny-1]
-xv[nlocal+0+nx*ny+ny+iy+(nz-1)*ny]
-xv[nlocal+0+nx*ny+ny+iy+(nz-1)*ny+1]
;}
ix=nx-1;
iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+ny*nz+nx*ny+iy-1]
-xv[nlocal+0+nx*ny+ny+ny*nz+nx*ny+iy]
-xv[nlocal+0+nx*ny+ny+ny*nz+nx*ny+iy+1]
;}
;
ix=nx-1;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+(iz-1)*ny]
-xv[nlocal+0+nx*ny+ny+(iz-1)*ny+1]
-xv[nlocal+0+nx*ny+ny+(iz)*ny]
-xv[nlocal+0+nx*ny+ny+(iz-1)*ny+1]
-xv[nlocal+0+nx*ny+ny+(iz+1)*ny]
-xv[nlocal+0+nx*ny+ny+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+(iz-1)*ny+ny-2]
-xv[nlocal+0+nx*ny+ny+(iz-1)*ny+ny-1]
-xv[nlocal+0+nx*ny+ny+(iz)*ny+ny-2]
-xv[nlocal+0+nx*ny+ny+(iz)*ny+ny-1]
-xv[nlocal+0+nx*ny+ny+(iz+1)*ny+ny-2]
-xv[nlocal+0+nx*ny+ny+(iz+1)*ny+ny-1]
;}
;
ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+iy-1+(iz-1)*ny]
-xv[nlocal+0+nx*ny+ny+iy+(iz-1)*ny]
-xv[nlocal+0+nx*ny+ny+iy+1+(iz-1)*ny]
-xv[nlocal+0+nx*ny+ny+iy-1+(iz)*ny]
-xv[nlocal+0+nx*ny+ny+iy+iz*ny]
-xv[nlocal+0+nx*ny+ny+iy+1+iz*ny]
-xv[nlocal+0+nx*ny+ny+iy-1+(iz+1)*ny]
-xv[nlocal+0+nx*ny+ny+iy+(iz+1)*ny]
-xv[nlocal+0+nx*ny+ny+iy+1+(iz+1)*ny]
;}
}
iz=0;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1+(iy-1)*nx]
-xv[nlocal+0+ix+(iy-1)*nx]
-xv[nlocal+0+ix+1+(iy-1)*nx]
-xv[nlocal+0+ix-1+(iy)*nx]
-xv[nlocal+0+ix+iy*nx]
-xv[nlocal+0+ix+1+iy*nx]
-xv[nlocal+0+ix-1+(iy+1)*nx]
-xv[nlocal+0+ix+(iy+1)*nx]
-xv[nlocal+0+ix+1+(iy+1)*nx]
;}
}
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+ny*nz+ix-1+(iy-1)*nx]
-xv[nlocal+0+nx*ny+ny+ny*nz+ix+(iy-1)*nx]
-xv[nlocal+0+nx*ny+ny+ny*nz+ix+1+(iy-1)*nx]
-xv[nlocal+0+nx*ny+ny+ny*nz+ix-1+(iy)*nx]
-xv[nlocal+0+nx*ny+ny+ny*nz+ix+iy*nx]
-xv[nlocal+0+nx*ny+ny+ny*nz+ix+1+iy*nx]
-xv[nlocal+0+nx*ny+ny+ny*nz+ix-1+(iy+1)*nx]
-xv[nlocal+0+nx*ny+ny+ny*nz+ix+(iy+1)*nx]
-xv[nlocal+0+nx*ny+ny+ny*nz+ix+1+(iy+1)*nx]
;}
}
}//ipz < npz - 1
            else
{
iz=0;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
-xv[nlocal+0+nx+ix-1]
-xv[nlocal+0+nx+ix]
-xv[nlocal+0+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix+(ny-2)*nx-1]
-xv[nlocal+0+ix+(ny-2)*nx]
-xv[nlocal+0+ix+(ny-2)*nx+1]
-xv[nlocal+0+ix+(ny-1)*nx-1]
-xv[nlocal+0+ix+(ny-1)*nx]
-xv[nlocal+0+ix+(ny-1)*nx+1]
;}
;
iz=0;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iy-1)*nx]
-xv[nlocal+0+(iy-1)*nx+1]
-xv[nlocal+0+(iy)*nx]
-xv[nlocal+0+(iy-1)*nx+1]
-xv[nlocal+0+(iy+1)*nx]
-xv[nlocal+0+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iy-1)*nx+nx-2]
-xv[nlocal+0+(iy-1)*nx+nx-1]
-xv[nlocal+0+(iy)*nx+nx-2]
-xv[nlocal+0+(iy)*nx+nx-1]
-xv[nlocal+0+(iy+1)*nx+nx-2]
-xv[nlocal+0+(iy+1)*nx+nx-1]
;}
ix=nx-1;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+iy-1]
-xv[nlocal+0+nx*ny+iy]
-xv[nlocal+0+nx*ny+iy+1]
;}
ix=nx-1;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+iy-1]
-xv[nlocal+0+nx*ny+ny+iy]
-xv[nlocal+0+nx*ny+ny+iy+1]
-xv[nlocal+0+nx*ny+ny+ny+iy-1]
-xv[nlocal+0+nx*ny+ny+ny+iy]
-xv[nlocal+0+nx*ny+ny+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+iy+(nz-2)*ny-1]
-xv[nlocal+0+nx*ny+ny+iy+(nz-2)*ny]
-xv[nlocal+0+nx*ny+ny+iy+(nz-2)*ny+1]
-xv[nlocal+0+nx*ny+ny+iy+(nz-1)*ny-1]
-xv[nlocal+0+nx*ny+ny+iy+(nz-1)*ny]
-xv[nlocal+0+nx*ny+ny+iy+(nz-1)*ny+1]
;}
;
ix=nx-1;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+(iz-1)*ny]
-xv[nlocal+0+nx*ny+ny+(iz-1)*ny+1]
-xv[nlocal+0+nx*ny+ny+(iz)*ny]
-xv[nlocal+0+nx*ny+ny+(iz-1)*ny+1]
-xv[nlocal+0+nx*ny+ny+(iz+1)*ny]
-xv[nlocal+0+nx*ny+ny+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+(iz-1)*ny+ny-2]
-xv[nlocal+0+nx*ny+ny+(iz-1)*ny+ny-1]
-xv[nlocal+0+nx*ny+ny+(iz)*ny+ny-2]
-xv[nlocal+0+nx*ny+ny+(iz)*ny+ny-1]
-xv[nlocal+0+nx*ny+ny+(iz+1)*ny+ny-2]
-xv[nlocal+0+nx*ny+ny+(iz+1)*ny+ny-1]
;}
;
ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+iy-1+(iz-1)*ny]
-xv[nlocal+0+nx*ny+ny+iy+(iz-1)*ny]
-xv[nlocal+0+nx*ny+ny+iy+1+(iz-1)*ny]
-xv[nlocal+0+nx*ny+ny+iy-1+(iz)*ny]
-xv[nlocal+0+nx*ny+ny+iy+iz*ny]
-xv[nlocal+0+nx*ny+ny+iy+1+iz*ny]
-xv[nlocal+0+nx*ny+ny+iy-1+(iz+1)*ny]
-xv[nlocal+0+nx*ny+ny+iy+(iz+1)*ny]
-xv[nlocal+0+nx*ny+ny+iy+1+(iz+1)*ny]
;}
}
iz=0;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1+(iy-1)*nx]
-xv[nlocal+0+ix+(iy-1)*nx]
-xv[nlocal+0+ix+1+(iy-1)*nx]
-xv[nlocal+0+ix-1+(iy)*nx]
-xv[nlocal+0+ix+iy*nx]
-xv[nlocal+0+ix+1+iy*nx]
-xv[nlocal+0+ix-1+(iy+1)*nx]
-xv[nlocal+0+ix+(iy+1)*nx]
-xv[nlocal+0+ix+1+(iy+1)*nx]
;}
}
}//ipz < npz - 1
          }//ipz > 0 
        else
{
            if(ipz < npz - 1)
{
iz=nz-1;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ix-1]
-xv[nlocal+0+ny*nz+ix]
-xv[nlocal+0+ny*nz+ix+1]
-xv[nlocal+0+ny*nz+nx+ix-1]
-xv[nlocal+0+ny*nz+nx+ix]
-xv[nlocal+0+ny*nz+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ix+(ny-2)*nx-1]
-xv[nlocal+0+ny*nz+ix+(ny-2)*nx]
-xv[nlocal+0+ny*nz+ix+(ny-2)*nx+1]
-xv[nlocal+0+ny*nz+ix+(ny-1)*nx-1]
-xv[nlocal+0+ny*nz+ix+(ny-1)*nx]
-xv[nlocal+0+ny*nz+ix+(ny-1)*nx+1]
;}
;
iz=nz-1;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+(iy-1)*nx]
-xv[nlocal+0+ny*nz+(iy-1)*nx+1]
-xv[nlocal+0+ny*nz+(iy)*nx]
-xv[nlocal+0+ny*nz+(iy-1)*nx+1]
-xv[nlocal+0+ny*nz+(iy+1)*nx]
-xv[nlocal+0+ny*nz+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+(iy-1)*nx+nx-2]
-xv[nlocal+0+ny*nz+(iy-1)*nx+nx-1]
-xv[nlocal+0+ny*nz+(iy)*nx+nx-2]
-xv[nlocal+0+ny*nz+(iy)*nx+nx-1]
-xv[nlocal+0+ny*nz+(iy+1)*nx+nx-2]
-xv[nlocal+0+ny*nz+(iy+1)*nx+nx-1]
;}
ix=nx-1;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1]
-xv[nlocal+0+iy]
-xv[nlocal+0+iy+1]
-xv[nlocal+0+ny+iy-1]
-xv[nlocal+0+ny+iy]
-xv[nlocal+0+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy+(nz-2)*ny-1]
-xv[nlocal+0+iy+(nz-2)*ny]
-xv[nlocal+0+iy+(nz-2)*ny+1]
-xv[nlocal+0+iy+(nz-1)*ny-1]
-xv[nlocal+0+iy+(nz-1)*ny]
-xv[nlocal+0+iy+(nz-1)*ny+1]
;}
ix=nx-1;
iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+nx*ny+iy-1]
-xv[nlocal+0+ny*nz+nx*ny+iy]
-xv[nlocal+0+ny*nz+nx*ny+iy+1]
;}
;
ix=nx-1;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*ny]
-xv[nlocal+0+(iz-1)*ny+1]
-xv[nlocal+0+(iz)*ny]
-xv[nlocal+0+(iz-1)*ny+1]
-xv[nlocal+0+(iz+1)*ny]
-xv[nlocal+0+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*ny+ny-2]
-xv[nlocal+0+(iz-1)*ny+ny-1]
-xv[nlocal+0+(iz)*ny+ny-2]
-xv[nlocal+0+(iz)*ny+ny-1]
-xv[nlocal+0+(iz+1)*ny+ny-2]
-xv[nlocal+0+(iz+1)*ny+ny-1]
;}
;
ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1+(iz-1)*ny]
-xv[nlocal+0+iy+(iz-1)*ny]
-xv[nlocal+0+iy+1+(iz-1)*ny]
-xv[nlocal+0+iy-1+(iz)*ny]
-xv[nlocal+0+iy+iz*ny]
-xv[nlocal+0+iy+1+iz*ny]
-xv[nlocal+0+iy-1+(iz+1)*ny]
-xv[nlocal+0+iy+(iz+1)*ny]
-xv[nlocal+0+iy+1+(iz+1)*ny]
;}
}
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ix-1+(iy-1)*nx]
-xv[nlocal+0+ny*nz+ix+(iy-1)*nx]
-xv[nlocal+0+ny*nz+ix+1+(iy-1)*nx]
-xv[nlocal+0+ny*nz+ix-1+(iy)*nx]
-xv[nlocal+0+ny*nz+ix+iy*nx]
-xv[nlocal+0+ny*nz+ix+1+iy*nx]
-xv[nlocal+0+ny*nz+ix-1+(iy+1)*nx]
-xv[nlocal+0+ny*nz+ix+(iy+1)*nx]
-xv[nlocal+0+ny*nz+ix+1+(iy+1)*nx]
;}
}
}//ipz < npz - 1
            else
{
;
ix=nx-1;
iz=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1]
-xv[nlocal+0+iy]
-xv[nlocal+0+iy+1]
-xv[nlocal+0+ny+iy-1]
-xv[nlocal+0+ny+iy]
-xv[nlocal+0+ny+iy+1]
;}iz=nz-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy+(nz-2)*ny-1]
-xv[nlocal+0+iy+(nz-2)*ny]
-xv[nlocal+0+iy+(nz-2)*ny+1]
-xv[nlocal+0+iy+(nz-1)*ny-1]
-xv[nlocal+0+iy+(nz-1)*ny]
-xv[nlocal+0+iy+(nz-1)*ny+1]
;}
;
ix=nx-1;
iy=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*ny]
-xv[nlocal+0+(iz-1)*ny+1]
-xv[nlocal+0+(iz)*ny]
-xv[nlocal+0+(iz-1)*ny+1]
-xv[nlocal+0+(iz+1)*ny]
-xv[nlocal+0+(iz+1)*ny+1]
;}iy=ny-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*ny+ny-2]
-xv[nlocal+0+(iz-1)*ny+ny-1]
-xv[nlocal+0+(iz)*ny+ny-2]
-xv[nlocal+0+(iz)*ny+ny-1]
-xv[nlocal+0+(iz+1)*ny+ny-2]
-xv[nlocal+0+(iz+1)*ny+ny-1]
;}
;
ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1+(iz-1)*ny]
-xv[nlocal+0+iy+(iz-1)*ny]
-xv[nlocal+0+iy+1+(iz-1)*ny]
-xv[nlocal+0+iy-1+(iz)*ny]
-xv[nlocal+0+iy+iz*ny]
-xv[nlocal+0+iy+1+iz*ny]
-xv[nlocal+0+iy-1+(iz+1)*ny]
-xv[nlocal+0+iy+(iz+1)*ny]
-xv[nlocal+0+iy+1+(iz+1)*ny]
;}
}
}//ipz < npz - 1
          }//ipz > 0 
        }//ipy < npy - 1 
      }//ipy > 0 
    }//ipx < npx - 1 
  else
{
    if(ipy > 0)
{
      if(ipy < npy - 1)
{
        if(ipz > 0)
{
            if(ipz < npz - 1)
{
ix = 0;
iy = 0;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
;
ix = 0;
iy = 0;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+1]
;
ix = 0;
iy = ny-1;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+nx*ny]
-xv[nlocal+0+nx+nx*ny+1]
;
ix = 0;
iy = ny-1;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+nx*ny]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+nx*ny+1]
;
ix = nx-1;
iy = 0;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx-2]
-xv[nlocal+0+nx-1]
;
ix = nx-1;
iy = 0;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx-2]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx-1]
;
ix = nx-1;
iy = ny-1;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+nx*ny+nx-2]
-xv[nlocal+0+nx+nx*ny+nx-1]
;
ix = nx-1;
iy = ny-1;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+nx*ny+nx-2]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+nx*ny+nx-1]
;
iy=0;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
;}
iy=0;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx+ix-1]
-xv[nlocal+0+nx+nx*ny+nx+ix]
-xv[nlocal+0+nx+nx*ny+nx+ix+1]
-xv[nlocal+0+nx+nx*ny+nx+nx+ix-1]
-xv[nlocal+0+nx+nx*ny+nx+nx+ix]
-xv[nlocal+0+nx+nx*ny+nx+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx+ix+(nz-2)*nx-1]
-xv[nlocal+0+nx+nx*ny+nx+ix+(nz-2)*nx]
-xv[nlocal+0+nx+nx*ny+nx+ix+(nz-2)*nx+1]
-xv[nlocal+0+nx+nx*ny+nx+ix+(nz-1)*nx-1]
-xv[nlocal+0+nx+nx*ny+nx+ix+(nz-1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+ix+(nz-1)*nx+1]
;}
iy=0;
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+ix-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+ix]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+ix+1]
;}
iz=0;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+ix-1]
-xv[nlocal+0+nx+ix]
-xv[nlocal+0+nx+ix+1]
-xv[nlocal+0+nx+nx+ix-1]
-xv[nlocal+0+nx+nx+ix]
-xv[nlocal+0+nx+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+ix+(ny-2)*nx-1]
-xv[nlocal+0+nx+ix+(ny-2)*nx]
-xv[nlocal+0+nx+ix+(ny-2)*nx+1]
-xv[nlocal+0+nx+ix+(ny-1)*nx-1]
-xv[nlocal+0+nx+ix+(ny-1)*nx]
-xv[nlocal+0+nx+ix+(ny-1)*nx+1]
;}
iz=nz-1;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+ix-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+ix]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+ix+1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+nx+ix-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+nx+ix]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+ix+(ny-2)*nx-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+ix+(ny-2)*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+ix+(ny-2)*nx+1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+ix+(ny-1)*nx-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+ix+(ny-1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+ix+(ny-1)*nx+1]
;}
iy=ny-1;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+ix-1]
-xv[nlocal+0+nx+nx*ny+ix]
-xv[nlocal+0+nx+nx*ny+ix+1]
;}
iy=ny-1;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix+1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx+ix-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx+ix]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix+(nz-2)*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix+(nz-1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix+(nz-1)*nx+1]
;}
iy=ny-1;
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+nx*ny+ix-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+nx*ny+ix]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+nx*ny+ix+1]
;}
;
iz=0;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+(iy-1)*nx]
-xv[nlocal+0+nx+(iy-1)*nx+1]
-xv[nlocal+0+nx+(iy)*nx]
-xv[nlocal+0+nx+(iy-1)*nx+1]
-xv[nlocal+0+nx+(iy+1)*nx]
-xv[nlocal+0+nx+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+(iy-1)*nx+nx-2]
-xv[nlocal+0+nx+(iy-1)*nx+nx-1]
-xv[nlocal+0+nx+(iy)*nx+nx-2]
-xv[nlocal+0+nx+(iy)*nx+nx-1]
-xv[nlocal+0+nx+(iy+1)*nx+nx-2]
-xv[nlocal+0+nx+(iy+1)*nx+nx-1]
;}
iz=nz-1;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+(iy-1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+(iy-1)*nx+1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+(iy)*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+(iy-1)*nx+1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+(iy+1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+(iy-1)*nx+nx-2]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+(iy-1)*nx+nx-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+(iy)*nx+nx-2]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+(iy)*nx+nx-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+(iy+1)*nx+nx-2]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+(iy+1)*nx+nx-1]
;}
;
iy=0;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx+(iz-1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+(iz-1)*nx+1]
-xv[nlocal+0+nx+nx*ny+nx+(iz)*nx]
-xv[nlocal+0+nx+nx*ny+nx+(iz-1)*nx+1]
-xv[nlocal+0+nx+nx*ny+nx+(iz+1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx+(iz-1)*nx+nx-2]
-xv[nlocal+0+nx+nx*ny+nx+(iz-1)*nx+nx-1]
-xv[nlocal+0+nx+nx*ny+nx+(iz)*nx+nx-2]
-xv[nlocal+0+nx+nx*ny+nx+(iz)*nx+nx-1]
-xv[nlocal+0+nx+nx*ny+nx+(iz+1)*nx+nx-2]
-xv[nlocal+0+nx+nx*ny+nx+(iz+1)*nx+nx-1]
;}
iy=ny-1;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+(iz-1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+(iz-1)*nx+1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+(iz)*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+(iz-1)*nx+1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+(iz+1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+(iz)*nx+nx-2]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+(iz)*nx+nx-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+(iz+1)*nx+nx-1]
;}
;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx+ix-1+(iz-1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+ix+(iz-1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+ix+1+(iz-1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+ix-1+(iz)*nx]
-xv[nlocal+0+nx+nx*ny+nx+ix+iz*nx]
-xv[nlocal+0+nx+nx*ny+nx+ix+1+iz*nx]
-xv[nlocal+0+nx+nx*ny+nx+ix-1+(iz+1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+ix+(iz+1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+ix+1+(iz+1)*nx]
;}
}
iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix-1+(iz-1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix+(iz-1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix+1+(iz-1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix-1+(iz)*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix+iz*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix+1+iz*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix-1+(iz+1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix+(iz+1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix+1+(iz+1)*nx]
;}
}
iz=0;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+ix-1+(iy-1)*nx]
-xv[nlocal+0+nx+ix+(iy-1)*nx]
-xv[nlocal+0+nx+ix+1+(iy-1)*nx]
-xv[nlocal+0+nx+ix-1+(iy)*nx]
-xv[nlocal+0+nx+ix+iy*nx]
-xv[nlocal+0+nx+ix+1+iy*nx]
-xv[nlocal+0+nx+ix-1+(iy+1)*nx]
-xv[nlocal+0+nx+ix+(iy+1)*nx]
-xv[nlocal+0+nx+ix+1+(iy+1)*nx]
;}
}
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+ix-1+(iy-1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+ix+(iy-1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+ix+1+(iy-1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+ix-1+(iy)*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+ix+iy*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+ix+1+iy*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+ix-1+(iy+1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+ix+(iy+1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+ix+1+(iy+1)*nx]
;}
}
}//ipz < npz - 1
            else
{
ix = 0;
iy = 0;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
;
ix = 0;
iy = ny-1;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+nx*ny]
-xv[nlocal+0+nx+nx*ny+1]
;
ix = nx-1;
iy = 0;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx-2]
-xv[nlocal+0+nx-1]
;
ix = nx-1;
iy = ny-1;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+nx*ny+nx-2]
-xv[nlocal+0+nx+nx*ny+nx-1]
;
iy=0;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
;}
iy=0;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx+ix-1]
-xv[nlocal+0+nx+nx*ny+nx+ix]
-xv[nlocal+0+nx+nx*ny+nx+ix+1]
-xv[nlocal+0+nx+nx*ny+nx+nx+ix-1]
-xv[nlocal+0+nx+nx*ny+nx+nx+ix]
-xv[nlocal+0+nx+nx*ny+nx+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx+ix+(nz-2)*nx-1]
-xv[nlocal+0+nx+nx*ny+nx+ix+(nz-2)*nx]
-xv[nlocal+0+nx+nx*ny+nx+ix+(nz-2)*nx+1]
-xv[nlocal+0+nx+nx*ny+nx+ix+(nz-1)*nx-1]
-xv[nlocal+0+nx+nx*ny+nx+ix+(nz-1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+ix+(nz-1)*nx+1]
;}
iz=0;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+ix-1]
-xv[nlocal+0+nx+ix]
-xv[nlocal+0+nx+ix+1]
-xv[nlocal+0+nx+nx+ix-1]
-xv[nlocal+0+nx+nx+ix]
-xv[nlocal+0+nx+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+ix+(ny-2)*nx-1]
-xv[nlocal+0+nx+ix+(ny-2)*nx]
-xv[nlocal+0+nx+ix+(ny-2)*nx+1]
-xv[nlocal+0+nx+ix+(ny-1)*nx-1]
-xv[nlocal+0+nx+ix+(ny-1)*nx]
-xv[nlocal+0+nx+ix+(ny-1)*nx+1]
;}
iy=ny-1;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+ix-1]
-xv[nlocal+0+nx+nx*ny+ix]
-xv[nlocal+0+nx+nx*ny+ix+1]
;}
iy=ny-1;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix+1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx+ix-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx+ix]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix+(nz-2)*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix+(nz-1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix+(nz-1)*nx+1]
;}
;
iz=0;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+(iy-1)*nx]
-xv[nlocal+0+nx+(iy-1)*nx+1]
-xv[nlocal+0+nx+(iy)*nx]
-xv[nlocal+0+nx+(iy-1)*nx+1]
-xv[nlocal+0+nx+(iy+1)*nx]
-xv[nlocal+0+nx+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+(iy-1)*nx+nx-2]
-xv[nlocal+0+nx+(iy-1)*nx+nx-1]
-xv[nlocal+0+nx+(iy)*nx+nx-2]
-xv[nlocal+0+nx+(iy)*nx+nx-1]
-xv[nlocal+0+nx+(iy+1)*nx+nx-2]
-xv[nlocal+0+nx+(iy+1)*nx+nx-1]
;}
;
iy=0;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx+(iz-1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+(iz-1)*nx+1]
-xv[nlocal+0+nx+nx*ny+nx+(iz)*nx]
-xv[nlocal+0+nx+nx*ny+nx+(iz-1)*nx+1]
-xv[nlocal+0+nx+nx*ny+nx+(iz+1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx+(iz-1)*nx+nx-2]
-xv[nlocal+0+nx+nx*ny+nx+(iz-1)*nx+nx-1]
-xv[nlocal+0+nx+nx*ny+nx+(iz)*nx+nx-2]
-xv[nlocal+0+nx+nx*ny+nx+(iz)*nx+nx-1]
-xv[nlocal+0+nx+nx*ny+nx+(iz+1)*nx+nx-2]
-xv[nlocal+0+nx+nx*ny+nx+(iz+1)*nx+nx-1]
;}
iy=ny-1;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+(iz-1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+(iz-1)*nx+1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+(iz)*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+(iz-1)*nx+1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+(iz+1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+(iz)*nx+nx-2]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+(iz)*nx+nx-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+(iz+1)*nx+nx-1]
;}
;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx+ix-1+(iz-1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+ix+(iz-1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+ix+1+(iz-1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+ix-1+(iz)*nx]
-xv[nlocal+0+nx+nx*ny+nx+ix+iz*nx]
-xv[nlocal+0+nx+nx*ny+nx+ix+1+iz*nx]
-xv[nlocal+0+nx+nx*ny+nx+ix-1+(iz+1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+ix+(iz+1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+ix+1+(iz+1)*nx]
;}
}
iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix-1+(iz-1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix+(iz-1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix+1+(iz-1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix-1+(iz)*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix+iz*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix+1+iz*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix-1+(iz+1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix+(iz+1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix+1+(iz+1)*nx]
;}
}
iz=0;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+ix-1+(iy-1)*nx]
-xv[nlocal+0+nx+ix+(iy-1)*nx]
-xv[nlocal+0+nx+ix+1+(iy-1)*nx]
-xv[nlocal+0+nx+ix-1+(iy)*nx]
-xv[nlocal+0+nx+ix+iy*nx]
-xv[nlocal+0+nx+ix+1+iy*nx]
-xv[nlocal+0+nx+ix-1+(iy+1)*nx]
-xv[nlocal+0+nx+ix+(iy+1)*nx]
-xv[nlocal+0+nx+ix+1+(iy+1)*nx]
;}
}
}//ipz < npz - 1
          }//ipz > 0 
        else
{
            if(ipz < npz - 1)
{
ix = 0;
iy = 0;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*nz+nx*nz]
-xv[nlocal+0+nx*nz+nx*nz+1]
;
ix = 0;
iy = ny-1;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*nz+nx*nz+nx+nx*ny]
-xv[nlocal+0+nx*nz+nx*nz+nx+nx*ny+1]
;
ix = nx-1;
iy = 0;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*nz+nx*nz+nx-2]
-xv[nlocal+0+nx*nz+nx*nz+nx-1]
;
ix = nx-1;
iy = ny-1;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*nz+nx*nz+nx+nx*ny+nx-2]
-xv[nlocal+0+nx*nz+nx*nz+nx+nx*ny+nx-1]
;
iy=0;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
-xv[nlocal+0+nx+ix-1]
-xv[nlocal+0+nx+ix]
-xv[nlocal+0+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix+(nz-2)*nx-1]
-xv[nlocal+0+ix+(nz-2)*nx]
-xv[nlocal+0+ix+(nz-2)*nx+1]
-xv[nlocal+0+ix+(nz-1)*nx-1]
-xv[nlocal+0+ix+(nz-1)*nx]
-xv[nlocal+0+ix+(nz-1)*nx+1]
;}
iy=0;
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nx*nz+ix-1]
-xv[nlocal+0+nx*nz+nx*nz+ix]
-xv[nlocal+0+nx*nz+nx*nz+ix+1]
;}
iz=nz-1;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nx*nz+nx+ix-1]
-xv[nlocal+0+nx*nz+nx*nz+nx+ix]
-xv[nlocal+0+nx*nz+nx*nz+nx+ix+1]
-xv[nlocal+0+nx*nz+nx*nz+nx+nx+ix-1]
-xv[nlocal+0+nx*nz+nx*nz+nx+nx+ix]
-xv[nlocal+0+nx*nz+nx*nz+nx+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nx*nz+nx+ix+(ny-2)*nx-1]
-xv[nlocal+0+nx*nz+nx*nz+nx+ix+(ny-2)*nx]
-xv[nlocal+0+nx*nz+nx*nz+nx+ix+(ny-2)*nx+1]
-xv[nlocal+0+nx*nz+nx*nz+nx+ix+(ny-1)*nx-1]
-xv[nlocal+0+nx*nz+nx*nz+nx+ix+(ny-1)*nx]
-xv[nlocal+0+nx*nz+nx*nz+nx+ix+(ny-1)*nx+1]
;}
iy=ny-1;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+ix-1]
-xv[nlocal+0+nx*nz+ix]
-xv[nlocal+0+nx*nz+ix+1]
-xv[nlocal+0+nx*nz+nx+ix-1]
-xv[nlocal+0+nx*nz+nx+ix]
-xv[nlocal+0+nx*nz+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+nx*nz+ix+(nz-2)*nx]
-xv[nlocal+0+nx*nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+nx*nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+nx*nz+ix+(nz-1)*nx]
-xv[nlocal+0+nx*nz+ix+(nz-1)*nx+1]
;}
iy=ny-1;
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nx*nz+nx+nx*ny+ix-1]
-xv[nlocal+0+nx*nz+nx*nz+nx+nx*ny+ix]
-xv[nlocal+0+nx*nz+nx*nz+nx+nx*ny+ix+1]
;}
;
iz=nz-1;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nx*nz+nx+(iy-1)*nx]
-xv[nlocal+0+nx*nz+nx*nz+nx+(iy-1)*nx+1]
-xv[nlocal+0+nx*nz+nx*nz+nx+(iy)*nx]
-xv[nlocal+0+nx*nz+nx*nz+nx+(iy-1)*nx+1]
-xv[nlocal+0+nx*nz+nx*nz+nx+(iy+1)*nx]
-xv[nlocal+0+nx*nz+nx*nz+nx+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nx*nz+nx+(iy-1)*nx+nx-2]
-xv[nlocal+0+nx*nz+nx*nz+nx+(iy-1)*nx+nx-1]
-xv[nlocal+0+nx*nz+nx*nz+nx+(iy)*nx+nx-2]
-xv[nlocal+0+nx*nz+nx*nz+nx+(iy)*nx+nx-1]
-xv[nlocal+0+nx*nz+nx*nz+nx+(iy+1)*nx+nx-2]
-xv[nlocal+0+nx*nz+nx*nz+nx+(iy+1)*nx+nx-1]
;}
;
iy=0;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*nx]
-xv[nlocal+0+(iz-1)*nx+1]
-xv[nlocal+0+(iz)*nx]
-xv[nlocal+0+(iz-1)*nx+1]
-xv[nlocal+0+(iz+1)*nx]
-xv[nlocal+0+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*nx+nx-2]
-xv[nlocal+0+(iz-1)*nx+nx-1]
-xv[nlocal+0+(iz)*nx+nx-2]
-xv[nlocal+0+(iz)*nx+nx-1]
-xv[nlocal+0+(iz+1)*nx+nx-2]
-xv[nlocal+0+(iz+1)*nx+nx-1]
;}
iy=ny-1;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+(iz-1)*nx]
-xv[nlocal+0+nx*nz+(iz-1)*nx+1]
-xv[nlocal+0+nx*nz+(iz)*nx]
-xv[nlocal+0+nx*nz+(iz-1)*nx+1]
-xv[nlocal+0+nx*nz+(iz+1)*nx]
-xv[nlocal+0+nx*nz+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+nx*nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+nx*nz+(iz)*nx+nx-2]
-xv[nlocal+0+nx*nz+(iz)*nx+nx-1]
-xv[nlocal+0+nx*nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+nx*nz+(iz+1)*nx+nx-1]
;}
;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1+(iz-1)*nx]
-xv[nlocal+0+ix+(iz-1)*nx]
-xv[nlocal+0+ix+1+(iz-1)*nx]
-xv[nlocal+0+ix-1+(iz)*nx]
-xv[nlocal+0+ix+iz*nx]
-xv[nlocal+0+ix+1+iz*nx]
-xv[nlocal+0+ix-1+(iz+1)*nx]
-xv[nlocal+0+ix+(iz+1)*nx]
-xv[nlocal+0+ix+1+(iz+1)*nx]
;}
}
iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+ix-1+(iz-1)*nx]
-xv[nlocal+0+nx*nz+ix+(iz-1)*nx]
-xv[nlocal+0+nx*nz+ix+1+(iz-1)*nx]
-xv[nlocal+0+nx*nz+ix-1+(iz)*nx]
-xv[nlocal+0+nx*nz+ix+iz*nx]
-xv[nlocal+0+nx*nz+ix+1+iz*nx]
-xv[nlocal+0+nx*nz+ix-1+(iz+1)*nx]
-xv[nlocal+0+nx*nz+ix+(iz+1)*nx]
-xv[nlocal+0+nx*nz+ix+1+(iz+1)*nx]
;}
}
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nx*nz+nx+ix-1+(iy-1)*nx]
-xv[nlocal+0+nx*nz+nx*nz+nx+ix+(iy-1)*nx]
-xv[nlocal+0+nx*nz+nx*nz+nx+ix+1+(iy-1)*nx]
-xv[nlocal+0+nx*nz+nx*nz+nx+ix-1+(iy)*nx]
-xv[nlocal+0+nx*nz+nx*nz+nx+ix+iy*nx]
-xv[nlocal+0+nx*nz+nx*nz+nx+ix+1+iy*nx]
-xv[nlocal+0+nx*nz+nx*nz+nx+ix-1+(iy+1)*nx]
-xv[nlocal+0+nx*nz+nx*nz+nx+ix+(iy+1)*nx]
-xv[nlocal+0+nx*nz+nx*nz+nx+ix+1+(iy+1)*nx]
;}
}
}//ipz < npz - 1
            else
{
iy=0;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
-xv[nlocal+0+nx+ix-1]
-xv[nlocal+0+nx+ix]
-xv[nlocal+0+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix+(nz-2)*nx-1]
-xv[nlocal+0+ix+(nz-2)*nx]
-xv[nlocal+0+ix+(nz-2)*nx+1]
-xv[nlocal+0+ix+(nz-1)*nx-1]
-xv[nlocal+0+ix+(nz-1)*nx]
-xv[nlocal+0+ix+(nz-1)*nx+1]
;}
iy=ny-1;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+ix-1]
-xv[nlocal+0+nx*nz+ix]
-xv[nlocal+0+nx*nz+ix+1]
-xv[nlocal+0+nx*nz+nx+ix-1]
-xv[nlocal+0+nx*nz+nx+ix]
-xv[nlocal+0+nx*nz+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+nx*nz+ix+(nz-2)*nx]
-xv[nlocal+0+nx*nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+nx*nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+nx*nz+ix+(nz-1)*nx]
-xv[nlocal+0+nx*nz+ix+(nz-1)*nx+1]
;}
;
;
iy=0;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*nx]
-xv[nlocal+0+(iz-1)*nx+1]
-xv[nlocal+0+(iz)*nx]
-xv[nlocal+0+(iz-1)*nx+1]
-xv[nlocal+0+(iz+1)*nx]
-xv[nlocal+0+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*nx+nx-2]
-xv[nlocal+0+(iz-1)*nx+nx-1]
-xv[nlocal+0+(iz)*nx+nx-2]
-xv[nlocal+0+(iz)*nx+nx-1]
-xv[nlocal+0+(iz+1)*nx+nx-2]
-xv[nlocal+0+(iz+1)*nx+nx-1]
;}
iy=ny-1;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+(iz-1)*nx]
-xv[nlocal+0+nx*nz+(iz-1)*nx+1]
-xv[nlocal+0+nx*nz+(iz)*nx]
-xv[nlocal+0+nx*nz+(iz-1)*nx+1]
-xv[nlocal+0+nx*nz+(iz+1)*nx]
-xv[nlocal+0+nx*nz+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+nx*nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+nx*nz+(iz)*nx+nx-2]
-xv[nlocal+0+nx*nz+(iz)*nx+nx-1]
-xv[nlocal+0+nx*nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+nx*nz+(iz+1)*nx+nx-1]
;}
;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1+(iz-1)*nx]
-xv[nlocal+0+ix+(iz-1)*nx]
-xv[nlocal+0+ix+1+(iz-1)*nx]
-xv[nlocal+0+ix-1+(iz)*nx]
-xv[nlocal+0+ix+iz*nx]
-xv[nlocal+0+ix+1+iz*nx]
-xv[nlocal+0+ix-1+(iz+1)*nx]
-xv[nlocal+0+ix+(iz+1)*nx]
-xv[nlocal+0+ix+1+(iz+1)*nx]
;}
}
iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+ix-1+(iz-1)*nx]
-xv[nlocal+0+nx*nz+ix+(iz-1)*nx]
-xv[nlocal+0+nx*nz+ix+1+(iz-1)*nx]
-xv[nlocal+0+nx*nz+ix-1+(iz)*nx]
-xv[nlocal+0+nx*nz+ix+iz*nx]
-xv[nlocal+0+nx*nz+ix+1+iz*nx]
-xv[nlocal+0+nx*nz+ix-1+(iz+1)*nx]
-xv[nlocal+0+nx*nz+ix+(iz+1)*nx]
-xv[nlocal+0+nx*nz+ix+1+(iz+1)*nx]
;}
}
}//ipz < npz - 1
          }//ipz > 0 
        }//ipy < npy - 1 
      else
{
        if(ipz > 0)
{
            if(ipz < npz - 1)
{
ix = 0;
iy = 0;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
;
ix = 0;
iy = 0;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+nx*ny+nx*nz]
-xv[nlocal+0+nx+nx*ny+nx*nz+1]
;
ix = nx-1;
iy = 0;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx-2]
-xv[nlocal+0+nx-1]
;
ix = nx-1;
iy = 0;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+nx*ny+nx*nz+nx-2]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx-1]
;
iy=0;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
;}
iy=0;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+ix-1]
-xv[nlocal+0+nx+nx*ny+ix]
-xv[nlocal+0+nx+nx*ny+ix+1]
-xv[nlocal+0+nx+nx*ny+nx+ix-1]
-xv[nlocal+0+nx+nx*ny+nx+ix]
-xv[nlocal+0+nx+nx*ny+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+ix+(nz-2)*nx-1]
-xv[nlocal+0+nx+nx*ny+ix+(nz-2)*nx]
-xv[nlocal+0+nx+nx*ny+ix+(nz-2)*nx+1]
-xv[nlocal+0+nx+nx*ny+ix+(nz-1)*nx-1]
-xv[nlocal+0+nx+nx*ny+ix+(nz-1)*nx]
-xv[nlocal+0+nx+nx*ny+ix+(nz-1)*nx+1]
;}
iy=0;
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx*nz+ix-1]
-xv[nlocal+0+nx+nx*ny+nx*nz+ix]
-xv[nlocal+0+nx+nx*ny+nx*nz+ix+1]
;}
iz=0;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+ix-1]
-xv[nlocal+0+nx+ix]
-xv[nlocal+0+nx+ix+1]
-xv[nlocal+0+nx+nx+ix-1]
-xv[nlocal+0+nx+nx+ix]
-xv[nlocal+0+nx+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+ix+(ny-2)*nx-1]
-xv[nlocal+0+nx+ix+(ny-2)*nx]
-xv[nlocal+0+nx+ix+(ny-2)*nx+1]
-xv[nlocal+0+nx+ix+(ny-1)*nx-1]
-xv[nlocal+0+nx+ix+(ny-1)*nx]
-xv[nlocal+0+nx+ix+(ny-1)*nx+1]
;}
iz=nz-1;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+ix-1]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+ix]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+ix+1]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+nx+ix-1]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+nx+ix]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+ix+(ny-2)*nx-1]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+ix+(ny-2)*nx]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+ix+(ny-2)*nx+1]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+ix+(ny-1)*nx-1]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+ix+(ny-1)*nx]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+ix+(ny-1)*nx+1]
;}
;
iz=0;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+(iy-1)*nx]
-xv[nlocal+0+nx+(iy-1)*nx+1]
-xv[nlocal+0+nx+(iy)*nx]
-xv[nlocal+0+nx+(iy-1)*nx+1]
-xv[nlocal+0+nx+(iy+1)*nx]
-xv[nlocal+0+nx+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+(iy-1)*nx+nx-2]
-xv[nlocal+0+nx+(iy-1)*nx+nx-1]
-xv[nlocal+0+nx+(iy)*nx+nx-2]
-xv[nlocal+0+nx+(iy)*nx+nx-1]
-xv[nlocal+0+nx+(iy+1)*nx+nx-2]
-xv[nlocal+0+nx+(iy+1)*nx+nx-1]
;}
iz=nz-1;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+(iy-1)*nx]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+(iy-1)*nx+1]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+(iy)*nx]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+(iy-1)*nx+1]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+(iy+1)*nx]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+(iy-1)*nx+nx-2]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+(iy-1)*nx+nx-1]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+(iy)*nx+nx-2]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+(iy)*nx+nx-1]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+(iy+1)*nx+nx-2]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+(iy+1)*nx+nx-1]
;}
;
iy=0;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+(iz-1)*nx]
-xv[nlocal+0+nx+nx*ny+(iz-1)*nx+1]
-xv[nlocal+0+nx+nx*ny+(iz)*nx]
-xv[nlocal+0+nx+nx*ny+(iz-1)*nx+1]
-xv[nlocal+0+nx+nx*ny+(iz+1)*nx]
-xv[nlocal+0+nx+nx*ny+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+(iz-1)*nx+nx-2]
-xv[nlocal+0+nx+nx*ny+(iz-1)*nx+nx-1]
-xv[nlocal+0+nx+nx*ny+(iz)*nx+nx-2]
-xv[nlocal+0+nx+nx*ny+(iz)*nx+nx-1]
-xv[nlocal+0+nx+nx*ny+(iz+1)*nx+nx-2]
-xv[nlocal+0+nx+nx*ny+(iz+1)*nx+nx-1]
;}
;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+ix-1+(iz-1)*nx]
-xv[nlocal+0+nx+nx*ny+ix+(iz-1)*nx]
-xv[nlocal+0+nx+nx*ny+ix+1+(iz-1)*nx]
-xv[nlocal+0+nx+nx*ny+ix-1+(iz)*nx]
-xv[nlocal+0+nx+nx*ny+ix+iz*nx]
-xv[nlocal+0+nx+nx*ny+ix+1+iz*nx]
-xv[nlocal+0+nx+nx*ny+ix-1+(iz+1)*nx]
-xv[nlocal+0+nx+nx*ny+ix+(iz+1)*nx]
-xv[nlocal+0+nx+nx*ny+ix+1+(iz+1)*nx]
;}
}
iz=0;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+ix-1+(iy-1)*nx]
-xv[nlocal+0+nx+ix+(iy-1)*nx]
-xv[nlocal+0+nx+ix+1+(iy-1)*nx]
-xv[nlocal+0+nx+ix-1+(iy)*nx]
-xv[nlocal+0+nx+ix+iy*nx]
-xv[nlocal+0+nx+ix+1+iy*nx]
-xv[nlocal+0+nx+ix-1+(iy+1)*nx]
-xv[nlocal+0+nx+ix+(iy+1)*nx]
-xv[nlocal+0+nx+ix+1+(iy+1)*nx]
;}
}
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+ix-1+(iy-1)*nx]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+ix+(iy-1)*nx]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+ix+1+(iy-1)*nx]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+ix-1+(iy)*nx]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+ix+iy*nx]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+ix+1+iy*nx]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+ix-1+(iy+1)*nx]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+ix+(iy+1)*nx]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+ix+1+(iy+1)*nx]
;}
}
}//ipz < npz - 1
            else
{
ix = 0;
iy = 0;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
;
ix = nx-1;
iy = 0;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx-2]
-xv[nlocal+0+nx-1]
;
iy=0;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
;}
iy=0;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+ix-1]
-xv[nlocal+0+nx+nx*ny+ix]
-xv[nlocal+0+nx+nx*ny+ix+1]
-xv[nlocal+0+nx+nx*ny+nx+ix-1]
-xv[nlocal+0+nx+nx*ny+nx+ix]
-xv[nlocal+0+nx+nx*ny+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+ix+(nz-2)*nx-1]
-xv[nlocal+0+nx+nx*ny+ix+(nz-2)*nx]
-xv[nlocal+0+nx+nx*ny+ix+(nz-2)*nx+1]
-xv[nlocal+0+nx+nx*ny+ix+(nz-1)*nx-1]
-xv[nlocal+0+nx+nx*ny+ix+(nz-1)*nx]
-xv[nlocal+0+nx+nx*ny+ix+(nz-1)*nx+1]
;}
iz=0;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+ix-1]
-xv[nlocal+0+nx+ix]
-xv[nlocal+0+nx+ix+1]
-xv[nlocal+0+nx+nx+ix-1]
-xv[nlocal+0+nx+nx+ix]
-xv[nlocal+0+nx+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+ix+(ny-2)*nx-1]
-xv[nlocal+0+nx+ix+(ny-2)*nx]
-xv[nlocal+0+nx+ix+(ny-2)*nx+1]
-xv[nlocal+0+nx+ix+(ny-1)*nx-1]
-xv[nlocal+0+nx+ix+(ny-1)*nx]
-xv[nlocal+0+nx+ix+(ny-1)*nx+1]
;}
;
iz=0;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+(iy-1)*nx]
-xv[nlocal+0+nx+(iy-1)*nx+1]
-xv[nlocal+0+nx+(iy)*nx]
-xv[nlocal+0+nx+(iy-1)*nx+1]
-xv[nlocal+0+nx+(iy+1)*nx]
-xv[nlocal+0+nx+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+(iy-1)*nx+nx-2]
-xv[nlocal+0+nx+(iy-1)*nx+nx-1]
-xv[nlocal+0+nx+(iy)*nx+nx-2]
-xv[nlocal+0+nx+(iy)*nx+nx-1]
-xv[nlocal+0+nx+(iy+1)*nx+nx-2]
-xv[nlocal+0+nx+(iy+1)*nx+nx-1]
;}
;
iy=0;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+(iz-1)*nx]
-xv[nlocal+0+nx+nx*ny+(iz-1)*nx+1]
-xv[nlocal+0+nx+nx*ny+(iz)*nx]
-xv[nlocal+0+nx+nx*ny+(iz-1)*nx+1]
-xv[nlocal+0+nx+nx*ny+(iz+1)*nx]
-xv[nlocal+0+nx+nx*ny+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+(iz-1)*nx+nx-2]
-xv[nlocal+0+nx+nx*ny+(iz-1)*nx+nx-1]
-xv[nlocal+0+nx+nx*ny+(iz)*nx+nx-2]
-xv[nlocal+0+nx+nx*ny+(iz)*nx+nx-1]
-xv[nlocal+0+nx+nx*ny+(iz+1)*nx+nx-2]
-xv[nlocal+0+nx+nx*ny+(iz+1)*nx+nx-1]
;}
;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+ix-1+(iz-1)*nx]
-xv[nlocal+0+nx+nx*ny+ix+(iz-1)*nx]
-xv[nlocal+0+nx+nx*ny+ix+1+(iz-1)*nx]
-xv[nlocal+0+nx+nx*ny+ix-1+(iz)*nx]
-xv[nlocal+0+nx+nx*ny+ix+iz*nx]
-xv[nlocal+0+nx+nx*ny+ix+1+iz*nx]
-xv[nlocal+0+nx+nx*ny+ix-1+(iz+1)*nx]
-xv[nlocal+0+nx+nx*ny+ix+(iz+1)*nx]
-xv[nlocal+0+nx+nx*ny+ix+1+(iz+1)*nx]
;}
}
iz=0;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+ix-1+(iy-1)*nx]
-xv[nlocal+0+nx+ix+(iy-1)*nx]
-xv[nlocal+0+nx+ix+1+(iy-1)*nx]
-xv[nlocal+0+nx+ix-1+(iy)*nx]
-xv[nlocal+0+nx+ix+iy*nx]
-xv[nlocal+0+nx+ix+1+iy*nx]
-xv[nlocal+0+nx+ix-1+(iy+1)*nx]
-xv[nlocal+0+nx+ix+(iy+1)*nx]
-xv[nlocal+0+nx+ix+1+(iy+1)*nx]
;}
}
}//ipz < npz - 1
          }//ipz > 0 
        else
{
            if(ipz < npz - 1)
{
ix = 0;
iy = 0;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*nz]
-xv[nlocal+0+nx*nz+1]
;
ix = nx-1;
iy = 0;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*nz+nx-2]
-xv[nlocal+0+nx*nz+nx-1]
;
iy=0;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
-xv[nlocal+0+nx+ix-1]
-xv[nlocal+0+nx+ix]
-xv[nlocal+0+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix+(nz-2)*nx-1]
-xv[nlocal+0+ix+(nz-2)*nx]
-xv[nlocal+0+ix+(nz-2)*nx+1]
-xv[nlocal+0+ix+(nz-1)*nx-1]
-xv[nlocal+0+ix+(nz-1)*nx]
-xv[nlocal+0+ix+(nz-1)*nx+1]
;}
iy=0;
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+ix-1]
-xv[nlocal+0+nx*nz+ix]
-xv[nlocal+0+nx*nz+ix+1]
;}
iz=nz-1;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nx+ix-1]
-xv[nlocal+0+nx*nz+nx+ix]
-xv[nlocal+0+nx*nz+nx+ix+1]
-xv[nlocal+0+nx*nz+nx+nx+ix-1]
-xv[nlocal+0+nx*nz+nx+nx+ix]
-xv[nlocal+0+nx*nz+nx+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nx+ix+(ny-2)*nx-1]
-xv[nlocal+0+nx*nz+nx+ix+(ny-2)*nx]
-xv[nlocal+0+nx*nz+nx+ix+(ny-2)*nx+1]
-xv[nlocal+0+nx*nz+nx+ix+(ny-1)*nx-1]
-xv[nlocal+0+nx*nz+nx+ix+(ny-1)*nx]
-xv[nlocal+0+nx*nz+nx+ix+(ny-1)*nx+1]
;}
;
iz=nz-1;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nx+(iy-1)*nx]
-xv[nlocal+0+nx*nz+nx+(iy-1)*nx+1]
-xv[nlocal+0+nx*nz+nx+(iy)*nx]
-xv[nlocal+0+nx*nz+nx+(iy-1)*nx+1]
-xv[nlocal+0+nx*nz+nx+(iy+1)*nx]
-xv[nlocal+0+nx*nz+nx+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nx+(iy-1)*nx+nx-2]
-xv[nlocal+0+nx*nz+nx+(iy-1)*nx+nx-1]
-xv[nlocal+0+nx*nz+nx+(iy)*nx+nx-2]
-xv[nlocal+0+nx*nz+nx+(iy)*nx+nx-1]
-xv[nlocal+0+nx*nz+nx+(iy+1)*nx+nx-2]
-xv[nlocal+0+nx*nz+nx+(iy+1)*nx+nx-1]
;}
;
iy=0;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*nx]
-xv[nlocal+0+(iz-1)*nx+1]
-xv[nlocal+0+(iz)*nx]
-xv[nlocal+0+(iz-1)*nx+1]
-xv[nlocal+0+(iz+1)*nx]
-xv[nlocal+0+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*nx+nx-2]
-xv[nlocal+0+(iz-1)*nx+nx-1]
-xv[nlocal+0+(iz)*nx+nx-2]
-xv[nlocal+0+(iz)*nx+nx-1]
-xv[nlocal+0+(iz+1)*nx+nx-2]
-xv[nlocal+0+(iz+1)*nx+nx-1]
;}
;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1+(iz-1)*nx]
-xv[nlocal+0+ix+(iz-1)*nx]
-xv[nlocal+0+ix+1+(iz-1)*nx]
-xv[nlocal+0+ix-1+(iz)*nx]
-xv[nlocal+0+ix+iz*nx]
-xv[nlocal+0+ix+1+iz*nx]
-xv[nlocal+0+ix-1+(iz+1)*nx]
-xv[nlocal+0+ix+(iz+1)*nx]
-xv[nlocal+0+ix+1+(iz+1)*nx]
;}
}
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nx+ix-1+(iy-1)*nx]
-xv[nlocal+0+nx*nz+nx+ix+(iy-1)*nx]
-xv[nlocal+0+nx*nz+nx+ix+1+(iy-1)*nx]
-xv[nlocal+0+nx*nz+nx+ix-1+(iy)*nx]
-xv[nlocal+0+nx*nz+nx+ix+iy*nx]
-xv[nlocal+0+nx*nz+nx+ix+1+iy*nx]
-xv[nlocal+0+nx*nz+nx+ix-1+(iy+1)*nx]
-xv[nlocal+0+nx*nz+nx+ix+(iy+1)*nx]
-xv[nlocal+0+nx*nz+nx+ix+1+(iy+1)*nx]
;}
}
}//ipz < npz - 1
            else
{
iy=0;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
-xv[nlocal+0+nx+ix-1]
-xv[nlocal+0+nx+ix]
-xv[nlocal+0+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix+(nz-2)*nx-1]
-xv[nlocal+0+ix+(nz-2)*nx]
-xv[nlocal+0+ix+(nz-2)*nx+1]
-xv[nlocal+0+ix+(nz-1)*nx-1]
-xv[nlocal+0+ix+(nz-1)*nx]
-xv[nlocal+0+ix+(nz-1)*nx+1]
;}
;
;
iy=0;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*nx]
-xv[nlocal+0+(iz-1)*nx+1]
-xv[nlocal+0+(iz)*nx]
-xv[nlocal+0+(iz-1)*nx+1]
-xv[nlocal+0+(iz+1)*nx]
-xv[nlocal+0+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*nx+nx-2]
-xv[nlocal+0+(iz-1)*nx+nx-1]
-xv[nlocal+0+(iz)*nx+nx-2]
-xv[nlocal+0+(iz)*nx+nx-1]
-xv[nlocal+0+(iz+1)*nx+nx-2]
-xv[nlocal+0+(iz+1)*nx+nx-1]
;}
;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1+(iz-1)*nx]
-xv[nlocal+0+ix+(iz-1)*nx]
-xv[nlocal+0+ix+1+(iz-1)*nx]
-xv[nlocal+0+ix-1+(iz)*nx]
-xv[nlocal+0+ix+iz*nx]
-xv[nlocal+0+ix+1+iz*nx]
-xv[nlocal+0+ix-1+(iz+1)*nx]
-xv[nlocal+0+ix+(iz+1)*nx]
-xv[nlocal+0+ix+1+(iz+1)*nx]
;}
}
}//ipz < npz - 1
          }//ipz > 0 
        }//ipy < npy - 1 
      }//ipy > 0 
    else
{
      if(ipy < npy - 1)
{
        if(ipz > 0)
{
            if(ipz < npz - 1)
{
ix = 0;
iy = ny-1;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*ny]
-xv[nlocal+0+nx*ny+1]
;
ix = 0;
iy = ny-1;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*ny+nx+nx*nz+nx*ny]
-xv[nlocal+0+nx*ny+nx+nx*nz+nx*ny+1]
;
ix = nx-1;
iy = ny-1;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*ny+nx-2]
-xv[nlocal+0+nx*ny+nx-1]
;
ix = nx-1;
iy = ny-1;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*ny+nx+nx*nz+nx*ny+nx-2]
-xv[nlocal+0+nx*ny+nx+nx*nz+nx*ny+nx-1]
;
iz=0;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
-xv[nlocal+0+nx+ix-1]
-xv[nlocal+0+nx+ix]
-xv[nlocal+0+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix+(ny-2)*nx-1]
-xv[nlocal+0+ix+(ny-2)*nx]
-xv[nlocal+0+ix+(ny-2)*nx+1]
-xv[nlocal+0+ix+(ny-1)*nx-1]
-xv[nlocal+0+ix+(ny-1)*nx]
-xv[nlocal+0+ix+(ny-1)*nx+1]
;}
iz=nz-1;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+nx+nx*nz+ix-1]
-xv[nlocal+0+nx*ny+nx+nx*nz+ix]
-xv[nlocal+0+nx*ny+nx+nx*nz+ix+1]
-xv[nlocal+0+nx*ny+nx+nx*nz+nx+ix-1]
-xv[nlocal+0+nx*ny+nx+nx*nz+nx+ix]
-xv[nlocal+0+nx*ny+nx+nx*nz+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+nx+nx*nz+ix+(ny-2)*nx-1]
-xv[nlocal+0+nx*ny+nx+nx*nz+ix+(ny-2)*nx]
-xv[nlocal+0+nx*ny+nx+nx*nz+ix+(ny-2)*nx+1]
-xv[nlocal+0+nx*ny+nx+nx*nz+ix+(ny-1)*nx-1]
-xv[nlocal+0+nx*ny+nx+nx*nz+ix+(ny-1)*nx]
-xv[nlocal+0+nx*ny+nx+nx*nz+ix+(ny-1)*nx+1]
;}
iy=ny-1;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ix-1]
-xv[nlocal+0+nx*ny+ix]
-xv[nlocal+0+nx*ny+ix+1]
;}
iy=ny-1;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+nx+ix-1]
-xv[nlocal+0+nx*ny+nx+ix]
-xv[nlocal+0+nx*ny+nx+ix+1]
-xv[nlocal+0+nx*ny+nx+nx+ix-1]
-xv[nlocal+0+nx*ny+nx+nx+ix]
-xv[nlocal+0+nx*ny+nx+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+nx+ix+(nz-2)*nx-1]
-xv[nlocal+0+nx*ny+nx+ix+(nz-2)*nx]
-xv[nlocal+0+nx*ny+nx+ix+(nz-2)*nx+1]
-xv[nlocal+0+nx*ny+nx+ix+(nz-1)*nx-1]
-xv[nlocal+0+nx*ny+nx+ix+(nz-1)*nx]
-xv[nlocal+0+nx*ny+nx+ix+(nz-1)*nx+1]
;}
iy=ny-1;
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+nx+nx*nz+nx*ny+ix-1]
-xv[nlocal+0+nx*ny+nx+nx*nz+nx*ny+ix]
-xv[nlocal+0+nx*ny+nx+nx*nz+nx*ny+ix+1]
;}
;
iz=0;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iy-1)*nx]
-xv[nlocal+0+(iy-1)*nx+1]
-xv[nlocal+0+(iy)*nx]
-xv[nlocal+0+(iy-1)*nx+1]
-xv[nlocal+0+(iy+1)*nx]
-xv[nlocal+0+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iy-1)*nx+nx-2]
-xv[nlocal+0+(iy-1)*nx+nx-1]
-xv[nlocal+0+(iy)*nx+nx-2]
-xv[nlocal+0+(iy)*nx+nx-1]
-xv[nlocal+0+(iy+1)*nx+nx-2]
-xv[nlocal+0+(iy+1)*nx+nx-1]
;}
iz=nz-1;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+nx+nx*nz+(iy-1)*nx]
-xv[nlocal+0+nx*ny+nx+nx*nz+(iy-1)*nx+1]
-xv[nlocal+0+nx*ny+nx+nx*nz+(iy)*nx]
-xv[nlocal+0+nx*ny+nx+nx*nz+(iy-1)*nx+1]
-xv[nlocal+0+nx*ny+nx+nx*nz+(iy+1)*nx]
-xv[nlocal+0+nx*ny+nx+nx*nz+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+nx+nx*nz+(iy-1)*nx+nx-2]
-xv[nlocal+0+nx*ny+nx+nx*nz+(iy-1)*nx+nx-1]
-xv[nlocal+0+nx*ny+nx+nx*nz+(iy)*nx+nx-2]
-xv[nlocal+0+nx*ny+nx+nx*nz+(iy)*nx+nx-1]
-xv[nlocal+0+nx*ny+nx+nx*nz+(iy+1)*nx+nx-2]
-xv[nlocal+0+nx*ny+nx+nx*nz+(iy+1)*nx+nx-1]
;}
;
iy=ny-1;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+nx+(iz-1)*nx]
-xv[nlocal+0+nx*ny+nx+(iz-1)*nx+1]
-xv[nlocal+0+nx*ny+nx+(iz)*nx]
-xv[nlocal+0+nx*ny+nx+(iz-1)*nx+1]
-xv[nlocal+0+nx*ny+nx+(iz+1)*nx]
-xv[nlocal+0+nx*ny+nx+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+nx+(iz-1)*nx+nx-2]
-xv[nlocal+0+nx*ny+nx+(iz-1)*nx+nx-1]
-xv[nlocal+0+nx*ny+nx+(iz)*nx+nx-2]
-xv[nlocal+0+nx*ny+nx+(iz)*nx+nx-1]
-xv[nlocal+0+nx*ny+nx+(iz+1)*nx+nx-2]
-xv[nlocal+0+nx*ny+nx+(iz+1)*nx+nx-1]
;}
;
iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+nx+ix-1+(iz-1)*nx]
-xv[nlocal+0+nx*ny+nx+ix+(iz-1)*nx]
-xv[nlocal+0+nx*ny+nx+ix+1+(iz-1)*nx]
-xv[nlocal+0+nx*ny+nx+ix-1+(iz)*nx]
-xv[nlocal+0+nx*ny+nx+ix+iz*nx]
-xv[nlocal+0+nx*ny+nx+ix+1+iz*nx]
-xv[nlocal+0+nx*ny+nx+ix-1+(iz+1)*nx]
-xv[nlocal+0+nx*ny+nx+ix+(iz+1)*nx]
-xv[nlocal+0+nx*ny+nx+ix+1+(iz+1)*nx]
;}
}
iz=0;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1+(iy-1)*nx]
-xv[nlocal+0+ix+(iy-1)*nx]
-xv[nlocal+0+ix+1+(iy-1)*nx]
-xv[nlocal+0+ix-1+(iy)*nx]
-xv[nlocal+0+ix+iy*nx]
-xv[nlocal+0+ix+1+iy*nx]
-xv[nlocal+0+ix-1+(iy+1)*nx]
-xv[nlocal+0+ix+(iy+1)*nx]
-xv[nlocal+0+ix+1+(iy+1)*nx]
;}
}
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+nx+nx*nz+ix-1+(iy-1)*nx]
-xv[nlocal+0+nx*ny+nx+nx*nz+ix+(iy-1)*nx]
-xv[nlocal+0+nx*ny+nx+nx*nz+ix+1+(iy-1)*nx]
-xv[nlocal+0+nx*ny+nx+nx*nz+ix-1+(iy)*nx]
-xv[nlocal+0+nx*ny+nx+nx*nz+ix+iy*nx]
-xv[nlocal+0+nx*ny+nx+nx*nz+ix+1+iy*nx]
-xv[nlocal+0+nx*ny+nx+nx*nz+ix-1+(iy+1)*nx]
-xv[nlocal+0+nx*ny+nx+nx*nz+ix+(iy+1)*nx]
-xv[nlocal+0+nx*ny+nx+nx*nz+ix+1+(iy+1)*nx]
;}
}
}//ipz < npz - 1
            else
{
ix = 0;
iy = ny-1;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*ny]
-xv[nlocal+0+nx*ny+1]
;
ix = nx-1;
iy = ny-1;
iz = 0;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*ny+nx-2]
-xv[nlocal+0+nx*ny+nx-1]
;
iz=0;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
-xv[nlocal+0+nx+ix-1]
-xv[nlocal+0+nx+ix]
-xv[nlocal+0+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix+(ny-2)*nx-1]
-xv[nlocal+0+ix+(ny-2)*nx]
-xv[nlocal+0+ix+(ny-2)*nx+1]
-xv[nlocal+0+ix+(ny-1)*nx-1]
-xv[nlocal+0+ix+(ny-1)*nx]
-xv[nlocal+0+ix+(ny-1)*nx+1]
;}
iy=ny-1;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ix-1]
-xv[nlocal+0+nx*ny+ix]
-xv[nlocal+0+nx*ny+ix+1]
;}
iy=ny-1;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+nx+ix-1]
-xv[nlocal+0+nx*ny+nx+ix]
-xv[nlocal+0+nx*ny+nx+ix+1]
-xv[nlocal+0+nx*ny+nx+nx+ix-1]
-xv[nlocal+0+nx*ny+nx+nx+ix]
-xv[nlocal+0+nx*ny+nx+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+nx+ix+(nz-2)*nx-1]
-xv[nlocal+0+nx*ny+nx+ix+(nz-2)*nx]
-xv[nlocal+0+nx*ny+nx+ix+(nz-2)*nx+1]
-xv[nlocal+0+nx*ny+nx+ix+(nz-1)*nx-1]
-xv[nlocal+0+nx*ny+nx+ix+(nz-1)*nx]
-xv[nlocal+0+nx*ny+nx+ix+(nz-1)*nx+1]
;}
;
iz=0;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iy-1)*nx]
-xv[nlocal+0+(iy-1)*nx+1]
-xv[nlocal+0+(iy)*nx]
-xv[nlocal+0+(iy-1)*nx+1]
-xv[nlocal+0+(iy+1)*nx]
-xv[nlocal+0+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iy-1)*nx+nx-2]
-xv[nlocal+0+(iy-1)*nx+nx-1]
-xv[nlocal+0+(iy)*nx+nx-2]
-xv[nlocal+0+(iy)*nx+nx-1]
-xv[nlocal+0+(iy+1)*nx+nx-2]
-xv[nlocal+0+(iy+1)*nx+nx-1]
;}
;
iy=ny-1;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+nx+(iz-1)*nx]
-xv[nlocal+0+nx*ny+nx+(iz-1)*nx+1]
-xv[nlocal+0+nx*ny+nx+(iz)*nx]
-xv[nlocal+0+nx*ny+nx+(iz-1)*nx+1]
-xv[nlocal+0+nx*ny+nx+(iz+1)*nx]
-xv[nlocal+0+nx*ny+nx+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+nx+(iz-1)*nx+nx-2]
-xv[nlocal+0+nx*ny+nx+(iz-1)*nx+nx-1]
-xv[nlocal+0+nx*ny+nx+(iz)*nx+nx-2]
-xv[nlocal+0+nx*ny+nx+(iz)*nx+nx-1]
-xv[nlocal+0+nx*ny+nx+(iz+1)*nx+nx-2]
-xv[nlocal+0+nx*ny+nx+(iz+1)*nx+nx-1]
;}
;
iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+nx+ix-1+(iz-1)*nx]
-xv[nlocal+0+nx*ny+nx+ix+(iz-1)*nx]
-xv[nlocal+0+nx*ny+nx+ix+1+(iz-1)*nx]
-xv[nlocal+0+nx*ny+nx+ix-1+(iz)*nx]
-xv[nlocal+0+nx*ny+nx+ix+iz*nx]
-xv[nlocal+0+nx*ny+nx+ix+1+iz*nx]
-xv[nlocal+0+nx*ny+nx+ix-1+(iz+1)*nx]
-xv[nlocal+0+nx*ny+nx+ix+(iz+1)*nx]
-xv[nlocal+0+nx*ny+nx+ix+1+(iz+1)*nx]
;}
}
iz=0;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1+(iy-1)*nx]
-xv[nlocal+0+ix+(iy-1)*nx]
-xv[nlocal+0+ix+1+(iy-1)*nx]
-xv[nlocal+0+ix-1+(iy)*nx]
-xv[nlocal+0+ix+iy*nx]
-xv[nlocal+0+ix+1+iy*nx]
-xv[nlocal+0+ix-1+(iy+1)*nx]
-xv[nlocal+0+ix+(iy+1)*nx]
-xv[nlocal+0+ix+1+(iy+1)*nx]
;}
}
}//ipz < npz - 1
          }//ipz > 0 
        else
{
            if(ipz < npz - 1)
{
ix = 0;
iy = ny-1;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*nz+nx*ny]
-xv[nlocal+0+nx*nz+nx*ny+1]
;
ix = nx-1;
iy = ny-1;
iz = nz-1;
yv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*nz+nx*ny+nx-2]
-xv[nlocal+0+nx*nz+nx*ny+nx-1]
;
iz=nz-1;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+ix-1]
-xv[nlocal+0+nx*nz+ix]
-xv[nlocal+0+nx*nz+ix+1]
-xv[nlocal+0+nx*nz+nx+ix-1]
-xv[nlocal+0+nx*nz+nx+ix]
-xv[nlocal+0+nx*nz+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+ix+(ny-2)*nx-1]
-xv[nlocal+0+nx*nz+ix+(ny-2)*nx]
-xv[nlocal+0+nx*nz+ix+(ny-2)*nx+1]
-xv[nlocal+0+nx*nz+ix+(ny-1)*nx-1]
-xv[nlocal+0+nx*nz+ix+(ny-1)*nx]
-xv[nlocal+0+nx*nz+ix+(ny-1)*nx+1]
;}
iy=ny-1;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
-xv[nlocal+0+nx+ix-1]
-xv[nlocal+0+nx+ix]
-xv[nlocal+0+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix+(nz-2)*nx-1]
-xv[nlocal+0+ix+(nz-2)*nx]
-xv[nlocal+0+ix+(nz-2)*nx+1]
-xv[nlocal+0+ix+(nz-1)*nx-1]
-xv[nlocal+0+ix+(nz-1)*nx]
-xv[nlocal+0+ix+(nz-1)*nx+1]
;}
iy=ny-1;
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nx*ny+ix-1]
-xv[nlocal+0+nx*nz+nx*ny+ix]
-xv[nlocal+0+nx*nz+nx*ny+ix+1]
;}
;
iz=nz-1;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+(iy-1)*nx]
-xv[nlocal+0+nx*nz+(iy-1)*nx+1]
-xv[nlocal+0+nx*nz+(iy)*nx]
-xv[nlocal+0+nx*nz+(iy-1)*nx+1]
-xv[nlocal+0+nx*nz+(iy+1)*nx]
-xv[nlocal+0+nx*nz+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+(iy-1)*nx+nx-2]
-xv[nlocal+0+nx*nz+(iy-1)*nx+nx-1]
-xv[nlocal+0+nx*nz+(iy)*nx+nx-2]
-xv[nlocal+0+nx*nz+(iy)*nx+nx-1]
-xv[nlocal+0+nx*nz+(iy+1)*nx+nx-2]
-xv[nlocal+0+nx*nz+(iy+1)*nx+nx-1]
;}
;
iy=ny-1;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*nx]
-xv[nlocal+0+(iz-1)*nx+1]
-xv[nlocal+0+(iz)*nx]
-xv[nlocal+0+(iz-1)*nx+1]
-xv[nlocal+0+(iz+1)*nx]
-xv[nlocal+0+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*nx+nx-2]
-xv[nlocal+0+(iz-1)*nx+nx-1]
-xv[nlocal+0+(iz)*nx+nx-2]
-xv[nlocal+0+(iz)*nx+nx-1]
-xv[nlocal+0+(iz+1)*nx+nx-2]
-xv[nlocal+0+(iz+1)*nx+nx-1]
;}
;
iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1+(iz-1)*nx]
-xv[nlocal+0+ix+(iz-1)*nx]
-xv[nlocal+0+ix+1+(iz-1)*nx]
-xv[nlocal+0+ix-1+(iz)*nx]
-xv[nlocal+0+ix+iz*nx]
-xv[nlocal+0+ix+1+iz*nx]
-xv[nlocal+0+ix-1+(iz+1)*nx]
-xv[nlocal+0+ix+(iz+1)*nx]
-xv[nlocal+0+ix+1+(iz+1)*nx]
;}
}
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+ix-1+(iy-1)*nx]
-xv[nlocal+0+nx*nz+ix+(iy-1)*nx]
-xv[nlocal+0+nx*nz+ix+1+(iy-1)*nx]
-xv[nlocal+0+nx*nz+ix-1+(iy)*nx]
-xv[nlocal+0+nx*nz+ix+iy*nx]
-xv[nlocal+0+nx*nz+ix+1+iy*nx]
-xv[nlocal+0+nx*nz+ix-1+(iy+1)*nx]
-xv[nlocal+0+nx*nz+ix+(iy+1)*nx]
-xv[nlocal+0+nx*nz+ix+1+(iy+1)*nx]
;}
}
}//ipz < npz - 1
            else
{
iy=ny-1;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
-xv[nlocal+0+nx+ix-1]
-xv[nlocal+0+nx+ix]
-xv[nlocal+0+nx+ix+1]
;}iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix+(nz-2)*nx-1]
-xv[nlocal+0+ix+(nz-2)*nx]
-xv[nlocal+0+ix+(nz-2)*nx+1]
-xv[nlocal+0+ix+(nz-1)*nx-1]
-xv[nlocal+0+ix+(nz-1)*nx]
-xv[nlocal+0+ix+(nz-1)*nx+1]
;}
;
;
iy=ny-1;
ix=0;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*nx]
-xv[nlocal+0+(iz-1)*nx+1]
-xv[nlocal+0+(iz)*nx]
-xv[nlocal+0+(iz-1)*nx+1]
-xv[nlocal+0+(iz+1)*nx]
-xv[nlocal+0+(iz+1)*nx+1]
;}ix=nx-1;
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*nx+nx-2]
-xv[nlocal+0+(iz-1)*nx+nx-1]
-xv[nlocal+0+(iz)*nx+nx-2]
-xv[nlocal+0+(iz)*nx+nx-1]
-xv[nlocal+0+(iz+1)*nx+nx-2]
-xv[nlocal+0+(iz+1)*nx+nx-1]
;}
;
iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
for (iz=1; iz<nz-1;iz++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1+(iz-1)*nx]
-xv[nlocal+0+ix+(iz-1)*nx]
-xv[nlocal+0+ix+1+(iz-1)*nx]
-xv[nlocal+0+ix-1+(iz)*nx]
-xv[nlocal+0+ix+iz*nx]
-xv[nlocal+0+ix+1+iz*nx]
-xv[nlocal+0+ix-1+(iz+1)*nx]
-xv[nlocal+0+ix+(iz+1)*nx]
-xv[nlocal+0+ix+1+(iz+1)*nx]
;}
}
}//ipz < npz - 1
          }//ipz > 0 
        }//ipy < npy - 1 
      else
{
        if(ipz > 0)
{
            if(ipz < npz - 1)
{
iz=0;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
-xv[nlocal+0+nx+ix-1]
-xv[nlocal+0+nx+ix]
-xv[nlocal+0+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix+(ny-2)*nx-1]
-xv[nlocal+0+ix+(ny-2)*nx]
-xv[nlocal+0+ix+(ny-2)*nx+1]
-xv[nlocal+0+ix+(ny-1)*nx-1]
-xv[nlocal+0+ix+(ny-1)*nx]
-xv[nlocal+0+ix+(ny-1)*nx+1]
;}
iz=nz-1;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ix-1]
-xv[nlocal+0+nx*ny+ix]
-xv[nlocal+0+nx*ny+ix+1]
-xv[nlocal+0+nx*ny+nx+ix-1]
-xv[nlocal+0+nx*ny+nx+ix]
-xv[nlocal+0+nx*ny+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+nx*ny+ix+(ny-2)*nx]
-xv[nlocal+0+nx*ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+nx*ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+nx*ny+ix+(ny-1)*nx]
-xv[nlocal+0+nx*ny+ix+(ny-1)*nx+1]
;}
;
iz=0;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iy-1)*nx]
-xv[nlocal+0+(iy-1)*nx+1]
-xv[nlocal+0+(iy)*nx]
-xv[nlocal+0+(iy-1)*nx+1]
-xv[nlocal+0+(iy+1)*nx]
-xv[nlocal+0+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iy-1)*nx+nx-2]
-xv[nlocal+0+(iy-1)*nx+nx-1]
-xv[nlocal+0+(iy)*nx+nx-2]
-xv[nlocal+0+(iy)*nx+nx-1]
-xv[nlocal+0+(iy+1)*nx+nx-2]
-xv[nlocal+0+(iy+1)*nx+nx-1]
;}
iz=nz-1;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+(iy-1)*nx]
-xv[nlocal+0+nx*ny+(iy-1)*nx+1]
-xv[nlocal+0+nx*ny+(iy)*nx]
-xv[nlocal+0+nx*ny+(iy-1)*nx+1]
-xv[nlocal+0+nx*ny+(iy+1)*nx]
-xv[nlocal+0+nx*ny+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+nx*ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+nx*ny+(iy)*nx+nx-2]
-xv[nlocal+0+nx*ny+(iy)*nx+nx-1]
-xv[nlocal+0+nx*ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+nx*ny+(iy+1)*nx+nx-1]
;}
;
;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1+(iy-1)*nx]
-xv[nlocal+0+ix+(iy-1)*nx]
-xv[nlocal+0+ix+1+(iy-1)*nx]
-xv[nlocal+0+ix-1+(iy)*nx]
-xv[nlocal+0+ix+iy*nx]
-xv[nlocal+0+ix+1+iy*nx]
-xv[nlocal+0+ix-1+(iy+1)*nx]
-xv[nlocal+0+ix+(iy+1)*nx]
-xv[nlocal+0+ix+1+(iy+1)*nx]
;}
}
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ix-1+(iy-1)*nx]
-xv[nlocal+0+nx*ny+ix+(iy-1)*nx]
-xv[nlocal+0+nx*ny+ix+1+(iy-1)*nx]
-xv[nlocal+0+nx*ny+ix-1+(iy)*nx]
-xv[nlocal+0+nx*ny+ix+iy*nx]
-xv[nlocal+0+nx*ny+ix+1+iy*nx]
-xv[nlocal+0+nx*ny+ix-1+(iy+1)*nx]
-xv[nlocal+0+nx*ny+ix+(iy+1)*nx]
-xv[nlocal+0+nx*ny+ix+1+(iy+1)*nx]
;}
}
}//ipz < npz - 1
            else
{
iz=0;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
-xv[nlocal+0+nx+ix-1]
-xv[nlocal+0+nx+ix]
-xv[nlocal+0+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix+(ny-2)*nx-1]
-xv[nlocal+0+ix+(ny-2)*nx]
-xv[nlocal+0+ix+(ny-2)*nx+1]
-xv[nlocal+0+ix+(ny-1)*nx-1]
-xv[nlocal+0+ix+(ny-1)*nx]
-xv[nlocal+0+ix+(ny-1)*nx+1]
;}
;
iz=0;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iy-1)*nx]
-xv[nlocal+0+(iy-1)*nx+1]
-xv[nlocal+0+(iy)*nx]
-xv[nlocal+0+(iy-1)*nx+1]
-xv[nlocal+0+(iy+1)*nx]
-xv[nlocal+0+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iy-1)*nx+nx-2]
-xv[nlocal+0+(iy-1)*nx+nx-1]
-xv[nlocal+0+(iy)*nx+nx-2]
-xv[nlocal+0+(iy)*nx+nx-1]
-xv[nlocal+0+(iy+1)*nx+nx-2]
-xv[nlocal+0+(iy+1)*nx+nx-1]
;}
;
;
iz=0;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1+(iy-1)*nx]
-xv[nlocal+0+ix+(iy-1)*nx]
-xv[nlocal+0+ix+1+(iy-1)*nx]
-xv[nlocal+0+ix-1+(iy)*nx]
-xv[nlocal+0+ix+iy*nx]
-xv[nlocal+0+ix+1+iy*nx]
-xv[nlocal+0+ix-1+(iy+1)*nx]
-xv[nlocal+0+ix+(iy+1)*nx]
-xv[nlocal+0+ix+1+(iy+1)*nx]
;}
}
}//ipz < npz - 1
          }//ipz > 0 
        else
{
            if(ipz < npz - 1)
{
iz=nz-1;
iy=0;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
-xv[nlocal+0+nx+ix-1]
-xv[nlocal+0+nx+ix]
-xv[nlocal+0+nx+ix+1]
;}iy=ny-1;
for (ix=1; ix<nx-1;ix++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix+(ny-2)*nx-1]
-xv[nlocal+0+ix+(ny-2)*nx]
-xv[nlocal+0+ix+(ny-2)*nx+1]
-xv[nlocal+0+ix+(ny-1)*nx-1]
-xv[nlocal+0+ix+(ny-1)*nx]
-xv[nlocal+0+ix+(ny-1)*nx+1]
;}
;
iz=nz-1;
ix=0;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iy-1)*nx]
-xv[nlocal+0+(iy-1)*nx+1]
-xv[nlocal+0+(iy)*nx]
-xv[nlocal+0+(iy-1)*nx+1]
-xv[nlocal+0+(iy+1)*nx]
-xv[nlocal+0+(iy+1)*nx+1]
;}ix=nx-1;
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iy-1)*nx+nx-2]
-xv[nlocal+0+(iy-1)*nx+nx-1]
-xv[nlocal+0+(iy)*nx+nx-2]
-xv[nlocal+0+(iy)*nx+nx-1]
-xv[nlocal+0+(iy+1)*nx+nx-2]
-xv[nlocal+0+(iy+1)*nx+nx-1]
;}
;
;
iz=nz-1;
for (ix=1; ix<nx-1;ix++)
{
for (iy=1; iy<ny-1;iy++)
{
 yv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1+(iy-1)*nx]
-xv[nlocal+0+ix+(iy-1)*nx]
-xv[nlocal+0+ix+1+(iy-1)*nx]
-xv[nlocal+0+ix-1+(iy)*nx]
-xv[nlocal+0+ix+iy*nx]
-xv[nlocal+0+ix+1+iy*nx]
-xv[nlocal+0+ix-1+(iy+1)*nx]
-xv[nlocal+0+ix+(iy+1)*nx]
-xv[nlocal+0+ix+1+(iy+1)*nx]
;}
}
}//ipz < npz - 1
            else
{
;
;
;
}//ipz < npz - 1
          }//ipz > 0 
        }//ipy < npy - 1 
      }//ipy > 0 
    }//ipx < npx - 1 
 }//ipx > 0 
return 0;}
