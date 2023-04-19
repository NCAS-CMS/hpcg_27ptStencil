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
 @file ComputeSYMGS_ref.cpp
 HPCG routine
 */

#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#endif
#include "ComputeSYMGS_ref.hpp"
#include <cassert>

/*!
  Computes one step of symmetric Gauss-Seidel:
  Assumption about the structure of matrix A:
  - Each row 'i' of the matrix has nonzero diagonal value whose address is matrixDiagonal[i]
  - Entries in row 'i' are ordered such that:
       - lower triangular terms are stored before the diagonal element.
       - upper triangular terms are stored after the diagonal element.
       - No other assumptions are made about entry ordering.
  Symmetric Gauss-Seidel notes:
  - We use the input vector x as the RHS and start with an initial guess for y of all zeros.
  - We perform one forward sweep.  x should be initially zero on the first GS sweep, but we do not attempt to exploit this fact.
  - We then perform one back sweep.
  - For simplicity we include the diagonal contribution in the for-j loop, then correct the sum after
  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On entry, x should contain relevant values, on exit x contains the result of one symmetric GS sweep with r as the RHS.
  @warning Early versions of this kernel (Version 1.1 and earlier) had the r and x arguments in reverse order, and out of sync with other kernels.
  @return returns 0 upon success and non-zero otherwise
  @see ComputeSYMGS
*/
int ComputeSYMGS_ref( const SparseMatrix & A, const Vector & r, Vector & x) {

  assert(x.localLength==A.localNumberOfColumns); // Make sure x contain space for halo values

#ifndef HPCG_NO_MPI
  ExchangeHalo(A,x);
#endif

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
  int npx          = A.geom->npx;
  int npy          = A.geom->npy;
  int npz          = A.geom->npz;
  int ipx          = A.geom->ipx;
  int ipy          = A.geom->ipy;
  int ipz          = A.geom->ipz;
  double diagonal_element = A.matrixDiagonal[0][0];// big assumption?
//  const local_int_t nrow = A.localNumberOfRows;
//  double ** matrixDiagonal = A.matrixDiagonal;  // An array of pointers to the diagonal entries A.matrixValues
  const double * const rv = r.values;
  double * const xv = x.values;

// forward sweep
for( iz=0; iz< nz; iz++){
    for( iy=0; iy< ny; iy++){
        for (ix=0; ix < nx; ix++){

if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+1)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+1)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+1)*nx+(iz+1)*ny*nx]
;}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+1)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+1)*nx+(iz+0)*ny*nx]
;}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+1)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+-1)*nx+(iz+1)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+1)*ny*nx]
;}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+0)+(iy+-1)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+-1)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
;}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+-1)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+1)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+1)*ny*nx]
;}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+-1)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+1)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
;}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+-1)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+-1)*nx+(iz+1)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
;}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+-1)+(iy+-1)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
;}

//edges 
else if (ix == 0 && iy == 0 &&iz>0 && iz<nz-1){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+1)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+1)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+1)+(iy+1)*nx+(iz+1)*ny*nx]
;}
else if (ix == 0 && iy == ny-1 &&iz>0 && iz<nz-1){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+0)+(iy+-1)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+-1)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+1)+(iy+-1)*nx+(iz+1)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+1)*ny*nx]
;}
else if (ix == nx-1 && iy == 0 && iz>0 && iz<nz-1){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+-1)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+1)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+-1)+(iy+1)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+1)*ny*nx]
;}
else if (ix == nx-1 && iy == ny-1 &&iz>0 && iz<nz-1){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+-1)+(iy+-1)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+-1)*nx+(iz+1)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
;}
else if (ix == 0 && iz == 0 &&iy>0 && iy<ny-1){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+1)*ny*nx]
+xv[(ix+1)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+-1)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+1)*ny*nx]
+xv[(ix+1)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+1)*nx+(iz+1)*ny*nx]
;}
else if (ix == 0 && iz == nz-1 &&iy>0 && iy<ny-1){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+0)+(iy+-1)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+-1)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+1)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+1)*nx+(iz+0)*ny*nx]
;}
else if (ix == nx-1 && iz == 0 && iy>0 && iy<ny-1){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+-1)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+-1)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+1)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+-1)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+1)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+1)*ny*nx]
;}
else if (ix == nx-1 && iz == nz-1 &&iy>0 && iy<ny-1){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+-1)+(iy+-1)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+1)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
;}
else if (iy == 0 && iz == 0 &&ix>0 && ix<nx-1){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+-1)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+1)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+1)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+1)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+1)*nx+(iz+1)*ny*nx]
;}
else if (iy == 0 && iz == nz-1 &&ix>0 && ix<nx-1){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+-1)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+1)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+1)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+1)*nx+(iz+0)*ny*nx]
;}
else if (iy == ny-1 && iz == 0 && ix>0 && ix<nx-1){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+-1)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+-1)*nx+(iz+1)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+1)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+-1)*nx+(iz+1)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+1)*ny*nx]
;}
else if (iy == ny-1 && iz == nz-1 &&ix>0 && ix<nx-1){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+-1)+(iy+-1)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+-1)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
;}

//sides 
else if (iz == 0 && ix>0 && ix<nx-1 && iy>0 && iy<ny-1){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+-1)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+-1)*nx+(iz+1)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+-1)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+1)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+1)*ny*nx]
+xv[(ix+1)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+-1)*nx+(iz+1)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+1)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+1)*nx+(iz+1)*ny*nx]
;}
else if (iz == nz-1 && ix>0 && ix<nx-1 && iy>0 && iy<ny-1){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+-1)+(iy+-1)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+1)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+-1)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+1)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+1)*nx+(iz+0)*ny*nx]
;}
else if (iy == 0 && ix>0 && ix<nx-1 && iz>0 && iz<nz-1){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+-1)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+1)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+-1)+(iy+1)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+1)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+1)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+1)+(iy+1)*nx+(iz+1)*ny*nx]
;}
else if (iy == ny-1 && ix>0 && ix<nx-1 && iz>0 && iz<nz-1){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+-1)+(iy+-1)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+-1)*nx+(iz+1)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+1)+(iy+-1)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+-1)*nx+(iz+1)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+1)*ny*nx]
;}
else if (ix == 0 && iy>0 && iy<ny-1 && iz>0 && iz<nz-1){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+0)+(iy+-1)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+-1)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+1)*ny*nx]
+xv[(ix+1)+(iy+-1)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+1)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+1)*ny*nx]
+xv[(ix+1)+(iy+1)*nx+(iz+1)*ny*nx]
;}
else if (ix == nx-1 && iy>0 && iy<ny-1 && iz>0 && iz<nz-1){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+-1)+(iy+-1)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+-1)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+1)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+-1)+(iy+1)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+1)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+1)*ny*nx]
;}

//bulk
else{
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx]
+ xv[ix+-1 + (iy+-1)*nx + (iz+-1)*nx*ny]
+ xv[ix+-1 + (iy+-1)*nx + (iz+0)*nx*ny]
+ xv[ix+-1 + (iy+-1)*nx + (iz+1)*nx*ny]
+ xv[ix+-1 + (iy+0)*nx + (iz+-1)*nx*ny]
+ xv[ix+-1 + (iy+0)*nx + (iz+0)*nx*ny]
+ xv[ix+-1 + (iy+0)*nx + (iz+1)*nx*ny]
+ xv[ix+-1 + (iy+1)*nx + (iz+-1)*nx*ny]
+ xv[ix+-1 + (iy+1)*nx + (iz+0)*nx*ny]
+ xv[ix+-1 + (iy+1)*nx + (iz+1)*nx*ny]
+ xv[ix+0 + (iy+-1)*nx + (iz+-1)*nx*ny]
+ xv[ix+0 + (iy+-1)*nx + (iz+0)*nx*ny]
+ xv[ix+0 + (iy+-1)*nx + (iz+1)*nx*ny]
+ xv[ix+0 + (iy+0)*nx + (iz+-1)*nx*ny]
+ xv[ix+0 + (iy+0)*nx + (iz+1)*nx*ny]
+ xv[ix+0 + (iy+1)*nx + (iz+-1)*nx*ny]
+ xv[ix+0 + (iy+1)*nx + (iz+0)*nx*ny]
+ xv[ix+0 + (iy+1)*nx + (iz+1)*nx*ny]
+ xv[ix+1 + (iy+-1)*nx + (iz+-1)*nx*ny]
+ xv[ix+1 + (iy+-1)*nx + (iz+0)*nx*ny]
+ xv[ix+1 + (iy+-1)*nx + (iz+1)*nx*ny]
+ xv[ix+1 + (iy+0)*nx + (iz+-1)*nx*ny]
+ xv[ix+1 + (iy+0)*nx + (iz+0)*nx*ny]
+ xv[ix+1 + (iy+0)*nx + (iz+1)*nx*ny]
+ xv[ix+1 + (iy+1)*nx + (iz+-1)*nx*ny]
+ xv[ix+1 + (iy+1)*nx + (iz+0)*nx*ny]
+ xv[ix+1 + (iy+1)*nx + (iz+1)*nx*ny]
;}

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
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+1]
-xv[nlocal+0+1+nx+1]
-xv[nlocal+0+1+nx+1+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny+1]
-xv[nlocal+0+1]
-xv[nlocal+0+1+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx+1]
-xv[nlocal+0+1+nx+1+ny]
-xv[nlocal+0+1+nx+1+ny+1]
-xv[nlocal+0+1+nx+1+ny+nx]
-xv[nlocal+0+1+nx+1+ny+nx+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*(nz-2)]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*(nz-2)+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*(nz-1)]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*(nz-1)+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*(nz-2)]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*(nz-2)+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*(nz-1)]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*(nz-1)+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+1+ny-2]
-xv[nlocal+0+1+nx+1+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+2*ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+2*ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+1]
-xv[nlocal+0+1+nx+1+ny+nx*(ny-2)]
-xv[nlocal+0+1+nx+1+ny+nx*(ny-2)+1]
-xv[nlocal+0+1+nx+1+ny+nx*(ny-1)]
-xv[nlocal+0+1+nx+1+ny+nx*(ny-1)+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*(nz-1)-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*(nz-1)-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*(ny-2)]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*(ny-2)+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*(ny-1)]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*(ny-1)+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*(nz-2)]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*(nz-2)+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*(nz-1)]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*(nz-1)+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+ny+1+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx-2]
-xv[nlocal+0+1+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+2*nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+2*nx-1]
-xv[nlocal+0+1+nx+1+ny+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx-1]
-xv[nlocal+0+1+nx+1+ny+2*nx-2]
-xv[nlocal+0+1+nx+1+ny+2*nx-1]
-xv[nlocal+0+1+nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*(nz-1)-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*(nz-1)-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+2*nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+2*nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*(nz-2)]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*(nz-2)+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*(nz-1)]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*(nz-1)+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+1+ny+nx*(ny-1)-2]
-xv[nlocal+0+1+nx+1+ny+nx*(ny-1)-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+2*nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+2*nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+2*ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+2*ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*(ny-1)-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*(ny-1)-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*(nz-1)-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*(nz-1)-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+ny+1+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+ny+1+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*(nz-1)-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*(nz-1)-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+ny+1+nx]
;
}
else if(iy==0&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+ix-1]
-xv[nlocal+0+1+ix]
-xv[nlocal+0+1+ix+1]
;}
else if(iy==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx+ix-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx+ix]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx+ix+1]
;}else if(iy==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix+(nz-2)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix+(nz-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix+(nz-1)*nx+1]
;}
else if(iy==0&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+ix-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+ix]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+ix+1]
;}
else if(iz==0&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+ix-1]
-xv[nlocal+0+1+nx+1+ny+ix]
-xv[nlocal+0+1+nx+1+ny+ix+1]
-xv[nlocal+0+1+nx+1+ny+nx+ix-1]
-xv[nlocal+0+1+nx+1+ny+nx+ix]
-xv[nlocal+0+1+nx+1+ny+nx+ix+1]
;}else if(iz==0&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+1+nx+1+ny+ix+(ny-2)*nx]
-xv[nlocal+0+1+nx+1+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+1+nx+1+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+1+nx+1+ny+ix+(ny-1)*nx]
-xv[nlocal+0+1+nx+1+ny+ix+(ny-1)*nx+1]
;}
else if(iz==nz-1&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx+ix-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx+ix]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx+ix+1]
;}else if(iz==nz-1&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix+(ny-2)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix+(ny-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix+(ny-1)*nx+1]
;}
else if(iy==ny-1&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+ix-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+ix]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+ix+1]
;}
else if(iy==ny-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx+ix-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx+ix]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx+ix+1]
;}else if(iy==ny-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(nz-2)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(nz-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(nz-1)*nx+1]
;}
else if(iy==ny-1&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+ny+1+ix-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+ny+1+ix]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+ny+1+ix+1]
;}
else if(ix==0&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+iy-1]
-xv[nlocal+0+1+nx+1+iy]
-xv[nlocal+0+1+nx+1+iy+1]
;}
else if(ix==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny+iy-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny+iy]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny+iy+1]
;}else if(ix==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy+(nz-2)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy+(nz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy+(nz-1)*ny+1]
;}
else if(ix==0&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+iy-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+iy]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+iy+1]
;}
else if(iz==0 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+(iy-1)*nx]
-xv[nlocal+0+1+nx+1+ny+(iy-1)*nx+1]
-xv[nlocal+0+1+nx+1+ny+(iy)*nx]
-xv[nlocal+0+1+nx+1+ny+(iy)*nx+1]
-xv[nlocal+0+1+nx+1+ny+(iy+1)*nx]
-xv[nlocal+0+1+nx+1+ny+(iy+1)*nx+1]
;}else if(iz==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+1+nx+1+ny+(iy)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+(iy)*nx+nx-1]
-xv[nlocal+0+1+nx+1+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+(iy+1)*nx+nx-1]
;}
else if(iz==nz-1 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+(iy-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+(iy-1)*nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+(iy)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+(iy)*nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+(iy+1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+(iy+1)*nx+1]
;}else if(iz==nz-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+(iy)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+(iy)*nx+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+(iy+1)*nx+nx-1]
;}
else if(ix==nx-1&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+iy-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+iy]
-xv[nlocal+0+1+nx+1+ny+nx*ny+iy+1]
;}
else if(ix==nx-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny+iy-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny+iy]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny+iy+1]
;}else if(ix==nx-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy+(nz-2)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy+(nz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy+(nz-1)*ny+1]
;}
else if(ix==nx-1&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+iy-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+iy]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+iy+1]
;}
else if(ix==0&&iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+iz-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+iz]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+iz+1]
;}
else if(ix==0&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+(iz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+(iz-1)*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+(iz)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+(iz)*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+(iz+1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+(iz+1)*ny+1]
;}else if(ix==0&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+(iz)*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+(iz)*ny+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+(iz+1)*ny+ny-1]
;}
else if(ix==0&&iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+iz-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+iz]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+iz+1]
;}
else if(iy==0&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+(iz-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+(iz-1)*nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+(iz)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+(iz)*nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+(iz+1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+(iz+1)*nx+1]
;}else if(iy==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+(iz)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+(iz)*nx+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+(iz+1)*nx+nx-1]
;}
else if(iy==ny-1&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz-1)*nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz)*nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz+1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz+1)*nx+1]
;}else if(iy==ny-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz)*nx+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz+1)*nx+nx-1]
;}
else if(ix==nx-1&&iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+iz-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+iz]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+iz+1]
;}
else if(ix==nx-1&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+(iz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+(iz-1)*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+(iz)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+(iz)*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+(iz+1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+(iz+1)*ny+1]
;}else if(ix==nx-1&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+(iz)*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+(iz)*ny+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+(iz+1)*ny+ny-1]
;}
else if(ix==nx-1&&iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+iz-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+iz]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+iz+1]
;}
else if (ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
            else
{
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+1]
-xv[nlocal+0+1+nx+1]
-xv[nlocal+0+1+nx+1+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny+1]
-xv[nlocal+0+1]
-xv[nlocal+0+1+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx+1]
-xv[nlocal+0+1+nx+1+ny]
-xv[nlocal+0+1+nx+1+ny+1]
-xv[nlocal+0+1+nx+1+ny+nx]
-xv[nlocal+0+1+nx+1+ny+nx+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*(nz-2)]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*(nz-2)+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*(nz-1)]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*(nz-1)+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*(nz-2)]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*(nz-2)+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*(nz-1)]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*(nz-1)+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+1+ny-2]
-xv[nlocal+0+1+nx+1+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+2*ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+2*ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+1]
-xv[nlocal+0+1+nx+1+ny+nx*(ny-2)]
-xv[nlocal+0+1+nx+1+ny+nx*(ny-2)+1]
-xv[nlocal+0+1+nx+1+ny+nx*(ny-1)]
-xv[nlocal+0+1+nx+1+ny+nx*(ny-1)+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*(nz-1)-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*(nz-1)-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*(nz-2)]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*(nz-2)+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*(nz-1)]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*(nz-1)+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx-2]
-xv[nlocal+0+1+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+2*nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+2*nx-1]
-xv[nlocal+0+1+nx+1+ny+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx-1]
-xv[nlocal+0+1+nx+1+ny+2*nx-2]
-xv[nlocal+0+1+nx+1+ny+2*nx-1]
-xv[nlocal+0+1+nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*(nz-1)-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*(nz-1)-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*(nz-2)]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*(nz-2)+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*(nz-1)]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*(nz-1)+1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+1+ny+nx*(ny-1)-2]
-xv[nlocal+0+1+nx+1+ny+nx*(ny-1)-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+2*nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+2*nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+2*ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+2*ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*(nz-1)-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*(nz-1)-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*(nz-1)-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*(nz-1)-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz-1]
;
}
else if(iy==0&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+ix-1]
-xv[nlocal+0+1+ix]
-xv[nlocal+0+1+ix+1]
;}
else if(iy==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx+ix-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx+ix]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx+ix+1]
;}else if(iy==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix+(nz-2)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix+(nz-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+ix+(nz-1)*nx+1]
;}
else if(iz==0&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+ix-1]
-xv[nlocal+0+1+nx+1+ny+ix]
-xv[nlocal+0+1+nx+1+ny+ix+1]
-xv[nlocal+0+1+nx+1+ny+nx+ix-1]
-xv[nlocal+0+1+nx+1+ny+nx+ix]
-xv[nlocal+0+1+nx+1+ny+nx+ix+1]
;}else if(iz==0&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+1+nx+1+ny+ix+(ny-2)*nx]
-xv[nlocal+0+1+nx+1+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+1+nx+1+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+1+nx+1+ny+ix+(ny-1)*nx]
-xv[nlocal+0+1+nx+1+ny+ix+(ny-1)*nx+1]
;}
else if(iy==ny-1&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+ix-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+ix]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+ix+1]
;}
else if(iy==ny-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx+ix-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx+ix]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx+ix+1]
;}else if(iy==ny-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(nz-2)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(nz-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(nz-1)*nx+1]
;}
else if(ix==0&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+iy-1]
-xv[nlocal+0+1+nx+1+iy]
-xv[nlocal+0+1+nx+1+iy+1]
;}
else if(ix==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny+iy-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny+iy]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny+iy+1]
;}else if(ix==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy+(nz-2)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy+(nz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+iy+(nz-1)*ny+1]
;}
else if(iz==0 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+(iy-1)*nx]
-xv[nlocal+0+1+nx+1+ny+(iy-1)*nx+1]
-xv[nlocal+0+1+nx+1+ny+(iy)*nx]
-xv[nlocal+0+1+nx+1+ny+(iy)*nx+1]
-xv[nlocal+0+1+nx+1+ny+(iy+1)*nx]
-xv[nlocal+0+1+nx+1+ny+(iy+1)*nx+1]
;}else if(iz==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+1+nx+1+ny+(iy)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+(iy)*nx+nx-1]
-xv[nlocal+0+1+nx+1+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+(iy+1)*nx+nx-1]
;}
else if(ix==nx-1&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+iy-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+iy]
-xv[nlocal+0+1+nx+1+ny+nx*ny+iy+1]
;}
else if(ix==nx-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny+iy-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny+iy]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny+iy+1]
;}else if(ix==nx-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy+(nz-2)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy+(nz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+iy+(nz-1)*ny+1]
;}
else if(ix==0&&iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+iz-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+iz]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+iz+1]
;}
else if(ix==0&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+(iz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+(iz-1)*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+(iz)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+(iz)*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+(iz+1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+(iz+1)*ny+1]
;}else if(ix==0&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+(iz)*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+(iz)*ny+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+(iz+1)*ny+ny-1]
;}
else if(ix==0&&iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+iz-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+iz]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+iz+1]
;}
else if(iy==0&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+(iz-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+(iz-1)*nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+(iz)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+(iz)*nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+(iz+1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+(iz+1)*nx+1]
;}else if(iy==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+(iz)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+(iz)*nx+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+(iz+1)*nx+nx-1]
;}
else if(iy==ny-1&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz-1)*nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz)*nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz+1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz+1)*nx+1]
;}else if(iy==ny-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz)*nx+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz+1)*nx+nx-1]
;}
else if(ix==nx-1&&iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+iz-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+iz]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+iz+1]
;}
else if(ix==nx-1&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+(iz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+(iz-1)*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+(iz)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+(iz)*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+(iz+1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+(iz+1)*ny+1]
;}else if(ix==nx-1&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+(iz)*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+(iz)*ny+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+(iz+1)*ny+ny-1]
;}
else if(ix==nx-1&&iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+iz-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+iz]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1+nx+1+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+iz+1]
;}
else if (ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
          }//ipz > 0 
        else
{
            if(ipz < npz - 1)
{
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+nz+nx*nz+nz]
-xv[nlocal+0+nz+nx*nz+nz+1]
-xv[nlocal+0+nz+nx*nz+nz+ny]
-xv[nlocal+0+nz+nx*nz+nz+ny+1]
-xv[nlocal+0+nz]
-xv[nlocal+0+nz+1]
-xv[nlocal+0+nz+nx]
-xv[nlocal+0+nz+nx+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz-2]
-xv[nlocal+0+nz-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz]
-xv[nlocal+0+nz+nx*nz+nz+ny*(nz-2)]
-xv[nlocal+0+nz+nx*nz+nz+ny*(nz-2)+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*(nz-1)]
-xv[nlocal+0+nz+nx*nz+nz+ny*(nz-1)+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+1]
-xv[nlocal+0+nz+nx*(nz-2)]
-xv[nlocal+0+nz+nx*(nz-2)+1]
-xv[nlocal+0+nz+nx*(nz-1)]
-xv[nlocal+0+nz+nx*(nz-1)+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx*nz+nz+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+ny-1]
-xv[nlocal+0+nz+nx*nz+nz+2*ny-2]
-xv[nlocal+0+nz+nx*nz+nz+2*ny-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx*nz+nz+ny*(nz-1)-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*(nz-1)-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*(ny-2)]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*(ny-2)+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*(ny-1)]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*(ny-1)+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*(nz-2)]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*(nz-2)+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*(nz-1)]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*(nz-1)+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+ny+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+ny+1+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx-2]
-xv[nlocal+0+nz+nx-1]
-xv[nlocal+0+nz+2*nx-2]
-xv[nlocal+0+nz+2*nx-1]
-xv[nlocal+0+nz+nx*nz]
-xv[nlocal+0+nz+nx*nz+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx*(nz-1)-2]
-xv[nlocal+0+nz+nx*(nz-1)-1]
-xv[nlocal+0+nz+nx*nz-2]
-xv[nlocal+0+nz+nx*nz-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+2*nx-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+2*nx-1]
-xv[nlocal+0+nz+nx*nz+nz-2]
-xv[nlocal+0+nz+nx*nz+nz-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*(nz-2)]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*(nz-2)+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*(nz-1)]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*(nz-1)+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+2*nx-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+2*nx-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+2*ny-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+2*ny-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*(ny-1)-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*(ny-1)-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*(nz-1)-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*(nz-1)-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+ny+1+nx-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+ny+1+nx-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*(nz-1)-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*(nz-1)-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+ny-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+ny+1+nx]
;
}
else if(iy==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+ix-1]
-xv[nlocal+0+nz+ix]
-xv[nlocal+0+nz+ix+1]
-xv[nlocal+0+nz+nx+ix-1]
-xv[nlocal+0+nz+nx+ix]
-xv[nlocal+0+nz+nx+ix+1]
;}else if(iy==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+nz+ix+(nz-2)*nx]
-xv[nlocal+0+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+nz+ix+(nz-1)*nx]
-xv[nlocal+0+nz+ix+(nz-1)*nx+1]
;}
else if(iy==0&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+ix-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+ix]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+ix+1]
;}
else if(iz==nz-1&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx+ix-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx+ix]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx+ix+1]
;}else if(iz==nz-1&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix+(ny-2)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix+(ny-1)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+ix+(ny-1)*nx+1]
;}
else if(iy==ny-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx+ix-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx+ix]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx+ix+1]
;}else if(iy==ny-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(nz-2)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(nz-1)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(nz-1)*nx+1]
;}
else if(iy==ny-1&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+ny+1+ix-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+ny+1+ix]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+ny+1+ix+1]
;}
else if(ix==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+iy-1]
-xv[nlocal+0+nz+nx*nz+nz+iy]
-xv[nlocal+0+nz+nx*nz+nz+iy+1]
-xv[nlocal+0+nz+nx*nz+nz+ny+iy-1]
-xv[nlocal+0+nz+nx*nz+nz+ny+iy]
-xv[nlocal+0+nz+nx*nz+nz+ny+iy+1]
;}else if(ix==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+nz+nx*nz+nz+iy+(nz-2)*ny]
-xv[nlocal+0+nz+nx*nz+nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+nz+nx*nz+nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+nz+nx*nz+nz+iy+(nz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+iy+(nz-1)*ny+1]
;}
else if(ix==0&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+iy-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+iy]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+iy+1]
;}
else if(iz==nz-1 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+(iy-1)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+(iy-1)*nx+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+(iy)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+(iy)*nx+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+(iy+1)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+(iy+1)*nx+1]
;}else if(iz==nz-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+(iy)*nx+nx-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+(iy)*nx+nx-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+(iy+1)*nx+nx-1]
;}
else if(ix==nx-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny+iy-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny+iy]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny+iy+1]
;}else if(ix==nx-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(nz-2)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(nz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(nz-1)*ny+1]
;}
else if(ix==nx-1&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+iy-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+iy]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz+1+nx+1+ny+nx*ny+iy+1]
;}
else if(ix==0&&iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iz-1]
-xv[nlocal+0+iz]
-xv[nlocal+0+iz+1]
;}
else if(ix==0&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+(iz-1)*ny+1]
-xv[nlocal+0+nz+nx*nz+nz+(iz)*ny]
-xv[nlocal+0+nz+nx*nz+nz+(iz)*ny+1]
-xv[nlocal+0+nz+nx*nz+nz+(iz+1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+(iz+1)*ny+1]
;}else if(ix==0&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+nz+nx*nz+nz+(iz)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+(iz)*ny+ny-1]
-xv[nlocal+0+nz+nx*nz+nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+(iz+1)*ny+ny-1]
;}
else if(ix==0&&iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+iz-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+iz]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+iz+1]
;}
else if(iy==0&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+(iz-1)*nx]
-xv[nlocal+0+nz+(iz-1)*nx+1]
-xv[nlocal+0+nz+(iz)*nx]
-xv[nlocal+0+nz+(iz)*nx+1]
-xv[nlocal+0+nz+(iz+1)*nx]
-xv[nlocal+0+nz+(iz+1)*nx+1]
;}else if(iy==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+nz+(iz)*nx+nx-2]
-xv[nlocal+0+nz+(iz)*nx+nx-1]
-xv[nlocal+0+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+nz+(iz+1)*nx+nx-1]
;}
else if(iy==ny-1&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz-1)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz-1)*nx+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz)*nx+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz+1)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz+1)*nx+1]
;}else if(iy==ny-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz)*nx+nx-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz)*nx+nx-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz+1)*nx+nx-1]
;}
else if(ix==nx-1&&iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+iz-1]
-xv[nlocal+0+nz+nx*nz+iz]
-xv[nlocal+0+nz+nx*nz+iz+1]
;}
else if(ix==nx-1&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz-1)*ny+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz)*ny+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz+1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz+1)*ny+1]
;}else if(ix==nx-1&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz)*ny+ny-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz+1)*ny+ny-1]
;}
else if(ix==nx-1&&iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+iz-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+iz]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+iz+1]
;}
else if (ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
            else
{
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+nz+nx*nz+nz]
-xv[nlocal+0+nz+nx*nz+nz+1]
-xv[nlocal+0+nz+nx*nz+nz+ny]
-xv[nlocal+0+nz+nx*nz+nz+ny+1]
-xv[nlocal+0+nz]
-xv[nlocal+0+nz+1]
-xv[nlocal+0+nz+nx]
-xv[nlocal+0+nz+nx+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz-2]
-xv[nlocal+0+nz-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*(nz-2)]
-xv[nlocal+0+nz+nx*nz+nz+ny*(nz-2)+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*(nz-1)]
-xv[nlocal+0+nz+nx*nz+nz+ny*(nz-1)+1]
-xv[nlocal+0+nz+nx*(nz-2)]
-xv[nlocal+0+nz+nx*(nz-2)+1]
-xv[nlocal+0+nz+nx*(nz-1)]
-xv[nlocal+0+nz+nx*(nz-1)+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx*nz+nz+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+ny-1]
-xv[nlocal+0+nz+nx*nz+nz+2*ny-2]
-xv[nlocal+0+nz+nx*nz+nz+2*ny-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx*nz+nz+ny*(nz-1)-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*(nz-1)-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*(nz-2)]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*(nz-2)+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*(nz-1)]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*(nz-1)+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx-2]
-xv[nlocal+0+nz+nx-1]
-xv[nlocal+0+nz+2*nx-2]
-xv[nlocal+0+nz+2*nx-1]
-xv[nlocal+0+nz+nx*nz]
-xv[nlocal+0+nz+nx*nz+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx*(nz-1)-2]
-xv[nlocal+0+nz+nx*(nz-1)-1]
-xv[nlocal+0+nz+nx*nz-2]
-xv[nlocal+0+nz+nx*nz-1]
-xv[nlocal+0+nz+nx*nz+nz-2]
-xv[nlocal+0+nz+nx*nz+nz-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*(nz-2)]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*(nz-2)+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*(nz-1)]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*(nz-1)+1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+2*nx-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+2*nx-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+2*ny-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+2*ny-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*(nz-1)-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*(nz-1)-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*(nz-1)-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*(nz-1)-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+nz-1]
;
}
else if(iy==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+ix-1]
-xv[nlocal+0+nz+ix]
-xv[nlocal+0+nz+ix+1]
-xv[nlocal+0+nz+nx+ix-1]
-xv[nlocal+0+nz+nx+ix]
-xv[nlocal+0+nz+nx+ix+1]
;}else if(iy==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+nz+ix+(nz-2)*nx]
-xv[nlocal+0+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+nz+ix+(nz-1)*nx]
-xv[nlocal+0+nz+ix+(nz-1)*nx+1]
;}
else if(iy==ny-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx+ix-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx+ix]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx+ix+1]
;}else if(iy==ny-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(nz-2)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(nz-1)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+ix+(nz-1)*nx+1]
;}
else if(ix==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+iy-1]
-xv[nlocal+0+nz+nx*nz+nz+iy]
-xv[nlocal+0+nz+nx*nz+nz+iy+1]
-xv[nlocal+0+nz+nx*nz+nz+ny+iy-1]
-xv[nlocal+0+nz+nx*nz+nz+ny+iy]
-xv[nlocal+0+nz+nx*nz+nz+ny+iy+1]
;}else if(ix==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+nz+nx*nz+nz+iy+(nz-2)*ny]
-xv[nlocal+0+nz+nx*nz+nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+nz+nx*nz+nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+nz+nx*nz+nz+iy+(nz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+iy+(nz-1)*ny+1]
;}
else if(ix==nx-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny+iy-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny+iy]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny+iy+1]
;}else if(ix==nx-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(nz-2)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(nz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(nz-1)*ny+1]
;}
else if(ix==0&&iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iz-1]
-xv[nlocal+0+iz]
-xv[nlocal+0+iz+1]
;}
else if(ix==0&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+(iz-1)*ny+1]
-xv[nlocal+0+nz+nx*nz+nz+(iz)*ny]
-xv[nlocal+0+nz+nx*nz+nz+(iz)*ny+1]
-xv[nlocal+0+nz+nx*nz+nz+(iz+1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+(iz+1)*ny+1]
;}else if(ix==0&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+nz+nx*nz+nz+(iz)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+(iz)*ny+ny-1]
-xv[nlocal+0+nz+nx*nz+nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+(iz+1)*ny+ny-1]
;}
else if(ix==0&&iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+iz-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+iz]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+iz+1]
;}
else if(iy==0&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+(iz-1)*nx]
-xv[nlocal+0+nz+(iz-1)*nx+1]
-xv[nlocal+0+nz+(iz)*nx]
-xv[nlocal+0+nz+(iz)*nx+1]
-xv[nlocal+0+nz+(iz+1)*nx]
-xv[nlocal+0+nz+(iz+1)*nx+1]
;}else if(iy==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+nz+(iz)*nx+nx-2]
-xv[nlocal+0+nz+(iz)*nx+nx-1]
-xv[nlocal+0+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+nz+(iz+1)*nx+nx-1]
;}
else if(iy==ny-1&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz-1)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz-1)*nx+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz)*nx+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz+1)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz+1)*nx+1]
;}else if(iy==ny-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz)*nx+nx-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz)*nx+nx-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+(iz+1)*nx+nx-1]
;}
else if(ix==nx-1&&iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+iz-1]
-xv[nlocal+0+nz+nx*nz+iz]
-xv[nlocal+0+nz+nx*nz+iz+1]
;}
else if(ix==nx-1&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz-1)*ny+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz)*ny+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz+1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz+1)*ny+1]
;}else if(ix==nx-1&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz)*ny+ny-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz+1)*ny+ny-1]
;}
else if(ix==nx-1&&iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+iz-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+iz]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+nz+nx*nz+iz+1]
;}
else if (ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
          }//ipz > 0 
        }//ipy < npy - 1 
      else
{
        if(ipz > 0)
{
            if(ipz < npz - 1)
{
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1]
-xv[nlocal+0+1+nx+1]
-xv[nlocal+0+1+nx+1+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny+1]
-xv[nlocal+0+1]
-xv[nlocal+0+1+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx+1]
-xv[nlocal+0+1+nx+1+ny]
-xv[nlocal+0+1+nx+1+ny+1]
-xv[nlocal+0+1+nx+1+ny+nx]
-xv[nlocal+0+1+nx+1+ny+nx+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*(nz-2)]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*(nz-2)+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*(nz-1)]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*(nz-1)+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*(nz-2)]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*(nz-2)+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*(nz-1)]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*(nz-1)+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+1+ny-2]
-xv[nlocal+0+1+nx+1+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+2*ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+2*ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*(ny-2)]
-xv[nlocal+0+1+nx+1+ny+nx*(ny-2)+1]
-xv[nlocal+0+1+nx+1+ny+nx*(ny-1)]
-xv[nlocal+0+1+nx+1+ny+nx*(ny-1)+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*(nz-1)-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*(nz-1)-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx*(ny-2)]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx*(ny-2)+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx*(ny-1)]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx*(ny-1)+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx-2]
-xv[nlocal+0+1+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+2*nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+2*nx-1]
-xv[nlocal+0+1+nx+1+ny+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx-1]
-xv[nlocal+0+1+nx+1+ny+2*nx-2]
-xv[nlocal+0+1+nx+1+ny+2*nx-1]
-xv[nlocal+0+1+nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*(nz-1)-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*(nz-1)-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+2*nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+2*nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*(nz-2)]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*(nz-2)+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*(nz-1)]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*(nz-1)+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx*ny+1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+1+ny+nx*(ny-1)-2]
-xv[nlocal+0+1+nx+1+ny+nx*(ny-1)-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+2*ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+2*ny-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx*(ny-1)-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx*(ny-1)-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx*ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx*ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*(nz-1)-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*(nz-1)-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx*ny+ny-1]
;
}
else if(iy==0&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+ix-1]
-xv[nlocal+0+1+ix]
-xv[nlocal+0+1+ix+1]
;}
else if(iy==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx+ix-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx+ix]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx+ix+1]
;}else if(iy==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix+(nz-2)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix+(nz-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix+(nz-1)*nx+1]
;}
else if(iy==0&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+ix-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+ix]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+ix+1]
;}
else if(iz==0&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+ix-1]
-xv[nlocal+0+1+nx+1+ny+ix]
-xv[nlocal+0+1+nx+1+ny+ix+1]
-xv[nlocal+0+1+nx+1+ny+nx+ix-1]
-xv[nlocal+0+1+nx+1+ny+nx+ix]
-xv[nlocal+0+1+nx+1+ny+nx+ix+1]
;}else if(iz==0&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+1+nx+1+ny+ix+(ny-2)*nx]
-xv[nlocal+0+1+nx+1+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+1+nx+1+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+1+nx+1+ny+ix+(ny-1)*nx]
-xv[nlocal+0+1+nx+1+ny+ix+(ny-1)*nx+1]
;}
else if(iz==nz-1&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx+ix-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx+ix]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx+ix+1]
;}else if(iz==nz-1&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix+(ny-2)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix+(ny-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix+(ny-1)*nx+1]
;}
else if(ix==0&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+iy-1]
-xv[nlocal+0+1+nx+1+iy]
-xv[nlocal+0+1+nx+1+iy+1]
;}
else if(ix==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny+iy-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny+iy]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny+iy+1]
;}else if(ix==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy+(nz-2)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy+(nz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy+(nz-1)*ny+1]
;}
else if(ix==0&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+iy-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+iy]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+iy+1]
;}
else if(iz==0 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+(iy-1)*nx]
-xv[nlocal+0+1+nx+1+ny+(iy-1)*nx+1]
-xv[nlocal+0+1+nx+1+ny+(iy)*nx]
-xv[nlocal+0+1+nx+1+ny+(iy)*nx+1]
-xv[nlocal+0+1+nx+1+ny+(iy+1)*nx]
-xv[nlocal+0+1+nx+1+ny+(iy+1)*nx+1]
;}else if(iz==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+1+nx+1+ny+(iy)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+(iy)*nx+nx-1]
-xv[nlocal+0+1+nx+1+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+(iy+1)*nx+nx-1]
;}
else if(iz==nz-1 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+(iy-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+(iy-1)*nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+(iy)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+(iy)*nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+(iy+1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+(iy+1)*nx+1]
;}else if(iz==nz-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+(iy)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+(iy)*nx+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+(iy+1)*nx+nx-1]
;}
else if(ix==nx-1&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+iy-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+iy]
-xv[nlocal+0+1+nx+1+ny+nx*ny+iy+1]
;}
else if(ix==nx-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny+iy-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny+iy]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny+iy+1]
;}else if(ix==nx-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy+(nz-2)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy+(nz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy+(nz-1)*ny+1]
;}
else if(ix==nx-1&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx*ny+iy-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx*ny+iy]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx*ny+iy+1]
;}
else if(ix==0&&iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+iz-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+iz]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+iz+1]
;}
else if(ix==0&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+(iz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+(iz-1)*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+(iz)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+(iz)*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+(iz+1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+(iz+1)*ny+1]
;}else if(ix==0&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+(iz)*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+(iz)*ny+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+(iz+1)*ny+ny-1]
;}
else if(iy==0&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+(iz-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+(iz-1)*nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+(iz)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+(iz)*nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+(iz+1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+(iz+1)*nx+1]
;}else if(iy==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+(iz)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+(iz)*nx+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+(iz+1)*nx+nx-1]
;}
else if(ix==nx-1&&iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+iz-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+iz]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+iz+1]
;}
else if(ix==nx-1&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+(iz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+(iz-1)*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+(iz)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+(iz)*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+(iz+1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+(iz+1)*ny+1]
;}else if(ix==nx-1&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+(iz)*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+(iz)*ny+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+(iz+1)*ny+ny-1]
;}
else if (ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
            else
{
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+1]
-xv[nlocal+0+1+nx+1]
-xv[nlocal+0+1+nx+1+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny+1]
-xv[nlocal+0+1]
-xv[nlocal+0+1+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx+1]
-xv[nlocal+0+1+nx+1+ny]
-xv[nlocal+0+1+nx+1+ny+1]
-xv[nlocal+0+1+nx+1+ny+nx]
-xv[nlocal+0+1+nx+1+ny+nx+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*(nz-2)]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*(nz-2)+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*(nz-1)]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*(nz-1)+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*(nz-2)]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*(nz-2)+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*(nz-1)]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*(nz-1)+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+1+ny-2]
-xv[nlocal+0+1+nx+1+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+2*ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+2*ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*(ny-2)]
-xv[nlocal+0+1+nx+1+ny+nx*(ny-2)+1]
-xv[nlocal+0+1+nx+1+ny+nx*(ny-1)]
-xv[nlocal+0+1+nx+1+ny+nx*(ny-1)+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*(nz-1)-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*(nz-1)-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz-1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx-2]
-xv[nlocal+0+1+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+2*nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+2*nx-1]
-xv[nlocal+0+1+nx+1+ny+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx-1]
-xv[nlocal+0+1+nx+1+ny+2*nx-2]
-xv[nlocal+0+1+nx+1+ny+2*nx-1]
-xv[nlocal+0+1+nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*(nz-1)-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*(nz-1)-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*(nz-2)]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*(nz-2)+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*(nz-1)]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*(nz-1)+1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+1+ny+nx*(ny-1)-2]
-xv[nlocal+0+1+nx+1+ny+nx*(ny-1)-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+2*ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+2*ny-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*(nz-1)-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*(nz-1)-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny*nz-1]
;
}
else if(iy==0&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+ix-1]
-xv[nlocal+0+1+ix]
-xv[nlocal+0+1+ix+1]
;}
else if(iy==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx+ix-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx+ix]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx+ix+1]
;}else if(iy==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix+(nz-2)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix+(nz-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+ix+(nz-1)*nx+1]
;}
else if(iz==0&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+ix-1]
-xv[nlocal+0+1+nx+1+ny+ix]
-xv[nlocal+0+1+nx+1+ny+ix+1]
-xv[nlocal+0+1+nx+1+ny+nx+ix-1]
-xv[nlocal+0+1+nx+1+ny+nx+ix]
-xv[nlocal+0+1+nx+1+ny+nx+ix+1]
;}else if(iz==0&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+1+nx+1+ny+ix+(ny-2)*nx]
-xv[nlocal+0+1+nx+1+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+1+nx+1+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+1+nx+1+ny+ix+(ny-1)*nx]
-xv[nlocal+0+1+nx+1+ny+ix+(ny-1)*nx+1]
;}
else if(ix==0&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+iy-1]
-xv[nlocal+0+1+nx+1+iy]
-xv[nlocal+0+1+nx+1+iy+1]
;}
else if(ix==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny+iy-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny+iy]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny+iy+1]
;}else if(ix==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy+(nz-2)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy+(nz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+iy+(nz-1)*ny+1]
;}
else if(iz==0 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+(iy-1)*nx]
-xv[nlocal+0+1+nx+1+ny+(iy-1)*nx+1]
-xv[nlocal+0+1+nx+1+ny+(iy)*nx]
-xv[nlocal+0+1+nx+1+ny+(iy)*nx+1]
-xv[nlocal+0+1+nx+1+ny+(iy+1)*nx]
-xv[nlocal+0+1+nx+1+ny+(iy+1)*nx+1]
;}else if(iz==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+1+nx+1+ny+(iy)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+(iy)*nx+nx-1]
-xv[nlocal+0+1+nx+1+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+(iy+1)*nx+nx-1]
;}
else if(ix==nx-1&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+iy-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+iy]
-xv[nlocal+0+1+nx+1+ny+nx*ny+iy+1]
;}
else if(ix==nx-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny+iy-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny+iy]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+ny+iy+1]
;}else if(ix==nx-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy+(nz-2)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy+(nz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+iy+(nz-1)*ny+1]
;}
else if(ix==0&&iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+iz-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+iz]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+iz+1]
;}
else if(ix==0&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+(iz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+(iz-1)*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+(iz)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+(iz)*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+(iz+1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+(iz+1)*ny+1]
;}else if(ix==0&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+(iz)*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+(iz)*ny+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+(iz+1)*ny+ny-1]
;}
else if(iy==0&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+(iz-1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+(iz-1)*nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+(iz)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+(iz)*nx+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+(iz+1)*nx]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+(iz+1)*nx+1]
;}else if(iy==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+(iz)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+(iz)*nx+nx-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+(iz+1)*nx+nx-1]
;}
else if(ix==nx-1&&iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+iz-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+iz]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+iz+1]
;}
else if(ix==nx-1&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+(iz-1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+(iz-1)*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+(iz)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+(iz)*ny+1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+(iz+1)*ny]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+(iz+1)*ny+1]
;}else if(ix==nx-1&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+(iz)*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+(iz)*ny+ny-1]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+1+nx+1+ny+nx*ny+ny+nz+nx*nz+nz+ny*nz+(iz+1)*ny+ny-1]
;}
else if (ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
          }//ipz > 0 
        else
{
            if(ipz < npz - 1)
{
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+nz+nx*nz+nz]
-xv[nlocal+0+nz+nx*nz+nz+1]
-xv[nlocal+0+nz+nx*nz+nz+ny]
-xv[nlocal+0+nz+nx*nz+nz+ny+1]
-xv[nlocal+0+nz]
-xv[nlocal+0+nz+1]
-xv[nlocal+0+nz+nx]
-xv[nlocal+0+nz+nx+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz-2]
-xv[nlocal+0+nz-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz]
-xv[nlocal+0+nz+nx*nz+nz+ny*(nz-2)]
-xv[nlocal+0+nz+nx*nz+nz+ny*(nz-2)+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*(nz-1)]
-xv[nlocal+0+nz+nx*nz+nz+ny*(nz-1)+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+1]
-xv[nlocal+0+nz+nx*(nz-2)]
-xv[nlocal+0+nz+nx*(nz-2)+1]
-xv[nlocal+0+nz+nx*(nz-1)]
-xv[nlocal+0+nz+nx*(nz-1)+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx*nz+nz+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+ny-1]
-xv[nlocal+0+nz+nx*nz+nz+2*ny-2]
-xv[nlocal+0+nz+nx*nz+nz+2*ny-1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx*nz+nz+ny*(nz-1)-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*(nz-1)-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx*(ny-2)]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx*(ny-2)+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx*(ny-1)]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx*(ny-1)+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx-2]
-xv[nlocal+0+nz+nx-1]
-xv[nlocal+0+nz+2*nx-2]
-xv[nlocal+0+nz+2*nx-1]
-xv[nlocal+0+nz+nx*nz]
-xv[nlocal+0+nz+nx*nz+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx*(nz-1)-2]
-xv[nlocal+0+nz+nx*(nz-1)-1]
-xv[nlocal+0+nz+nx*nz-2]
-xv[nlocal+0+nz+nx*nz-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+2*nx-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+2*nx-1]
-xv[nlocal+0+nz+nx*nz+nz-2]
-xv[nlocal+0+nz+nx*nz+nz-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*(nz-2)]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*(nz-2)+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*(nz-1)]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*(nz-1)+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx*ny+1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+2*ny-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+2*ny-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx*(ny-1)-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx*(ny-1)-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx*ny-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx*ny-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*(nz-1)-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*(nz-1)-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx*ny+ny-1]
;
}
else if(iy==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+ix-1]
-xv[nlocal+0+nz+ix]
-xv[nlocal+0+nz+ix+1]
-xv[nlocal+0+nz+nx+ix-1]
-xv[nlocal+0+nz+nx+ix]
-xv[nlocal+0+nz+nx+ix+1]
;}else if(iy==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+nz+ix+(nz-2)*nx]
-xv[nlocal+0+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+nz+ix+(nz-1)*nx]
-xv[nlocal+0+nz+ix+(nz-1)*nx+1]
;}
else if(iy==0&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+ix-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+ix]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+ix+1]
;}
else if(iz==nz-1&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx+ix-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx+ix]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx+ix+1]
;}else if(iz==nz-1&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix+(ny-2)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix+(ny-1)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+ix+(ny-1)*nx+1]
;}
else if(ix==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+iy-1]
-xv[nlocal+0+nz+nx*nz+nz+iy]
-xv[nlocal+0+nz+nx*nz+nz+iy+1]
-xv[nlocal+0+nz+nx*nz+nz+ny+iy-1]
-xv[nlocal+0+nz+nx*nz+nz+ny+iy]
-xv[nlocal+0+nz+nx*nz+nz+ny+iy+1]
;}else if(ix==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+nz+nx*nz+nz+iy+(nz-2)*ny]
-xv[nlocal+0+nz+nx*nz+nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+nz+nx*nz+nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+nz+nx*nz+nz+iy+(nz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+iy+(nz-1)*ny+1]
;}
else if(ix==0&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+iy-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+iy]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+iy+1]
;}
else if(iz==nz-1 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+(iy-1)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+(iy-1)*nx+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+(iy)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+(iy)*nx+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+(iy+1)*nx]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+(iy+1)*nx+1]
;}else if(iz==nz-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+(iy)*nx+nx-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+(iy)*nx+nx-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+(iy+1)*nx+nx-1]
;}
else if(ix==nx-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny+iy-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny+iy]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny+iy+1]
;}else if(ix==nx-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(nz-2)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(nz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(nz-1)*ny+1]
;}
else if(ix==nx-1&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx*ny+iy-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx*ny+iy]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz+1+nx+1+ny+nx*ny+iy+1]
;}
else if(ix==0&&iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iz-1]
-xv[nlocal+0+iz]
-xv[nlocal+0+iz+1]
;}
else if(ix==0&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+(iz-1)*ny+1]
-xv[nlocal+0+nz+nx*nz+nz+(iz)*ny]
-xv[nlocal+0+nz+nx*nz+nz+(iz)*ny+1]
-xv[nlocal+0+nz+nx*nz+nz+(iz+1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+(iz+1)*ny+1]
;}else if(ix==0&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+nz+nx*nz+nz+(iz)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+(iz)*ny+ny-1]
-xv[nlocal+0+nz+nx*nz+nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+(iz+1)*ny+ny-1]
;}
else if(iy==0&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+(iz-1)*nx]
-xv[nlocal+0+nz+(iz-1)*nx+1]
-xv[nlocal+0+nz+(iz)*nx]
-xv[nlocal+0+nz+(iz)*nx+1]
-xv[nlocal+0+nz+(iz+1)*nx]
-xv[nlocal+0+nz+(iz+1)*nx+1]
;}else if(iy==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+nz+(iz)*nx+nx-2]
-xv[nlocal+0+nz+(iz)*nx+nx-1]
-xv[nlocal+0+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+nz+(iz+1)*nx+nx-1]
;}
else if(ix==nx-1&&iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+iz-1]
-xv[nlocal+0+nz+nx*nz+iz]
-xv[nlocal+0+nz+nx*nz+iz+1]
;}
else if(ix==nx-1&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz-1)*ny+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz)*ny+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz+1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz+1)*ny+1]
;}else if(ix==nx-1&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz)*ny+ny-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz+1)*ny+ny-1]
;}
else if (ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
            else
{
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+nz+nx*nz+nz]
-xv[nlocal+0+nz+nx*nz+nz+1]
-xv[nlocal+0+nz+nx*nz+nz+ny]
-xv[nlocal+0+nz+nx*nz+nz+ny+1]
-xv[nlocal+0+nz]
-xv[nlocal+0+nz+1]
-xv[nlocal+0+nz+nx]
-xv[nlocal+0+nz+nx+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz-2]
-xv[nlocal+0+nz-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*(nz-2)]
-xv[nlocal+0+nz+nx*nz+nz+ny*(nz-2)+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*(nz-1)]
-xv[nlocal+0+nz+nx*nz+nz+ny*(nz-1)+1]
-xv[nlocal+0+nz+nx*(nz-2)]
-xv[nlocal+0+nz+nx*(nz-2)+1]
-xv[nlocal+0+nz+nx*(nz-1)]
-xv[nlocal+0+nz+nx*(nz-1)+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx*nz+nz+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+ny-1]
-xv[nlocal+0+nz+nx*nz+nz+2*ny-2]
-xv[nlocal+0+nz+nx*nz+nz+2*ny-1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx*nz+nz+ny*(nz-1)-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*(nz-1)-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz-1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx-2]
-xv[nlocal+0+nz+nx-1]
-xv[nlocal+0+nz+2*nx-2]
-xv[nlocal+0+nz+2*nx-1]
-xv[nlocal+0+nz+nx*nz]
-xv[nlocal+0+nz+nx*nz+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx*(nz-1)-2]
-xv[nlocal+0+nz+nx*(nz-1)-1]
-xv[nlocal+0+nz+nx*nz-2]
-xv[nlocal+0+nz+nx*nz-1]
-xv[nlocal+0+nz+nx*nz+nz-2]
-xv[nlocal+0+nz+nx*nz+nz-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*(nz-2)]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*(nz-2)+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*(nz-1)]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*(nz-1)+1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+2*ny-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+2*ny-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*(nz-1)-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*(nz-1)-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny*nz-1]
;
}
else if(iy==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+ix-1]
-xv[nlocal+0+nz+ix]
-xv[nlocal+0+nz+ix+1]
-xv[nlocal+0+nz+nx+ix-1]
-xv[nlocal+0+nz+nx+ix]
-xv[nlocal+0+nz+nx+ix+1]
;}else if(iy==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+nz+ix+(nz-2)*nx]
-xv[nlocal+0+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+nz+ix+(nz-1)*nx]
-xv[nlocal+0+nz+ix+(nz-1)*nx+1]
;}
else if(ix==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+iy-1]
-xv[nlocal+0+nz+nx*nz+nz+iy]
-xv[nlocal+0+nz+nx*nz+nz+iy+1]
-xv[nlocal+0+nz+nx*nz+nz+ny+iy-1]
-xv[nlocal+0+nz+nx*nz+nz+ny+iy]
-xv[nlocal+0+nz+nx*nz+nz+ny+iy+1]
;}else if(ix==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+nz+nx*nz+nz+iy+(nz-2)*ny]
-xv[nlocal+0+nz+nx*nz+nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+nz+nx*nz+nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+nz+nx*nz+nz+iy+(nz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+iy+(nz-1)*ny+1]
;}
else if(ix==nx-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny+iy-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny+iy]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+ny+iy+1]
;}else if(ix==nx-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(nz-2)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(nz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+iy+(nz-1)*ny+1]
;}
else if(ix==0&&iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iz-1]
-xv[nlocal+0+iz]
-xv[nlocal+0+iz+1]
;}
else if(ix==0&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+(iz-1)*ny+1]
-xv[nlocal+0+nz+nx*nz+nz+(iz)*ny]
-xv[nlocal+0+nz+nx*nz+nz+(iz)*ny+1]
-xv[nlocal+0+nz+nx*nz+nz+(iz+1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+(iz+1)*ny+1]
;}else if(ix==0&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+nz+nx*nz+nz+(iz)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+(iz)*ny+ny-1]
-xv[nlocal+0+nz+nx*nz+nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+(iz+1)*ny+ny-1]
;}
else if(iy==0&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+(iz-1)*nx]
-xv[nlocal+0+nz+(iz-1)*nx+1]
-xv[nlocal+0+nz+(iz)*nx]
-xv[nlocal+0+nz+(iz)*nx+1]
-xv[nlocal+0+nz+(iz+1)*nx]
-xv[nlocal+0+nz+(iz+1)*nx+1]
;}else if(iy==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+nz+(iz)*nx+nx-2]
-xv[nlocal+0+nz+(iz)*nx+nx-1]
-xv[nlocal+0+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+nz+(iz+1)*nx+nx-1]
;}
else if(ix==nx-1&&iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+iz-1]
-xv[nlocal+0+nz+nx*nz+iz]
-xv[nlocal+0+nz+nx*nz+iz+1]
;}
else if(ix==nx-1&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz-1)*ny+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz)*ny+1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz+1)*ny]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz+1)*ny+1]
;}else if(ix==nx-1&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz)*ny+ny-1]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+nz+ny*nz+(iz+1)*ny+ny-1]
;}
else if (ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny+1]
-xv[nlocal+0+ny]
-xv[nlocal+0+ny+1]
-xv[nlocal+0+ny+nx]
-xv[nlocal+0+ny+nx+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*(nz-2)]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*(nz-2)+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*(nz-1)]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*(nz-1)+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny-2]
-xv[nlocal+0+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+2*ny-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+2*ny-1]
-xv[nlocal+0+ny+nx*ny+ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+1]
-xv[nlocal+0+ny+nx*(ny-2)]
-xv[nlocal+0+ny+nx*(ny-2)+1]
-xv[nlocal+0+ny+nx*(ny-1)]
-xv[nlocal+0+ny+nx*(ny-1)+1]
-xv[nlocal+0+ny+nx*ny+ny+1]
-xv[nlocal+0+ny+nx*ny+ny+1+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*(nz-1)-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*(nz-1)-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*(ny-2)]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*(ny-2)+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*(ny-1)]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*(ny-1)+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*(nz-2)]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*(nz-2)+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*(nz-1)]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*(nz-1)+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+ny+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+ny+1+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx-2]
-xv[nlocal+0+ny+nx-1]
-xv[nlocal+0+ny+2*nx-2]
-xv[nlocal+0+ny+2*nx-1]
-xv[nlocal+0+ny+nx*ny]
-xv[nlocal+0+ny+nx*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+2*nx-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+2*nx-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*(nz-2)]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*(nz-2)+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*(nz-1)]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*(nz-1)+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx*(ny-1)-2]
-xv[nlocal+0+ny+nx*(ny-1)-1]
-xv[nlocal+0+ny+nx*ny-2]
-xv[nlocal+0+ny+nx*ny-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+2*nx-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+2*nx-1]
-xv[nlocal+0+ny+nx*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+2*ny-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+2*ny-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*(ny-1)-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*(ny-1)-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*(nz-1)-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*(nz-1)-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+ny+1+nx-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+ny+1+nx-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*(nz-1)-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*(nz-1)-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+ny+1+nx]
;
}
else if(iz==0&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+ix-1]
-xv[nlocal+0+ny+ix]
-xv[nlocal+0+ny+ix+1]
-xv[nlocal+0+ny+nx+ix-1]
-xv[nlocal+0+ny+nx+ix]
-xv[nlocal+0+ny+nx+ix+1]
;}else if(iz==0&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+ny+ix+(ny-2)*nx]
-xv[nlocal+0+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+ny+ix+(ny-1)*nx]
-xv[nlocal+0+ny+ix+(ny-1)*nx+1]
;}
else if(iz==nz-1&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx+ix-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx+ix]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx+ix+1]
;}else if(iz==nz-1&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix+(ny-2)*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix+(ny-1)*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix+(ny-1)*nx+1]
;}
else if(iy==ny-1&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+ix-1]
-xv[nlocal+0+ny+nx*ny+ny+1+ix]
-xv[nlocal+0+ny+nx*ny+ny+1+ix+1]
;}
else if(iy==ny-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx+ix-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx+ix]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx+ix+1]
;}else if(iy==ny-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix+(nz-2)*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix+(nz-1)*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix+(nz-1)*nx+1]
;}
else if(iy==ny-1&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+ny+1+ix-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+ny+1+ix]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+ny+1+ix+1]
;}
else if(ix==0&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1]
-xv[nlocal+0+iy]
-xv[nlocal+0+iy+1]
;}
else if(ix==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny+iy-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny+iy]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny+iy+1]
;}else if(ix==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy+(nz-2)*ny-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy+(nz-2)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy+(nz-2)*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy+(nz-1)*ny-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy+(nz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy+(nz-1)*ny+1]
;}
else if(ix==0&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+iy-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+iy]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+iy+1]
;}
else if(iz==0 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+(iy-1)*nx]
-xv[nlocal+0+ny+(iy-1)*nx+1]
-xv[nlocal+0+ny+(iy)*nx]
-xv[nlocal+0+ny+(iy)*nx+1]
-xv[nlocal+0+ny+(iy+1)*nx]
-xv[nlocal+0+ny+(iy+1)*nx+1]
;}else if(iz==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+ny+(iy)*nx+nx-2]
-xv[nlocal+0+ny+(iy)*nx+nx-1]
-xv[nlocal+0+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+ny+(iy+1)*nx+nx-1]
;}
else if(iz==nz-1 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+(iy-1)*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+(iy-1)*nx+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+(iy)*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+(iy)*nx+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+(iy+1)*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+(iy+1)*nx+1]
;}else if(iz==nz-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+(iy)*nx+nx-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+(iy)*nx+nx-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+(iy+1)*nx+nx-1]
;}
else if(ix==nx-1&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+iy-1]
-xv[nlocal+0+ny+nx*ny+iy]
-xv[nlocal+0+ny+nx*ny+iy+1]
;}
else if(ix==nx-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny+iy-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny+iy]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny+iy+1]
;}else if(ix==nx-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy+(nz-2)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy+(nz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy+(nz-1)*ny+1]
;}
else if(ix==nx-1&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+iy-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+iy]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+iy+1]
;}
else if(ix==0&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+(iz-1)*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+(iz)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+(iz)*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+(iz+1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+(iz+1)*ny+1]
;}else if(ix==0&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+(iz-1)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+(iz-1)*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+(iz)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+(iz)*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+(iz+1)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+(iz+1)*ny+ny-1]
;}
else if(ix==0&&iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+iz-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+iz]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+iz+1]
;}
else if(iy==ny-1&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+(iz-1)*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+(iz-1)*nx+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+(iz)*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+(iz)*nx+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+(iz+1)*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+(iz+1)*nx+1]
;}else if(iy==ny-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+(iz)*nx+nx-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+(iz)*nx+nx-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+(iz+1)*nx+nx-1]
;}
else if(ix==nx-1&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+(iz-1)*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+(iz)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+(iz)*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+(iz+1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+(iz+1)*ny+1]
;}else if(ix==nx-1&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+(iz)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+(iz)*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+(iz+1)*ny+ny-1]
;}
else if(ix==nx-1&&iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+iz-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+iz]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+iz+1]
;}
else if (ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
            else
{
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny+1]
-xv[nlocal+0+ny]
-xv[nlocal+0+ny+1]
-xv[nlocal+0+ny+nx]
-xv[nlocal+0+ny+nx+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*(nz-2)]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*(nz-2)+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*(nz-1)]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*(nz-1)+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny-2]
-xv[nlocal+0+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+2*ny-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+2*ny-1]
-xv[nlocal+0+ny+nx*ny+ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+1]
-xv[nlocal+0+ny+nx*(ny-2)]
-xv[nlocal+0+ny+nx*(ny-2)+1]
-xv[nlocal+0+ny+nx*(ny-1)]
-xv[nlocal+0+ny+nx*(ny-1)+1]
-xv[nlocal+0+ny+nx*ny+ny+1]
-xv[nlocal+0+ny+nx*ny+ny+1+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*(nz-1)-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*(nz-1)-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*(nz-2)]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*(nz-2)+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*(nz-1)]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*(nz-1)+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx-2]
-xv[nlocal+0+ny+nx-1]
-xv[nlocal+0+ny+2*nx-2]
-xv[nlocal+0+ny+2*nx-1]
-xv[nlocal+0+ny+nx*ny]
-xv[nlocal+0+ny+nx*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*(nz-2)]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*(nz-2)+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*(nz-1)]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*(nz-1)+1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx*(ny-1)-2]
-xv[nlocal+0+ny+nx*(ny-1)-1]
-xv[nlocal+0+ny+nx*ny-2]
-xv[nlocal+0+ny+nx*ny-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+2*nx-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+2*nx-1]
-xv[nlocal+0+ny+nx*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+2*ny-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+2*ny-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*(nz-1)-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*(nz-1)-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*(nz-1)-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*(nz-1)-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+nz-1]
;
}
else if(iz==0&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+ix-1]
-xv[nlocal+0+ny+ix]
-xv[nlocal+0+ny+ix+1]
-xv[nlocal+0+ny+nx+ix-1]
-xv[nlocal+0+ny+nx+ix]
-xv[nlocal+0+ny+nx+ix+1]
;}else if(iz==0&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+ny+ix+(ny-2)*nx]
-xv[nlocal+0+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+ny+ix+(ny-1)*nx]
-xv[nlocal+0+ny+ix+(ny-1)*nx+1]
;}
else if(iy==ny-1&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+ix-1]
-xv[nlocal+0+ny+nx*ny+ny+1+ix]
-xv[nlocal+0+ny+nx*ny+ny+1+ix+1]
;}
else if(iy==ny-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx+ix-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx+ix]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx+ix+1]
;}else if(iy==ny-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix+(nz-2)*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix+(nz-1)*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+ix+(nz-1)*nx+1]
;}
else if(ix==0&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1]
-xv[nlocal+0+iy]
-xv[nlocal+0+iy+1]
;}
else if(ix==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny+iy-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny+iy]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny+iy+1]
;}else if(ix==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy+(nz-2)*ny-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy+(nz-2)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy+(nz-2)*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy+(nz-1)*ny-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy+(nz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+iy+(nz-1)*ny+1]
;}
else if(iz==0 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+(iy-1)*nx]
-xv[nlocal+0+ny+(iy-1)*nx+1]
-xv[nlocal+0+ny+(iy)*nx]
-xv[nlocal+0+ny+(iy)*nx+1]
-xv[nlocal+0+ny+(iy+1)*nx]
-xv[nlocal+0+ny+(iy+1)*nx+1]
;}else if(iz==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+ny+(iy)*nx+nx-2]
-xv[nlocal+0+ny+(iy)*nx+nx-1]
-xv[nlocal+0+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+ny+(iy+1)*nx+nx-1]
;}
else if(ix==nx-1&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+iy-1]
-xv[nlocal+0+ny+nx*ny+iy]
-xv[nlocal+0+ny+nx*ny+iy+1]
;}
else if(ix==nx-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny+iy-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny+iy]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny+iy+1]
;}else if(ix==nx-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy+(nz-2)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy+(nz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+iy+(nz-1)*ny+1]
;}
else if(ix==0&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+(iz-1)*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+(iz)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+(iz)*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+(iz+1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+(iz+1)*ny+1]
;}else if(ix==0&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+(iz-1)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+(iz-1)*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+(iz)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+(iz)*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+(iz+1)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+(iz+1)*ny+ny-1]
;}
else if(ix==0&&iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+iz-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+iz]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+iz+1]
;}
else if(iy==ny-1&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+(iz-1)*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+(iz-1)*nx+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+(iz)*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+(iz)*nx+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+(iz+1)*nx]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+(iz+1)*nx+1]
;}else if(iy==ny-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+(iz)*nx+nx-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+(iz)*nx+nx-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+(iz+1)*nx+nx-1]
;}
else if(ix==nx-1&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+(iz-1)*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+(iz)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+(iz)*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+(iz+1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+(iz+1)*ny+1]
;}else if(ix==nx-1&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+(iz)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+(iz)*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+(iz+1)*ny+ny-1]
;}
else if(ix==nx-1&&iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+iz-1]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+iz]
-xv[nlocal+0+ny+nx*ny+ny+1+nx+1+ny*nz+ny*nz+nz+nx*nz+iz+1]
;}
else if (ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
          }//ipz > 0 
        else
{
            if(ipz < npz - 1)
{
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+ny]
-xv[nlocal+0+ny+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*(nz-2)]
-xv[nlocal+0+ny*(nz-2)+1]
-xv[nlocal+0+ny*(nz-1)]
-xv[nlocal+0+ny*(nz-1)+1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny-2]
-xv[nlocal+0+ny-1]
-xv[nlocal+0+2*ny-2]
-xv[nlocal+0+2*ny-1]
-xv[nlocal+0+ny*nz+ny*nz]
-xv[nlocal+0+ny*nz+ny*nz+1]
-xv[nlocal+0+ny*nz+ny*nz+nz]
-xv[nlocal+0+ny*nz+ny*nz+nz+1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*(nz-1)-2]
-xv[nlocal+0+ny*(nz-1)-1]
-xv[nlocal+0+ny*nz-2]
-xv[nlocal+0+ny*nz-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny-2]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny-1]
-xv[nlocal+0+ny*nz+ny*nz+nz-2]
-xv[nlocal+0+ny*nz+ny*nz+nz-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+ny]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*(ny-2)]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*(ny-2)+1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*(ny-1)]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*(ny-1)+1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*(nz-2)]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*(nz-2)+1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*(nz-1)]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*(nz-1)+1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+ny+1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+ny+1+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*nz]
-xv[nlocal+0+ny*nz+1]
-xv[nlocal+0+ny*nz+ny]
-xv[nlocal+0+ny*nz+ny+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx-2]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+2*nx-2]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+2*nx-1]
-xv[nlocal+0+ny*nz+ny*(nz-2)]
-xv[nlocal+0+ny*nz+ny*(nz-2)+1]
-xv[nlocal+0+ny*nz+ny*(nz-1)]
-xv[nlocal+0+ny*nz+ny*(nz-1)+1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*nz+ny*nz+nz+nx-2]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+2*nx-2]
-xv[nlocal+0+ny*nz+ny*nz+nz+2*nx-1]
-xv[nlocal+0+ny*nz+ny-2]
-xv[nlocal+0+ny*nz+ny-1]
-xv[nlocal+0+ny*nz+2*ny-2]
-xv[nlocal+0+ny*nz+2*ny-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*(ny-1)-2]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*(ny-1)-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny-2]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*(nz-1)-2]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*(nz-1)-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz-2]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+ny+1+nx-2]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+ny+1+nx-1]
-xv[nlocal+0+ny*nz+ny*(nz-1)-2]
-xv[nlocal+0+ny*nz+ny*(nz-1)-1]
-xv[nlocal+0+ny*nz+ny*nz-2]
-xv[nlocal+0+ny*nz+ny*nz-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+ny-2]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+ny-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz-2]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+ny+1+nx]
;
}
else if(iz==nz-1&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix+1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx+ix-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx+ix]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx+ix+1]
;}else if(iz==nz-1&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix+(ny-2)*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix+(ny-1)*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+ix+(ny-1)*nx+1]
;}
else if(iy==ny-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+nz+ix-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+ix]
-xv[nlocal+0+ny*nz+ny*nz+nz+ix+1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx+ix-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx+ix]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx+ix+1]
;}else if(iy==ny-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+ix+(nz-2)*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+ny*nz+ny*nz+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+ix+(nz-1)*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+ix+(nz-1)*nx+1]
;}
else if(iy==ny-1&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+ny+1+ix-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+ny+1+ix]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+ny+1+ix+1]
;}
else if(ix==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1]
-xv[nlocal+0+iy]
-xv[nlocal+0+iy+1]
-xv[nlocal+0+ny+iy-1]
-xv[nlocal+0+ny+iy]
-xv[nlocal+0+ny+iy+1]
;}else if(ix==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy+(nz-2)*ny-1]
-xv[nlocal+0+iy+(nz-2)*ny]
-xv[nlocal+0+iy+(nz-2)*ny+1]
-xv[nlocal+0+iy+(nz-1)*ny-1]
-xv[nlocal+0+iy+(nz-1)*ny]
-xv[nlocal+0+iy+(nz-1)*ny+1]
;}
else if(ix==0&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+iy-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+iy]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+iy+1]
;}
else if(iz==nz-1 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+(iy-1)*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+(iy-1)*nx+1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+(iy)*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+(iy)*nx+1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+(iy+1)*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+(iy+1)*nx+1]
;}else if(iz==nz-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+(iy)*nx+nx-2]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+(iy)*nx+nx-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+(iy+1)*nx+nx-1]
;}
else if(ix==nx-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+iy-1]
-xv[nlocal+0+ny*nz+iy]
-xv[nlocal+0+ny*nz+iy+1]
-xv[nlocal+0+ny*nz+ny+iy-1]
-xv[nlocal+0+ny*nz+ny+iy]
-xv[nlocal+0+ny*nz+ny+iy+1]
;}else if(ix==nx-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+ny*nz+iy+(nz-2)*ny]
-xv[nlocal+0+ny*nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+ny*nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+ny*nz+iy+(nz-1)*ny]
-xv[nlocal+0+ny*nz+iy+(nz-1)*ny+1]
;}
else if(ix==nx-1&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+iy-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+iy]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz+ny+nx*ny+iy+1]
;}
else if(ix==0&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*ny]
-xv[nlocal+0+(iz-1)*ny+1]
-xv[nlocal+0+(iz)*ny]
-xv[nlocal+0+(iz)*ny+1]
-xv[nlocal+0+(iz+1)*ny]
-xv[nlocal+0+(iz+1)*ny+1]
;}else if(ix==0&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*ny+ny-2]
-xv[nlocal+0+(iz-1)*ny+ny-1]
-xv[nlocal+0+(iz)*ny+ny-2]
-xv[nlocal+0+(iz)*ny+ny-1]
-xv[nlocal+0+(iz+1)*ny+ny-2]
-xv[nlocal+0+(iz+1)*ny+ny-1]
;}
else if(ix==0&&iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+iz-1]
-xv[nlocal+0+ny*nz+ny*nz+iz]
-xv[nlocal+0+ny*nz+ny*nz+iz+1]
;}
else if(iy==ny-1&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+nz+(iz-1)*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+(iz-1)*nx+1]
-xv[nlocal+0+ny*nz+ny*nz+nz+(iz)*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+(iz)*nx+1]
-xv[nlocal+0+ny*nz+ny*nz+nz+(iz+1)*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+(iz+1)*nx+1]
;}else if(iy==ny-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+ny*nz+ny*nz+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+(iz)*nx+nx-2]
-xv[nlocal+0+ny*nz+ny*nz+nz+(iz)*nx+nx-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+ny*nz+ny*nz+nz+(iz+1)*nx+nx-1]
;}
else if(ix==nx-1&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+(iz-1)*ny]
-xv[nlocal+0+ny*nz+(iz-1)*ny+1]
-xv[nlocal+0+ny*nz+(iz)*ny]
-xv[nlocal+0+ny*nz+(iz)*ny+1]
-xv[nlocal+0+ny*nz+(iz+1)*ny]
-xv[nlocal+0+ny*nz+(iz+1)*ny+1]
;}else if(ix==nx-1&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+ny*nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+ny*nz+(iz)*ny+ny-2]
-xv[nlocal+0+ny*nz+(iz)*ny+ny-1]
-xv[nlocal+0+ny*nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+ny*nz+(iz+1)*ny+ny-1]
;}
else if(ix==nx-1&&iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+iz-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+iz]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+iz+1]
;}
else if (ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
            else
{
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+ny]
-xv[nlocal+0+ny+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*(nz-2)]
-xv[nlocal+0+ny*(nz-2)+1]
-xv[nlocal+0+ny*(nz-1)]
-xv[nlocal+0+ny*(nz-1)+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny-2]
-xv[nlocal+0+ny-1]
-xv[nlocal+0+2*ny-2]
-xv[nlocal+0+2*ny-1]
-xv[nlocal+0+ny*nz+ny*nz]
-xv[nlocal+0+ny*nz+ny*nz+1]
-xv[nlocal+0+ny*nz+ny*nz+nz]
-xv[nlocal+0+ny*nz+ny*nz+nz+1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*(nz-1)-2]
-xv[nlocal+0+ny*(nz-1)-1]
-xv[nlocal+0+ny*nz-2]
-xv[nlocal+0+ny*nz-1]
-xv[nlocal+0+ny*nz+ny*nz+nz-2]
-xv[nlocal+0+ny*nz+ny*nz+nz-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*(nz-2)]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*(nz-2)+1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*(nz-1)]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*(nz-1)+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*nz]
-xv[nlocal+0+ny*nz+1]
-xv[nlocal+0+ny*nz+ny]
-xv[nlocal+0+ny*nz+ny+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*nz+ny*(nz-2)]
-xv[nlocal+0+ny*nz+ny*(nz-2)+1]
-xv[nlocal+0+ny*nz+ny*(nz-1)]
-xv[nlocal+0+ny*nz+ny*(nz-1)+1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*nz+ny*nz+nz+nx-2]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+2*nx-2]
-xv[nlocal+0+ny*nz+ny*nz+nz+2*nx-1]
-xv[nlocal+0+ny*nz+ny-2]
-xv[nlocal+0+ny*nz+ny-1]
-xv[nlocal+0+ny*nz+2*ny-2]
-xv[nlocal+0+ny*nz+2*ny-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*(nz-1)-2]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*(nz-1)-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz-2]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz-1]
-xv[nlocal+0+ny*nz+ny*(nz-1)-2]
-xv[nlocal+0+ny*nz+ny*(nz-1)-1]
-xv[nlocal+0+ny*nz+ny*nz-2]
-xv[nlocal+0+ny*nz+ny*nz-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz-2]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+nz-1]
;
}
else if(iy==ny-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+nz+ix-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+ix]
-xv[nlocal+0+ny*nz+ny*nz+nz+ix+1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx+ix-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx+ix]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx+ix+1]
;}else if(iy==ny-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+ix+(nz-2)*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+ny*nz+ny*nz+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+ix+(nz-1)*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+ix+(nz-1)*nx+1]
;}
else if(ix==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1]
-xv[nlocal+0+iy]
-xv[nlocal+0+iy+1]
-xv[nlocal+0+ny+iy-1]
-xv[nlocal+0+ny+iy]
-xv[nlocal+0+ny+iy+1]
;}else if(ix==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy+(nz-2)*ny-1]
-xv[nlocal+0+iy+(nz-2)*ny]
-xv[nlocal+0+iy+(nz-2)*ny+1]
-xv[nlocal+0+iy+(nz-1)*ny-1]
-xv[nlocal+0+iy+(nz-1)*ny]
-xv[nlocal+0+iy+(nz-1)*ny+1]
;}
else if(ix==nx-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+iy-1]
-xv[nlocal+0+ny*nz+iy]
-xv[nlocal+0+ny*nz+iy+1]
-xv[nlocal+0+ny*nz+ny+iy-1]
-xv[nlocal+0+ny*nz+ny+iy]
-xv[nlocal+0+ny*nz+ny+iy+1]
;}else if(ix==nx-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+ny*nz+iy+(nz-2)*ny]
-xv[nlocal+0+ny*nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+ny*nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+ny*nz+iy+(nz-1)*ny]
-xv[nlocal+0+ny*nz+iy+(nz-1)*ny+1]
;}
else if(ix==0&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*ny]
-xv[nlocal+0+(iz-1)*ny+1]
-xv[nlocal+0+(iz)*ny]
-xv[nlocal+0+(iz)*ny+1]
-xv[nlocal+0+(iz+1)*ny]
-xv[nlocal+0+(iz+1)*ny+1]
;}else if(ix==0&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*ny+ny-2]
-xv[nlocal+0+(iz-1)*ny+ny-1]
-xv[nlocal+0+(iz)*ny+ny-2]
-xv[nlocal+0+(iz)*ny+ny-1]
-xv[nlocal+0+(iz+1)*ny+ny-2]
-xv[nlocal+0+(iz+1)*ny+ny-1]
;}
else if(ix==0&&iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+iz-1]
-xv[nlocal+0+ny*nz+ny*nz+iz]
-xv[nlocal+0+ny*nz+ny*nz+iz+1]
;}
else if(iy==ny-1&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+nz+(iz-1)*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+(iz-1)*nx+1]
-xv[nlocal+0+ny*nz+ny*nz+nz+(iz)*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+(iz)*nx+1]
-xv[nlocal+0+ny*nz+ny*nz+nz+(iz+1)*nx]
-xv[nlocal+0+ny*nz+ny*nz+nz+(iz+1)*nx+1]
;}else if(iy==ny-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+ny*nz+ny*nz+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+(iz)*nx+nx-2]
-xv[nlocal+0+ny*nz+ny*nz+nz+(iz)*nx+nx-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+ny*nz+ny*nz+nz+(iz+1)*nx+nx-1]
;}
else if(ix==nx-1&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+(iz-1)*ny]
-xv[nlocal+0+ny*nz+(iz-1)*ny+1]
-xv[nlocal+0+ny*nz+(iz)*ny]
-xv[nlocal+0+ny*nz+(iz)*ny+1]
-xv[nlocal+0+ny*nz+(iz+1)*ny]
-xv[nlocal+0+ny*nz+(iz+1)*ny+1]
;}else if(ix==nx-1&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+ny*nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+ny*nz+(iz)*ny+ny-2]
-xv[nlocal+0+ny*nz+(iz)*ny+ny-1]
-xv[nlocal+0+ny*nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+ny*nz+(iz+1)*ny+ny-1]
;}
else if(ix==nx-1&&iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+iz-1]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+iz]
-xv[nlocal+0+ny*nz+ny*nz+nz+nx*nz+iz+1]
;}
else if (ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
          }//ipz > 0 
        }//ipy < npy - 1 
      else
{
        if(ipz > 0)
{
            if(ipz < npz - 1)
{
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+ny+nx*ny+ny]
-xv[nlocal+0+ny+nx*ny+ny+1]
-xv[nlocal+0+ny+nx*ny+ny+ny]
-xv[nlocal+0+ny+nx*ny+ny+ny+1]
-xv[nlocal+0+ny]
-xv[nlocal+0+ny+1]
-xv[nlocal+0+ny+nx]
-xv[nlocal+0+ny+nx+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx*ny+ny+ny*(nz-2)]
-xv[nlocal+0+ny+nx*ny+ny+ny*(nz-2)+1]
-xv[nlocal+0+ny+nx*ny+ny+ny*(nz-1)]
-xv[nlocal+0+ny+nx*ny+ny+ny*(nz-1)+1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+nx]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny-2]
-xv[nlocal+0+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+2*ny-2]
-xv[nlocal+0+ny+nx*ny+ny+2*ny-1]
-xv[nlocal+0+ny+nx*(ny-2)]
-xv[nlocal+0+ny+nx*(ny-2)+1]
-xv[nlocal+0+ny+nx*(ny-1)]
-xv[nlocal+0+ny+nx*(ny-1)+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx*ny+ny+ny*(nz-1)-2]
-xv[nlocal+0+ny+nx*ny+ny+ny*(nz-1)-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz-2]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+nx*(ny-2)]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+nx*(ny-2)+1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+nx*(ny-1)]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+nx*(ny-1)+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx-2]
-xv[nlocal+0+ny+nx-1]
-xv[nlocal+0+ny+2*nx-2]
-xv[nlocal+0+ny+2*nx-1]
-xv[nlocal+0+ny+nx*ny]
-xv[nlocal+0+ny+nx*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+nx-2]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+nx-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+2*nx-2]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+2*nx-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*(nz-2)]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*(nz-2)+1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*(nz-1)]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*(nz-1)+1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+nx*ny]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+nx*ny+1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx*(ny-1)-2]
-xv[nlocal+0+ny+nx*(ny-1)-1]
-xv[nlocal+0+ny+nx*ny-2]
-xv[nlocal+0+ny+nx*ny-1]
-xv[nlocal+0+ny+nx*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+2*ny-2]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+2*ny-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+nx*(ny-1)-2]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+nx*(ny-1)-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+nx*ny-2]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+nx*ny-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*(nz-1)-2]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*(nz-1)-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz-2]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+nx*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+nx*ny+ny-1]
;
}
else if(iz==0&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+ix-1]
-xv[nlocal+0+ny+ix]
-xv[nlocal+0+ny+ix+1]
-xv[nlocal+0+ny+nx+ix-1]
-xv[nlocal+0+ny+nx+ix]
-xv[nlocal+0+ny+nx+ix+1]
;}else if(iz==0&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+ny+ix+(ny-2)*nx]
-xv[nlocal+0+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+ny+ix+(ny-1)*nx]
-xv[nlocal+0+ny+ix+(ny-1)*nx+1]
;}
else if(iz==nz-1&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+ix-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+ix]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+ix+1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+nx+ix-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+nx+ix]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+nx+ix+1]
;}else if(iz==nz-1&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+ix+(ny-2)*nx]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+ix+(ny-1)*nx]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+ix+(ny-1)*nx+1]
;}
else if(ix==0&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1]
-xv[nlocal+0+iy]
-xv[nlocal+0+iy+1]
;}
else if(ix==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+iy-1]
-xv[nlocal+0+ny+nx*ny+ny+iy]
-xv[nlocal+0+ny+nx*ny+ny+iy+1]
-xv[nlocal+0+ny+nx*ny+ny+ny+iy-1]
-xv[nlocal+0+ny+nx*ny+ny+ny+iy]
-xv[nlocal+0+ny+nx*ny+ny+ny+iy+1]
;}else if(ix==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+iy+(nz-2)*ny-1]
-xv[nlocal+0+ny+nx*ny+ny+iy+(nz-2)*ny]
-xv[nlocal+0+ny+nx*ny+ny+iy+(nz-2)*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+iy+(nz-1)*ny-1]
-xv[nlocal+0+ny+nx*ny+ny+iy+(nz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+iy+(nz-1)*ny+1]
;}
else if(ix==0&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+iy-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+iy]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+iy+1]
;}
else if(iz==0 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+(iy-1)*nx]
-xv[nlocal+0+ny+(iy-1)*nx+1]
-xv[nlocal+0+ny+(iy)*nx]
-xv[nlocal+0+ny+(iy)*nx+1]
-xv[nlocal+0+ny+(iy+1)*nx]
-xv[nlocal+0+ny+(iy+1)*nx+1]
;}else if(iz==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+ny+(iy)*nx+nx-2]
-xv[nlocal+0+ny+(iy)*nx+nx-1]
-xv[nlocal+0+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+ny+(iy+1)*nx+nx-1]
;}
else if(iz==nz-1 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+(iy-1)*nx]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+(iy-1)*nx+1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+(iy)*nx]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+(iy)*nx+1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+(iy+1)*nx]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+(iy+1)*nx+1]
;}else if(iz==nz-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+(iy)*nx+nx-2]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+(iy)*nx+nx-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+(iy+1)*nx+nx-1]
;}
else if(ix==nx-1&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+iy-1]
-xv[nlocal+0+ny+nx*ny+iy]
-xv[nlocal+0+ny+nx*ny+iy+1]
;}
else if(ix==nx-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy+1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny+iy-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny+iy]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny+iy+1]
;}else if(ix==nx-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy+(nz-2)*ny]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy+(nz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy+(nz-1)*ny+1]
;}
else if(ix==nx-1&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+nx*ny+iy-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+nx*ny+iy]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz+ny+nx*ny+iy+1]
;}
else if(ix==0&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+(iz-1)*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+(iz)*ny]
-xv[nlocal+0+ny+nx*ny+ny+(iz)*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+(iz+1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+(iz+1)*ny+1]
;}else if(ix==0&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+(iz-1)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+(iz-1)*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+(iz)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+(iz)*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+(iz+1)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+(iz+1)*ny+ny-1]
;}
else if(ix==nx-1&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+(iz-1)*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+(iz)*ny]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+(iz)*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+(iz+1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+(iz+1)*ny+1]
;}else if(ix==nx-1&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+(iz)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+(iz)*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+(iz+1)*ny+ny-1]
;}
else if (ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
            else
{
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+ny+nx*ny+ny]
-xv[nlocal+0+ny+nx*ny+ny+1]
-xv[nlocal+0+ny+nx*ny+ny+ny]
-xv[nlocal+0+ny+nx*ny+ny+ny+1]
-xv[nlocal+0+ny]
-xv[nlocal+0+ny+1]
-xv[nlocal+0+ny+nx]
-xv[nlocal+0+ny+nx+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx*ny+ny+ny*(nz-2)]
-xv[nlocal+0+ny+nx*ny+ny+ny*(nz-2)+1]
-xv[nlocal+0+ny+nx*ny+ny+ny*(nz-1)]
-xv[nlocal+0+ny+nx*ny+ny+ny*(nz-1)+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny-2]
-xv[nlocal+0+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+2*ny-2]
-xv[nlocal+0+ny+nx*ny+ny+2*ny-1]
-xv[nlocal+0+ny+nx*(ny-2)]
-xv[nlocal+0+ny+nx*(ny-2)+1]
-xv[nlocal+0+ny+nx*(ny-1)]
-xv[nlocal+0+ny+nx*(ny-1)+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx*ny+ny+ny*(nz-1)-2]
-xv[nlocal+0+ny+nx*ny+ny+ny*(nz-1)-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz-2]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz-1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx-2]
-xv[nlocal+0+ny+nx-1]
-xv[nlocal+0+ny+2*nx-2]
-xv[nlocal+0+ny+2*nx-1]
-xv[nlocal+0+ny+nx*ny]
-xv[nlocal+0+ny+nx*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*(nz-2)]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*(nz-2)+1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*(nz-1)]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*(nz-1)+1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx*(ny-1)-2]
-xv[nlocal+0+ny+nx*(ny-1)-1]
-xv[nlocal+0+ny+nx*ny-2]
-xv[nlocal+0+ny+nx*ny-1]
-xv[nlocal+0+ny+nx*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+2*ny-2]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+2*ny-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*(nz-1)-2]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*(nz-1)-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz-2]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny*nz-1]
;
}
else if(iz==0&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+ix-1]
-xv[nlocal+0+ny+ix]
-xv[nlocal+0+ny+ix+1]
-xv[nlocal+0+ny+nx+ix-1]
-xv[nlocal+0+ny+nx+ix]
-xv[nlocal+0+ny+nx+ix+1]
;}else if(iz==0&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+ny+ix+(ny-2)*nx]
-xv[nlocal+0+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+ny+ix+(ny-1)*nx]
-xv[nlocal+0+ny+ix+(ny-1)*nx+1]
;}
else if(ix==0&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1]
-xv[nlocal+0+iy]
-xv[nlocal+0+iy+1]
;}
else if(ix==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+iy-1]
-xv[nlocal+0+ny+nx*ny+ny+iy]
-xv[nlocal+0+ny+nx*ny+ny+iy+1]
-xv[nlocal+0+ny+nx*ny+ny+ny+iy-1]
-xv[nlocal+0+ny+nx*ny+ny+ny+iy]
-xv[nlocal+0+ny+nx*ny+ny+ny+iy+1]
;}else if(ix==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+iy+(nz-2)*ny-1]
-xv[nlocal+0+ny+nx*ny+ny+iy+(nz-2)*ny]
-xv[nlocal+0+ny+nx*ny+ny+iy+(nz-2)*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+iy+(nz-1)*ny-1]
-xv[nlocal+0+ny+nx*ny+ny+iy+(nz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+iy+(nz-1)*ny+1]
;}
else if(iz==0 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+(iy-1)*nx]
-xv[nlocal+0+ny+(iy-1)*nx+1]
-xv[nlocal+0+ny+(iy)*nx]
-xv[nlocal+0+ny+(iy)*nx+1]
-xv[nlocal+0+ny+(iy+1)*nx]
-xv[nlocal+0+ny+(iy+1)*nx+1]
;}else if(iz==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+ny+(iy)*nx+nx-2]
-xv[nlocal+0+ny+(iy)*nx+nx-1]
-xv[nlocal+0+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+ny+(iy+1)*nx+nx-1]
;}
else if(ix==nx-1&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+iy-1]
-xv[nlocal+0+ny+nx*ny+iy]
-xv[nlocal+0+ny+nx*ny+iy+1]
;}
else if(ix==nx-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy+1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny+iy-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny+iy]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+ny+iy+1]
;}else if(ix==nx-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy+(nz-2)*ny]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy+(nz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+iy+(nz-1)*ny+1]
;}
else if(ix==0&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+(iz-1)*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+(iz)*ny]
-xv[nlocal+0+ny+nx*ny+ny+(iz)*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+(iz+1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+(iz+1)*ny+1]
;}else if(ix==0&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+(iz-1)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+(iz-1)*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+(iz)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+(iz)*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+(iz+1)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+(iz+1)*ny+ny-1]
;}
else if(ix==nx-1&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+(iz-1)*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+(iz)*ny]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+(iz)*ny+1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+(iz+1)*ny]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+(iz+1)*ny+1]
;}else if(ix==nx-1&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+(iz)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+(iz)*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny+ny*nz+(iz+1)*ny+ny-1]
;}
else if (ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
          }//ipz > 0 
        else
{
            if(ipz < npz - 1)
{
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+ny]
-xv[nlocal+0+ny+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*(nz-2)]
-xv[nlocal+0+ny*(nz-2)+1]
-xv[nlocal+0+ny*(nz-1)]
-xv[nlocal+0+ny*(nz-1)+1]
-xv[nlocal+0+ny*nz+ny*nz]
-xv[nlocal+0+ny*nz+ny*nz+1]
-xv[nlocal+0+ny*nz+ny*nz+ny]
-xv[nlocal+0+ny*nz+ny*nz+ny+1]
-xv[nlocal+0+ny*nz+ny*nz+ny+nx]
-xv[nlocal+0+ny*nz+ny*nz+ny+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny-2]
-xv[nlocal+0+ny-1]
-xv[nlocal+0+2*ny-2]
-xv[nlocal+0+2*ny-1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*(nz-1)-2]
-xv[nlocal+0+ny*(nz-1)-1]
-xv[nlocal+0+ny*nz-2]
-xv[nlocal+0+ny*nz-1]
-xv[nlocal+0+ny*nz+ny*nz+ny-2]
-xv[nlocal+0+ny*nz+ny*nz+ny-1]
-xv[nlocal+0+ny*nz+ny*nz+ny+nx*(ny-2)]
-xv[nlocal+0+ny*nz+ny*nz+ny+nx*(ny-2)+1]
-xv[nlocal+0+ny*nz+ny*nz+ny+nx*(ny-1)]
-xv[nlocal+0+ny*nz+ny*nz+ny+nx*(ny-1)+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*nz]
-xv[nlocal+0+ny*nz+1]
-xv[nlocal+0+ny*nz+ny]
-xv[nlocal+0+ny*nz+ny+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*nz+ny*nz+ny+nx-2]
-xv[nlocal+0+ny*nz+ny*nz+ny+nx-1]
-xv[nlocal+0+ny*nz+ny*nz+ny+2*nx-2]
-xv[nlocal+0+ny*nz+ny*nz+ny+2*nx-1]
-xv[nlocal+0+ny*nz+ny*(nz-2)]
-xv[nlocal+0+ny*nz+ny*(nz-2)+1]
-xv[nlocal+0+ny*nz+ny*(nz-1)]
-xv[nlocal+0+ny*nz+ny*(nz-1)+1]
-xv[nlocal+0+ny*nz+ny*nz+ny+nx*ny]
-xv[nlocal+0+ny*nz+ny*nz+ny+nx*ny+1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*nz+ny-2]
-xv[nlocal+0+ny*nz+ny-1]
-xv[nlocal+0+ny*nz+2*ny-2]
-xv[nlocal+0+ny*nz+2*ny-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*nz+ny*nz+ny+nx*(ny-1)-2]
-xv[nlocal+0+ny*nz+ny*nz+ny+nx*(ny-1)-1]
-xv[nlocal+0+ny*nz+ny*nz+ny+nx*ny-2]
-xv[nlocal+0+ny*nz+ny*nz+ny+nx*ny-1]
-xv[nlocal+0+ny*nz+ny*(nz-1)-2]
-xv[nlocal+0+ny*nz+ny*(nz-1)-1]
-xv[nlocal+0+ny*nz+ny*nz-2]
-xv[nlocal+0+ny*nz+ny*nz-1]
-xv[nlocal+0+ny*nz+ny*nz+ny+nx*ny+ny-2]
-xv[nlocal+0+ny*nz+ny*nz+ny+nx*ny+ny-1]
;
}
else if(iz==nz-1&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+ny+ix-1]
-xv[nlocal+0+ny*nz+ny*nz+ny+ix]
-xv[nlocal+0+ny*nz+ny*nz+ny+ix+1]
-xv[nlocal+0+ny*nz+ny*nz+ny+nx+ix-1]
-xv[nlocal+0+ny*nz+ny*nz+ny+nx+ix]
-xv[nlocal+0+ny*nz+ny*nz+ny+nx+ix+1]
;}else if(iz==nz-1&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+ny*nz+ny*nz+ny+ix+(ny-2)*nx]
-xv[nlocal+0+ny*nz+ny*nz+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+ny*nz+ny*nz+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+ny*nz+ny*nz+ny+ix+(ny-1)*nx]
-xv[nlocal+0+ny*nz+ny*nz+ny+ix+(ny-1)*nx+1]
;}
else if(ix==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1]
-xv[nlocal+0+iy]
-xv[nlocal+0+iy+1]
-xv[nlocal+0+ny+iy-1]
-xv[nlocal+0+ny+iy]
-xv[nlocal+0+ny+iy+1]
;}else if(ix==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy+(nz-2)*ny-1]
-xv[nlocal+0+iy+(nz-2)*ny]
-xv[nlocal+0+iy+(nz-2)*ny+1]
-xv[nlocal+0+iy+(nz-1)*ny-1]
-xv[nlocal+0+iy+(nz-1)*ny]
-xv[nlocal+0+iy+(nz-1)*ny+1]
;}
else if(ix==0&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+iy-1]
-xv[nlocal+0+ny*nz+ny*nz+iy]
-xv[nlocal+0+ny*nz+ny*nz+iy+1]
;}
else if(iz==nz-1 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+ny+(iy-1)*nx]
-xv[nlocal+0+ny*nz+ny*nz+ny+(iy-1)*nx+1]
-xv[nlocal+0+ny*nz+ny*nz+ny+(iy)*nx]
-xv[nlocal+0+ny*nz+ny*nz+ny+(iy)*nx+1]
-xv[nlocal+0+ny*nz+ny*nz+ny+(iy+1)*nx]
-xv[nlocal+0+ny*nz+ny*nz+ny+(iy+1)*nx+1]
;}else if(iz==nz-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+ny*nz+ny*nz+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+ny*nz+ny*nz+ny+(iy)*nx+nx-2]
-xv[nlocal+0+ny*nz+ny*nz+ny+(iy)*nx+nx-1]
-xv[nlocal+0+ny*nz+ny*nz+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+ny*nz+ny*nz+ny+(iy+1)*nx+nx-1]
;}
else if(ix==nx-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+iy-1]
-xv[nlocal+0+ny*nz+iy]
-xv[nlocal+0+ny*nz+iy+1]
-xv[nlocal+0+ny*nz+ny+iy-1]
-xv[nlocal+0+ny*nz+ny+iy]
-xv[nlocal+0+ny*nz+ny+iy+1]
;}else if(ix==nx-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+ny*nz+iy+(nz-2)*ny]
-xv[nlocal+0+ny*nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+ny*nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+ny*nz+iy+(nz-1)*ny]
-xv[nlocal+0+ny*nz+iy+(nz-1)*ny+1]
;}
else if(ix==nx-1&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny*nz+ny+nx*ny+iy-1]
-xv[nlocal+0+ny*nz+ny*nz+ny+nx*ny+iy]
-xv[nlocal+0+ny*nz+ny*nz+ny+nx*ny+iy+1]
;}
else if(ix==0&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*ny]
-xv[nlocal+0+(iz-1)*ny+1]
-xv[nlocal+0+(iz)*ny]
-xv[nlocal+0+(iz)*ny+1]
-xv[nlocal+0+(iz+1)*ny]
-xv[nlocal+0+(iz+1)*ny+1]
;}else if(ix==0&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*ny+ny-2]
-xv[nlocal+0+(iz-1)*ny+ny-1]
-xv[nlocal+0+(iz)*ny+ny-2]
-xv[nlocal+0+(iz)*ny+ny-1]
-xv[nlocal+0+(iz+1)*ny+ny-2]
-xv[nlocal+0+(iz+1)*ny+ny-1]
;}
else if(ix==nx-1&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+(iz-1)*ny]
-xv[nlocal+0+ny*nz+(iz-1)*ny+1]
-xv[nlocal+0+ny*nz+(iz)*ny]
-xv[nlocal+0+ny*nz+(iz)*ny+1]
-xv[nlocal+0+ny*nz+(iz+1)*ny]
-xv[nlocal+0+ny*nz+(iz+1)*ny+1]
;}else if(ix==nx-1&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+ny*nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+ny*nz+(iz)*ny+ny-2]
-xv[nlocal+0+ny*nz+(iz)*ny+ny-1]
-xv[nlocal+0+ny*nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+ny*nz+(iz+1)*ny+ny-1]
;}
else if (ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
            else
{
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+ny]
-xv[nlocal+0+ny+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*(nz-2)]
-xv[nlocal+0+ny*(nz-2)+1]
-xv[nlocal+0+ny*(nz-1)]
-xv[nlocal+0+ny*(nz-1)+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny-2]
-xv[nlocal+0+ny-1]
-xv[nlocal+0+2*ny-2]
-xv[nlocal+0+2*ny-1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*(nz-1)-2]
-xv[nlocal+0+ny*(nz-1)-1]
-xv[nlocal+0+ny*nz-2]
-xv[nlocal+0+ny*nz-1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*nz]
-xv[nlocal+0+ny*nz+1]
-xv[nlocal+0+ny*nz+ny]
-xv[nlocal+0+ny*nz+ny+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*nz+ny*(nz-2)]
-xv[nlocal+0+ny*nz+ny*(nz-2)+1]
-xv[nlocal+0+ny*nz+ny*(nz-1)]
-xv[nlocal+0+ny*nz+ny*(nz-1)+1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*nz+ny-2]
-xv[nlocal+0+ny*nz+ny-1]
-xv[nlocal+0+ny*nz+2*ny-2]
-xv[nlocal+0+ny*nz+2*ny-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*nz+ny*(nz-1)-2]
-xv[nlocal+0+ny*nz+ny*(nz-1)-1]
-xv[nlocal+0+ny*nz+ny*nz-2]
-xv[nlocal+0+ny*nz+ny*nz-1]
;
}
else if(ix==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1]
-xv[nlocal+0+iy]
-xv[nlocal+0+iy+1]
-xv[nlocal+0+ny+iy-1]
-xv[nlocal+0+ny+iy]
-xv[nlocal+0+ny+iy+1]
;}else if(ix==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy+(nz-2)*ny-1]
-xv[nlocal+0+iy+(nz-2)*ny]
-xv[nlocal+0+iy+(nz-2)*ny+1]
-xv[nlocal+0+iy+(nz-1)*ny-1]
-xv[nlocal+0+iy+(nz-1)*ny]
-xv[nlocal+0+iy+(nz-1)*ny+1]
;}
else if(ix==nx-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+iy-1]
-xv[nlocal+0+ny*nz+iy]
-xv[nlocal+0+ny*nz+iy+1]
-xv[nlocal+0+ny*nz+ny+iy-1]
-xv[nlocal+0+ny*nz+ny+iy]
-xv[nlocal+0+ny*nz+ny+iy+1]
;}else if(ix==nx-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+ny*nz+iy+(nz-2)*ny]
-xv[nlocal+0+ny*nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+ny*nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+ny*nz+iy+(nz-1)*ny]
-xv[nlocal+0+ny*nz+iy+(nz-1)*ny+1]
;}
else if(ix==0&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*ny]
-xv[nlocal+0+(iz-1)*ny+1]
-xv[nlocal+0+(iz)*ny]
-xv[nlocal+0+(iz)*ny+1]
-xv[nlocal+0+(iz+1)*ny]
-xv[nlocal+0+(iz+1)*ny+1]
;}else if(ix==0&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*ny+ny-2]
-xv[nlocal+0+(iz-1)*ny+ny-1]
-xv[nlocal+0+(iz)*ny+ny-2]
-xv[nlocal+0+(iz)*ny+ny-1]
-xv[nlocal+0+(iz+1)*ny+ny-2]
-xv[nlocal+0+(iz+1)*ny+ny-1]
;}
else if(ix==nx-1&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+(iz-1)*ny]
-xv[nlocal+0+ny*nz+(iz-1)*ny+1]
-xv[nlocal+0+ny*nz+(iz)*ny]
-xv[nlocal+0+ny*nz+(iz)*ny+1]
-xv[nlocal+0+ny*nz+(iz+1)*ny]
-xv[nlocal+0+ny*nz+(iz+1)*ny+1]
;}else if(ix==nx-1&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+ny*nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+ny*nz+(iz)*ny+ny-2]
-xv[nlocal+0+ny*nz+(iz)*ny+ny-1]
-xv[nlocal+0+ny*nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+ny*nz+(iz+1)*ny+ny-1]
;}
else if (ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+1]
-xv[nlocal+0+1+nx]
-xv[nlocal+0+1+nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny+1]
-xv[nlocal+0+1]
-xv[nlocal+0+1+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx+1]
-xv[nlocal+0+1+nx+ny]
-xv[nlocal+0+1+nx+ny+1]
-xv[nlocal+0+1+nx+ny+nx]
-xv[nlocal+0+1+nx+ny+nx+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*(nz-2)]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*(nz-2)+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*(nz-1)]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*(nz-1)+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*(nz-2)]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*(nz-2)+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*(nz-1)]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*(nz-1)+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+ny-2]
-xv[nlocal+0+1+nx+ny-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+2*ny-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+2*ny-1]
-xv[nlocal+0+1+nx+ny+nx*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+1]
-xv[nlocal+0+1+nx+ny+nx*(ny-2)]
-xv[nlocal+0+1+nx+ny+nx*(ny-2)+1]
-xv[nlocal+0+1+nx+ny+nx*(ny-1)]
-xv[nlocal+0+1+nx+ny+nx*(ny-1)+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*(nz-1)-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*(nz-1)-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx*(ny-2)]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx*(ny-2)+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx*(ny-1)]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx*(ny-1)+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*(nz-2)]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*(nz-2)+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*(nz-1)]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*(nz-1)+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx*ny+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx*ny+1+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx-2]
-xv[nlocal+0+1+nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+2*nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+2*nx-1]
-xv[nlocal+0+1+nx+ny+nx-2]
-xv[nlocal+0+1+nx+ny+nx-1]
-xv[nlocal+0+1+nx+ny+2*nx-2]
-xv[nlocal+0+1+nx+ny+2*nx-1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*(nz-1)-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*(nz-1)-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+2*nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+2*nx-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+ny+nx*(ny-1)-2]
-xv[nlocal+0+1+nx+ny+nx*(ny-1)-1]
-xv[nlocal+0+1+nx+ny+nx*ny-2]
-xv[nlocal+0+1+nx+ny+nx*ny-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+2*nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+2*nx-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx*(ny-1)-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx*(ny-1)-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx*ny-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx*ny-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*(nz-1)-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*(nz-1)-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx*ny+1+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx*ny+1+nx-1]
;
}
else if(iy==0&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+ix-1]
-xv[nlocal+0+1+ix]
-xv[nlocal+0+1+ix+1]
;}
else if(iy==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx+ix-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx+ix]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx+ix+1]
;}else if(iy==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix+(nz-2)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix+(nz-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix+(nz-1)*nx+1]
;}
else if(iy==0&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+ix-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+ix]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+ix+1]
;}
else if(iz==0&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+ix-1]
-xv[nlocal+0+1+nx+ny+ix]
-xv[nlocal+0+1+nx+ny+ix+1]
-xv[nlocal+0+1+nx+ny+nx+ix-1]
-xv[nlocal+0+1+nx+ny+nx+ix]
-xv[nlocal+0+1+nx+ny+nx+ix+1]
;}else if(iz==0&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+1+nx+ny+ix+(ny-2)*nx]
-xv[nlocal+0+1+nx+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+1+nx+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+1+nx+ny+ix+(ny-1)*nx]
-xv[nlocal+0+1+nx+ny+ix+(ny-1)*nx+1]
;}
else if(iz==nz-1&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx+ix-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx+ix]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx+ix+1]
;}else if(iz==nz-1&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix+(ny-2)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix+(ny-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix+(ny-1)*nx+1]
;}
else if(iy==ny-1&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+ix-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+ix]
-xv[nlocal+0+1+nx+ny+nx*ny+1+ix+1]
;}
else if(iy==ny-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx+ix-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx+ix]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx+ix+1]
;}else if(iy==ny-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix+(nz-2)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix+(nz-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix+(nz-1)*nx+1]
;}
else if(iy==ny-1&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx*ny+1+ix-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx*ny+1+ix]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx*ny+1+ix+1]
;}
else if(ix==0&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+iy-1]
-xv[nlocal+0+1+nx+iy]
-xv[nlocal+0+1+nx+iy+1]
;}
else if(ix==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny+iy-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny+iy]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny+iy+1]
;}else if(ix==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy+(nz-2)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy+(nz-1)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy+(nz-1)*ny+1]
;}
else if(ix==0&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+iy-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+iy]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+iy+1]
;}
else if(iz==0 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+(iy-1)*nx]
-xv[nlocal+0+1+nx+ny+(iy-1)*nx+1]
-xv[nlocal+0+1+nx+ny+(iy)*nx]
-xv[nlocal+0+1+nx+ny+(iy)*nx+1]
-xv[nlocal+0+1+nx+ny+(iy+1)*nx]
-xv[nlocal+0+1+nx+ny+(iy+1)*nx+1]
;}else if(iz==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+1+nx+ny+(iy)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+(iy)*nx+nx-1]
-xv[nlocal+0+1+nx+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+(iy+1)*nx+nx-1]
;}
else if(iz==nz-1 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+(iy-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+(iy-1)*nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+(iy)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+(iy)*nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+(iy+1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+(iy+1)*nx+1]
;}else if(iz==nz-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+(iy)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+(iy)*nx+nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+(iy+1)*nx+nx-1]
;}
else if(ix==0&&iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+iz-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+iz]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+iz+1]
;}
else if(ix==0&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+(iz-1)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+(iz-1)*ny+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+(iz)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+(iz)*ny+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+(iz+1)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+(iz+1)*ny+1]
;}else if(ix==0&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+(iz)*ny+ny-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+(iz)*ny+ny-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+(iz+1)*ny+ny-1]
;}
else if(ix==0&&iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+iz-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+iz]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+iz+1]
;}
else if(iy==0&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+(iz-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+(iz-1)*nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+(iz)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+(iz)*nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+(iz+1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+(iz+1)*nx+1]
;}else if(iy==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+(iz)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+(iz)*nx+nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+(iz+1)*nx+nx-1]
;}
else if(iy==ny-1&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+(iz-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+(iz-1)*nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+(iz)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+(iz)*nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+(iz+1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+(iz+1)*nx+1]
;}else if(iy==ny-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+(iz)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+(iz)*nx+nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+(iz+1)*nx+nx-1]
;}
else if (ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
            else
{
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+1]
-xv[nlocal+0+1+nx]
-xv[nlocal+0+1+nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny+1]
-xv[nlocal+0+1]
-xv[nlocal+0+1+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx+1]
-xv[nlocal+0+1+nx+ny]
-xv[nlocal+0+1+nx+ny+1]
-xv[nlocal+0+1+nx+ny+nx]
-xv[nlocal+0+1+nx+ny+nx+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*(nz-2)]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*(nz-2)+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*(nz-1)]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*(nz-1)+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*(nz-2)]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*(nz-2)+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*(nz-1)]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*(nz-1)+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+ny-2]
-xv[nlocal+0+1+nx+ny-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+2*ny-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+2*ny-1]
-xv[nlocal+0+1+nx+ny+nx*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+1]
-xv[nlocal+0+1+nx+ny+nx*(ny-2)]
-xv[nlocal+0+1+nx+ny+nx*(ny-2)+1]
-xv[nlocal+0+1+nx+ny+nx*(ny-1)]
-xv[nlocal+0+1+nx+ny+nx*(ny-1)+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*(nz-1)-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*(nz-1)-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*(nz-2)]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*(nz-2)+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*(nz-1)]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*(nz-1)+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx-2]
-xv[nlocal+0+1+nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+2*nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+2*nx-1]
-xv[nlocal+0+1+nx+ny+nx-2]
-xv[nlocal+0+1+nx+ny+nx-1]
-xv[nlocal+0+1+nx+ny+2*nx-2]
-xv[nlocal+0+1+nx+ny+2*nx-1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*(nz-1)-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*(nz-1)-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+ny+nx*(ny-1)-2]
-xv[nlocal+0+1+nx+ny+nx*(ny-1)-1]
-xv[nlocal+0+1+nx+ny+nx*ny-2]
-xv[nlocal+0+1+nx+ny+nx*ny-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+2*nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+2*nx-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*(nz-1)-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*(nz-1)-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx*nz-1]
;
}
else if(iy==0&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+ix-1]
-xv[nlocal+0+1+ix]
-xv[nlocal+0+1+ix+1]
;}
else if(iy==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx+ix-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx+ix]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx+ix+1]
;}else if(iy==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix+(nz-2)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix+(nz-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+ix+(nz-1)*nx+1]
;}
else if(iz==0&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+ix-1]
-xv[nlocal+0+1+nx+ny+ix]
-xv[nlocal+0+1+nx+ny+ix+1]
-xv[nlocal+0+1+nx+ny+nx+ix-1]
-xv[nlocal+0+1+nx+ny+nx+ix]
-xv[nlocal+0+1+nx+ny+nx+ix+1]
;}else if(iz==0&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+1+nx+ny+ix+(ny-2)*nx]
-xv[nlocal+0+1+nx+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+1+nx+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+1+nx+ny+ix+(ny-1)*nx]
-xv[nlocal+0+1+nx+ny+ix+(ny-1)*nx+1]
;}
else if(iy==ny-1&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+ix-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+ix]
-xv[nlocal+0+1+nx+ny+nx*ny+1+ix+1]
;}
else if(iy==ny-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx+ix-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx+ix]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+nx+ix+1]
;}else if(iy==ny-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix+(nz-2)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix+(nz-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+ix+(nz-1)*nx+1]
;}
else if(ix==0&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+iy-1]
-xv[nlocal+0+1+nx+iy]
-xv[nlocal+0+1+nx+iy+1]
;}
else if(ix==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny+iy-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny+iy]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny+iy+1]
;}else if(ix==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy+(nz-2)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy+(nz-1)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+iy+(nz-1)*ny+1]
;}
else if(iz==0 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+(iy-1)*nx]
-xv[nlocal+0+1+nx+ny+(iy-1)*nx+1]
-xv[nlocal+0+1+nx+ny+(iy)*nx]
-xv[nlocal+0+1+nx+ny+(iy)*nx+1]
-xv[nlocal+0+1+nx+ny+(iy+1)*nx]
-xv[nlocal+0+1+nx+ny+(iy+1)*nx+1]
;}else if(iz==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+1+nx+ny+(iy)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+(iy)*nx+nx-1]
-xv[nlocal+0+1+nx+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+(iy+1)*nx+nx-1]
;}
else if(ix==0&&iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+iz-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+iz]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+iz+1]
;}
else if(ix==0&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+(iz-1)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+(iz-1)*ny+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+(iz)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+(iz)*ny+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+(iz+1)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+(iz+1)*ny+1]
;}else if(ix==0&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+(iz)*ny+ny-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+(iz)*ny+ny-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+(iz+1)*ny+ny-1]
;}
else if(ix==0&&iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+iz-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+iz]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+iz+1]
;}
else if(iy==0&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+(iz-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+(iz-1)*nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+(iz)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+(iz)*nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+(iz+1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+(iz+1)*nx+1]
;}else if(iy==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+(iz)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+(iz)*nx+nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+(iz+1)*nx+nx-1]
;}
else if(iy==ny-1&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+(iz-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+(iz-1)*nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+(iz)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+(iz)*nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+(iz+1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+(iz+1)*nx+1]
;}else if(iy==ny-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+(iz)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+(iz)*nx+nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+1+nx+nz+nx*nz+ny*nz+nz+(iz+1)*nx+nx-1]
;}
else if (ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
          }//ipz > 0 
        else
{
            if(ipz < npz - 1)
{
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+nz+nx*nz]
-xv[nlocal+0+nz+nx*nz+1]
-xv[nlocal+0+nz+nx*nz+ny]
-xv[nlocal+0+nz+nx*nz+ny+1]
-xv[nlocal+0+nz]
-xv[nlocal+0+nz+1]
-xv[nlocal+0+nz+nx]
-xv[nlocal+0+nz+nx+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz-2]
-xv[nlocal+0+nz-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz]
-xv[nlocal+0+nz+nx*nz+ny*(nz-2)]
-xv[nlocal+0+nz+nx*nz+ny*(nz-2)+1]
-xv[nlocal+0+nz+nx*nz+ny*(nz-1)]
-xv[nlocal+0+nz+nx*nz+ny*(nz-1)+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+1]
-xv[nlocal+0+nz+nx*(nz-2)]
-xv[nlocal+0+nz+nx*(nz-2)+1]
-xv[nlocal+0+nz+nx*(nz-1)]
-xv[nlocal+0+nz+nx*(nz-1)+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx*nz+ny-2]
-xv[nlocal+0+nz+nx*nz+ny-1]
-xv[nlocal+0+nz+nx*nz+2*ny-2]
-xv[nlocal+0+nz+nx*nz+2*ny-1]
-xv[nlocal+0+nz+nx*nz+ny*nz]
-xv[nlocal+0+nz+nx*nz+ny*nz+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx*nz+ny*(nz-1)-2]
-xv[nlocal+0+nz+nx*nz+ny*(nz-1)-1]
-xv[nlocal+0+nz+nx*nz+ny*nz-2]
-xv[nlocal+0+nz+nx*nz+ny*nz-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny-2]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz-2]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx*ny]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx*(ny-2)]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx*(ny-2)+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx*(ny-1)]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx*(ny-1)+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*(nz-2)]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*(nz-2)+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*(nz-1)]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*(nz-1)+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx*ny+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx*ny+1+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx-2]
-xv[nlocal+0+nz+nx-1]
-xv[nlocal+0+nz+2*nx-2]
-xv[nlocal+0+nz+2*nx-1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx*(nz-1)-2]
-xv[nlocal+0+nz+nx*(nz-1)-1]
-xv[nlocal+0+nz+nx*nz-2]
-xv[nlocal+0+nz+nx*nz-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx-2]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx-2]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+2*nx-2]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+2*nx-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx-2]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+2*nx-2]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+2*nx-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx*(ny-1)-2]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx*(ny-1)-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx*ny-2]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx*ny-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*(nz-1)-2]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*(nz-1)-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz-2]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx*ny+1+nx-2]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx*ny+1+nx-1]
;
}
else if(iy==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+ix-1]
-xv[nlocal+0+nz+ix]
-xv[nlocal+0+nz+ix+1]
-xv[nlocal+0+nz+nx+ix-1]
-xv[nlocal+0+nz+nx+ix]
-xv[nlocal+0+nz+nx+ix+1]
;}else if(iy==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+nz+ix+(nz-2)*nx]
-xv[nlocal+0+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+nz+ix+(nz-1)*nx]
-xv[nlocal+0+nz+ix+(nz-1)*nx+1]
;}
else if(iy==0&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+ix-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+ix]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+ix+1]
;}
else if(iz==nz-1&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx+ix-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx+ix]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx+ix+1]
;}else if(iz==nz-1&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix+(ny-2)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix+(ny-1)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+ix+(ny-1)*nx+1]
;}
else if(iy==ny-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx+ix-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx+ix]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx+ix+1]
;}else if(iy==ny-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix+(nz-2)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix+(nz-1)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix+(nz-1)*nx+1]
;}
else if(iy==ny-1&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx*ny+1+ix-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx*ny+1+ix]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+nx*ny+1+ix+1]
;}
else if(ix==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+iy-1]
-xv[nlocal+0+nz+nx*nz+iy]
-xv[nlocal+0+nz+nx*nz+iy+1]
-xv[nlocal+0+nz+nx*nz+ny+iy-1]
-xv[nlocal+0+nz+nx*nz+ny+iy]
-xv[nlocal+0+nz+nx*nz+ny+iy+1]
;}else if(ix==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+nz+nx*nz+iy+(nz-2)*ny]
-xv[nlocal+0+nz+nx*nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+nz+nx*nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+nz+nx*nz+iy+(nz-1)*ny]
-xv[nlocal+0+nz+nx*nz+iy+(nz-1)*ny+1]
;}
else if(ix==0&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+iy-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+iy]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+iy+1]
;}
else if(iz==nz-1 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+(iy-1)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+(iy-1)*nx+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+(iy)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+(iy)*nx+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+(iy+1)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+(iy+1)*nx+1]
;}else if(iz==nz-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+(iy)*nx+nx-2]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+(iy)*nx+nx-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz+1+nx+ny+(iy+1)*nx+nx-1]
;}
else if(ix==0&&iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iz-1]
-xv[nlocal+0+iz]
-xv[nlocal+0+iz+1]
;}
else if(ix==0&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+(iz-1)*ny+1]
-xv[nlocal+0+nz+nx*nz+(iz)*ny]
-xv[nlocal+0+nz+nx*nz+(iz)*ny+1]
-xv[nlocal+0+nz+nx*nz+(iz+1)*ny]
-xv[nlocal+0+nz+nx*nz+(iz+1)*ny+1]
;}else if(ix==0&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+nz+nx*nz+(iz)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+(iz)*ny+ny-1]
-xv[nlocal+0+nz+nx*nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+(iz+1)*ny+ny-1]
;}
else if(ix==0&&iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+ny*nz+iz-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+iz]
-xv[nlocal+0+nz+nx*nz+ny*nz+iz+1]
;}
else if(iy==0&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+(iz-1)*nx]
-xv[nlocal+0+nz+(iz-1)*nx+1]
-xv[nlocal+0+nz+(iz)*nx]
-xv[nlocal+0+nz+(iz)*nx+1]
-xv[nlocal+0+nz+(iz+1)*nx]
-xv[nlocal+0+nz+(iz+1)*nx+1]
;}else if(iy==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+nz+(iz)*nx+nx-2]
-xv[nlocal+0+nz+(iz)*nx+nx-1]
-xv[nlocal+0+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+nz+(iz+1)*nx+nx-1]
;}
else if(iy==ny-1&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+(iz-1)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+(iz-1)*nx+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+(iz)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+(iz)*nx+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+(iz+1)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+(iz+1)*nx+1]
;}else if(iy==ny-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+(iz)*nx+nx-2]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+(iz)*nx+nx-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+(iz+1)*nx+nx-1]
;}
else if (ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
            else
{
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+nz+nx*nz]
-xv[nlocal+0+nz+nx*nz+1]
-xv[nlocal+0+nz+nx*nz+ny]
-xv[nlocal+0+nz+nx*nz+ny+1]
-xv[nlocal+0+nz]
-xv[nlocal+0+nz+1]
-xv[nlocal+0+nz+nx]
-xv[nlocal+0+nz+nx+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz-2]
-xv[nlocal+0+nz-1]
-xv[nlocal+0+nz+nx*nz+ny*(nz-2)]
-xv[nlocal+0+nz+nx*nz+ny*(nz-2)+1]
-xv[nlocal+0+nz+nx*nz+ny*(nz-1)]
-xv[nlocal+0+nz+nx*nz+ny*(nz-1)+1]
-xv[nlocal+0+nz+nx*(nz-2)]
-xv[nlocal+0+nz+nx*(nz-2)+1]
-xv[nlocal+0+nz+nx*(nz-1)]
-xv[nlocal+0+nz+nx*(nz-1)+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx*nz+ny-2]
-xv[nlocal+0+nz+nx*nz+ny-1]
-xv[nlocal+0+nz+nx*nz+2*ny-2]
-xv[nlocal+0+nz+nx*nz+2*ny-1]
-xv[nlocal+0+nz+nx*nz+ny*nz]
-xv[nlocal+0+nz+nx*nz+ny*nz+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx*nz+ny*(nz-1)-2]
-xv[nlocal+0+nz+nx*nz+ny*(nz-1)-1]
-xv[nlocal+0+nz+nx*nz+ny*nz-2]
-xv[nlocal+0+nz+nx*nz+ny*nz-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz-2]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*(nz-2)]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*(nz-2)+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*(nz-1)]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*(nz-1)+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx-2]
-xv[nlocal+0+nz+nx-1]
-xv[nlocal+0+nz+2*nx-2]
-xv[nlocal+0+nz+2*nx-1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx*(nz-1)-2]
-xv[nlocal+0+nz+nx*(nz-1)-1]
-xv[nlocal+0+nz+nx*nz-2]
-xv[nlocal+0+nz+nx*nz-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx-2]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+2*nx-2]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+2*nx-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*(nz-1)-2]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*(nz-1)-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz-2]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx*nz-1]
;
}
else if(iy==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+ix-1]
-xv[nlocal+0+nz+ix]
-xv[nlocal+0+nz+ix+1]
-xv[nlocal+0+nz+nx+ix-1]
-xv[nlocal+0+nz+nx+ix]
-xv[nlocal+0+nz+nx+ix+1]
;}else if(iy==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+nz+ix+(nz-2)*nx]
-xv[nlocal+0+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+nz+ix+(nz-1)*nx]
-xv[nlocal+0+nz+ix+(nz-1)*nx+1]
;}
else if(iy==ny-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx+ix-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx+ix]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+nx+ix+1]
;}else if(iy==ny-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix+(nz-2)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix+(nz-1)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+ix+(nz-1)*nx+1]
;}
else if(ix==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+iy-1]
-xv[nlocal+0+nz+nx*nz+iy]
-xv[nlocal+0+nz+nx*nz+iy+1]
-xv[nlocal+0+nz+nx*nz+ny+iy-1]
-xv[nlocal+0+nz+nx*nz+ny+iy]
-xv[nlocal+0+nz+nx*nz+ny+iy+1]
;}else if(ix==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+nz+nx*nz+iy+(nz-2)*ny]
-xv[nlocal+0+nz+nx*nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+nz+nx*nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+nz+nx*nz+iy+(nz-1)*ny]
-xv[nlocal+0+nz+nx*nz+iy+(nz-1)*ny+1]
;}
else if(ix==0&&iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iz-1]
-xv[nlocal+0+iz]
-xv[nlocal+0+iz+1]
;}
else if(ix==0&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+(iz-1)*ny+1]
-xv[nlocal+0+nz+nx*nz+(iz)*ny]
-xv[nlocal+0+nz+nx*nz+(iz)*ny+1]
-xv[nlocal+0+nz+nx*nz+(iz+1)*ny]
-xv[nlocal+0+nz+nx*nz+(iz+1)*ny+1]
;}else if(ix==0&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+nz+nx*nz+(iz)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+(iz)*ny+ny-1]
-xv[nlocal+0+nz+nx*nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+(iz+1)*ny+ny-1]
;}
else if(ix==0&&iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+ny*nz+iz-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+iz]
-xv[nlocal+0+nz+nx*nz+ny*nz+iz+1]
;}
else if(iy==0&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+(iz-1)*nx]
-xv[nlocal+0+nz+(iz-1)*nx+1]
-xv[nlocal+0+nz+(iz)*nx]
-xv[nlocal+0+nz+(iz)*nx+1]
-xv[nlocal+0+nz+(iz+1)*nx]
-xv[nlocal+0+nz+(iz+1)*nx+1]
;}else if(iy==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+nz+(iz)*nx+nx-2]
-xv[nlocal+0+nz+(iz)*nx+nx-1]
-xv[nlocal+0+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+nz+(iz+1)*nx+nx-1]
;}
else if(iy==ny-1&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+(iz-1)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+(iz-1)*nx+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+(iz)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+(iz)*nx+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+(iz+1)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+(iz+1)*nx+1]
;}else if(iy==ny-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+(iz)*nx+nx-2]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+(iz)*nx+nx-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+nz+nx*nz+ny*nz+nz+(iz+1)*nx+nx-1]
;}
else if (ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
          }//ipz > 0 
        }//ipy < npy - 1 
      else
{
        if(ipz > 0)
{
            if(ipz < npz - 1)
{
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1+nx+ny+nx*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+1]
-xv[nlocal+0+1+nx]
-xv[nlocal+0+1+nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny+1]
-xv[nlocal+0+1]
-xv[nlocal+0+1+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx+1]
-xv[nlocal+0+1+nx+ny]
-xv[nlocal+0+1+nx+ny+1]
-xv[nlocal+0+1+nx+ny+nx]
-xv[nlocal+0+1+nx+ny+nx+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+ny+nx*ny+nz-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*(nz-2)]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*(nz-2)+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*(nz-1)]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*(nz-1)+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*(nz-2)]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*(nz-2)+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*(nz-1)]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*(nz-1)+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+ny-2]
-xv[nlocal+0+1+nx+ny-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+2*ny-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+2*ny-1]
-xv[nlocal+0+1+nx+ny+nx*(ny-2)]
-xv[nlocal+0+1+nx+ny+nx*(ny-2)+1]
-xv[nlocal+0+1+nx+ny+nx*(ny-1)]
-xv[nlocal+0+1+nx+ny+nx*(ny-1)+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*(nz-1)-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*(nz-1)-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+nx*(ny-2)]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+nx*(ny-2)+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+nx*(ny-1)]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+nx*(ny-1)+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx-2]
-xv[nlocal+0+1+nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+2*nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+2*nx-1]
-xv[nlocal+0+1+nx+ny+nx-2]
-xv[nlocal+0+1+nx+ny+nx-1]
-xv[nlocal+0+1+nx+ny+2*nx-2]
-xv[nlocal+0+1+nx+ny+2*nx-1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*(nz-1)-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*(nz-1)-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+2*nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+2*nx-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+ny+nx*(ny-1)-2]
-xv[nlocal+0+1+nx+ny+nx*(ny-1)-1]
-xv[nlocal+0+1+nx+ny+nx*ny-2]
-xv[nlocal+0+1+nx+ny+nx*ny-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+nx*(ny-1)-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+nx*(ny-1)-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+nx*ny-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+nx*ny-1]
;
}
else if(iy==0&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+ix-1]
-xv[nlocal+0+1+ix]
-xv[nlocal+0+1+ix+1]
;}
else if(iy==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx+ix-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx+ix]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx+ix+1]
;}else if(iy==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix+(nz-2)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix+(nz-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix+(nz-1)*nx+1]
;}
else if(iy==0&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+ix-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+ix]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+ix+1]
;}
else if(iz==0&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+ix-1]
-xv[nlocal+0+1+nx+ny+ix]
-xv[nlocal+0+1+nx+ny+ix+1]
-xv[nlocal+0+1+nx+ny+nx+ix-1]
-xv[nlocal+0+1+nx+ny+nx+ix]
-xv[nlocal+0+1+nx+ny+nx+ix+1]
;}else if(iz==0&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+1+nx+ny+ix+(ny-2)*nx]
-xv[nlocal+0+1+nx+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+1+nx+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+1+nx+ny+ix+(ny-1)*nx]
-xv[nlocal+0+1+nx+ny+ix+(ny-1)*nx+1]
;}
else if(iz==nz-1&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+ix-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+ix]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+ix+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+nx+ix-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+nx+ix]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+nx+ix+1]
;}else if(iz==nz-1&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+ix+(ny-2)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+ix+(ny-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+ix+(ny-1)*nx+1]
;}
else if(ix==0&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+iy-1]
-xv[nlocal+0+1+nx+iy]
-xv[nlocal+0+1+nx+iy+1]
;}
else if(ix==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny+iy-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny+iy]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny+iy+1]
;}else if(ix==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy+(nz-2)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy+(nz-1)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy+(nz-1)*ny+1]
;}
else if(ix==0&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+iy-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+iy]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+iy+1]
;}
else if(iz==0 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+(iy-1)*nx]
-xv[nlocal+0+1+nx+ny+(iy-1)*nx+1]
-xv[nlocal+0+1+nx+ny+(iy)*nx]
-xv[nlocal+0+1+nx+ny+(iy)*nx+1]
-xv[nlocal+0+1+nx+ny+(iy+1)*nx]
-xv[nlocal+0+1+nx+ny+(iy+1)*nx+1]
;}else if(iz==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+1+nx+ny+(iy)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+(iy)*nx+nx-1]
-xv[nlocal+0+1+nx+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+(iy+1)*nx+nx-1]
;}
else if(iz==nz-1 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+(iy-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+(iy-1)*nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+(iy)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+(iy)*nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+(iy+1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+(iy+1)*nx+1]
;}else if(iz==nz-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+(iy)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+(iy)*nx+nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz+1+nx+ny+(iy+1)*nx+nx-1]
;}
else if(ix==0&&iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+iz-1]
-xv[nlocal+0+1+nx+ny+nx*ny+iz]
-xv[nlocal+0+1+nx+ny+nx*ny+iz+1]
;}
else if(ix==0&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+(iz-1)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+(iz-1)*ny+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+(iz)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+(iz)*ny+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+(iz+1)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+(iz+1)*ny+1]
;}else if(ix==0&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+(iz)*ny+ny-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+(iz)*ny+ny-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+(iz+1)*ny+ny-1]
;}
else if(iy==0&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+nz+(iz-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+(iz-1)*nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+(iz)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+(iz)*nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+(iz+1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+(iz+1)*nx+1]
;}else if(iy==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+(iz)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+(iz)*nx+nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+(iz+1)*nx+nx-1]
;}
else if (ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
            else
{
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1+nx+ny+nx*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+1]
-xv[nlocal+0+1+nx]
-xv[nlocal+0+1+nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny+1]
-xv[nlocal+0+1]
-xv[nlocal+0+1+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx+1]
-xv[nlocal+0+1+nx+ny]
-xv[nlocal+0+1+nx+ny+1]
-xv[nlocal+0+1+nx+ny+nx]
-xv[nlocal+0+1+nx+ny+nx+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+ny+nx*ny+nz-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*(nz-2)]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*(nz-2)+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*(nz-1)]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*(nz-1)+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*(nz-2)]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*(nz-2)+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*(nz-1)]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*(nz-1)+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+ny-2]
-xv[nlocal+0+1+nx+ny-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+2*ny-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+2*ny-1]
-xv[nlocal+0+1+nx+ny+nx*(ny-2)]
-xv[nlocal+0+1+nx+ny+nx*(ny-2)+1]
-xv[nlocal+0+1+nx+ny+nx*(ny-1)]
-xv[nlocal+0+1+nx+ny+nx*(ny-1)+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*(nz-1)-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*(nz-1)-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny*nz-1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx-2]
-xv[nlocal+0+1+nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+2*nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+2*nx-1]
-xv[nlocal+0+1+nx+ny+nx-2]
-xv[nlocal+0+1+nx+ny+nx-1]
-xv[nlocal+0+1+nx+ny+2*nx-2]
-xv[nlocal+0+1+nx+ny+2*nx-1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*(nz-1)-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*(nz-1)-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+1+nx+ny+nx*(ny-1)-2]
-xv[nlocal+0+1+nx+ny+nx*(ny-1)-1]
-xv[nlocal+0+1+nx+ny+nx*ny-2]
-xv[nlocal+0+1+nx+ny+nx*ny-1]
;
}
else if(iy==0&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+ix-1]
-xv[nlocal+0+1+ix]
-xv[nlocal+0+1+ix+1]
;}
else if(iy==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx+ix-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx+ix]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx+ix+1]
;}else if(iy==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix+(nz-2)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix+(nz-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+ix+(nz-1)*nx+1]
;}
else if(iz==0&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+ix-1]
-xv[nlocal+0+1+nx+ny+ix]
-xv[nlocal+0+1+nx+ny+ix+1]
-xv[nlocal+0+1+nx+ny+nx+ix-1]
-xv[nlocal+0+1+nx+ny+nx+ix]
-xv[nlocal+0+1+nx+ny+nx+ix+1]
;}else if(iz==0&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+1+nx+ny+ix+(ny-2)*nx]
-xv[nlocal+0+1+nx+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+1+nx+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+1+nx+ny+ix+(ny-1)*nx]
-xv[nlocal+0+1+nx+ny+ix+(ny-1)*nx+1]
;}
else if(ix==0&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+iy-1]
-xv[nlocal+0+1+nx+iy]
-xv[nlocal+0+1+nx+iy+1]
;}
else if(ix==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny+iy-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny+iy]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+ny+iy+1]
;}else if(ix==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy+(nz-2)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy+(nz-1)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+iy+(nz-1)*ny+1]
;}
else if(iz==0 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+(iy-1)*nx]
-xv[nlocal+0+1+nx+ny+(iy-1)*nx+1]
-xv[nlocal+0+1+nx+ny+(iy)*nx]
-xv[nlocal+0+1+nx+ny+(iy)*nx+1]
-xv[nlocal+0+1+nx+ny+(iy+1)*nx]
-xv[nlocal+0+1+nx+ny+(iy+1)*nx+1]
;}else if(iz==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+1+nx+ny+(iy)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+(iy)*nx+nx-1]
-xv[nlocal+0+1+nx+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+(iy+1)*nx+nx-1]
;}
else if(ix==0&&iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+iz-1]
-xv[nlocal+0+1+nx+ny+nx*ny+iz]
-xv[nlocal+0+1+nx+ny+nx*ny+iz+1]
;}
else if(ix==0&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+(iz-1)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+(iz-1)*ny+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+(iz)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+(iz)*ny+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+(iz+1)*ny]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+(iz+1)*ny+1]
;}else if(ix==0&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+(iz)*ny+ny-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+(iz)*ny+ny-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+nx*nz+(iz+1)*ny+ny-1]
;}
else if(iy==0&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+nz+(iz-1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+(iz-1)*nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+(iz)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+(iz)*nx+1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+(iz+1)*nx]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+(iz+1)*nx+1]
;}else if(iy==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+1+nx+ny+nx*ny+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+(iz)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+(iz)*nx+nx-1]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+1+nx+ny+nx*ny+nz+(iz+1)*nx+nx-1]
;}
else if (ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
          }//ipz > 0 
        else
{
            if(ipz < npz - 1)
{
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+nz+nx*nz]
-xv[nlocal+0+nz+nx*nz+1]
-xv[nlocal+0+nz+nx*nz+ny]
-xv[nlocal+0+nz+nx*nz+ny+1]
-xv[nlocal+0+nz]
-xv[nlocal+0+nz+1]
-xv[nlocal+0+nz+nx]
-xv[nlocal+0+nz+nx+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz-2]
-xv[nlocal+0+nz-1]
-xv[nlocal+0+nz+nx*nz+ny*nz]
-xv[nlocal+0+nz+nx*nz+ny*(nz-2)]
-xv[nlocal+0+nz+nx*nz+ny*(nz-2)+1]
-xv[nlocal+0+nz+nx*nz+ny*(nz-1)]
-xv[nlocal+0+nz+nx*nz+ny*(nz-1)+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+1]
-xv[nlocal+0+nz+nx*(nz-2)]
-xv[nlocal+0+nz+nx*(nz-2)+1]
-xv[nlocal+0+nz+nx*(nz-1)]
-xv[nlocal+0+nz+nx*(nz-1)+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx*nz+ny-2]
-xv[nlocal+0+nz+nx*nz+ny-1]
-xv[nlocal+0+nz+nx*nz+2*ny-2]
-xv[nlocal+0+nz+nx*nz+2*ny-1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx*nz+ny*(nz-1)-2]
-xv[nlocal+0+nz+nx*nz+ny*(nz-1)-1]
-xv[nlocal+0+nz+nx*nz+ny*nz-2]
-xv[nlocal+0+nz+nx*nz+ny*nz-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny-2]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+nx*(ny-2)]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+nx*(ny-2)+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+nx*(ny-1)]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+nx*(ny-1)+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx-2]
-xv[nlocal+0+nz+nx-1]
-xv[nlocal+0+nz+2*nx-2]
-xv[nlocal+0+nz+2*nx-1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx*(nz-1)-2]
-xv[nlocal+0+nz+nx*(nz-1)-1]
-xv[nlocal+0+nz+nx*nz-2]
-xv[nlocal+0+nz+nx*nz-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx-2]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+nx-2]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+nx-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+2*nx-2]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+2*nx-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+nx*(ny-1)-2]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+nx*(ny-1)-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+nx*ny-2]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+nx*ny-1]
;
}
else if(iy==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+ix-1]
-xv[nlocal+0+nz+ix]
-xv[nlocal+0+nz+ix+1]
-xv[nlocal+0+nz+nx+ix-1]
-xv[nlocal+0+nz+nx+ix]
-xv[nlocal+0+nz+nx+ix+1]
;}else if(iy==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+nz+ix+(nz-2)*nx]
-xv[nlocal+0+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+nz+ix+(nz-1)*nx]
-xv[nlocal+0+nz+ix+(nz-1)*nx+1]
;}
else if(iy==0&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+ny*nz+1+ix-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+ix]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+ix+1]
;}
else if(iz==nz-1&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+ix-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+ix]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+ix+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+nx+ix-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+nx+ix]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+nx+ix+1]
;}else if(iz==nz-1&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+ix+(ny-2)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+ix+(ny-1)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+ix+(ny-1)*nx+1]
;}
else if(ix==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+iy-1]
-xv[nlocal+0+nz+nx*nz+iy]
-xv[nlocal+0+nz+nx*nz+iy+1]
-xv[nlocal+0+nz+nx*nz+ny+iy-1]
-xv[nlocal+0+nz+nx*nz+ny+iy]
-xv[nlocal+0+nz+nx*nz+ny+iy+1]
;}else if(ix==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+nz+nx*nz+iy+(nz-2)*ny]
-xv[nlocal+0+nz+nx*nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+nz+nx*nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+nz+nx*nz+iy+(nz-1)*ny]
-xv[nlocal+0+nz+nx*nz+iy+(nz-1)*ny+1]
;}
else if(ix==0&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+iy-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+iy]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+iy+1]
;}
else if(iz==nz-1 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+(iy-1)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+(iy-1)*nx+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+(iy)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+(iy)*nx+1]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+(iy+1)*nx]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+(iy+1)*nx+1]
;}else if(iz==nz-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+(iy)*nx+nx-2]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+(iy)*nx+nx-1]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+nz+nx*nz+ny*nz+1+nx+ny+(iy+1)*nx+nx-1]
;}
else if(ix==0&&iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iz-1]
-xv[nlocal+0+iz]
-xv[nlocal+0+iz+1]
;}
else if(ix==0&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+(iz-1)*ny+1]
-xv[nlocal+0+nz+nx*nz+(iz)*ny]
-xv[nlocal+0+nz+nx*nz+(iz)*ny+1]
-xv[nlocal+0+nz+nx*nz+(iz+1)*ny]
-xv[nlocal+0+nz+nx*nz+(iz+1)*ny+1]
;}else if(ix==0&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+nz+nx*nz+(iz)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+(iz)*ny+ny-1]
-xv[nlocal+0+nz+nx*nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+(iz+1)*ny+ny-1]
;}
else if(iy==0&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+(iz-1)*nx]
-xv[nlocal+0+nz+(iz-1)*nx+1]
-xv[nlocal+0+nz+(iz)*nx]
-xv[nlocal+0+nz+(iz)*nx+1]
-xv[nlocal+0+nz+(iz+1)*nx]
-xv[nlocal+0+nz+(iz+1)*nx+1]
;}else if(iy==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+nz+(iz)*nx+nx-2]
-xv[nlocal+0+nz+(iz)*nx+nx-1]
-xv[nlocal+0+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+nz+(iz+1)*nx+nx-1]
;}
else if (ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
            else
{
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+nz+nx*nz]
-xv[nlocal+0+nz+nx*nz+1]
-xv[nlocal+0+nz+nx*nz+ny]
-xv[nlocal+0+nz+nx*nz+ny+1]
-xv[nlocal+0+nz]
-xv[nlocal+0+nz+1]
-xv[nlocal+0+nz+nx]
-xv[nlocal+0+nz+nx+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz-2]
-xv[nlocal+0+nz-1]
-xv[nlocal+0+nz+nx*nz+ny*(nz-2)]
-xv[nlocal+0+nz+nx*nz+ny*(nz-2)+1]
-xv[nlocal+0+nz+nx*nz+ny*(nz-1)]
-xv[nlocal+0+nz+nx*nz+ny*(nz-1)+1]
-xv[nlocal+0+nz+nx*(nz-2)]
-xv[nlocal+0+nz+nx*(nz-2)+1]
-xv[nlocal+0+nz+nx*(nz-1)]
-xv[nlocal+0+nz+nx*(nz-1)+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx*nz+ny-2]
-xv[nlocal+0+nz+nx*nz+ny-1]
-xv[nlocal+0+nz+nx*nz+2*ny-2]
-xv[nlocal+0+nz+nx*nz+2*ny-1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx*nz+ny*(nz-1)-2]
-xv[nlocal+0+nz+nx*nz+ny*(nz-1)-1]
-xv[nlocal+0+nz+nx*nz+ny*nz-2]
-xv[nlocal+0+nz+nx*nz+ny*nz-1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx-2]
-xv[nlocal+0+nz+nx-1]
-xv[nlocal+0+nz+2*nx-2]
-xv[nlocal+0+nz+2*nx-1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nz+nx*(nz-1)-2]
-xv[nlocal+0+nz+nx*(nz-1)-1]
-xv[nlocal+0+nz+nx*nz-2]
-xv[nlocal+0+nz+nx*nz-1]
;
}
else if(iy==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+ix-1]
-xv[nlocal+0+nz+ix]
-xv[nlocal+0+nz+ix+1]
-xv[nlocal+0+nz+nx+ix-1]
-xv[nlocal+0+nz+nx+ix]
-xv[nlocal+0+nz+nx+ix+1]
;}else if(iy==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+nz+ix+(nz-2)*nx]
-xv[nlocal+0+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+nz+ix+(nz-1)*nx]
-xv[nlocal+0+nz+ix+(nz-1)*nx+1]
;}
else if(ix==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+iy-1]
-xv[nlocal+0+nz+nx*nz+iy]
-xv[nlocal+0+nz+nx*nz+iy+1]
-xv[nlocal+0+nz+nx*nz+ny+iy-1]
-xv[nlocal+0+nz+nx*nz+ny+iy]
-xv[nlocal+0+nz+nx*nz+ny+iy+1]
;}else if(ix==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+nz+nx*nz+iy+(nz-2)*ny]
-xv[nlocal+0+nz+nx*nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+nz+nx*nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+nz+nx*nz+iy+(nz-1)*ny]
-xv[nlocal+0+nz+nx*nz+iy+(nz-1)*ny+1]
;}
else if(ix==0&&iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iz-1]
-xv[nlocal+0+iz]
-xv[nlocal+0+iz+1]
;}
else if(ix==0&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+(iz-1)*ny]
-xv[nlocal+0+nz+nx*nz+(iz-1)*ny+1]
-xv[nlocal+0+nz+nx*nz+(iz)*ny]
-xv[nlocal+0+nz+nx*nz+(iz)*ny+1]
-xv[nlocal+0+nz+nx*nz+(iz+1)*ny]
-xv[nlocal+0+nz+nx*nz+(iz+1)*ny+1]
;}else if(ix==0&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+nx*nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+nz+nx*nz+(iz)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+(iz)*ny+ny-1]
-xv[nlocal+0+nz+nx*nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+nz+nx*nz+(iz+1)*ny+ny-1]
;}
else if(iy==0&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+(iz-1)*nx]
-xv[nlocal+0+nz+(iz-1)*nx+1]
-xv[nlocal+0+nz+(iz)*nx]
-xv[nlocal+0+nz+(iz)*nx+1]
-xv[nlocal+0+nz+(iz+1)*nx]
-xv[nlocal+0+nz+(iz+1)*nx+1]
;}else if(iy==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+nz+(iz)*nx+nx-2]
-xv[nlocal+0+nz+(iz)*nx+nx-1]
-xv[nlocal+0+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+nz+(iz+1)*nx+nx-1]
;}
else if (ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+ny+nx*ny+1+nx]
-xv[nlocal+0+ny+nx*ny+1+nx+1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny]
-xv[nlocal+0+ny+nx*ny+1+nx+ny+1]
-xv[nlocal+0+ny]
-xv[nlocal+0+ny+1]
-xv[nlocal+0+ny+nx]
-xv[nlocal+0+ny+nx+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx*ny+1+nx+ny*(nz-2)]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*(nz-2)+1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*(nz-1)]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*(nz-1)+1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny-2]
-xv[nlocal+0+ny-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny-2]
-xv[nlocal+0+ny+nx*ny+1+nx+ny-1]
-xv[nlocal+0+ny+nx*ny+1+nx+2*ny-2]
-xv[nlocal+0+ny+nx*ny+1+nx+2*ny-1]
-xv[nlocal+0+ny+nx*ny]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+1]
-xv[nlocal+0+ny+nx*(ny-2)]
-xv[nlocal+0+ny+nx*(ny-2)+1]
-xv[nlocal+0+ny+nx*(ny-1)]
-xv[nlocal+0+ny+nx*(ny-1)+1]
-xv[nlocal+0+ny+nx*ny+1]
-xv[nlocal+0+ny+nx*ny+1+1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx*ny+1+nx+ny*(nz-1)-2]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*(nz-1)-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz-2]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny-2]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz-2]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+nx*ny]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+nx*(ny-2)]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+nx*(ny-2)+1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+nx*(ny-1)]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+nx*(ny-1)+1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*(nz-2)]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*(nz-2)+1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*(nz-1)]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*(nz-1)+1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+nx*ny+1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+nx*ny+1+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx-2]
-xv[nlocal+0+ny+nx-1]
-xv[nlocal+0+ny+2*nx-2]
-xv[nlocal+0+ny+2*nx-1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+nx-2]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+nx-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+2*nx-2]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+2*nx-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx*(ny-1)-2]
-xv[nlocal+0+ny+nx*(ny-1)-1]
-xv[nlocal+0+ny+nx*ny-2]
-xv[nlocal+0+ny+nx*ny-1]
-xv[nlocal+0+ny+nx*ny+1+nx-2]
-xv[nlocal+0+ny+nx*ny+1+nx-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx-2]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+2*nx-2]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+2*nx-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+nx*(ny-1)-2]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+nx*(ny-1)-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+nx*ny-2]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+nx*ny-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*(nz-1)-2]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*(nz-1)-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz-2]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+nx*ny+1+nx-2]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+nx*ny+1+nx-1]
;
}
else if(iz==0&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+ix-1]
-xv[nlocal+0+ny+ix]
-xv[nlocal+0+ny+ix+1]
-xv[nlocal+0+ny+nx+ix-1]
-xv[nlocal+0+ny+nx+ix]
-xv[nlocal+0+ny+nx+ix+1]
;}else if(iz==0&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+ny+ix+(ny-2)*nx]
-xv[nlocal+0+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+ny+ix+(ny-1)*nx]
-xv[nlocal+0+ny+ix+(ny-1)*nx+1]
;}
else if(iz==nz-1&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+ix-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+ix]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+ix+1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+nx+ix-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+nx+ix]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+nx+ix+1]
;}else if(iz==nz-1&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+ix+(ny-2)*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+ix+(ny-1)*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+ix+(ny-1)*nx+1]
;}
else if(iy==ny-1&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+ix-1]
-xv[nlocal+0+ny+nx*ny+1+ix]
-xv[nlocal+0+ny+nx*ny+1+ix+1]
;}
else if(iy==ny-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix+1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx+ix-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx+ix]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx+ix+1]
;}else if(iy==ny-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix+(nz-2)*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix+(nz-1)*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix+(nz-1)*nx+1]
;}
else if(iy==ny-1&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+nx*ny+1+ix-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+nx*ny+1+ix]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+nx*ny+1+ix+1]
;}
else if(ix==0&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1]
-xv[nlocal+0+iy]
-xv[nlocal+0+iy+1]
;}
else if(ix==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+iy-1]
-xv[nlocal+0+ny+nx*ny+1+nx+iy]
-xv[nlocal+0+ny+nx*ny+1+nx+iy+1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny+iy-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny+iy]
-xv[nlocal+0+ny+nx*ny+1+nx+ny+iy+1]
;}else if(ix==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+iy+(nz-2)*ny-1]
-xv[nlocal+0+ny+nx*ny+1+nx+iy+(nz-2)*ny]
-xv[nlocal+0+ny+nx*ny+1+nx+iy+(nz-2)*ny+1]
-xv[nlocal+0+ny+nx*ny+1+nx+iy+(nz-1)*ny-1]
-xv[nlocal+0+ny+nx*ny+1+nx+iy+(nz-1)*ny]
-xv[nlocal+0+ny+nx*ny+1+nx+iy+(nz-1)*ny+1]
;}
else if(ix==0&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+iy-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+iy]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+iy+1]
;}
else if(iz==0 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+(iy-1)*nx]
-xv[nlocal+0+ny+(iy-1)*nx+1]
-xv[nlocal+0+ny+(iy)*nx]
-xv[nlocal+0+ny+(iy)*nx+1]
-xv[nlocal+0+ny+(iy+1)*nx]
-xv[nlocal+0+ny+(iy+1)*nx+1]
;}else if(iz==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+ny+(iy)*nx+nx-2]
-xv[nlocal+0+ny+(iy)*nx+nx-1]
-xv[nlocal+0+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+ny+(iy+1)*nx+nx-1]
;}
else if(iz==nz-1 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+(iy-1)*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+(iy-1)*nx+1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+(iy)*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+(iy)*nx+1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+(iy+1)*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+(iy+1)*nx+1]
;}else if(iz==nz-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+(iy)*nx+nx-2]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+(iy)*nx+nx-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz+ny+(iy+1)*nx+nx-1]
;}
else if(ix==0&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+1+nx+(iz-1)*ny+1]
-xv[nlocal+0+ny+nx*ny+1+nx+(iz)*ny]
-xv[nlocal+0+ny+nx*ny+1+nx+(iz)*ny+1]
-xv[nlocal+0+ny+nx*ny+1+nx+(iz+1)*ny]
-xv[nlocal+0+ny+nx*ny+1+nx+(iz+1)*ny+1]
;}else if(ix==0&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+(iz-1)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+1+nx+(iz-1)*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+1+nx+(iz)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+1+nx+(iz)*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+1+nx+(iz+1)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+1+nx+(iz+1)*ny+ny-1]
;}
else if(ix==0&&iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+iz-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+iz]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+iz+1]
;}
else if(iy==ny-1&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+(iz-1)*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+(iz-1)*nx+1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+(iz)*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+(iz)*nx+1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+(iz+1)*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+(iz+1)*nx+1]
;}else if(iy==ny-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+(iz)*nx+nx-2]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+(iz)*nx+nx-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+(iz+1)*nx+nx-1]
;}
else if (ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
            else
{
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+ny+nx*ny+1+nx]
-xv[nlocal+0+ny+nx*ny+1+nx+1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny]
-xv[nlocal+0+ny+nx*ny+1+nx+ny+1]
-xv[nlocal+0+ny]
-xv[nlocal+0+ny+1]
-xv[nlocal+0+ny+nx]
-xv[nlocal+0+ny+nx+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx*ny+1+nx+ny*(nz-2)]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*(nz-2)+1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*(nz-1)]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*(nz-1)+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny-2]
-xv[nlocal+0+ny-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny-2]
-xv[nlocal+0+ny+nx*ny+1+nx+ny-1]
-xv[nlocal+0+ny+nx*ny+1+nx+2*ny-2]
-xv[nlocal+0+ny+nx*ny+1+nx+2*ny-1]
-xv[nlocal+0+ny+nx*ny]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+1]
-xv[nlocal+0+ny+nx*(ny-2)]
-xv[nlocal+0+ny+nx*(ny-2)+1]
-xv[nlocal+0+ny+nx*(ny-1)]
-xv[nlocal+0+ny+nx*(ny-1)+1]
-xv[nlocal+0+ny+nx*ny+1]
-xv[nlocal+0+ny+nx*ny+1+1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx*ny+1+nx+ny*(nz-1)-2]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*(nz-1)-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz-2]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz-2]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*(nz-2)]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*(nz-2)+1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*(nz-1)]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*(nz-1)+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx-2]
-xv[nlocal+0+ny+nx-1]
-xv[nlocal+0+ny+2*nx-2]
-xv[nlocal+0+ny+2*nx-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx*(ny-1)-2]
-xv[nlocal+0+ny+nx*(ny-1)-1]
-xv[nlocal+0+ny+nx*ny-2]
-xv[nlocal+0+ny+nx*ny-1]
-xv[nlocal+0+ny+nx*ny+1+nx-2]
-xv[nlocal+0+ny+nx*ny+1+nx-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx-2]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+2*nx-2]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+2*nx-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*(nz-1)-2]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*(nz-1)-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz-2]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx*nz-1]
;
}
else if(iz==0&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+ix-1]
-xv[nlocal+0+ny+ix]
-xv[nlocal+0+ny+ix+1]
-xv[nlocal+0+ny+nx+ix-1]
-xv[nlocal+0+ny+nx+ix]
-xv[nlocal+0+ny+nx+ix+1]
;}else if(iz==0&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+ny+ix+(ny-2)*nx]
-xv[nlocal+0+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+ny+ix+(ny-1)*nx]
-xv[nlocal+0+ny+ix+(ny-1)*nx+1]
;}
else if(iy==ny-1&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+ix-1]
-xv[nlocal+0+ny+nx*ny+1+ix]
-xv[nlocal+0+ny+nx*ny+1+ix+1]
;}
else if(iy==ny-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix+1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx+ix-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx+ix]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+nx+ix+1]
;}else if(iy==ny-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix+(nz-2)*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix+(nz-1)*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+ix+(nz-1)*nx+1]
;}
else if(ix==0&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1]
-xv[nlocal+0+iy]
-xv[nlocal+0+iy+1]
;}
else if(ix==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+iy-1]
-xv[nlocal+0+ny+nx*ny+1+nx+iy]
-xv[nlocal+0+ny+nx*ny+1+nx+iy+1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny+iy-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny+iy]
-xv[nlocal+0+ny+nx*ny+1+nx+ny+iy+1]
;}else if(ix==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+iy+(nz-2)*ny-1]
-xv[nlocal+0+ny+nx*ny+1+nx+iy+(nz-2)*ny]
-xv[nlocal+0+ny+nx*ny+1+nx+iy+(nz-2)*ny+1]
-xv[nlocal+0+ny+nx*ny+1+nx+iy+(nz-1)*ny-1]
-xv[nlocal+0+ny+nx*ny+1+nx+iy+(nz-1)*ny]
-xv[nlocal+0+ny+nx*ny+1+nx+iy+(nz-1)*ny+1]
;}
else if(iz==0 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+(iy-1)*nx]
-xv[nlocal+0+ny+(iy-1)*nx+1]
-xv[nlocal+0+ny+(iy)*nx]
-xv[nlocal+0+ny+(iy)*nx+1]
-xv[nlocal+0+ny+(iy+1)*nx]
-xv[nlocal+0+ny+(iy+1)*nx+1]
;}else if(iz==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+ny+(iy)*nx+nx-2]
-xv[nlocal+0+ny+(iy)*nx+nx-1]
-xv[nlocal+0+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+ny+(iy+1)*nx+nx-1]
;}
else if(ix==0&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+1+nx+(iz-1)*ny+1]
-xv[nlocal+0+ny+nx*ny+1+nx+(iz)*ny]
-xv[nlocal+0+ny+nx*ny+1+nx+(iz)*ny+1]
-xv[nlocal+0+ny+nx*ny+1+nx+(iz+1)*ny]
-xv[nlocal+0+ny+nx*ny+1+nx+(iz+1)*ny+1]
;}else if(ix==0&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+(iz-1)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+1+nx+(iz-1)*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+1+nx+(iz)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+1+nx+(iz)*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+1+nx+(iz+1)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+1+nx+(iz+1)*ny+ny-1]
;}
else if(ix==0&&iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+iz-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+iz]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+iz+1]
;}
else if(iy==ny-1&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+(iz-1)*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+(iz-1)*nx+1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+(iz)*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+(iz)*nx+1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+(iz+1)*nx]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+(iz+1)*nx+1]
;}else if(iy==ny-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+(iz)*nx+nx-2]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+(iz)*nx+nx-1]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+ny+nx*ny+1+nx+ny*nz+nz+(iz+1)*nx+nx-1]
;}
else if (ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
          }//ipz > 0 
        else
{
            if(ipz < npz - 1)
{
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+ny]
-xv[nlocal+0+ny+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*(nz-2)]
-xv[nlocal+0+ny*(nz-2)+1]
-xv[nlocal+0+ny*(nz-1)]
-xv[nlocal+0+ny*(nz-1)+1]
-xv[nlocal+0+ny*nz+nz+nx*nz]
-xv[nlocal+0+ny*nz+nz+nx*nz+1]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+1]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+nx]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny-2]
-xv[nlocal+0+ny-1]
-xv[nlocal+0+2*ny-2]
-xv[nlocal+0+2*ny-1]
-xv[nlocal+0+ny*nz]
-xv[nlocal+0+ny*nz+1]
-xv[nlocal+0+ny*nz+nz]
-xv[nlocal+0+ny*nz+nz+1]
-xv[nlocal+0+ny*nz+nz+nx]
-xv[nlocal+0+ny*nz+nz+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*(nz-1)-2]
-xv[nlocal+0+ny*(nz-1)-1]
-xv[nlocal+0+ny*nz-2]
-xv[nlocal+0+ny*nz-1]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny-2]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny-1]
-xv[nlocal+0+ny*nz+nz-2]
-xv[nlocal+0+ny*nz+nz-1]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+nx*ny]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+nx*(ny-2)]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+nx*(ny-2)+1]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+nx*(ny-1)]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+nx*(ny-1)+1]
-xv[nlocal+0+ny*nz+nz+nx*(nz-2)]
-xv[nlocal+0+ny*nz+nz+nx*(nz-2)+1]
-xv[nlocal+0+ny*nz+nz+nx*(nz-1)]
-xv[nlocal+0+ny*nz+nz+nx*(nz-1)+1]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+nx*ny+1]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+nx*ny+1+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+nx-2]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+nx-1]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+2*nx-2]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+2*nx-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*nz+nz+nx-2]
-xv[nlocal+0+ny*nz+nz+nx-1]
-xv[nlocal+0+ny*nz+nz+2*nx-2]
-xv[nlocal+0+ny*nz+nz+2*nx-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+nx*(ny-1)-2]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+nx*(ny-1)-1]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+nx*ny-2]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+nx*ny-1]
-xv[nlocal+0+ny*nz+nz+nx*(nz-1)-2]
-xv[nlocal+0+ny*nz+nz+nx*(nz-1)-1]
-xv[nlocal+0+ny*nz+nz+nx*nz-2]
-xv[nlocal+0+ny*nz+nz+nx*nz-1]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+nx*ny+1+nx-2]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+nx*ny+1+nx-1]
;
}
else if(iz==nz-1&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+ix-1]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+ix]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+ix+1]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+nx+ix-1]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+nx+ix]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+nx+ix+1]
;}else if(iz==nz-1&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+ix+(ny-2)*nx]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+ix+(ny-1)*nx]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+ix+(ny-1)*nx+1]
;}
else if(iy==ny-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+nz+ix-1]
-xv[nlocal+0+ny*nz+nz+ix]
-xv[nlocal+0+ny*nz+nz+ix+1]
-xv[nlocal+0+ny*nz+nz+nx+ix-1]
-xv[nlocal+0+ny*nz+nz+nx+ix]
-xv[nlocal+0+ny*nz+nz+nx+ix+1]
;}else if(iy==ny-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+ny*nz+nz+ix+(nz-2)*nx]
-xv[nlocal+0+ny*nz+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+ny*nz+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+ny*nz+nz+ix+(nz-1)*nx]
-xv[nlocal+0+ny*nz+nz+ix+(nz-1)*nx+1]
;}
else if(iy==ny-1&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+nx*ny+1+ix-1]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+nx*ny+1+ix]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+nx*ny+1+ix+1]
;}
else if(ix==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1]
-xv[nlocal+0+iy]
-xv[nlocal+0+iy+1]
-xv[nlocal+0+ny+iy-1]
-xv[nlocal+0+ny+iy]
-xv[nlocal+0+ny+iy+1]
;}else if(ix==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy+(nz-2)*ny-1]
-xv[nlocal+0+iy+(nz-2)*ny]
-xv[nlocal+0+iy+(nz-2)*ny+1]
-xv[nlocal+0+iy+(nz-1)*ny-1]
-xv[nlocal+0+iy+(nz-1)*ny]
-xv[nlocal+0+iy+(nz-1)*ny+1]
;}
else if(ix==0&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+nz+nx*nz+iy-1]
-xv[nlocal+0+ny*nz+nz+nx*nz+iy]
-xv[nlocal+0+ny*nz+nz+nx*nz+iy+1]
;}
else if(iz==nz-1 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+(iy-1)*nx]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+(iy-1)*nx+1]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+(iy)*nx]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+(iy)*nx+1]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+(iy+1)*nx]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+(iy+1)*nx+1]
;}else if(iz==nz-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+(iy)*nx+nx-2]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+(iy)*nx+nx-1]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+ny*nz+nz+nx*nz+ny+(iy+1)*nx+nx-1]
;}
else if(ix==0&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*ny]
-xv[nlocal+0+(iz-1)*ny+1]
-xv[nlocal+0+(iz)*ny]
-xv[nlocal+0+(iz)*ny+1]
-xv[nlocal+0+(iz+1)*ny]
-xv[nlocal+0+(iz+1)*ny+1]
;}else if(ix==0&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*ny+ny-2]
-xv[nlocal+0+(iz-1)*ny+ny-1]
-xv[nlocal+0+(iz)*ny+ny-2]
-xv[nlocal+0+(iz)*ny+ny-1]
-xv[nlocal+0+(iz+1)*ny+ny-2]
-xv[nlocal+0+(iz+1)*ny+ny-1]
;}
else if(ix==0&&iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+iz-1]
-xv[nlocal+0+ny*nz+iz]
-xv[nlocal+0+ny*nz+iz+1]
;}
else if(iy==ny-1&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+nz+(iz-1)*nx]
-xv[nlocal+0+ny*nz+nz+(iz-1)*nx+1]
-xv[nlocal+0+ny*nz+nz+(iz)*nx]
-xv[nlocal+0+ny*nz+nz+(iz)*nx+1]
-xv[nlocal+0+ny*nz+nz+(iz+1)*nx]
-xv[nlocal+0+ny*nz+nz+(iz+1)*nx+1]
;}else if(iy==ny-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+ny*nz+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+ny*nz+nz+(iz)*nx+nx-2]
-xv[nlocal+0+ny*nz+nz+(iz)*nx+nx-1]
-xv[nlocal+0+ny*nz+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+ny*nz+nz+(iz+1)*nx+nx-1]
;}
else if (ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
            else
{
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+ny]
-xv[nlocal+0+ny+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*(nz-2)]
-xv[nlocal+0+ny*(nz-2)+1]
-xv[nlocal+0+ny*(nz-1)]
-xv[nlocal+0+ny*(nz-1)+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny-2]
-xv[nlocal+0+ny-1]
-xv[nlocal+0+2*ny-2]
-xv[nlocal+0+2*ny-1]
-xv[nlocal+0+ny*nz]
-xv[nlocal+0+ny*nz+1]
-xv[nlocal+0+ny*nz+nz]
-xv[nlocal+0+ny*nz+nz+1]
-xv[nlocal+0+ny*nz+nz+nx]
-xv[nlocal+0+ny*nz+nz+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*(nz-1)-2]
-xv[nlocal+0+ny*(nz-1)-1]
-xv[nlocal+0+ny*nz-2]
-xv[nlocal+0+ny*nz-1]
-xv[nlocal+0+ny*nz+nz-2]
-xv[nlocal+0+ny*nz+nz-1]
-xv[nlocal+0+ny*nz+nz+nx*(nz-2)]
-xv[nlocal+0+ny*nz+nz+nx*(nz-2)+1]
-xv[nlocal+0+ny*nz+nz+nx*(nz-1)]
-xv[nlocal+0+ny*nz+nz+nx*(nz-1)+1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*nz+nz+nx-2]
-xv[nlocal+0+ny*nz+nz+nx-1]
-xv[nlocal+0+ny*nz+nz+2*nx-2]
-xv[nlocal+0+ny*nz+nz+2*nx-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*nz+nz+nx*(nz-1)-2]
-xv[nlocal+0+ny*nz+nz+nx*(nz-1)-1]
-xv[nlocal+0+ny*nz+nz+nx*nz-2]
-xv[nlocal+0+ny*nz+nz+nx*nz-1]
;
}
else if(iy==ny-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+nz+ix-1]
-xv[nlocal+0+ny*nz+nz+ix]
-xv[nlocal+0+ny*nz+nz+ix+1]
-xv[nlocal+0+ny*nz+nz+nx+ix-1]
-xv[nlocal+0+ny*nz+nz+nx+ix]
-xv[nlocal+0+ny*nz+nz+nx+ix+1]
;}else if(iy==ny-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+ny*nz+nz+ix+(nz-2)*nx]
-xv[nlocal+0+ny*nz+nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+ny*nz+nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+ny*nz+nz+ix+(nz-1)*nx]
-xv[nlocal+0+ny*nz+nz+ix+(nz-1)*nx+1]
;}
else if(ix==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1]
-xv[nlocal+0+iy]
-xv[nlocal+0+iy+1]
-xv[nlocal+0+ny+iy-1]
-xv[nlocal+0+ny+iy]
-xv[nlocal+0+ny+iy+1]
;}else if(ix==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy+(nz-2)*ny-1]
-xv[nlocal+0+iy+(nz-2)*ny]
-xv[nlocal+0+iy+(nz-2)*ny+1]
-xv[nlocal+0+iy+(nz-1)*ny-1]
-xv[nlocal+0+iy+(nz-1)*ny]
-xv[nlocal+0+iy+(nz-1)*ny+1]
;}
else if(ix==0&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*ny]
-xv[nlocal+0+(iz-1)*ny+1]
-xv[nlocal+0+(iz)*ny]
-xv[nlocal+0+(iz)*ny+1]
-xv[nlocal+0+(iz+1)*ny]
-xv[nlocal+0+(iz+1)*ny+1]
;}else if(ix==0&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*ny+ny-2]
-xv[nlocal+0+(iz-1)*ny+ny-1]
-xv[nlocal+0+(iz)*ny+ny-2]
-xv[nlocal+0+(iz)*ny+ny-1]
-xv[nlocal+0+(iz+1)*ny+ny-2]
-xv[nlocal+0+(iz+1)*ny+ny-1]
;}
else if(ix==0&&iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+iz-1]
-xv[nlocal+0+ny*nz+iz]
-xv[nlocal+0+ny*nz+iz+1]
;}
else if(iy==ny-1&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+nz+(iz-1)*nx]
-xv[nlocal+0+ny*nz+nz+(iz-1)*nx+1]
-xv[nlocal+0+ny*nz+nz+(iz)*nx]
-xv[nlocal+0+ny*nz+nz+(iz)*nx+1]
-xv[nlocal+0+ny*nz+nz+(iz+1)*nx]
-xv[nlocal+0+ny*nz+nz+(iz+1)*nx+1]
;}else if(iy==ny-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+ny*nz+nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+ny*nz+nz+(iz)*nx+nx-2]
-xv[nlocal+0+ny*nz+nz+(iz)*nx+nx-1]
-xv[nlocal+0+ny*nz+nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+ny*nz+nz+(iz+1)*nx+nx-1]
;}
else if (ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
          }//ipz > 0 
        }//ipy < npy - 1 
      else
{
        if(ipz > 0)
{
            if(ipz < npz - 1)
{
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+ny+nx*ny]
-xv[nlocal+0+ny+nx*ny+1]
-xv[nlocal+0+ny+nx*ny+ny]
-xv[nlocal+0+ny+nx*ny+ny+1]
-xv[nlocal+0+ny]
-xv[nlocal+0+ny+1]
-xv[nlocal+0+ny+nx]
-xv[nlocal+0+ny+nx+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx*ny+ny*(nz-2)]
-xv[nlocal+0+ny+nx*ny+ny*(nz-2)+1]
-xv[nlocal+0+ny+nx*ny+ny*(nz-1)]
-xv[nlocal+0+ny+nx*ny+ny*(nz-1)+1]
-xv[nlocal+0+ny+nx*ny+ny*nz]
-xv[nlocal+0+ny+nx*ny+ny*nz+1]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+1]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+nx]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny-2]
-xv[nlocal+0+ny-1]
-xv[nlocal+0+ny+nx*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+2*ny-2]
-xv[nlocal+0+ny+nx*ny+2*ny-1]
-xv[nlocal+0+ny+nx*(ny-2)]
-xv[nlocal+0+ny+nx*(ny-2)+1]
-xv[nlocal+0+ny+nx*(ny-1)]
-xv[nlocal+0+ny+nx*(ny-1)+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx*ny+ny*(nz-1)-2]
-xv[nlocal+0+ny+nx*ny+ny*(nz-1)-1]
-xv[nlocal+0+ny+nx*ny+ny*nz-2]
-xv[nlocal+0+ny+nx*ny+ny*nz-1]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny-2]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny-1]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+nx*(ny-2)]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+nx*(ny-2)+1]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+nx*(ny-1)]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+nx*(ny-1)+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx-2]
-xv[nlocal+0+ny+nx-1]
-xv[nlocal+0+ny+2*nx-2]
-xv[nlocal+0+ny+2*nx-1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+nx-2]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+nx-1]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+2*nx-2]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+2*nx-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx*(ny-1)-2]
-xv[nlocal+0+ny+nx*(ny-1)-1]
-xv[nlocal+0+ny+nx*ny-2]
-xv[nlocal+0+ny+nx*ny-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+nx*(ny-1)-2]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+nx*(ny-1)-1]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+nx*ny-2]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+nx*ny-1]
;
}
else if(iz==0&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+ix-1]
-xv[nlocal+0+ny+ix]
-xv[nlocal+0+ny+ix+1]
-xv[nlocal+0+ny+nx+ix-1]
-xv[nlocal+0+ny+nx+ix]
-xv[nlocal+0+ny+nx+ix+1]
;}else if(iz==0&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+ny+ix+(ny-2)*nx]
-xv[nlocal+0+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+ny+ix+(ny-1)*nx]
-xv[nlocal+0+ny+ix+(ny-1)*nx+1]
;}
else if(iz==nz-1&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+ix-1]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+ix]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+ix+1]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+nx+ix-1]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+nx+ix]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+nx+ix+1]
;}else if(iz==nz-1&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+ix+(ny-2)*nx]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+ix+(ny-1)*nx]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+ix+(ny-1)*nx+1]
;}
else if(ix==0&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1]
-xv[nlocal+0+iy]
-xv[nlocal+0+iy+1]
;}
else if(ix==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+iy-1]
-xv[nlocal+0+ny+nx*ny+iy]
-xv[nlocal+0+ny+nx*ny+iy+1]
-xv[nlocal+0+ny+nx*ny+ny+iy-1]
-xv[nlocal+0+ny+nx*ny+ny+iy]
-xv[nlocal+0+ny+nx*ny+ny+iy+1]
;}else if(ix==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+iy+(nz-2)*ny-1]
-xv[nlocal+0+ny+nx*ny+iy+(nz-2)*ny]
-xv[nlocal+0+ny+nx*ny+iy+(nz-2)*ny+1]
-xv[nlocal+0+ny+nx*ny+iy+(nz-1)*ny-1]
-xv[nlocal+0+ny+nx*ny+iy+(nz-1)*ny]
-xv[nlocal+0+ny+nx*ny+iy+(nz-1)*ny+1]
;}
else if(ix==0&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny*nz+iy-1]
-xv[nlocal+0+ny+nx*ny+ny*nz+iy]
-xv[nlocal+0+ny+nx*ny+ny*nz+iy+1]
;}
else if(iz==0 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+(iy-1)*nx]
-xv[nlocal+0+ny+(iy-1)*nx+1]
-xv[nlocal+0+ny+(iy)*nx]
-xv[nlocal+0+ny+(iy)*nx+1]
-xv[nlocal+0+ny+(iy+1)*nx]
-xv[nlocal+0+ny+(iy+1)*nx+1]
;}else if(iz==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+ny+(iy)*nx+nx-2]
-xv[nlocal+0+ny+(iy)*nx+nx-1]
-xv[nlocal+0+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+ny+(iy+1)*nx+nx-1]
;}
else if(iz==nz-1 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+(iy-1)*nx]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+(iy-1)*nx+1]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+(iy)*nx]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+(iy)*nx+1]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+(iy+1)*nx]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+(iy+1)*nx+1]
;}else if(iz==nz-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+(iy)*nx+nx-2]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+(iy)*nx+nx-1]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+ny+nx*ny+ny*nz+ny+(iy+1)*nx+nx-1]
;}
else if(ix==0&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+(iz-1)*ny+1]
-xv[nlocal+0+ny+nx*ny+(iz)*ny]
-xv[nlocal+0+ny+nx*ny+(iz)*ny+1]
-xv[nlocal+0+ny+nx*ny+(iz+1)*ny]
-xv[nlocal+0+ny+nx*ny+(iz+1)*ny+1]
;}else if(ix==0&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+(iz-1)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+(iz-1)*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+(iz)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+(iz)*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+(iz+1)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+(iz+1)*ny+ny-1]
;}
else if (ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
            else
{
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+ny+nx*ny]
-xv[nlocal+0+ny+nx*ny+1]
-xv[nlocal+0+ny+nx*ny+ny]
-xv[nlocal+0+ny+nx*ny+ny+1]
-xv[nlocal+0+ny]
-xv[nlocal+0+ny+1]
-xv[nlocal+0+ny+nx]
-xv[nlocal+0+ny+nx+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx*ny+ny*(nz-2)]
-xv[nlocal+0+ny+nx*ny+ny*(nz-2)+1]
-xv[nlocal+0+ny+nx*ny+ny*(nz-1)]
-xv[nlocal+0+ny+nx*ny+ny*(nz-1)+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny-2]
-xv[nlocal+0+ny-1]
-xv[nlocal+0+ny+nx*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+2*ny-2]
-xv[nlocal+0+ny+nx*ny+2*ny-1]
-xv[nlocal+0+ny+nx*(ny-2)]
-xv[nlocal+0+ny+nx*(ny-2)+1]
-xv[nlocal+0+ny+nx*(ny-1)]
-xv[nlocal+0+ny+nx*(ny-1)+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx*ny+ny*(nz-1)-2]
-xv[nlocal+0+ny+nx*ny+ny*(nz-1)-1]
-xv[nlocal+0+ny+nx*ny+ny*nz-2]
-xv[nlocal+0+ny+nx*ny+ny*nz-1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx-2]
-xv[nlocal+0+ny+nx-1]
-xv[nlocal+0+ny+2*nx-2]
-xv[nlocal+0+ny+2*nx-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny+nx*(ny-1)-2]
-xv[nlocal+0+ny+nx*(ny-1)-1]
-xv[nlocal+0+ny+nx*ny-2]
-xv[nlocal+0+ny+nx*ny-1]
;
}
else if(iz==0&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+ix-1]
-xv[nlocal+0+ny+ix]
-xv[nlocal+0+ny+ix+1]
-xv[nlocal+0+ny+nx+ix-1]
-xv[nlocal+0+ny+nx+ix]
-xv[nlocal+0+ny+nx+ix+1]
;}else if(iz==0&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+ny+ix+(ny-2)*nx]
-xv[nlocal+0+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+ny+ix+(ny-1)*nx]
-xv[nlocal+0+ny+ix+(ny-1)*nx+1]
;}
else if(ix==0&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1]
-xv[nlocal+0+iy]
-xv[nlocal+0+iy+1]
;}
else if(ix==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+iy-1]
-xv[nlocal+0+ny+nx*ny+iy]
-xv[nlocal+0+ny+nx*ny+iy+1]
-xv[nlocal+0+ny+nx*ny+ny+iy-1]
-xv[nlocal+0+ny+nx*ny+ny+iy]
-xv[nlocal+0+ny+nx*ny+ny+iy+1]
;}else if(ix==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+iy+(nz-2)*ny-1]
-xv[nlocal+0+ny+nx*ny+iy+(nz-2)*ny]
-xv[nlocal+0+ny+nx*ny+iy+(nz-2)*ny+1]
-xv[nlocal+0+ny+nx*ny+iy+(nz-1)*ny-1]
-xv[nlocal+0+ny+nx*ny+iy+(nz-1)*ny]
-xv[nlocal+0+ny+nx*ny+iy+(nz-1)*ny+1]
;}
else if(iz==0 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+(iy-1)*nx]
-xv[nlocal+0+ny+(iy-1)*nx+1]
-xv[nlocal+0+ny+(iy)*nx]
-xv[nlocal+0+ny+(iy)*nx+1]
-xv[nlocal+0+ny+(iy+1)*nx]
-xv[nlocal+0+ny+(iy+1)*nx+1]
;}else if(iz==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+ny+(iy)*nx+nx-2]
-xv[nlocal+0+ny+(iy)*nx+nx-1]
-xv[nlocal+0+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+ny+(iy+1)*nx+nx-1]
;}
else if(ix==0&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+(iz-1)*ny]
-xv[nlocal+0+ny+nx*ny+(iz-1)*ny+1]
-xv[nlocal+0+ny+nx*ny+(iz)*ny]
-xv[nlocal+0+ny+nx*ny+(iz)*ny+1]
-xv[nlocal+0+ny+nx*ny+(iz+1)*ny]
-xv[nlocal+0+ny+nx*ny+(iz+1)*ny+1]
;}else if(ix==0&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny+nx*ny+(iz-1)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+(iz-1)*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+(iz)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+(iz)*ny+ny-1]
-xv[nlocal+0+ny+nx*ny+(iz+1)*ny+ny-2]
-xv[nlocal+0+ny+nx*ny+(iz+1)*ny+ny-1]
;}
else if (ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
          }//ipz > 0 
        else
{
            if(ipz < npz - 1)
{
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+ny]
-xv[nlocal+0+ny+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*(nz-2)]
-xv[nlocal+0+ny*(nz-2)+1]
-xv[nlocal+0+ny*(nz-1)]
-xv[nlocal+0+ny*(nz-1)+1]
-xv[nlocal+0+ny*nz]
-xv[nlocal+0+ny*nz+1]
-xv[nlocal+0+ny*nz+ny]
-xv[nlocal+0+ny*nz+ny+1]
-xv[nlocal+0+ny*nz+ny+nx]
-xv[nlocal+0+ny*nz+ny+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny-2]
-xv[nlocal+0+ny-1]
-xv[nlocal+0+2*ny-2]
-xv[nlocal+0+2*ny-1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*(nz-1)-2]
-xv[nlocal+0+ny*(nz-1)-1]
-xv[nlocal+0+ny*nz-2]
-xv[nlocal+0+ny*nz-1]
-xv[nlocal+0+ny*nz+ny-2]
-xv[nlocal+0+ny*nz+ny-1]
-xv[nlocal+0+ny*nz+ny+nx*(ny-2)]
-xv[nlocal+0+ny*nz+ny+nx*(ny-2)+1]
-xv[nlocal+0+ny*nz+ny+nx*(ny-1)]
-xv[nlocal+0+ny*nz+ny+nx*(ny-1)+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*nz+ny+nx-2]
-xv[nlocal+0+ny*nz+ny+nx-1]
-xv[nlocal+0+ny*nz+ny+2*nx-2]
-xv[nlocal+0+ny*nz+ny+2*nx-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*nz+ny+nx*(ny-1)-2]
-xv[nlocal+0+ny*nz+ny+nx*(ny-1)-1]
-xv[nlocal+0+ny*nz+ny+nx*ny-2]
-xv[nlocal+0+ny*nz+ny+nx*ny-1]
;
}
else if(iz==nz-1&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny+ix-1]
-xv[nlocal+0+ny*nz+ny+ix]
-xv[nlocal+0+ny*nz+ny+ix+1]
-xv[nlocal+0+ny*nz+ny+nx+ix-1]
-xv[nlocal+0+ny*nz+ny+nx+ix]
-xv[nlocal+0+ny*nz+ny+nx+ix+1]
;}else if(iz==nz-1&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+ny*nz+ny+ix+(ny-2)*nx]
-xv[nlocal+0+ny*nz+ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+ny*nz+ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+ny*nz+ny+ix+(ny-1)*nx]
-xv[nlocal+0+ny*nz+ny+ix+(ny-1)*nx+1]
;}
else if(ix==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1]
-xv[nlocal+0+iy]
-xv[nlocal+0+iy+1]
-xv[nlocal+0+ny+iy-1]
-xv[nlocal+0+ny+iy]
-xv[nlocal+0+ny+iy+1]
;}else if(ix==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy+(nz-2)*ny-1]
-xv[nlocal+0+iy+(nz-2)*ny]
-xv[nlocal+0+iy+(nz-2)*ny+1]
-xv[nlocal+0+iy+(nz-1)*ny-1]
-xv[nlocal+0+iy+(nz-1)*ny]
-xv[nlocal+0+iy+(nz-1)*ny+1]
;}
else if(ix==0&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+iy-1]
-xv[nlocal+0+ny*nz+iy]
-xv[nlocal+0+ny*nz+iy+1]
;}
else if(iz==nz-1 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny+(iy-1)*nx]
-xv[nlocal+0+ny*nz+ny+(iy-1)*nx+1]
-xv[nlocal+0+ny*nz+ny+(iy)*nx]
-xv[nlocal+0+ny*nz+ny+(iy)*nx+1]
-xv[nlocal+0+ny*nz+ny+(iy+1)*nx]
-xv[nlocal+0+ny*nz+ny+(iy+1)*nx+1]
;}else if(iz==nz-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+ny*nz+ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+ny*nz+ny+(iy)*nx+nx-2]
-xv[nlocal+0+ny*nz+ny+(iy)*nx+nx-1]
-xv[nlocal+0+ny*nz+ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+ny*nz+ny+(iy+1)*nx+nx-1]
;}
else if(ix==0&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*ny]
-xv[nlocal+0+(iz-1)*ny+1]
-xv[nlocal+0+(iz)*ny]
-xv[nlocal+0+(iz)*ny+1]
-xv[nlocal+0+(iz+1)*ny]
-xv[nlocal+0+(iz+1)*ny+1]
;}else if(ix==0&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*ny+ny-2]
-xv[nlocal+0+(iz-1)*ny+ny-1]
-xv[nlocal+0+(iz)*ny+ny-2]
-xv[nlocal+0+(iz)*ny+ny-1]
-xv[nlocal+0+(iz+1)*ny+ny-2]
-xv[nlocal+0+(iz+1)*ny+ny-1]
;}
else if (ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
            else
{
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+ny]
-xv[nlocal+0+ny+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*(nz-2)]
-xv[nlocal+0+ny*(nz-2)+1]
-xv[nlocal+0+ny*(nz-1)]
-xv[nlocal+0+ny*(nz-1)+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny-2]
-xv[nlocal+0+ny-1]
-xv[nlocal+0+2*ny-2]
-xv[nlocal+0+2*ny-1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*(nz-1)-2]
-xv[nlocal+0+ny*(nz-1)-1]
-xv[nlocal+0+ny*nz-2]
-xv[nlocal+0+ny*nz-1]
;
}
else if(ix==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1]
-xv[nlocal+0+iy]
-xv[nlocal+0+iy+1]
-xv[nlocal+0+ny+iy-1]
-xv[nlocal+0+ny+iy]
-xv[nlocal+0+ny+iy+1]
;}else if(ix==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy+(nz-2)*ny-1]
-xv[nlocal+0+iy+(nz-2)*ny]
-xv[nlocal+0+iy+(nz-2)*ny+1]
-xv[nlocal+0+iy+(nz-1)*ny-1]
-xv[nlocal+0+iy+(nz-1)*ny]
-xv[nlocal+0+iy+(nz-1)*ny+1]
;}
else if(ix==0&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*ny]
-xv[nlocal+0+(iz-1)*ny+1]
-xv[nlocal+0+(iz)*ny]
-xv[nlocal+0+(iz)*ny+1]
-xv[nlocal+0+(iz+1)*ny]
-xv[nlocal+0+(iz+1)*ny+1]
;}else if(ix==0&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*ny+ny-2]
-xv[nlocal+0+(iz-1)*ny+ny-1]
-xv[nlocal+0+(iz)*ny+ny-2]
-xv[nlocal+0+(iz)*ny+ny-1]
-xv[nlocal+0+(iz+1)*ny+ny-2]
-xv[nlocal+0+(iz+1)*ny+ny-1]
;}
else if (ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx+1]
-xv[nlocal+0+nx+1]
-xv[nlocal+0+nx+1+1]
-xv[nlocal+0+nx+1+nx]
-xv[nlocal+0+nx+1+nx+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*(nz-2)]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*(nz-2)+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*(nz-1)]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*(nz-1)+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+1+nx*(ny-2)]
-xv[nlocal+0+nx+1+nx*(ny-2)+1]
-xv[nlocal+0+nx+1+nx*(ny-1)]
-xv[nlocal+0+nx+1+nx*(ny-1)+1]
-xv[nlocal+0+nx+1+nx*ny+ny]
-xv[nlocal+0+nx+1+nx*ny+ny+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*(ny-2)]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*(ny-2)+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*(ny-1)]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*(ny-1)+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*(nz-2)]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*(nz-2)+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*(nz-1)]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*(nz-1)+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny+ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny+ny+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx-2]
-xv[nlocal+0+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+2*nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+2*nx-1]
-xv[nlocal+0+nx+1+nx-2]
-xv[nlocal+0+nx+1+nx-1]
-xv[nlocal+0+nx+1+2*nx-2]
-xv[nlocal+0+nx+1+2*nx-1]
-xv[nlocal+0+nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+1]
-xv[nlocal+0+nx+1+nx*ny]
-xv[nlocal+0+nx+1+nx*ny+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*(nz-1)-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*(nz-1)-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+2*nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+2*nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*(nz-2)]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*(nz-2)+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*(nz-1)]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*(nz-1)+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny+1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+1+nx*(ny-1)-2]
-xv[nlocal+0+nx+1+nx*(ny-1)-1]
-xv[nlocal+0+nx+1+nx*ny-2]
-xv[nlocal+0+nx+1+nx*ny-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+2*nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+2*nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny-2]
-xv[nlocal+0+nx+1+nx*ny+ny-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+2*ny-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+2*ny-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*(ny-1)-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*(ny-1)-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*(nz-1)-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*(nz-1)-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny+ny+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny+ny+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*(nz-1)-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*(nz-1)-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny+ny-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny+ny-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny+ny+nx]
;
}
else if(iy==0&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
;}
else if(iy==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx+ix-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx+ix]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx+ix+1]
;}else if(iy==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix+(nz-2)*nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix+(nz-2)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix+(nz-2)*nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix+(nz-1)*nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix+(nz-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix+(nz-1)*nx+1]
;}
else if(iy==0&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+ix-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+ix]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+ix+1]
;}
else if(iz==0&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+ix-1]
-xv[nlocal+0+nx+1+ix]
-xv[nlocal+0+nx+1+ix+1]
-xv[nlocal+0+nx+1+nx+ix-1]
-xv[nlocal+0+nx+1+nx+ix]
-xv[nlocal+0+nx+1+nx+ix+1]
;}else if(iz==0&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+ix+(ny-2)*nx-1]
-xv[nlocal+0+nx+1+ix+(ny-2)*nx]
-xv[nlocal+0+nx+1+ix+(ny-2)*nx+1]
-xv[nlocal+0+nx+1+ix+(ny-1)*nx-1]
-xv[nlocal+0+nx+1+ix+(ny-1)*nx]
-xv[nlocal+0+nx+1+ix+(ny-1)*nx+1]
;}
else if(iz==nz-1&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx+ix-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx+ix]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx+ix+1]
;}else if(iz==nz-1&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix+(ny-2)*nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix+(ny-2)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix+(ny-2)*nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix+(ny-1)*nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix+(ny-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix+(ny-1)*nx+1]
;}
else if(iy==ny-1&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+ix-1]
-xv[nlocal+0+nx+1+nx*ny+ny+ix]
-xv[nlocal+0+nx+1+nx*ny+ny+ix+1]
;}
else if(iy==ny-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx+ix-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx+ix]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx+ix+1]
;}else if(iy==ny-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix+(nz-2)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix+(nz-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix+(nz-1)*nx+1]
;}
else if(iy==ny-1&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny+ny+ix-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny+ny+ix]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny+ny+ix+1]
;}
else if(iz==0 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+(iy-1)*nx]
-xv[nlocal+0+nx+1+(iy-1)*nx+1]
-xv[nlocal+0+nx+1+(iy)*nx]
-xv[nlocal+0+nx+1+(iy)*nx+1]
-xv[nlocal+0+nx+1+(iy+1)*nx]
-xv[nlocal+0+nx+1+(iy+1)*nx+1]
;}else if(iz==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+(iy-1)*nx+nx-2]
-xv[nlocal+0+nx+1+(iy-1)*nx+nx-1]
-xv[nlocal+0+nx+1+(iy)*nx+nx-2]
-xv[nlocal+0+nx+1+(iy)*nx+nx-1]
-xv[nlocal+0+nx+1+(iy+1)*nx+nx-2]
-xv[nlocal+0+nx+1+(iy+1)*nx+nx-1]
;}
else if(iz==nz-1 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+(iy-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+(iy-1)*nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+(iy)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+(iy)*nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+(iy+1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+(iy+1)*nx+1]
;}else if(iz==nz-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+(iy-1)*nx+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+(iy-1)*nx+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+(iy)*nx+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+(iy)*nx+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+(iy+1)*nx+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+(iy+1)*nx+nx-1]
;}
else if(ix==nx-1&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+iy-1]
-xv[nlocal+0+nx+1+nx*ny+iy]
-xv[nlocal+0+nx+1+nx*ny+iy+1]
;}
else if(ix==nx-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny+iy-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny+iy]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny+iy+1]
;}else if(ix==nx-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy+(nz-2)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy+(nz-1)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy+(nz-1)*ny+1]
;}
else if(ix==nx-1&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny+iy-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny+iy]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny+iy+1]
;}
else if(iy==0&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+(iz-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+(iz-1)*nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+(iz)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+(iz)*nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+(iz+1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+(iz+1)*nx+1]
;}else if(iy==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+(iz-1)*nx+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+(iz-1)*nx+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+(iz)*nx+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+(iz)*nx+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+(iz+1)*nx+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+(iz+1)*nx+nx-1]
;}
else if(iy==ny-1&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+(iz-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+(iz-1)*nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+(iz)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+(iz)*nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+(iz+1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+(iz+1)*nx+1]
;}else if(iy==ny-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+(iz)*nx+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+(iz)*nx+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+(iz+1)*nx+nx-1]
;}
else if(ix==nx-1&&iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+iz-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+iz]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+iz+1]
;}
else if(ix==nx-1&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+(iz-1)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+(iz-1)*ny+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+(iz)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+(iz)*ny+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+(iz+1)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+(iz+1)*ny+1]
;}else if(ix==nx-1&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+(iz)*ny+ny-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+(iz)*ny+ny-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+(iz+1)*ny+ny-1]
;}
else if(ix==nx-1&&iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+iz-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+iz]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+iz+1]
;}
else if (ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
            else
{
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx+1]
-xv[nlocal+0+nx+1]
-xv[nlocal+0+nx+1+1]
-xv[nlocal+0+nx+1+nx]
-xv[nlocal+0+nx+1+nx+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*(nz-2)]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*(nz-2)+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*(nz-1)]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*(nz-1)+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+1+nx*(ny-2)]
-xv[nlocal+0+nx+1+nx*(ny-2)+1]
-xv[nlocal+0+nx+1+nx*(ny-1)]
-xv[nlocal+0+nx+1+nx*(ny-1)+1]
-xv[nlocal+0+nx+1+nx*ny+ny]
-xv[nlocal+0+nx+1+nx*ny+ny+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*(nz-2)]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*(nz-2)+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*(nz-1)]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*(nz-1)+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx-2]
-xv[nlocal+0+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+2*nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+2*nx-1]
-xv[nlocal+0+nx+1+nx-2]
-xv[nlocal+0+nx+1+nx-1]
-xv[nlocal+0+nx+1+2*nx-2]
-xv[nlocal+0+nx+1+2*nx-1]
-xv[nlocal+0+nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+1]
-xv[nlocal+0+nx+1+nx*ny]
-xv[nlocal+0+nx+1+nx*ny+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*(nz-1)-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*(nz-1)-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*(nz-2)]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*(nz-2)+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*(nz-1)]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*(nz-1)+1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+1+nx*(ny-1)-2]
-xv[nlocal+0+nx+1+nx*(ny-1)-1]
-xv[nlocal+0+nx+1+nx*ny-2]
-xv[nlocal+0+nx+1+nx*ny-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+2*nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+2*nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny-2]
-xv[nlocal+0+nx+1+nx*ny+ny-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+2*ny-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+2*ny-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*(nz-1)-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*(nz-1)-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*(nz-1)-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*(nz-1)-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+nz-1]
;
}
else if(iy==0&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
;}
else if(iy==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx+ix-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx+ix]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx+ix+1]
;}else if(iy==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix+(nz-2)*nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix+(nz-2)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix+(nz-2)*nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix+(nz-1)*nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix+(nz-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+ix+(nz-1)*nx+1]
;}
else if(iz==0&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+ix-1]
-xv[nlocal+0+nx+1+ix]
-xv[nlocal+0+nx+1+ix+1]
-xv[nlocal+0+nx+1+nx+ix-1]
-xv[nlocal+0+nx+1+nx+ix]
-xv[nlocal+0+nx+1+nx+ix+1]
;}else if(iz==0&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+ix+(ny-2)*nx-1]
-xv[nlocal+0+nx+1+ix+(ny-2)*nx]
-xv[nlocal+0+nx+1+ix+(ny-2)*nx+1]
-xv[nlocal+0+nx+1+ix+(ny-1)*nx-1]
-xv[nlocal+0+nx+1+ix+(ny-1)*nx]
-xv[nlocal+0+nx+1+ix+(ny-1)*nx+1]
;}
else if(iy==ny-1&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+ix-1]
-xv[nlocal+0+nx+1+nx*ny+ny+ix]
-xv[nlocal+0+nx+1+nx*ny+ny+ix+1]
;}
else if(iy==ny-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx+ix-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx+ix]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx+ix+1]
;}else if(iy==ny-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix+(nz-2)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix+(nz-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+ix+(nz-1)*nx+1]
;}
else if(iz==0 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+(iy-1)*nx]
-xv[nlocal+0+nx+1+(iy-1)*nx+1]
-xv[nlocal+0+nx+1+(iy)*nx]
-xv[nlocal+0+nx+1+(iy)*nx+1]
-xv[nlocal+0+nx+1+(iy+1)*nx]
-xv[nlocal+0+nx+1+(iy+1)*nx+1]
;}else if(iz==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+(iy-1)*nx+nx-2]
-xv[nlocal+0+nx+1+(iy-1)*nx+nx-1]
-xv[nlocal+0+nx+1+(iy)*nx+nx-2]
-xv[nlocal+0+nx+1+(iy)*nx+nx-1]
-xv[nlocal+0+nx+1+(iy+1)*nx+nx-2]
-xv[nlocal+0+nx+1+(iy+1)*nx+nx-1]
;}
else if(ix==nx-1&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+iy-1]
-xv[nlocal+0+nx+1+nx*ny+iy]
-xv[nlocal+0+nx+1+nx*ny+iy+1]
;}
else if(ix==nx-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny+iy-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny+iy]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny+iy+1]
;}else if(ix==nx-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy+(nz-2)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy+(nz-1)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+iy+(nz-1)*ny+1]
;}
else if(iy==0&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+(iz-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+(iz-1)*nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+(iz)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+(iz)*nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+(iz+1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+(iz+1)*nx+1]
;}else if(iy==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+(iz-1)*nx+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+(iz-1)*nx+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+(iz)*nx+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+(iz)*nx+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+(iz+1)*nx+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+(iz+1)*nx+nx-1]
;}
else if(iy==ny-1&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+(iz-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+(iz-1)*nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+(iz)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+(iz)*nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+(iz+1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+(iz+1)*nx+1]
;}else if(iy==ny-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+(iz)*nx+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+(iz)*nx+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+(iz+1)*nx+nx-1]
;}
else if(ix==nx-1&&iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+iz-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+iz]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+iz+1]
;}
else if(ix==nx-1&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+(iz-1)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+(iz-1)*ny+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+(iz)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+(iz)*ny+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+(iz+1)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+(iz+1)*ny+1]
;}else if(ix==nx-1&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+(iz)*ny+ny-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+(iz)*ny+ny-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+(iz+1)*ny+ny-1]
;}
else if(ix==nx-1&&iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+iz-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+iz]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1+nx*nz+nz+ny*nz+nx*nz+iz+1]
;}
else if (ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
          }//ipz > 0 
        else
{
            if(ipz < npz - 1)
{
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+nx]
-xv[nlocal+0+nx+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*(nz-2)]
-xv[nlocal+0+nx*(nz-2)+1]
-xv[nlocal+0+nx*(nz-1)]
-xv[nlocal+0+nx*(nz-1)+1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*nz+nz+ny*nz]
-xv[nlocal+0+nx*nz+nz+ny*nz+1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*(ny-2)]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*(ny-2)+1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*(ny-1)]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*(ny-1)+1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*(nz-2)]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*(nz-2)+1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*(nz-1)]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*(nz-1)+1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny+ny]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny+ny+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx-2]
-xv[nlocal+0+nx-1]
-xv[nlocal+0+2*nx-2]
-xv[nlocal+0+2*nx-1]
-xv[nlocal+0+nx*nz]
-xv[nlocal+0+nx*nz+1]
-xv[nlocal+0+nx*nz+nz]
-xv[nlocal+0+nx*nz+nz+1]
-xv[nlocal+0+nx*nz+nz+ny]
-xv[nlocal+0+nx*nz+nz+ny+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*(nz-1)-2]
-xv[nlocal+0+nx*(nz-1)-1]
-xv[nlocal+0+nx*nz-2]
-xv[nlocal+0+nx*nz-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx-2]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx-2]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+2*nx-2]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+2*nx-1]
-xv[nlocal+0+nx*nz+nz-2]
-xv[nlocal+0+nx*nz+nz-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx]
-xv[nlocal+0+nx*nz+nz+ny*(nz-2)]
-xv[nlocal+0+nx*nz+nz+ny*(nz-2)+1]
-xv[nlocal+0+nx*nz+nz+ny*(nz-1)]
-xv[nlocal+0+nx*nz+nz+ny*(nz-1)+1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny+1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*nz+nz+ny*nz+nx-2]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+2*nx-2]
-xv[nlocal+0+nx*nz+nz+ny*nz+2*nx-1]
-xv[nlocal+0+nx*nz+nz+ny-2]
-xv[nlocal+0+nx*nz+nz+ny-1]
-xv[nlocal+0+nx*nz+nz+2*ny-2]
-xv[nlocal+0+nx*nz+nz+2*ny-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*(ny-1)-2]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*(ny-1)-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny-2]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*(nz-1)-2]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*(nz-1)-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz-2]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny+ny+nx-2]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny+ny+nx-1]
-xv[nlocal+0+nx*nz+nz+ny*(nz-1)-2]
-xv[nlocal+0+nx*nz+nz+ny*(nz-1)-1]
-xv[nlocal+0+nx*nz+nz+ny*nz-2]
-xv[nlocal+0+nx*nz+nz+ny*nz-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny+ny-2]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny+ny-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz-2]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny+ny+nx]
;
}
else if(iy==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
-xv[nlocal+0+nx+ix-1]
-xv[nlocal+0+nx+ix]
-xv[nlocal+0+nx+ix+1]
;}else if(iy==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix+(nz-2)*nx-1]
-xv[nlocal+0+ix+(nz-2)*nx]
-xv[nlocal+0+ix+(nz-2)*nx+1]
-xv[nlocal+0+ix+(nz-1)*nx-1]
-xv[nlocal+0+ix+(nz-1)*nx]
-xv[nlocal+0+ix+(nz-1)*nx+1]
;}
else if(iy==0&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+ix-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+ix]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+ix+1]
;}
else if(iz==nz-1&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix+1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx+ix-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx+ix]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx+ix+1]
;}else if(iz==nz-1&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix+(ny-2)*nx-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix+(ny-2)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix+(ny-2)*nx+1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix+(ny-1)*nx-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix+(ny-1)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+ix+(ny-1)*nx+1]
;}
else if(iy==ny-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+ny*nz+ix-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+ix]
-xv[nlocal+0+nx*nz+nz+ny*nz+ix+1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+ix-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+ix]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+ix+1]
;}else if(iy==ny-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+ny*nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+ix+(nz-2)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+nx*nz+nz+ny*nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+ix+(nz-1)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+ix+(nz-1)*nx+1]
;}
else if(iy==ny-1&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny+ny+ix-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny+ny+ix]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny+ny+ix+1]
;}
else if(iz==nz-1 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+(iy-1)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+(iy-1)*nx+1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+(iy)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+(iy)*nx+1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+(iy+1)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+(iy+1)*nx+1]
;}else if(iz==nz-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+(iy-1)*nx+nx-2]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+(iy-1)*nx+nx-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+(iy)*nx+nx-2]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+(iy)*nx+nx-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+(iy+1)*nx+nx-2]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+(iy+1)*nx+nx-1]
;}
else if(ix==nx-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+iy-1]
-xv[nlocal+0+nx*nz+nz+iy]
-xv[nlocal+0+nx*nz+nz+iy+1]
-xv[nlocal+0+nx*nz+nz+ny+iy-1]
-xv[nlocal+0+nx*nz+nz+ny+iy]
-xv[nlocal+0+nx*nz+nz+ny+iy+1]
;}else if(ix==nx-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+nx*nz+nz+iy+(nz-2)*ny]
-xv[nlocal+0+nx*nz+nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+nx*nz+nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+nx*nz+nz+iy+(nz-1)*ny]
-xv[nlocal+0+nx*nz+nz+iy+(nz-1)*ny+1]
;}
else if(ix==nx-1&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny+iy-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny+iy]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz+nx+1+nx*ny+iy+1]
;}
else if(iy==0&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*nx]
-xv[nlocal+0+(iz-1)*nx+1]
-xv[nlocal+0+(iz)*nx]
-xv[nlocal+0+(iz)*nx+1]
-xv[nlocal+0+(iz+1)*nx]
-xv[nlocal+0+(iz+1)*nx+1]
;}else if(iy==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*nx+nx-2]
-xv[nlocal+0+(iz-1)*nx+nx-1]
-xv[nlocal+0+(iz)*nx+nx-2]
-xv[nlocal+0+(iz)*nx+nx-1]
-xv[nlocal+0+(iz+1)*nx+nx-2]
-xv[nlocal+0+(iz+1)*nx+nx-1]
;}
else if(iy==ny-1&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+ny*nz+(iz-1)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+(iz-1)*nx+1]
-xv[nlocal+0+nx*nz+nz+ny*nz+(iz)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+(iz)*nx+1]
-xv[nlocal+0+nx*nz+nz+ny*nz+(iz+1)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+(iz+1)*nx+1]
;}else if(iy==ny-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+ny*nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+nx*nz+nz+ny*nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+(iz)*nx+nx-2]
-xv[nlocal+0+nx*nz+nz+ny*nz+(iz)*nx+nx-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+nx*nz+nz+ny*nz+(iz+1)*nx+nx-1]
;}
else if(ix==nx-1&&iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+iz-1]
-xv[nlocal+0+nx*nz+iz]
-xv[nlocal+0+nx*nz+iz+1]
;}
else if(ix==nx-1&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+(iz-1)*ny]
-xv[nlocal+0+nx*nz+nz+(iz-1)*ny+1]
-xv[nlocal+0+nx*nz+nz+(iz)*ny]
-xv[nlocal+0+nx*nz+nz+(iz)*ny+1]
-xv[nlocal+0+nx*nz+nz+(iz+1)*ny]
-xv[nlocal+0+nx*nz+nz+(iz+1)*ny+1]
;}else if(ix==nx-1&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+nx*nz+nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+nx*nz+nz+(iz)*ny+ny-2]
-xv[nlocal+0+nx*nz+nz+(iz)*ny+ny-1]
-xv[nlocal+0+nx*nz+nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+nx*nz+nz+(iz+1)*ny+ny-1]
;}
else if(ix==nx-1&&iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+iz-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+iz]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+iz+1]
;}
else if (ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
            else
{
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+nx]
-xv[nlocal+0+nx+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*(nz-2)]
-xv[nlocal+0+nx*(nz-2)+1]
-xv[nlocal+0+nx*(nz-1)]
-xv[nlocal+0+nx*(nz-1)+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*nz+nz+ny*nz]
-xv[nlocal+0+nx*nz+nz+ny*nz+1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*(nz-2)]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*(nz-2)+1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*(nz-1)]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*(nz-1)+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx-2]
-xv[nlocal+0+nx-1]
-xv[nlocal+0+2*nx-2]
-xv[nlocal+0+2*nx-1]
-xv[nlocal+0+nx*nz]
-xv[nlocal+0+nx*nz+1]
-xv[nlocal+0+nx*nz+nz]
-xv[nlocal+0+nx*nz+nz+1]
-xv[nlocal+0+nx*nz+nz+ny]
-xv[nlocal+0+nx*nz+nz+ny+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*(nz-1)-2]
-xv[nlocal+0+nx*(nz-1)-1]
-xv[nlocal+0+nx*nz-2]
-xv[nlocal+0+nx*nz-1]
-xv[nlocal+0+nx*nz+nz-2]
-xv[nlocal+0+nx*nz+nz-1]
-xv[nlocal+0+nx*nz+nz+ny*(nz-2)]
-xv[nlocal+0+nx*nz+nz+ny*(nz-2)+1]
-xv[nlocal+0+nx*nz+nz+ny*(nz-1)]
-xv[nlocal+0+nx*nz+nz+ny*(nz-1)+1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*nz+nz+ny*nz+nx-2]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+2*nx-2]
-xv[nlocal+0+nx*nz+nz+ny*nz+2*nx-1]
-xv[nlocal+0+nx*nz+nz+ny-2]
-xv[nlocal+0+nx*nz+nz+ny-1]
-xv[nlocal+0+nx*nz+nz+2*ny-2]
-xv[nlocal+0+nx*nz+nz+2*ny-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*(nz-1)-2]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*(nz-1)-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz-2]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz-1]
-xv[nlocal+0+nx*nz+nz+ny*(nz-1)-2]
-xv[nlocal+0+nx*nz+nz+ny*(nz-1)-1]
-xv[nlocal+0+nx*nz+nz+ny*nz-2]
-xv[nlocal+0+nx*nz+nz+ny*nz-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz-2]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+nz-1]
;
}
else if(iy==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
-xv[nlocal+0+nx+ix-1]
-xv[nlocal+0+nx+ix]
-xv[nlocal+0+nx+ix+1]
;}else if(iy==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix+(nz-2)*nx-1]
-xv[nlocal+0+ix+(nz-2)*nx]
-xv[nlocal+0+ix+(nz-2)*nx+1]
-xv[nlocal+0+ix+(nz-1)*nx-1]
-xv[nlocal+0+ix+(nz-1)*nx]
-xv[nlocal+0+ix+(nz-1)*nx+1]
;}
else if(iy==ny-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+ny*nz+ix-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+ix]
-xv[nlocal+0+nx*nz+nz+ny*nz+ix+1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+ix-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+ix]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+ix+1]
;}else if(iy==ny-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+ny*nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+ix+(nz-2)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+nx*nz+nz+ny*nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+ix+(nz-1)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+ix+(nz-1)*nx+1]
;}
else if(ix==nx-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+iy-1]
-xv[nlocal+0+nx*nz+nz+iy]
-xv[nlocal+0+nx*nz+nz+iy+1]
-xv[nlocal+0+nx*nz+nz+ny+iy-1]
-xv[nlocal+0+nx*nz+nz+ny+iy]
-xv[nlocal+0+nx*nz+nz+ny+iy+1]
;}else if(ix==nx-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+nx*nz+nz+iy+(nz-2)*ny]
-xv[nlocal+0+nx*nz+nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+nx*nz+nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+nx*nz+nz+iy+(nz-1)*ny]
-xv[nlocal+0+nx*nz+nz+iy+(nz-1)*ny+1]
;}
else if(iy==0&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*nx]
-xv[nlocal+0+(iz-1)*nx+1]
-xv[nlocal+0+(iz)*nx]
-xv[nlocal+0+(iz)*nx+1]
-xv[nlocal+0+(iz+1)*nx]
-xv[nlocal+0+(iz+1)*nx+1]
;}else if(iy==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*nx+nx-2]
-xv[nlocal+0+(iz-1)*nx+nx-1]
-xv[nlocal+0+(iz)*nx+nx-2]
-xv[nlocal+0+(iz)*nx+nx-1]
-xv[nlocal+0+(iz+1)*nx+nx-2]
-xv[nlocal+0+(iz+1)*nx+nx-1]
;}
else if(iy==ny-1&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+ny*nz+(iz-1)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+(iz-1)*nx+1]
-xv[nlocal+0+nx*nz+nz+ny*nz+(iz)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+(iz)*nx+1]
-xv[nlocal+0+nx*nz+nz+ny*nz+(iz+1)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+(iz+1)*nx+1]
;}else if(iy==ny-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+ny*nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+nx*nz+nz+ny*nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+(iz)*nx+nx-2]
-xv[nlocal+0+nx*nz+nz+ny*nz+(iz)*nx+nx-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+nx*nz+nz+ny*nz+(iz+1)*nx+nx-1]
;}
else if(ix==nx-1&&iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+iz-1]
-xv[nlocal+0+nx*nz+iz]
-xv[nlocal+0+nx*nz+iz+1]
;}
else if(ix==nx-1&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+(iz-1)*ny]
-xv[nlocal+0+nx*nz+nz+(iz-1)*ny+1]
-xv[nlocal+0+nx*nz+nz+(iz)*ny]
-xv[nlocal+0+nx*nz+nz+(iz)*ny+1]
-xv[nlocal+0+nx*nz+nz+(iz+1)*ny]
-xv[nlocal+0+nx*nz+nz+(iz+1)*ny+1]
;}else if(ix==nx-1&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+nx*nz+nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+nx*nz+nz+(iz)*ny+ny-2]
-xv[nlocal+0+nx*nz+nz+(iz)*ny+ny-1]
-xv[nlocal+0+nx*nz+nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+nx*nz+nz+(iz+1)*ny+ny-1]
;}
else if(ix==nx-1&&iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+iz-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+iz]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx*nz+iz+1]
;}
else if (ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
          }//ipz > 0 
        }//ipy < npy - 1 
      else
{
        if(ipz > 0)
{
            if(ipz < npz - 1)
{
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+nx+1+nx*ny+ny]
-xv[nlocal+0+nx+1+nx*ny+ny+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1]
-xv[nlocal+0+nx+1]
-xv[nlocal+0+nx+1+1]
-xv[nlocal+0+nx+1+nx]
-xv[nlocal+0+nx+1+nx+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+1+nx*ny+ny+nx*(nz-2)]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*(nz-2)+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*(nz-1)]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*(nz-1)+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+1+nx*(ny-2)]
-xv[nlocal+0+nx+1+nx*(ny-2)+1]
-xv[nlocal+0+nx+1+nx*(ny-1)]
-xv[nlocal+0+nx+1+nx*(ny-1)+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+nx*(ny-2)]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+nx*(ny-2)+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+nx*(ny-1)]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+nx*(ny-1)+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx-2]
-xv[nlocal+0+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+2*nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+2*nx-1]
-xv[nlocal+0+nx+1+nx-2]
-xv[nlocal+0+nx+1+nx-1]
-xv[nlocal+0+nx+1+2*nx-2]
-xv[nlocal+0+nx+1+2*nx-1]
-xv[nlocal+0+nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+1]
-xv[nlocal+0+nx+1+nx*ny]
-xv[nlocal+0+nx+1+nx*ny+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+1+nx*ny+ny+nx*(nz-1)-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*(nz-1)-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+2*nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+2*nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*(nz-2)]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*(nz-2)+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*(nz-1)]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*(nz-1)+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+nx*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+nx*ny+1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+1+nx*(ny-1)-2]
-xv[nlocal+0+nx+1+nx*(ny-1)-1]
-xv[nlocal+0+nx+1+nx*ny-2]
-xv[nlocal+0+nx+1+nx*ny-1]
-xv[nlocal+0+nx+1+nx*ny+ny-2]
-xv[nlocal+0+nx+1+nx*ny+ny-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+2*ny-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+2*ny-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+nx*(ny-1)-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+nx*(ny-1)-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+nx*ny-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+nx*ny-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*(nz-1)-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*(nz-1)-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+nx*ny+ny-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+nx*ny+ny-1]
;
}
else if(iy==0&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
;}
else if(iy==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+ix-1]
-xv[nlocal+0+nx+1+nx*ny+ny+ix]
-xv[nlocal+0+nx+1+nx*ny+ny+ix+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+ix-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+ix]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+ix+1]
;}else if(iy==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+ix+(nz-2)*nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+ix+(nz-2)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+ix+(nz-2)*nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+ix+(nz-1)*nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+ix+(nz-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+ix+(nz-1)*nx+1]
;}
else if(iy==0&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+ix-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+ix]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+ix+1]
;}
else if(iz==0&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+ix-1]
-xv[nlocal+0+nx+1+ix]
-xv[nlocal+0+nx+1+ix+1]
-xv[nlocal+0+nx+1+nx+ix-1]
-xv[nlocal+0+nx+1+nx+ix]
-xv[nlocal+0+nx+1+nx+ix+1]
;}else if(iz==0&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+ix+(ny-2)*nx-1]
-xv[nlocal+0+nx+1+ix+(ny-2)*nx]
-xv[nlocal+0+nx+1+ix+(ny-2)*nx+1]
-xv[nlocal+0+nx+1+ix+(ny-1)*nx-1]
-xv[nlocal+0+nx+1+ix+(ny-1)*nx]
-xv[nlocal+0+nx+1+ix+(ny-1)*nx+1]
;}
else if(iz==nz-1&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+ix-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+ix]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+ix+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+nx+ix-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+nx+ix]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+nx+ix+1]
;}else if(iz==nz-1&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+ix+(ny-2)*nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+ix+(ny-2)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+ix+(ny-2)*nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+ix+(ny-1)*nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+ix+(ny-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+ix+(ny-1)*nx+1]
;}
else if(iz==0 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+(iy-1)*nx]
-xv[nlocal+0+nx+1+(iy-1)*nx+1]
-xv[nlocal+0+nx+1+(iy)*nx]
-xv[nlocal+0+nx+1+(iy)*nx+1]
-xv[nlocal+0+nx+1+(iy+1)*nx]
-xv[nlocal+0+nx+1+(iy+1)*nx+1]
;}else if(iz==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+(iy-1)*nx+nx-2]
-xv[nlocal+0+nx+1+(iy-1)*nx+nx-1]
-xv[nlocal+0+nx+1+(iy)*nx+nx-2]
-xv[nlocal+0+nx+1+(iy)*nx+nx-1]
-xv[nlocal+0+nx+1+(iy+1)*nx+nx-2]
-xv[nlocal+0+nx+1+(iy+1)*nx+nx-1]
;}
else if(iz==nz-1 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+(iy-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+(iy-1)*nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+(iy)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+(iy)*nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+(iy+1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+(iy+1)*nx+1]
;}else if(iz==nz-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+(iy-1)*nx+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+(iy-1)*nx+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+(iy)*nx+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+(iy)*nx+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+(iy+1)*nx+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+(iy+1)*nx+nx-1]
;}
else if(ix==nx-1&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+iy-1]
-xv[nlocal+0+nx+1+nx*ny+iy]
-xv[nlocal+0+nx+1+nx*ny+iy+1]
;}
else if(ix==nx-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny+iy-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny+iy]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny+iy+1]
;}else if(ix==nx-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy+(nz-2)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy+(nz-1)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy+(nz-1)*ny+1]
;}
else if(ix==nx-1&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+nx*ny+iy-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+nx*ny+iy]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz+nx+1+nx*ny+iy+1]
;}
else if(iy==0&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+(iz-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+(iz-1)*nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+(iz)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+(iz)*nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+(iz+1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+(iz+1)*nx+1]
;}else if(iy==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+(iz-1)*nx+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+(iz-1)*nx+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+(iz)*nx+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+(iz)*nx+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+(iz+1)*nx+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+(iz+1)*nx+nx-1]
;}
else if(ix==nx-1&&iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+iz-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+iz]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+iz+1]
;}
else if(ix==nx-1&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+(iz-1)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+(iz-1)*ny+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+(iz)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+(iz)*ny+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+(iz+1)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+(iz+1)*ny+1]
;}else if(ix==nx-1&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+(iz)*ny+ny-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+(iz)*ny+ny-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+(iz+1)*ny+ny-1]
;}
else if (ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
            else
{
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+nx+1+nx*ny+ny]
-xv[nlocal+0+nx+1+nx*ny+ny+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+1]
-xv[nlocal+0+nx+1]
-xv[nlocal+0+nx+1+1]
-xv[nlocal+0+nx+1+nx]
-xv[nlocal+0+nx+1+nx+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+1+nx*ny+ny+nx*(nz-2)]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*(nz-2)+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*(nz-1)]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*(nz-1)+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+1+nx*(ny-2)]
-xv[nlocal+0+nx+1+nx*(ny-2)+1]
-xv[nlocal+0+nx+1+nx*(ny-1)]
-xv[nlocal+0+nx+1+nx*(ny-1)+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx-2]
-xv[nlocal+0+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+2*nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+2*nx-1]
-xv[nlocal+0+nx+1+nx-2]
-xv[nlocal+0+nx+1+nx-1]
-xv[nlocal+0+nx+1+2*nx-2]
-xv[nlocal+0+nx+1+2*nx-1]
-xv[nlocal+0+nx]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+1]
-xv[nlocal+0+nx+1+nx*ny]
-xv[nlocal+0+nx+1+nx*ny+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+1+nx*ny+ny+nx*(nz-1)-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*(nz-1)-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*(nz-2)]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*(nz-2)+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*(nz-1)]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*(nz-1)+1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+1+nx*(ny-1)-2]
-xv[nlocal+0+nx+1+nx*(ny-1)-1]
-xv[nlocal+0+nx+1+nx*ny-2]
-xv[nlocal+0+nx+1+nx*ny-1]
-xv[nlocal+0+nx+1+nx*ny+ny-2]
-xv[nlocal+0+nx+1+nx*ny+ny-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+2*ny-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+2*ny-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*(nz-1)-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*(nz-1)-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny*nz-1]
;
}
else if(iy==0&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
;}
else if(iy==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+ix-1]
-xv[nlocal+0+nx+1+nx*ny+ny+ix]
-xv[nlocal+0+nx+1+nx*ny+ny+ix+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+ix-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+ix]
-xv[nlocal+0+nx+1+nx*ny+ny+nx+ix+1]
;}else if(iy==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+ix+(nz-2)*nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+ix+(nz-2)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+ix+(nz-2)*nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+ix+(nz-1)*nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+ix+(nz-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+ix+(nz-1)*nx+1]
;}
else if(iz==0&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+ix-1]
-xv[nlocal+0+nx+1+ix]
-xv[nlocal+0+nx+1+ix+1]
-xv[nlocal+0+nx+1+nx+ix-1]
-xv[nlocal+0+nx+1+nx+ix]
-xv[nlocal+0+nx+1+nx+ix+1]
;}else if(iz==0&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+ix+(ny-2)*nx-1]
-xv[nlocal+0+nx+1+ix+(ny-2)*nx]
-xv[nlocal+0+nx+1+ix+(ny-2)*nx+1]
-xv[nlocal+0+nx+1+ix+(ny-1)*nx-1]
-xv[nlocal+0+nx+1+ix+(ny-1)*nx]
-xv[nlocal+0+nx+1+ix+(ny-1)*nx+1]
;}
else if(iz==0 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+(iy-1)*nx]
-xv[nlocal+0+nx+1+(iy-1)*nx+1]
-xv[nlocal+0+nx+1+(iy)*nx]
-xv[nlocal+0+nx+1+(iy)*nx+1]
-xv[nlocal+0+nx+1+(iy+1)*nx]
-xv[nlocal+0+nx+1+(iy+1)*nx+1]
;}else if(iz==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+(iy-1)*nx+nx-2]
-xv[nlocal+0+nx+1+(iy-1)*nx+nx-1]
-xv[nlocal+0+nx+1+(iy)*nx+nx-2]
-xv[nlocal+0+nx+1+(iy)*nx+nx-1]
-xv[nlocal+0+nx+1+(iy+1)*nx+nx-2]
-xv[nlocal+0+nx+1+(iy+1)*nx+nx-1]
;}
else if(ix==nx-1&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+iy-1]
-xv[nlocal+0+nx+1+nx*ny+iy]
-xv[nlocal+0+nx+1+nx*ny+iy+1]
;}
else if(ix==nx-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny+iy-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny+iy]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+ny+iy+1]
;}else if(ix==nx-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy+(nz-2)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy+(nz-1)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+iy+(nz-1)*ny+1]
;}
else if(iy==0&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+(iz-1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+(iz-1)*nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+(iz)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+(iz)*nx+1]
-xv[nlocal+0+nx+1+nx*ny+ny+(iz+1)*nx]
-xv[nlocal+0+nx+1+nx*ny+ny+(iz+1)*nx+1]
;}else if(iy==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+(iz-1)*nx+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+(iz-1)*nx+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+(iz)*nx+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+(iz)*nx+nx-1]
-xv[nlocal+0+nx+1+nx*ny+ny+(iz+1)*nx+nx-2]
-xv[nlocal+0+nx+1+nx*ny+ny+(iz+1)*nx+nx-1]
;}
else if(ix==nx-1&&iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+iz-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+iz]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+iz+1]
;}
else if(ix==nx-1&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+(iz-1)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+(iz-1)*ny+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+(iz)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+(iz)*ny+1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+(iz+1)*ny]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+(iz+1)*ny+1]
;}else if(ix==nx-1&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+(iz)*ny+ny-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+(iz)*ny+ny-1]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+nx+1+nx*ny+ny+nx*nz+nz+(iz+1)*ny+ny-1]
;}
else if (ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
          }//ipz > 0 
        else
{
            if(ipz < npz - 1)
{
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+nx]
-xv[nlocal+0+nx+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*(nz-2)]
-xv[nlocal+0+nx*(nz-2)+1]
-xv[nlocal+0+nx*(nz-1)]
-xv[nlocal+0+nx*(nz-1)+1]
-xv[nlocal+0+nx*nz+nz+ny*nz]
-xv[nlocal+0+nx*nz+nz+ny*nz+1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+nx*(ny-2)]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+nx*(ny-2)+1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+nx*(ny-1)]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+nx*(ny-1)+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx-2]
-xv[nlocal+0+nx-1]
-xv[nlocal+0+2*nx-2]
-xv[nlocal+0+2*nx-1]
-xv[nlocal+0+nx*nz]
-xv[nlocal+0+nx*nz+1]
-xv[nlocal+0+nx*nz+nz]
-xv[nlocal+0+nx*nz+nz+1]
-xv[nlocal+0+nx*nz+nz+ny]
-xv[nlocal+0+nx*nz+nz+ny+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*(nz-1)-2]
-xv[nlocal+0+nx*(nz-1)-1]
-xv[nlocal+0+nx*nz-2]
-xv[nlocal+0+nx*nz-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx-2]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+nx-2]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+nx-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+2*nx-2]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+2*nx-1]
-xv[nlocal+0+nx*nz+nz-2]
-xv[nlocal+0+nx*nz+nz-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx]
-xv[nlocal+0+nx*nz+nz+ny*(nz-2)]
-xv[nlocal+0+nx*nz+nz+ny*(nz-2)+1]
-xv[nlocal+0+nx*nz+nz+ny*(nz-1)]
-xv[nlocal+0+nx*nz+nz+ny*(nz-1)+1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+nx*ny]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+nx*ny+1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*nz+nz+ny-2]
-xv[nlocal+0+nx*nz+nz+ny-1]
-xv[nlocal+0+nx*nz+nz+2*ny-2]
-xv[nlocal+0+nx*nz+nz+2*ny-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+nx*(ny-1)-2]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+nx*(ny-1)-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+nx*ny-2]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+nx*ny-1]
-xv[nlocal+0+nx*nz+nz+ny*(nz-1)-2]
-xv[nlocal+0+nx*nz+nz+ny*(nz-1)-1]
-xv[nlocal+0+nx*nz+nz+ny*nz-2]
-xv[nlocal+0+nx*nz+nz+ny*nz-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+nx*ny+ny-2]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+nx*ny+ny-1]
;
}
else if(iy==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
-xv[nlocal+0+nx+ix-1]
-xv[nlocal+0+nx+ix]
-xv[nlocal+0+nx+ix+1]
;}else if(iy==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix+(nz-2)*nx-1]
-xv[nlocal+0+ix+(nz-2)*nx]
-xv[nlocal+0+ix+(nz-2)*nx+1]
-xv[nlocal+0+ix+(nz-1)*nx-1]
-xv[nlocal+0+ix+(nz-1)*nx]
-xv[nlocal+0+ix+(nz-1)*nx+1]
;}
else if(iy==0&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+ny*nz+ix-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+ix]
-xv[nlocal+0+nx*nz+nz+ny*nz+ix+1]
;}
else if(iz==nz-1&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+ix-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+ix]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+ix+1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+nx+ix-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+nx+ix]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+nx+ix+1]
;}else if(iz==nz-1&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+ix+(ny-2)*nx-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+ix+(ny-2)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+ix+(ny-2)*nx+1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+ix+(ny-1)*nx-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+ix+(ny-1)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+ix+(ny-1)*nx+1]
;}
else if(iz==nz-1 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+(iy-1)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+(iy-1)*nx+1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+(iy)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+(iy)*nx+1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+(iy+1)*nx]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+(iy+1)*nx+1]
;}else if(iz==nz-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+(iy-1)*nx+nx-2]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+(iy-1)*nx+nx-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+(iy)*nx+nx-2]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+(iy)*nx+nx-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+(iy+1)*nx+nx-2]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+(iy+1)*nx+nx-1]
;}
else if(ix==nx-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+iy-1]
-xv[nlocal+0+nx*nz+nz+iy]
-xv[nlocal+0+nx*nz+nz+iy+1]
-xv[nlocal+0+nx*nz+nz+ny+iy-1]
-xv[nlocal+0+nx*nz+nz+ny+iy]
-xv[nlocal+0+nx*nz+nz+ny+iy+1]
;}else if(ix==nx-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+nx*nz+nz+iy+(nz-2)*ny]
-xv[nlocal+0+nx*nz+nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+nx*nz+nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+nx*nz+nz+iy+(nz-1)*ny]
-xv[nlocal+0+nx*nz+nz+iy+(nz-1)*ny+1]
;}
else if(ix==nx-1&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+nx*ny+iy-1]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+nx*ny+iy]
-xv[nlocal+0+nx*nz+nz+ny*nz+nx+1+nx*ny+iy+1]
;}
else if(iy==0&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*nx]
-xv[nlocal+0+(iz-1)*nx+1]
-xv[nlocal+0+(iz)*nx]
-xv[nlocal+0+(iz)*nx+1]
-xv[nlocal+0+(iz+1)*nx]
-xv[nlocal+0+(iz+1)*nx+1]
;}else if(iy==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*nx+nx-2]
-xv[nlocal+0+(iz-1)*nx+nx-1]
-xv[nlocal+0+(iz)*nx+nx-2]
-xv[nlocal+0+(iz)*nx+nx-1]
-xv[nlocal+0+(iz+1)*nx+nx-2]
-xv[nlocal+0+(iz+1)*nx+nx-1]
;}
else if(ix==nx-1&&iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+iz-1]
-xv[nlocal+0+nx*nz+iz]
-xv[nlocal+0+nx*nz+iz+1]
;}
else if(ix==nx-1&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+(iz-1)*ny]
-xv[nlocal+0+nx*nz+nz+(iz-1)*ny+1]
-xv[nlocal+0+nx*nz+nz+(iz)*ny]
-xv[nlocal+0+nx*nz+nz+(iz)*ny+1]
-xv[nlocal+0+nx*nz+nz+(iz+1)*ny]
-xv[nlocal+0+nx*nz+nz+(iz+1)*ny+1]
;}else if(ix==nx-1&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+nx*nz+nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+nx*nz+nz+(iz)*ny+ny-2]
-xv[nlocal+0+nx*nz+nz+(iz)*ny+ny-1]
-xv[nlocal+0+nx*nz+nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+nx*nz+nz+(iz+1)*ny+ny-1]
;}
else if (ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
            else
{
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+nx]
-xv[nlocal+0+nx+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*(nz-2)]
-xv[nlocal+0+nx*(nz-2)+1]
-xv[nlocal+0+nx*(nz-1)]
-xv[nlocal+0+nx*(nz-1)+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx-2]
-xv[nlocal+0+nx-1]
-xv[nlocal+0+2*nx-2]
-xv[nlocal+0+2*nx-1]
-xv[nlocal+0+nx*nz]
-xv[nlocal+0+nx*nz+1]
-xv[nlocal+0+nx*nz+nz]
-xv[nlocal+0+nx*nz+nz+1]
-xv[nlocal+0+nx*nz+nz+ny]
-xv[nlocal+0+nx*nz+nz+ny+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*(nz-1)-2]
-xv[nlocal+0+nx*(nz-1)-1]
-xv[nlocal+0+nx*nz-2]
-xv[nlocal+0+nx*nz-1]
-xv[nlocal+0+nx*nz+nz-2]
-xv[nlocal+0+nx*nz+nz-1]
-xv[nlocal+0+nx*nz+nz+ny*(nz-2)]
-xv[nlocal+0+nx*nz+nz+ny*(nz-2)+1]
-xv[nlocal+0+nx*nz+nz+ny*(nz-1)]
-xv[nlocal+0+nx*nz+nz+ny*(nz-1)+1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*nz+nz+ny-2]
-xv[nlocal+0+nx*nz+nz+ny-1]
-xv[nlocal+0+nx*nz+nz+2*ny-2]
-xv[nlocal+0+nx*nz+nz+2*ny-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*nz+nz+ny*(nz-1)-2]
-xv[nlocal+0+nx*nz+nz+ny*(nz-1)-1]
-xv[nlocal+0+nx*nz+nz+ny*nz-2]
-xv[nlocal+0+nx*nz+nz+ny*nz-1]
;
}
else if(iy==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
-xv[nlocal+0+nx+ix-1]
-xv[nlocal+0+nx+ix]
-xv[nlocal+0+nx+ix+1]
;}else if(iy==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix+(nz-2)*nx-1]
-xv[nlocal+0+ix+(nz-2)*nx]
-xv[nlocal+0+ix+(nz-2)*nx+1]
-xv[nlocal+0+ix+(nz-1)*nx-1]
-xv[nlocal+0+ix+(nz-1)*nx]
-xv[nlocal+0+ix+(nz-1)*nx+1]
;}
else if(ix==nx-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+iy-1]
-xv[nlocal+0+nx*nz+nz+iy]
-xv[nlocal+0+nx*nz+nz+iy+1]
-xv[nlocal+0+nx*nz+nz+ny+iy-1]
-xv[nlocal+0+nx*nz+nz+ny+iy]
-xv[nlocal+0+nx*nz+nz+ny+iy+1]
;}else if(ix==nx-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+iy+(nz-2)*ny-1]
-xv[nlocal+0+nx*nz+nz+iy+(nz-2)*ny]
-xv[nlocal+0+nx*nz+nz+iy+(nz-2)*ny+1]
-xv[nlocal+0+nx*nz+nz+iy+(nz-1)*ny-1]
-xv[nlocal+0+nx*nz+nz+iy+(nz-1)*ny]
-xv[nlocal+0+nx*nz+nz+iy+(nz-1)*ny+1]
;}
else if(iy==0&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*nx]
-xv[nlocal+0+(iz-1)*nx+1]
-xv[nlocal+0+(iz)*nx]
-xv[nlocal+0+(iz)*nx+1]
-xv[nlocal+0+(iz+1)*nx]
-xv[nlocal+0+(iz+1)*nx+1]
;}else if(iy==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*nx+nx-2]
-xv[nlocal+0+(iz-1)*nx+nx-1]
-xv[nlocal+0+(iz)*nx+nx-2]
-xv[nlocal+0+(iz)*nx+nx-1]
-xv[nlocal+0+(iz+1)*nx+nx-2]
-xv[nlocal+0+(iz+1)*nx+nx-1]
;}
else if(ix==nx-1&&iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+iz-1]
-xv[nlocal+0+nx*nz+iz]
-xv[nlocal+0+nx*nz+iz+1]
;}
else if(ix==nx-1&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+(iz-1)*ny]
-xv[nlocal+0+nx*nz+nz+(iz-1)*ny+1]
-xv[nlocal+0+nx*nz+nz+(iz)*ny]
-xv[nlocal+0+nx*nz+nz+(iz)*ny+1]
-xv[nlocal+0+nx*nz+nz+(iz+1)*ny]
-xv[nlocal+0+nx*nz+nz+(iz+1)*ny+1]
;}else if(ix==nx-1&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nz+(iz-1)*ny+ny-2]
-xv[nlocal+0+nx*nz+nz+(iz-1)*ny+ny-1]
-xv[nlocal+0+nx*nz+nz+(iz)*ny+ny-2]
-xv[nlocal+0+nx*nz+nz+(iz)*ny+ny-1]
-xv[nlocal+0+nx*nz+nz+(iz+1)*ny+ny-2]
-xv[nlocal+0+nx*nz+nz+(iz+1)*ny+ny-1]
;}
else if (ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+nx]
-xv[nlocal+0+nx+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*(ny-2)]
-xv[nlocal+0+nx*(ny-2)+1]
-xv[nlocal+0+nx*(ny-1)]
-xv[nlocal+0+nx*(ny-1)+1]
-xv[nlocal+0+nx*ny+ny]
-xv[nlocal+0+nx*ny+ny+1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+nx*(ny-2)]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+nx*(ny-2)+1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+nx*(ny-1)]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+nx*(ny-1)+1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*(nz-2)]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*(nz-2)+1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*(nz-1)]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*(nz-1)+1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+nx*ny+ny]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+nx*ny+ny+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx-2]
-xv[nlocal+0+nx-1]
-xv[nlocal+0+2*nx-2]
-xv[nlocal+0+2*nx-1]
-xv[nlocal+0+nx*ny]
-xv[nlocal+0+nx*ny+1]
-xv[nlocal+0+nx*ny+ny+nx+1]
-xv[nlocal+0+nx*ny+ny+nx+1+1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny]
-xv[nlocal+0+nx*ny+ny+nx+1+ny+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+nx-2]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+nx-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+2*nx-2]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+2*nx-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*(nz-2)]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*(nz-2)+1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*(nz-1)]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*(nz-1)+1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+nx*ny]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+nx*ny+1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*(ny-1)-2]
-xv[nlocal+0+nx*(ny-1)-1]
-xv[nlocal+0+nx*ny-2]
-xv[nlocal+0+nx*ny-1]
-xv[nlocal+0+nx*ny+ny+nx-2]
-xv[nlocal+0+nx*ny+ny+nx-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx-2]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+2*nx-2]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+2*nx-1]
-xv[nlocal+0+nx*ny+ny-2]
-xv[nlocal+0+nx*ny+ny-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny-2]
-xv[nlocal+0+nx*ny+ny+nx+1+ny-1]
-xv[nlocal+0+nx*ny+ny+nx+1+2*ny-2]
-xv[nlocal+0+nx*ny+ny+nx+1+2*ny-1]
-xv[nlocal+0+nx*ny+ny+nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+nx*(ny-1)-2]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+nx*(ny-1)-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+nx*ny-2]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+nx*ny-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*(nz-1)-2]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*(nz-1)-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz-2]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+nx*ny+ny+nx-2]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+nx*ny+ny+nx-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*(nz-1)-2]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*(nz-1)-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz-2]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+nx*ny+ny-2]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+nx*ny+ny-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz-2]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+nx*ny+ny+nx]
;
}
else if(iz==0&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
-xv[nlocal+0+nx+ix-1]
-xv[nlocal+0+nx+ix]
-xv[nlocal+0+nx+ix+1]
;}else if(iz==0&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix+(ny-2)*nx-1]
-xv[nlocal+0+ix+(ny-2)*nx]
-xv[nlocal+0+ix+(ny-2)*nx+1]
-xv[nlocal+0+ix+(ny-1)*nx-1]
-xv[nlocal+0+ix+(ny-1)*nx]
-xv[nlocal+0+ix+(ny-1)*nx+1]
;}
else if(iz==nz-1&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+ix-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+ix]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+ix+1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+nx+ix-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+nx+ix]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+nx+ix+1]
;}else if(iz==nz-1&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+ix+(ny-2)*nx-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+ix+(ny-2)*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+ix+(ny-2)*nx+1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+ix+(ny-1)*nx-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+ix+(ny-1)*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+ix+(ny-1)*nx+1]
;}
else if(iy==ny-1&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+ix-1]
-xv[nlocal+0+nx*ny+ny+ix]
-xv[nlocal+0+nx*ny+ny+ix+1]
;}
else if(iy==ny-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix+1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx+ix-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx+ix]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx+ix+1]
;}else if(iy==ny-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix+(nz-2)*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix+(nz-1)*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix+(nz-1)*nx+1]
;}
else if(iy==ny-1&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+nx*ny+ny+ix-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+nx*ny+ny+ix]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+nx*ny+ny+ix+1]
;}
else if(iz==0 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iy-1)*nx]
-xv[nlocal+0+(iy-1)*nx+1]
-xv[nlocal+0+(iy)*nx]
-xv[nlocal+0+(iy)*nx+1]
-xv[nlocal+0+(iy+1)*nx]
-xv[nlocal+0+(iy+1)*nx+1]
;}else if(iz==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iy-1)*nx+nx-2]
-xv[nlocal+0+(iy-1)*nx+nx-1]
-xv[nlocal+0+(iy)*nx+nx-2]
-xv[nlocal+0+(iy)*nx+nx-1]
-xv[nlocal+0+(iy+1)*nx+nx-2]
-xv[nlocal+0+(iy+1)*nx+nx-1]
;}
else if(iz==nz-1 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+(iy-1)*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+(iy-1)*nx+1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+(iy)*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+(iy)*nx+1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+(iy+1)*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+(iy+1)*nx+1]
;}else if(iz==nz-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+(iy-1)*nx+nx-2]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+(iy-1)*nx+nx-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+(iy)*nx+nx-2]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+(iy)*nx+nx-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+(iy+1)*nx+nx-2]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+(iy+1)*nx+nx-1]
;}
else if(ix==nx-1&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+iy-1]
-xv[nlocal+0+nx*ny+iy]
-xv[nlocal+0+nx*ny+iy+1]
;}
else if(ix==nx-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+iy-1]
-xv[nlocal+0+nx*ny+ny+nx+1+iy]
-xv[nlocal+0+nx*ny+ny+nx+1+iy+1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny+iy-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny+iy]
-xv[nlocal+0+nx*ny+ny+nx+1+ny+iy+1]
;}else if(ix==nx-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+iy+(nz-2)*ny-1]
-xv[nlocal+0+nx*ny+ny+nx+1+iy+(nz-2)*ny]
-xv[nlocal+0+nx*ny+ny+nx+1+iy+(nz-2)*ny+1]
-xv[nlocal+0+nx*ny+ny+nx+1+iy+(nz-1)*ny-1]
-xv[nlocal+0+nx*ny+ny+nx+1+iy+(nz-1)*ny]
-xv[nlocal+0+nx*ny+ny+nx+1+iy+(nz-1)*ny+1]
;}
else if(ix==nx-1&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+nx*ny+iy-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+nx*ny+iy]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz+nx*ny+iy+1]
;}
else if(iy==ny-1&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+(iz-1)*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+(iz-1)*nx+1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+(iz)*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+(iz)*nx+1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+(iz+1)*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+(iz+1)*nx+1]
;}else if(iy==ny-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+(iz)*nx+nx-2]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+(iz)*nx+nx-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+(iz+1)*nx+nx-1]
;}
else if(ix==nx-1&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+(iz-1)*ny]
-xv[nlocal+0+nx*ny+ny+nx+1+(iz-1)*ny+1]
-xv[nlocal+0+nx*ny+ny+nx+1+(iz)*ny]
-xv[nlocal+0+nx*ny+ny+nx+1+(iz)*ny+1]
-xv[nlocal+0+nx*ny+ny+nx+1+(iz+1)*ny]
-xv[nlocal+0+nx*ny+ny+nx+1+(iz+1)*ny+1]
;}else if(ix==nx-1&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+(iz-1)*ny+ny-2]
-xv[nlocal+0+nx*ny+ny+nx+1+(iz-1)*ny+ny-1]
-xv[nlocal+0+nx*ny+ny+nx+1+(iz)*ny+ny-2]
-xv[nlocal+0+nx*ny+ny+nx+1+(iz)*ny+ny-1]
-xv[nlocal+0+nx*ny+ny+nx+1+(iz+1)*ny+ny-2]
-xv[nlocal+0+nx*ny+ny+nx+1+(iz+1)*ny+ny-1]
;}
else if(ix==nx-1&&iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+iz-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+iz]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+iz+1]
;}
else if (ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
            else
{
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+nx]
-xv[nlocal+0+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*(ny-2)]
-xv[nlocal+0+nx*(ny-2)+1]
-xv[nlocal+0+nx*(ny-1)]
-xv[nlocal+0+nx*(ny-1)+1]
-xv[nlocal+0+nx*ny+ny]
-xv[nlocal+0+nx*ny+ny+1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*(nz-2)]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*(nz-2)+1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*(nz-1)]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*(nz-1)+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx-2]
-xv[nlocal+0+nx-1]
-xv[nlocal+0+2*nx-2]
-xv[nlocal+0+2*nx-1]
-xv[nlocal+0+nx*ny]
-xv[nlocal+0+nx*ny+1]
-xv[nlocal+0+nx*ny+ny+nx+1]
-xv[nlocal+0+nx*ny+ny+nx+1+1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny]
-xv[nlocal+0+nx*ny+ny+nx+1+ny+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*ny+ny+nx+1+ny*(nz-2)]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*(nz-2)+1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*(nz-1)]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*(nz-1)+1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*(ny-1)-2]
-xv[nlocal+0+nx*(ny-1)-1]
-xv[nlocal+0+nx*ny-2]
-xv[nlocal+0+nx*ny-1]
-xv[nlocal+0+nx*ny+ny+nx-2]
-xv[nlocal+0+nx*ny+ny+nx-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx-2]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+2*nx-2]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+2*nx-1]
-xv[nlocal+0+nx*ny+ny-2]
-xv[nlocal+0+nx*ny+ny-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny-2]
-xv[nlocal+0+nx*ny+ny+nx+1+ny-1]
-xv[nlocal+0+nx*ny+ny+nx+1+2*ny-2]
-xv[nlocal+0+nx*ny+ny+nx+1+2*ny-1]
-xv[nlocal+0+nx*ny+ny+nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*(nz-1)-2]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*(nz-1)-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz-2]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*(nz-1)-2]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*(nz-1)-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz-2]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz-2]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+nz-1]
;
}
else if(iz==0&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
-xv[nlocal+0+nx+ix-1]
-xv[nlocal+0+nx+ix]
-xv[nlocal+0+nx+ix+1]
;}else if(iz==0&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix+(ny-2)*nx-1]
-xv[nlocal+0+ix+(ny-2)*nx]
-xv[nlocal+0+ix+(ny-2)*nx+1]
-xv[nlocal+0+ix+(ny-1)*nx-1]
-xv[nlocal+0+ix+(ny-1)*nx]
-xv[nlocal+0+ix+(ny-1)*nx+1]
;}
else if(iy==ny-1&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+ix-1]
-xv[nlocal+0+nx*ny+ny+ix]
-xv[nlocal+0+nx*ny+ny+ix+1]
;}
else if(iy==ny-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix+1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx+ix-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx+ix]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx+ix+1]
;}else if(iy==ny-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix+(nz-2)*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix+(nz-1)*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+ix+(nz-1)*nx+1]
;}
else if(iz==0 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iy-1)*nx]
-xv[nlocal+0+(iy-1)*nx+1]
-xv[nlocal+0+(iy)*nx]
-xv[nlocal+0+(iy)*nx+1]
-xv[nlocal+0+(iy+1)*nx]
-xv[nlocal+0+(iy+1)*nx+1]
;}else if(iz==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iy-1)*nx+nx-2]
-xv[nlocal+0+(iy-1)*nx+nx-1]
-xv[nlocal+0+(iy)*nx+nx-2]
-xv[nlocal+0+(iy)*nx+nx-1]
-xv[nlocal+0+(iy+1)*nx+nx-2]
-xv[nlocal+0+(iy+1)*nx+nx-1]
;}
else if(ix==nx-1&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+iy-1]
-xv[nlocal+0+nx*ny+iy]
-xv[nlocal+0+nx*ny+iy+1]
;}
else if(ix==nx-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+iy-1]
-xv[nlocal+0+nx*ny+ny+nx+1+iy]
-xv[nlocal+0+nx*ny+ny+nx+1+iy+1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny+iy-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny+iy]
-xv[nlocal+0+nx*ny+ny+nx+1+ny+iy+1]
;}else if(ix==nx-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+iy+(nz-2)*ny-1]
-xv[nlocal+0+nx*ny+ny+nx+1+iy+(nz-2)*ny]
-xv[nlocal+0+nx*ny+ny+nx+1+iy+(nz-2)*ny+1]
-xv[nlocal+0+nx*ny+ny+nx+1+iy+(nz-1)*ny-1]
-xv[nlocal+0+nx*ny+ny+nx+1+iy+(nz-1)*ny]
-xv[nlocal+0+nx*ny+ny+nx+1+iy+(nz-1)*ny+1]
;}
else if(iy==ny-1&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+(iz-1)*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+(iz-1)*nx+1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+(iz)*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+(iz)*nx+1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+(iz+1)*nx]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+(iz+1)*nx+1]
;}else if(iy==ny-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+(iz)*nx+nx-2]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+(iz)*nx+nx-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+(iz+1)*nx+nx-1]
;}
else if(ix==nx-1&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+(iz-1)*ny]
-xv[nlocal+0+nx*ny+ny+nx+1+(iz-1)*ny+1]
-xv[nlocal+0+nx*ny+ny+nx+1+(iz)*ny]
-xv[nlocal+0+nx*ny+ny+nx+1+(iz)*ny+1]
-xv[nlocal+0+nx*ny+ny+nx+1+(iz+1)*ny]
-xv[nlocal+0+nx*ny+ny+nx+1+(iz+1)*ny+1]
;}else if(ix==nx-1&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+(iz-1)*ny+ny-2]
-xv[nlocal+0+nx*ny+ny+nx+1+(iz-1)*ny+ny-1]
-xv[nlocal+0+nx*ny+ny+nx+1+(iz)*ny+ny-2]
-xv[nlocal+0+nx*ny+ny+nx+1+(iz)*ny+ny-1]
-xv[nlocal+0+nx*ny+ny+nx+1+(iz+1)*ny+ny-2]
-xv[nlocal+0+nx*ny+ny+nx+1+(iz+1)*ny+ny-1]
;}
else if(ix==nx-1&&iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+iz-1]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+iz]
-xv[nlocal+0+nx*ny+ny+nx+1+ny*nz+nx*nz+iz+1]
;}
else if (ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
          }//ipz > 0 
        else
{
            if(ipz < npz - 1)
{
if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*nz+nx*nz+nz]
-xv[nlocal+0+ny*nz+nx*nz+nz+1]
-xv[nlocal+0+ny*nz+nx*nz+nz+nx]
-xv[nlocal+0+ny*nz+nx*nz+nz+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*nz]
-xv[nlocal+0+ny*nz+1]
-xv[nlocal+0+ny*nz+nx]
-xv[nlocal+0+ny*nz+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*nz+nx*nz+nz+nx*(ny-2)]
-xv[nlocal+0+ny*nz+nx*nz+nz+nx*(ny-2)+1]
-xv[nlocal+0+ny*nz+nx*nz+nz+nx*(ny-1)]
-xv[nlocal+0+ny*nz+nx*nz+nz+nx*(ny-1)+1]
-xv[nlocal+0+ny*nz+nx*(nz-2)]
-xv[nlocal+0+ny*nz+nx*(nz-2)+1]
-xv[nlocal+0+ny*nz+nx*(nz-1)]
-xv[nlocal+0+ny*nz+nx*(nz-1)+1]
-xv[nlocal+0+ny*nz+nx*nz+nz+nx*ny+ny]
-xv[nlocal+0+ny*nz+nx*nz+nz+nx*ny+ny+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+ny]
-xv[nlocal+0+ny+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*nz+nx*nz+nz+nx-2]
-xv[nlocal+0+ny*nz+nx*nz+nz+nx-1]
-xv[nlocal+0+ny*nz+nx*nz+nz+2*nx-2]
-xv[nlocal+0+ny*nz+nx*nz+nz+2*nx-1]
-xv[nlocal+0+ny*(nz-2)]
-xv[nlocal+0+ny*(nz-2)+1]
-xv[nlocal+0+ny*(nz-1)]
-xv[nlocal+0+ny*(nz-1)+1]
-xv[nlocal+0+ny*nz+nx*nz+nz+nx*ny]
-xv[nlocal+0+ny*nz+nx*nz+nz+nx*ny+1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*nz+nx-2]
-xv[nlocal+0+ny*nz+nx-1]
-xv[nlocal+0+ny*nz+2*nx-2]
-xv[nlocal+0+ny*nz+2*nx-1]
-xv[nlocal+0+ny-2]
-xv[nlocal+0+ny-1]
-xv[nlocal+0+2*ny-2]
-xv[nlocal+0+2*ny-1]
-xv[nlocal+0+ny*nz+nx*nz]
-xv[nlocal+0+ny*nz+nx*nz+1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*nz+nx*nz+nz+nx*(ny-1)-2]
-xv[nlocal+0+ny*nz+nx*nz+nz+nx*(ny-1)-1]
-xv[nlocal+0+ny*nz+nx*nz+nz+nx*ny-2]
-xv[nlocal+0+ny*nz+nx*nz+nz+nx*ny-1]
-xv[nlocal+0+ny*nz+nx*(nz-1)-2]
-xv[nlocal+0+ny*nz+nx*(nz-1)-1]
-xv[nlocal+0+ny*nz+nx*nz-2]
-xv[nlocal+0+ny*nz+nx*nz-1]
-xv[nlocal+0+ny*nz+nx*nz+nz+nx*ny+ny+nx-2]
-xv[nlocal+0+ny*nz+nx*nz+nz+nx*ny+ny+nx-1]
-xv[nlocal+0+ny*(nz-1)-2]
-xv[nlocal+0+ny*(nz-1)-1]
-xv[nlocal+0+ny*nz-2]
-xv[nlocal+0+ny*nz-1]
-xv[nlocal+0+ny*nz+nx*nz+nz+nx*ny+ny-2]
-xv[nlocal+0+ny*nz+nx*nz+nz+nx*ny+ny-1]
-xv[nlocal+0+ny*nz+nx*nz+nz-2]
-xv[nlocal+0+ny*nz+nx*nz+nz-1]
-xv[nlocal+0+ny*nz+nx*nz+nz+nx*ny+ny+nx]
;
}
else if(iz==nz-1&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+nx*nz+nz+ix-1]
-xv[nlocal+0+ny*nz+nx*nz+nz+ix]
-xv[nlocal+0+ny*nz+nx*nz+nz+ix+1]
-xv[nlocal+0+ny*nz+nx*nz+nz+nx+ix-1]
-xv[nlocal+0+ny*nz+nx*nz+nz+nx+ix]
-xv[nlocal+0+ny*nz+nx*nz+nz+nx+ix+1]
;}else if(iz==nz-1&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+nx*nz+nz+ix+(ny-2)*nx-1]
-xv[nlocal+0+ny*nz+nx*nz+nz+ix+(ny-2)*nx]
-xv[nlocal+0+ny*nz+nx*nz+nz+ix+(ny-2)*nx+1]
-xv[nlocal+0+ny*nz+nx*nz+nz+ix+(ny-1)*nx-1]
-xv[nlocal+0+ny*nz+nx*nz+nz+ix+(ny-1)*nx]
-xv[nlocal+0+ny*nz+nx*nz+nz+ix+(ny-1)*nx+1]
;}
else if(iy==ny-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ix-1]
-xv[nlocal+0+ny*nz+ix]
-xv[nlocal+0+ny*nz+ix+1]
-xv[nlocal+0+ny*nz+nx+ix-1]
-xv[nlocal+0+ny*nz+nx+ix]
-xv[nlocal+0+ny*nz+nx+ix+1]
;}else if(iy==ny-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+ny*nz+ix+(nz-2)*nx]
-xv[nlocal+0+ny*nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+ny*nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+ny*nz+ix+(nz-1)*nx]
-xv[nlocal+0+ny*nz+ix+(nz-1)*nx+1]
;}
else if(iy==ny-1&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+nx*nz+nz+nx*ny+ny+ix-1]
-xv[nlocal+0+ny*nz+nx*nz+nz+nx*ny+ny+ix]
-xv[nlocal+0+ny*nz+nx*nz+nz+nx*ny+ny+ix+1]
;}
else if(iz==nz-1 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+nx*nz+nz+(iy-1)*nx]
-xv[nlocal+0+ny*nz+nx*nz+nz+(iy-1)*nx+1]
-xv[nlocal+0+ny*nz+nx*nz+nz+(iy)*nx]
-xv[nlocal+0+ny*nz+nx*nz+nz+(iy)*nx+1]
-xv[nlocal+0+ny*nz+nx*nz+nz+(iy+1)*nx]
-xv[nlocal+0+ny*nz+nx*nz+nz+(iy+1)*nx+1]
;}else if(iz==nz-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+nx*nz+nz+(iy-1)*nx+nx-2]
-xv[nlocal+0+ny*nz+nx*nz+nz+(iy-1)*nx+nx-1]
-xv[nlocal+0+ny*nz+nx*nz+nz+(iy)*nx+nx-2]
-xv[nlocal+0+ny*nz+nx*nz+nz+(iy)*nx+nx-1]
-xv[nlocal+0+ny*nz+nx*nz+nz+(iy+1)*nx+nx-2]
-xv[nlocal+0+ny*nz+nx*nz+nz+(iy+1)*nx+nx-1]
;}
else if(ix==nx-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1]
-xv[nlocal+0+iy]
-xv[nlocal+0+iy+1]
-xv[nlocal+0+ny+iy-1]
-xv[nlocal+0+ny+iy]
-xv[nlocal+0+ny+iy+1]
;}else if(ix==nx-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy+(nz-2)*ny-1]
-xv[nlocal+0+iy+(nz-2)*ny]
-xv[nlocal+0+iy+(nz-2)*ny+1]
-xv[nlocal+0+iy+(nz-1)*ny-1]
-xv[nlocal+0+iy+(nz-1)*ny]
-xv[nlocal+0+iy+(nz-1)*ny+1]
;}
else if(ix==nx-1&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+nx*nz+nz+nx*ny+iy-1]
-xv[nlocal+0+ny*nz+nx*nz+nz+nx*ny+iy]
-xv[nlocal+0+ny*nz+nx*nz+nz+nx*ny+iy+1]
;}
else if(iy==ny-1&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+(iz-1)*nx]
-xv[nlocal+0+ny*nz+(iz-1)*nx+1]
-xv[nlocal+0+ny*nz+(iz)*nx]
-xv[nlocal+0+ny*nz+(iz)*nx+1]
-xv[nlocal+0+ny*nz+(iz+1)*nx]
-xv[nlocal+0+ny*nz+(iz+1)*nx+1]
;}else if(iy==ny-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+ny*nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+ny*nz+(iz)*nx+nx-2]
-xv[nlocal+0+ny*nz+(iz)*nx+nx-1]
-xv[nlocal+0+ny*nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+ny*nz+(iz+1)*nx+nx-1]
;}
else if(ix==nx-1&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*ny]
-xv[nlocal+0+(iz-1)*ny+1]
-xv[nlocal+0+(iz)*ny]
-xv[nlocal+0+(iz)*ny+1]
-xv[nlocal+0+(iz+1)*ny]
-xv[nlocal+0+(iz+1)*ny+1]
;}else if(ix==nx-1&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*ny+ny-2]
-xv[nlocal+0+(iz-1)*ny+ny-1]
-xv[nlocal+0+(iz)*ny+ny-2]
-xv[nlocal+0+(iz)*ny+ny-1]
-xv[nlocal+0+(iz+1)*ny+ny-2]
-xv[nlocal+0+(iz+1)*ny+ny-1]
;}
else if(ix==nx-1&&iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+nx*nz+iz-1]
-xv[nlocal+0+ny*nz+nx*nz+iz]
-xv[nlocal+0+ny*nz+nx*nz+iz+1]
;}
else if (ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
            else
{
if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*nz]
-xv[nlocal+0+ny*nz+1]
-xv[nlocal+0+ny*nz+nx]
-xv[nlocal+0+ny*nz+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*nz+nx*(nz-2)]
-xv[nlocal+0+ny*nz+nx*(nz-2)+1]
-xv[nlocal+0+ny*nz+nx*(nz-1)]
-xv[nlocal+0+ny*nz+nx*(nz-1)+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+ny]
-xv[nlocal+0+ny+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*(nz-2)]
-xv[nlocal+0+ny*(nz-2)+1]
-xv[nlocal+0+ny*(nz-1)]
-xv[nlocal+0+ny*(nz-1)+1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*nz+nx-2]
-xv[nlocal+0+ny*nz+nx-1]
-xv[nlocal+0+ny*nz+2*nx-2]
-xv[nlocal+0+ny*nz+2*nx-1]
-xv[nlocal+0+ny-2]
-xv[nlocal+0+ny-1]
-xv[nlocal+0+2*ny-2]
-xv[nlocal+0+2*ny-1]
-xv[nlocal+0+ny*nz+nx*nz]
-xv[nlocal+0+ny*nz+nx*nz+1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*nz+nx*(nz-1)-2]
-xv[nlocal+0+ny*nz+nx*(nz-1)-1]
-xv[nlocal+0+ny*nz+nx*nz-2]
-xv[nlocal+0+ny*nz+nx*nz-1]
-xv[nlocal+0+ny*(nz-1)-2]
-xv[nlocal+0+ny*(nz-1)-1]
-xv[nlocal+0+ny*nz-2]
-xv[nlocal+0+ny*nz-1]
-xv[nlocal+0+ny*nz+nx*nz+nz-2]
-xv[nlocal+0+ny*nz+nx*nz+nz-1]
;
}
else if(iy==ny-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ix-1]
-xv[nlocal+0+ny*nz+ix]
-xv[nlocal+0+ny*nz+ix+1]
-xv[nlocal+0+ny*nz+nx+ix-1]
-xv[nlocal+0+ny*nz+nx+ix]
-xv[nlocal+0+ny*nz+nx+ix+1]
;}else if(iy==ny-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+ny*nz+ix+(nz-2)*nx]
-xv[nlocal+0+ny*nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+ny*nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+ny*nz+ix+(nz-1)*nx]
-xv[nlocal+0+ny*nz+ix+(nz-1)*nx+1]
;}
else if(ix==nx-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1]
-xv[nlocal+0+iy]
-xv[nlocal+0+iy+1]
-xv[nlocal+0+ny+iy-1]
-xv[nlocal+0+ny+iy]
-xv[nlocal+0+ny+iy+1]
;}else if(ix==nx-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy+(nz-2)*ny-1]
-xv[nlocal+0+iy+(nz-2)*ny]
-xv[nlocal+0+iy+(nz-2)*ny+1]
-xv[nlocal+0+iy+(nz-1)*ny-1]
-xv[nlocal+0+iy+(nz-1)*ny]
-xv[nlocal+0+iy+(nz-1)*ny+1]
;}
else if(iy==ny-1&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+(iz-1)*nx]
-xv[nlocal+0+ny*nz+(iz-1)*nx+1]
-xv[nlocal+0+ny*nz+(iz)*nx]
-xv[nlocal+0+ny*nz+(iz)*nx+1]
-xv[nlocal+0+ny*nz+(iz+1)*nx]
-xv[nlocal+0+ny*nz+(iz+1)*nx+1]
;}else if(iy==ny-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+ny*nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+ny*nz+(iz)*nx+nx-2]
-xv[nlocal+0+ny*nz+(iz)*nx+nx-1]
-xv[nlocal+0+ny*nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+ny*nz+(iz+1)*nx+nx-1]
;}
else if(ix==nx-1&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*ny]
-xv[nlocal+0+(iz-1)*ny+1]
-xv[nlocal+0+(iz)*ny]
-xv[nlocal+0+(iz)*ny+1]
-xv[nlocal+0+(iz+1)*ny]
-xv[nlocal+0+(iz+1)*ny+1]
;}else if(ix==nx-1&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*ny+ny-2]
-xv[nlocal+0+(iz-1)*ny+ny-1]
-xv[nlocal+0+(iz)*ny+ny-2]
-xv[nlocal+0+(iz)*ny+ny-1]
-xv[nlocal+0+(iz+1)*ny+ny-2]
-xv[nlocal+0+(iz+1)*ny+ny-1]
;}
else if(ix==nx-1&&iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+nx*nz+iz-1]
-xv[nlocal+0+ny*nz+nx*nz+iz]
-xv[nlocal+0+ny*nz+nx*nz+iz+1]
;}
else if (ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
          }//ipz > 0 
        }//ipy < npy - 1 
      else
{
        if(ipz > 0)
{
            if(ipz < npz - 1)
{
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+nx]
-xv[nlocal+0+nx+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*ny+ny+ny*nz]
-xv[nlocal+0+nx*ny+ny+ny*nz+1]
-xv[nlocal+0+nx*ny+ny+ny*nz+nx]
-xv[nlocal+0+nx*ny+ny+ny*nz+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*(ny-2)]
-xv[nlocal+0+nx*(ny-2)+1]
-xv[nlocal+0+nx*(ny-1)]
-xv[nlocal+0+nx*(ny-1)+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*ny+ny+ny*nz+nx*(ny-2)]
-xv[nlocal+0+nx*ny+ny+ny*nz+nx*(ny-2)+1]
-xv[nlocal+0+nx*ny+ny+ny*nz+nx*(ny-1)]
-xv[nlocal+0+nx*ny+ny+ny*nz+nx*(ny-1)+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx-2]
-xv[nlocal+0+nx-1]
-xv[nlocal+0+2*nx-2]
-xv[nlocal+0+2*nx-1]
-xv[nlocal+0+nx*ny]
-xv[nlocal+0+nx*ny+1]
-xv[nlocal+0+nx*ny+ny]
-xv[nlocal+0+nx*ny+ny+1]
-xv[nlocal+0+nx*ny+ny+ny]
-xv[nlocal+0+nx*ny+ny+ny+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*ny+ny+ny*nz+nx-2]
-xv[nlocal+0+nx*ny+ny+ny*nz+nx-1]
-xv[nlocal+0+nx*ny+ny+ny*nz+2*nx-2]
-xv[nlocal+0+nx*ny+ny+ny*nz+2*nx-1]
-xv[nlocal+0+nx*ny+ny+ny*(nz-2)]
-xv[nlocal+0+nx*ny+ny+ny*(nz-2)+1]
-xv[nlocal+0+nx*ny+ny+ny*(nz-1)]
-xv[nlocal+0+nx*ny+ny+ny*(nz-1)+1]
-xv[nlocal+0+nx*ny+ny+ny*nz+nx*ny]
-xv[nlocal+0+nx*ny+ny+ny*nz+nx*ny+1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*(ny-1)-2]
-xv[nlocal+0+nx*(ny-1)-1]
-xv[nlocal+0+nx*ny-2]
-xv[nlocal+0+nx*ny-1]
-xv[nlocal+0+nx*ny+ny-2]
-xv[nlocal+0+nx*ny+ny-1]
-xv[nlocal+0+nx*ny+ny+ny-2]
-xv[nlocal+0+nx*ny+ny+ny-1]
-xv[nlocal+0+nx*ny+ny+2*ny-2]
-xv[nlocal+0+nx*ny+ny+2*ny-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*ny+ny+ny*nz+nx*(ny-1)-2]
-xv[nlocal+0+nx*ny+ny+ny*nz+nx*(ny-1)-1]
-xv[nlocal+0+nx*ny+ny+ny*nz+nx*ny-2]
-xv[nlocal+0+nx*ny+ny+ny*nz+nx*ny-1]
-xv[nlocal+0+nx*ny+ny+ny*(nz-1)-2]
-xv[nlocal+0+nx*ny+ny+ny*(nz-1)-1]
-xv[nlocal+0+nx*ny+ny+ny*nz-2]
-xv[nlocal+0+nx*ny+ny+ny*nz-1]
-xv[nlocal+0+nx*ny+ny+ny*nz+nx*ny+ny-2]
-xv[nlocal+0+nx*ny+ny+ny*nz+nx*ny+ny-1]
;
}
else if(iz==0&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
-xv[nlocal+0+nx+ix-1]
-xv[nlocal+0+nx+ix]
-xv[nlocal+0+nx+ix+1]
;}else if(iz==0&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix+(ny-2)*nx-1]
-xv[nlocal+0+ix+(ny-2)*nx]
-xv[nlocal+0+ix+(ny-2)*nx+1]
-xv[nlocal+0+ix+(ny-1)*nx-1]
-xv[nlocal+0+ix+(ny-1)*nx]
-xv[nlocal+0+ix+(ny-1)*nx+1]
;}
else if(iz==nz-1&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+ny*nz+ix-1]
-xv[nlocal+0+nx*ny+ny+ny*nz+ix]
-xv[nlocal+0+nx*ny+ny+ny*nz+ix+1]
-xv[nlocal+0+nx*ny+ny+ny*nz+nx+ix-1]
-xv[nlocal+0+nx*ny+ny+ny*nz+nx+ix]
-xv[nlocal+0+nx*ny+ny+ny*nz+nx+ix+1]
;}else if(iz==nz-1&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+ny*nz+ix+(ny-2)*nx-1]
-xv[nlocal+0+nx*ny+ny+ny*nz+ix+(ny-2)*nx]
-xv[nlocal+0+nx*ny+ny+ny*nz+ix+(ny-2)*nx+1]
-xv[nlocal+0+nx*ny+ny+ny*nz+ix+(ny-1)*nx-1]
-xv[nlocal+0+nx*ny+ny+ny*nz+ix+(ny-1)*nx]
-xv[nlocal+0+nx*ny+ny+ny*nz+ix+(ny-1)*nx+1]
;}
else if(iz==0 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iy-1)*nx]
-xv[nlocal+0+(iy-1)*nx+1]
-xv[nlocal+0+(iy)*nx]
-xv[nlocal+0+(iy)*nx+1]
-xv[nlocal+0+(iy+1)*nx]
-xv[nlocal+0+(iy+1)*nx+1]
;}else if(iz==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iy-1)*nx+nx-2]
-xv[nlocal+0+(iy-1)*nx+nx-1]
-xv[nlocal+0+(iy)*nx+nx-2]
-xv[nlocal+0+(iy)*nx+nx-1]
-xv[nlocal+0+(iy+1)*nx+nx-2]
-xv[nlocal+0+(iy+1)*nx+nx-1]
;}
else if(iz==nz-1 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+ny*nz+(iy-1)*nx]
-xv[nlocal+0+nx*ny+ny+ny*nz+(iy-1)*nx+1]
-xv[nlocal+0+nx*ny+ny+ny*nz+(iy)*nx]
-xv[nlocal+0+nx*ny+ny+ny*nz+(iy)*nx+1]
-xv[nlocal+0+nx*ny+ny+ny*nz+(iy+1)*nx]
-xv[nlocal+0+nx*ny+ny+ny*nz+(iy+1)*nx+1]
;}else if(iz==nz-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+ny*nz+(iy-1)*nx+nx-2]
-xv[nlocal+0+nx*ny+ny+ny*nz+(iy-1)*nx+nx-1]
-xv[nlocal+0+nx*ny+ny+ny*nz+(iy)*nx+nx-2]
-xv[nlocal+0+nx*ny+ny+ny*nz+(iy)*nx+nx-1]
-xv[nlocal+0+nx*ny+ny+ny*nz+(iy+1)*nx+nx-2]
-xv[nlocal+0+nx*ny+ny+ny*nz+(iy+1)*nx+nx-1]
;}
else if(ix==nx-1&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+iy-1]
-xv[nlocal+0+nx*ny+iy]
-xv[nlocal+0+nx*ny+iy+1]
;}
else if(ix==nx-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+iy-1]
-xv[nlocal+0+nx*ny+ny+iy]
-xv[nlocal+0+nx*ny+ny+iy+1]
-xv[nlocal+0+nx*ny+ny+ny+iy-1]
-xv[nlocal+0+nx*ny+ny+ny+iy]
-xv[nlocal+0+nx*ny+ny+ny+iy+1]
;}else if(ix==nx-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+iy+(nz-2)*ny-1]
-xv[nlocal+0+nx*ny+ny+iy+(nz-2)*ny]
-xv[nlocal+0+nx*ny+ny+iy+(nz-2)*ny+1]
-xv[nlocal+0+nx*ny+ny+iy+(nz-1)*ny-1]
-xv[nlocal+0+nx*ny+ny+iy+(nz-1)*ny]
-xv[nlocal+0+nx*ny+ny+iy+(nz-1)*ny+1]
;}
else if(ix==nx-1&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+ny*nz+nx*ny+iy-1]
-xv[nlocal+0+nx*ny+ny+ny*nz+nx*ny+iy]
-xv[nlocal+0+nx*ny+ny+ny*nz+nx*ny+iy+1]
;}
else if(ix==nx-1&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+(iz-1)*ny]
-xv[nlocal+0+nx*ny+ny+(iz-1)*ny+1]
-xv[nlocal+0+nx*ny+ny+(iz)*ny]
-xv[nlocal+0+nx*ny+ny+(iz)*ny+1]
-xv[nlocal+0+nx*ny+ny+(iz+1)*ny]
-xv[nlocal+0+nx*ny+ny+(iz+1)*ny+1]
;}else if(ix==nx-1&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+(iz-1)*ny+ny-2]
-xv[nlocal+0+nx*ny+ny+(iz-1)*ny+ny-1]
-xv[nlocal+0+nx*ny+ny+(iz)*ny+ny-2]
-xv[nlocal+0+nx*ny+ny+(iz)*ny+ny-1]
-xv[nlocal+0+nx*ny+ny+(iz+1)*ny+ny-2]
-xv[nlocal+0+nx*ny+ny+(iz+1)*ny+ny-1]
;}
else if (ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
            else
{
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+nx]
-xv[nlocal+0+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*(ny-2)]
-xv[nlocal+0+nx*(ny-2)+1]
-xv[nlocal+0+nx*(ny-1)]
-xv[nlocal+0+nx*(ny-1)+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx-2]
-xv[nlocal+0+nx-1]
-xv[nlocal+0+2*nx-2]
-xv[nlocal+0+2*nx-1]
-xv[nlocal+0+nx*ny]
-xv[nlocal+0+nx*ny+1]
-xv[nlocal+0+nx*ny+ny]
-xv[nlocal+0+nx*ny+ny+1]
-xv[nlocal+0+nx*ny+ny+ny]
-xv[nlocal+0+nx*ny+ny+ny+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*ny+ny+ny*(nz-2)]
-xv[nlocal+0+nx*ny+ny+ny*(nz-2)+1]
-xv[nlocal+0+nx*ny+ny+ny*(nz-1)]
-xv[nlocal+0+nx*ny+ny+ny*(nz-1)+1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*(ny-1)-2]
-xv[nlocal+0+nx*(ny-1)-1]
-xv[nlocal+0+nx*ny-2]
-xv[nlocal+0+nx*ny-1]
-xv[nlocal+0+nx*ny+ny-2]
-xv[nlocal+0+nx*ny+ny-1]
-xv[nlocal+0+nx*ny+ny+ny-2]
-xv[nlocal+0+nx*ny+ny+ny-1]
-xv[nlocal+0+nx*ny+ny+2*ny-2]
-xv[nlocal+0+nx*ny+ny+2*ny-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*ny+ny+ny*(nz-1)-2]
-xv[nlocal+0+nx*ny+ny+ny*(nz-1)-1]
-xv[nlocal+0+nx*ny+ny+ny*nz-2]
-xv[nlocal+0+nx*ny+ny+ny*nz-1]
;
}
else if(iz==0&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
-xv[nlocal+0+nx+ix-1]
-xv[nlocal+0+nx+ix]
-xv[nlocal+0+nx+ix+1]
;}else if(iz==0&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix+(ny-2)*nx-1]
-xv[nlocal+0+ix+(ny-2)*nx]
-xv[nlocal+0+ix+(ny-2)*nx+1]
-xv[nlocal+0+ix+(ny-1)*nx-1]
-xv[nlocal+0+ix+(ny-1)*nx]
-xv[nlocal+0+ix+(ny-1)*nx+1]
;}
else if(iz==0 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iy-1)*nx]
-xv[nlocal+0+(iy-1)*nx+1]
-xv[nlocal+0+(iy)*nx]
-xv[nlocal+0+(iy)*nx+1]
-xv[nlocal+0+(iy+1)*nx]
-xv[nlocal+0+(iy+1)*nx+1]
;}else if(iz==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iy-1)*nx+nx-2]
-xv[nlocal+0+(iy-1)*nx+nx-1]
-xv[nlocal+0+(iy)*nx+nx-2]
-xv[nlocal+0+(iy)*nx+nx-1]
-xv[nlocal+0+(iy+1)*nx+nx-2]
-xv[nlocal+0+(iy+1)*nx+nx-1]
;}
else if(ix==nx-1&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+iy-1]
-xv[nlocal+0+nx*ny+iy]
-xv[nlocal+0+nx*ny+iy+1]
;}
else if(ix==nx-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+iy-1]
-xv[nlocal+0+nx*ny+ny+iy]
-xv[nlocal+0+nx*ny+ny+iy+1]
-xv[nlocal+0+nx*ny+ny+ny+iy-1]
-xv[nlocal+0+nx*ny+ny+ny+iy]
-xv[nlocal+0+nx*ny+ny+ny+iy+1]
;}else if(ix==nx-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+iy+(nz-2)*ny-1]
-xv[nlocal+0+nx*ny+ny+iy+(nz-2)*ny]
-xv[nlocal+0+nx*ny+ny+iy+(nz-2)*ny+1]
-xv[nlocal+0+nx*ny+ny+iy+(nz-1)*ny-1]
-xv[nlocal+0+nx*ny+ny+iy+(nz-1)*ny]
-xv[nlocal+0+nx*ny+ny+iy+(nz-1)*ny+1]
;}
else if(ix==nx-1&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+(iz-1)*ny]
-xv[nlocal+0+nx*ny+ny+(iz-1)*ny+1]
-xv[nlocal+0+nx*ny+ny+(iz)*ny]
-xv[nlocal+0+nx*ny+ny+(iz)*ny+1]
-xv[nlocal+0+nx*ny+ny+(iz+1)*ny]
-xv[nlocal+0+nx*ny+ny+(iz+1)*ny+1]
;}else if(ix==nx-1&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ny+(iz-1)*ny+ny-2]
-xv[nlocal+0+nx*ny+ny+(iz-1)*ny+ny-1]
-xv[nlocal+0+nx*ny+ny+(iz)*ny+ny-2]
-xv[nlocal+0+nx*ny+ny+(iz)*ny+ny-1]
-xv[nlocal+0+nx*ny+ny+(iz+1)*ny+ny-2]
-xv[nlocal+0+nx*ny+ny+(iz+1)*ny+ny-1]
;}
else if (ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
          }//ipz > 0 
        else
{
            if(ipz < npz - 1)
{
if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*nz]
-xv[nlocal+0+ny*nz+1]
-xv[nlocal+0+ny*nz+nx]
-xv[nlocal+0+ny*nz+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*nz+nx*(ny-2)]
-xv[nlocal+0+ny*nz+nx*(ny-2)+1]
-xv[nlocal+0+ny*nz+nx*(ny-1)]
-xv[nlocal+0+ny*nz+nx*(ny-1)+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+ny]
-xv[nlocal+0+ny+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*nz+nx-2]
-xv[nlocal+0+ny*nz+nx-1]
-xv[nlocal+0+ny*nz+2*nx-2]
-xv[nlocal+0+ny*nz+2*nx-1]
-xv[nlocal+0+ny*(nz-2)]
-xv[nlocal+0+ny*(nz-2)+1]
-xv[nlocal+0+ny*(nz-1)]
-xv[nlocal+0+ny*(nz-1)+1]
-xv[nlocal+0+ny*nz+nx*ny]
-xv[nlocal+0+ny*nz+nx*ny+1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny-2]
-xv[nlocal+0+ny-1]
-xv[nlocal+0+2*ny-2]
-xv[nlocal+0+2*ny-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*nz+nx*(ny-1)-2]
-xv[nlocal+0+ny*nz+nx*(ny-1)-1]
-xv[nlocal+0+ny*nz+nx*ny-2]
-xv[nlocal+0+ny*nz+nx*ny-1]
-xv[nlocal+0+ny*(nz-1)-2]
-xv[nlocal+0+ny*(nz-1)-1]
-xv[nlocal+0+ny*nz-2]
-xv[nlocal+0+ny*nz-1]
-xv[nlocal+0+ny*nz+nx*ny+ny-2]
-xv[nlocal+0+ny*nz+nx*ny+ny-1]
;
}
else if(iz==nz-1&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ix-1]
-xv[nlocal+0+ny*nz+ix]
-xv[nlocal+0+ny*nz+ix+1]
-xv[nlocal+0+ny*nz+nx+ix-1]
-xv[nlocal+0+ny*nz+nx+ix]
-xv[nlocal+0+ny*nz+nx+ix+1]
;}else if(iz==nz-1&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+ix+(ny-2)*nx-1]
-xv[nlocal+0+ny*nz+ix+(ny-2)*nx]
-xv[nlocal+0+ny*nz+ix+(ny-2)*nx+1]
-xv[nlocal+0+ny*nz+ix+(ny-1)*nx-1]
-xv[nlocal+0+ny*nz+ix+(ny-1)*nx]
-xv[nlocal+0+ny*nz+ix+(ny-1)*nx+1]
;}
else if(iz==nz-1 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+(iy-1)*nx]
-xv[nlocal+0+ny*nz+(iy-1)*nx+1]
-xv[nlocal+0+ny*nz+(iy)*nx]
-xv[nlocal+0+ny*nz+(iy)*nx+1]
-xv[nlocal+0+ny*nz+(iy+1)*nx]
-xv[nlocal+0+ny*nz+(iy+1)*nx+1]
;}else if(iz==nz-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+(iy-1)*nx+nx-2]
-xv[nlocal+0+ny*nz+(iy-1)*nx+nx-1]
-xv[nlocal+0+ny*nz+(iy)*nx+nx-2]
-xv[nlocal+0+ny*nz+(iy)*nx+nx-1]
-xv[nlocal+0+ny*nz+(iy+1)*nx+nx-2]
-xv[nlocal+0+ny*nz+(iy+1)*nx+nx-1]
;}
else if(ix==nx-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1]
-xv[nlocal+0+iy]
-xv[nlocal+0+iy+1]
-xv[nlocal+0+ny+iy-1]
-xv[nlocal+0+ny+iy]
-xv[nlocal+0+ny+iy+1]
;}else if(ix==nx-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy+(nz-2)*ny-1]
-xv[nlocal+0+iy+(nz-2)*ny]
-xv[nlocal+0+iy+(nz-2)*ny+1]
-xv[nlocal+0+iy+(nz-1)*ny-1]
-xv[nlocal+0+iy+(nz-1)*ny]
-xv[nlocal+0+iy+(nz-1)*ny+1]
;}
else if(ix==nx-1&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ny*nz+nx*ny+iy-1]
-xv[nlocal+0+ny*nz+nx*ny+iy]
-xv[nlocal+0+ny*nz+nx*ny+iy+1]
;}
else if(ix==nx-1&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*ny]
-xv[nlocal+0+(iz-1)*ny+1]
-xv[nlocal+0+(iz)*ny]
-xv[nlocal+0+(iz)*ny+1]
-xv[nlocal+0+(iz+1)*ny]
-xv[nlocal+0+(iz+1)*ny+1]
;}else if(ix==nx-1&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*ny+ny-2]
-xv[nlocal+0+(iz-1)*ny+ny-1]
-xv[nlocal+0+(iz)*ny+ny-2]
-xv[nlocal+0+(iz)*ny+ny-1]
-xv[nlocal+0+(iz+1)*ny+ny-2]
-xv[nlocal+0+(iz+1)*ny+ny-1]
;}
else if (ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
            else
{
if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+ny]
-xv[nlocal+0+ny+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*(nz-2)]
-xv[nlocal+0+ny*(nz-2)+1]
-xv[nlocal+0+ny*(nz-1)]
-xv[nlocal+0+ny*(nz-1)+1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny-2]
-xv[nlocal+0+ny-1]
-xv[nlocal+0+2*ny-2]
-xv[nlocal+0+2*ny-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+ny*(nz-1)-2]
-xv[nlocal+0+ny*(nz-1)-1]
-xv[nlocal+0+ny*nz-2]
-xv[nlocal+0+ny*nz-1]
;
}
else if(ix==nx-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy-1]
-xv[nlocal+0+iy]
-xv[nlocal+0+iy+1]
-xv[nlocal+0+ny+iy-1]
-xv[nlocal+0+ny+iy]
-xv[nlocal+0+ny+iy+1]
;}else if(ix==nx-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+iy+(nz-2)*ny-1]
-xv[nlocal+0+iy+(nz-2)*ny]
-xv[nlocal+0+iy+(nz-2)*ny+1]
-xv[nlocal+0+iy+(nz-1)*ny-1]
-xv[nlocal+0+iy+(nz-1)*ny]
-xv[nlocal+0+iy+(nz-1)*ny+1]
;}
else if(ix==nx-1&&iy==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*ny]
-xv[nlocal+0+(iz-1)*ny+1]
-xv[nlocal+0+(iz)*ny]
-xv[nlocal+0+(iz)*ny+1]
-xv[nlocal+0+(iz+1)*ny]
-xv[nlocal+0+(iz+1)*ny+1]
;}else if(ix==nx-1&&iy==ny-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*ny+ny-2]
-xv[nlocal+0+(iz-1)*ny+ny-1]
-xv[nlocal+0+(iz)*ny+ny-2]
-xv[nlocal+0+(iz)*ny+ny-1]
-xv[nlocal+0+(iz+1)*ny+ny-2]
-xv[nlocal+0+(iz+1)*ny+ny-1]
;}
else if (ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+nx+nx*ny+nx]
-xv[nlocal+0+nx+nx*ny+nx+1]
-xv[nlocal+0+nx+nx*ny+nx+nx]
-xv[nlocal+0+nx+nx*ny+nx+nx+1]
-xv[nlocal+0+nx]
-xv[nlocal+0+nx+1]
-xv[nlocal+0+nx+nx]
-xv[nlocal+0+nx+nx+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+nx*ny+nx+nx*(nz-2)]
-xv[nlocal+0+nx+nx*ny+nx+nx*(nz-2)+1]
-xv[nlocal+0+nx+nx*ny+nx+nx*(nz-1)]
-xv[nlocal+0+nx+nx*ny+nx+nx*(nz-1)+1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+nx*(ny-2)]
-xv[nlocal+0+nx+nx*(ny-2)+1]
-xv[nlocal+0+nx+nx*(ny-1)]
-xv[nlocal+0+nx+nx*(ny-1)+1]
-xv[nlocal+0+nx+nx*ny]
-xv[nlocal+0+nx+nx*ny+1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+nx*(ny-2)]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+nx*(ny-2)+1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+nx*(ny-1)]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+nx*(ny-1)+1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*(nz-2)]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*(nz-2)+1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*(nz-1)]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*(nz-1)+1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+nx*ny]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+nx*ny+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx-2]
-xv[nlocal+0+nx-1]
-xv[nlocal+0+nx+nx*ny+nx+nx-2]
-xv[nlocal+0+nx+nx*ny+nx+nx-1]
-xv[nlocal+0+nx+nx*ny+nx+2*nx-2]
-xv[nlocal+0+nx+nx*ny+nx+2*nx-1]
-xv[nlocal+0+nx+nx-2]
-xv[nlocal+0+nx+nx-1]
-xv[nlocal+0+nx+2*nx-2]
-xv[nlocal+0+nx+2*nx-1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+nx*ny+nx+nx*(nz-1)-2]
-xv[nlocal+0+nx+nx*ny+nx+nx*(nz-1)-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz-2]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx-2]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+nx-2]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+nx-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+2*nx-2]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+2*nx-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+nx*(ny-1)-2]
-xv[nlocal+0+nx+nx*(ny-1)-1]
-xv[nlocal+0+nx+nx*ny-2]
-xv[nlocal+0+nx+nx*ny-1]
-xv[nlocal+0+nx+nx*ny+nx-2]
-xv[nlocal+0+nx+nx*ny+nx-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx-2]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+2*nx-2]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+2*nx-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+nx*(ny-1)-2]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+nx*(ny-1)-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+nx*ny-2]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+nx*ny-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*(nz-1)-2]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*(nz-1)-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz-2]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+nx*ny+nx-2]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+nx*ny+nx-1]
;
}
else if(iy==0&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
;}
else if(iy==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx+ix-1]
-xv[nlocal+0+nx+nx*ny+nx+ix]
-xv[nlocal+0+nx+nx*ny+nx+ix+1]
-xv[nlocal+0+nx+nx*ny+nx+nx+ix-1]
-xv[nlocal+0+nx+nx*ny+nx+nx+ix]
-xv[nlocal+0+nx+nx*ny+nx+nx+ix+1]
;}else if(iy==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx+ix+(nz-2)*nx-1]
-xv[nlocal+0+nx+nx*ny+nx+ix+(nz-2)*nx]
-xv[nlocal+0+nx+nx*ny+nx+ix+(nz-2)*nx+1]
-xv[nlocal+0+nx+nx*ny+nx+ix+(nz-1)*nx-1]
-xv[nlocal+0+nx+nx*ny+nx+ix+(nz-1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+ix+(nz-1)*nx+1]
;}
else if(iy==0&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+ix-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+ix]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+ix+1]
;}
else if(iz==0&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+ix-1]
-xv[nlocal+0+nx+ix]
-xv[nlocal+0+nx+ix+1]
-xv[nlocal+0+nx+nx+ix-1]
-xv[nlocal+0+nx+nx+ix]
-xv[nlocal+0+nx+nx+ix+1]
;}else if(iz==0&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+ix+(ny-2)*nx-1]
-xv[nlocal+0+nx+ix+(ny-2)*nx]
-xv[nlocal+0+nx+ix+(ny-2)*nx+1]
-xv[nlocal+0+nx+ix+(ny-1)*nx-1]
-xv[nlocal+0+nx+ix+(ny-1)*nx]
-xv[nlocal+0+nx+ix+(ny-1)*nx+1]
;}
else if(iz==nz-1&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+ix-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+ix]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+ix+1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+nx+ix-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+nx+ix]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+nx+ix+1]
;}else if(iz==nz-1&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+ix+(ny-2)*nx-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+ix+(ny-2)*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+ix+(ny-2)*nx+1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+ix+(ny-1)*nx-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+ix+(ny-1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+ix+(ny-1)*nx+1]
;}
else if(iy==ny-1&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+ix-1]
-xv[nlocal+0+nx+nx*ny+ix]
-xv[nlocal+0+nx+nx*ny+ix+1]
;}
else if(iy==ny-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix+1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx+ix-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx+ix]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx+ix+1]
;}else if(iy==ny-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix+(nz-2)*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix+(nz-1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix+(nz-1)*nx+1]
;}
else if(iy==ny-1&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+nx*ny+ix-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+nx*ny+ix]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+nx*ny+ix+1]
;}
else if(iz==0 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+(iy-1)*nx]
-xv[nlocal+0+nx+(iy-1)*nx+1]
-xv[nlocal+0+nx+(iy)*nx]
-xv[nlocal+0+nx+(iy)*nx+1]
-xv[nlocal+0+nx+(iy+1)*nx]
-xv[nlocal+0+nx+(iy+1)*nx+1]
;}else if(iz==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+(iy-1)*nx+nx-2]
-xv[nlocal+0+nx+(iy-1)*nx+nx-1]
-xv[nlocal+0+nx+(iy)*nx+nx-2]
-xv[nlocal+0+nx+(iy)*nx+nx-1]
-xv[nlocal+0+nx+(iy+1)*nx+nx-2]
-xv[nlocal+0+nx+(iy+1)*nx+nx-1]
;}
else if(iz==nz-1 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+(iy-1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+(iy-1)*nx+1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+(iy)*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+(iy)*nx+1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+(iy+1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+(iy+1)*nx+1]
;}else if(iz==nz-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+(iy-1)*nx+nx-2]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+(iy-1)*nx+nx-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+(iy)*nx+nx-2]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+(iy)*nx+nx-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+(iy+1)*nx+nx-2]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz+nx+(iy+1)*nx+nx-1]
;}
else if(iy==0&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx+(iz-1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+(iz-1)*nx+1]
-xv[nlocal+0+nx+nx*ny+nx+(iz)*nx]
-xv[nlocal+0+nx+nx*ny+nx+(iz)*nx+1]
-xv[nlocal+0+nx+nx*ny+nx+(iz+1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+(iz+1)*nx+1]
;}else if(iy==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx+(iz-1)*nx+nx-2]
-xv[nlocal+0+nx+nx*ny+nx+(iz-1)*nx+nx-1]
-xv[nlocal+0+nx+nx*ny+nx+(iz)*nx+nx-2]
-xv[nlocal+0+nx+nx*ny+nx+(iz)*nx+nx-1]
-xv[nlocal+0+nx+nx*ny+nx+(iz+1)*nx+nx-2]
-xv[nlocal+0+nx+nx*ny+nx+(iz+1)*nx+nx-1]
;}
else if(iy==ny-1&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+(iz-1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+(iz-1)*nx+1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+(iz)*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+(iz)*nx+1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+(iz+1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+(iz+1)*nx+1]
;}else if(iy==ny-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+(iz)*nx+nx-2]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+(iz)*nx+nx-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+(iz+1)*nx+nx-1]
;}
else if (iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
            else
{
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+nx+nx*ny+nx]
-xv[nlocal+0+nx+nx*ny+nx+1]
-xv[nlocal+0+nx+nx*ny+nx+nx]
-xv[nlocal+0+nx+nx*ny+nx+nx+1]
-xv[nlocal+0+nx]
-xv[nlocal+0+nx+1]
-xv[nlocal+0+nx+nx]
-xv[nlocal+0+nx+nx+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+nx*ny+nx+nx*(nz-2)]
-xv[nlocal+0+nx+nx*ny+nx+nx*(nz-2)+1]
-xv[nlocal+0+nx+nx*ny+nx+nx*(nz-1)]
-xv[nlocal+0+nx+nx*ny+nx+nx*(nz-1)+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+nx*(ny-2)]
-xv[nlocal+0+nx+nx*(ny-2)+1]
-xv[nlocal+0+nx+nx*(ny-1)]
-xv[nlocal+0+nx+nx*(ny-1)+1]
-xv[nlocal+0+nx+nx*ny]
-xv[nlocal+0+nx+nx*ny+1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*(nz-2)]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*(nz-2)+1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*(nz-1)]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*(nz-1)+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx-2]
-xv[nlocal+0+nx-1]
-xv[nlocal+0+nx+nx*ny+nx+nx-2]
-xv[nlocal+0+nx+nx*ny+nx+nx-1]
-xv[nlocal+0+nx+nx*ny+nx+2*nx-2]
-xv[nlocal+0+nx+nx*ny+nx+2*nx-1]
-xv[nlocal+0+nx+nx-2]
-xv[nlocal+0+nx+nx-1]
-xv[nlocal+0+nx+2*nx-2]
-xv[nlocal+0+nx+2*nx-1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+nx*ny+nx+nx*(nz-1)-2]
-xv[nlocal+0+nx+nx*ny+nx+nx*(nz-1)-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz-2]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+nx*(ny-1)-2]
-xv[nlocal+0+nx+nx*(ny-1)-1]
-xv[nlocal+0+nx+nx*ny-2]
-xv[nlocal+0+nx+nx*ny-1]
-xv[nlocal+0+nx+nx*ny+nx-2]
-xv[nlocal+0+nx+nx*ny+nx-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx-2]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+2*nx-2]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+2*nx-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*(nz-1)-2]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*(nz-1)-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz-2]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx*nz-1]
;
}
else if(iy==0&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
;}
else if(iy==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx+ix-1]
-xv[nlocal+0+nx+nx*ny+nx+ix]
-xv[nlocal+0+nx+nx*ny+nx+ix+1]
-xv[nlocal+0+nx+nx*ny+nx+nx+ix-1]
-xv[nlocal+0+nx+nx*ny+nx+nx+ix]
-xv[nlocal+0+nx+nx*ny+nx+nx+ix+1]
;}else if(iy==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx+ix+(nz-2)*nx-1]
-xv[nlocal+0+nx+nx*ny+nx+ix+(nz-2)*nx]
-xv[nlocal+0+nx+nx*ny+nx+ix+(nz-2)*nx+1]
-xv[nlocal+0+nx+nx*ny+nx+ix+(nz-1)*nx-1]
-xv[nlocal+0+nx+nx*ny+nx+ix+(nz-1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+ix+(nz-1)*nx+1]
;}
else if(iz==0&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+ix-1]
-xv[nlocal+0+nx+ix]
-xv[nlocal+0+nx+ix+1]
-xv[nlocal+0+nx+nx+ix-1]
-xv[nlocal+0+nx+nx+ix]
-xv[nlocal+0+nx+nx+ix+1]
;}else if(iz==0&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+ix+(ny-2)*nx-1]
-xv[nlocal+0+nx+ix+(ny-2)*nx]
-xv[nlocal+0+nx+ix+(ny-2)*nx+1]
-xv[nlocal+0+nx+ix+(ny-1)*nx-1]
-xv[nlocal+0+nx+ix+(ny-1)*nx]
-xv[nlocal+0+nx+ix+(ny-1)*nx+1]
;}
else if(iy==ny-1&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+ix-1]
-xv[nlocal+0+nx+nx*ny+ix]
-xv[nlocal+0+nx+nx*ny+ix+1]
;}
else if(iy==ny-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix+1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx+ix-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx+ix]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+nx+ix+1]
;}else if(iy==ny-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix+(nz-2)*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix+(nz-1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+ix+(nz-1)*nx+1]
;}
else if(iz==0 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+(iy-1)*nx]
-xv[nlocal+0+nx+(iy-1)*nx+1]
-xv[nlocal+0+nx+(iy)*nx]
-xv[nlocal+0+nx+(iy)*nx+1]
-xv[nlocal+0+nx+(iy+1)*nx]
-xv[nlocal+0+nx+(iy+1)*nx+1]
;}else if(iz==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+(iy-1)*nx+nx-2]
-xv[nlocal+0+nx+(iy-1)*nx+nx-1]
-xv[nlocal+0+nx+(iy)*nx+nx-2]
-xv[nlocal+0+nx+(iy)*nx+nx-1]
-xv[nlocal+0+nx+(iy+1)*nx+nx-2]
-xv[nlocal+0+nx+(iy+1)*nx+nx-1]
;}
else if(iy==0&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx+(iz-1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+(iz-1)*nx+1]
-xv[nlocal+0+nx+nx*ny+nx+(iz)*nx]
-xv[nlocal+0+nx+nx*ny+nx+(iz)*nx+1]
-xv[nlocal+0+nx+nx*ny+nx+(iz+1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+(iz+1)*nx+1]
;}else if(iy==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx+(iz-1)*nx+nx-2]
-xv[nlocal+0+nx+nx*ny+nx+(iz-1)*nx+nx-1]
-xv[nlocal+0+nx+nx*ny+nx+(iz)*nx+nx-2]
-xv[nlocal+0+nx+nx*ny+nx+(iz)*nx+nx-1]
-xv[nlocal+0+nx+nx*ny+nx+(iz+1)*nx+nx-2]
-xv[nlocal+0+nx+nx*ny+nx+(iz+1)*nx+nx-1]
;}
else if(iy==ny-1&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+(iz-1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+(iz-1)*nx+1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+(iz)*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+(iz)*nx+1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+(iz+1)*nx]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+(iz+1)*nx+1]
;}else if(iy==ny-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+(iz)*nx+nx-2]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+(iz)*nx+nx-1]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+nx+nx*ny+nx+nx*nz+(iz+1)*nx+nx-1]
;}
else if (iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
          }//ipz > 0 
        else
{
            if(ipz < npz - 1)
{
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+nx]
-xv[nlocal+0+nx+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*(nz-2)]
-xv[nlocal+0+nx*(nz-2)+1]
-xv[nlocal+0+nx*(nz-1)]
-xv[nlocal+0+nx*(nz-1)+1]
-xv[nlocal+0+nx*nz+nx*nz]
-xv[nlocal+0+nx*nz+nx*nz+1]
-xv[nlocal+0+nx*nz+nx*nz+nx]
-xv[nlocal+0+nx*nz+nx*nz+nx+1]
-xv[nlocal+0+nx*nz+nx*nz+nx+nx]
-xv[nlocal+0+nx*nz+nx*nz+nx+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*nz]
-xv[nlocal+0+nx*nz+1]
-xv[nlocal+0+nx*nz+nx]
-xv[nlocal+0+nx*nz+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*nz+nx*nz+nx+nx*(ny-2)]
-xv[nlocal+0+nx*nz+nx*nz+nx+nx*(ny-2)+1]
-xv[nlocal+0+nx*nz+nx*nz+nx+nx*(ny-1)]
-xv[nlocal+0+nx*nz+nx*nz+nx+nx*(ny-1)+1]
-xv[nlocal+0+nx*nz+nx*(nz-2)]
-xv[nlocal+0+nx*nz+nx*(nz-2)+1]
-xv[nlocal+0+nx*nz+nx*(nz-1)]
-xv[nlocal+0+nx*nz+nx*(nz-1)+1]
-xv[nlocal+0+nx*nz+nx*nz+nx+nx*ny]
-xv[nlocal+0+nx*nz+nx*nz+nx+nx*ny+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx-2]
-xv[nlocal+0+nx-1]
-xv[nlocal+0+2*nx-2]
-xv[nlocal+0+2*nx-1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*(nz-1)-2]
-xv[nlocal+0+nx*(nz-1)-1]
-xv[nlocal+0+nx*nz-2]
-xv[nlocal+0+nx*nz-1]
-xv[nlocal+0+nx*nz+nx*nz+nx-2]
-xv[nlocal+0+nx*nz+nx*nz+nx-1]
-xv[nlocal+0+nx*nz+nx*nz+nx+nx-2]
-xv[nlocal+0+nx*nz+nx*nz+nx+nx-1]
-xv[nlocal+0+nx*nz+nx*nz+nx+2*nx-2]
-xv[nlocal+0+nx*nz+nx*nz+nx+2*nx-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*nz+nx-2]
-xv[nlocal+0+nx*nz+nx-1]
-xv[nlocal+0+nx*nz+2*nx-2]
-xv[nlocal+0+nx*nz+2*nx-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*nz+nx*nz+nx+nx*(ny-1)-2]
-xv[nlocal+0+nx*nz+nx*nz+nx+nx*(ny-1)-1]
-xv[nlocal+0+nx*nz+nx*nz+nx+nx*ny-2]
-xv[nlocal+0+nx*nz+nx*nz+nx+nx*ny-1]
-xv[nlocal+0+nx*nz+nx*(nz-1)-2]
-xv[nlocal+0+nx*nz+nx*(nz-1)-1]
-xv[nlocal+0+nx*nz+nx*nz-2]
-xv[nlocal+0+nx*nz+nx*nz-1]
-xv[nlocal+0+nx*nz+nx*nz+nx+nx*ny+nx-2]
-xv[nlocal+0+nx*nz+nx*nz+nx+nx*ny+nx-1]
;
}
else if(iy==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
-xv[nlocal+0+nx+ix-1]
-xv[nlocal+0+nx+ix]
-xv[nlocal+0+nx+ix+1]
;}else if(iy==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix+(nz-2)*nx-1]
-xv[nlocal+0+ix+(nz-2)*nx]
-xv[nlocal+0+ix+(nz-2)*nx+1]
-xv[nlocal+0+ix+(nz-1)*nx-1]
-xv[nlocal+0+ix+(nz-1)*nx]
-xv[nlocal+0+ix+(nz-1)*nx+1]
;}
else if(iy==0&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nx*nz+ix-1]
-xv[nlocal+0+nx*nz+nx*nz+ix]
-xv[nlocal+0+nx*nz+nx*nz+ix+1]
;}
else if(iz==nz-1&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nx*nz+nx+ix-1]
-xv[nlocal+0+nx*nz+nx*nz+nx+ix]
-xv[nlocal+0+nx*nz+nx*nz+nx+ix+1]
-xv[nlocal+0+nx*nz+nx*nz+nx+nx+ix-1]
-xv[nlocal+0+nx*nz+nx*nz+nx+nx+ix]
-xv[nlocal+0+nx*nz+nx*nz+nx+nx+ix+1]
;}else if(iz==nz-1&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nx*nz+nx+ix+(ny-2)*nx-1]
-xv[nlocal+0+nx*nz+nx*nz+nx+ix+(ny-2)*nx]
-xv[nlocal+0+nx*nz+nx*nz+nx+ix+(ny-2)*nx+1]
-xv[nlocal+0+nx*nz+nx*nz+nx+ix+(ny-1)*nx-1]
-xv[nlocal+0+nx*nz+nx*nz+nx+ix+(ny-1)*nx]
-xv[nlocal+0+nx*nz+nx*nz+nx+ix+(ny-1)*nx+1]
;}
else if(iy==ny-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+ix-1]
-xv[nlocal+0+nx*nz+ix]
-xv[nlocal+0+nx*nz+ix+1]
-xv[nlocal+0+nx*nz+nx+ix-1]
-xv[nlocal+0+nx*nz+nx+ix]
-xv[nlocal+0+nx*nz+nx+ix+1]
;}else if(iy==ny-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+nx*nz+ix+(nz-2)*nx]
-xv[nlocal+0+nx*nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+nx*nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+nx*nz+ix+(nz-1)*nx]
-xv[nlocal+0+nx*nz+ix+(nz-1)*nx+1]
;}
else if(iy==ny-1&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nx*nz+nx+nx*ny+ix-1]
-xv[nlocal+0+nx*nz+nx*nz+nx+nx*ny+ix]
-xv[nlocal+0+nx*nz+nx*nz+nx+nx*ny+ix+1]
;}
else if(iz==nz-1 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nx*nz+nx+(iy-1)*nx]
-xv[nlocal+0+nx*nz+nx*nz+nx+(iy-1)*nx+1]
-xv[nlocal+0+nx*nz+nx*nz+nx+(iy)*nx]
-xv[nlocal+0+nx*nz+nx*nz+nx+(iy)*nx+1]
-xv[nlocal+0+nx*nz+nx*nz+nx+(iy+1)*nx]
-xv[nlocal+0+nx*nz+nx*nz+nx+(iy+1)*nx+1]
;}else if(iz==nz-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nx*nz+nx+(iy-1)*nx+nx-2]
-xv[nlocal+0+nx*nz+nx*nz+nx+(iy-1)*nx+nx-1]
-xv[nlocal+0+nx*nz+nx*nz+nx+(iy)*nx+nx-2]
-xv[nlocal+0+nx*nz+nx*nz+nx+(iy)*nx+nx-1]
-xv[nlocal+0+nx*nz+nx*nz+nx+(iy+1)*nx+nx-2]
-xv[nlocal+0+nx*nz+nx*nz+nx+(iy+1)*nx+nx-1]
;}
else if(iy==0&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*nx]
-xv[nlocal+0+(iz-1)*nx+1]
-xv[nlocal+0+(iz)*nx]
-xv[nlocal+0+(iz)*nx+1]
-xv[nlocal+0+(iz+1)*nx]
-xv[nlocal+0+(iz+1)*nx+1]
;}else if(iy==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*nx+nx-2]
-xv[nlocal+0+(iz-1)*nx+nx-1]
-xv[nlocal+0+(iz)*nx+nx-2]
-xv[nlocal+0+(iz)*nx+nx-1]
-xv[nlocal+0+(iz+1)*nx+nx-2]
-xv[nlocal+0+(iz+1)*nx+nx-1]
;}
else if(iy==ny-1&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+(iz-1)*nx]
-xv[nlocal+0+nx*nz+(iz-1)*nx+1]
-xv[nlocal+0+nx*nz+(iz)*nx]
-xv[nlocal+0+nx*nz+(iz)*nx+1]
-xv[nlocal+0+nx*nz+(iz+1)*nx]
-xv[nlocal+0+nx*nz+(iz+1)*nx+1]
;}else if(iy==ny-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+nx*nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+nx*nz+(iz)*nx+nx-2]
-xv[nlocal+0+nx*nz+(iz)*nx+nx-1]
-xv[nlocal+0+nx*nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+nx*nz+(iz+1)*nx+nx-1]
;}
else if (iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
            else
{
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+nx]
-xv[nlocal+0+nx+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*(nz-2)]
-xv[nlocal+0+nx*(nz-2)+1]
-xv[nlocal+0+nx*(nz-1)]
-xv[nlocal+0+nx*(nz-1)+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*nz]
-xv[nlocal+0+nx*nz+1]
-xv[nlocal+0+nx*nz+nx]
-xv[nlocal+0+nx*nz+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*nz+nx*(nz-2)]
-xv[nlocal+0+nx*nz+nx*(nz-2)+1]
-xv[nlocal+0+nx*nz+nx*(nz-1)]
-xv[nlocal+0+nx*nz+nx*(nz-1)+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx-2]
-xv[nlocal+0+nx-1]
-xv[nlocal+0+2*nx-2]
-xv[nlocal+0+2*nx-1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*(nz-1)-2]
-xv[nlocal+0+nx*(nz-1)-1]
-xv[nlocal+0+nx*nz-2]
-xv[nlocal+0+nx*nz-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*nz+nx-2]
-xv[nlocal+0+nx*nz+nx-1]
-xv[nlocal+0+nx*nz+2*nx-2]
-xv[nlocal+0+nx*nz+2*nx-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*nz+nx*(nz-1)-2]
-xv[nlocal+0+nx*nz+nx*(nz-1)-1]
-xv[nlocal+0+nx*nz+nx*nz-2]
-xv[nlocal+0+nx*nz+nx*nz-1]
;
}
else if(iy==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
-xv[nlocal+0+nx+ix-1]
-xv[nlocal+0+nx+ix]
-xv[nlocal+0+nx+ix+1]
;}else if(iy==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix+(nz-2)*nx-1]
-xv[nlocal+0+ix+(nz-2)*nx]
-xv[nlocal+0+ix+(nz-2)*nx+1]
-xv[nlocal+0+ix+(nz-1)*nx-1]
-xv[nlocal+0+ix+(nz-1)*nx]
-xv[nlocal+0+ix+(nz-1)*nx+1]
;}
else if(iy==ny-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+ix-1]
-xv[nlocal+0+nx*nz+ix]
-xv[nlocal+0+nx*nz+ix+1]
-xv[nlocal+0+nx*nz+nx+ix-1]
-xv[nlocal+0+nx*nz+nx+ix]
-xv[nlocal+0+nx*nz+nx+ix+1]
;}else if(iy==ny-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+ix+(nz-2)*nx-1]
-xv[nlocal+0+nx*nz+ix+(nz-2)*nx]
-xv[nlocal+0+nx*nz+ix+(nz-2)*nx+1]
-xv[nlocal+0+nx*nz+ix+(nz-1)*nx-1]
-xv[nlocal+0+nx*nz+ix+(nz-1)*nx]
-xv[nlocal+0+nx*nz+ix+(nz-1)*nx+1]
;}
else if(iy==0&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*nx]
-xv[nlocal+0+(iz-1)*nx+1]
-xv[nlocal+0+(iz)*nx]
-xv[nlocal+0+(iz)*nx+1]
-xv[nlocal+0+(iz+1)*nx]
-xv[nlocal+0+(iz+1)*nx+1]
;}else if(iy==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*nx+nx-2]
-xv[nlocal+0+(iz-1)*nx+nx-1]
-xv[nlocal+0+(iz)*nx+nx-2]
-xv[nlocal+0+(iz)*nx+nx-1]
-xv[nlocal+0+(iz+1)*nx+nx-2]
-xv[nlocal+0+(iz+1)*nx+nx-1]
;}
else if(iy==ny-1&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+(iz-1)*nx]
-xv[nlocal+0+nx*nz+(iz-1)*nx+1]
-xv[nlocal+0+nx*nz+(iz)*nx]
-xv[nlocal+0+nx*nz+(iz)*nx+1]
-xv[nlocal+0+nx*nz+(iz+1)*nx]
-xv[nlocal+0+nx*nz+(iz+1)*nx+1]
;}else if(iy==ny-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+(iz-1)*nx+nx-2]
-xv[nlocal+0+nx*nz+(iz-1)*nx+nx-1]
-xv[nlocal+0+nx*nz+(iz)*nx+nx-2]
-xv[nlocal+0+nx*nz+(iz)*nx+nx-1]
-xv[nlocal+0+nx*nz+(iz+1)*nx+nx-2]
-xv[nlocal+0+nx*nz+(iz+1)*nx+nx-1]
;}
else if (iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
          }//ipz > 0 
        }//ipy < npy - 1 
      else
{
        if(ipz > 0)
{
            if(ipz < npz - 1)
{
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+nx+nx*ny]
-xv[nlocal+0+nx+nx*ny+1]
-xv[nlocal+0+nx+nx*ny+nx]
-xv[nlocal+0+nx+nx*ny+nx+1]
-xv[nlocal+0+nx]
-xv[nlocal+0+nx+1]
-xv[nlocal+0+nx+nx]
-xv[nlocal+0+nx+nx+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+nx*ny+nx*(nz-2)]
-xv[nlocal+0+nx+nx*ny+nx*(nz-2)+1]
-xv[nlocal+0+nx+nx*ny+nx*(nz-1)]
-xv[nlocal+0+nx+nx*ny+nx*(nz-1)+1]
-xv[nlocal+0+nx+nx*ny+nx*nz]
-xv[nlocal+0+nx+nx*ny+nx*nz+1]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+1]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+nx]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+nx*(ny-2)]
-xv[nlocal+0+nx+nx*(ny-2)+1]
-xv[nlocal+0+nx+nx*(ny-1)]
-xv[nlocal+0+nx+nx*(ny-1)+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+nx*(ny-2)]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+nx*(ny-2)+1]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+nx*(ny-1)]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+nx*(ny-1)+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx-2]
-xv[nlocal+0+nx-1]
-xv[nlocal+0+nx+nx*ny+nx-2]
-xv[nlocal+0+nx+nx*ny+nx-1]
-xv[nlocal+0+nx+nx*ny+2*nx-2]
-xv[nlocal+0+nx+nx*ny+2*nx-1]
-xv[nlocal+0+nx+nx-2]
-xv[nlocal+0+nx+nx-1]
-xv[nlocal+0+nx+2*nx-2]
-xv[nlocal+0+nx+2*nx-1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+nx*ny+nx*(nz-1)-2]
-xv[nlocal+0+nx+nx*ny+nx*(nz-1)-1]
-xv[nlocal+0+nx+nx*ny+nx*nz-2]
-xv[nlocal+0+nx+nx*ny+nx*nz-1]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx-2]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx-1]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+nx-2]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+nx-1]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+2*nx-2]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+2*nx-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+nx*(ny-1)-2]
-xv[nlocal+0+nx+nx*(ny-1)-1]
-xv[nlocal+0+nx+nx*ny-2]
-xv[nlocal+0+nx+nx*ny-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+nx*(ny-1)-2]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+nx*(ny-1)-1]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+nx*ny-2]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+nx*ny-1]
;
}
else if(iy==0&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
;}
else if(iy==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+ix-1]
-xv[nlocal+0+nx+nx*ny+ix]
-xv[nlocal+0+nx+nx*ny+ix+1]
-xv[nlocal+0+nx+nx*ny+nx+ix-1]
-xv[nlocal+0+nx+nx*ny+nx+ix]
-xv[nlocal+0+nx+nx*ny+nx+ix+1]
;}else if(iy==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+ix+(nz-2)*nx-1]
-xv[nlocal+0+nx+nx*ny+ix+(nz-2)*nx]
-xv[nlocal+0+nx+nx*ny+ix+(nz-2)*nx+1]
-xv[nlocal+0+nx+nx*ny+ix+(nz-1)*nx-1]
-xv[nlocal+0+nx+nx*ny+ix+(nz-1)*nx]
-xv[nlocal+0+nx+nx*ny+ix+(nz-1)*nx+1]
;}
else if(iy==0&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx*nz+ix-1]
-xv[nlocal+0+nx+nx*ny+nx*nz+ix]
-xv[nlocal+0+nx+nx*ny+nx*nz+ix+1]
;}
else if(iz==0&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+ix-1]
-xv[nlocal+0+nx+ix]
-xv[nlocal+0+nx+ix+1]
-xv[nlocal+0+nx+nx+ix-1]
-xv[nlocal+0+nx+nx+ix]
-xv[nlocal+0+nx+nx+ix+1]
;}else if(iz==0&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+ix+(ny-2)*nx-1]
-xv[nlocal+0+nx+ix+(ny-2)*nx]
-xv[nlocal+0+nx+ix+(ny-2)*nx+1]
-xv[nlocal+0+nx+ix+(ny-1)*nx-1]
-xv[nlocal+0+nx+ix+(ny-1)*nx]
-xv[nlocal+0+nx+ix+(ny-1)*nx+1]
;}
else if(iz==nz-1&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+ix-1]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+ix]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+ix+1]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+nx+ix-1]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+nx+ix]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+nx+ix+1]
;}else if(iz==nz-1&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+ix+(ny-2)*nx-1]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+ix+(ny-2)*nx]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+ix+(ny-2)*nx+1]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+ix+(ny-1)*nx-1]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+ix+(ny-1)*nx]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+ix+(ny-1)*nx+1]
;}
else if(iz==0 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+(iy-1)*nx]
-xv[nlocal+0+nx+(iy-1)*nx+1]
-xv[nlocal+0+nx+(iy)*nx]
-xv[nlocal+0+nx+(iy)*nx+1]
-xv[nlocal+0+nx+(iy+1)*nx]
-xv[nlocal+0+nx+(iy+1)*nx+1]
;}else if(iz==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+(iy-1)*nx+nx-2]
-xv[nlocal+0+nx+(iy-1)*nx+nx-1]
-xv[nlocal+0+nx+(iy)*nx+nx-2]
-xv[nlocal+0+nx+(iy)*nx+nx-1]
-xv[nlocal+0+nx+(iy+1)*nx+nx-2]
-xv[nlocal+0+nx+(iy+1)*nx+nx-1]
;}
else if(iz==nz-1 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+(iy-1)*nx]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+(iy-1)*nx+1]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+(iy)*nx]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+(iy)*nx+1]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+(iy+1)*nx]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+(iy+1)*nx+1]
;}else if(iz==nz-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+(iy-1)*nx+nx-2]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+(iy-1)*nx+nx-1]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+(iy)*nx+nx-2]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+(iy)*nx+nx-1]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+(iy+1)*nx+nx-2]
-xv[nlocal+0+nx+nx*ny+nx*nz+nx+(iy+1)*nx+nx-1]
;}
else if(iy==0&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+(iz-1)*nx]
-xv[nlocal+0+nx+nx*ny+(iz-1)*nx+1]
-xv[nlocal+0+nx+nx*ny+(iz)*nx]
-xv[nlocal+0+nx+nx*ny+(iz)*nx+1]
-xv[nlocal+0+nx+nx*ny+(iz+1)*nx]
-xv[nlocal+0+nx+nx*ny+(iz+1)*nx+1]
;}else if(iy==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+(iz-1)*nx+nx-2]
-xv[nlocal+0+nx+nx*ny+(iz-1)*nx+nx-1]
-xv[nlocal+0+nx+nx*ny+(iz)*nx+nx-2]
-xv[nlocal+0+nx+nx*ny+(iz)*nx+nx-1]
-xv[nlocal+0+nx+nx*ny+(iz+1)*nx+nx-2]
-xv[nlocal+0+nx+nx*ny+(iz+1)*nx+nx-1]
;}
else if (iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
            else
{
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+nx+nx*ny]
-xv[nlocal+0+nx+nx*ny+1]
-xv[nlocal+0+nx+nx*ny+nx]
-xv[nlocal+0+nx+nx*ny+nx+1]
-xv[nlocal+0+nx]
-xv[nlocal+0+nx+1]
-xv[nlocal+0+nx+nx]
-xv[nlocal+0+nx+nx+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+nx*ny+nx*(nz-2)]
-xv[nlocal+0+nx+nx*ny+nx*(nz-2)+1]
-xv[nlocal+0+nx+nx*ny+nx*(nz-1)]
-xv[nlocal+0+nx+nx*ny+nx*(nz-1)+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+nx*(ny-2)]
-xv[nlocal+0+nx+nx*(ny-2)+1]
-xv[nlocal+0+nx+nx*(ny-1)]
-xv[nlocal+0+nx+nx*(ny-1)+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx-2]
-xv[nlocal+0+nx-1]
-xv[nlocal+0+nx+nx*ny+nx-2]
-xv[nlocal+0+nx+nx*ny+nx-1]
-xv[nlocal+0+nx+nx*ny+2*nx-2]
-xv[nlocal+0+nx+nx*ny+2*nx-1]
-xv[nlocal+0+nx+nx-2]
-xv[nlocal+0+nx+nx-1]
-xv[nlocal+0+nx+2*nx-2]
-xv[nlocal+0+nx+2*nx-1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+nx*ny+nx*(nz-1)-2]
-xv[nlocal+0+nx+nx*ny+nx*(nz-1)-1]
-xv[nlocal+0+nx+nx*ny+nx*nz-2]
-xv[nlocal+0+nx+nx*ny+nx*nz-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx+nx*(ny-1)-2]
-xv[nlocal+0+nx+nx*(ny-1)-1]
-xv[nlocal+0+nx+nx*ny-2]
-xv[nlocal+0+nx+nx*ny-1]
;
}
else if(iy==0&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
;}
else if(iy==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+ix-1]
-xv[nlocal+0+nx+nx*ny+ix]
-xv[nlocal+0+nx+nx*ny+ix+1]
-xv[nlocal+0+nx+nx*ny+nx+ix-1]
-xv[nlocal+0+nx+nx*ny+nx+ix]
-xv[nlocal+0+nx+nx*ny+nx+ix+1]
;}else if(iy==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+ix+(nz-2)*nx-1]
-xv[nlocal+0+nx+nx*ny+ix+(nz-2)*nx]
-xv[nlocal+0+nx+nx*ny+ix+(nz-2)*nx+1]
-xv[nlocal+0+nx+nx*ny+ix+(nz-1)*nx-1]
-xv[nlocal+0+nx+nx*ny+ix+(nz-1)*nx]
-xv[nlocal+0+nx+nx*ny+ix+(nz-1)*nx+1]
;}
else if(iz==0&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+ix-1]
-xv[nlocal+0+nx+ix]
-xv[nlocal+0+nx+ix+1]
-xv[nlocal+0+nx+nx+ix-1]
-xv[nlocal+0+nx+nx+ix]
-xv[nlocal+0+nx+nx+ix+1]
;}else if(iz==0&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+ix+(ny-2)*nx-1]
-xv[nlocal+0+nx+ix+(ny-2)*nx]
-xv[nlocal+0+nx+ix+(ny-2)*nx+1]
-xv[nlocal+0+nx+ix+(ny-1)*nx-1]
-xv[nlocal+0+nx+ix+(ny-1)*nx]
-xv[nlocal+0+nx+ix+(ny-1)*nx+1]
;}
else if(iz==0 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+(iy-1)*nx]
-xv[nlocal+0+nx+(iy-1)*nx+1]
-xv[nlocal+0+nx+(iy)*nx]
-xv[nlocal+0+nx+(iy)*nx+1]
-xv[nlocal+0+nx+(iy+1)*nx]
-xv[nlocal+0+nx+(iy+1)*nx+1]
;}else if(iz==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+(iy-1)*nx+nx-2]
-xv[nlocal+0+nx+(iy-1)*nx+nx-1]
-xv[nlocal+0+nx+(iy)*nx+nx-2]
-xv[nlocal+0+nx+(iy)*nx+nx-1]
-xv[nlocal+0+nx+(iy+1)*nx+nx-2]
-xv[nlocal+0+nx+(iy+1)*nx+nx-1]
;}
else if(iy==0&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+(iz-1)*nx]
-xv[nlocal+0+nx+nx*ny+(iz-1)*nx+1]
-xv[nlocal+0+nx+nx*ny+(iz)*nx]
-xv[nlocal+0+nx+nx*ny+(iz)*nx+1]
-xv[nlocal+0+nx+nx*ny+(iz+1)*nx]
-xv[nlocal+0+nx+nx*ny+(iz+1)*nx+1]
;}else if(iy==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx+nx*ny+(iz-1)*nx+nx-2]
-xv[nlocal+0+nx+nx*ny+(iz-1)*nx+nx-1]
-xv[nlocal+0+nx+nx*ny+(iz)*nx+nx-2]
-xv[nlocal+0+nx+nx*ny+(iz)*nx+nx-1]
-xv[nlocal+0+nx+nx*ny+(iz+1)*nx+nx-2]
-xv[nlocal+0+nx+nx*ny+(iz+1)*nx+nx-1]
;}
else if (iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
          }//ipz > 0 
        else
{
            if(ipz < npz - 1)
{
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+nx]
-xv[nlocal+0+nx+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*(nz-2)]
-xv[nlocal+0+nx*(nz-2)+1]
-xv[nlocal+0+nx*(nz-1)]
-xv[nlocal+0+nx*(nz-1)+1]
-xv[nlocal+0+nx*nz]
-xv[nlocal+0+nx*nz+1]
-xv[nlocal+0+nx*nz+nx]
-xv[nlocal+0+nx*nz+nx+1]
-xv[nlocal+0+nx*nz+nx+nx]
-xv[nlocal+0+nx*nz+nx+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*nz+nx+nx*(ny-2)]
-xv[nlocal+0+nx*nz+nx+nx*(ny-2)+1]
-xv[nlocal+0+nx*nz+nx+nx*(ny-1)]
-xv[nlocal+0+nx*nz+nx+nx*(ny-1)+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx-2]
-xv[nlocal+0+nx-1]
-xv[nlocal+0+2*nx-2]
-xv[nlocal+0+2*nx-1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*(nz-1)-2]
-xv[nlocal+0+nx*(nz-1)-1]
-xv[nlocal+0+nx*nz-2]
-xv[nlocal+0+nx*nz-1]
-xv[nlocal+0+nx*nz+nx-2]
-xv[nlocal+0+nx*nz+nx-1]
-xv[nlocal+0+nx*nz+nx+nx-2]
-xv[nlocal+0+nx*nz+nx+nx-1]
-xv[nlocal+0+nx*nz+nx+2*nx-2]
-xv[nlocal+0+nx*nz+nx+2*nx-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*nz+nx+nx*(ny-1)-2]
-xv[nlocal+0+nx*nz+nx+nx*(ny-1)-1]
-xv[nlocal+0+nx*nz+nx+nx*ny-2]
-xv[nlocal+0+nx*nz+nx+nx*ny-1]
;
}
else if(iy==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
-xv[nlocal+0+nx+ix-1]
-xv[nlocal+0+nx+ix]
-xv[nlocal+0+nx+ix+1]
;}else if(iy==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix+(nz-2)*nx-1]
-xv[nlocal+0+ix+(nz-2)*nx]
-xv[nlocal+0+ix+(nz-2)*nx+1]
-xv[nlocal+0+ix+(nz-1)*nx-1]
-xv[nlocal+0+ix+(nz-1)*nx]
-xv[nlocal+0+ix+(nz-1)*nx+1]
;}
else if(iy==0&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+ix-1]
-xv[nlocal+0+nx*nz+ix]
-xv[nlocal+0+nx*nz+ix+1]
;}
else if(iz==nz-1&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nx+ix-1]
-xv[nlocal+0+nx*nz+nx+ix]
-xv[nlocal+0+nx*nz+nx+ix+1]
-xv[nlocal+0+nx*nz+nx+nx+ix-1]
-xv[nlocal+0+nx*nz+nx+nx+ix]
-xv[nlocal+0+nx*nz+nx+nx+ix+1]
;}else if(iz==nz-1&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nx+ix+(ny-2)*nx-1]
-xv[nlocal+0+nx*nz+nx+ix+(ny-2)*nx]
-xv[nlocal+0+nx*nz+nx+ix+(ny-2)*nx+1]
-xv[nlocal+0+nx*nz+nx+ix+(ny-1)*nx-1]
-xv[nlocal+0+nx*nz+nx+ix+(ny-1)*nx]
-xv[nlocal+0+nx*nz+nx+ix+(ny-1)*nx+1]
;}
else if(iz==nz-1 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nx+(iy-1)*nx]
-xv[nlocal+0+nx*nz+nx+(iy-1)*nx+1]
-xv[nlocal+0+nx*nz+nx+(iy)*nx]
-xv[nlocal+0+nx*nz+nx+(iy)*nx+1]
-xv[nlocal+0+nx*nz+nx+(iy+1)*nx]
-xv[nlocal+0+nx*nz+nx+(iy+1)*nx+1]
;}else if(iz==nz-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nx+(iy-1)*nx+nx-2]
-xv[nlocal+0+nx*nz+nx+(iy-1)*nx+nx-1]
-xv[nlocal+0+nx*nz+nx+(iy)*nx+nx-2]
-xv[nlocal+0+nx*nz+nx+(iy)*nx+nx-1]
-xv[nlocal+0+nx*nz+nx+(iy+1)*nx+nx-2]
-xv[nlocal+0+nx*nz+nx+(iy+1)*nx+nx-1]
;}
else if(iy==0&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*nx]
-xv[nlocal+0+(iz-1)*nx+1]
-xv[nlocal+0+(iz)*nx]
-xv[nlocal+0+(iz)*nx+1]
-xv[nlocal+0+(iz+1)*nx]
-xv[nlocal+0+(iz+1)*nx+1]
;}else if(iy==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*nx+nx-2]
-xv[nlocal+0+(iz-1)*nx+nx-1]
-xv[nlocal+0+(iz)*nx+nx-2]
-xv[nlocal+0+(iz)*nx+nx-1]
-xv[nlocal+0+(iz+1)*nx+nx-2]
-xv[nlocal+0+(iz+1)*nx+nx-1]
;}
else if (iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
            else
{
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+nx]
-xv[nlocal+0+nx+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*(nz-2)]
-xv[nlocal+0+nx*(nz-2)+1]
-xv[nlocal+0+nx*(nz-1)]
-xv[nlocal+0+nx*(nz-1)+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx-2]
-xv[nlocal+0+nx-1]
-xv[nlocal+0+2*nx-2]
-xv[nlocal+0+2*nx-1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*(nz-1)-2]
-xv[nlocal+0+nx*(nz-1)-1]
-xv[nlocal+0+nx*nz-2]
-xv[nlocal+0+nx*nz-1]
;
}
else if(iy==0&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
-xv[nlocal+0+nx+ix-1]
-xv[nlocal+0+nx+ix]
-xv[nlocal+0+nx+ix+1]
;}else if(iy==0&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix+(nz-2)*nx-1]
-xv[nlocal+0+ix+(nz-2)*nx]
-xv[nlocal+0+ix+(nz-2)*nx+1]
-xv[nlocal+0+ix+(nz-1)*nx-1]
-xv[nlocal+0+ix+(nz-1)*nx]
-xv[nlocal+0+ix+(nz-1)*nx+1]
;}
else if(iy==0&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*nx]
-xv[nlocal+0+(iz-1)*nx+1]
-xv[nlocal+0+(iz)*nx]
-xv[nlocal+0+(iz)*nx+1]
-xv[nlocal+0+(iz+1)*nx]
-xv[nlocal+0+(iz+1)*nx+1]
;}else if(iy==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*nx+nx-2]
-xv[nlocal+0+(iz-1)*nx+nx-1]
-xv[nlocal+0+(iz)*nx+nx-2]
-xv[nlocal+0+(iz)*nx+nx-1]
-xv[nlocal+0+(iz+1)*nx+nx-2]
-xv[nlocal+0+(iz+1)*nx+nx-1]
;}
else if (iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+nx]
-xv[nlocal+0+nx+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*ny+nx+nx*nz]
-xv[nlocal+0+nx*ny+nx+nx*nz+1]
-xv[nlocal+0+nx*ny+nx+nx*nz+nx]
-xv[nlocal+0+nx*ny+nx+nx*nz+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*(ny-2)]
-xv[nlocal+0+nx*(ny-2)+1]
-xv[nlocal+0+nx*(ny-1)]
-xv[nlocal+0+nx*(ny-1)+1]
-xv[nlocal+0+nx*ny]
-xv[nlocal+0+nx*ny+1]
-xv[nlocal+0+nx*ny+nx]
-xv[nlocal+0+nx*ny+nx+1]
-xv[nlocal+0+nx*ny+nx+nx]
-xv[nlocal+0+nx*ny+nx+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*ny+nx+nx*nz+nx*(ny-2)]
-xv[nlocal+0+nx*ny+nx+nx*nz+nx*(ny-2)+1]
-xv[nlocal+0+nx*ny+nx+nx*nz+nx*(ny-1)]
-xv[nlocal+0+nx*ny+nx+nx*nz+nx*(ny-1)+1]
-xv[nlocal+0+nx*ny+nx+nx*(nz-2)]
-xv[nlocal+0+nx*ny+nx+nx*(nz-2)+1]
-xv[nlocal+0+nx*ny+nx+nx*(nz-1)]
-xv[nlocal+0+nx*ny+nx+nx*(nz-1)+1]
-xv[nlocal+0+nx*ny+nx+nx*nz+nx*ny]
-xv[nlocal+0+nx*ny+nx+nx*nz+nx*ny+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx-2]
-xv[nlocal+0+nx-1]
-xv[nlocal+0+2*nx-2]
-xv[nlocal+0+2*nx-1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*ny+nx+nx*nz+nx-2]
-xv[nlocal+0+nx*ny+nx+nx*nz+nx-1]
-xv[nlocal+0+nx*ny+nx+nx*nz+2*nx-2]
-xv[nlocal+0+nx*ny+nx+nx*nz+2*nx-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*(ny-1)-2]
-xv[nlocal+0+nx*(ny-1)-1]
-xv[nlocal+0+nx*ny-2]
-xv[nlocal+0+nx*ny-1]
-xv[nlocal+0+nx*ny+nx-2]
-xv[nlocal+0+nx*ny+nx-1]
-xv[nlocal+0+nx*ny+nx+nx-2]
-xv[nlocal+0+nx*ny+nx+nx-1]
-xv[nlocal+0+nx*ny+nx+2*nx-2]
-xv[nlocal+0+nx*ny+nx+2*nx-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*ny+nx+nx*nz+nx*(ny-1)-2]
-xv[nlocal+0+nx*ny+nx+nx*nz+nx*(ny-1)-1]
-xv[nlocal+0+nx*ny+nx+nx*nz+nx*ny-2]
-xv[nlocal+0+nx*ny+nx+nx*nz+nx*ny-1]
-xv[nlocal+0+nx*ny+nx+nx*(nz-1)-2]
-xv[nlocal+0+nx*ny+nx+nx*(nz-1)-1]
-xv[nlocal+0+nx*ny+nx+nx*nz-2]
-xv[nlocal+0+nx*ny+nx+nx*nz-1]
-xv[nlocal+0+nx*ny+nx+nx*nz+nx*ny+nx-2]
-xv[nlocal+0+nx*ny+nx+nx*nz+nx*ny+nx-1]
;
}
else if(iz==0&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
-xv[nlocal+0+nx+ix-1]
-xv[nlocal+0+nx+ix]
-xv[nlocal+0+nx+ix+1]
;}else if(iz==0&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix+(ny-2)*nx-1]
-xv[nlocal+0+ix+(ny-2)*nx]
-xv[nlocal+0+ix+(ny-2)*nx+1]
-xv[nlocal+0+ix+(ny-1)*nx-1]
-xv[nlocal+0+ix+(ny-1)*nx]
-xv[nlocal+0+ix+(ny-1)*nx+1]
;}
else if(iz==nz-1&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+nx+nx*nz+ix-1]
-xv[nlocal+0+nx*ny+nx+nx*nz+ix]
-xv[nlocal+0+nx*ny+nx+nx*nz+ix+1]
-xv[nlocal+0+nx*ny+nx+nx*nz+nx+ix-1]
-xv[nlocal+0+nx*ny+nx+nx*nz+nx+ix]
-xv[nlocal+0+nx*ny+nx+nx*nz+nx+ix+1]
;}else if(iz==nz-1&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+nx+nx*nz+ix+(ny-2)*nx-1]
-xv[nlocal+0+nx*ny+nx+nx*nz+ix+(ny-2)*nx]
-xv[nlocal+0+nx*ny+nx+nx*nz+ix+(ny-2)*nx+1]
-xv[nlocal+0+nx*ny+nx+nx*nz+ix+(ny-1)*nx-1]
-xv[nlocal+0+nx*ny+nx+nx*nz+ix+(ny-1)*nx]
-xv[nlocal+0+nx*ny+nx+nx*nz+ix+(ny-1)*nx+1]
;}
else if(iy==ny-1&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ix-1]
-xv[nlocal+0+nx*ny+ix]
-xv[nlocal+0+nx*ny+ix+1]
;}
else if(iy==ny-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+nx+ix-1]
-xv[nlocal+0+nx*ny+nx+ix]
-xv[nlocal+0+nx*ny+nx+ix+1]
-xv[nlocal+0+nx*ny+nx+nx+ix-1]
-xv[nlocal+0+nx*ny+nx+nx+ix]
-xv[nlocal+0+nx*ny+nx+nx+ix+1]
;}else if(iy==ny-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+nx+ix+(nz-2)*nx-1]
-xv[nlocal+0+nx*ny+nx+ix+(nz-2)*nx]
-xv[nlocal+0+nx*ny+nx+ix+(nz-2)*nx+1]
-xv[nlocal+0+nx*ny+nx+ix+(nz-1)*nx-1]
-xv[nlocal+0+nx*ny+nx+ix+(nz-1)*nx]
-xv[nlocal+0+nx*ny+nx+ix+(nz-1)*nx+1]
;}
else if(iy==ny-1&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+nx+nx*nz+nx*ny+ix-1]
-xv[nlocal+0+nx*ny+nx+nx*nz+nx*ny+ix]
-xv[nlocal+0+nx*ny+nx+nx*nz+nx*ny+ix+1]
;}
else if(iz==0 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iy-1)*nx]
-xv[nlocal+0+(iy-1)*nx+1]
-xv[nlocal+0+(iy)*nx]
-xv[nlocal+0+(iy)*nx+1]
-xv[nlocal+0+(iy+1)*nx]
-xv[nlocal+0+(iy+1)*nx+1]
;}else if(iz==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iy-1)*nx+nx-2]
-xv[nlocal+0+(iy-1)*nx+nx-1]
-xv[nlocal+0+(iy)*nx+nx-2]
-xv[nlocal+0+(iy)*nx+nx-1]
-xv[nlocal+0+(iy+1)*nx+nx-2]
-xv[nlocal+0+(iy+1)*nx+nx-1]
;}
else if(iz==nz-1 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+nx+nx*nz+(iy-1)*nx]
-xv[nlocal+0+nx*ny+nx+nx*nz+(iy-1)*nx+1]
-xv[nlocal+0+nx*ny+nx+nx*nz+(iy)*nx]
-xv[nlocal+0+nx*ny+nx+nx*nz+(iy)*nx+1]
-xv[nlocal+0+nx*ny+nx+nx*nz+(iy+1)*nx]
-xv[nlocal+0+nx*ny+nx+nx*nz+(iy+1)*nx+1]
;}else if(iz==nz-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+nx+nx*nz+(iy-1)*nx+nx-2]
-xv[nlocal+0+nx*ny+nx+nx*nz+(iy-1)*nx+nx-1]
-xv[nlocal+0+nx*ny+nx+nx*nz+(iy)*nx+nx-2]
-xv[nlocal+0+nx*ny+nx+nx*nz+(iy)*nx+nx-1]
-xv[nlocal+0+nx*ny+nx+nx*nz+(iy+1)*nx+nx-2]
-xv[nlocal+0+nx*ny+nx+nx*nz+(iy+1)*nx+nx-1]
;}
else if(iy==ny-1&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+nx+(iz-1)*nx]
-xv[nlocal+0+nx*ny+nx+(iz-1)*nx+1]
-xv[nlocal+0+nx*ny+nx+(iz)*nx]
-xv[nlocal+0+nx*ny+nx+(iz)*nx+1]
-xv[nlocal+0+nx*ny+nx+(iz+1)*nx]
-xv[nlocal+0+nx*ny+nx+(iz+1)*nx+1]
;}else if(iy==ny-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+nx+(iz-1)*nx+nx-2]
-xv[nlocal+0+nx*ny+nx+(iz-1)*nx+nx-1]
-xv[nlocal+0+nx*ny+nx+(iz)*nx+nx-2]
-xv[nlocal+0+nx*ny+nx+(iz)*nx+nx-1]
-xv[nlocal+0+nx*ny+nx+(iz+1)*nx+nx-2]
-xv[nlocal+0+nx*ny+nx+(iz+1)*nx+nx-1]
;}
else if (iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
            else
{
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+nx]
-xv[nlocal+0+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*(ny-2)]
-xv[nlocal+0+nx*(ny-2)+1]
-xv[nlocal+0+nx*(ny-1)]
-xv[nlocal+0+nx*(ny-1)+1]
-xv[nlocal+0+nx*ny]
-xv[nlocal+0+nx*ny+1]
-xv[nlocal+0+nx*ny+nx]
-xv[nlocal+0+nx*ny+nx+1]
-xv[nlocal+0+nx*ny+nx+nx]
-xv[nlocal+0+nx*ny+nx+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*ny+nx+nx*(nz-2)]
-xv[nlocal+0+nx*ny+nx+nx*(nz-2)+1]
-xv[nlocal+0+nx*ny+nx+nx*(nz-1)]
-xv[nlocal+0+nx*ny+nx+nx*(nz-1)+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx-2]
-xv[nlocal+0+nx-1]
-xv[nlocal+0+2*nx-2]
-xv[nlocal+0+2*nx-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*(ny-1)-2]
-xv[nlocal+0+nx*(ny-1)-1]
-xv[nlocal+0+nx*ny-2]
-xv[nlocal+0+nx*ny-1]
-xv[nlocal+0+nx*ny+nx-2]
-xv[nlocal+0+nx*ny+nx-1]
-xv[nlocal+0+nx*ny+nx+nx-2]
-xv[nlocal+0+nx*ny+nx+nx-1]
-xv[nlocal+0+nx*ny+nx+2*nx-2]
-xv[nlocal+0+nx*ny+nx+2*nx-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*ny+nx+nx*(nz-1)-2]
-xv[nlocal+0+nx*ny+nx+nx*(nz-1)-1]
-xv[nlocal+0+nx*ny+nx+nx*nz-2]
-xv[nlocal+0+nx*ny+nx+nx*nz-1]
;
}
else if(iz==0&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
-xv[nlocal+0+nx+ix-1]
-xv[nlocal+0+nx+ix]
-xv[nlocal+0+nx+ix+1]
;}else if(iz==0&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix+(ny-2)*nx-1]
-xv[nlocal+0+ix+(ny-2)*nx]
-xv[nlocal+0+ix+(ny-2)*nx+1]
-xv[nlocal+0+ix+(ny-1)*nx-1]
-xv[nlocal+0+ix+(ny-1)*nx]
-xv[nlocal+0+ix+(ny-1)*nx+1]
;}
else if(iy==ny-1&&iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ix-1]
-xv[nlocal+0+nx*ny+ix]
-xv[nlocal+0+nx*ny+ix+1]
;}
else if(iy==ny-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+nx+ix-1]
-xv[nlocal+0+nx*ny+nx+ix]
-xv[nlocal+0+nx*ny+nx+ix+1]
-xv[nlocal+0+nx*ny+nx+nx+ix-1]
-xv[nlocal+0+nx*ny+nx+nx+ix]
-xv[nlocal+0+nx*ny+nx+nx+ix+1]
;}else if(iy==ny-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+nx+ix+(nz-2)*nx-1]
-xv[nlocal+0+nx*ny+nx+ix+(nz-2)*nx]
-xv[nlocal+0+nx*ny+nx+ix+(nz-2)*nx+1]
-xv[nlocal+0+nx*ny+nx+ix+(nz-1)*nx-1]
-xv[nlocal+0+nx*ny+nx+ix+(nz-1)*nx]
-xv[nlocal+0+nx*ny+nx+ix+(nz-1)*nx+1]
;}
else if(iz==0 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iy-1)*nx]
-xv[nlocal+0+(iy-1)*nx+1]
-xv[nlocal+0+(iy)*nx]
-xv[nlocal+0+(iy)*nx+1]
-xv[nlocal+0+(iy+1)*nx]
-xv[nlocal+0+(iy+1)*nx+1]
;}else if(iz==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iy-1)*nx+nx-2]
-xv[nlocal+0+(iy-1)*nx+nx-1]
-xv[nlocal+0+(iy)*nx+nx-2]
-xv[nlocal+0+(iy)*nx+nx-1]
-xv[nlocal+0+(iy+1)*nx+nx-2]
-xv[nlocal+0+(iy+1)*nx+nx-1]
;}
else if(iy==ny-1&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+nx+(iz-1)*nx]
-xv[nlocal+0+nx*ny+nx+(iz-1)*nx+1]
-xv[nlocal+0+nx*ny+nx+(iz)*nx]
-xv[nlocal+0+nx*ny+nx+(iz)*nx+1]
-xv[nlocal+0+nx*ny+nx+(iz+1)*nx]
-xv[nlocal+0+nx*ny+nx+(iz+1)*nx+1]
;}else if(iy==ny-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+nx+(iz-1)*nx+nx-2]
-xv[nlocal+0+nx*ny+nx+(iz-1)*nx+nx-1]
-xv[nlocal+0+nx*ny+nx+(iz)*nx+nx-2]
-xv[nlocal+0+nx*ny+nx+(iz)*nx+nx-1]
-xv[nlocal+0+nx*ny+nx+(iz+1)*nx+nx-2]
-xv[nlocal+0+nx*ny+nx+(iz+1)*nx+nx-1]
;}
else if (iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
          }//ipz > 0 
        else
{
            if(ipz < npz - 1)
{
if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*nz]
-xv[nlocal+0+nx*nz+1]
-xv[nlocal+0+nx*nz+nx]
-xv[nlocal+0+nx*nz+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+nx]
-xv[nlocal+0+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*nz+nx*(ny-2)]
-xv[nlocal+0+nx*nz+nx*(ny-2)+1]
-xv[nlocal+0+nx*nz+nx*(ny-1)]
-xv[nlocal+0+nx*nz+nx*(ny-1)+1]
-xv[nlocal+0+nx*(nz-2)]
-xv[nlocal+0+nx*(nz-2)+1]
-xv[nlocal+0+nx*(nz-1)]
-xv[nlocal+0+nx*(nz-1)+1]
-xv[nlocal+0+nx*nz+nx*ny]
-xv[nlocal+0+nx*nz+nx*ny+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*nz+nx-2]
-xv[nlocal+0+nx*nz+nx-1]
-xv[nlocal+0+nx*nz+2*nx-2]
-xv[nlocal+0+nx*nz+2*nx-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx-2]
-xv[nlocal+0+nx-1]
-xv[nlocal+0+2*nx-2]
-xv[nlocal+0+2*nx-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*nz+nx*(ny-1)-2]
-xv[nlocal+0+nx*nz+nx*(ny-1)-1]
-xv[nlocal+0+nx*nz+nx*ny-2]
-xv[nlocal+0+nx*nz+nx*ny-1]
-xv[nlocal+0+nx*(nz-1)-2]
-xv[nlocal+0+nx*(nz-1)-1]
-xv[nlocal+0+nx*nz-2]
-xv[nlocal+0+nx*nz-1]
-xv[nlocal+0+nx*nz+nx*ny+nx-2]
-xv[nlocal+0+nx*nz+nx*ny+nx-1]
;
}
else if(iz==nz-1&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+ix-1]
-xv[nlocal+0+nx*nz+ix]
-xv[nlocal+0+nx*nz+ix+1]
-xv[nlocal+0+nx*nz+nx+ix-1]
-xv[nlocal+0+nx*nz+nx+ix]
-xv[nlocal+0+nx*nz+nx+ix+1]
;}else if(iz==nz-1&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+ix+(ny-2)*nx-1]
-xv[nlocal+0+nx*nz+ix+(ny-2)*nx]
-xv[nlocal+0+nx*nz+ix+(ny-2)*nx+1]
-xv[nlocal+0+nx*nz+ix+(ny-1)*nx-1]
-xv[nlocal+0+nx*nz+ix+(ny-1)*nx]
-xv[nlocal+0+nx*nz+ix+(ny-1)*nx+1]
;}
else if(iy==ny-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
-xv[nlocal+0+nx+ix-1]
-xv[nlocal+0+nx+ix]
-xv[nlocal+0+nx+ix+1]
;}else if(iy==ny-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix+(nz-2)*nx-1]
-xv[nlocal+0+ix+(nz-2)*nx]
-xv[nlocal+0+ix+(nz-2)*nx+1]
-xv[nlocal+0+ix+(nz-1)*nx-1]
-xv[nlocal+0+ix+(nz-1)*nx]
-xv[nlocal+0+ix+(nz-1)*nx+1]
;}
else if(iy==ny-1&&iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+nx*ny+ix-1]
-xv[nlocal+0+nx*nz+nx*ny+ix]
-xv[nlocal+0+nx*nz+nx*ny+ix+1]
;}
else if(iz==nz-1 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+(iy-1)*nx]
-xv[nlocal+0+nx*nz+(iy-1)*nx+1]
-xv[nlocal+0+nx*nz+(iy)*nx]
-xv[nlocal+0+nx*nz+(iy)*nx+1]
-xv[nlocal+0+nx*nz+(iy+1)*nx]
-xv[nlocal+0+nx*nz+(iy+1)*nx+1]
;}else if(iz==nz-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*nz+(iy-1)*nx+nx-2]
-xv[nlocal+0+nx*nz+(iy-1)*nx+nx-1]
-xv[nlocal+0+nx*nz+(iy)*nx+nx-2]
-xv[nlocal+0+nx*nz+(iy)*nx+nx-1]
-xv[nlocal+0+nx*nz+(iy+1)*nx+nx-2]
-xv[nlocal+0+nx*nz+(iy+1)*nx+nx-1]
;}
else if(iy==ny-1&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*nx]
-xv[nlocal+0+(iz-1)*nx+1]
-xv[nlocal+0+(iz)*nx]
-xv[nlocal+0+(iz)*nx+1]
-xv[nlocal+0+(iz+1)*nx]
-xv[nlocal+0+(iz+1)*nx+1]
;}else if(iy==ny-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*nx+nx-2]
-xv[nlocal+0+(iz-1)*nx+nx-1]
-xv[nlocal+0+(iz)*nx+nx-2]
-xv[nlocal+0+(iz)*nx+nx-1]
-xv[nlocal+0+(iz+1)*nx+nx-2]
-xv[nlocal+0+(iz+1)*nx+nx-1]
;}
else if (iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
            else
{
if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+nx]
-xv[nlocal+0+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*(nz-2)]
-xv[nlocal+0+nx*(nz-2)+1]
-xv[nlocal+0+nx*(nz-1)]
-xv[nlocal+0+nx*(nz-1)+1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx-2]
-xv[nlocal+0+nx-1]
-xv[nlocal+0+2*nx-2]
-xv[nlocal+0+2*nx-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*(nz-1)-2]
-xv[nlocal+0+nx*(nz-1)-1]
-xv[nlocal+0+nx*nz-2]
-xv[nlocal+0+nx*nz-1]
;
}
else if(iy==ny-1&&iz==0)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
-xv[nlocal+0+nx+ix-1]
-xv[nlocal+0+nx+ix]
-xv[nlocal+0+nx+ix+1]
;}else if(iy==ny-1&&iz==nz-1)
{
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix+(nz-2)*nx-1]
-xv[nlocal+0+ix+(nz-2)*nx]
-xv[nlocal+0+ix+(nz-2)*nx+1]
-xv[nlocal+0+ix+(nz-1)*nx-1]
-xv[nlocal+0+ix+(nz-1)*nx]
-xv[nlocal+0+ix+(nz-1)*nx+1]
;}
else if(iy==ny-1&&ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*nx]
-xv[nlocal+0+(iz-1)*nx+1]
-xv[nlocal+0+(iz)*nx]
-xv[nlocal+0+(iz)*nx+1]
-xv[nlocal+0+(iz+1)*nx]
-xv[nlocal+0+(iz+1)*nx+1]
;}else if(iy==ny-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iz-1)*nx+nx-2]
-xv[nlocal+0+(iz-1)*nx+nx-1]
-xv[nlocal+0+(iz)*nx+nx-2]
-xv[nlocal+0+(iz)*nx+nx-1]
-xv[nlocal+0+(iz+1)*nx+nx-2]
-xv[nlocal+0+(iz+1)*nx+nx-1]
;}
else if (iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
          }//ipz > 0 
        }//ipy < npy - 1 
      else
{
        if(ipz > 0)
{
            if(ipz < npz - 1)
{
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+nx]
-xv[nlocal+0+nx+1]
;
}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*ny]
-xv[nlocal+0+nx*ny+1]
-xv[nlocal+0+nx*ny+nx]
-xv[nlocal+0+nx*ny+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*(ny-2)]
-xv[nlocal+0+nx*(ny-2)+1]
-xv[nlocal+0+nx*(ny-1)]
-xv[nlocal+0+nx*(ny-1)+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*ny+nx*(ny-2)]
-xv[nlocal+0+nx*ny+nx*(ny-2)+1]
-xv[nlocal+0+nx*ny+nx*(ny-1)]
-xv[nlocal+0+nx*ny+nx*(ny-1)+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx-2]
-xv[nlocal+0+nx-1]
-xv[nlocal+0+2*nx-2]
-xv[nlocal+0+2*nx-1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*ny+nx-2]
-xv[nlocal+0+nx*ny+nx-1]
-xv[nlocal+0+nx*ny+2*nx-2]
-xv[nlocal+0+nx*ny+2*nx-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*(ny-1)-2]
-xv[nlocal+0+nx*(ny-1)-1]
-xv[nlocal+0+nx*ny-2]
-xv[nlocal+0+nx*ny-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*ny+nx*(ny-1)-2]
-xv[nlocal+0+nx*ny+nx*(ny-1)-1]
-xv[nlocal+0+nx*ny+nx*ny-2]
-xv[nlocal+0+nx*ny+nx*ny-1]
;
}
else if(iz==0&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
-xv[nlocal+0+nx+ix-1]
-xv[nlocal+0+nx+ix]
-xv[nlocal+0+nx+ix+1]
;}else if(iz==0&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix+(ny-2)*nx-1]
-xv[nlocal+0+ix+(ny-2)*nx]
-xv[nlocal+0+ix+(ny-2)*nx+1]
-xv[nlocal+0+ix+(ny-1)*nx-1]
-xv[nlocal+0+ix+(ny-1)*nx]
-xv[nlocal+0+ix+(ny-1)*nx+1]
;}
else if(iz==nz-1&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ix-1]
-xv[nlocal+0+nx*ny+ix]
-xv[nlocal+0+nx*ny+ix+1]
-xv[nlocal+0+nx*ny+nx+ix-1]
-xv[nlocal+0+nx*ny+nx+ix]
-xv[nlocal+0+nx*ny+nx+ix+1]
;}else if(iz==nz-1&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+ix+(ny-2)*nx-1]
-xv[nlocal+0+nx*ny+ix+(ny-2)*nx]
-xv[nlocal+0+nx*ny+ix+(ny-2)*nx+1]
-xv[nlocal+0+nx*ny+ix+(ny-1)*nx-1]
-xv[nlocal+0+nx*ny+ix+(ny-1)*nx]
-xv[nlocal+0+nx*ny+ix+(ny-1)*nx+1]
;}
else if(iz==0 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iy-1)*nx]
-xv[nlocal+0+(iy-1)*nx+1]
-xv[nlocal+0+(iy)*nx]
-xv[nlocal+0+(iy)*nx+1]
-xv[nlocal+0+(iy+1)*nx]
-xv[nlocal+0+(iy+1)*nx+1]
;}else if(iz==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iy-1)*nx+nx-2]
-xv[nlocal+0+(iy-1)*nx+nx-1]
-xv[nlocal+0+(iy)*nx+nx-2]
-xv[nlocal+0+(iy)*nx+nx-1]
-xv[nlocal+0+(iy+1)*nx+nx-2]
-xv[nlocal+0+(iy+1)*nx+nx-1]
;}
else if(iz==nz-1 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+(iy-1)*nx]
-xv[nlocal+0+nx*ny+(iy-1)*nx+1]
-xv[nlocal+0+nx*ny+(iy)*nx]
-xv[nlocal+0+nx*ny+(iy)*nx+1]
-xv[nlocal+0+nx*ny+(iy+1)*nx]
-xv[nlocal+0+nx*ny+(iy+1)*nx+1]
;}else if(iz==nz-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+nx*ny+(iy-1)*nx+nx-2]
-xv[nlocal+0+nx*ny+(iy-1)*nx+nx-1]
-xv[nlocal+0+nx*ny+(iy)*nx+nx-2]
-xv[nlocal+0+nx*ny+(iy)*nx+nx-1]
-xv[nlocal+0+nx*ny+(iy+1)*nx+nx-2]
-xv[nlocal+0+nx*ny+(iy+1)*nx+nx-1]
;}
else if (iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
else if (iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
            else
{
if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+nx]
-xv[nlocal+0+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*(ny-2)]
-xv[nlocal+0+nx*(ny-2)+1]
-xv[nlocal+0+nx*(ny-1)]
-xv[nlocal+0+nx*(ny-1)+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx-2]
-xv[nlocal+0+nx-1]
-xv[nlocal+0+2*nx-2]
-xv[nlocal+0+2*nx-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*(ny-1)-2]
-xv[nlocal+0+nx*(ny-1)-1]
-xv[nlocal+0+nx*ny-2]
-xv[nlocal+0+nx*ny-1]
;
}
else if(iz==0&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
-xv[nlocal+0+nx+ix-1]
-xv[nlocal+0+nx+ix]
-xv[nlocal+0+nx+ix+1]
;}else if(iz==0&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix+(ny-2)*nx-1]
-xv[nlocal+0+ix+(ny-2)*nx]
-xv[nlocal+0+ix+(ny-2)*nx+1]
-xv[nlocal+0+ix+(ny-1)*nx-1]
-xv[nlocal+0+ix+(ny-1)*nx]
-xv[nlocal+0+ix+(ny-1)*nx+1]
;}
else if(iz==0 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iy-1)*nx]
-xv[nlocal+0+(iy-1)*nx+1]
-xv[nlocal+0+(iy)*nx]
-xv[nlocal+0+(iy)*nx+1]
-xv[nlocal+0+(iy+1)*nx]
-xv[nlocal+0+(iy+1)*nx+1]
;}else if(iz==0&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iy-1)*nx+nx-2]
-xv[nlocal+0+(iy-1)*nx+nx-1]
-xv[nlocal+0+(iy)*nx+nx-2]
-xv[nlocal+0+(iy)*nx+nx-1]
-xv[nlocal+0+(iy+1)*nx+nx-2]
-xv[nlocal+0+(iy+1)*nx+nx-1]
;}
else if (iz==0){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
          }//ipz > 0 
        else
{
            if(ipz < npz - 1)
{
if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0]
-xv[nlocal+0+1]
-xv[nlocal+0+nx]
-xv[nlocal+0+nx+1]
;
}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*(ny-2)]
-xv[nlocal+0+nx*(ny-2)+1]
-xv[nlocal+0+nx*(ny-1)]
-xv[nlocal+0+nx*(ny-1)+1]
;
}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx-2]
-xv[nlocal+0+nx-1]
-xv[nlocal+0+2*nx-2]
-xv[nlocal+0+2*nx-1]
;
}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] += 
-xv[nlocal+0+nx*(ny-1)-2]
-xv[nlocal+0+nx*(ny-1)-1]
-xv[nlocal+0+nx*ny-2]
-xv[nlocal+0+nx*ny-1]
;
}
else if(iz==nz-1&& iy==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix-1]
-xv[nlocal+0+ix]
-xv[nlocal+0+ix+1]
-xv[nlocal+0+nx+ix-1]
-xv[nlocal+0+nx+ix]
-xv[nlocal+0+nx+ix+1]
;}else if(iz==nz-1&& iy==ny-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+ix+(ny-2)*nx-1]
-xv[nlocal+0+ix+(ny-2)*nx]
-xv[nlocal+0+ix+(ny-2)*nx+1]
-xv[nlocal+0+ix+(ny-1)*nx-1]
-xv[nlocal+0+ix+(ny-1)*nx]
-xv[nlocal+0+ix+(ny-1)*nx+1]
;}
else if(iz==nz-1 && ix==0){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iy-1)*nx]
-xv[nlocal+0+(iy-1)*nx+1]
-xv[nlocal+0+(iy)*nx]
-xv[nlocal+0+(iy)*nx+1]
-xv[nlocal+0+(iy+1)*nx]
-xv[nlocal+0+(iy+1)*nx+1]
;}else if(iz==nz-1&&ix==nx-1){
 xv[ix+iy*nx+iz*ny*nx] +=
-xv[nlocal+0+(iy-1)*nx+nx-2]
-xv[nlocal+0+(iy-1)*nx+nx-1]
-xv[nlocal+0+(iy)*nx+nx-2]
-xv[nlocal+0+(iy)*nx+nx-1]
-xv[nlocal+0+(iy+1)*nx+nx-2]
-xv[nlocal+0+(iy+1)*nx+nx-1]
;}
else if (iz==nz-1){
 xv[ix+iy*nx+iz*ny*nx] +=
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
}//ipz < npz - 1
            else
{
}//ipz < npz - 1
          }//ipz > 0 
        }//ipy < npy - 1 
      }//ipy > 0 
    }//ipx < npx - 1 
 }//ipx > 0 

      
xv[ix+iy*nx+iz*ny*nx] /= diagonal_element;
        }
      }
    }

// backwards sweep
for( iz=nz-1; iz>=0; iz--){
    for( iy=ny-1; iy>=0; iy--){
        for (ix=nx-1; ix>=0; ix--){

if(ix == 0&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+1)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+1)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+1)*nx+(iz+1)*ny*nx]
;}
else if(ix == 0&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+1)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+1)*nx+(iz+0)*ny*nx]
;}
else if(ix == 0&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+1)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+-1)*nx+(iz+1)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+1)*ny*nx]
;}
else if(ix == 0&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+0)+(iy+-1)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+-1)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
;}
else if(ix == nx-1&& iy == 0&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+-1)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+1)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+1)*ny*nx]
;}
else if(ix == nx-1&& iy == 0&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+-1)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+1)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
;}
else if(ix == nx-1&& iy == ny-1&& iz == 0){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+-1)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+-1)*nx+(iz+1)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
;}
else if(ix == nx-1&& iy == ny-1&& iz == nz-1){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+-1)+(iy+-1)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
;}

//edges 
else if (ix == 0 && iy == 0 &&iz>0 && iz<nz-1){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+1)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+1)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+1)+(iy+1)*nx+(iz+1)*ny*nx]
;}
else if (ix == 0 && iy == ny-1 &&iz>0 && iz<nz-1){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+0)+(iy+-1)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+-1)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+1)+(iy+-1)*nx+(iz+1)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+1)*ny*nx]
;}
else if (ix == nx-1 && iy == 0 && iz>0 && iz<nz-1){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+-1)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+1)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+-1)+(iy+1)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+1)*ny*nx]
;}
else if (ix == nx-1 && iy == ny-1 &&iz>0 && iz<nz-1){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+-1)+(iy+-1)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+-1)*nx+(iz+1)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
;}
else if (ix == 0 && iz == 0 &&iy>0 && iy<ny-1){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+1)*ny*nx]
+xv[(ix+1)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+-1)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+1)*ny*nx]
+xv[(ix+1)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+1)*nx+(iz+1)*ny*nx]
;}
else if (ix == 0 && iz == nz-1 &&iy>0 && iy<ny-1){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+0)+(iy+-1)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+-1)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+1)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+1)*nx+(iz+0)*ny*nx]
;}
else if (ix == nx-1 && iz == 0 && iy>0 && iy<ny-1){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+-1)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+-1)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+1)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+-1)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+1)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+1)*ny*nx]
;}
else if (ix == nx-1 && iz == nz-1 &&iy>0 && iy<ny-1){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+-1)+(iy+-1)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+1)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
;}
else if (iy == 0 && iz == 0 &&ix>0 && ix<nx-1){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+-1)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+1)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+1)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+1)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+1)*nx+(iz+1)*ny*nx]
;}
else if (iy == 0 && iz == nz-1 &&ix>0 && ix<nx-1){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+-1)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+1)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+1)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+1)*nx+(iz+0)*ny*nx]
;}
else if (iy == ny-1 && iz == 0 && ix>0 && ix<nx-1){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+-1)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+-1)*nx+(iz+1)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+1)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+-1)*nx+(iz+1)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+1)*ny*nx]
;}
else if (iy == ny-1 && iz == nz-1 &&ix>0 && ix<nx-1){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+-1)+(iy+-1)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+-1)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
;}

//sides 
else if (iz == 0 && ix>0 && ix<nx-1 && iy>0 && iy<ny-1){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+-1)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+-1)*nx+(iz+1)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+-1)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+1)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+1)*ny*nx]
+xv[(ix+1)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+-1)*nx+(iz+1)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+1)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+1)*nx+(iz+1)*ny*nx]
;}
else if (iz == nz-1 && ix>0 && ix<nx-1 && iy>0 && iy<ny-1){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+-1)+(iy+-1)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+1)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+-1)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+1)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+1)*nx+(iz+0)*ny*nx]
;}
else if (iy == 0 && ix>0 && ix<nx-1 && iz>0 && iz<nz-1){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+-1)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+1)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+-1)+(iy+1)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+1)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+1)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+1)+(iy+1)*nx+(iz+1)*ny*nx]
;}
else if (iy == ny-1 && ix>0 && ix<nx-1 && iz>0 && iz<nz-1){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+-1)+(iy+-1)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+-1)*nx+(iz+1)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+1)+(iy+-1)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+-1)*nx+(iz+1)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+1)*ny*nx]
;}
else if (ix == 0 && iy>0 && iy<ny-1 && iz>0 && iz<nz-1){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+0)+(iy+-1)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+-1)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+1)*ny*nx]
+xv[(ix+1)+(iy+-1)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+1)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+-1)*ny*nx]
+xv[(ix+1)+(iy+1)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+1)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+1)*ny*nx]
+xv[(ix+1)+(iy+1)*nx+(iz+1)*ny*nx]
;}
else if (ix == nx-1 && iy>0 && iy<ny-1 && iz>0 && iz<nz-1){
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx] 
+xv[(ix+-1)+(iy+-1)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+-1)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+-1)*nx+(iz+1)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+0)*nx+(iz+1)*ny*nx]
+xv[(ix+-1)+(iy+1)*nx+(iz+-1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+-1)*ny*nx]
+xv[(ix+-1)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+0)*ny*nx]
+xv[(ix+-1)+(iy+1)*nx+(iz+1)*ny*nx]
+xv[(ix+0)+(iy+1)*nx+(iz+1)*ny*nx]
;}

//bulk
else{
xv[ix+iy*nx+iz*ny*nx] = rv[ix+iy*nx+iz*ny*nx]
+ xv[ix+-1 + (iy+-1)*nx + (iz+-1)*nx*ny]
+ xv[ix+-1 + (iy+-1)*nx + (iz+0)*nx*ny]
+ xv[ix+-1 + (iy+-1)*nx + (iz+1)*nx*ny]
+ xv[ix+-1 + (iy+0)*nx + (iz+-1)*nx*ny]
+ xv[ix+-1 + (iy+0)*nx + (iz+0)*nx*ny]
+ xv[ix+-1 + (iy+0)*nx + (iz+1)*nx*ny]
+ xv[ix+-1 + (iy+1)*nx + (iz+-1)*nx*ny]
+ xv[ix+-1 + (iy+1)*nx + (iz+0)*nx*ny]
+ xv[ix+-1 + (iy+1)*nx + (iz+1)*nx*ny]
+ xv[ix+0 + (iy+-1)*nx + (iz+-1)*nx*ny]
+ xv[ix+0 + (iy+-1)*nx + (iz+0)*nx*ny]
+ xv[ix+0 + (iy+-1)*nx + (iz+1)*nx*ny]
+ xv[ix+0 + (iy+0)*nx + (iz+-1)*nx*ny]
+ xv[ix+0 + (iy+0)*nx + (iz+1)*nx*ny]
+ xv[ix+0 + (iy+1)*nx + (iz+-1)*nx*ny]
+ xv[ix+0 + (iy+1)*nx + (iz+0)*nx*ny]
+ xv[ix+0 + (iy+1)*nx + (iz+1)*nx*ny]
+ xv[ix+1 + (iy+-1)*nx + (iz+-1)*nx*ny]
+ xv[ix+1 + (iy+-1)*nx + (iz+0)*nx*ny]
+ xv[ix+1 + (iy+-1)*nx + (iz+1)*nx*ny]
+ xv[ix+1 + (iy+0)*nx + (iz+-1)*nx*ny]
+ xv[ix+1 + (iy+0)*nx + (iz+0)*nx*ny]
+ xv[ix+1 + (iy+0)*nx + (iz+1)*nx*ny]
+ xv[ix+1 + (iy+1)*nx + (iz+-1)*nx*ny]
+ xv[ix+1 + (iy+1)*nx + (iz+0)*nx*ny]
+ xv[ix+1 + (iy+1)*nx + (iz+1)*nx*ny]
;}

      
xv[ix+iy*nx+iz*ny*nx] /= diagonal_element;
        }
      }
    }

  
return 0;
}