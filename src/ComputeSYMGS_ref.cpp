
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
#include "idx.hpp"
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

  //const local_int_t nrow = A.localNumberOfRows;
  //double ** matrixDiagonal = A.matrixDiagonal;  // An array of pointers to the diagonal entries A.matrixValues
  //const double * const rv = r.values;
  //double * const xv = x.values;

  //for (local_int_t i=0; i< nrow; i++) {
    //const double * const currentValues = A.matrixValues[i];
    //const local_int_t * const currentColIndices = A.mtxIndL[i];
    //const int currentNumberOfNonzeros = A.nonzerosInRow[i];
    //const double  currentDiagonal = matrixDiagonal[i][0]; // Current diagonal value
    //double sum = rv[i]; // RHS value

    //for (int j=0; j< currentNumberOfNonzeros; j++) {
      //local_int_t curCol = currentColIndices[j];
      //sum -= currentValues[j] * xv[curCol];
    //}
    //sum += xv[i]*currentDiagonal; // Remove diagonal contribution from previous loop

    //xv[i] = sum/currentDiagonal;

  //}

  //// Now the back sweep.

  //for (local_int_t i=nrow-1; i>=0; i--) {
    //const double * const currentValues = A.matrixValues[i];
    //const local_int_t * const currentColIndices = A.mtxIndL[i];
    //const int currentNumberOfNonzeros = A.nonzerosInRow[i];
    //const double  currentDiagonal = matrixDiagonal[i][0]; // Current diagonal value
    //double sum = rv[i]; // RHS value

    //for (int j = 0; j< currentNumberOfNonzeros; j++) {
      //local_int_t curCol = currentColIndices[j];
      //sum -= currentValues[j]*xv[curCol];
    //}
    //sum += xv[i]*currentDiagonal; // Remove diagonal contribution from previous loop

    //xv[i] = sum/currentDiagonal;
  //}

////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////

  const double * const rv = r.values;
  double * const xv = x.values;

  global_int_t nx = A.geom->nx;
  global_int_t ny = A.geom->ny;
  global_int_t nz = A.geom->nz;
  global_int_t ng = 1; // number of ghost points FIXME should be declared "more globally"
  const local_int_t nrow_ghost = (nx+2*ng)*(ny+2*ng)*(nz+2*ng); // local number of rows *including* ghost points
  double* xg = new double[nrow_ghost]{0.0}; // Copy of x with ghost points

  // copy xv to xg TODO HACK fix this
  for (local_int_t ix=0; ix<nx; ++ix) {
    for (local_int_t iy=0; iy<ny; ++iy) {
      for (local_int_t iz=0; iz<nz; ++iz) {
        xg[idx(ix,iy,iz,nx,ny,nz,ng)] = xv[idx(ix,iy,iz,nx,ny,nz)];
      }
    }
  }

  const double currentDiagonal = 26.0; // Current diagonal value CHANGEME when changing stencil

#ifndef HPCG_NO_OPENMP
  #pragma omp parallel for
#endif
  for (local_int_t ix=0; ix<nx; ++ix) {
    for (local_int_t iy=0; iy<ny; ++iy) {
      for (local_int_t iz=0; iz<nz; ++iz) {
        double sum = rv[idx(ix,iy,iz,nx,ny,nz)]; // RHS value
        for(int sx=-1; sx<=1; ++sx) {
          for(int sy=-1; sy<=1; ++sy) {
            for(int sz=-1; sz<=1; ++sz) {
              sum -= -1.*xg[idx(ix+sx, iy+sy, iz+sz, nx, ny, nz, ng)];
            }
          }
        }
        sum += -1.*xg[idx(ix+0, iy+0, iz+0, nx, ny, nz, ng)]; // Remove diagnoal component entirely

        xg[idx(ix,iy,iz,nx,ny,nz,ng)] = sum/currentDiagonal;
      }
    }
  }

#ifndef HPCG_NO_OPENMP
  #pragma omp parallel for
#endif
  for (local_int_t ix=nx-1; ix>=0; --ix) {
    for (local_int_t iy=ny-1; iy>=0; --iy) {
      for (local_int_t iz=nz-1; iz>=0; --iz) {
        double sum = rv[idx(ix,iy,iz,nx,ny,nz)]; // RHS value
        for(int sx=-1; sx<=1; ++sx) {
          for(int sy=-1; sy<=1; ++sy) {
            for(int sz=-1; sz<=1; ++sz) {
              sum -= -1.*xg[idx(ix+sx, iy+sy, iz+sz, nx, ny, nz, ng)];
            }
          }
        }
        sum += -1.*xg[idx(ix+0, iy+0, iz+0, nx, ny, nz, ng)]; // Remove diagnoal component entirely

        xv[idx(ix,iy,iz,nx,ny,nz)] = sum/currentDiagonal;
      }
    }
  }

  delete[] xg;

  return 0;
}

