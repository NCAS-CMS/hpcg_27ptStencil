
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

#include "ComputeSPMV_ref.hpp"
#include <iostream> // remove

#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#endif

#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif
#include <cassert>

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
int ComputeSPMV_ref( const SparseMatrix & A, Vector & x, Vector & y) {

  assert(x.localLength>=A.localNumberOfColumns); // Test vector lengths
  assert(y.localLength>=A.localNumberOfRows);

#ifndef HPCG_NO_MPI
    ExchangeHalo(A,x);
#endif

  const double * const xv = x.values;
  double * const yv = y.values;
  const int* map_fn = (*A.geom->map_w3);//map function
  int*** dofmap_fn = (*A.geom->dofmap);//map function
  double * op1 = A.op1;
  double * op2 = A.op2;
  double * op3 = A.op3;
  double * op4 = A.op4;
  double * op5 = A.op5;
  double * op6 = A.op6;
  double * op7 = A.op7;
  double * op8 = A.op8;
  double * op9 = A.op9;
// A would need the stuff about indexing etc - x and y would be in custom vector form
  // cell always start at 0 for this
//  std::cout << yv[map_fn[0] ] << " " << op1[map_fn[0] ] << " " << xv[dofmap_fn[0][0][0]] <<std::endl;

  for (int cell = 0; cell< A.geom->nxy; cell++)
  {
    // should really do map_fn[cell] and dofmap_fn[cell] here - but no fortran equivalent
    for (int k = 0; k < A.geom->nz; k++)
    {
//        std::cout << " here " << cell << " " << k << std::endl;
//        yv[0] = 1.0;
//        yv[map_fn[cell]+k] = op1[map_fn[cell]+k] * xv[dofmap_fn[cell][0][0] + k];
        yv[map_fn[cell] + k] = op1[map_fn[cell] + k] * xv[dofmap_fn[cell][0][0] + k]
                  +op5[map_fn[cell] + k]  * xv[dofmap_fn[cell][0][1] + k]
                  +op4[map_fn[cell] + k]  * xv[dofmap_fn[cell][1][1] + k]
                  +op3[map_fn[cell] + k]  * xv[dofmap_fn[cell][2][1] + k]
                  +op2[map_fn[cell] + k]  * xv[dofmap_fn[cell][3][1] + k];
    }

    for (int k = 0; k <=A.geom->nz-3; k++)
    {
      yv[map_fn[cell] + k] = yv[map_fn[cell] + k] + op6[map_fn[cell] + k] * xv[map_fn[cell] + k+1]
                            + op7[map_fn[cell] + k] * xv[map_fn[cell] + k+2];

    }

    int k = A.geom->nz - 2;
    yv[map_fn[cell] + k] = yv[map_fn[cell] + k] + op6[map_fn[cell] + k] * xv[map_fn[cell] + k+1];

    //! Coefficients on layers below
    k = 1;
    yv[map_fn[cell] + k] = yv[map_fn[cell] + k] + op8[map_fn[cell] + k] * xv[map_fn[cell] + k-1];
    for( k = 2; k <= A.geom->nz - 1;k++)
    {
      yv[map_fn[cell] + k] = yv[map_fn[cell] + k] + op8[map_fn[cell] + k] * xv[map_fn[cell] + k-1]
                              + op9[map_fn[cell] + k] * xv[map_fn[cell] + k-2];
    }

  }

/*
  const double * const xv = x.values;
  double * const yv = y.values;
  const local_int_t nrow = A.localNumberOfRows;
#ifndef HPCG_NO_OPENMP
  #pragma omp parallel for
#endif
  for (local_int_t i=0; i< nrow; i++)  {
    double sum = 0.0;
    const double * const cur_vals = A.matrixValues[i];
    const local_int_t * const cur_inds = A.mtxIndL[i];
    const int cur_nnz = A.nonzerosInRow[i];

    for (int j=0; j< cur_nnz; j++)
      sum += cur_vals[j]*xv[cur_inds[j]];
    yv[i] = sum;
  }
*/
  return 0;
}
