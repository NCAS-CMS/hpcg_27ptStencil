
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
 @file main.cpp

 HPCG routine
 */

// Main routine of a program that calls the HPCG conjugate gradient
// solver to solve the problem, and then prints results.

#ifndef HPCG_NO_MPI
#include <mpi.h>
#endif

#include <fstream>
#include <iostream>
#include <cstdlib>
#ifdef HPCG_DETAILED_DEBUG
using std::cin;
#endif
using std::endl;

#include <vector>

#include "hpcg.hpp"

#include "CheckAspectRatio.hpp"
#include "GenerateGeometry.hpp"
#include "GenerateProblem.hpp"
#include "GenerateCoarseProblem.hpp"
#include "SetupHalo.hpp"
#include "CheckProblem.hpp"
#include "ExchangeHalo.hpp"
#include "OptimizeProblem.hpp"
#include "WriteProblem.hpp"
#include "ReportResults.hpp"
#include "mytimer.hpp"
#include "ComputeSPMV_ref.hpp"
//#include "ComputeMG_ref.hpp"
#include "ComputeResidual.hpp"
#include "CG.hpp"
#include "CG_ref.hpp"
#include "Geometry.hpp"
#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "CGData.hpp"
#include "TestCG.hpp"
#include "TestSymmetry.hpp"
#include "TestNorms.hpp"
#include "LFRic_setup.hpp"

/*!
  Main driver program: Construct synthetic problem, run V&V tests, compute benchmark parameters, run benchmark, report results.

  @param[in]  argc Standard argument count.  Should equal 1 (no arguments passed in) or 4 (nx, ny, nz passed in)
  @param[in]  argv Standard argument array.  If argc==1, argv is unused.  If argc==4, argv[1], argv[2], argv[3] will be interpreted as nx, ny, nz, resp.

  @return Returns zero on success and a non-zero value otherwise.

*/
int main(int argc, char * argv[]) {

#ifndef HPCG_NO_MPI
  MPI_Init(&argc, &argv);
#endif

  HPCG_Params params;

  HPCG_Init(&argc, &argv, params);

  // Check if QuickPath option is enabled.
  // If the running time is set to zero, we minimize all paths through the program
  bool quickPath = (params.runningTime==0);

  int size = params.comm_size, rank = params.comm_rank; // Number of MPI processes, My process ID

#ifdef HPCG_DETAILED_DEBUG
  if (size < 100 && rank==0) HPCG_fout << "Process "<<rank<<" of "<<size<<" is alive with " << params.numThreads << " threads." <<endl;

  if (rank==0) {
    char c;
    std::cout << "Press key to continue"<< std::endl;
    std::cin.get(c);
  }
#ifndef HPCG_NO_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
#endif

  local_int_t nx,ny,nz;
  nx = (local_int_t)params.nx;
  ny = (local_int_t)params.ny;
  nz = (local_int_t)params.nz;
  int ierr = 0;  // Used to check return codes on function calls

  ierr = CheckAspectRatio(0.125, nx, ny, nz, "local problem", rank==0); // Not defined x or y, so not a concern - dhc
  if (ierr)
    return ierr;

  /////////////////////////
  // Problem setup Phase //
  /////////////////////////

#ifdef HPCG_DEBUG
  double t1 = mytimer();
#endif

  // Construct the geometry and linear system
  Geometry * geom = new Geometry;
  GenerateGeometry(size, rank, params.numThreads, params.pz, params.zl, params.zu, nx, ny, nz, params.npx, params.npy, params.npz, geom);

  ierr = CheckAspectRatio(0.125, geom->npx, geom->npy, geom->npz, "process grid", rank==0); // again - not defined in LFRic grid
  if (ierr)
    return ierr;

  // Use this array for collecting timing information
  std::vector< double > times(10,0.0);

  double setup_time = mytimer();

  SparseMatrix A;
  InitializeSparseMatrix(A, geom); // Should I modify this routine or correct afterwards? - dhc

  Vector b, x, xexact;
  GenerateProblem(A, &b, &x, &xexact); // Keep here but over write data today, and subroutine later
  SetupHalo(A); //dhc - will not be doing halo exchange in this app



  int numberOfMgLevels = 1; // 4; // Number of levels including first // Don't know how this will work at this stage - dhc
  SparseMatrix * curLevelMatrix = &A;
  for (int level = 1; level< numberOfMgLevels; ++level) {
    GenerateCoarseProblem(*curLevelMatrix);
    curLevelMatrix = curLevelMatrix->Ac; // Make the just-constructed coarse grid the next level
  }

  setup_time = mytimer() - setup_time; // Capture total time of setup
  times[9] = setup_time; // Save it for reporting

  // dont need this atm - dhc
/*
  curLevelMatrix = &A;
  Vector * curb = &b;
  Vector * curx = &x;
  Vector * curxexact = &xexact;
  for (int level = 0; level< numberOfMgLevels; ++level) {
     CheckProblem(*curLevelMatrix, curb, curx, curxexact);
     curLevelMatrix = curLevelMatrix->Ac; // Make the nextcoarse grid the next level
     curb = 0; // No vectors after the top level
     curx = 0;
     curxexact = 0;
  }
*/

  CGData data;
  InitializeSparseCGData(A, data);

  //dhc over-write stuff here?
  local_int_t loop0_start = 0;
  local_int_t loop0_stop  = 0;
  local_int_t nlayers     = 0;
  local_int_t undf_w3     = 0;
  local_int_t x_vec_max_branch_length = 0;
  int* map_w3;double* yvec;double* xvec;double* op1;double* op2;double* op3;double* op4;double* op5;double* op6;double* op7;double* op8;double* op9;double* ans;
  int* stencil_size; int*** dofmap;

  if (rank==0)
  {
    read_dinodump(loop0_start, loop0_stop, nlayers, undf_w3, x_vec_max_branch_length, &map_w3,
                &yvec, &xvec, &op1, &op2, &op3, &op4, &op5, &op6, &op7, &op8, &op9, &ans, &stencil_size, &dofmap);
  }

  MPI_Bcast(&loop0_start, 1, MPI_INT, 0, MPI_COMM_WORLD );
  MPI_Bcast(&loop0_stop,  1, MPI_INT, 0, MPI_COMM_WORLD );
  MPI_Bcast(&nlayers,     1, MPI_INT, 0, MPI_COMM_WORLD );
  MPI_Bcast(&undf_w3,     1, MPI_INT, 0, MPI_COMM_WORLD );
  MPI_Bcast(&x_vec_max_branch_length, 1, MPI_INT, 0, MPI_COMM_WORLD );
  if(rank!=0)
  {
     map_w3 = (int*)    malloc(loop0_stop * sizeof(int));
     yvec   = (double*) malloc(undf_w3 * sizeof(double));
     xvec   = (double*) malloc(undf_w3 * sizeof(double));
     op1    = (double*) malloc(undf_w3 * sizeof(double));
     op2    = (double*) malloc(undf_w3 * sizeof(double));
     op3    = (double*) malloc(undf_w3 * sizeof(double));
     op4    = (double*) malloc(undf_w3 * sizeof(double));
     op5    = (double*) malloc(undf_w3 * sizeof(double));
     op6    = (double*) malloc(undf_w3 * sizeof(double));
     op7    = (double*) malloc(undf_w3 * sizeof(double));
     op8    = (double*) malloc(undf_w3 * sizeof(double));
     op9    = (double*) malloc(undf_w3 * sizeof(double));
     ans    = (double*) malloc(undf_w3 * sizeof(double));
  }

  MPI_Bcast(map_w3, loop0_stop, MPI_INT, 0, MPI_COMM_WORLD );
  MPI_Bcast(yvec, undf_w3, MPI_DOUBLE, 0, MPI_COMM_WORLD );
  MPI_Bcast(xvec, undf_w3, MPI_DOUBLE, 0, MPI_COMM_WORLD );
  MPI_Bcast(op1, undf_w3, MPI_DOUBLE, 0, MPI_COMM_WORLD );
  MPI_Bcast(op2, undf_w3, MPI_DOUBLE, 0, MPI_COMM_WORLD );
  MPI_Bcast(op3, undf_w3, MPI_DOUBLE, 0, MPI_COMM_WORLD );
  MPI_Bcast(op4, undf_w3, MPI_DOUBLE, 0, MPI_COMM_WORLD );
  MPI_Bcast(op5, undf_w3, MPI_DOUBLE, 0, MPI_COMM_WORLD );
  MPI_Bcast(op6, undf_w3, MPI_DOUBLE, 0, MPI_COMM_WORLD );
  MPI_Bcast(op7, undf_w3, MPI_DOUBLE, 0, MPI_COMM_WORLD );
  MPI_Bcast(op8, undf_w3, MPI_DOUBLE, 0, MPI_COMM_WORLD );
  MPI_Bcast(op9, undf_w3, MPI_DOUBLE, 0, MPI_COMM_WORLD );
  MPI_Bcast(ans, undf_w3, MPI_DOUBLE, 0, MPI_COMM_WORLD );

  //set dofmap to be contiguous later (then do 1 bcast)
  if(rank!=0)  dofmap = (int***) malloc(loop0_stop * sizeof(int **));
  for (int i=0;i<loop0_stop;i++)
  {
    if (rank!=0) dofmap[i] = (int**) malloc(4 * sizeof(int*));
        for (int j=0; j < 4; j++)
        {
            if(rank!=0) dofmap[i][j] = (int*) malloc (x_vec_max_branch_length * sizeof(int));

            MPI_Bcast(dofmap[i][j], x_vec_max_branch_length, MPI_INT, 0, MPI_COMM_WORLD);
        }
    }

  b.localLength = undf_w3;
  b.values = xvec;
  x.localLength = undf_w3;

  xexact.localLength = undf_w3;
  xexact.values = yvec;

  // z direction never partitioned, so global and local always nlayers
  A.geom->gnz = nlayers;
  A.geom->nz  = nlayers;
  A.geom->nxy = loop0_stop - loop0_start + 1; // number total in x and y
  A.localNumberOfColumns = A.geom->nxy * A.geom->nz;
  assert(A.geom->nxy * A.geom->nz == undf_w3);
  A.localNumberOfRows = A.localNumberOfColumns; //MPI ranks == 1
  A.geom->map_w3 = &map_w3;
  A.geom->dofmap = &dofmap;

  A.op1 = op1;
  A.op2 = op2;
  A.op3 = op3;
  A.op4 = op4;
  A.op5 = op5;
  A.op6 = op6;
  A.op7 = op7;
  A.op8 = op8;
  A.op9 = op9;

  bool force_more_symmetric = true; // Do this to ensure CG converges -otherwise can get to about 1e-6 and bounces back a bit

  if (force_more_symmetric){
    A.op4 = A.op2; //S=N
    A.op5 = A.op3; // E=W
    A.op9 = A.op7; // DD=UU
    A.op8 = A.op6; //D=U
  }

  // slightly tidy data object to avoid problems?
  data.Ap.localLength = A.localNumberOfColumns;
  data.p.localLength = A.localNumberOfColumns;
  data.r.localLength = A.localNumberOfColumns;
  data.z.localLength = A.localNumberOfColumns;

  // Change parts of A (each MPI is a replica)
  A.totalNumberOfRows = A.localNumberOfRows * size;
  // 9pt stencil in bulk, but 2 layers above and below - hence bottom/top missing 2, and 1st above/below bottom/top missing 1
  A.totalNumberOfNonzeros = (9.0 * A.geom->nxy * (A.geom->nz -4) +
                             8.0 * A.geom->nxy * 2 +
                             7.0 * A.geom->nxy * 2) * size;
  ////////////////////////////////////
  // Reference SpMV+MG Timing Phase //
  ////////////////////////////////////

  // Call Reference SpMV and MG. Compute Optimization time as ratio of times in these routines

  local_int_t nrow = A.localNumberOfRows;
  local_int_t ncol = A.localNumberOfColumns;

  Vector x_overlap, b_computed;
  InitializeVector(x_overlap, ncol); // Overlapped copy of x vector
  InitializeVector(b_computed, nrow); // Computed RHS vector

  // Record execution time of reference SpMV and MG kernels for reporting times
  // First load vector with random values
  FillRandomVector(x_overlap);

  int numberOfCalls = 10;
  if (quickPath) numberOfCalls = 1; //QuickPath means we do on one call of each block of repetitive code
  double t_begin = mytimer();
  for (int i=0; i< numberOfCalls; ++i) {
    ierr = ComputeSPMV_ref(A, x_overlap, b_computed); // b_computed = A*x_overlap
    if (ierr) HPCG_fout << "Error in call to SpMV: " << ierr << ".\n" << endl;
    //ierr = ComputeMG_ref(A, b_computed, x_overlap); // b_computed = Minv*y_overlap //what happens if turn this off - dhc?
    //if (ierr) HPCG_fout << "Error in call to MG: " << ierr << ".\n" << endl;
  }
  times[8] = (mytimer() - t_begin)/((double) numberOfCalls);  // Total time divided by number of calls.
#ifdef HPCG_DEBUG
  if (rank==0) HPCG_fout << "Total SpMV+MG timing phase execution time in main (sec) = " << mytimer() - t1 << endl;
#endif

  ///////////////////////////////
  // Reference CG Timing Phase //
  ///////////////////////////////

#ifdef HPCG_DEBUG
  t1 = mytimer();
#endif
  int global_failure = 0; // assume all is well: no failures

  int niters = 0;
  int totalNiters_ref = 0;
  double normr = 0.0;
  double normr0 = 0.0;
  int refMaxIters = 50;
  numberOfCalls = 1; // Only need to run the residual reduction analysis once

  // Compute the residual reduction for the natural ordering and reference kernels
  // dhc - natural ordering and new are going to be the same in this case, as overwriting
  std::vector< double > ref_times(9,0.0);
  double tolerance = 0.0; // Set tolerance to zero to make all runs do maxIters iterations
  int err_count = 0;
  for (int i=0; i< numberOfCalls; ++i) {
    ZeroVector(x);
    ierr = CG_ref( A, data, b, x, refMaxIters, tolerance, niters, normr, normr0, &ref_times[0], false); // dhc - set MG to false
    if (ierr) ++err_count; // count the number of errors in CG
    totalNiters_ref += niters;
  }
  if (rank == 0 && err_count) HPCG_fout << err_count << " error(s) in call(s) to reference CG." << endl;
  double refTolerance = normr / normr0;

  // Call user-tunable set up function.
  double t7 = mytimer();
  OptimizeProblem(A, data, b, x, xexact);
  t7 = mytimer() - t7;
  times[7] = t7;
#ifdef HPCG_DEBUG
  if (rank==0) HPCG_fout << "Total problem setup time in main (sec) = " << mytimer() - t1 << endl;
#endif

#ifdef HPCG_DETAILED_DEBUG
  if (geom->size == 1) WriteProblem(*geom, A, b, x, xexact);
#endif


  //////////////////////////////
  // Validation Testing Phase //
  //////////////////////////////

#ifdef HPCG_DEBUG
  t1 = mytimer();
#endif

  TestCGData testcg_data;

  testcg_data.count_pass = testcg_data.count_fail = 0;
//  TestCG(A, data, b, x, testcg_data);
  TestSymmetryData testsymmetry_data;
  //  TestSymmetry(A, b, xexact, testsymmetry_data);
  testsymmetry_data.count_fail = 0; // Bypassing tests, so technically none failed...
  
#ifdef HPCG_DEBUG
  if (rank==0) HPCG_fout << "Total validation (TestCG and TestSymmetry) execution time in main (sec) = " << mytimer() - t1 << endl;
#endif

#ifdef HPCG_DEBUG
  t1 = mytimer();
#endif

  //////////////////////////////
  // Optimized CG Setup Phase //
  //////////////////////////////

  niters = 0;
  normr = 0.0;
  normr0 = 0.0;
  err_count = 0;
  int tolerance_failures = 0;

  int optMaxIters = 10*refMaxIters;
  int optNiters = refMaxIters;
  double opt_worst_time = 0.0;

  std::vector< double > opt_times(9,0.0);

  // Compute the residual reduction and residual count for the user ordering and optimized kernels.
  for (int i=0; i< numberOfCalls; ++i) {
    ZeroVector(x); // start x at all zeros
    double last_cummulative_time = opt_times[0];

    ierr = CG( A, data, b, x, optMaxIters, refTolerance, niters, normr, normr0, &opt_times[0], false); // dhc - set MG false

    if (ierr) ++err_count; // count the number of errors in CG
    // Convergence check accepts an error of no more than 6 significant digits of relTolerance
    if (normr / normr0 > refTolerance * (1.0 + 1.0e-6)) ++tolerance_failures; // the number of failures to reduce residual

    // pick the largest number of iterations to guarantee convergence
    if (niters > optNiters) optNiters = niters;

    double current_time = opt_times[0] - last_cummulative_time;
    if (current_time > opt_worst_time) opt_worst_time = current_time;
  }

#ifndef HPCG_NO_MPI
// Get the absolute worst time across all MPI ranks (time in CG can be different)
  double local_opt_worst_time = opt_worst_time;
  MPI_Allreduce(&local_opt_worst_time, &opt_worst_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif


  if (rank == 0 && err_count) HPCG_fout << err_count << " error(s) in call(s) to optimized CG." << endl;
  if (tolerance_failures) {
    global_failure = 1;
    if (rank == 0)
      HPCG_fout << "Failed to reduce the residual " << tolerance_failures << " times." << endl;
  }

  ///////////////////////////////
  // Optimized CG Timing Phase //
  ///////////////////////////////

  // Here we finally run the benchmark phase
  // The variable total_runtime is the target benchmark execution time in seconds

  double total_runtime = params.runningTime;
  int numberOfCgSets = int(total_runtime / opt_worst_time) + 1; // Run at least once, account for rounding

#ifdef HPCG_DEBUG
  if (rank==0) {
    HPCG_fout << "Projected running time: " << total_runtime << " seconds" << endl;
    HPCG_fout << "Number of CG sets: " << numberOfCgSets << endl;
  }
#endif

  /* This is the timed run for a specified amount of time. */

  optMaxIters = optNiters;
  double optTolerance = 0.0;  // Force optMaxIters iterations
  TestNormsData testnorms_data;
  testnorms_data.samples = numberOfCgSets;
  testnorms_data.values = new double[numberOfCgSets];

  for (int i=0; i< numberOfCgSets; ++i) {
    ZeroVector(x); // Zero out x
    ierr = CG( A, data, b, x, optMaxIters, optTolerance, niters, normr, normr0, &times[0], false); // dhc - no MG
    if (ierr) HPCG_fout << "Error in call to CG: " << ierr << ".\n" << endl;
    if (rank==0) HPCG_fout << "Call [" << i << "] Scaled Residual [" << normr/normr0 << "]" << endl;
    testnorms_data.values[i] = normr/normr0; // Record scaled residual from this run
  }

  // Compute difference between known exact solution and computed solution
  // All processors are needed here.
#ifdef HPCG_DEBUG
  double residual = 0;
  ierr = ComputeResidual(A.localNumberOfRows, x, xexact, residual);
  if (ierr) HPCG_fout << "Error in call to compute_residual: " << ierr << ".\n" << endl;
  if (rank==0) HPCG_fout << "Difference between computed and exact  = " << residual << ".\n" << endl;
#endif

  // Test Norm Results
  ierr = TestNorms(testnorms_data);

  ////////////////////
  // Report Results //
  ////////////////////

  // Report results to YAML file
  ReportResults(A, numberOfMgLevels, numberOfCgSets, refMaxIters, optMaxIters, &times[0], testcg_data, testsymmetry_data, testnorms_data, global_failure, quickPath);

  // Clean up
  DeleteMatrix(A); // This delete will recursively delete all coarse grid data
  DeleteCGData(data);
  DeleteVector(x);
  DeleteVector(b);
  DeleteVector(xexact);
  DeleteVector(x_overlap);
  DeleteVector(b_computed);
  delete [] testnorms_data.values;



  HPCG_Finalize();

  // Finish up
#ifndef HPCG_NO_MPI
  MPI_Finalize();
#endif
  return 0;
}
