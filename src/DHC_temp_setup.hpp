#ifndef DHC_temp_setup_HPP
#define DHC_temp_setup_HPP
#include "Geometry.hpp"

int check_similarity_arrays(double** v1, double** v2, const int len_arrays);

//void apply_helmholtz_operator_code(local_int_t&nlayers, double** y_vec, local_int_t& undf_w3);
void apply_helmholtz_operator_code(local_int_t&nlayers, local_int_t& cell, int** map_w3, double** yvec, double** xvec,
                                   double** op1, double** op2, double** op3, double** op4, double** op5, double** op6,
                                   double** op7, double** op8, double** op9, double** ans, local_int_t& undf_w3, int** stencil_size, int**** dofmap);

void read_dinodump(local_int_t& loop0_start, local_int_t& loop0_stop, local_int_t& nlayers, local_int_t& undf_w3, local_int_t& x_vec_max_branch_length,
                   int** map_w3, double** yvec, double** xvec, double** op1, double** op2, double** op3, double** op4, double** op5, double** op6,
                   double** op7, double** op8, double** op9, double** ans, int** stencil_size, int**** dofmap);
#endif // DHC_temp_setup_HPP
