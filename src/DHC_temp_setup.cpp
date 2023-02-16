// read dinodump.dat and save some stuff
// recreating microbenchmark

#include "GenerateGeometry.hpp"
#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>
#include <algorithm>
using namespace std;

int check_similarity_arrays(double** v1, double** v2, const int len_arrays)//, int &n_different){}
{
    // how many elements of v1 and v2 differ by more than hard coded diff
    double diff = 0.0001;
    int n_different = 0;
    for (int i = 0; i < len_arrays; i++)
    {
        if (abs((*v1)[i] - (*v2)[i]) > diff) {n_different++;}
    }
    return n_different;
}

void read_part_vector_from_string(string line, double** local_array, int &array_counter, string space_delimiter){
    vector<string> words{};
    size_t pos = 0;
    while ((pos = line.find(space_delimiter)) != string::npos) {
        words.push_back(line.substr(0, pos));
           line.erase(0, pos + space_delimiter.length());
        }
//    std::cout << "last bit " << line << std::endl; exit(9);
    // add the last bit additionally
    words.push_back(line);
    for (const auto &str : words) {
       if(!str.empty()){
//              temp_stencil[array_counter] = stod(str);
              (*local_array)[array_counter] = stod(str);
          array_counter++;
       }
    }
}

void read_part_vector_from_string(string line, int** local_array, int &array_counter, string space_delimiter){
    vector<string> words{};
    size_t pos = 0;
    while ((pos = line.find(space_delimiter)) != string::npos) {
        words.push_back(line.substr(0, pos));
           line.erase(0, pos + space_delimiter.length());
        }
//    std::cout << "last bit " << line << std::endl; exit(9);
    // add the last bit additionally
    words.push_back(line);
    for (const auto &str : words) {
       if(!str.empty()){
//              temp_stencil[array_counter] = stod(str);
              (*local_array)[array_counter] = stoi(str);
          array_counter++;
       }
    }
}



void apply_helmholtz_operator_code(local_int_t &nlayers, local_int_t &cell, int** map_w3, double** yvec, double** xvec,
                                   double** op1, double** op2, double** op3, double** op4, double** op5, double** op6,
                                   double** op7, double** op8, double** op9, double** ans, local_int_t& undf_w3, int** stencil_size, int**** dofmap)
{
  /*
     this plus the loop over the surface grid makes the SPMV equivalent
     op1 = Helm_C; op2 = Helm_N; op3 = Helm_E; op4 = Helm_S;
     op5 = Helm_W; op6 = Helm_U; op7 = Helm_UU; op8 = Helm_D; op9 = Helm_DD;
  */

  //checking numbers vs spreadsheet
/*  std::cout << "map " << (*map_w3)[0] << " " << (*map_w3)[1] << " " << (*map_w3)[3456 - 1] << std::endl;
  std::cout << "map " << (*map_w3)[4] << " " << (*map_w3)[5] << " " << (*map_w3)[6] << std::endl;
  std::cout << "yvec " << (*yvec)[0] << " " << (*yvec)[1] << " " << (*yvec)[undf_w3 - 1] << std::endl;
  std::cout << "xvec " << (*xvec)[0] << " " << (*xvec)[1] << " " << (*xvec)[undf_w3 - 1] << std::endl;
  std::cout << "op1  " << (*op1)[0] << " " << (*op1)[undf_w3 - 1] << std::endl;
  std::cout << "op2  " << (*op2)[0] << " " << (*op2)[undf_w3 - 1] << std::endl;
  std::cout << "op3  " << (*op3)[0] << " " << (*op3)[undf_w3 - 1] << std::endl;
  std::cout << "op4  " << (*op4)[0] << " " << (*op4)[undf_w3 - 1] << std::endl;
  std::cout << "op5  " << (*op5)[0] << " " << (*op5)[undf_w3 - 1] << std::endl;
  std::cout << "op6  " << (*op6)[0] << " " << (*op6)[undf_w3 - 1] << std::endl;
  std::cout << "op7  " << (*op7)[0] << " " << (*op7)[undf_w3 - 1] << std::endl;
  std::cout << "op8  " << (*op8)[0] << " " << (*op8)[undf_w3 - 1] << std::endl;
  std::cout << "op9  " << (*op9)[0] << " " << (*op9)[undf_w3 - 1] << std::endl;
  std::cout << "ans  " << (*ans)[0] << " " << (*ans)[undf_w3 - 1] << std::endl;
  std::cout <<"FIX stencil size if needed" << std::endl;
  std::cout << "dofmap " << (*dofmap)[0][0][0] << " " << (*dofmap)[3456-1][4-1][2-1] << std::endl;*/
//  std::cout << "stsi " << (*stencil_size)[0] <<" " << (*stencil_size)[4*3456 - 1] << std::endl;
  // dimensions dofmap [cells][4][branch length
  // use pointer arithmetic for speed?
  // map_w3 is just treated as a vector, as dimension size is 1 in other axis
//  int cell = 0;
/*  std::cout << (*map_w3)[0]<< " " << (*yvec)[0]<< " " << (*xvec)[0] << std::endl;
  std::cout << (*op1)[(*map_w3)[0]]<< " " << (*xvec)[(*dofmap)[cell][0][0]]<< " " <<
               (*op5)[(*map_w3)[0]]<< " " << (*xvec)[(*dofmap)[cell][1][0]]<< " " <<
               (*op4)[(*map_w3)[0]]<< " " << (*xvec)[(*dofmap)[cell][1][1]]<< " " <<
               (*op3)[(*map_w3)[0]]<< " " << (*xvec)[(*dofmap)[cell][1][2]]<< " " <<
               (*op2)[(*map_w3)[0]]<< " " << (*xvec)[(*dofmap)[cell][1][3]]<< std::endl;*/
  // first match map and dofmap - then change then so that they are one less (in readdinodump below) - c vs fort
  for (int k = 0; k < nlayers; k++)
  {
/*      std::cout << (*yvec)[(*map_w3)[cell] + k] << " " << (*op1)[(*map_w3)[cell] + k] << " " << (*xvec)[(*dofmap)[cell][0][0] + k]
                    << " " <<(*map_w3)[cell]<< " " << (*dofmap)[cell][0][0]<< " " <<
                     (*dofmap)[cell][0][1]<< " " << (*dofmap)[cell][1][1]<< " " << (*dofmap)[cell][2][1]<<
                     " " << (*dofmap)[cell][3][1]<< std::endl;
*/
      (*yvec)[(*map_w3)[cell] + k] = (*op1)[(*map_w3)[cell] + k] * (*xvec)[(*dofmap)[cell][0][0] + k]
                  +(*op5)[(*map_w3)[cell] + k]  * (*xvec)[(*dofmap)[cell][0][1] + k]
                  +(*op4)[(*map_w3)[cell] + k]  * (*xvec)[(*dofmap)[cell][1][1] + k]
                  +(*op3)[(*map_w3)[cell] + k]  * (*xvec)[(*dofmap)[cell][2][1] + k]
                  +(*op2)[(*map_w3)[cell] + k]  * (*xvec)[(*dofmap)[cell][3][1] + k];
  }

  for (int k = 0; k <=nlayers-3; k++)
  {
    (*yvec)[(*map_w3)[cell] + k] = (*yvec)[(*map_w3)[cell] + k] + (*op6)[(*map_w3)[cell] + k] * (*xvec)[(*map_w3)[cell] + k+1]
                            + (*op7)[(*map_w3)[cell] + k] * (*xvec)[(*map_w3)[cell] + k+2];

  }

  int k = nlayers - 2;
  (*yvec)[(*map_w3)[cell] + k] = (*yvec)[(*map_w3)[cell] + k] + (*op6)[(*map_w3)[cell] + k] * (*xvec)[(*map_w3)[cell] + k+1];

  //! Coefficients on layers below
  k = 1;
  (*yvec)[(*map_w3)[cell] + k] = (*yvec)[(*map_w3)[cell] + k] + (*op8)[(*map_w3)[cell] + k] * (*xvec)[(*map_w3)[cell] + k-1];
  for( k = 2; k <= nlayers-1;k++)
  {
    (*yvec)[(*map_w3)[cell] + k] = (*yvec)[(*map_w3)[cell] + k] + (*op8)[(*map_w3)[0] + k] * (*xvec)[(*map_w3)[0] + k-1]
                              + (*op9)[(*map_w3)[0] + k] * (*xvec)[(*map_w3)[0] + k-2];
  }

/*
    ! Use a more efficient method for the global
    do k = 0,nlayers-1
      y(map(1)+k) = Helm_C(map(1)+k)*x(smap(1,1,1)+k) &
                  + Helm_W(map(1)+k)*x(smap(1,2,1)+k) &
                  + Helm_S(map(1)+k)*x(smap(1,2,2)+k) &
                  + Helm_E(map(1)+k)*x(smap(1,2,3)+k) &
                  + Helm_N(map(1)+k)*x(smap(1,2,4)+k)
    end do

  end if

  ! Coefficients on layers above
  do k = 0,nlayers-3
    y(map(1)+k) = y(map(1)+k) + Helm_U(map(1)+k) *x(map(1)+k+1) &
                              + Helm_UU(map(1)+k)*x(map(1)+k+2)
  end do
  k = nlayers - 2
  y(map(1)+k) = y(map(1)+k) + Helm_U(map(1)+k)*x(map(1)+k+1)

  ! Coefficients on layers below
  k = 1
  y(map(1)+k) = y(map(1)+k) + Helm_D(map(1)+k)*x(map(1)+k-1)
  do k = 2,nlayers-1
    y(map(1)+k) = y(map(1)+k) + Helm_D(map(1)+k) *x(map(1)+k-1) &
                              + Helm_DD(map(1)+k)*x(map(1)+k-2)
  end do
*/

}




void read_dinodump(local_int_t& loop0_start, local_int_t& loop0_stop, local_int_t& nlayers, local_int_t& undf_w3, local_int_t& x_vec_max_branch_length,
                   int** map_w3, double** yvec, double** xvec, double** op1, double** op2, double** op3, double** op4, double** op5, double** op6,
                   double** op7, double** op8, double** op9, double** ans, int** stencil_size, int**** dofmap)
{
  // read dinodump.dat
  // remember from fortran
  // temp objects made as vectors for higher ranks, then rearranged at bottom
  const int numbers_per_line = 6;
  const int wide_numbers_per_line = 3;
  bool constants_set = false;
  string line;
  int counter = 0;
  ifstream myfile ("dinodump.dat");
//  double tempvec[];
/*  call steggy%input_array(x_vec_stencil_size, 4, undf_w3 loop0stop)
  call steggy%input_array(x_vec_stencil_dofmap,&
       ndf_w3,x_vec_max_branch_length, 4, loop0)
  call steggy%input_array(map_w3,ndf_w3,loop0)
*/
  size_t pos = 0;
  const string space_delimiter = " ";
  double* temp_stencil;
  double* temp_dofmap;
  int array_counter = 0; // use to add to arrays
  int scalars_stop  = 4;
  int stencil_start;
  int stencil_stop;
  int dofmap_start;  //x_vec_stencil_dofmap , smap
  int dofmap_stop;
  int map_start; // map_w3
  int map_stop;
  int yvec_start;
  int yvec_stop;
  int xvec_start;
  int xvec_stop;
  int op1_start;
  int op1_stop;
  int op2_start;
  int op2_stop;
  int op3_start;
  int op3_stop;
  int op4_start;
  int op4_stop;
  int op5_start;
  int op5_stop;
  int op6_start;
  int op6_stop;
  int op7_start;
  int op7_stop;
  int op8_start;
  int op8_stop;
  int op9_start;
  int op9_stop;
  int ans_start;
  int ans_stop;
  if (myfile.is_open())
  {
    while ( getline (myfile,line) )
    {
      if (!constants_set){
        if (counter == 0) loop0_start = (local_int_t) stoi(line);
        if (counter == 1) loop0_stop  = (local_int_t) stoi(line);
        if (counter == 2) nlayers     = (local_int_t) stoi(line);
        if (counter == 3) undf_w3     = (local_int_t) stoi(line);
        if (counter == 4) {
                x_vec_max_branch_length = (local_int_t) stoi(line);
                constants_set = true;
//              set bounds
                stencil_start = scalars_stop + 1;
                stencil_stop  = stencil_start + 4*loop0_stop/numbers_per_line -1;
                dofmap_start  = stencil_stop + 1;
                dofmap_stop   = dofmap_start + 4*loop0_stop*x_vec_max_branch_length/numbers_per_line -1;
                map_start     = dofmap_stop + 1;
                map_stop      = map_start + loop0_stop/numbers_per_line -1;
                yvec_start    = map_stop + 1;
                yvec_stop     = yvec_start + undf_w3/wide_numbers_per_line -1;
                xvec_start    = yvec_stop + 1;
                xvec_stop     = xvec_start + undf_w3/wide_numbers_per_line -1;
                op1_start     = xvec_stop + 1;
                op1_stop      = op1_start + undf_w3/wide_numbers_per_line -1;
                op2_start     = op1_stop + 1;
                op2_stop      = op2_start + undf_w3/wide_numbers_per_line -1;
                op3_start     = op2_stop + 1;
                op3_stop      = op3_start + undf_w3/wide_numbers_per_line -1;
                op4_start     = op3_stop + 1;
                op4_stop      = op4_start + undf_w3/wide_numbers_per_line -1;
                op5_start     = op4_stop + 1;
                op5_stop      = op5_start + undf_w3/wide_numbers_per_line -1;
                op6_start     = op5_stop + 1;
                op6_stop      = op6_start + undf_w3/wide_numbers_per_line -1;
                op7_start     = op6_stop + 1;
                op7_stop      = op7_start + undf_w3/wide_numbers_per_line -1;
                op8_start     = op7_stop + 1;
                op8_stop      = op8_start + undf_w3/wide_numbers_per_line -1;
                op9_start     = op8_stop + 1;
                op9_stop      = op9_start + undf_w3/wide_numbers_per_line -1;
                ans_start     = op9_stop + 1;
                ans_stop      = ans_start + undf_w3/wide_numbers_per_line -1;

                // allocate memory
                temp_stencil  = (double*) malloc(4*loop0_stop * sizeof(double));
                temp_dofmap   = (double*) malloc(4*loop0_stop*x_vec_max_branch_length * sizeof(double));
                *map_w3       = (int*) malloc(loop0_stop * sizeof(int));
                *yvec         = (double*) malloc(undf_w3 * sizeof(double));
                *xvec         = (double*) malloc(undf_w3 * sizeof(double));
                *op1          = (double*) malloc(undf_w3 * sizeof(double));
                *op2          = (double*) malloc(undf_w3 * sizeof(double));
                *op3          = (double*) malloc(undf_w3 * sizeof(double));
                *op4          = (double*) malloc(undf_w3 * sizeof(double));
                *op5          = (double*) malloc(undf_w3 * sizeof(double));
                *op6          = (double*) malloc(undf_w3 * sizeof(double));
                *op7          = (double*) malloc(undf_w3 * sizeof(double));
                *op8          = (double*) malloc(undf_w3 * sizeof(double));
                *op9          = (double*) malloc(undf_w3 * sizeof(double));
                *ans          = (double*) malloc(undf_w3 * sizeof(double));

                std::cout << "Scalars " << loop0_start << " "<< loop0_stop << " "<< nlayers << " "<< undf_w3 << " "<< x_vec_max_branch_length << std::endl;
                std::cout << "Points in file " << stencil_start << " " << stencil_stop << " " << dofmap_start << " " << dofmap_stop << " "
                << map_start << " " << map_stop << std::endl;
         }
      }
      else{
//     stencil
        if (counter >= stencil_start and counter <= stencil_stop)
        {
          read_part_vector_from_string(line, &temp_stencil, array_counter, space_delimiter);
        }
//     dofmap       (ndf_w3,x_vec_max_branch_length, 4, loop0)
        if (counter == dofmap_start) array_counter = 0;
        if (counter >= dofmap_start and counter <= dofmap_stop)
        {
          read_part_vector_from_string(line, &temp_dofmap, array_counter, space_delimiter);
        }//dofmap */
        if (counter == map_start) array_counter = 0;
        if (counter >= map_start and counter <= map_stop)
        {
          read_part_vector_from_string(line, map_w3, array_counter, space_delimiter);
        }//map_w3
        //yvec
        if (counter == yvec_start) array_counter = 0;
        if (counter >= yvec_start and counter <= yvec_stop)
        {
           read_part_vector_from_string(line, yvec, array_counter, space_delimiter);
        }
        if (counter == xvec_start) array_counter = 0;
        if (counter >= xvec_start and counter <= xvec_stop)
        {
           read_part_vector_from_string(line, xvec, array_counter, space_delimiter);
        }
        if (counter == op1_start) array_counter = 0;
        if (counter >= op1_start and counter <= op1_stop)
        {
           read_part_vector_from_string(line, op1, array_counter, space_delimiter);
        }
        if (counter == op2_start) array_counter = 0;
        if (counter >= op2_start and counter <= op2_stop)
        {
           read_part_vector_from_string(line, op2, array_counter, space_delimiter);
        }
        if (counter == op3_start) array_counter = 0;
        if (counter >= op3_start and counter <= op3_stop)
        {
           read_part_vector_from_string(line, op3, array_counter, space_delimiter);
        }
        if (counter == op4_start) array_counter = 0;
        if (counter >= op4_start and counter <= op4_stop)
        {
           read_part_vector_from_string(line, op4, array_counter, space_delimiter);
        }
        if (counter == op5_start) array_counter = 0;
        if (counter >= op5_start and counter <= op5_stop)
        {
           read_part_vector_from_string(line, op5, array_counter, space_delimiter);
        }
        if (counter == op6_start) array_counter = 0;
        if (counter >= op6_start and counter <= op6_stop)
        {
           read_part_vector_from_string(line, op6, array_counter, space_delimiter);
        }
        if (counter == op7_start) array_counter = 0;
        if (counter >= op7_start and counter <= op7_stop)
        {
           read_part_vector_from_string(line, op7, array_counter, space_delimiter);
        }
        if (counter == op8_start) array_counter = 0;
        if (counter >= op8_start and counter <= op8_stop)
        {
           read_part_vector_from_string(line, op8, array_counter, space_delimiter);
        }
        if (counter == op9_start) array_counter = 0;
        if (counter >= op9_start and counter <= op9_stop)
        {
           read_part_vector_from_string(line, op9, array_counter, space_delimiter);
        }
        if (counter == ans_start) array_counter = 0;
        if (counter >= ans_start and counter <= ans_stop)
        {
           read_part_vector_from_string(line, ans, array_counter, space_delimiter);
        }

      }//constants set
      counter++;
    }//while
    if (counter >= 4){

    // rearrange stencil sizes - strictly 4 * loop0stop but all numbers are 2 and not used so be lazy atm
/* stencil size is only used for limited area
    *stencil_size = (int*) malloc(4*loop0_stop * sizeof(int));
    for (int i=0; i < 4 *loop0_stop; i++)
    {
        (*stencil_size)[i] = (int) temp_stencil[i];
    }
*/
    // lower all values map by one (c vs fort)
    for (int i=0;i<loop0_stop;i++)
    {
        (*map_w3)[i]--;
    }
    // rearrange dofmap (smap) smap[nloop0stop, 4,maxlen, 1]
    // fortran   allocate(x_vec_stencil_dofmap(ndf_w3, x_vec_max_branch_length,4,undf_w3))
//                temp_dofmap   = (double*) malloc(4*loop0_stop*x_vec_max_branch_length * sizeof(double));
    *dofmap = (int***) malloc(loop0_stop * sizeof(int **));
    int temp_counter = 0;
    for (int i=0;i<loop0_stop;i++)
    {
        (*dofmap)[i] = (int**) malloc(4 * sizeof(int*));
        for (int j=0; j < 4; j++)
        {
            (*dofmap)[i][j] = (int*) malloc (x_vec_max_branch_length * sizeof(int));
            for (int k=0; k < x_vec_max_branch_length; k++)
            {
//               (*dofmap)[i][j][k] = temp_dofmap[k*4*loop0_stop + j * 4 + i];
               (*dofmap)[i][j][k] = temp_dofmap[temp_counter] - 1;
//               std::cout << "rearr " << temp_counter << " " << i << " " << j<< " "<< k << " " << temp_dofmap[temp_counter] <<std::endl;
//               << (*dofmap)[i][j][k] << " " << temp_dofmap[temp_counter]<< std::endl;
               temp_counter++;
            }
        }
  //      exit(789);
    }
    free(temp_stencil);
    free(temp_dofmap);
    }//if counter large
    myfile.close();
  }//file open
};
