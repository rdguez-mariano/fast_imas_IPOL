/**
  * @file mex_and_omp.h
  * @author Mariano Rodríguez
  * @date 2017
  * @brief Handles Mex and OpenMP. So if they're not present, you won't have any compilation error.
  */

#ifndef MEX_AND_OMP_H
#define MEX_AND_OMP_H

#ifndef _NOMEX
#include "mex.h"
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * @brief If MEX is available this will call "mexEvalString(msg)". If not, it won't do anything.
 * @param msg Expression to evaluate.
 */
void my_mexEvalString(char * msg);

/**
 * @brief If MEX is available this will call "my_Printf(...)". If not, it won't do anything.
 * @code
my_Printf("Please print: float %.2f ; int %i; string %s \n",14.82,10,'hello');
 * @endcode
 */
void my_Printf(const char* format, ...);

void my_mexErrMsgTxt(char * msg);

/**
 * @brief If OpenMP is available this will call "omp_get_thread_num()". If not, it will always return 1.
 * @return The unique thread identification number within the current team.
 */
int my_omp_get_thread_num();

/**
 * @brief If OpenMP is available this will call "omp_get_max_threads()". If not, it will always return 1.
 * @return The maximum number of threads you could use.
 */
int my_omp_get_max_threads();

/**
 * @brief If OpenMP is available this will call "omp_get_num_threads()". If not, it will always return 1.
 * @return The number of threads used in a parallel region.
 */
int my_omp_get_num_threads();


void my_omp_set_thread_num(int num);

#endif // MEX_AND_OMP_H
