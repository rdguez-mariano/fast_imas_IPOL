#include <mex_and_omp.h>
#include <stdio.h>
#include <cstdarg>

using namespace std;


void my_mexEvalString(char * msg)
{
    #ifdef _NOMEX
        printf(msg);
    #else
       if (my_omp_get_num_threads()==1)
            mexEvalString(msg);
    #endif
}

void my_Printf(const char* format, ...)
{
    char dest[1024 * 16];
    va_list argptr;
    va_start(argptr, format);
    vsprintf(dest, format, argptr);
    va_end(argptr);

    #ifdef _NOMEX
        printf(dest);
    #else
        if (my_omp_get_num_threads()==1)
        {
            mexPrintf(dest);
            mexEvalString("drawnow");
        }
    #endif
}


void my_mexErrMsgTxt(char * msg)
{
    #ifdef _NOMEX
        printf(msg);
    #else
        if (my_omp_get_num_threads()==1)
            mexErrMsgTxt(msg);
    #endif
}


int my_omp_get_thread_num()
{
    #ifdef _OPENMP
        return (omp_get_thread_num());
    #else
        return (1);
    #endif
}

void my_omp_set_thread_num(int num)
{
    #ifdef _OPENMP
        omp_set_num_threads(num);
    #else
        //Do nothing
    #endif
}

int my_omp_get_max_threads()
{
    #ifdef _OPENMP
        return (omp_get_max_threads());
    #else
        return (1);
    #endif
}

int my_omp_get_num_threads()
{
    #ifdef _OPENMP
        return (omp_get_num_threads());
    #else
        return (1);
    #endif
}
