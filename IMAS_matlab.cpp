#include "math.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vector>
using namespace std;

#ifdef _OPENMP
#include <omp.h>
#endif

 #include "mex.h"
 #include "imas.h"

# define IM_X 800
# define IM_Y 600



// Fonction principale (g�re la liaison avec Matlab)
void mexFunction(int nlhs, mxArray *plhs[], int nrhs,
const mxArray *prhs[])
{
  /* V�rification du nombre d'arguments */
    if ( !(nrhs >= 4) ) {
        my_mexErrMsgTxt("Wrong number of input arguments!");
    } else if ( nlhs != 1) {
        my_mexErrMsgTxt("Wrong number of output arguments!");
    }


  /* matrices d'entr�e*/
  float edge_thres = -1.0f, tensor_thres = -1.0f;
  //size_t w1,h1,w2,h2,w3=-1,h3=-1;

    long w1,h1,w2,h2,w3=-1,h3=-1;  // W = number of rows, H = number of columns
    w1 = mxGetM(prhs[0]);
    h1 = mxGetN(prhs[0]);
    w2 = mxGetM(prhs[1]);
    h2 = mxGetN(prhs[1]);
    vector<float> ipixels1(mxGetPr(prhs[0]), mxGetPr(prhs[0]) + w1*h1);
    vector<float> ipixels2(mxGetPr(prhs[1]), mxGetPr(prhs[1]) + w2*h2);

    int flag_resize = 0;
    int applyfilter = (int) mxGetScalar(prhs[2]); //variants of applying the filter
	  float covering = (float) mxGetScalar(prhs[3]);
    int IMAS_INDEX = (int) mxGetScalar(prhs[4]);
    bool coveringfile = (bool) mxGetScalar(prhs[5]);
	float matchratio = (float) mxGetScalar(prhs[6]);


    string algo_name = SetDetectorDescriptor(IMAS_INDEX);

    if (covering==-1.0f)
        covering = default_radius;

    if (covering!=1.0f)
        algo_name ="Optimal-Affine-"+algo_name;

        imasCoverings ic;
        ic.loadsimulations2do(covering,default_radius,true);

    if (matchratio>0.0f)
        update_matchratio(matchratio);

    if (edge_thres>0.0f)
        update_edge_threshold(edge_thres);

    if (tensor_thres>0.0f)
        update_tensor_threshold(tensor_thres);

        // Number of threads to use
        int nthreads, maxthreads;
        /* Display info on OpenMP*/
    #pragma omp parallel
        {
    #pragma omp master
            {
                nthreads = my_omp_get_num_threads();
                maxthreads = my_omp_get_max_threads();
            }
        }
        my_Printf("--> Using %d threads out of %d for executing %s <--\n\n",nthreads,maxthreads,algo_name.c_str());




        // Performing IMAS
        matchingslist matchings;
        vector< float > data;


        IMAS_Impl(ipixels1, (int)w1, (int)h1, ipixels2, (int)w2, (int)h2, data, matchings,ic, applyfilter);



    int wo = 14;
    plhs[0] = mxCreateDoubleMatrix(wo, matchings.size(), mxREAL);
    double *datamatrix = mxGetPr(plhs[0]);
    for ( int i = 0; i < (int) wo*matchings.size(); i++ )
        datamatrix[i] = (double) data[i];

        data.clear();
        matchings.clear();

}
