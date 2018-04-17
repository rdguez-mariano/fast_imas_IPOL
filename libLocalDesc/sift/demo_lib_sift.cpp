// Copyright (c) 2018 Mariano Rodríguez and Rafael Grompone von Gioi
// An original code of SIFT was modified as needed to compute
// some descriptors based on SIFT.

// Authors of this original code of SIFT: Unknown.
// Please, if you are the author of this file, or if you
// know who are the authors of this file, let us know, so we can give the 
// adequate credits and/or get the adequate authorizations.

// WARNING: 
// This file implements an algorithm possibly linked to the patent
//
// David Lowe  "Method and apparatus for identifying scale invariant 
// features in an image and use of same for locating an object in an 
// image",  U.S. Patent 6,711,293.
//
// This file is made available for the exclusive aim of serving as
// scientific tool to verify of the soundness and
// completeness of the algorithm description. Compilation,
// execution and redistribution of this file may violate exclusive
// patents rights in certain countries.
// The situation being different for every country and changing
// over time, it is your responsibility to determine which patent
// rights restrictions apply to you before you compile, use,
// modify, or redistribute this file. A patent lawyer is qualified
// to make this determination.
// If and only if they don't conflict with any patent terms, you
// can benefit from the following license terms attached to this
// file.
//
// This program is provided for scientific and educational only:
// you can use and/or modify it for these purposes, but you are
// not allowed to redistribute this work or derivative works in
// source or executable form. A license must be obtained from the
// patent right holders for any other use.

#define DEBUG 0

#define ABS(x)    (((x) > 0) ? (x) : (-(x)))


#include "libLocalDesc/sift/demo_lib_sift.h"

#include "math.h"

#include <algorithm>    // std::random_shuffle
#include <ctime>        // std::time
#include <cstdlib>      // std::rand, std::srand

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vector>

#include <string>
#include <sstream>
#include <iostream>

#define BIG_NUMBER_L1 2800.0f
#define BIG_NUMBER_L2 1000000000000.0f

std::stringstream tobeprinted;






void default_sift_parameters(siftPar &par)
{
    par.OctaveMax=100000;
    par.DoubleImSize = 0;
    par.order = 3;
    par.InitSigma = 1.6;
    par.BorderDist = 5;
    par.Scales = 3;
    par.PeakThresh = 255.0 * 0.04 / 3.0;
    par.EdgeThresh = 0.06;
    par.EdgeThresh1 = 0.08;
    par.TensorThresh = 0.08;
    par.OriBins  = 36;
    par.OriSigma = 1.5;
    par.OriHistThresh = 0.8;
    par.MaxIndexVal = 0.2;
    par.MagFactor  = 3;
    par.IndexSigma  = 1.0;
    par.IgnoreGradSign = 0;

    par.MatchRatio = 0.6; // 0.73; Guoshen Yu
    par.MatchXradius = 1000000.0f;
    par.MatchYradius = 1000000.0f;

    par.noncorrectlylocalized = 0;
    par.MODE_ROOT=true;
    par.half_sift_trick=false;
    par.L2norm = true; //false = L1 Norm

}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// SIFT Keypoint detection 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void OctaveKeypoints(flimage & image, float octSize, keypointslist& keys,siftPar &par);

void FindMaxMin(  flimage* dogs,  flimage* blur, float octSize ,keypointslist& keys,siftPar &par);

bool LocalMax(float val, flimage& dog, int y0, int x0);

bool LocalMin(float val, flimage& dog, int y0, int x0);

bool LocalMaxMin(float val, const flimage& dog, int y0, int x0);

int NotOnEdge( flimage& dog, int r, int c, float octSize,siftPar &par);

float FitQuadratic(float offset[3],  flimage* dogs, int s, int r, int c);

void InterpKeyPoint(flimage& blur,
                    flimage* dogs, int s, int r, int c,
                    const flimage& grad, const flimage& ori, flimage& map,
                    float octSize, keypointslist& keys, int movesRemain, siftPar &par);

void AssignOriHist(const flimage& blur,
                   const flimage& grad, const flimage& ori, float octSize,
                   float octScale, float octRow, float octCol, keypointslist& keys, siftPar &par);

void SmoothHistogram(
        float* hist, int bins);

float InterpPeak(
        float a, float b, float c);

void MakeKeypoint(const flimage& blur,
                  const flimage& grad, const flimage& ori, float octSize, float octScale,
                  float octRow, float octCol, float angle, keypointslist& keys, siftPar &par);

template <unsigned int OriSize,unsigned int IndexSize>
void MakeKeypointSample(keypoint_base<OriSize, IndexSize> &key, const flimage& grad, const flimage& ori,
                        float scale, float row, float col, siftPar &par);

void NormalizeVec(
        float* vec, int VecLength);

template <unsigned int OriSize,unsigned int IndexSize>
void KeySampleVec(
        keypoint_base<OriSize, IndexSize>& key, const flimage& grad, const flimage& ori,
        float scale, float row, float col,siftPar &par);

template <unsigned int OriSize,unsigned int IndexSize>
void KeySample(
        float index[IndexSize][IndexSize][OriSize], keypoint_base<OriSize, IndexSize>& key,
        const flimage& grad, const flimage& ori,
        float scale, float row, float col,siftPar &par);

template <unsigned int OriSize,unsigned int IndexSize>
void AddSample(
        float index[IndexSize][IndexSize][OriSize], keypoint_base<OriSize, IndexSize>& key,
        const flimage& grad, const flimage& orim,
        int r, int c, float rpos, float cpos, float rx, float cx,siftPar &par);

template <unsigned int OriSize,unsigned int IndexSize>
void PlaceInIndex(
        float index[IndexSize][IndexSize][OriSize],
        float mag, float ori, float rx, float cx,siftPar &par);



// Modified by Mariano Rodríguez to obtain Root-SIFT
/* tobeprinted - just for avoiding calling MATLAB printing function inside a parallel region, which will cast errors */
void compute_sift_keypoints(float *input, keypointslist& keypoints, int width, int height, siftPar &par)
{

    flimage image;

    /// Make zoom of image if necessary
    float octSize = 1.0;
    if (par.DoubleImSize){

        //printf("... compute_sift_keypoints :: applying zoom\n");
        //		image.create(2*width, 2*height);
        //		apply_zoom(input,image.getPlane(),2.0,par.order,width,height);
        //		octSize *= 0.5;

        tobeprinted << "Doulbe image size not allowed. Guoshen Yu\n";
        exit(-1);



    } else
    {

        image.create(width,height,input);
    }

    // tobeprinted << "Using initial Dog value: " << par.PeakThresh << "\n";
    // tobeprinted << "Double image size: "<< par.DoubleImSize << "\n";
    // tobeprinted << "Interpolation order: "<< par.order <<"\n";


    /// Apply initial smoothing to input image to raise its smoothing to par.InitSigma.
    /// We assume image from camera has smoothing of sigma = 0.5, which becomes sigma = 1.0 if image has been doubled.
    /// increase = sqrt(Init^2 - Current^2)
    float	curSigma;
    if (par.DoubleImSize) curSigma = 1.0; else curSigma = 0.5;


    if (par.InitSigma > curSigma ) {

        // tobeprinted << "Convolving initial image to achieve std:" << par.InitSigma <<"\n";

        float sigma = (float) sqrt((double)(par.InitSigma * par.InitSigma - curSigma * curSigma));

        gaussian_convolution( image.getPlane(), image.getPlane(), image.nwidth(), image.nheight(), sigma);

    }

    /// Convolve by par.InitSigma at each step inside OctaveKeypoints by steps of
    /// Subsample of factor 2 while reasonable image size

    /// Keep reducing image by factors of 2 until one dimension is
    /// smaller than minimum size at which a feature could be detected.
    int 	minsize = 2 * par.BorderDist + 2;
    int     OctaveCounter = 0;

    // tobeprinted<<"... compute_sift_keypoints :: maximum number of scales : "<<par.OctaveMax<<"\n";

    while (image.nwidth() > minsize &&  image.nheight() > minsize && OctaveCounter < par.OctaveMax) {

        OctaveKeypoints(image, octSize, keypoints,par);

        // image is blurred inside OctaveKeypoints and therefore can be sampled
        flimage aux( (int)((float) image.nwidth() / 2.0f) , (int)((float) image.nheight() / 2.0f));


        sample(image.getPlane(), aux.getPlane(), 2.0f, image.nwidth(), image.nheight());

        image = aux;

        octSize *= 2.0;

        OctaveCounter++;

    }

    /* Root-SIFT*/
    if ((par.MODE_ROOT || desc_type==IMAS_ROOTSIFT)&&(keypoints.size()>0))
    {
        int VecLength = keypoints[0].veclength;
        for (int i=0; i<(int)keypoints.size();i++)
        {
            float total = 0;
            for (int j=0; j<VecLength;j++)
            {
                total += ABS(keypoints[i].vec[j]);
            }
            for (int j=0; j<VecLength;j++)
            {
                keypoints[i].vec[j] = sqrt( 512*keypoints[i].vec[j]/total );
            }
        }
    }
    tobeprinted << "sift:: "<< keypoints.size() <<" keypoints \n";
    tobeprinted <<"sift::  plus non correctly localized: "<< par.noncorrectlylocalized <<"\n";
}


/////////////////////////////////////////////////
/// EXTREMA DETECTION IN ONE SCALE-SPACE OCTAVE:
/////////////////////////////////////////////////

/// par.Scales determine how many steps we perform to pass from one scale to the next one:  sigI --> 2*sigI
/// At each step we pass from    sigma_0 --> sigma0 * (1 + R)
/// At the last step  sigI * (1 + R)^par.Scales = 2 * sigI
/// (1+R) = 2 ^(1 / par.Scales)  it is called sigmaRatio

/// It seems that blur[par.Scales+1] is compared in two succesive iterations
///
void OctaveKeypoints(flimage & image, float octSize, keypointslist& keys,siftPar &par)
{
    // Guoshen Yu, 2010.09.21, Windows version
    // flimage blur[par.Scales+3], dogs[par.Scales+2];
    int size_blur = par.Scales+3;
    int size_dogs = par.Scales+2;
    flimage *blur = new flimage[size_blur];
    flimage *dogs = new flimage[size_dogs];

    float sigmaRatio = (float) pow(2.0, 1.0 / (double) par.Scales);


    /* Build array, blur, holding par.Scales+3 blurred versions of the image. */
    blur[0] = flimage(image);	/* First level is input to this routine. */
    float prevSigma = par.InitSigma;	/* Input image has par.InitSigma smoothing. */


    /* Form each level by adding incremental blur from previous level.
    Increase in blur is from prevSigma to prevSigma * sigmaRatio, so
    increase^2 = (prevSigma * sigmaRatio)^2 - prevSigma^2
    */
    for (int i = 1; i < par.Scales + 3; i++) {

        //if (DEBUG) printf("Convolving scale: %d \n", i);

        blur[i] = flimage(blur[i-1]);

        float increase = prevSigma*(float)sqrt((double)(sigmaRatio*sigmaRatio-1.0));

        gaussian_convolution( blur[i].getPlane(), blur[i].getPlane(), blur[i].nwidth(), blur[i].nheight(), increase);

        prevSigma *= sigmaRatio;

    }


    /* Compute an array, dogs, of difference-of-Gaussian images by
    subtracting each image from its next blurred version. */
    for (int i = 0; i < par.Scales + 2; i++) {

        dogs[i] = flimage(blur[i]);

        /// dogs[i] = dogs[i] - blur[i+1]
        combine(dogs[i].getPlane(),1.0f, blur[i+1].getPlane(),-1.0f, dogs[i].getPlane(),  dogs[i].nwidth() * dogs[i].nheight());
    }


    // Image with exact blur to be subsampled is blur[scales]
    image = blur[par.Scales];

    /* Scale-space extrema detection in this octave	*/
    //if (DEBUG) printf("Looking for local maxima \n");

    FindMaxMin(dogs, blur, octSize, keys,par);

    // Guoshen Yu, 2010.09.22, Windows version
    delete [] blur;
    delete [] dogs;
}


/////////////////////////////////////////////////
///Find the local maxima and minima of the DOG images in scale space.  Return the keypoints for these locations.
/////////////////////////////////////////////////

/// For each point at each scale we decide if it is a local maxima:
/// - |dogs(x,y,s)| > 0.8 * par.PeakThresh
/// - Local maximum or minimum in s-1,s,s+1
/// - NotonEdge:  ratio of the two principle curvatures of the DOG function at this point be below a threshold.


/// blur[par.Scales+1] is not used in order to look for extrema
/// while these could be computed using avalaible blur and dogs
void FindMaxMin(
        flimage* dogs,  flimage* blur,
        float octSize, keypointslist& keys,siftPar &par)
{

    int width = dogs[0].nwidth(), height = dogs[0].nheight();

    /* Create an image map in which locations that have a keypoint are
    marked with value 1.0, to prevent two keypoints being located at
    same position.  This may seem an inefficient data structure, but
    does not add significant overhead.
    */
    flimage map(width,height,0.0f);
    flimage grad(width,height,0.0f);
    flimage ori(width,height,0.0f);

    /* Search through each scale, leaving 1 scale below and 1 above.
        There are par.Scales+2 dog images.
    */
    for (int s = 1; s < par.Scales+1; s++) {

        //if (DEBUG) printf("************************scale: %d\n", s);

        //getchar();

        /* For each intermediate image, compute gradient and orientation
        images to be used for keypoint description.  */
        compute_gradient_orientation(blur[s].getPlane(), grad.getPlane(), ori.getPlane(), blur[s].nwidth(), blur[s].nheight());


        /* Only find peaks at least par.BorderDist samples from image border, as
        peaks centered close to the border will lack stability. */
        assert(par.BorderDist >= 2);
        float val;
        int partialcounter = 0;
        for (int r = par.BorderDist; r < height - par.BorderDist; r++)
            for (int c = par.BorderDist; c < width - par.BorderDist; c++) {

                /* Pixel value at (c,r) position. */
                val = dogs[s](c,r);

                /* DOG magnitude must be above 0.8 * par.PeakThresh threshold
                (precise threshold check will be done once peak
                interpolation is performed).  Then check whether this
                point is a peak in 3x3 region at each level, and is not
                on an elongated edge.
                */

                if (fabs(val) > 0.8 * par.PeakThresh)
                {

                    /*

                    // If local maxima
                    if (LocalMax(val, dogs[s-1], r, c,par) && LocalMax(val, dogs[s], r, c, par) && LocalMax(val, dogs[s+1], r, c,par) && NotOnEdge(dogs[s], r, c, octSize,par))
                    {
                        if (DEBUG) printf("Maximum Keypoint found (%d,%d,%d)  val: %f\n",s,r,c,val);
                        InterpKeyPoint(
                            dogs, s, r, c, grad, ori,
                            map, octSize, keys, 5,par);

                    } else  if (LocalMin(val, dogs[s-1], r, c,par) && LocalMin(val, dogs[s], r, c,par) && LocalMin(val, dogs[s+1], r, c,par) && NotOnEdge(dogs[s], r, c, octSize,par))
                    {
                        if (DEBUG) printf("Minimum Keypoint found (%d,%d,%d)  val: %f\n",s,r,c,val);
                        InterpKeyPoint(
                            dogs, s, r, c, grad, ori,
                            map, octSize, keys, 5,par);
                    }
*/
                    if (LocalMaxMin(val, dogs[s-1], r, c) && LocalMaxMin(val, dogs[s], r, c) && LocalMaxMin(val, dogs[s+1], r, c) && NotOnEdge(dogs[s], r, c, octSize,par))
                    {
                        partialcounter++;
                        //if (DEBUG) printf("%d:  (%d,%d,%d)  val: %f\n",partialcounter, s,r,c,val);

                        InterpKeyPoint(blur[s],
                                       dogs, s, r, c, grad, ori,
                                       map, octSize, keys, 5,par);

                        //getchar();
                    }


                }

            }
    }

}


//bool LocalMax(float val, flimage& dog, int y0, int x0, siftPar &par)
bool LocalMax(float val, flimage& dog, int y0, int x0)
{
    for (int x = x0 - 1; x <= x0 + 1; x++)
        for (int y = y0  - 1; y <= y0 + 1; y++){
            //printf("%f \t", dog(x,y));
            if (dog(x,y) > val) return 0;
        }

    return 1;
}

bool LocalMin(float val,  flimage& dog, int y0, int x0)
{
    for (int x = x0 - 1; x <= x0 + 1; x++)
        for (int y = y0  - 1; y <= y0 + 1; y++){
            //printf("%f \t", dog(x,y));
            if (dog(x,y) < val) return 0;
        }

    return 1;
}


/* Return TRUE iff val is a local maximum (positive value) or
   minimum (negative value) compared to the 3x3 neighbourhood that
   is centered at (row,col).
*/
bool LocalMaxMin(float val, const flimage& dog, int y0, int x0)
{
    // For efficiency, use separate cases for maxima or minima, and
    // return as soon as possible
    if (val > 0.0) {
        for (int x = x0 - 1; x <= x0 + 1; x++)
            for (int y = y0  - 1; y <= y0 + 1; y++){
                if (dog(x,y) > val) return false;
            }
    } else {
        for (int x = x0 - 1; x <= x0 + 1; x++)
            for (int y = y0  - 1; y <= y0 + 1; y++){
                if (dog(x,y) < val) return false;
            }
    }

    return true;
}



/* Returns FALSE if this point on the DOG function lies on an edge.
   This test is done early because it is very efficient and eliminates
   many points.  It requires that the ratio of the two principle
   curvatures of the DOG function at this point be below a threshold.

   Edge threshold is higher on the first scale where SNR is small in
   order to reduce the number of unstable keypoints.
*/
int NotOnEdge(flimage& dog, int r, int c, float octSize,siftPar &par)
{
    /* Compute 2x2 Hessian values from pixel differences. */
    float	H00 = dog(c,r-1) - 2.0 * dog(c,r) + dog(c,r+1), /* AMIR: div by ? */
            H11 = dog(c-1,r) - 2.0 * dog(c,r) + dog(c+1,r),
            H01 = ( (dog(c+1,r+1) - dog(c-1,r+1)) - (dog(c+1,r-1) - dog(c-1,r-1)) ) / 4.0;

    /* Compute determinant and trace of the Hessian. */
    float	det = H00 * H11 - H01 * H01,	/// Det H = \prod l_i
            trace = H00 + H11;		/// tr H = \sum l_i

    /// As we do not desire edges but only corners we demand l_max / l_min less than a threshold
    /// In practice if A = k B,     A*B = k B^2
    ///				(A + B)^2 = (k+1)^2 * B^2
    ///				k B^2 >  t * (k+1)^2 * B^2 sii   k  / (k+1)^2 > t
    /// This is a decreasing function for k > 1 and value 0.3 at k=1.
    /// Setting t = 0.08, means k<=10

    /* To detect an edge response, we require the ratio of smallest
       to largest principle curvatures of the DOG function
       (eigenvalues of the Hessian) to be below a threshold.  For
       efficiency, we use Harris' idea of requiring the determinant to
       be above par.EdgeThresh times the squared trace, as for eigenvalues
       A and B, det = AB, trace = A+B.  So if A = 10B, then det = 10B**2,
       and trace**2 = (11B)**2 = 121B**2, so par.EdgeThresh = 10/121 =
       0.08 to require ratio of eigenvalues less than 10.
    */
    if (octSize <= 1)
        return (det > par.EdgeThresh1 * trace * trace);
    else
        return (det > par.EdgeThresh * trace * trace);

}


/* Create a keypoint at a peak near scale space location (s,r,c), where
   s is scale (index of DOGs image), and (r,c) is (row, col) location.
   Add to the list of keys with any new keys added.
*/
void InterpKeyPoint( flimage& blur,
                     flimage* dogs, int s, int r, int c,
                     const flimage& grad, const flimage& ori, flimage& map,
                     float octSize, keypointslist& keys, int movesRemain,siftPar &par)
{

    /* Fit quadratic to determine offset and peak value. */
    float offset[3];
    float peakval = FitQuadratic(offset, dogs, s, r, c);
    //if (DEBUG) printf("peakval: %f, of[0]: %f  of[1]: %f  of[2]: %f\n", peakval, offset[0], offset[1], offset[2]);

    /* Move to an adjacent (row,col) location if quadratic interpolation
       is larger than 0.6 units in some direction (we use 0.6 instead of
       0.5 to avoid jumping back and forth near boundary).  We do not
       perform move to adjacent scales, as it is seldom useful and we
       do not have easy access to adjacent scale structures.  The
       movesRemain counter allows only a fixed number of moves to
       prevent possibility of infinite loops.
    */
    int newr = r, newc = c;
    if (offset[1] > 0.6 && r < dogs[0].nheight() - 3)
        newr++;
    else if (offset[1] < -0.6 && r > 3)
        newr--;

    if (offset[2] > 0.6 && c < dogs[0].nwidth() - 3)
        newc++;
    else if (offset[2] < -0.6 && c > 3)
        newc--;

    if (movesRemain > 0  &&  (newr != r || newc != c)) {
        InterpKeyPoint(blur,
                       dogs, s, newr, newc, grad, ori, map,
                       octSize, keys,movesRemain - 1,par);
        return;
    }

    /* Do not create a keypoint if interpolation still remains far
       outside expected limits, or if magnitude of peak value is below
       threshold (i.e., contrast is too low). */
    if (	fabs(offset[0]) > 1.5 || fabs(offset[1]) > 1.5 ||
            fabs(offset[2]) > 1.5 || fabs(peakval) < par.PeakThresh)
    {
        //if (DEBUG) printf("Point not well localized by FitQuadratic\n");
        par.noncorrectlylocalized++;
        return;
    }

    /* Check that no keypoint has been created at this location (to avoid
       duplicates).  Otherwise, mark this map location.
    */
    if (map(c,r) > 0.0) return;
    map(c,r) = 1.0;

    /* The scale relative to this octave is given by octScale.  The scale
       units are in terms of sigma for the smallest of the Gaussians in the
       DOG used to identify that scale.
    */
    // Guoshen Yu, 2010.09.21 Windows version
    // float octScale = par.InitSigma * pow(2.0, (s + offset[0]) / (float) par.Scales);
    float octScale = par.InitSigma * pow(2.0, (s + offset[0]) / (double) par.Scales);

    /// always use histogram of orientations
    //if (UseHistogramOri)
    AssignOriHist(blur,
                  grad, ori, octSize, octScale,
                  r + offset[1], c + offset[2], keys, par);
    //else
    //	AssignOriAvg(
    //		grad, ori, octSize, octScale,
    //		r + offset[1], c + offset[2], keys);
}



/* Apply the method developed by Matthew Brown (see BMVC 02 paper) to
   fit a 3D quadratic function through the DOG function values around
   the location (s,r,c), i.e., (scale,row,col), at which a peak has
   been detected.  Return the interpolated peak position as a vector
   in "offset", which gives offset from position (s,r,c).  The
   returned value is the interpolated DOG magnitude at this peak.
*/
float FitQuadratic(float offset[3], flimage* dogs, int s, int r, int c)
{
    float g[3];
    flimage *dog0, *dog1, *dog2;
    int i;

    //s = 1; r = 128; c = 128;

    float ** H =  allocate_float_matrix(3, 3);

    /* Select the dog images at peak scale, dog1, as well as the scale
       below, dog0, and scale above, dog2. */
    dog0 = &dogs[s-1];
    dog1 = &dogs[s];
    dog2 = &dogs[s+1];

    /* Fill in the values of the gradient from pixel differences. */
    g[0] = ((*dog2)(c,r) - (*dog0)(c,r)) / 2.0;
    g[1] = ((*dog1)(c,r+1) - (*dog1)(c,r-1)) / 2.0;
    g[2] = ((*dog1)(c+1,r) - (*dog1)(c-1,r)) / 2.0;

    /* Fill in the values of the Hessian from pixel differences. */
    H[0][0] = (*dog0)(c,r)   - 2.0 * (*dog1)(c,r) + (*dog2)(c,r);
    H[1][1] = (*dog1)(c,r-1) - 2.0 * (*dog1)(c,r) + (*dog1)(c,r+1);
    H[2][2] = (*dog1)(c-1,r) - 2.0 * (*dog1)(c,r) + (*dog1)(c+1,r);
    H[0][1] = H[1][0] = ( ((*dog2)(c,r+1) - (*dog2)(c,r-1)) -
                          ((*dog0)(c,r+1) - (*dog0)(c,r-1)) ) / 4.0;
    H[0][2] = H[2][0] = ( ((*dog2)(c+1,r) - (*dog2)(c-1,r)) -
                          ((*dog0)(c+1,r) - (*dog0)(c-1,r)) ) / 4.0;
    H[1][2] = H[2][1] = ( ((*dog1)(c+1,r+1) - (*dog1)(c-1,r+1)) -
                          ((*dog1)(c+1,r-1) - (*dog1)(c-1,r-1)) ) / 4.0;

    /* Solve the 3x3 linear sytem, Hx = -g. Result, x, gives peak offset.
       Note that SolveLinearSystem destroys contents of H. */
    offset[0] = - g[0];
    offset[1] = - g[1];
    offset[2] = - g[2];

    // 	for(i=0; i < 3; i++){
    //
    // 		for(j=0; j < 3; j++) printf("%f  ", H[i][j]);
    // 		printf("\n");
    // 	}

    // 	printf("\n");
    //
    // 	for(i=0; i < 3; i++) printf("%f  ", offset[i]);
    // 	printf("\n");

    float solution[3];
    lusolve(H, solution, offset,3);

    // 	printf("\n");
    // 	for(i=0; i < 3; i++) printf("%f  ", solution[i]);
    // 	printf("\n");


    desallocate_float_matrix(H,3,3);
    delete[] H; /*memcheck*/

    /* Also return value of DOG at peak location using initial value plus
       0.5 times linear interpolation with gradient to peak position
       (this is correct for a quadratic approximation).
    */
    for(i=0; i < 3; i++) offset[i] = solution[i];

    return ((*dog1)(c,r) + 0.5 * (solution[0]*g[0]+solution[1]*g[1]+solution[2]*g[2]));
}



/// - Compute histogram of orientation in a neighborhood weighted by gradient and distance to center
/// - Look for local (3-neighborhood) maximum with valuer larger or equal than par.OriHistThresh * maxval


/* Assign an orientation to this keypoint.  This is done by creating a
   Gaussian weighted histogram of the gradient directions in the
   region.  The histogram is smoothed and the largest peak selected.
   The results are in the range of -PI to PI.
*/
void AssignOriHist(const flimage& blur,
                   const flimage& grad, const flimage& ori, float octSize,
                   float octScale, float octRow, float octCol, keypointslist& keys, siftPar &par)
{
    int	bin, prev, next;

    // Guoshen Yu, 2010.09.21 Windows version
    //	float	hist[par.OriBins], distsq, dif, gval, weight, angle, interp;
    float distsq, dif, gval, weight, angle, interp;
    int tmp_size = par.OriBins;
    float *hist = new float[tmp_size];

    float radius2, sigma2;

    int	row = (int) (octRow+0.5),
            col = (int) (octCol+0.5),
            rows = grad.nheight(),
            cols = grad.nwidth();

    for (int i = 0; i < par.OriBins; i++) hist[i] = 0.0;

    /* Look at pixels within 3 sigma around the point and sum their
      Gaussian weighted gradient magnitudes into the histogram. */
    float	sigma = par.OriSigma * octScale;
    int	radius = (int) (sigma * 3.0);
    int rmin = MAX(0,row-radius);
    int cmin = MAX(0,col-radius);
    int rmax = MIN(row+radius,rows-2);
    int cmax = MIN(col+radius,cols-2);
    radius2 = (float)(radius * radius);
    sigma2 = 2.0*sigma*sigma;

    for (int r = rmin; r <= rmax; r++) {
        for (int c = cmin; c <= cmax; c++) {

            gval = grad(c,r);

            dif = (r - octRow);	distsq = dif*dif;
            dif = (c - octCol);	distsq += dif*dif;

            if (gval > 0.0  &&  distsq < radius2 + 0.5) {

                weight = exp(- distsq / sigma2);

                /* Ori is in range of -PI to PI. */
                angle = ori(c,r);
                bin = (int) (par.OriBins * (angle + PI + 0.001) / (2.0 * PI));
                assert(bin >= 0 && bin <= par.OriBins);
                bin = MIN(bin, par.OriBins - 1);
                hist[bin] += weight * gval;

            }

        }
    }


    /* Apply smoothing 6 times for accurate Gaussian approximation. */
    for (int i = 0; i < 6; i++)
        SmoothHistogram(hist, par.OriBins);

    /* Find maximum value in histogram. */
    float maxval = 0.0;
    for (int i = 0; i < par.OriBins; i++)
        if (hist[i] > maxval) maxval = hist[i];

    /* Look for each local peak in histogram.  If value is within
      par.OriHistThresh of maximum value, then generate a keypoint. */
    for (int i = 0; i < par.OriBins; i++) {
        prev = (i == 0 ? par.OriBins - 1 : i - 1);
        next = (i == par.OriBins - 1 ? 0 : i + 1);

        if (	hist[i] > hist[prev]  &&  hist[i] > hist[next]  &&
                hist[i] >= par.OriHistThresh * maxval ) {

            /* Use parabolic fit to interpolate peak location from 3 samples.
              Set angle in range -PI to PI. */
            interp = InterpPeak(hist[prev], hist[i], hist[next]);
            angle = 2.0 * PI * (i + 0.5 + interp) / par.OriBins - PI;
            assert(angle >= -PI  &&  angle <= PI);

            //if (DEBUG) printf("angle selected: %f \t location: (%f,%f)\n", angle, octRow, octCol);
            ;
            /* Create a keypoint with this orientation. */
            MakeKeypoint(blur,
                         grad, ori, octSize, octScale,
                         octRow, octCol, angle, keys,par);
        }

    }

    // Guoshen Yu, 2010.09.22, Windows version
    delete [] hist;
}



/* Smooth a histogram by using a [1/3 1/3 1/3] kernel.  Assume the histogram
   is connected in a circular buffer.
*/
void SmoothHistogram(float* hist, int bins)
{
    float prev, temp;

    prev = hist[bins - 1];
    for (int i = 0; i < bins; i++) {
        temp = hist[i];
        hist[i] = ( prev + hist[i] + hist[(i + 1 == bins) ? 0 : i + 1] ) / 3.0;
        prev = temp;
    }
}


/* Return a number in the range [-0.5, 0.5] that represents the
   location of the peak of a parabola passing through the 3 evenly
   spaced samples.  The center value is assumed to be greater than or
   equal to the other values if positive, or less than if negative.
*/
float InterpPeak(float a, float b, float c)
{
    if (b < 0.0) {
        a = -a; b = -b; c = -c;
    }
    assert(b >= a  &&  b >= c);
    return 0.5 * (a - c) / (a - 2.0 * b + c);
}

/* Joan Pau: Add a new keypoint to a vector of keypoints
   Create a new keypoint and return list of keypoints with new one added.

   Modified by Mariano Rodríguez to obtain HALF-SIFT
 */
void MakeKeypoint(const flimage& blur,
                  const flimage& grad, const flimage& ori, float octSize, float octScale,
                  float octRow, float octCol, float angle, keypointslist& keys,siftPar &par)
{
    if (par.half_sift_trick || desc_type==IMAS_HALFROOTSIFT || desc_type ==IMAS_HALFSIFT)
    {
        if (true) // Fast but looses some matches
        {
            /*
             * two temporal keypoints are created in the <angle> and oposite <angle> directions.
             * Then the energy of the first <IndexSize1> histograms is used to determine which direction
             * should be taken into account to create a unique KEYPOINT.
            */
            keypoint_base<2*OriSize1,IndexSize1> newkeypoint_double;
            newkeypoint_double.x = octSize * octCol;	/*x coordinate */
            newkeypoint_double.y = octSize * octRow;	/*y coordinate */
            newkeypoint_double.scale = octSize * octScale;	/* scale */
            newkeypoint_double.angle = angle;//fmod(angle,M_PI/2);		/* orientation */
            MakeKeypointSample(newkeypoint_double,grad,ori,octScale,octRow,octCol,par);

            keypoint_base<2*OriSize1,IndexSize1> newkeypoint_double2;
            newkeypoint_double2.x = octSize * octCol;	/*x coordinate */
            newkeypoint_double2.y = octSize * octRow;	/*y coordinate */
            newkeypoint_double2.scale = octSize * octScale;	/* scale */
            if (angle<0)
                newkeypoint_double2.angle = angle + PI; /* orientation */
            else
                newkeypoint_double2.angle = angle - PI;
            MakeKeypointSample(newkeypoint_double2,grad,ori,octScale,octRow,octCol,par);

            float energy=0.0f, energy2 = 0.0f;
            for (unsigned int k = 0; k < 2*OriSize1*IndexSize1; k++)
            {
                energy += newkeypoint_double.vec[k]*newkeypoint_double.vec[k];
                energy2 += newkeypoint_double2.vec[k]*newkeypoint_double2.vec[k];
            }

            keypoint newkeypoint;
            newkeypoint.x = octSize * octCol;	/*x coordinate */
            newkeypoint.y = octSize * octRow;	/*y coordinate */
            newkeypoint.scale = octSize * octScale;	/* scale */
            if (energy>energy2)
            {
                newkeypoint.radius = newkeypoint_double.radius;
                newkeypoint.angle = newkeypoint_double.angle;
                int ind1 = 0, ind2 = 0;
                for (unsigned int i = 0; i < IndexSize1; i++)
                    for (unsigned int j = 0; j < IndexSize1; j++)
                    {
                        ind1 = i*IndexSize1*OriSize1 + j*OriSize1;
                        ind2 = i*IndexSize1*2*OriSize1 + j*2*OriSize1;
                        for (unsigned int k = 0; k < OriSize1; k++)
                            newkeypoint.vec[ind1+k] = (newkeypoint_double.vec[ind2+k] + newkeypoint_double.vec[ind2+k+OriSize1])/2;
                    }
            }
            else
            {
                newkeypoint.radius = newkeypoint_double2.radius;
                newkeypoint.angle = newkeypoint_double2.angle;
                int ind1 = 0, ind2 = 0;
                for (unsigned int i = 0; i < IndexSize1; i++)
                    for (unsigned int j = 0; j < IndexSize1; j++)
                    {
                        ind1 = i*IndexSize1*OriSize1 + j*OriSize1;
                        ind2 = i*IndexSize1*2*OriSize1 + j*2*OriSize1;
                        for (unsigned int k = 0; k < OriSize1; k++)
                            newkeypoint.vec[ind1+k] = (newkeypoint_double2.vec[ind2+k] + newkeypoint_double2.vec[ind2+k+OriSize1])/2;
                    }
            }

            if (newkeypoint.radius>0.0f)
                keys.push_back(newkeypoint);
        }
        else // All possible matches but too slow
        {
            /*
             * Creates two keypoints in the <angle> and oposite <angle> directions.
            */
            keypoint_base<2*OriSize1,IndexSize1> newkeypoint_double;
            newkeypoint_double.x = octSize * octCol;	/*x coordinate */
            newkeypoint_double.y = octSize * octRow;	/*y coordinate */
            newkeypoint_double.scale = octSize * octScale;	/* scale */
            newkeypoint_double.angle = angle;//fmod(angle,M_PI/2);		/* orientation */
            MakeKeypointSample(newkeypoint_double,grad,ori,octScale,octRow,octCol,par);

            keypoint newkeypoint;
            newkeypoint.x = octSize * octCol;	/*x coordinate */
            newkeypoint.y = octSize * octRow;	/*y coordinate */
            newkeypoint.scale = octSize * octScale;	/* scale */
            newkeypoint.angle = angle;//fmod(angle,M_PI/2);		/* fmod(angle,M_PI) */
            newkeypoint.radius = newkeypoint_double.radius;

            int ind1 = 0, ind2 = 0;
            for (unsigned int i = 0; i < IndexSize1; i++)
                for (unsigned int j = 0; j < IndexSize1; j++)
                {
                    ind1 = i*IndexSize1*OriSize1 + j*OriSize1;
                    ind2 = i*IndexSize1*2*OriSize1 + j*2*OriSize1;
                    for (unsigned int k = 0; k < OriSize1; k++)
                        newkeypoint.vec[ind1+k] = (newkeypoint_double.vec[ind2+k] + newkeypoint_double.vec[ind2+k+OriSize1])/2;
                }
            keys.push_back(newkeypoint);


            newkeypoint_double.x = octSize * octCol;	/*x coordinate */
            newkeypoint_double.y = octSize * octRow;	/*y coordinate */
            newkeypoint_double.scale = octSize * octScale;	/* scale */
            if (angle<0)
                newkeypoint_double.angle = angle + PI; /* orientation */
            else
                newkeypoint_double.angle = angle - PI;
            MakeKeypointSample(newkeypoint_double,grad,ori,octScale,octRow,octCol,par);

            //keypoint newkeypoint2;
            newkeypoint.x = octSize * octCol;	/*x coordinate */
            newkeypoint.y = octSize * octRow;	/*y coordinate */
            newkeypoint.scale = octSize * octScale;	/* scale */
            newkeypoint.radius = newkeypoint_double.radius;
            if (angle<0)
                newkeypoint.angle = angle + PI;//fmod(angle,M_PI/2);		/* fmod(angle,M_PI) */
            else
                newkeypoint.angle = angle - PI;//fmod(angle,M_PI/2);		/* fmod(angle,M_PI) */

            for (unsigned int i = 0; i < IndexSize1; i++)
                for (unsigned int j = 0; j < IndexSize1; j++)
                {
                    ind1 = i*IndexSize1*OriSize1 + j*OriSize1;
                    ind2 = i*IndexSize1*2*OriSize1 + j*2*OriSize1;
                    for (unsigned int k = 0; k < OriSize1; k++)
                        newkeypoint.vec[ind1+k] = (newkeypoint_double.vec[ind2+k] + newkeypoint_double.vec[ind2+k+OriSize1])/2;
                }
            keys.push_back(newkeypoint);
        }
    }
    else
    {
        keypoint newkeypoint;
        newkeypoint.x = octSize * octCol;	/*x coordinate */
        newkeypoint.y = octSize * octRow;	/*y coordinate */
        newkeypoint.scale = octSize * octScale;	/* scale */
        newkeypoint.angle = angle;		/* orientation */
        MakeKeypointSample(newkeypoint,grad,ori,octScale,octRow,octCol,par);
        if (newkeypoint.radius>0.0f)
            keys.push_back(newkeypoint);
    }
}



/* Use the parameters of this keypoint to sample the gradient images
     at a set of locations within a circular region around the keypoint.
     The (scale,row,col) values are relative to current octave sampling.
     The resulting vector is stored in the key.
*/
template <unsigned int OriSize,unsigned int IndexSize>
void MakeKeypointSample(
        keypoint_base<OriSize,IndexSize>& key, const flimage& grad, const flimage& ori,
        float scale, float row, float col,siftPar &par)
{
    const int VecLength = IndexSize * IndexSize * OriSize;
    /* Produce sample vector. */
    KeySampleVec(key, grad, ori, scale, row, col,par);


    /* Normalize vector.  This should provide illumination invariance
    for planar lambertian surfaces (except for saturation effects).
    Normalization also improves nearest-neighbor metric by
    increasing relative distance for vectors with few features.
    It is also useful to implement a distance threshold and to
    allow conversion to integer format.
    */
    NormalizeVec(key.vec,VecLength);

    /* Now that normalization has been done, threshold elements of
    index vector to decrease emphasis on large gradient magnitudes.
    Admittedly, the large magnitude values will have affected the
    normalization, and therefore the threshold, so this is of
    limited value.
    */
    bool changed = false;
    for (int i = 0; i < VecLength; i++)
        if (key.vec[i] > par.MaxIndexVal) {
            key.vec[i] = par.MaxIndexVal;
            changed = true;
        }

    if (changed) NormalizeVec(key.vec,VecLength);

    /* Convert float vector to integer. Assume largest value in normalized
    vector is likely to be less than 0.5. */
    /// QUESTION: why is the vector quantized to integer
    int intval;
    for (int i = 0; i < VecLength; i++) {
        intval =  (int)(512.0 * key.vec[i]);
        key.vec[i] = (int) MIN(255, intval);
    }
}

/* Normalize length of vec to 1.0.
*/
void NormalizeVec(float* vec, int VecLength)
{
    float val, fac;

    float sqlen = 0.0;
    for (int i = 0; i < VecLength; i++) {
        val = vec[i];
        sqlen += val * val;
    }
    fac = 1.0 / sqrt(sqlen);

    for (int i = 0; i < VecLength; i++)
        vec[i] *= fac;
}


/* Create a 3D index array into which gradient values are accumulated.
   After filling array, copy values back into vec.
*/
template <unsigned int OriSize,unsigned int IndexSize>
void KeySampleVec(
        keypoint_base<OriSize,IndexSize>& key, const flimage& grad, const flimage& ori,
        float scale, float row, float col,siftPar &par)
{

    float index[IndexSize][IndexSize][OriSize];

    /* Initialize index array. */
    for (unsigned int i = 0; i < IndexSize; i++)
        for (unsigned int j = 0; j < IndexSize; j++)
            for (unsigned int k = 0; k < OriSize; k++)
                index[i][j][k] = 0.0;


    KeySample(index, key, grad, ori, scale, row, col, par);


    /* Unwrap the 3D index values into 1D vec. */
    int v = 0;
    for (unsigned int i = 0; i < IndexSize; i++)
        for (unsigned int j = 0; j < IndexSize; j++)
            for (unsigned int k = 0; k < OriSize; k++)
                key.vec[v++] = index[i][j][k];
}



/* Add features to vec obtained from sampling the grad and ori images
   for a particular scale.  Location of key is (scale,row,col) with respect
   to images at this scale.  We examine each pixel within a circular
   region containing the keypoint, and distribute the gradient for that
   pixel into the appropriate bins of the index array.
*/
template <unsigned int OriSize,unsigned int IndexSize>
void KeySample(
        float index[IndexSize][IndexSize][OriSize], keypoint_base<OriSize,IndexSize>& key,
        const flimage& grad, const flimage& ori, float scale, float row, float col,siftPar &par)
{
    float rpos, cpos, rx, cx;

    int	irow = (int) (row + 0.5),
            icol = (int) (col + 0.5);
    float	sine   = (float) sin(key.angle),
            cosine = (float) cos(key.angle);

    /* The spacing of index samples in terms of pixels at this scale. */
    float	spacing = scale * par.MagFactor;

    /* Radius of index sample region must extend to diagonal corner of
    index patch plus half sample for interpolation. */
    float	radius = 1.414 * spacing * (IndexSize + 1) / 2.0;
    int	iradius = (int) (radius + 0.5);

    // Initialising structure tensor components // Mariano Rodríguez
    float ts_xx = 0.0f, ts_yy = 0.0f, ts_xy = 0.0f;

    //key.radius = (int) (par.OriSigma * scale * par.MagFactor);
    key.radius = (float) iradius; // Mariano Rodríguez

    /* Examine all points from the gradient image that could lie within the
    index square. */
    for (int i = -iradius; i <= iradius; i++) {
        for (int j = -iradius; j <= iradius; j++) {

            /* Rotate sample offset to make it relative to key orientation.
             Uses (row,col) instead of (x,y) coords.  Also, make subpixel
             correction as later image offset must be an integer.  Divide
             by spacing to put in index units.
            */

            /* Guoshen Yu, inverse the rotation */
            rpos = ((cosine * i - sine * j) - (row - irow)) / spacing;
            cpos = ((sine * i + cosine * j) - (col - icol)) / spacing;

            /*
             rpos = ((cosine * i + sine * j) - (row - irow)) / spacing;
             cpos = ((- sine * i + cosine * j) - (col - icol)) / spacing;*/

            /* Compute location of sample in terms of real-valued index array
             coordinates.  Subtract 0.5 so that rx of 1.0 means to put full
             weight on index[1] (e.g., when rpos is 0 and IndexSize is 3. */
            rx = rpos + IndexSize / 2.0 - 0.5;
            cx = cpos + IndexSize / 2.0 - 0.5;

            /* Test whether this sample falls within boundary of index patch. */
            if (	rx > -1.0 && rx < (float) IndexSize  &&
                    cx > -1.0 && cx < (float) IndexSize )
            {
                int r = (int)(irow + i), c = (int)(icol + j);

                if (r >= 0  &&  r < grad.nheight()  &&  c >= 0  &&  c < grad.nwidth())
                {
                    AddSample(index, key, grad, ori, r, c, rpos, cpos, rx, cx,par);

                    // Computing structure tensor // Mariano Rodríguez
                    float mag = grad(c,r);
                    float theta = ori(c,r);
                    float dx = mag*cos(theta), dy = mag*sin(theta);
                    ts_xx += pow(dx,2);
                    ts_yy += pow(dy,2);
                    ts_xy += dx*dy;
                }

            }
        }
    }
    // eigen values of the structure tensor // Mariano Rodríguez
    float	det = ts_xx * ts_yy - ts_xy * ts_xy,	/// Det H = \prod l_i
            trace = ts_xx + ts_yy;		/// tr H = \sum l_i

    /// As we do not desire edges but only corners we demand l_max / l_min less than a threshold
    /// In practice if A = k B,     A*B = k B^2
    ///				(A + B)^2 = (k+1)^2 * B^2
    ///				k B^2 >  t * (k+1)^2 * B^2 sii   k  / (k+1)^2 > t
    /// This is a decreasing function for k > 1 and value 0.3 at k=1.
    ///       f(k) = k  / (k+1)^2
    /// Setting t = 0.08, means k<=10
    //std::cout<<trace<<std::endl;
    //trace < 0.5*128*128
    if ((det < par.TensorThresh * trace * trace))
        //if ( (det - 0.05 *(trace*trace)) > par.TensorThresh) //harris condition
        key.radius = -1.0f;
}


/* Given a sample from the image gradient, place it in the index array.
*/
template <unsigned int OriSize,unsigned int IndexSize>
void AddSample(
        float index[IndexSize][IndexSize][OriSize], keypoint_base<OriSize,IndexSize>& key,
        const flimage& grad, const flimage& orim,
        int r, int c, float rpos, float cpos, float rx, float cx,siftPar &par)
{

    /* Compute Gaussian weight for sample, as function of radial distance
       from center.  Sigma is relative to half-width of index. */
    float	sigma  = par.IndexSigma * 0.5 * IndexSize,
            weight = exp(- (rpos * rpos + cpos * cpos) / (2.0 * sigma * sigma)),
            mag    = weight *  grad(c,r);
    //		mag    = weight *  grad((int)c,(int)r); // Guoshen Yu, explicitely cast to int to avoid warning


    /* Subtract keypoint orientation to give ori relative to keypoint. */
    float	ori = orim(c,r) -  key.angle;
    //	float	ori = orim((int)c,(int)r) -  key.angle; // Guoshen Yu, explicitely cast to int to avoid warning


    /* Put orientation in range [0, 2*PI].  If sign of gradient is to
       be ignored, then put in range [0, PI]. */
    if (par.IgnoreGradSign) {
        while (ori > PI ) ori -= PI;
        while (ori < 0.0) ori += PI;
    } else {
        while (ori > 2.0*PI) ori -= 2.0*PI;
        while (ori < 0.0   ) ori += 2.0*PI;
    }
    PlaceInIndex(index, mag, ori, rx, cx,par);
}


/* Increment the appropriate locations in the index to incorporate
   this image sample.  The location of the sample in the index is (rx,cx).
*/
template <unsigned int OriSize,unsigned int IndexSize>
void PlaceInIndex(
        float index[IndexSize][IndexSize][OriSize],
        float mag, float ori, float rx, float cx,siftPar &par)
{
    int	orr, rindex, cindex, oindex;
    float	rweight, cweight, oweight;
    float  *ivec;

    float	oval = OriSize * ori / (par.IgnoreGradSign ? PI : 2.0*PI);

    //	int	ri = (rx >= 0.0) ? rx : rx - 1.0,	/* Round down to next integer. */
    //		ci = (cx >= 0.0) ? cx : cx - 1.0,
    //		oi = (oval >= 0.0) ? oval : oval - 1.0;

    int	ri = (int)((rx >= 0.0) ? rx : rx - 1.0),	/* Round down to next integer. */ // Guoshen Yu, explicitely cast to int to avoid warning
            ci = (int)((cx >= 0.0) ? cx : cx - 1.0), // Guoshen Yu, explicitely cast to int to avoid warning
            oi = (int)((oval >= 0.0) ? oval : oval - 1.0); // Guoshen Yu, explicitely cast to int to avoid warning

    float	rfrac = rx - ri,			/* Fractional part of location. */
            cfrac = cx - ci,
            ofrac = oval - oi;
    assert(
                ri >= -1  &&  ri < (int)IndexSize  &&
                oi >=  0  &&  oi <= (int)OriSize  &&
                rfrac >= 0.0  &&  rfrac <= 1.0);

    /* Put appropriate fraction in each of 8 buckets around this point
        in the (row,col,ori) dimensions.  This loop is written for
        efficiency, as it is the inner loop of key sampling. */
    for (int r = 0; r < 2; r++) {
        rindex = ri + r;
        if (rindex >=0 && rindex < (int) IndexSize) {
            rweight = mag * ((r == 0) ? 1.0 - rfrac : rfrac);

            for (int c = 0; c < 2; c++) {
                cindex = ci + c;
                if (cindex >=0 && cindex < (int) IndexSize) {
                    cweight = rweight * ((c == 0) ? 1.0 - cfrac : cfrac);
                    ivec = index[rindex][cindex];
                    for (orr = 0; orr < 2; orr++) {
                        oindex = oi + orr;
                        if (oindex >= (int) OriSize)  /* Orientation wraps around at PI. */
                            oindex = 0;
                        oweight = cweight * ((orr == 0) ? 1.0 - ofrac : ofrac);
                        ivec[oindex] += oweight;
                    }
                }
            }
        }
    }
}



// generate a vector of random indices
std::vector<int> random_inds(int amount)
{
    std::srand ( unsigned ( std::time(0) ) );
    std::vector<int> myvec;
    for (int i=0; i< (int) amount; i++)
    {
        myvec.push_back(i);
    }

    std::random_shuffle ( myvec.begin(), myvec.end() );
    return myvec;

}
