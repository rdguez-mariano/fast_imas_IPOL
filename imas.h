/**
  * @file imas.h
  * @author Mariano Rodríguez
  * @date 2018
  * @brief The IMAS algorithm implementation.
  */

#ifndef _LIB_IMAS_H
#define _LIB_IMAS_H

#include <vector>
#include "IMAS_coverings.h"
#include "libNumerics/numerics.h"
#include <string>
#include <time.h>

#ifdef _NO_OPENCV
#include "libLocalDesc/sift/demo_lib_sift.h"
#include "libLocalDesc/surf/extract_surf.h"
#include "libLocalDesc/surf/lib_match_surf.h"
#else
//opencv
#include <opencv2/core.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/xfeatures2d.hpp>
#include "opencv2/flann/miniflann.hpp"
#endif

#include <vector>

#define IMAS_SIFT 1
#define IMAS_SURF 2
#define IMAS_ROOTSIFT 11
#define IMAS_SIFT2 12
#define IMAS_HALFSIFT 21

#ifdef _NO_OPENCV
    #define IMAS_HALFROOTSIFT 22

#ifdef _ACD
    #define IMAS_AC 30              // Acontrario matcher
    #define IMAS_AC_W 31            // Acontrario matcher with weights
    #define IMAS_AC_Q 32            // Acontrario matcher quantised
#endif

#ifdef _LDAHASH
    #define IMAS_DIF128   41
    #define IMAS_LDA128   42
    #define IMAS_DIF64    43
    #define IMAS_LDA64    44
#endif

#else
    #define IMAS_BRISK 3
    #define IMAS_BRIEF 4
    #define IMAS_ORB 5
    #define IMAS_DAISY 6
    #define IMAS_AKAZE 7
    #define IMAS_LATCH 8
    #define IMAS_FREAK 9
    #define IMAS_LUCID 10
    #define IMAS_AGAST 13
#endif

// applyfilter equal to the sum of the desired filters to apply
#define ORSA_FUNDAMENTAL 1
#define ORSA_HOMOGRAPHY 2
#define USAC_HOMOGRAPHY 4
#define USAC_FUNDAMENTAL 3



extern int Filter_num_min;
extern double Filter_precision;


typedef libNumerics::matrix<double> TypeMap;

extern std::vector<TypeMap> IdentifiedMaps;

extern int rho;

#ifdef _NO_OPENCV
typedef double IMAS_time;
#else
typedef int64 IMAS_time;
#endif

extern int desc_type;




/**
 * @brief The local descriptor name as in the literature
 */
extern std::string desc_name;

/**
 * @brief Determines which log(radius)-covering to use. Normally set to the maximum tilt tolerance of the matching method.
 */
extern float default_radius;

namespace IMAS
{
#ifdef _NO_OPENCV
class IMAS_Matrix
{
public:
    IMAS_Matrix();
    IMAS_Matrix(std::vector<float> input_data, int width, int height);
    float* data;
    int cols; //= width
    int rows; //= height
    bool empty();
};
#else
typedef cv::Mat IMAS_Matrix;
#endif
IMAS_time IMAS_getTickCount();
double IMAS_getTickFrequency();

struct point_data // Use keypoint from libsift.h
{
    float x;
    float y;
    void* kp_ptr;
    //struct keypoint* kp_ptr;
};

struct skewed_KeyPoint
{
    float size;
    float angle;
    float t;
    float theta;
    float scale;
    point_data pt;
};

struct IMAS_KeyPoint
{
    float x, y, sum_x, sum_y;
    std::vector< skewed_KeyPoint> KPvec;
};

const int NORM_L1 = 1;
const int NORM_L2 = 2;
const int NORM_HAMMING = 3;

}



/**
 * @brief IMAS_keypointlist is only to be used inside local_descriptor.cpp or handled when filtering in compute_IMAS_matches.cpp
 */
typedef std::vector<IMAS::skewed_KeyPoint> IMAS_keypointlist;

/**
 * @brief A simple version of keypoints that will be handle inside IMAS.
 */
struct keypoint_simple {
    float	x,y,
    scale, size,theta,t,
    angle;
};

/**
 * @brief The very definition of a match for IMAS.
 */
typedef std::pair<keypoint_simple,keypoint_simple> matching;

/**
 * @brief A list of matches following the structure of IMAS.
 */
typedef std::vector<matching> matchingslist;

extern std::vector<IMAS::IMAS_KeyPoint*> keys3;


/**
 * @brief It loads the detector and descriptor index in IMAS.
 * @param DDIndex The IMAS Index of the matching method.
 * @return The name of the matching method that has been selected.
 * @code
 cout<< "The " << SetDetectorDescriptor(IMAS_SIFT) << " has been selected" << endl;
 * @endcode
 */
std::string SetDetectorDescriptor(int DDIndex);


void update_matchratio(float matchratio);
void update_edge_threshold(float edge_thres);
void update_tensor_threshold(float tensor_thres);


/**
 * @brief Computes all hyper-keypoints comming from a set of optical tilts digitally generated.
 * @param image Input image.
 * @param width Width of the input image.
 * @param height Height of the input image.
 * @param imasKP Returns a list of generalised keypoints.
 * @param simu_details Specifies the optical tilts that are to be simulated.
 * @param stats A vector with statistics on found generalised keypoints. Mean, min or max of SIIM keypoints over all found generalised keypoints.
 * @return The total number of generalised keypoints that have been found.
 * @author Mariano Rodríguez
 */
int IMAS_detectAndCompute(std::vector<float>& image, int width, int height, std::vector<IMAS::IMAS_KeyPoint *> &imasKP, const std::vector<tilt_simu>& simu_details, std::vector<float> &stats);


/**
 * @brief Performs the Formal IMAS algorithm.
 * @param ipixels1 image1
 * @param w1 Width of image1
 * @param h1 Height of image1
 * @param ipixels2 image2
 * @param w2 Width of image2
 * @param h2 Height of image2
 * @param data Returns a Nx14 matrix representing data for N matches.
 * <table>
  <tr>
    <th>Columns</th>
    <th>Comments</th>
  </tr>
  <tr>
    <td> x_1 </td>
    <td> First coordinate from keypoints on image1 </td>
  </tr>
  <tr>
    <td> y_1 </td>
    <td> Second coordinate from keypoints on image1 </td>
  </tr>
  <tr>
    <td> scale_1 </td>
    <td> scale from keypoints from image1 </td>
  </tr>
  <tr>
    <td> angle_1 </td>
    <td> angle from keypoints from image1 </td>
  </tr>
  <tr>
    <td> t1_1 </td>
    <td> Tilt on image1 in the x-direction from which the keypoints come </td>
  </tr>
  <tr>
    <td> t2_1 </td>
    <td>Tilt on image1 in the y-direction from which the keypoints come  </td>
  </tr>
  <tr>
    <td> theta_1 </td>
    <td> Rotation that was applied before simulating the optical tilt on image1 </td>
  </tr>
  <tr>
    <td> x_2 </td>
    <td> First coordinate from keypoints on image2 </td>
  </tr>
  <tr>
    <td> y_2 </td>
    <td> Second coordinate from keypoints on image2 </td>
  </tr>
  <tr>
    <td> scale_2 </td>
    <td> scale from keypoints from image2 </td>
  </tr>
  <tr>
    <td> angle_2 </td>
    <td> angle from keypoints from image2 </td>
  </tr>
  <tr>
    <td> t1_2 </td>
    <td> Tilt on image2 in the x-direction from which the keypoints come </td>
  </tr>
  <tr>
    <td> t2_2 </td>
    <td> Tilt on image2 in the y-direction from which the keypoints come  </td>
  </tr>
  <tr>
    <td> theta_2 </td>
    <td> Rotation that was applied before simulating the optical tilt on image2 </td>
  </tr>

</table>
 * @param matchings Returns the matches
 * @param Minfoall Returns more info on the matches
 * @param flag_resize Tells the algo if you want to resize the image
 * @param applyfilter Tells which filters should be applied in the function compute_IMAS_matches()
 */
void IMAS_Impl(std::vector<float>& ipixels1, int w1, int h1, std::vector<float>& ipixels2, int w2, int h2, std::vector<float>& data, matchingslist& matchings, imasCoverings &ic, int applyfilter);
#endif // _LIB_IMAS_H
