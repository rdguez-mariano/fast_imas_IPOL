#include "imas.h"
#include <math.h>
#include <algorithm>
#include <ctime>
#include <cstdlib>

#include "mex_and_omp.h"

#include "libNumerics/numerics.h"
#include "libSimuTilts/digital_tilt.h"

#include "libSimuTilts/frot.h"
#include "libSimuTilts/fproj.h"


#ifdef _NO_OPENCV
#include "libLocalDesc/sift/demo_lib_sift.h"
#endif

#include "libOrsa/orsa_fundamental.hpp"
#include "libOrsa/orsa_homography.hpp"




#ifdef _USAC
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <unistd.h>
#include "ConfigParams.h"
#include "libUSAC/estimators/FundmatrixEstimator.h"
#include "libUSAC/estimators/HomogEstimator.h"
#include "libUSAC/estimators/FundmatrixEstimator.h"
#endif


#ifdef _LDAHASH
#include "libLocalDesc/ldahash/lib_ldahash.h"
#endif


#define BIG_NUMBER_L1 2800.0f
#define BIG_NUMBER_L2 1000000000000.0f


using namespace std;


/**
 * @brief The minimum number of matches for ORSA (depending on the type: Homography or Fundamental) to be applied.
 */
int Filter_num_min = 5;

/**
 * @brief The precision in pixels in which ORSA is to find the underlying meaningful transformation.
 * The meaning of this precision changes with ORSA Fundamental or ORSA Homography
 */
double Filter_precision=24;

/**
 * @brief IdentifiedMaps Stores meaningful transformations identified by ORSA
 */
std::vector<TypeMap> IdentifiedMaps;

/**
 * @brief Stores generalised keypoints (possible from a third image) to be used as a backgroud a-contrario model.
 */
std::vector<IMAS::IMAS_KeyPoint*> keys3;

/**
 * @brief Fixes the number of generalised keypoints in the third image to be used.
 * If this number is less than the size of <keys3> then a random part of it is selected as a-contrario model.
 * If set to -1 it considers all elements in <keys3>.
 */
int fixed_number_of_keypoints = -1;
/**
 * @brief SIIM keypoints must lie within a radius of <rho> in order to be considered as the same generalised keypoint.
 */
int rho = 4;




float _arearatio;


//these are global variables
std::string desc_name="SIFT"; // descriptor being used
float default_radius = 1.8f; // the default radius for the covering


int desc_type=1;   // index for the descriptor being used
float nndrRatio = 0.8f; // Find correspondences by NNDR (Nearest Neighbor Distance Ratio)
int normType = -1; // Use default norm proposed by our program
bool binary_desc = false; // kind of descriptor is being used : binary or float
bool rooted = true; // use root versions of descriptors ex. ROOTSIFT
bool sift_desc = true;

#ifndef _OPENMP
#include <time.h>
#endif


#ifdef _NO_OPENCV
siftPar siftparameters;
#endif


IMAS_time IMAS::IMAS_getTickCount()
{
#ifndef _NO_OPENCV
    return(cv::getTickCount());
#else
#ifdef _OPENMP
    return(omp_get_wtime());
#else
    return(time(0));
#endif
#endif

}
double IMAS::IMAS_getTickFrequency()
{
#ifndef _NO_OPENCV
    return(cv::getTickFrequency());
#else
    return(1.0);
#endif
}

#ifdef _NO_OPENCV
bool IMAS::IMAS_Matrix::empty()
{
    return((this->cols+this->rows)==0);
}

IMAS::IMAS_Matrix::IMAS_Matrix()
{
    cols = 0;
    rows = 0;
}

IMAS::IMAS_Matrix::IMAS_Matrix(std::vector<float> input_data, int width, int height)
{
    cols = width;
    rows = height;
    data = new float[cols*rows];
    for (int cc = 0; cc < cols*rows; cc++)
        data[cc] =input_data[cc];
}

void update_tensor_threshold(float tensor_thres)
{
    siftparameters.TensorThresh = tensor_thres;
}

void update_edge_threshold(float edge_thres)
{
    siftparameters.EdgeThresh = edge_thres;
    siftparameters.EdgeThresh1 = edge_thres;
}
#endif

void update_matchratio(float matchratio)
{
    //siftparameters.MatchRatio = matchratio;
    nndrRatio = matchratio;
}


void updateparams()
{
#ifdef _NO_OPENCV
    default_sift_parameters(siftparameters);
    //siftparameters.MatchRatio = nndrRatio;
    siftparameters.L2norm = (normType==IMAS::NORM_L2);
    siftparameters.MODE_ROOT = rooted;
    if (desc_type==IMAS_HALFSIFT)
    {
        siftparameters.half_sift_trick = true;
    }
#else
    // .... Put params to update here
#endif
}


std::string SetDetectorDescriptor(int DDIndex)
{
    switch (DDIndex)
    {
    case IMAS_SIFT:
    {
        desc_name="SIFT L2";
        nndrRatio = 0.8f;
        desc_type = IMAS_SIFT;
        binary_desc = false;
        default_radius = 1.7f;
        rooted = false;
        sift_desc = true;

#ifndef _NO_OPENCV
        normType = cv::NORM_L2;
        desc_name=desc_name + " (opencv)";
#else
        normType = IMAS::NORM_L2;
#endif
        break;
    }
    case IMAS_SIFT2:
    {
        desc_name="SIFT L1";
        nndrRatio = 0.73f;
        desc_type = IMAS_SIFT;
        binary_desc = false;
        default_radius = 1.7f;
        rooted = false;
        sift_desc = true;

#ifndef _NO_OPENCV
        normType = cv::NORM_L1;
        desc_name=desc_name + "_opencv";
#else
        normType = IMAS::NORM_L1;
#endif
        break;
    }
    case IMAS_HALFSIFT:
    {
        desc_name="HalfSIFT L1";
        nndrRatio = 0.8f;
        desc_type = IMAS_HALFSIFT;
        binary_desc = false;
        default_radius = 1.7f;
        sift_desc = true;
        rooted = false;

#ifndef _NO_OPENCV
        normType = cv::NORM_L1;
        desc_name=desc_name + "_opencv";
#else
        normType = IMAS::NORM_L1;
#endif
        break;

    }
    case IMAS_ROOTSIFT:
    {
        desc_name="RootSIFT";
        nndrRatio = 0.8f;
        desc_type = IMAS_ROOTSIFT;
        binary_desc = false;
        default_radius = 1.7f;
        rooted = true;
        sift_desc = true;

#ifndef _NO_OPENCV
        normType = cv::NORM_L2;
        desc_name=desc_name + " (opencv)";
#else
        normType = IMAS::NORM_L2;
#endif
        break;
    }
    case IMAS_SURF:
    {
        desc_name="SURF";
        nndrRatio = 0.6f;
        desc_type = IMAS_SURF;
        binary_desc = false;
        default_radius = 1.4f;
        sift_desc = false;

#ifndef _NO_OPENCV
        normType = cv::NORM_L1;
        desc_name=desc_name + " (opencv)";
#else
        normType = IMAS::NORM_L1;
#endif
        break;
    }

#ifndef _NO_OPENCV
    case IMAS_BRISK:
    {
        desc_name="BRISK";
        nndrRatio = 0.8f;
        desc_type = IMAS_BRISK;
        binary_desc = true;
        normType = cv::NORM_HAMMING;
        default_radius = 1.7f;
        sift_desc = false;
        break;
    }
    case IMAS_FREAK:
    {
        desc_name="FREAK";
        nndrRatio = 0.8f;
        desc_type = IMAS_FREAK;
        binary_desc = true;
        normType = cv::NORM_HAMMING;
        default_radius = 1.6f;
        sift_desc = false;
        break;
    }
    case IMAS_ORB:
    {
        desc_name="ORB";
        nndrRatio = 0.8f;
        desc_type = IMAS_ORB;
        binary_desc = true;
        normType = cv::NORM_HAMMING;
        default_radius = 1.4f;
        sift_desc = false;
        break;
    }
    case IMAS_BRIEF:
    {
        desc_name="BRIEF";
        nndrRatio = 0.8f;
        desc_type = IMAS_BRIEF;
        binary_desc = true;
        normType = cv::NORM_HAMMING;
        default_radius = 1.4f;
        sift_desc = false;
        break;
    }
    case IMAS_AGAST:
    {
        desc_name="AGAST";
        nndrRatio = 0.8f;
        desc_type = IMAS_AGAST;
        binary_desc = true;
        normType = cv::NORM_HAMMING;
        sift_desc = false;
        break;
    }
    case IMAS_LATCH:
    {
        desc_name="LATCH";
        nndrRatio = 0.8f;
        desc_type = IMAS_LATCH;
        binary_desc = true;
        normType = cv::NORM_HAMMING;
        default_radius = 1.55f;
        sift_desc = false;
        break;
    }
    case IMAS_LUCID:
    {
        desc_name="LUCID";
        nndrRatio = 0.8f;
        desc_type = IMAS_LUCID;
        binary_desc = true;
        normType = cv::NORM_HAMMING;
        sift_desc = false;
        break;
    }
    case IMAS_DAISY:
    {
        desc_name="DAISY";
        nndrRatio = 0.8f;
        desc_type = IMAS_DAISY;
        binary_desc = false;
        normType = cv::NORM_L1;
        default_radius = 1.55f;
        sift_desc = false;
        break;
    }
    case IMAS_AKAZE:
    {
        desc_name="AKAZE";
        nndrRatio = 0.8f;
        desc_type = IMAS_AKAZE;
        binary_desc = true;
        normType = cv::NORM_HAMMING;
        default_radius = 1.7f;
        sift_desc = false;
        break;
    }

#else
        //// Stand alone
    case IMAS_HALFROOTSIFT:
    {
        desc_name="HalfRootSIFT";
        normType = IMAS::NORM_L2;
        nndrRatio = 0.8f;
        desc_type = IMAS_HALFSIFT;
        binary_desc = false;
        rooted = true;
        default_radius = 1.7f;
        sift_desc = true;
        break;
    }
#ifdef _ACD
    case IMAS_AC:
    {
        desc_name="AC";
        desc_type = IMAS_AC;
        binary_desc = false;
        rooted = false;
        default_radius = 1.7f;
        sift_desc = true;
        NewOriSize1 = 22;
        step_sigma = 1.5;
        break;
    }
    case IMAS_AC_W:
    {
        desc_name="AC-W";
        desc_type = IMAS_AC_W;
        binary_desc = false;
        rooted = false;
        default_radius = 1.7f;
        sift_desc = true;
        NewOriSize1 = 22;
        step_sigma = 1.5;
        break;
    }
    case IMAS_AC_Q:
    {
        desc_name="AC-Q";
        desc_type = IMAS_AC_Q;
        binary_desc = false;
        rooted = false;
        default_radius = 1.7f;
        sift_desc = true;
        NewOriSize1 = 18;
        quant_prec = 0.032;
        step_sigma = 1.5;
        break;
    }
#endif
#ifdef _LDAHASH
    case IMAS_DIF128:
    {
        desc_name="DIF128";
        nndrRatio = 0.625f;
        desc_type = IMAS_DIF128;
        binary_desc = true;
        default_radius = 1.7f;
        rooted = false;
        sift_desc = true;
        normType = IMAS::NORM_HAMMING;
        break;
    }
    case IMAS_DIF64:
    {
        desc_name="DIF64";
        nndrRatio = 0.625f;
        desc_type = IMAS_DIF64;
        binary_desc = true;
        default_radius = 1.7f;
        rooted = false;
        sift_desc = true;
        normType = IMAS::NORM_HAMMING;
        break;
    }
    case IMAS_LDA128:
    {
        desc_name="LDA128";
        nndrRatio = 0.625f;
        desc_type = IMAS_LDA128;
        binary_desc = true;
        default_radius = 1.7f;
        rooted = false;
        sift_desc = true;
        normType = IMAS::NORM_HAMMING;
        break;
    }
    case IMAS_LDA64:
    {
        desc_name="LDA64";
        nndrRatio = 0.625f;
        desc_type = IMAS_LDA64;
        binary_desc = true;
        default_radius = 1.7f;
        rooted = false;
        sift_desc = true;
        normType = IMAS::NORM_HAMMING;
        break;
    }
#endif
#endif
    }
    updateparams();
    return(desc_name);
}



void vectorimage2imasimage(std::vector<float>& input_image, IMAS::IMAS_Matrix &output_image, int width, int height)
{
#ifdef _NO_OPENCV
    //COPY
    //output_image = *(new IMAS::IMAS_Matrix(input_image,width,height));

    //JUST BIND
    output_image = *(new IMAS::IMAS_Matrix());
    output_image.data = input_image.data();
    output_image.cols = width;
    output_image.rows = height;
#else
    output_image.create(height, width, CV_8UC1);//cv::imread(argv[2]);

    for (int i = 0;  i < output_image.rows; i++)
    {
        for (int j = 0; j < output_image.cols; j++)
        {
            output_image.data[((output_image.cols)*i)+j] = (uchar) floor(input_image[((output_image.cols)*i)+j]);
        }
    }
#endif
}

void imasimage2vectorimage(IMAS::IMAS_Matrix input_image,std::vector<float>& output_image,int& width, int& height)
{
#ifdef _NO_OPENCV
    width = input_image.cols;
    height = input_image.rows;
    output_image.resize(width*height);
    for (int i = 0;  i < width*height; i++)
        output_image[i] = input_image.data[i];
#else
    output_image.clear();
    width = input_image.cols;
    height = input_image.rows;

    for (int i = 0;  i < input_image.rows; i++)
    {
        for (int j = 0; j < input_image.cols; j++)
        {
            output_image.push_back( (float) input_image.data[((input_image.cols)*i)+j] );
        }
    }
#endif
}


void floatarray2imasimage(float *input_image, IMAS::IMAS_Matrix &output_image, int width, int height)
{
#ifdef _NO_OPENCV
    output_image = *(new IMAS::IMAS_Matrix);
    output_image.data = input_image;
    output_image.cols = width;
    output_image.rows = height;
#else
    output_image.create(height, width, CV_8UC1);//cv::imread(argv[2]);

    for (int i = 0;  i < output_image.rows; i++)
    {
        for (int j = 0; j < output_image.cols; j++)
        {
            output_image.data[((output_image.cols)*i)+j] = (uchar) floor(input_image[((output_image.cols)*i)+j]);
        }
    }
#endif
}



#ifndef _NO_OPENCV
void detector_and_descriptor(cv::Ptr<cv::FeatureDetector> &detector, cv::Ptr<cv::DescriptorExtractor> &extractor)
{
    switch (desc_type)
    {
    case IMAS_SIFT:
    {
        detector = cv::xfeatures2d::SIFT::create();
        extractor = cv::xfeatures2d::SIFT::create();
        break;
    }
    case IMAS_SIFT2:
    {
        detector = cv::xfeatures2d::SIFT::create();
        extractor = cv::xfeatures2d::SIFT::create();
        break;
    }
    case IMAS_HALFSIFT:
    {
        detector = cv::xfeatures2d::SIFT::create();
        extractor = cv::xfeatures2d::SIFT::create();
        break;
    }
    case IMAS_ROOTSIFT:
    {
        detector = cv::xfeatures2d::SIFT::create();
        extractor = cv::xfeatures2d::SIFT::create();
        break;
    }
    case IMAS_SURF:
    {
        detector = cv::xfeatures2d::SURF::create();
        extractor = cv::xfeatures2d::SURF::create();
        break;
    }
    case IMAS_BRISK:
    {
        detector = cv::BRISK::create();
        extractor = cv::BRISK::create();
        break;
    }
    case IMAS_FREAK:
    {
        detector = cv::xfeatures2d::SURF::create();
        extractor = cv::xfeatures2d::FREAK::create();
        break;
    }
    case IMAS_ORB:
    {
        detector = cv::ORB::create(1500);
        extractor = cv::ORB::create();
        break;
    }
    case IMAS_BRIEF:
    {
        detector = cv::xfeatures2d::StarDetector::create();
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
        break;
    }
    case IMAS_AGAST:
    {
        detector = cv::ORB::create();
        extractor = cv::AgastFeatureDetector::create();
        break;
    }
    case IMAS_LATCH:
    {
        detector = cv::ORB::create(1500);
        extractor = cv::xfeatures2d::LATCH::create();
        break;
    }
    case IMAS_LUCID:
    {
        detector = cv::ORB::create();
        extractor = cv::xfeatures2d::LUCID::create(); // Needs 3-channels image
        break;
    }
    case IMAS_DAISY:
    {
        detector = cv::ORB::create(1500);
        extractor = cv::xfeatures2d::DAISY::create();
        break;
    }
    case IMAS_AKAZE:
    {
        detector = cv::AKAZE::create();
        extractor = cv::AKAZE::create();
        break;
    }

    }
}
#endif



void compute_local_descriptor_keypoints(IMAS::IMAS_Matrix &queryImg,  IMAS_keypointlist& KPs, float t, float theta)
{

    if(!queryImg.empty())
    {
#ifdef _NO_OPENCV
        if (sift_desc)
        {
            keypointslist* keys = new keypointslist;
            compute_sift_keypoints(queryImg.data,*keys,queryImg.cols,queryImg.rows,siftparameters);
            //KPs.DescList.resize(keys->size());
            KPs.resize(keys->size());
            for(int i=0; i<(int)keys->size();i++)
            {
                KPs[i].pt.x = (*keys)[i].x;
                KPs[i].pt.y = (*keys)[i].y;

#ifdef _LDAHASH
                if (desc_type>=41 && desc_type<=44)
                    KPs[i].pt.kp_ptr = lda_describe_from_SIFT( (*keys)[i], desc_type);
                else
                    KPs[i].pt.kp_ptr = &((*keys)[i]);
#else
                KPs[i].pt.kp_ptr = &((*keys)[i]);
#endif
                KPs[i].size = (*keys)[i].radius;
                KPs[i].scale = (*keys)[i].scale;
                KPs[i].angle = (*keys)[i].angle;
                KPs[i].t = t;
                KPs[i].theta = theta;
            }
        }
        else
        {
            listDescriptor* keys = extract_surf(queryImg.data, queryImg.cols,queryImg.rows);
            KPs.resize(keys->size());
            for(int i=0; i<(int)keys->size();i++)
            {
                KPs[i].pt.x = (*keys)[i]->kP->x;
                KPs[i].pt.y = (*keys)[i]->kP->y;
                KPs[i].pt.kp_ptr = (*keys)[i];
                KPs[i].size = 2*(*keys)[i]->kP->scale-1;
                KPs[i].scale = (*keys)[i]->kP->scale;
                KPs[i].angle = (*keys)[i]->kP->orientation;
                KPs[i].t = t;
                KPs[i].theta = theta;
            }
        }
#else
        ////////////////////////////
        // EXTRACT KEYPOINTS and FEATURES
        ////////////////////////////
        cv::Ptr<cv::FeatureDetector> detector;
        cv::Ptr<cv::DescriptorExtractor> extractor;
        detector_and_descriptor(detector, extractor);


        cv::Mat dlist;
        std::vector<cv::KeyPoint> klist;
        detector->detect(queryImg, klist);
        extractor->compute(queryImg, klist, dlist);

        if (desc_type==IMAS_ROOTSIFT)
        {
            int rows;
            rows = dlist.rows;

            for (int ii=0;ii<rows;ii++)
            {
                cv::Mat temp,temp2;
                cv::normalize( dlist.row(ii), temp, 1, cv::NORM_L1);
                cv::sqrt(temp, temp2);
                temp2.row(0).copyTo(dlist.row(ii));
            }

        }
        if (desc_type==IMAS_HALFSIFT)
        {
            int rows;
            cv::Mat temp;
            rows = dlist.rows;

            for (int ii=0;ii<rows;ii++)
                for (int jhist=0;jhist<16;jhist++)
                    for(int jori=0;jori<4;jori++)
                    {
                        temp = ( dlist.row(ii).col(jhist*8+jori) + dlist.row(ii).col(jhist*8+jori+4) );
                        temp.copyTo(dlist.row(ii).col(jhist*8+jori));
                        temp.copyTo(dlist.row(ii).col(jhist*8+jori+4));

                    }
        }


        int rows;
        rows = dlist.rows;
        KPs.resize(klist.size());
        for (int i=0;i<rows;i++)
        {
            KPs[i].pt.x = klist[i].pt.x;
            KPs[i].pt.y = klist[i].pt.y;
            KPs[i].size = klist[i].size;
            KPs[i].scale = klist[i].octave;
            KPs[i].angle = klist[i].angle;
            KPs[i].t = t;
            KPs[i].theta = theta;

            KPs[i].pt.kp_ptr = new cv::Mat(dlist.row(i));
        }
#endif
    }
}






//******************************************* MATCHER and FILTERS


void USAC_Filter(matchingslist& matchings, double precision, bool doFundamental, bool verb)
{
#ifdef _USAC
    // store the coordinates of the matching points
    std::vector<double> point_data;
    std::vector<unsigned int> prosac_data;
    prosac_data.resize(matchings.size());
    point_data.resize(6*matchings.size());
    for (int i = 0; i < (int) matchings.size(); i++ )
    {
        point_data[6*i] = matchings[i].first.x;
        point_data[6*i+1] = matchings[i].first.y;
        point_data[6*i+3] = matchings[i].second.x;
        point_data[6*i+4] = matchings[i].second.y;
        point_data[6*i+2] = 1.0;
        point_data[6*i+5] = 1.0;
    }

    // Estimation of fundamental matrix with ORSA
    matchingslist matchings_unique;
    matchings_unique.clear();
    libNumerics::matrix<double> F(3,3);


    std::string cfg_file_path, dir(get_current_dir_name());
    if(doFundamental)
        cfg_file_path =dir+"/fundamental.cfg";
    else
        cfg_file_path =dir+"/homography.cfg";



    if (doFundamental)
    {
        // ------------------------------------------------------------------------
        // initialize the fundamental matrix estimation problem
        ConfigParamsFund cfg;
        if ( !cfg.initParamsFromConfigFile((cfg_file_path)) )
        {
            std::cerr << "Error during initialization" << std::endl;
        }
        cfg.common.numDataPoints = matchings.size();
        cfg.common.inlierThreshold = precision;
        FundMatrixEstimator* fund = new FundMatrixEstimator;
        fund->initParamsUSAC(cfg);

        // set up the fundamental matrix estimation problem

        fund->initDataUSAC(cfg);
        fund->initProblem(cfg, &point_data[0]);
        fund->solve();

        for (unsigned int i = 0; i < 3; ++i)
        {
            for (unsigned int j = 0; j < 3; ++j)
            {
                F(i,j) = fund->final_model_params_[3*i+j];
            }
        }

        for (unsigned int i = 0; i < matchings.size(); ++i)
        {
            if(fund->usac_results_.inlier_flags_[i])
                matchings_unique.push_back(matchings[i]);
        }

        // clean up
        point_data.clear();
        prosac_data.clear();
        fund->cleanupProblem();
        delete fund;

    } else
    {
        // ------------------------------------------------------------------------
        // initialize the homography estimation problem
        ConfigParamsHomog cfg;
        if ( !cfg.initParamsFromConfigFile((cfg_file_path)) )
            std::cerr << "Error during initialization" << std::endl;

        HomogEstimator* homog = new HomogEstimator;
        cfg.common.numDataPoints = matchings.size();
        cfg.common.inlierThreshold = precision;
        homog->initParamsUSAC(cfg);

        // set up the homography estimation problem
        homog->initDataUSAC(cfg);
        homog->initProblem(cfg, &point_data[0]);
        homog->solve();

        // write out results
        for(unsigned int i = 0; i < 3; ++i)
        {
            for (unsigned int j = 0; j < 3; ++j)
            {
                F(i,j) = homog->final_model_params_[3*i+j];
            }
        }

        for (unsigned int i = 0; i < matchings.size(); ++i)
        {
            if (homog->usac_results_.inlier_flags_[i])
                matchings_unique.push_back(matchings[i]);
        }


        // clean up
        point_data.clear();
        prosac_data.clear();
        homog->cleanupProblem();
        delete homog;
    }



    // if the matching is significant, register the good matches
    if ( matchings_unique.size()>0 )
    {
        matchings.clear();

        for (int cc = 0; cc < (int) matchings_unique.size(); cc++ )
            matchings.push_back(matchings_unique[cc]);

        IdentifiedMaps.clear();
        IdentifiedMaps.push_back(F);

        if (verb)
        {
            my_Printf("The two images match! %d matchings are identified.\n", (int) matchings.size());
            if (doFundamental)
                std::cout << "*************** Fundamental Matrix ***************"<< std::endl;
            else
                std::cout << "*************** Homography ***************"<< std::endl;
            std::cout << F <<std::endl;
            std::cout << "**************************************************"<< std::endl;
        }
    }
    else
    {
        matchings.clear();
        if (verb)
            my_Printf("The two images do not match. The matching is not significant");
    }
#else
    matchings.clear();
    cerr<<"The CMakeLists.txt file has the option USAC set to OFF. Please set it to ON to use USAC."<<endl;
#endif
}


/**
 * @brief Applies an Epipolar filter based on the article: <a href="http://www.ipol.im/pub/art/2016/147/">Lionel Moisan, Pierre Moulon, and Pascal Monasse, Fundamental Matrix of a Stereo Pair, with A Contrario Elimination of Outliers, Image Processing On Line, 6 (2016), pp. 89–113</a>.
 * @param matchings Current matches
 * @param w1 Width of image1
 * @param h1 Height of image1
 * @param w2 Width of image2
 * @param h2 Height of image2
 * @param nfa_max Maximum accepted nfa to consider a map as meaningul.
 * @param ITER_ORSA Maximum number of iterations to be applied in Orsa.
 * @param precision Imposed precision in pixels for ORSA Fundamental. In other words, the distance from a point to the epipolar line should be less than equal to precision. Put precision<=0 if no precision is needed.
 * @param verb Verbose mode
 */
void ORSA_EpipolarFilter(matchingslist& matchings,int w1,int h1,int w2,int h2, double nfa_max,int ITER_ORSA,double precision,bool verb)
{
    //////// Use ORSA to filter out the incorrect matches.
    // store the coordinates of the matching points
    vector<Match> match_coor;
    for (int cc = 0; cc < (int) matchings.size(); cc++ )
    {
        Match match1_coor;
        match1_coor.x1 = matchings[cc].first.x;
        match1_coor.y1 = matchings[cc].first.y;
        match1_coor.x2 = matchings[cc].second.x;
        match1_coor.y2 = matchings[cc].second.y;

        match_coor.push_back(match1_coor);
    }
    // Estimation of fundamental matrix with ORSA
    matchingslist matchings_unique;
    libNumerics::matrix<double> F(3,3);
    std::vector<int> vec_inliers;
    double nfa;
    orsa::orsa_fundamental(match_coor, w1,h1,w2,h2, precision, ITER_ORSA,F, vec_inliers,nfa,verb);



    // if the matching is significant, register the good matches
    if ( nfa < nfa_max )
    {
        //  std::cout << "F=" << F <<std::endl;
        // extract meaningful matches
        matchings_unique.clear();
        for (int cc = 0; cc < (int) vec_inliers.size(); cc++ )
        {
            matchings_unique.push_back(matchings[vec_inliers[cc]]);
        }
        matchings.clear();

        for (int cc = 0; cc < (int) vec_inliers.size(); cc++ )
        {
            matchings.push_back(matchings_unique[cc]);
        }

        IdentifiedMaps.clear();
        IdentifiedMaps.push_back(F);

        if (verb)
        {
            my_Printf("The two images match! %d matchings are identified. log(nfa)=%.2f.\n", (int) matchings.size(), nfa);
            std::cout << "*************** Fundamental Matrix ***************"<< std::endl;
            std::cout << F <<std::endl;
            std::cout << "**************************************************"<< std::endl;
        }



    }
    else
    {
        matchings.clear();
        if (verb)
            my_Printf("The two images do not match. The matching is not significant:  log(nfa)=%.2f.\n", nfa);
    }
}

/**
 * @brief Applies an Homography filter based on the article: <a href="http://www.ipol.im/pub/art/2012/mmm-oh/">Lionel Moisan, Pierre Moulon, and Pascal Monasse, Automatic Homographic Registration of a Pair of Images, with A Contrario Elimination of Outliers, Image Processing On Line, 2 (2012), pp. 56–73</a>.
 * @param matchings Current matches
 * @param w1 Width of image1
 * @param h1 Height of image1
 * @param w2 Width of image2
 * @param h2 Height of image2
 * @param nfa_max Maximum accepted nfa to consider a map as meaningul.
 * @param ITER_ORSA Maximum number of iterations to be applied in Orsa.
 * @param precision Imposed precision in pixels for ORSA Homography. In other words, the distance from a point to the plane should be less than equal to precision. Put precision<=0 if no precision is needed.
 * @param verb Verbose mode
 */
void ORSA_HomographyFilter(matchingslist& matchings,int w1,int h1,int w2,int h2, double nfa_max,int ITER_ORSA,double precision,bool verb)
{
    //////// Use ORSA to filter out the incorrect matches.
    // store the coordinates of the matching points
    vector<Match> match_coor;
    for (int cc = 0; cc < (int) matchings.size(); cc++ )
    {
        Match match1_coor;
        match1_coor.x1 = matchings[cc].first.x;
        match1_coor.y1 = matchings[cc].first.y;
        match1_coor.x2 = matchings[cc].second.x;
        match1_coor.y2 = matchings[cc].second.y;

        match_coor.push_back(match1_coor);
    }
    // Estimation of fundamental matrix with ORSA
    matchingslist matchings_unique;
    libNumerics::matrix<double> H(3,3);
    std::vector<int> vec_inliers;
    double nfa;
    orsa::ORSA_homography(match_coor, w1,h1,w2,h2, precision, ITER_ORSA,H, vec_inliers,nfa,verb);



    // if the matching is significant, register the good matches
    if ( nfa < nfa_max )
    {
        //  std::cout << "F=" << F <<std::endl;
        // extract meaningful matches
        matchings_unique.resize(vec_inliers.size());

        for (int cc = 0; cc < (int) vec_inliers.size(); cc++ )
        {
            matchings_unique[cc] = matchings[vec_inliers[cc]];
        }

        matchings.resize(vec_inliers.size());
        for (int cc = 0; cc < (int) vec_inliers.size(); cc++ )
        {
            matchings[cc] = matchings_unique[cc];
        }

        H /= H(2,2);

        IdentifiedMaps.clear();
        IdentifiedMaps.push_back(H);

        if (verb)
        {
            my_Printf("The two images match! %d matchings are identified. log(nfa)=%.2f.\n", (int) matchings.size(), nfa);
            std::cout << "*************** Homography ***************"<< std::endl;
            std::cout << H <<std::endl;
            std::cout << "******************************************"<< std::endl;
        }
    }
    else
    {
        matchings.clear();
        if (verb)
            my_Printf("The two images do not match. The matching is not significant:  log(nfa)=%.2f.\n", nfa);
    }
}



#ifdef _NO_OPENCV
//unsigned int contador = 0;
template <unsigned int OriSize,unsigned int IndexSize>
/**
 * @brief Computes the classical real distance but stops computing
 * once this distance gets bigger than tdist.
 * @param (k1,k2) Pair of keypoints
 * @param tdist Current minimum distance
 * @param par Which norm to use (either L1 or L2)
 * @return Either \f$\Vert k1 - k2 \Vert_{L_1} \f$ or \f$\Vert k1 - k2 \Vert_{L_2}^2 \f$
 * @author Mariano Rodríguez
 */
float distance_sift(keypoint_base<OriSize,IndexSize> *k1,keypoint_base<OriSize,IndexSize> *k2, float tdist, bool L2norm)
{
    float dif;
    float distsq = 0.0;

    float *ik1 = k1->vec;
    float *ik2 = k2->vec;

    for (int i = 0; (i < (int)k1->veclength)&&(distsq <= tdist); i++)
    {
        dif = ik1[i] - ik2[i];
        if (L2norm)
            distsq += dif * dif;
        else
            distsq += std::abs(dif);
    }

    return distsq;
}
#else
/**
 * @brief Computes the classical cv::norm distance but stops computing
 * once this distance gets bigger than tdist.
 * @param (k1,k2) Pair of cv keypoints
 * @param tdist Current minimum distance
 * @param par Which norm to use (either L1 or L2)
 * @return Either \f$\Vert k1 - k2 \Vert_{L_1} \f$ or \f$\Vert k1 - k2 \Vert_{L_2}^2 \f$
 * @author Mariano Rodríguez
 */
float distance_sift(cv::Mat* k1, cv::Mat* k2, float tdist, bool L2norm)
{
    float dif;
    float distsq = 0.f;

    for (int i = 0; (i < k1->cols)&&(distsq <= tdist); i++)
    {
        dif = k1->at<float>(0,i) - k2->at<float>(0,i);
        if (L2norm)
            distsq += dif * dif;
        else
            distsq += std::abs(dif);
    }
    return distsq;
}
#endif


/**
 * @brief Computes the generalised distance proposed in \cite imas_IPOL_2017 but stops computing
 * when this distance gets bigger than tdist.
 * @param (k1,k2) Pair of generalised keypoints
 * @param dist Current minimum distance
 * @param (ind1,ind2) Returns where the minimum was found.  \f$(ind1,ind2) \in k1 \times k2\f$
 * @param par Which norm to use (either L1 or L2)
 * @return \f$\min_{(\alpha,\beta)\in k1 \times k2} \delta(\alpha,\beta) \f$
 *   where   \f$\delta(x,y)\f$  is either  \f$\Vert x - y \Vert_{L_1} \f$  or  \f$\Vert x - y \Vert_{L_2} \f$
 * @author Mariano Rodríguez
 */
float distance_imasKP(IMAS::IMAS_KeyPoint *k1,IMAS::IMAS_KeyPoint *k2, float& dist,int &ind1, int &ind2, int tnorm)
{
    float tdist = dist;
    for(int i1=0;i1<(int)k1->KPvec.size();i1++)
        for(int i2=0;i2<(int)k2->KPvec.size();i2++)
        {
#ifndef _NO_OPENCV
            if (sift_desc)
                tdist = distance_sift(static_cast<IMAS::IMAS_Matrix*>(k1->KPvec[i1].pt.kp_ptr),static_cast<IMAS::IMAS_Matrix*>(k2->KPvec[i2].pt.kp_ptr),dist,tnorm==cv::NORM_L2);
            else
                tdist = (float)cv::norm(*static_cast<IMAS::IMAS_Matrix*>(k1->KPvec[i1].pt.kp_ptr),*static_cast<IMAS::IMAS_Matrix*>(k2->KPvec[i2].pt.kp_ptr),tnorm);
            //#pragma omp critical
            //            cout<<tdist<<" "<<tdist1<<endl;
            //tdist = opencv_distance(static_cast<IMAS::IMAS_Matrix*>(k1->KPvec[i1].pt.kp_ptr) , static_cast<IMAS::IMAS_Matrix*>(k2->KPvec[i2].pt.kp_ptr));
#else
            if (sift_desc)
#ifdef _LDAHASH
                if (desc_type>=41 && desc_type<=44)
                    tdist = lda_hamming_distance(static_cast<ldadescriptor*>(k1->KPvec[i1].pt.kp_ptr) , static_cast<ldadescriptor*>(k2->KPvec[i2].pt.kp_ptr), dist);
                else
                    tdist = distance_sift(static_cast<keypoint*>(k1->KPvec[i1].pt.kp_ptr) , static_cast<keypoint*>(k2->KPvec[i2].pt.kp_ptr), dist, tnorm==IMAS::NORM_L2);
#else
                tdist = distance_sift(static_cast<keypoint*>(k1->KPvec[i1].pt.kp_ptr) , static_cast<keypoint*>(k2->KPvec[i2].pt.kp_ptr), dist, tnorm==IMAS::NORM_L2);
#endif
            else
                if (static_cast<descriptor*>(k1->KPvec[i1].pt.kp_ptr)->kP->signLaplacian==static_cast<descriptor*>(k2->KPvec[i2].pt.kp_ptr)->kP->signLaplacian)
                    tdist = euclideanDistance(static_cast<descriptor*>(k1->KPvec[i1].pt.kp_ptr) , static_cast<descriptor*>(k2->KPvec[i2].pt.kp_ptr));
#endif
            if ( dist>tdist )
            {
                dist = tdist;
                ind1 = i1;
                ind2 = i2;
            }

        }

    return(dist);

}


/**
 * @brief Implements the ratio between first and second closest generalised keypoints proposed in \cite imas_IPOL_2017  based on the second-closest neighbor acceptance criterion
initially proposed by D. Lowe in \cite Lowe2004.
 * @param key A generalised query keypoint.
 * @param klist The whole list of target generalised keypoints.
 * @param min Returns the index for which the minimum distance is attained
 * @param (ind1,ind2) Returns where the minimum was found in  \f$(ind1,ind2) \in key \times klist[min]\f$
 * @param par Which norm to use (either L1 or L2) for computing distances
 * @return Found minimal ratio
 * @author Mariano Rodríguez
 */
float CheckForMatchIMAS(IMAS::IMAS_KeyPoint* key, std::vector<IMAS::IMAS_KeyPoint*>& klist, int& min, int& ind1, int& ind2, int tnorm)
{
    float	dsq, distsq1, distsq2;
#ifdef _NO_OPENCV
    if (tnorm==IMAS::NORM_L2)
        distsq1 = distsq2 = BIG_NUMBER_L2;
    else
        distsq1 = distsq2 = BIG_NUMBER_L1;
#else
    if (tnorm==cv::NORM_L2)
        distsq1 = distsq2 = BIG_NUMBER_L2;
    else
        distsq1 = distsq2 = BIG_NUMBER_L1;
#endif

    for (int j=0; j< (int) klist.size(); j++)
    {
        int i1=-1 ,i2=-1;
        dsq = distance_imasKP(key, klist[j], distsq2,i1,i2, tnorm);

        if (dsq < distsq1) {
            distsq2 = distsq1;
            distsq1 = dsq;
            min = j;
            ind1 = i1;
            ind2 = i2;
        } else if (dsq < distsq2) {
            distsq2 = dsq;
        }
    }
    if (distsq2==0)
        return BIG_NUMBER_L2;
    else
        return distsq1/distsq2 ;
}


/**
 * @brief Implements an acontrario version of the ratio between first and second closest generalised keypoints proposed in \cite imas_IPOL_2017.
 * @param key A generalised query keypoint.
 * @param klist The whole list of target generalised keypoints.
 * @param min Returns the index for which the minimum distance is attained
 * @param (ind1,ind2) Returns where the minimum was found in  \f$(ind1,ind2) \in key \times klist[min]\f$
 * @param par Which norm to use (either L1 or L2) for computing distances
 * @return Found minimal ratio
 * @author Mariano Rodríguez
 */
float CheckForMatchIMAS_acontrario(IMAS::IMAS_KeyPoint* key, std::vector<IMAS::IMAS_KeyPoint*>& klist, int& min, int& ind1, int& ind2, int tnorm)
{
    float	dsq, distsq1, distsq2, distsq3;
#ifdef _NO_OPENCV
    if (tnorm==IMAS::NORM_L2)
        distsq1 = distsq2 = BIG_NUMBER_L2;
    else
        distsq1 = distsq2 = BIG_NUMBER_L1;
#else
    if (tnorm==cv::NORM_L2)
        distsq1 = distsq2 = BIG_NUMBER_L2;
    else
        distsq1 = distsq2 = BIG_NUMBER_L1;
#endif

    for (int j=0; j< (int) klist.size(); j++)
    {
        int i1=-1, i2=-1;
        dsq = distance_imasKP(key, klist[j], distsq2,i1,i2, tnorm);

        if (dsq < distsq1) {
            distsq2 = distsq1;
            distsq1 = dsq;
            min = j;
            ind1 = i1;
            ind2 = i2;
        } else if (dsq < distsq2) {
            distsq2 = dsq;
        }
    }

#ifdef _NO_OPENCV
    if (tnorm==IMAS::NORM_L2)
        distsq2 = distsq3 = BIG_NUMBER_L2;
    else
        distsq2 = distsq3 =  BIG_NUMBER_L1;
#else
    if (tnorm==cv::NORM_L2)
        distsq2 = distsq3 = BIG_NUMBER_L2;
    else
        distsq2 = distsq3 =  BIG_NUMBER_L1;
#endif

    for (int j=0; j< (int) keys3.size(); j++)
    {
        int i1=-1, i2=-1;
        dsq = distance_imasKP(key, keys3[j], distsq3,i1,i2, tnorm);

        if (dsq < distsq2) {
            distsq3 = distsq2;
            distsq2 = dsq;
        } else if (dsq < distsq3) {
            distsq3 = dsq;
        }
    }
    if (distsq2==0)
        return BIG_NUMBER_L2;
    else
        return distsq1/distsq2 ;
}


#ifdef _ACD

#ifndef FALSE
#define FALSE 0
#endif /* !FALSE */

#ifndef TRUE
#define TRUE 1
#endif /* !TRUE */

/*----------------------------------------------------------------------------*/
/** PI */
#ifndef M_PI
#define M_PI   3.14159265358979323846
#endif /* !M_PI */

/*----------------------------------------------------------------------------*/
/** max value */
#define max(a,b) (((a)>(b))?(a):(b))

/** min value */
#define min(a,b) (((a)>(b))?(b):(a))

/*----------------------------------------------------------------------------*/
/**
 * @brief Normalized angle difference between 'a' and the symmetric of 'b'
    relative to a vertical axis.
* @author Rafael Grompone von Gioi
 */
static inline double norm_angle(double a, double b)
{
    a -= b;
    while( a <= -M_PI ) a += 2.0*M_PI;
    while( a >   M_PI ) a -= 2.0*M_PI;

    return fabs(a) / M_PI;
}

/*----------------------------------------------------------------------------*/
/**
 * @brief Simple patch_comparison
 * @param grad_angle1
 * @param grad_angle2
 * @param X
 * @param Y
 * @param logNT
 * @return
 * @author Rafael Grompone von Gioi, Mariano Rodríguez
 */
double patch_comparison( double * grad_angle1, double * grad_angle2,
                         int X, int Y, double logNT )
{
    int n = 0;      /* count of angles compared */
    double k = 0.0; /* measure of symmetric angles */
    double logNFAC;

    int nmax = (X-2)*(Y-2);
    /* logNFAC-logNT <= 0 */
    /* logNFAC-logNT = nmax * log10(k) - 0.5 * log10(2.0 * M_PI) - (nmax+0.5) * log10(nmax) + nmax * log10(exp(1.0)) */
    double threshold = pow(10, (0.5 * log10(2.0 * M_PI) + (nmax+0.5) * log10(nmax) - nmax * log10(exp(1.0)))/nmax );
    int x,y,A,B;


    for(x=1; x<X-1; x++)
        for(y=1; y<Y-1; y++)
        {
            A = grad_angle1[x+y*X] != NOTDEF;  /* A has defined gradient */
            B = grad_angle2[x+y*X] != NOTDEF;  /* B has defined gradient */

            /* if at least one of the corresponding pixels has defined gradient,
           count it in the total number of pixels evaluated */
            if( A || B ) ++n;

            if( A && B) k += norm_angle(grad_angle1[x+y*X], grad_angle2[x+y*X]);
            else if( (A && !B) || (!A && B) ) k += 1.0;

            if (k>threshold)
                return (logNT);
        }

    /* NFAC = NT * k^n / n!
     log(n!) is bounded by Stirling's approximation:
       n! >= sqrt(2pi) * n^(n+0.5) * exp(-n)
     then, log10(NFA) <= log10(NT) + n*log10(k) - log10(latter expansion) */
    logNFAC = logNT + n * log10(k)
            - 0.5 * log10(2.0 * M_PI) - (n+0.5) * log10(n) + n * log10(exp(1.0));

    return logNFAC;
}

double* weights;
double sum_log_w;
double threshold_AC;
double sigma_default = -1.0;


/**
 * @brief create_weights_for_patch_comparison
 * @param X
 * @param Y
 * @author Rafael Grompone von Gioi, Mariano Rodríguez
 */
void create_weights_for_patch_comparison(int X, int Y)
{
    sum_log_w = 0.0;
    delete[] weights;
    weights = new double[X*Y];
    int r = (int) (X/2), c = (int) (Y/2);
    double w;

    for(int x=0; x<X*Y; x++)
        weights[x] = NOTDEF;

    for(int x=1; x<X-1; x++)
        for(int y=1; y<Y-1; y++)
        {
            //w = exp( -(pow(r-x,2)+pow(c-y,2))/(2.0*X*sqrt(X)) );
            if (sigma_default>0)
                w = exp( -(pow(r-x,2)+pow(c-y,2))/(2.0*sigma_default) );
            else
                w = exp( -(pow(r-x,2)+pow(c-y,2))/(2.0*X*Y) );
            weights[x+y*X] = w;
            sum_log_w += log10(w);
        }
    int nmax = (X-2)*(Y-2);
    /* logNFAC-logNT <= 0 */
    /* logNFAC-logNT = nmax * log10(k) - 0.5 * log10(2.0 * M_PI) - (nmax+0.5) * log10(nmax) + nmax * log10(exp(1.0)) */
    threshold_AC = pow(10, (sum_log_w + 0.5 * log10(2.0 * M_PI) + (nmax+0.5) * log10(nmax) - nmax * log10(exp(1.0)))/nmax );
}

/*----------------------------------------------------------------------------*/
/**
 * @brief weighted_patch_comparison
 * @param grad_angle1
 * @param grad_angle2
 * @param grad_mod1
 * @param grad_mod2
 * @param X
 * @param Y
 * @param logNT
 * @return
 * @author Rafael Grompone von Gioi, Mariano Rodríguez
 */
double weighted_patch_comparison( double * grad_angle1, double * grad_angle2,
                                  double * grad_mod1,   double * grad_mod2,
                                  int X, int Y, double logNT )
{
    int n = 0;      /* count of angles compared */
    double k = 0.0; /* measure of symmetric angles */
    double logNFAC;
    int x,y;

    for(x=1; x<X-1; x++)
        for(y=1; y<Y-1; y++)
        {
            double a = grad_angle1[x+y*X];
            double b = grad_angle2[x+y*X];
            int A = (a != NOTDEF);  /* A has defined gradient */
            int B = (b != NOTDEF);  /* B has defined gradient */

            /* if at least one of the corresponding pixels has defined gradient,
           count it in the total number of pixels evaluated */
            if( A || B )
            {
                ++n;
                if( A && B) k += weights[x+y*X] * norm_angle(a,b);
                else        k += weights[x+y*X];  /* one angle not defined, maximal error = 1 */
            }

            if (k>threshold_AC)
                return (logNT);
        }

    /* NFAC = NT * k^n / (n! * prod_i w_i)
     log(n!) is bounded by Stirling's approximation:
       n! >= sqrt(2pi) * n^(n+0.5) * exp(-n)
     then, log10(NFA) <= log10(NT) + n*log10(k) - log10(latter expansion) */
    logNFAC = logNT + n * log10(k) - sum_log_w - 0.5 * log10(2.0 * M_PI) - (n+0.5) * log10(n) + n * log10(exp(1.0));

    return logNFAC;
}



/*----------------------------------------------------------------------------*/
/**
 * @brief Computes log10( NT * B(n,k,p) ), where B(,,) is the tail of the binomial
    distribution, and logNT is log10(NT); the value is estimated using
    Hoeffding's inequality.
 * @param logNT
 * @param n
 * @param k
 * @param p
 * @return
 * @author Rafael Grompone von Gioi
 */
double nfa(double logNT, int n, int k, double p)
{
    double r = (double) k / (double) n;

    if( r <= p ) return logNT;

    double log_binom = k * log10(p/r) + n*(1-r) * log10( (1-p)/(1-r) );
    return logNT + log_binom;
}

/**
 * @brief quantised_patch_comparison
 * @param grad_angle1
 * @param grad_angle2
 * @param grad_mod1
 * @param grad_mod2
 * @param X
 * @param Y
 * @param logNT
 * @return
 * @author Rafael Grompone von Gioi, Mariano Rodríguez
 */
double quantised_patch_comparison( double * grad_angle1, double * grad_angle2,
                                   double * grad_mod1,   double * grad_mod2,
                                   int X, int Y, double logNT )
{
    int n = 0;      /* count of angles compared */
    int k = 0; /* measure of symmetric angles */
    double logNFAC;
    int x,y;


    for(x=1; x<X-1; x++)
        for(y=1; y<Y-1; y++)
        {
            double a = grad_angle1[x+y*X];
            double b = grad_angle2[x+y*X];
            int A = (a != NOTDEF);  /* A has defined gradient */
            int B = (b != NOTDEF);  /* B has defined gradient */

            /* if at least one of the corresponding pixels has defined gradient,
           count it in the total number of pixels evaluated */
            if( A || B ) //if( A || B )
            {
                ++n;

                if(( A && B)&&(norm_angle(a,b)<quant_prec))
                    k ++ ;
                /* one angle not defined, maximal error = 1 */
            }
        }

    /* NFAC = NT * k^n / (n! * prod_i w_i)
     log(n!) is bounded by Stirling's approximation:
       n! >= sqrt(2pi) * n^(n+0.5) * exp(-n)
     then, log10(NFA) <= log10(NT) + n*log10(k) - log10(latter expansion) */

    logNFAC = nfa(logNT,n,k,quant_prec);

    return logNFAC;
}

#endif


/**
 * @brief Computes matches among hyper-descriptors coming from query and target images as described in \cite imas_IPOL_2017
 * @param w1 Width of image1
 * @param h1 Height of image1
 * @param w2 Width of image2
 * @param h2 Height of image2
 * @param keys1 Keypoints and hyper-descriptors found on all simulated optical tilts of query image
 * @param keys2 Keypoints and hyper-descriptors found on all simulated optical tilts of target image
 * @param matchings Returns a vector of matches after filtering
 * @param applyfilter filter to apply to RAW matches. It could be ORSA Homography \cite Moisan2012 or ORSA Fundamental \cite Moisan2016.
 * @return Total number of matches
 * @author Mariano Rodríguez
 */
int IMAS_matcher(int w1, int h1, int w2, int h2, std::vector<IMAS::IMAS_KeyPoint*>& keys1, std::vector<IMAS::IMAS_KeyPoint*>& keys2, matchingslist &matchings, int applyfilter)
{
    IMAS_time tstart = IMAS::IMAS_getTickCount();
    my_Printf("IMAS-Matcher...\n");

    float	minratio, sqratio;

    minratio = nndrRatio;
#ifdef _ACD
    if (!(desc_type == IMAS_AC || desc_type ==IMAS_AC_Q || desc_type == IMAS_AC_W))
#endif
    {
#pragma omp parallel for
        for (int i=0; i< (int) keys1.size(); i++)
        {
            int imatch=-1, ind1 = -1, ind2 = -1;

            if (!keys3.empty())
            {
                sqratio = CheckForMatchIMAS_acontrario(keys1[i], keys2, imatch,ind1,ind2,normType);
            }
            else
            {
                sqratio = CheckForMatchIMAS(keys1[i], keys2, imatch,ind1,ind2,normType);
            }
            if (sqratio< minratio)
            {


                //my_Printf("par.MatchRatio = %f, sqratio = %f, sqratiomin = %f \n",par.MatchRatio,sqratio,sqminratio);
                keypoint_simple k1, k2;

                k1.x = keys1[i]->KPvec[ind1].pt.x;
                k1.y = keys1[i]->KPvec[ind1].pt.y;
                k1.scale = keys1[i]->KPvec[ind1].scale;
                k1.angle = keys1[i]->KPvec[ind1].angle;
                k1.theta = keys1[i]->KPvec[ind1].theta;
                k1.t = keys1[i]->KPvec[ind1].t;
                k1.size = keys1[i]->KPvec[ind1].size;

                k2.x = keys2[imatch]->KPvec[ind2].pt.x;
                k2.y = keys2[imatch]->KPvec[ind2].pt.y;
                k2.scale = keys2[imatch]->KPvec[ind2].scale;
                k2.angle = keys2[imatch]->KPvec[ind2].angle;
                k2.theta = keys2[imatch]->KPvec[ind2].theta;
                k2.t = keys2[imatch]->KPvec[ind2].t;
                k2.size = keys2[imatch]->KPvec[ind2].size;

#pragma omp critical
                matchings.push_back( matching(k1,k2) );
            }
        }
    }
#ifdef _ACD
    else
    {
        if(desc_type==IMAS_AC_W)
            create_weights_for_patch_comparison(NewOriSize1, NewOriSize1);
        int X1 = w1, Y1 = h1, X2 = w2, Y2 = h2;
        double logNT;
        logNT = 1.5*log10(X1) + 1.5*log10(Y1)
                + 1.5*log10(X2) + 1.5*log10(Y2)
                + log10( log( 2.0 * max(X1,Y1) ) / log(2.0) )
                + log10( log( 2.0 * max(X2,Y2) ) / log(2.0) )
                + 2.0*log10(_arearatio);

#pragma omp parallel for
        for (int n1=0; n1< (int) keys1.size(); n1++)
            for (int n2=0; n2< (int) keys2.size(); n2++)
            {
                IMAS::IMAS_KeyPoint *k1, *k2;
                k1 = keys1[n1];
                k2 = keys2[n2];
                int ind1 = -1, ind2 = -1;
                double bestlogNFA = logNT, logNFA = 1000.0;
                for(int i1=0;i1<(int)k1->KPvec.size();i1++)
                    for(int i2=0;i2<(int)k2->KPvec.size();i2++)
                    {

                        switch (desc_type) {
                        case IMAS_AC: // without weights
                        {
                            logNFA = patch_comparison(
                                        static_cast<keypoint*>(k1->KPvec[i1].pt.kp_ptr)->gradangle,
                                        static_cast<keypoint*>(k2->KPvec[i2].pt.kp_ptr)->gradangle,
                                        NewOriSize1,NewOriSize1,logNT);
                            break;
                        }
                        case IMAS_AC_W: //weighted
                        {
                            logNFA = weighted_patch_comparison(
                                        static_cast<keypoint*>(k1->KPvec[i1].pt.kp_ptr)->gradangle,
                                        static_cast<keypoint*>(k2->KPvec[i2].pt.kp_ptr)->gradangle,
                                        static_cast<keypoint*>(k1->KPvec[i1].pt.kp_ptr)->gradmod,
                                        static_cast<keypoint*>(k2->KPvec[i2].pt.kp_ptr)->gradmod,
                                        NewOriSize1,NewOriSize1,logNT);
                            break;
                        }
                        case IMAS_AC_Q: //quantised
                        {
                            logNFA = quantised_patch_comparison(
                                        static_cast<keypoint*>(k1->KPvec[i1].pt.kp_ptr)->gradangle,
                                        static_cast<keypoint*>(k2->KPvec[i2].pt.kp_ptr)->gradangle,
                                        static_cast<keypoint*>(k1->KPvec[i1].pt.kp_ptr)->gradmod,
                                        static_cast<keypoint*>(k2->KPvec[i2].pt.kp_ptr)->gradmod,
                                        NewOriSize1,NewOriSize1,logNT);
                            break;
                        }

                        }

                        if ( (0>logNFA) && (bestlogNFA>logNFA) )
                        {
                            ind1 = i1;
                            ind2 = i2;
                            bestlogNFA = logNFA;
                        }
                    }

                if (bestlogNFA<0)
                {
                    {
                        keypoint_simple k1, k2;

                        k1.x = keys1[n1]->KPvec[ind1].pt.x;
                        k1.y = keys1[n1]->KPvec[ind1].pt.y;
                        k1.scale = keys1[n1]->KPvec[ind1].scale;
                        k1.angle = keys1[n1]->KPvec[ind1].angle;
                        k1.theta = keys1[n1]->KPvec[ind1].theta;
                        k1.t = keys1[n1]->KPvec[ind1].t;
                        k1.size = keys1[n1]->KPvec[ind1].size;

                        k2.x = keys2[n2]->KPvec[ind2].pt.x;
                        k2.y = keys2[n2]->KPvec[ind2].pt.y;
                        k2.scale = keys2[n2]->KPvec[ind2].scale;
                        k2.angle = keys2[n2]->KPvec[ind2].angle;
                        k2.theta = keys2[n2]->KPvec[ind2].theta;
                        k2.t = keys2[n2]->KPvec[ind2].t;
                        k2.size = keys2[n2]->KPvec[ind2].size;


#pragma omp critical
                        matchings.push_back( matching(k1,k2) );
                    }

                }
            }
    }
#endif
    my_Printf("   %d possible matches have been found. \n", (int) matchings.size());
    my_Printf("IMAS-Matcher accomplished in %.2f seconds.\n \n", (IMAS::IMAS_getTickCount() - tstart)/ IMAS::IMAS_getTickFrequency());


    tstart = IMAS::IMAS_getTickCount();

    // If (enough matches to do epipolar filtering)
    if ( ( (int) matchings.size() >= Filter_num_min ) )
    {
        my_Printf("Filters... \n");
        float nfa_max = -2;
        const int ITER_ORSA=10000;
        const bool verb=true;
        //std::srand(std::time(0));
        if (applyfilter==ORSA_FUNDAMENTAL)
        {
            // Lionel Moisan, Pierre Moulon, and Pascal Monasse, Fundamental Matrix of a Stereo Pair, with A Contrario Elimination of Outliers, Image Processing On Line, 6 (2016), pp. 89–113.
            my_Printf("-> Applying ORSA filter (Fundamental Matrix) \n");

            // Fundamental matrix Estimation with ORSA
            ORSA_EpipolarFilter( matchings, w1, h1, w2, h2, nfa_max, ITER_ORSA, Filter_precision, verb);
        }

        if (applyfilter==ORSA_HOMOGRAPHY)
        {
            // Lionel Moisan, Pierre Moulon, and Pascal Monasse, Automatic Homographic Registration of a Pair of Images, with A Contrario Elimination of Outliers, Image Processing On Line, 2 (2012), pp. 56–73.
            my_Printf("-> Applying ORSA filter (Homography) \n");

            // Homography Estimation with ORSA
            ORSA_HomographyFilter( matchings, w1, h1, w2, h2, nfa_max, ITER_ORSA, Filter_precision, verb);
        }

        if (applyfilter==USAC_HOMOGRAPHY || applyfilter==USAC_FUNDAMENTAL )
        {
            // Raguram, R., Chum, O., Pollefeys, M., Matas, J., & Frahm, J. M. (2013). USAC: a universal framework for random sample consensus. IEEE transactions on pattern analysis and machine intelligence, 35(8), 2022-2038.
            if (applyfilter==USAC_HOMOGRAPHY)
                my_Printf("-> Applying USAC filter (Homography) \n");
            else
                my_Printf("-> Applying USAC filter (Fundamental) \n");

            cout<<"Imposed precision <= "<< Filter_precision<<endl;

            // Estimation with USAC
            USAC_Filter( matchings,Filter_precision, applyfilter==USAC_FUNDAMENTAL,  verb);
        }



        my_Printf("   Number of filtered matches =  %d. \n", (int) matchings.size());
        my_Printf("Filters were applied in %.2f seconds.\n", (IMAS::IMAS_getTickCount() - tstart)/ IMAS::IMAS_getTickFrequency());
    }
    else
    {
        if (( (int) matchings.size() < Filter_num_min ) )
            my_Printf("Not enough matches to extract the underlying meaningful transformation\n");
        matchings.clear();
    }

    my_Printf("\n   Final number of matches =  %d. \n\n", (int) matchings.size());



    return matchings.size();

}






//********************************************** DETECTOR AND EXTRACTOR





/**
 * @brief Computes tilted coordinates \f$ (x^\prime,y^\prime) \in [1,m_x]\times[1,m_y]\f$ from image coordinates \f$(x,y) \in [1,n_x]\times[1,n_y]\f$ where
 * \f$ T_tR_\theta (x,y) = (x^\prime,y^\prime) \f$. This takes into account the fact that a digital rotation \f$R_\theta\f$ applies a translation to reframe the image.
 * @param (x,y) Enters image coordiantes and returns tilted coordinates
 * @param w Width from the original image (\f$ w = n_x \f$)
 * @param h Width from the original image (\f$ h = n_y \f$)
 * @param t Tilt applied in the y direction
 * @param theta Direction \f$\theta\f$ of the tilt
 * @author Mariano Rodríguez
 */
void imagecoor2tiltedcoor(float& x, float& y, const int& w, const int& h, const float& t, const float& theta)
{
    // Apply R_\theta

    // Prepare rotation matrix
    libNumerics::matrix<float> Rot(2,2), vec(2,1), corners(2,4);
    // Rot = [cos(Rtheta) -sin(Rtheta);sin(Rtheta) cos(Rtheta)];
    Rot(0,0) = cos(theta); Rot(0,1) = sin(theta);
    Rot(1,0) = -sin(theta); Rot(1,1) = cos(theta);


    vec(0,0) = x-1;
    vec(1,0) = y-1;

    // rotation -> [x1;y1] = Rot*[x1;y1]
    vec = (Rot*vec);
    x = vec(0,0);
    y = vec(1,0);


    // Translate so that image borders are in the first quadrant?
    float  x_ori, y_ori;
    // A = Rot*[ [0;h1] [w1;0] [w1;h1] [0;0] ];
    corners(0,0) = 0;   corners(0,1) = w-1;   corners(0,2) = w-1;   corners(0,3) = 0;
    corners(1,0) = 0;  corners(1,1) = 0;    corners(1,2) = h-1;   corners(1,3) = h-1;

    corners = Rot*corners;

    //x_ori = min(corners(1,:));
    //y_ori = min(corners(2,:));
    x_ori = corners(0,0); y_ori = corners(1,0);
    for(int i=1; i<4; i++)
    {
        if (x_ori>corners(0,i))
            x_ori = corners(0,i);

        if (y_ori>corners(1,i))
            y_ori = corners(1,i);
    }

    // translation and Tilt T_t
    x = x - x_ori + 1;
    y = (y - y_ori)/t + 1;
}


/**
 * @brief Computes image coordinates \f$(x,y) \in [1,n_x]\times[1,n_y]\f$ from tilted coordinates \f$T_tR_\theta(x,y) \in [1,m_x]\times[1,m_y]\f$
 * @param (x,y) Enters image coordinates and returns tilted coordinates
 * @param w Width from the original image (\f$ w = n_x \f$)
 * @param h Width from the original image (\f$ h = n_y \f$)
 * @param t Tilt that was applied in the y direction
 * @param theta Direction \f$\theta\f$ in which the tilt that was applied
 * @return True if the computed image-coordinates fall inside the true image boundaries.
 * @author Mariano Rodríguez
 */
bool tiltedcoor2imagecoor(float& x, float& y, const int& w, const int& h, const float& threshold , const float& t, const float& theta)
{

    // Get initial translation \tau_{x_ori,y_ori}

    // Prepare rotation matrix
    libNumerics::matrix<float> Rot(2,2), vec(2,1), corners(2,4);
    // Rot = [cos(Rtheta) -sin(Rtheta);sin(Rtheta) cos(Rtheta)];
    Rot(0,0) = cos(theta); Rot(0,1) = sin(theta);
    Rot(1,0) = -sin(theta); Rot(1,1) = cos(theta);

    // Translate so that image borders are in the first quadrant?
    float  x_ori, y_ori;
    // A = Rot*[ [0;h-1] [w-1;0] [w-1;h-1] [0;0] ];
    corners(0,0) = 0;   corners(0,1) = w-1;   corners(0,2) = w-1;   corners(0,3) = 0;
    corners(1,0) = 0;  corners(1,1) = 0;    corners(1,2) = h-1;   corners(1,3) = h-1;
    corners = Rot*corners;

    //x_ori = min(corners(1,:));
    //y_ori = min(corners(2,:));
    x_ori = corners(0,0); y_ori = corners(1,0);
    for(int i=1; i<4; i++)
    {
        if (x_ori>corners(0,i))
            x_ori = corners(0,i);

        if (y_ori>corners(1,i))
            y_ori = corners(1,i);
    }

    // Distance from the point to the true image borders
    // if less than threshold stop computations and return false
    float xvec[5], yvec[5], d;
    for(int i=0; i<4; i++)
    {// translation and Tilt T_t
        xvec[i] = corners(0,i) - x_ori + 1;
        yvec[i] = (corners(1,i) - y_ori)/t + 1;
        //cout<<"xvec="<<xvec[i]<< " yvec="<<yvec[i]<<endl;
    }
    xvec[4] = xvec[0];
    yvec[4] = yvec[0];
    //cout<<"xvec="<<xvec[4]<< " yvec="<<yvec[4]<<endl;

    for(int i=0; i<4; i++)
    {
        d = std::abs( (yvec[i+1]-yvec[i])*x - (xvec[i+1]-xvec[i])*y + xvec[i+1]*yvec[i] - yvec[i+1]*xvec[i] ) / sqrt( pow(xvec[i+1]-xvec[i],2) + pow(yvec[i+1]-yvec[i],2) );
        //cout<<"d="<<d<<endl;
        if (d <= threshold)
            return(false);
    }


    // Inverse of Tilt T_t and translation
    x = (x-1)   + x_ori;
    y = (y-1)*t + y_ori;

    vec(0,0) = x;
    vec(1,0) = y;

    // rotation -> [x1;y1] = Rot^{-1}*[x1;y1]
    Rot(1,0) = -Rot(1,0); Rot(0,1) = -Rot(0,1);
    vec = (Rot*vec);
    x = vec(0,0) + 1;
    y = vec(1,0) + 1;

    if ( (x>w)||(y>h)||(x<1)||(y<1) )
        return(false);
    else
        return(true);
}


/**
 * @brief Each SIIM (Scale Invariant Image Matching) descriptor in <keys> is added to its corresponding hyper-descriptor. Already created hyper-descriptors are stored in <mapKP>.
 * If a SIIM keypoint doesn't correponds to a hyper-keypoint in <mapKP> then it is created.
 * @param keys List of SIIM descriptors to be added
 * @param mapKP A map to already created hyper-descriptors. For each pixel in the image there is possible a hyper-descriptor,
 * so the size of the image is equal to the size of the map.
 * @param width mapKP width
 * @param height mapKP height
 * @author Mariano Rodríguez
 */
void Add_IMAS_KP(IMAS_keypointlist& keys, std::vector<IMAS::IMAS_KeyPoint*>& mapKP, int width, int height)
{
    float x,y,t,theta;
    int xr,yr;
    bool only_center;
    for(int i=0; i<(int)keys.size();i++)
    {
        x = keys[i].pt.x;
        xr = (int) round(x);

        y = keys[i].pt.y;
        yr = (int) round(y);

        t = keys[i].t;
        theta = keys[i].theta;

        int ind =  yr*width + xr, newind;
        newind = ind;
        if ( mapKP[ind]==0 )
        {
            // create new imas element
            mapKP[ind] = new IMAS::IMAS_KeyPoint();
            mapKP[ind]->x = x;
            mapKP[ind]->y = y;
            mapKP[ind]->sum_x = x;
            mapKP[ind]->sum_y = y;
            mapKP[ind]->KPvec.push_back(keys[i]);
        }
        else
        {
            mapKP[ind]->KPvec.push_back(keys[i]);
            mapKP[ind]->sum_x += x;
            mapKP[ind]->sum_y += y;
            mapKP[ind]->x = mapKP[ind]->sum_x / mapKP[ind]->KPvec.size();
            mapKP[ind]->y = mapKP[ind]->sum_y / mapKP[ind]->KPvec.size();

        }

        only_center = false;
        int r = rho;
        while(!only_center)
        {
            only_center = true;
            for (int xi = xr-r; xi<=xr+r; xi++)
                for (int yi = yr-r; yi<=yr+r; yi++)
                {
                    if (( sqrt(pow(xi-xr,2) + pow(yi-yr,2))<=r )&&(xi>0)&&(xi<width)&&(yi>0)&&(yi<height))
                    {
                        int indi = yi*width + xi;
                        if ( (mapKP[indi]!=0)&&(indi!=ind) )
                        {
                            //merge indi to ind
                            only_center = false;
                            mapKP[ind]->sum_x += mapKP[indi]->sum_x;
                            mapKP[ind]->sum_y += mapKP[indi]->sum_y;

                            for (int k=0;k<(int)mapKP[indi]->KPvec.size();k++)
                            {
                                mapKP[ind]->KPvec.push_back(mapKP[indi]->KPvec[k]);
                            }
                            delete mapKP[indi];
                            mapKP[indi] = 0;

                            mapKP[ind]->x = mapKP[ind]->sum_x / mapKP[ind]->KPvec.size();
                            mapKP[ind]->y = mapKP[ind]->sum_y / mapKP[ind]->KPvec.size();

                            //update newind
                            x = mapKP[ind]->x;
                            xr = (int) round(x);
                            y = mapKP[ind]->y;
                            yr = (int) round(y);
                            newind = yr*width + xr;

                            if (newind!=ind)
                            {
                                if (mapKP[newind]==0)
                                {// newind empty
                                    mapKP[newind] = mapKP[ind];
                                    mapKP[ind] = 0;
                                    ind = newind;
                                }
                                else
                                { // newind not empty
                                    mapKP[newind]->sum_x += mapKP[ind]->sum_x;
                                    mapKP[newind]->sum_y += mapKP[ind]->sum_y;

                                    for (int k=0;k<(int)mapKP[ind]->KPvec.size();k++)
                                    {
                                        mapKP[newind]->KPvec.push_back(mapKP[ind]->KPvec[k]);
                                    }

                                    mapKP[newind]->x = mapKP[newind]->sum_x / mapKP[newind]->KPvec.size();
                                    mapKP[newind]->y = mapKP[newind]->sum_y / mapKP[newind]->KPvec.size();

                                    delete mapKP[ind];
                                    mapKP[ind] = 0;
                                    ind = newind;
                                }
                            }
                        }
                    }

                }
        }

    }

}




/**
 * @brief Computes all hyper-descriptors comming from a set of optical tilts digitally generated.
 * @param image Input image.
 * @param width Width of the input image.
 * @param height Height of the input image.
 * @param imasKP Returns a list of generalised keypoints.
 * @param simu_details Specifies the optical tilts that are to be simulated.
 * @param stats A vector with statistics on found generalised keypoints. Mean, min or max of SIIM keypoints over all found generalised keypoints.
 * @return The total number of generalised keypoints that have been found.
 * @author Mariano Rodríguez
 */
int IMAS_detectAndCompute(vector<float>& image, int width, int height,std::vector<IMAS::IMAS_KeyPoint*>& imasKP, const std::vector<tilt_simu>& simu_details,std::vector<float>& stats)
{
    std::vector<IMAS::IMAS_KeyPoint*> mapKP;
    mapKP.resize(width*height);

    for(int i=0;i<width*height;i++)
        mapKP[0]= 0;

    int num_tilt, tt;
    int num_keys_total=0;


    num_tilt = simu_details.size();
#pragma omp parallel
#pragma omp master
    {
        for (tt = 1; tt <= num_tilt; tt++)
        {
            float t = simu_details[tt-1].t;
            if ( t == 1 )  // it will ignore rotations for tilts=1 !!!
            {
#pragma omp task firstprivate(t) shared(image, mapKP)
                {
                    IMAS_keypointlist keys;
                    IMAS::IMAS_Matrix queryImg;
#pragma omp critical
                    vectorimage2imasimage(image, queryImg, width, height);

                    compute_local_descriptor_keypoints(queryImg,keys,t,0.0f);

                    //std::random_shuffle(keys.KeyList.begin(),keys.KeyList.end());
#pragma omp critical
                    Add_IMAS_KP(keys, mapKP, width,height);
                }

            }
            else
            {
                // The number of rotations to simulate under the current tilt.
                int num_rot1 = simu_details[tt-1].rots.size();

                // Loop on rotations.
                for ( int rr = 1; rr <= num_rot1; rr++ )
                {
#pragma omp task firstprivate(tt,rr,t,width,height) shared(mapKP,image)
                    {

                        float theta = simu_details[tt-1].rots[rr-1];
                        theta = theta * 180 / M_PI;

                        /* Anti-aliasing filtering along vertical direction */

                        // Mariano Rodríguez ( 07/02/2017 )
                        float sigma = 0.8 * sqrt( pow(t,2) - 1.0f ); /* As the optical tilt */
                        vector<float> image_tmp;
                        int width_t, height_t;

                        // simulate digital tilt: rotate and subsample the image along the vertical axis by a factor of t.
                        simulate_digital_tilt(image,width,height,image_tmp, width_t,height_t,theta,t,sigma);

                        IMAS::IMAS_Matrix queryImg;
                        vectorimage2imasimage(image_tmp, queryImg, width_t, height_t);

                        // compute keypoint descriptors on simulated images.
                        IMAS_keypointlist keypoints, keys;
                        IMAS_keypointlist* keypoints_filtered = &(keys);
                        keys.clear();


                        compute_local_descriptor_keypoints(queryImg,(keypoints),t,theta);


                        /* check if the keypoint is located on the boundary of the parallelogram (i.e., the boundary of the distorted input image). If so, remove it to avoid boundary artifacts. */
                        if ( keypoints.size() != 0 )
                        {
                            for ( int cc = 0; cc < (int) keypoints.size(); cc++ )
                            {

                                float x0, y0, BorderTh;

                                x0 = keypoints[cc].pt.x;
                                y0 = keypoints[cc].pt.y;

                                //Keep the descriptor off the border... BorderTh = diagonal length of the descriptor
                                BorderTh = keypoints[cc].size;

                                if (tiltedcoor2imagecoor(x0, y0, width, height,BorderTh, t, theta* M_PI / 180))
                                {
                                    // Normalize the coordinates of the matched points by compensate the simulate affine transformations
                                    keypoints[cc].pt.x = x0;
                                    keypoints[cc].pt.y = y0;

                                    keypoints_filtered->push_back(keypoints[cc]);

                                }

                            }

                            //std::random_shuffle(keys.KeyList.begin(),keys.KeyList.end());
#pragma omp critical
                            Add_IMAS_KP(keys, mapKP, width,height);
                        }
                    }
                }// end of for loop on rotation
            }
        } // end of foor loop on tilts
    }

    // save in imasKP and do stats
    int num_max = 0, num_min = 500000, total = 0;
    float num_mean = 0;
    for (int i = 0; i < (int) mapKP.size(); i++)
        if (mapKP[i]!=0)
        {
            imasKP.push_back(mapKP[i]);
            total +=mapKP[i]->KPvec.size();
            num_keys_total += 1;//(int) mapKP[i]->KPvec.size();
            if (num_max<(int)mapKP[i]->KPvec.size())
                num_max = mapKP[i]->KPvec.size();
            if (num_min>(int)mapKP[i]->KPvec.size())
                num_min = mapKP[i]->KPvec.size();
            num_mean +=mapKP[i]->KPvec.size();
        }
    num_mean = num_mean/num_keys_total;
    stats.push_back((float)total);
    stats.push_back((float)num_min);
    stats.push_back(num_mean);
    stats.push_back((float)num_max);

    return num_keys_total;
}




//************************************ IMAS Implementation

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
void IMAS_Impl(vector<float>& ipixels1, int w1, int h1, vector<float>& ipixels2, int w2, int h2, vector<float>& data, matchingslist& matchings,imasCoverings& ic, int applyfilter)
{

    ///// Compute IMAS keypoints
    std::vector<IMAS::IMAS_KeyPoint*> keys1;
    std::vector<IMAS::IMAS_KeyPoint*> keys2;
    keys1.clear();
    keys2.clear();

    int num_keys1=0, num_keys2=0;


    my_Printf("IMAS-Detector with %s...\n",desc_name.c_str());

    IMAS_time tstart = IMAS::IMAS_getTickCount();

    _arearatio = ic.getAreaRatio();

    std::vector<float> stats1,stats2;
    num_keys1 = IMAS_detectAndCompute(ipixels1, w1, h1, keys1, ic.getSimuDetails1(),stats1);
    num_keys2 = IMAS_detectAndCompute(ipixels2, w2, h2, keys2, ic.getSimuDetails2(),stats2);

    my_Printf("   %d hyper-descriptors from %d SIIM descriptors have been found in %d simulated versions of image 1\n", num_keys1,(int)stats1[0],ic.getTotSimu1());
    my_Printf("      stats: group_min = %d , group_mean = %.3f, group_max = %d\n",(int)stats1[1],stats1[2],(int)stats1[3]);

    my_Printf("   %d hyper-descriptors from %d SIIM descriptors have been found in %d simulated versions of image 2\n", num_keys2,(int)stats2[0],ic.getTotSimu2());
    my_Printf("      stats: group_min = %d , group_mean = %.3f, group_max = %d\n",(int)stats2[1],stats2[2],(int)stats2[3]);

    my_Printf("IMAS-Detector accomplished in %.2f seconds.\n \n", (IMAS::IMAS_getTickCount() - tstart)/ IMAS::IMAS_getTickFrequency());


    IMAS_matcher(w1, h1, w2, h2, keys1, keys2, matchings, applyfilter);



    /* Generate data matrix: matchinglist and Minfoall */
    for ( int i = 0; i < (int) matchings.size(); i++ )
    {
        matching *ptr_in = &(matchings[i]);
        // there are 14 rows of info
        data.push_back(ptr_in->first.x); //x1_in
        data.push_back(ptr_in->first.y); //y1_in
        data.push_back(ptr_in->first.scale); //s1_in
        data.push_back(ptr_in->first.angle); //a1_in
        data.push_back(1); //t1
        data.push_back(ptr_in->first.t); //t2
        data.push_back(ptr_in->first.theta); //theta

        data.push_back(ptr_in->second.x); //x2_in
        data.push_back(ptr_in->second.y); //y2_in
        data.push_back(ptr_in->second.scale); //s2_in
        data.push_back(ptr_in->second.angle); //a2_in
        data.push_back(1); //t_im2_1
        data.push_back(ptr_in->second.t); //t_im2_2
        data.push_back(ptr_in->second.theta); //theta2
    }


    my_Printf("Done.\n\n");
    keys1.clear();
    keys2.clear();
    ipixels1.clear();
    ipixels2.clear();

}
