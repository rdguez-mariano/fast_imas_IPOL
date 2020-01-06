/**
  * @file main.cpp
  * @author Mariano Rodríguez
  * @date 2018
  * @brief A caller for IMAS
  */

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include "mex_and_omp.h"
#include "imas.h"
#include "io_png/io_png.h"
#include "libNumerics/numerics.h"
#include "library.h"

#ifdef _GDAL
#include <gdal/gdal.h>
#include <gdal/cpl_conv.h>
#endif

float framewidth = 100;

void invert_contrast(std::vector<float>& image,int w, int h)
{    for (int i=0;i<h;i++)
        for (int j=0;j<w;j++)
        {
            image[j*h+i] = (float)(255 - image[j*h+i]);
        }
}



// Grow rectangle of corners (x0,y0) and (x1,y1) to include (x,y)
void growTo(float& x0, float& y0, float& x1, float& y1, float x, float y)
{
    if(x<x0) x0=x;
    if(x>x1) x1=x;
    if(y<y0) y0=y;
    if(y>y1) y1=y;
}


// Panorama construction
// IPOL demo modification by Mariano Rodríguez on the 28th September, 2018.
// Panorama is now reconstructed by default in a frame around the target image in order to avoid the creation of big images when strong homographies are present.
/**
 * @brief Panorama Construction
 * @author Pascal Monasse, Mariano Rodríguez
 */
void panorama(std::vector<float>& I1,int w1, int h1,std::vector<float>& I2, int w2, int h2, libNumerics::matrix<float> H, bool aroundI2)
{
    std::vector<float> I;
    int h, w;
    libNumerics::matrix<float> v(3,1);
    float x0=0, y0=0, x1=(float)w2, y1=(float)h2;

    if (aroundI2)
    {
        x0=-framewidth;
        y0=-framewidth;
        x1+=framewidth;
        y1+=framewidth;
    }
    else
    {
        v(0,0)=0; v(1,0)=0; v(2,0)=1;
        v=H*v; v/=v(2,0);
        growTo(x0, y0, x1, y1, v(0,0), v(1,0));

        v(0,0)=(float)w1; v(1,0)=0; v(2,0)=1;
        v=H*v; v/=v(2,0);
        growTo(x0, y0, x1, y1, v(0,0), v(1,0));

        v(0,0)=(float)w1; v(1,0)=(float)h1; v(2,0)=1;
        v=H*v; v/=v(2,0);
        growTo(x0, y0, x1, y1, v(0,0), v(1,0));

        v(0,0)=0; v(1,0)=(float)h1; v(2,0)=1;
        v=H*v; v/=v(2,0);
        growTo(x0, y0, x1, y1, v(0,0), v(1,0));
    }

    w = int(x1-x0);
    h = int(y1-y0);
    I.resize(h*w);

    for (int i =0; i<w*h;i++)
        I[i] = 255;

    H = H.inv(); // Pull from image I1
    for(int i=0; i<h; i++)
        for(int j=0; j<w; j++) {
            v(0,0) = j+x0; v(1,0) = i+y0; v(2,0) = 1;
            bool in=(0<=v(0,0) && round(v(0,0))<w2 && 0<=v(1,0) && round(v(1,0))<h2);
            if(in)
                I[i*w + j] = I2[ (int)round(v(0,0)) + (int)round(v(1,0))*w2 ];
            v = H*v;
            v /= v(2,0);
            if(0<=v(0,0) && round(v(0,0))<w1 && 0<=v(1,0) && round(v(1,0))<h1) {
                if(in) {
                    float vtemp = I1[ (int)round(v(0,0)) + (int)round(v(1,0))*w1 ];
                    I[i*w + j] = ( I[i*w + j] + vtemp )/2;
                } else
                {
                    I[i*w + j] = I1[ (int)round(v(0,0)) + (int)round(v(1,0))*w1 ];
                }
            }
        }

    float * rgb = new float[w*h];
    for(int i = 0; i < (int) h*w; i++)
        rgb[i] = I[i];

    write_png_f32("panorama.png", rgb, w, h, 1);
}

/**
 * @brief write_example_parallelograms
 * @author Mariano Rodríguez
 */
void write_example_parallelograms(std::vector<float>& ipixels1,int w1, int h1,std::vector<keypoint_simple>& matchings)
{
    int wo =  w1;
    int ho = h1;

    std::vector<float *> opixelsIMAS_rich;
    for(int c=0;c<3;c++)
    {
        opixelsIMAS_rich.push_back(new float[wo*ho]);
    }

    for(int c=0;c<3;c++)
        for(int j = 0; j < (int) h1; j++)
            for(int i = 0; i < (int) w1; i++)
            {
                opixelsIMAS_rich[c][j*wo+i] = ipixels1[j*w1+i];
            }


    //////////////////////////////////////////////////////////////////// Draw parallelograms
    float* colorlines = new float[3], *colordesc = new float[3];
    colorlines[0] = 250.0f;colorlines[1] = 1.0f; colorlines[2] = 1.0f;
    colordesc[0] = 1.0f;colordesc[1] = 250.0f; colordesc[2] = 1.0f;
    float value;

    for(int i=0; i < (int) matchings.size(); i++)
    {
        srand(i*6546);
        for(int c=0;c<3;c++)
        {
            value =  (float)(rand() % 150 + 50);
            /* DRAWING RICH KEYPOINTS */
            //draw_line(opixelsIMAS_rich[c],  round(matchings[i].first.x), round(matchings[i].first.y), round(matchings[i].second.x), round(matchings[i].second.y) + h1 + band_w, colorlines[c], wo, ho);
            draw_square_affine(opixelsIMAS_rich[c],wo,ho, matchings[i].x, matchings[i].y, matchings[i].angle, matchings[i].scale, matchings[i].t, 1.0f, matchings[i].theta*M_PI/180, value);//colordesc[c]


            draw_square(opixelsIMAS_rich[c],10+5*i,10,2,2,value,w1,h1);

            int pointradius = 2;
            int x = matchings[i].x, y = matchings[i].y;
            for(int x0 = -pointradius; x0<=pointradius; x0++)
                for(int y0 = -pointradius; y0<=pointradius; y0++)
                {
                    if(sqrt(pow(x0,2)+pow(y0,2))<=pointradius)
                    {
                        for(int c=0;c<3;c++)
                            opixelsIMAS_rich[c][(y-y0)*wo+(x-x0)] = 0;
                    }
                }
        }
    }


    float * rgb = new float[wo*ho*3];
    for(int c=0;c<3;c++)
        for(int j = 0; j < (int) ho; j++)
            for(int i = 0; i < (int) wo; i++)
                rgb[j*wo+i+c*(wo*ho)] = opixelsIMAS_rich[c][j*wo+i];
    write_png_f32("output_vert_rich.png", rgb, wo, ho, 3);

    for(int c=0;c<3;c++)
    {
        delete[] opixelsIMAS_rich[c]; /*memcheck*/
    }
    delete[] rgb;
}


/**
 * @brief Writes corresponding matches between two images
 * @author Mariano Rodríguez
 */
void write_images_matches(std::vector<float>& ipixels1,int w1, int h1,std::vector<float>& ipixels2, int w2, int h2,matchingslist& matchings)
{

    int sq = 2;
    ///////////////// Output image containing line matches (the two images are concatenated one above the other)
    int band_w = 20; // insert a black band of width band_w between the two images for better visibility

    int wo =  MAX(w1,w2);
    int ho = h1+h2+band_w;

    std::vector<float *> opixelsIMAS, opixelsIMAS_rich;
    for(int c=0;c<3;c++)
    {
        opixelsIMAS.push_back(new float[wo*ho]);
        opixelsIMAS_rich.push_back(new float[wo*ho]);
    }

    for(int c=0;c<3;c++)
        for(int j = 0; j < (int) ho; j++)
            for(int i = 0; i < (int) wo; i++)
            {
                opixelsIMAS[c][j*wo+i] = 255;
                opixelsIMAS_rich[c][j*wo+i] = 255;
            }

    /////////////////////////////////////////////////////////////////// Copy both images to output
    for(int c=0;c<3;c++)
        for(int j = 0; j < (int) h1; j++)
            for(int i = 0; i < (int) w1; i++)
            {
                opixelsIMAS[c][j*wo+i] = ipixels1[j*w1+i];
                opixelsIMAS_rich[c][j*wo+i] = ipixels1[j*w1+i];
            }

    for(int c=0;c<3;c++)
        for(int j = 0; j < (int) h2; j++)
            for(int i = 0; i < (int) (int)w2; i++)
            {
                opixelsIMAS[c][(h1 + band_w + j)*wo + i] = ipixels2[j*w2 + i];
                opixelsIMAS_rich[c][(h1 + band_w + j)*wo + i] = ipixels2[j*w2 + i];
            }

    //////////////////////////////////////////////////////////////////// Draw matches
    float* colorlines = new float[3], *colordesc = new float[3];
    colorlines[0] = 250.0f;colorlines[1] = 1.0f; colorlines[2] = 1.0f;
    colordesc[0] = 1.0f;colordesc[1] = 250.0f; colordesc[2] = 1.0f;
    float value;
    for(int i=0; i < (int) matchings.size(); i++)
        for(int c=0;c<3;c++)
        {
            /* DRAWING SQUARES */
            value =  (float)(rand() % 150 + 50);
            draw_line(opixelsIMAS[c],  round(matchings[i].first.x), round(matchings[i].first.y),
                      round(matchings[i].second.x), round(matchings[i].second.y) + h1 + band_w, value, wo, ho);

            draw_square(opixelsIMAS[c],  round(matchings[i].first.x)-sq, round(matchings[i].first.y)-sq, 2*sq, 2*sq, value, wo, ho);
            draw_square(opixelsIMAS[c],  round(matchings[i].second.x)-sq, round(matchings[i].second.y) + h1 + band_w-sq, 2*sq, 2*sq, value, wo, ho);

            /* DRAWING RICH KEYPOINTS */
            //draw_line(opixelsIMAS_rich[c],  round(matchings[i].first.x), round(matchings[i].first.y), round(matchings[i].second.x), round(matchings[i].second.y) + h1 + band_w, colorlines[c], wo, ho);
            draw_circle_affine(opixelsIMAS_rich[c],wo,ho, matchings[i].first.x, matchings[i].first.y, matchings[i].first.angle, matchings[i].first.scale, matchings[i].first.t, 1.0f, matchings[i].first.theta*M_PI/180, colordesc[c]);
            draw_circle_affine(opixelsIMAS_rich[c],wo,ho, matchings[i].second.x, matchings[i].second.y + h1 + band_w, matchings[i].second.angle, matchings[i].second.scale, matchings[i].second.t, 1.0f, matchings[i].second.theta*M_PI/180, colordesc[c]);
        }

    float * rgb = new float[wo*ho*3];
    for(int c=0;c<3;c++)
        for(int j = 0; j < (int) ho; j++)
            for(int i = 0; i < (int) wo; i++)
                rgb[j*wo+i+c*(wo*ho)] = opixelsIMAS[c][j*wo+i];
    write_png_f32("output_vert.png", rgb, wo, ho, 3);
    for(int c=0;c<3;c++)
        for(int j = 0; j < (int) ho; j++)
            for(int i = 0; i < (int) wo; i++)
                rgb[j*wo+i+c*(wo*ho)] = opixelsIMAS_rich[c][j*wo+i];
    write_png_f32("output_vert_rich.png", rgb, wo, ho, 3);

    for(int c=0;c<3;c++)
    {
        delete[] opixelsIMAS[c]; /*memcheck*/
        delete[] opixelsIMAS_rich[c]; /*memcheck*/
    }



    /////////// Output image containing line matches (the two images are concatenated one aside the other)
    int woH =  w1+w2+band_w;
    int hoH = MAX(h1,h2);

    std::vector<float *> opixelsIMAS_H, opixelsIMAS_H_rich;
    for(int c=0;c<3;c++)
    {
        opixelsIMAS_H.push_back(new float[woH*hoH]);
        opixelsIMAS_H_rich.push_back(new float[woH*hoH]);
    }

    for(int c=0;c<3;c++)
        for(int j = 0; j < (int) hoH; j++)
            for(int i = 0; i < (int) woH; i++)
            {
                opixelsIMAS_H[c][j*woH+i] = 255;
                opixelsIMAS_H_rich[c][j*woH+i] = 255;
            }

    /////////////////////////////////////////////////////////////////// Copy both images to output
    for(int c=0;c<3;c++)
        for(int j = 0; j < (int) h1; j++)
            for(int i = 0; i < (int) w1; i++)
            {
                opixelsIMAS_H[c][j*woH+i] = ipixels1[j*w1+i];
                opixelsIMAS_H_rich[c][j*woH+i] = ipixels1[j*w1+i];
            }

    for(int c=0;c<3;c++)
        for(int j = 0; j < (int) h2; j++)
            for(int i = 0; i < (int) w2; i++)
            {
                opixelsIMAS_H[c][j*woH + w1 + band_w + i] = ipixels2[j*w2 + i];
                opixelsIMAS_H_rich[c][j*woH + w1 + band_w + i] = ipixels2[j*w2 + i];
            }

    //////////////////////////////////////////////////////////////////// Draw matches

    for(int i=0; i < (int) matchings.size(); i++)
        for(int c=0;c<3;c++)
        {
            /* DRAWING SQUARES */
            value =  (float)(rand() % 150 + 50);

            draw_line(opixelsIMAS_H[c],  round(matchings[i].first.x), round(matchings[i].first.y),
                      round(matchings[i].second.x) + w1 + band_w, round(matchings[i].second.y), value, woH, hoH);

            draw_square(opixelsIMAS_H[c],  round(matchings[i].first.x)-sq, round(matchings[i].first.y)-sq, 2*sq, 2*sq, value, woH, hoH);
            draw_square(opixelsIMAS_H[c],  round(matchings[i].second.x) + w1 + band_w-sq, round(matchings[i].second.y)-sq, 2*sq, 2*sq, value, woH, hoH);

            /* DRAWING RICH KEYPOINTS */
            //draw_line(opixelsIMAS_H_rich[c],  round(matchings[i].first.x), round(matchings[i].first.y),round(matchings[i].second.x) + w1 + band_w, round(matchings[i].second.y), colorlines[c], woH, hoH);
            draw_circle_affine(opixelsIMAS_H_rich[c],woH,hoH, matchings[i].first.x, matchings[i].first.y, matchings[i].first.angle, matchings[i].first.scale, matchings[i].first.t, 1.0f, matchings[i].first.theta*M_PI/180, colordesc[c]);
            draw_circle_affine(opixelsIMAS_H_rich[c],woH,hoH, matchings[i].second.x + w1 + band_w, matchings[i].second.y, matchings[i].second.angle, matchings[i].second.scale,matchings[i].second.t, 1.0f, matchings[i].second.theta*M_PI/180, colordesc[c]);
        }

    delete[] rgb;
    rgb = new float[woH*hoH*3];
    for(int c=0;c<3;c++)
        for(int j = 0; j < (int) hoH; j++)
            for(int i = 0; i < (int) woH; i++)
                rgb[j*woH+i+c*(woH*hoH)] = opixelsIMAS_H[c][j*woH+i];

    write_png_f32("output_hori.png", rgb, woH, hoH, 3);

    for(int c=0;c<3;c++)
        for(int j = 0; j < (int) hoH; j++)
            for(int i = 0; i < (int) woH; i++)
                rgb[j*woH+i+c*(woH*hoH)] = opixelsIMAS_H_rich[c][j*woH+i];

    write_png_f32("output_hori_rich.png", rgb, woH, hoH, 3);


    delete[] rgb;
    for(int c=0;c<3;c++)
    {
        delete[] opixelsIMAS_H[c]; /*memcheck*/
        delete[] opixelsIMAS_H_rich[c]; /*memcheck*/
    }
}


#include "libSimuTilts/digital_tilt.h"
#include "libSimuTilts/fproj.h"
/**
 * @brief Resizes an image to keep the same area as areaS.
 * @author Guoshen Yu
 */
void areazoom_image(vector<float>& ipixels, size_t& w1, size_t& h1, float areaS)
{
    //float areaS = wS * hS;

    float zoom1=0;
    int wS1=0, hS1=0;
    vector<float> ipixels_temp(ipixels);
    float InitSigma_aa = 1.6;


    float fproj_p, fproj_bg;
    char fproj_i;
    float *fproj_x4, *fproj_y4;
    int fproj_o;

    fproj_o = 3;
    fproj_p = 0;
    fproj_i = 0;
    fproj_bg = 0;
    fproj_x4 = 0;
    fproj_y4 = 0;



    // Resize image 1
    float area1 = w1 * h1;
    zoom1 = sqrt(area1/areaS);

    wS1 = (int) (w1 / zoom1);
    hS1 = (int) (h1 / zoom1);

    int fproj_sx = wS1;
    int fproj_sy = hS1;

    float fproj_x1 = 0;
    float fproj_y1 = 0;
    float fproj_x2 = wS1;
    float fproj_y2 = 0;
    float fproj_x3 = 0;
    float fproj_y3 = hS1;

    /* Anti-aliasing filtering along vertical direction */
    if ( zoom1 > 1 )
    {
        float sigma_aa = InitSigma_aa * zoom1 / 2;
        GaussianBlur1D(ipixels_temp,w1,h1,sigma_aa,1);
        GaussianBlur1D(ipixels_temp,w1,h1,sigma_aa,0);
    }

    // simulate a tilt: subsample the image along the vertical axis by a factor of t.
    ipixels.resize(wS1*hS1);
    fproj (ipixels_temp, ipixels , w1, h1, &fproj_sx, &fproj_sy, &fproj_bg, &fproj_o, &fproj_p,
           &fproj_i , fproj_x1 , fproj_y1 , fproj_x2 , fproj_y2 , fproj_x3 , fproj_y3, fproj_x4, fproj_y4);

    w1 = wS1;
    h1 = hS1;
}

#include <map>
#include <string>
#include <iostream>
enum StringValue { _wrongvalue,_im1, _im2,_im3,_max_keys_im3,_im3_only, _applyfilter, _IMAS_INDEX, _covering,_match_ratio, _filter_precision, _eigen_threshold, _tensor_eigen_threshold, _filter_radius, _fixed_area,_im1_gdal, _im2_gdal, _bigpanorama, _framewidth};
static std::map<std::string, int> strmap;
void buildmap()
{
    strmap["wrongvalue"] = _wrongvalue;
    strmap["-im1"] = _im1;
    strmap["-im2"] = _im2;
    strmap["-im1_gdal"] = _im1_gdal;
    strmap["-im2_gdal"] = _im2_gdal;
    strmap["-im3"] = _im3;
    strmap["-max_keys_im3"] = _max_keys_im3;
    strmap["-im3_only"] = _im3_only;
    strmap["-applyfilter"] = _applyfilter;
    strmap["-desc"] = _IMAS_INDEX;
    strmap["-covering"] = _covering;
    strmap["-match_ratio"] = _match_ratio;
    strmap["-filter_precision"] = _filter_precision;
    strmap["-eigen_threshold"] = _eigen_threshold;
    strmap["-tensor_eigen_threshold"] = _tensor_eigen_threshold;
    strmap["-filter_radius"] = _filter_radius;
    strmap["-fixed_area"] = _fixed_area;
    strmap["-bigpanorama"] = _bigpanorama;
    strmap["-framewidth"] = _framewidth;


}

void get_arguments(int argc, char **argv, std::vector<float>& im1,size_t& w1,size_t& h1, std::vector<float>& im2,size_t& w2, size_t& h2,std::vector<float>& im3,size_t& w3, size_t& h3, int& applyfilter, int& IMAS_INDEX, float& covering,float& matchratio, float& edge_thres, float& tensor_thres, bool& fixed_area, bool& aroundI2)
{
    int count = 1;
    buildmap();
    while (count<argc)
    {
        string s(argv[count++]);
        //cout<<s<<" = "<<argv[count]<< endl;
        switch (strmap[s])
        {
        case _wrongvalue:
        {
            cout<<"unidentified: "<<s<<" = "<<argv[count]<<endl;
            break;
        }
        case _fixed_area:
        {
            fixed_area = true;
            count--;
            break;
        }
        case _bigpanorama:
        {
            aroundI2 = false;
            count--;
            break;
        }
        case _im3:
        {
            float * iarr1;
            if (NULL == (iarr1 = read_png_f32_gray(argv[count], &w3, &h3)))
            {
                std::cout << "**** a-contrario image not found **** " << std::endl;
            }
            else
            {
                im3 = *new vector<float>(iarr1, iarr1 + w3 * h3);
                free(iarr1); /*memcheck*/
            }
            break;
        }
        case _im1:
        {
            float * iarr1;
            if (NULL == (iarr1 = read_png_f32_gray(argv[count], &w1, &h1))) {
                std::cerr << "Unable to load image file " << argv[count] << std::endl;
            }
            im1 = *new vector<float>(iarr1, iarr1 + w1 * h1);
            free(iarr1); /*memcheck*/
            break;
        }
        case _im2:
        {
            // Read image2
            float * iarr2;
            if (NULL == (iarr2 = read_png_f32_gray(argv[count], &w2, &h2))) {
                std::cerr << "Unable to load image file " << argv[count] << std::endl;
            }
            std::vector<float> ipixels2(iarr2, iarr2 + w2 * h2);
            free(iarr2); /*memcheck*/
            im2 = ipixels2;
            break;
        }
        case _im1_gdal:
        {
#ifdef _GDAL
            GDALDatasetH  hDataset;
            GDALAllRegister();

            hDataset = GDALOpen( argv[count++], GA_ReadOnly );

//            int width = GDALGetRasterXSize(hDataset);
//            int height = GDALGetRasterYSize(hDataset);
            int bands = GDALGetRasterCount(hDataset);

            int xoff = atoi(argv[count++]);//23000;
            int yoff = atoi(argv[count++]);//5000;
            w1 = atoi(argv[count++]);
            h1 = atoi(argv[count]);


            float * iarr1 = (float *) CPLMalloc(sizeof(float)*w1*h1*bands);
            GDALDatasetRasterIO( hDataset, GF_Read,xoff,yoff, w1, h1,
                                 iarr1, w1, h1, GDT_Float32,
                                 bands, NULL, 0,0,0 );
            GDALClose( hDataset );

            //Normalise
            float max = iarr1[0];
            for (int i =1;i<w1*h1*bands;i++)
                if (max<iarr1[i])
                    max = iarr1[i];
            for (int i =0;i<w1*h1*bands;i++)
            {
                iarr1[i] = 255.0*iarr1[i]/max;
            }

            im1 = *new vector<float>(iarr1, iarr1 + w1 * h1);

            free(iarr1); /*memcheck*/
#else
     cerr<<"Error: CMAKE didn't include GDAL. Please turn on the proper flag in CMakeLists.txt"<<endl;
     count = count+4;
     w1=0;
     h1=0;
#endif
            break;
        }
        case _im2_gdal:
        {
#ifdef _GDAL
            GDALDatasetH  hDataset;
            GDALAllRegister();

            hDataset = GDALOpen( argv[count++], GA_ReadOnly );

//            int width = GDALGetRasterXSize(hDataset);
//            int height = GDALGetRasterYSize(hDataset);
            int bands = GDALGetRasterCount(hDataset);

            int xoff = atoi(argv[count++]);//23000;
            int yoff = atoi(argv[count++]);//5000;
            w2 = atoi(argv[count++]);
            h2 = atoi(argv[count]);


            float * iarr2 = (float *) CPLMalloc(sizeof(float)*w2*h2*bands);
            GDALDatasetRasterIO( hDataset, GF_Read,xoff,yoff, w2, h2,
                                 iarr2, w2, h2, GDT_Float32,
                                 bands, NULL, 0,0,0 );
            GDALClose( hDataset );

            //Normalise
            float max = iarr2[0];
            for (int i =1;i<w2*h2*bands;i++)
                if (max<iarr2[i])
                    max = iarr2[i];
            for (int i =0;i<w2*h2*bands;i++)
            {
                iarr2[i] = 255.0*iarr2[i]/max;
            }

            im2 = *new vector<float>(iarr2, iarr2 + w2 * h2);

            free(iarr2); /*memcheck*/
#else
     cerr<<"Error: CMAKE didn't include GDAL. Please turn on the proper flag in CMakeLists.txt"<<endl;
     count = count+4;
     w2=0;
     h2=0;
#endif
            break;
        }
        case _filter_precision:
        {
            Filter_precision = atof(argv[count]);
            break;
        }
        case _framewidth:
        {
            framewidth = atof(argv[count]);
            break;
        }
        case _applyfilter:
        {
            applyfilter = atoi(argv[count]);
            switch (applyfilter) {
            case ORSA_FUNDAMENTAL:
            {
                Filter_num_min = 8;
                Filter_precision=3;
                break;
            }
            case ORSA_HOMOGRAPHY:
            {
                Filter_num_min = 5;
                Filter_precision=24;
                break;
            }
            case USAC_FUNDAMENTAL:
            {
                Filter_num_min = 8;
                Filter_precision=3;
                break;
            }
            case USAC_HOMOGRAPHY:
            {
                Filter_num_min = 5;
                Filter_precision=24;
                break;
            }
            case 0:
            {
                Filter_num_min = 0;
                break;
            }
            }
            break;
        }
        case _IMAS_INDEX:
        {
            IMAS_INDEX = atoi(argv[count]);
            break;
        }
        case _covering:
        {
            covering = atof(argv[count]);
            break;
        }
        case _match_ratio:
        {
            matchratio = atof(argv[count]);
            break;
        }
        case _filter_radius:
        {
            rho = atoi(argv[count]);
            break;
        }
        case _eigen_threshold:
        {
            edge_thres = atof(argv[count]) / pow( 1 + atof(argv[count]) ,2);
            break;
        }
        case _tensor_eigen_threshold:
        {
            tensor_thres = atof(argv[count]) / pow( 1 + atof(argv[count]) ,2);
            break;
        }
        }
        count++;
    }
}



int main(int argc, char **argv)
{
    int IMAS_INDEX = 11, applyfilter = 7;
    float covering = -1.0f, matchratio = -1.0f, edge_thres = -1.0f, tensor_thres = -1.0f;
    std::vector<float> ipixels1,ipixels2,ipixels3;
    size_t w1=0,h1=0,w2=0,h2=0,w3=-1,h3=-1;
    bool fixed_area = false, aroundI2 = true;
    get_arguments(argc,argv,ipixels1,w1,h1,ipixels2,w2,h2,ipixels3,w3,h3,applyfilter,IMAS_INDEX,covering,matchratio,edge_thres, tensor_thres, fixed_area,aroundI2);

    if(argc==1)
    {
        cout<<"Arguments Example:"<<endl;
        cout<<"-im1 PATH/im1.png -im2 PATH/im2.png -applyfilter 2 -desc 11"<<endl;
    }

    if ((int)h1*h2*w1*w2==0)
    {
        cout<<"Wrong input images !"<<endl;
        return 0;
    }


    if (fixed_area)
    {
        areazoom_image(ipixels1,w1,h1,800*600);
        areazoom_image(ipixels2,w2,h2,800*600);
        if (((int)w3>0)&&((int)h3>0))
            areazoom_image(ipixels3,w3,h3,800*600);
    }

    string algo_name = SetDetectorDescriptor(IMAS_INDEX);

    if (covering==-1.0f)
        covering = default_radius;

    if (covering>1.0f)
        algo_name ="Optimal-Affine-"+algo_name;

    imasCoverings ic;
    if (covering<0.0f && covering>-1.0)
    {
        algo_name = "AdOPT-Affine-"+algo_name;
        ic.loadsimulations2do();
    }
    else
        ic.loadsimulations2do(covering,default_radius,true);


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




    if (matchratio>0.0f)
        update_matchratio(matchratio);

#ifdef _NO_OPENCV
    if (edge_thres>0.0f)
        update_edge_threshold(edge_thres);

    if (tensor_thres>0.0f)
        update_tensor_threshold(tensor_thres);
#endif

    if ( ((int)w3>0)&&((int)h3>0) )
    {
        IMAS_time tstart = IMAS::IMAS_getTickCount();
        my_Printf("Computing A-contrario hyper-descriptors...\n");
        std::vector<float> stats3;
        int num_keys1 = IMAS_detectAndCompute(ipixels3, w3, h3, keys3, ic.getSimuDetails2(),stats3);

        my_Printf("   %d hyper-descriptors from %d SIIM descriptors have been found in %d simulated versions of the A-contrario image\n", num_keys1,(int)stats3[0],ic.getTotSimu1());
        my_Printf("      stats: group_min = %d , group_mean = %.3f, group_max = %d\n",(int)stats3[1],stats3[2],(int)stats3[3]);
        my_Printf("Computation of A-contrario hyper-descriptors accomplished in %.2f seconds.\n \n", (IMAS::IMAS_getTickCount() - tstart)/ IMAS::IMAS_getTickFrequency());


    }


    // IMAS
    matchingslist matchings;
    vector< float > data;
    IMAS_Impl(ipixels1, (int)w1, (int)h1, ipixels2, (int)w2, (int)h2, data, matchings,ic, applyfilter);

    write_images_matches(ipixels1,(int) w1, (int) h1, ipixels2, (int) w2, (int) h2, matchings);


    //write panorama if homography is selected
    if ((applyfilter==ORSA_HOMOGRAPHY || applyfilter == USAC_HOMOGRAPHY)&&(IdentifiedMaps.size()!=0))
    {
        libNumerics::matrix<double> H1 = IdentifiedMaps[0];
        libNumerics::matrix<float> H(3,3);
        for(int i=0; i<3; i++)
            for(int j=0; j<3; j++)
                H(i,j) = (float) H1(i,j);

        panorama(ipixels1,(int) w1, (int) h1, ipixels2, (int) w2, (int) h2, H, aroundI2);
    }

    //Output file "data_matches.csv"

    int wo = 15;
    ofstream myfile;
    myfile.open ("data_matches.csv", std::ofstream::out | std::ofstream::trunc);
    myfile<<"x1, y1, sigma1, angle1, t1_x, t1_y, theta1, x2, y2, sigma2, angle2, t2_x, t2_y, theta2, distance"<<endl;

    if (matchings.size()>0)
    {
        int cont =1;
        myfile << ((double) data[0]) << ",";

        for ( int i = 1; i < (int) (wo*matchings.size()); i++ )
        {
            if (cont ==(wo-1))
            {
                myfile << ((double) data[i]) << endl;
                cont = 0;
            }
            else
            {
                myfile << ((double) data[i]) << ",";
                cont = cont +1;
            }

        }
    }
    myfile.close();



    //Showing Results


    // Clear memory
    data.clear();
    matchings.clear();

    return 0;
}
