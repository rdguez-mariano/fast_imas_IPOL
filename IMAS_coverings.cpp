/**
  * @file IMAS_covering.cpp
  * @author Mariano Rodríguez
  * @date 2017
  * @brief Handles IMAS coverings: either coverings proposed in the literature or near optimal coverings as in \cite imas2017.
  */
#include "IMAS_coverings.h"
//#include<fstream>   //string, ifstream
#include <algorithm>    // std::random_shuffle, atoi
#include "math.h" // M_PI

using namespace std;

const char delimiter = ',';
const float point_radius = 2.0f; // for drawing points in the output image

#ifdef _NOMEX
#include "io_png/io_png.h"
#endif

/**
 * @brief Computes the transition tilt by the formula in \cite imas2017
 * @param (t,psi1) Denotes \f$ [T_t,R_{\psi_1}] \f$
 * @param (s,psi2) Denotes \f$ [T_s,R_{\psi_2}] \f$
 * @return \f$ r  \colon = \tau( T_t,R_{\psi_1} (T_s,R_{\psi_2})^{-1} ) \f$    where \f$r\f$ satisfies:
 * \f$ \frac{e^{2r}+1}{2e^{r}} = \left(\frac{\frac{t}{s}+\frac{s}{t}}{2}\right)\cos^{2}\left(\psi_{1}-\psi_{2}\right)+\left(\frac{\frac{1}{st}+st}{2}\right)\sin^{2}\left(\psi_{1}-\psi_{2}\right)\f$
 */
float imasCoverings::transition_tilt(float t, float psi1, float s, float psi2)
{
    float cos_2 = pow(cos(psi1-psi2),2);
    float g = ( pow(t/s,2) + 1 )*cos_2 + ( 1/pow(s,2) + pow(t,2) )*(1-cos_2);
    float G = (s/t)*g/2;
    return( G + sqrt(pow(G,2) - 1) );
}


/**
 * @brief Computes Area Ratio of a feasible set. Feasible sets are indexed by \f$(t_1, \phi_1, \cdots , t_n, \phi_n)\f$ where each t_i, \phi_i will generate the a set of affine maps like
 * \f$J_i \colon = \left\lbrace T_{t_{i}}R_{\phi_{i}},\,T_{t_{i}}R_{2\phi_{i}}\,...,\,T_{t_{i}}R_{\left\lfloor \frac{\pi}{\phi_{i}}\right\rfloor \phi_{i}} \right\rbrace \f$
 * @param tilt_element A Vector representing \f$(t_1, \phi_1, \cdots , t_n, \phi_n)\f$
 * @return \f$ 1 + \sum^{n}_{i=1} \frac{|J_i|}{t_i}\f$  where \f$|J_i|\f$ denotes the cardinal of the set \f$J_i\f$.
 */
float imasCoverings::Fcout(const vector<float>& tilt_element)
{
    float ret = 1.0f;
    for(int n=1;n<=tilt_element.size()/2;n++)
        ret += (trunc(M_PI/tilt_element[2*n-1])+1)/tilt_element[2*n-2];
    return (ret);
}




void imasCoverings::drawpoint(float * data, int width, int i, int j, float radius, float value)
{
    for (int x=-radius;x<=radius;x++)
        for (int y=-radius;y<=radius;y++)
        {
            if ((y+j>0)&&(x+i>0)&&(y+j<width)&&(x+i<width)&&(sqrt(x*x+y*y)<=radius))
            {
                data[(y+j)*width + x+i] = value;
            }
        }
}


void imasCoverings::getdisks(float t,float psi, float r, float phistep, std::vector<float>& phivec, std::vector<float>& tupvec,std::vector<float>& tlowvec)
{
    float beta = ( pow(r,2)+1 )/(2*r);
    float phi = 0;
    while(phi <2*M_PI)
    {

        float Gphi = pow(cos(psi-phi),2);
        float dis = pow(beta,2) - ( Gphi/t + t*(1-Gphi) )*( (1-Gphi)/t + t*Gphi );
        float tupper, tlower;
        if( dis>0 )
        {
            tupper = ( beta + sqrt(dis) )/( Gphi/t + (1-Gphi)*t );
            tlower = ( beta - sqrt(dis) )/( Gphi/t + (1-Gphi)*t );
            if (tupper>1)
            {
                phivec.push_back(phi);
                tupvec.push_back(tupper);
                if (tlower<1)
                    tlowvec.push_back( 1.0f);
                else
                    tlowvec.push_back(tlower);
            }
        }
        else
            if (dis==0)
            {
                tupper = beta/( Gphi/t + (1-Gphi)*t );
                if (tupper>1)
                {
                    phivec.push_back(phi);
                    tupvec.push_back(tupper);
                    tlowvec.push_back(tupper);
                }
            }
        phi += phistep;
    }
}

void imasCoverings::drawborder(float *data, std::vector<float> phivec, std::vector<float>& tvec,float seenregion, int logseenregionpixels)
{
    int i,j,wo = logseenregionpixels*2 +1; float x,y;
    for(int n=0;n<phivec.size();n++)
    {
        x = log(tvec[n])*logseenregionpixels/log(seenregion)*cos(phivec[n]);
        y = log(tvec[n])*logseenregionpixels/log(seenregion)*sin(phivec[n]);
        i = (int)round(x+logseenregionpixels);
        j = (int)round(y+logseenregionpixels);

        //drawpoint(data, wo, i, j, point_radius, -1);
        data[j*wo+i] = -2;
    }

}

std::vector<float> imasCoverings::scalarmult(std::vector<float> vec, float scalar)
{
    for(int n=0;n<vec.size();n++)
        vec[n] = vec[n] + scalar;
    return(vec);
}

void imasCoverings::drawdisks(float * data, const vector<float>& covering_element, float r,float seenregion, int logseenregionpixels)
{
    const float phidiskstep = M_PI/100000;
    std::vector<float> phivec, tupvec, tlowvec;
    phivec.clear(); tupvec.clear(); tlowvec.clear();
    int N = covering_element.size()/2;

    int i,j,wo = logseenregionpixels*2 +1; float x,y;
    i = 0 + logseenregionpixels;
    j = 0 + logseenregionpixels;
    drawpoint(data, wo, i, j, point_radius, -1);
    getdisks(1.0f,0, r, phidiskstep, phivec, tupvec, tlowvec);
    drawborder(data, phivec, tupvec,seenregion,logseenregionpixels);

    for(int n=1;n<=N;n++)
    {
        float phistep1 = covering_element[2*n-1];
        float t1 = covering_element[2*n-2];
        float phi1 = 0;
        phivec.clear(); tupvec.clear(); tlowvec.clear();
        getdisks(t1,phi1, r, phidiskstep, phivec, tupvec, tlowvec);
        while(phi1<M_PI)
        {
            x = log(t1)*logseenregionpixels/log(seenregion)*cos(phi1);
            y = log(t1)*logseenregionpixels/log(seenregion)*sin(phi1);
            i = (int)round(x+logseenregionpixels);
            j = (int)round(y+logseenregionpixels);

            drawpoint(data, wo, i, j, point_radius, -1);
            drawborder(data, scalarmult(phivec,phi1), tupvec,seenregion,logseenregionpixels);
            drawborder(data, scalarmult(phivec,phi1), tlowvec,seenregion,logseenregionpixels);

            phi1 = phi1+phistep1;
        }
    }

    // plot(log(t).*cos(psi),log(t).*sin(psi),DRAW_CENTER_DISKS_OPTS,'MarkerSize',12);
    // tempball.curvsup = plot(log(pointsup).*cos(phivect),log(pointsup).*sin(phivect),'-','Color',colorvec);
}



void imasCoverings::write_image_covering(const vector<float>& covering_element, float r,float region,float seenregion, int logseenregionpixels)
{
    int wo =  2*logseenregionpixels + 1;
    int ho = wo;

    int N = covering_element.size()/2;

    float * data = new float[wo*ho];
    for (int i=0;i<wo*ho;i++)
        data[i] = 0;


    // status for each pixel
    for (int x=-logseenregionpixels;x<=logseenregionpixels;x++)
        for (int y=-logseenregionpixels;y<=logseenregionpixels;y++)
        {

            float rho = sqrt(x*x +y*y);
            float phi = atan2(y,x);
            float t = exp( (log(seenregion)*rho/(float)logseenregionpixels) );

            int i = x+logseenregionpixels;
            int j = y+logseenregionpixels;

            if (transition_tilt(t,phi,1,0)<r)
                data[j*wo+i]++;

            for(int n=1;n<=N;n++)
            {
                float phistep1 = covering_element[2*n-1];
                float t1 = covering_element[2*n-2];
                float phi1 = 0;
                while(phi1<M_PI)
                {
                    if (transition_tilt(t,phi,t1,phi1)<r)
                        data[j*wo+i]++;
                    phi1 = phi1+phistep1;
                }
            }
        }
    drawdisks(data, covering_element, r,seenregion, logseenregionpixels);

    //image with orientations as in MATLAB
    reverse(data,data+wo*ho);
    for (int i=0;i<wo;i++)
        reverse(data+i*wo,data+(i+1)*wo-1);


    float* red = new float[3], *green = new float[3], *gray = new float[3], *white = new float[3],*blue = new float[3],*black = new float[3];

    red[0] = 204.0f;red[1] = 1.0f; red[2] = 51.0f;
    green[0] = 1.0f;green[1] = 150.0f; green[2] = 51.0f;
    black[0] = 0.0f; black[1] = 0.0f; black[2] = 0.0f;
    blue[0] = 255*0.871f; blue[1] = 255*0.922f; blue[2] = 255*0.98f;
    gray[0] = 150.0f;gray[1] = 150.0f; gray[2] = 150.0f;
    white[0] = 250.0f;white[1] = 250.0f; white[2] = 250.0f;

    float * rgb = new float[wo*ho*3];
    for(int c=0;c<3;c++)
        for(int j = 0; j < (int) ho; j++)
            for(int i = 0; i < (int) wo; i++)
            {
                if (data[j*wo+i]>1)
                    rgb[j*wo+i+c*(wo*ho)] = blue[c];
                if (data[j*wo+i]==1)
                    rgb[j*wo+i+c*(wo*ho)] = white[c];
                if (data[j*wo+i]==0)
                    rgb[j*wo+i+c*(wo*ho)] = gray[c];

                if (data[j*wo+i]==-1)
                    rgb[j*wo+i+c*(wo*ho)] = green[c];

                if (data[j*wo+i]==-2)
                    rgb[j*wo+i+c*(wo*ho)] = red[c];
            }

    //dashed line GAMMA in black
    float phi = 0;
    while(phi<2*M_PI)
    {
        int x = (int) round( log(region)*cos(phi)*logseenregionpixels/log(seenregion) );
        int y = (int) round( log(region)*sin(phi)*logseenregionpixels/log(seenregion) );
        int i = x+logseenregionpixels;
        int j = y+logseenregionpixels;

        if((i>0)&&(j>0)&&(j<ho)&&(i<ho))
            for(int c=0;c<3;c++)
            {
                rgb[j*wo+i+c*(wo*ho)] = black[c];
            }

        phi += 2*M_PI/(logseenregionpixels);
    }
#ifdef _NOMEX
    write_png_f32("covering.png", rgb, wo, ho, 3);
#endif

    delete[] rgb;
}





void imasCoverings::insertsimu(float t1,float r1, std::vector<tilt_simu>& simu_details)
{
    tilt_simu temp;
    bool inserted=false;
    for (int i=0;i<(int)simu_details.size();i++)
    {
        if (simu_details[i].t==t1)
        {
            simu_details[i].rots.push_back(r1);
            inserted = true;
        }
    }
    if (inserted == false)
    {
        temp.t=t1;
        temp.rots.push_back(r1);
        simu_details.push_back(temp);
    }

}

std::vector<float> imasCoverings::string2vec(string s)
{
    std::vector<float> vec;
    string acc = "";
    for(int i = 0; i < (int)(s.size()); i++)
    {
        if((s)[i] == delimiter)
        {
            vec.push_back( std::atof(acc.c_str()) );
            acc = "";
        }
        else
            acc += s[i];
    }
    if (acc!="")
        vec.push_back( std::atof(acc.c_str()) );
    return vec;
}


/**
 * @brief Sets up tilt simulations to perform from file "2simu.csv"
 */
void imasCoverings::loadsimulations2do()
{
    std::ifstream filein;
    filein.open("2simu.csv");
    if (filein.good())
    {
        std::vector<string> stringvec;
        std::vector<std::string>::iterator it;
        while( !(filein.eof()) )
        {
            string s;
            std::getline(filein, s);
            stringvec.push_back(s);
        }
        filein.close();

        if (stringvec.size()<4)
        {
            my_Printf("Error: Wrong file structure -> 2simu.csv \n");
            my_Printf("       Needs 4 rows where values are separated by comma \n");
            exit(-1);
        }

        std::vector<float> vec_simtilts;
        std::vector<float> vec_simrot;

        vec_simtilts = string2vec( stringvec[0] );
        vec_simrot = string2vec( stringvec[1] );
        if (vec_simrot.size()==vec_simtilts.size())
        {
            for (int i=0;i<(int)vec_simrot.size();i++)
                insertsimu(vec_simtilts[i],vec_simrot[i],simu_details1);
        }
        else
        {
            my_Printf("Error: Wrong file structure for simulations in image 1 -> 2simu.csv \n");
            my_Printf("       Explicitely state t_1,...,t_n and r_1,...,r_n \n");
            exit(-1);
        }

        totsimu1 = vec_simrot.size();

        vec_simtilts = string2vec( stringvec[2] );
        vec_simrot = string2vec( stringvec[3] );
        if (vec_simrot.size()==vec_simtilts.size())
        {
            for (int i=0;i<(int)vec_simrot.size();i++)
                insertsimu(vec_simtilts[i],vec_simrot[i],simu_details2);
        }
        else
        {
            my_Printf("Error: Wrong file structure for simulations in image 2 -> 2simu.csv \n");
            my_Printf("       Explicitely state t_1,...,t_n and r_1,...,r_n \n");
            exit(-1);
        }

        totsimu2 = vec_simrot.size();

    }
    else
    {
        my_Printf("WARNING: No file 2simu.csv has been found. Selecting default covering !  \n");
        loadsimulations2do(1.7f,1.7f,false);
    }
}


/**
 * @brief Sets up tilt simulations to perform from 4 vectors having all the information on \f$[T^{1}_{t_{i}}R^{1}_{\phi_{i}}]\f$ and \f$[T^{2}_{t_j}R^{2}_{\phi_j}]\f$.
 * Both vectors for image1/image2 must be equal in length.
 * @param vec_simtilts1 Tilts for image1
 * @param vec_simrot1 Rotations for image1
 * @param vec_simtilts2 Tilts for image2
 * @param vec_simrot2 Rotations for image2
 */
void imasCoverings::loadsimulations2do(std::vector<float>& vec_simtilts1, std::vector<float>& vec_simrot1,std::vector<float>& vec_simtilts2, std::vector<float>& vec_simrot2)
{
    if ( (vec_simrot1.size()==vec_simtilts1.size())&&(vec_simrot2.size()==vec_simtilts2.size()) )
    {
        for (int i=0;i<(int)vec_simrot1.size();i++)
            insertsimu(vec_simtilts1[i],vec_simrot1[i],simu_details1);
        for (int i=0;i<(int)vec_simrot2.size();i++)
            insertsimu(vec_simtilts2[i],vec_simrot2[i],simu_details2);
    }
    else
    {
        my_Printf("Error: Wrong structure information for simulations \n");
        my_Printf("       Explicitely state t_1,...,t_n and r_1,...,r_n for each image \n");
        exit(-1);
    }

}

/**
 * @brief Loads into <simu_details1> and <simu_details2> the corresponding set of simulations as concentric groups given by tilt and phase.
 * In other words, it generates for each \f$i\f$ indexing both vectors <tilts> and <phase>, all the following affine maps to be applied:
 * \f$T_{t_{i}}R_{\phi_{i}},\,T_{t_{i}}R_{2\phi_{i}}\,...,\,T_{t_{i}}R_{\left\lfloor \frac{\pi}{\phi_{i}}\right\rfloor \phi_{i}}\f$
 * where \f$\phi_i=phase[i]\f$ and \f$t_i = tilts[i]\f$
 * @param tilts \f$t_i = tilts[i]\f$
 * @param phase \f$\phi_i=phase[i]\f$
 */
void imasCoverings::load_optimal_simulations2do(const vector<float>& tilts, const vector<float>& phase)
{
    insertsimu(1,0,simu_details1);
    insertsimu(1,0,simu_details2);
    totsimu1 = 1;
    totsimu2 = 1;

    if (tilts.size()==phase.size())
    {

        for (int i=0;i<tilts.size();i++)
        {
            float t1 = tilts[i];
            float phi1 = phase[i];
            if ( (t1>1)&&(phi1>0) )
            {
                float phi = 0;
                while(phi<M_PI)
                {
                    totsimu1++;
                    totsimu2++;
                    insertsimu(t1,phi,simu_details1);
                    insertsimu(t1,phi,simu_details2);
                    phi = phi1 + phi;

                }
            }
        }
    }

}


/**
 * @brief Sets up default tilt simulations for an specific radius.
 * @param covering_code If positive the function will give back the default near optimal covering (as in \cite imas2017) assigned to that radius. If negative the function will give back the default covering proposed in the literature.
 * @param radius Controls the radius of the covering
 * @param writeimage If true this function will also generate an image "covering.png" representing the log radius- covering in the space of tilts
 */
void imasCoverings::loadsimulations2do(float covering_code,float radius, bool writeimage)
{
    my_Printf("Covering Info:\n");
    vector<float> tilts, phase;
    if (radius<0)
        radius = -radius;
    int r = round(covering_code*100);
    float region = 0.0f;
    switch (r)
    {
    case 140: // OK
    {
        region=4.4f;
        radius=1.4f;
        my_Printf("Selecting a log%.1f-covering for tilts up to %.0f  \n",1.4f,region);

        tilts.resize(3); phase.resize(3);
        tilts[0]=1.7745; phase[0]=0.525449; tilts[1]=2.38807; phase[1]=0.286394; tilts[2]=3.91568; phase[2]=0.14296;

        load_optimal_simulations2do(tilts, phase);

        break;
    }
    case 100:
    {
        region = 0.0f;
        my_Printf("Selecting just one tilt: [Id]  \n");
        tilts.clear(); phase.clear();
        load_optimal_simulations2do(tilts, phase);
        break;
    }
    case 200://OK
    {
        region = 9.0f;
        radius = 2.0f;
        my_Printf("Selecting optimal log%.0f-covering for tilts up to %.0f  \n",2.0f,region);

        tilts.resize(2); phase.resize(2);
        tilts[0]=3.40531; phase[0]=0.396436; tilts[1]=8.08729; phase[1]=0.158663;
        load_optimal_simulations2do(tilts, phase);

        break;
    }
    case 190: //OK
    {
        region=7.2;
        radius=1.9;
        my_Printf("Selecting optimal log%.1f-covering for tilts up to %.0f  \n",1.9f,region);

        tilts.resize(2); phase.resize(2);
        tilts[0]=3.20805; phase[0]=0.349674; tilts[1]=7.78315; phase[1]=0.174837;
        load_optimal_simulations2do(tilts, phase);

        break;
    }
    case 180: //OK
    {
        region = 6.0f;
        radius=1.8f;
        my_Printf("Selecting optimal log%.1f-covering for tilts up to %.0f  \n",1.8f,region);

        tilts.resize(2); phase.resize(2);
        tilts[0]=2.89419; phase[0]=0.396183; tilts[1]=6.33474; phase[1]=0.198091;
        load_optimal_simulations2do(tilts, phase);

        break;
    }
    case 170: //OK
    {
        region=5.8; radius=1.7;
        my_Printf("Selecting optimal log%.1f-covering for tilts up to %.0f  \n",1.7f,region);

        tilts.resize(2); phase.resize(2);
        tilts[0]=2.61757; phase[0]=0.398032; tilts[1]=5.18168; phase[1]=0.198314;
        load_optimal_simulations2do(tilts, phase);
        break;
    }
    case 150: // OK
    {
        region=5.5f;
        radius=1.5f;
        my_Printf("Selecting optimal log%.2f-covering for tilts up to %.2f  \n",1.5f,region);
        tilts.resize(3); phase.resize(3);
        tilts[0]=1.99593; phase[0]=0.526597; tilts[1]=2.80717; phase[1]=0.286735; tilts[2]=4.95509; phase[2]=0.1431;
        load_optimal_simulations2do(tilts, phase);
        break;
    }
    case 160:
    {
        region=5.0f;
        radius=1.6f;
        my_Printf("Selecting optimal log%.1f-covering for tilts up to %.0f  \n",1.6f,region);

        tilts.resize(2); phase.resize(2);
        tilts[0]=2.34933; phase[0]=0.395055; tilts[1]=4.60284; phase[1]=0.197527;
        load_optimal_simulations2do(tilts, phase);

        break;
    }
    case -150: // Not at all optimal
    {
        region = 0.0f;
        my_Printf("Selecting the MODS SURF-SURF HARD proposed covering \n",1.5f,region);

        tilts.resize(8); phase.resize(8);
        for(int i=1;i<=8;i++)
        {
            tilts[i-1] = i+1;
            phase[i-1] = (M_PI*74/180)/tilts[i-1];
        }
        load_optimal_simulations2do(tilts, phase);
        break;
    }

    case -152: // Not at all optimal
    {
        region = 0.0f;
        my_Printf("Selecting the FAIR-SURF fixed-tilts proposed covering \n",1.5f,region);

        tilts.resize(4); phase.resize(4);
        tilts[0] = 2*sqrt(3)/3; tilts[1] = sqrt(2); tilts[2] = 2;tilts[3] = 4;
        for(int i=1;i<=8;i++)
        {
            phase[i-1] = (M_PI*73/180)/tilts[i-1];
        }
        load_optimal_simulations2do(tilts, phase);
        break;
    }

    case -180: // Not at all optimal
    {
        region = 5.5f;
        my_Printf("Selecting the ASIFT first log%.1f-covering for tilts up to %.0f  \n",1.8f,region);

        tilts.resize(5); phase.resize(5);
        for(int i=1;i<=5;i++)
        {
            tilts[i-1] = pow(sqrt(2),i);
            phase[i-1] = (M_PI*73/180)/tilts[i-1];
        }
        load_optimal_simulations2do(tilts, phase);
        break;
    }
    case -182: // Not at all optimal
    {
        region = 9.6f;
        my_Printf("Selecting the MODS DOG-SIFT HARD log%.1f-covering for tilts up to %.0f  \n",1.8f,region);

        tilts.resize(4); phase.resize(4);
        for(int i=1;i<=4;i++)
        {
            tilts[i-1] = i*2;
            phase[i-1] = (M_PI*60.5f/180)/tilts[i-1];
        }
        load_optimal_simulations2do(tilts, phase);
        break;
    }
    case -184: // Not at all optimal
    {
        region = radius;
        my_Printf("Selecting the MODS DOG-SIFT MEDIUM log%.1f-covering for tilts up to %.0f  \n",1.8f,region);

        tilts.resize(8); phase.resize(8);
        for(int i=1;i<=8;i++)
        {
            tilts[i-1] = i+1;
            phase[i-1] = (M_PI*1.001)/tilts[i-1];
        }
        load_optimal_simulations2do(tilts, phase);
        break;
    }


    default:
    {
        my_Printf("WARNING: The optimal log%.2f-covering hasn't been computed yet !  \n",covering_code);
        loadsimulations2do(1.7f,radius,true);
        writeimage = false;
        break;
    }

    }



    vector<float> covering_element(tilts.size()*2);
    my_Printf("( t_1 phi_1 ... t_n phi_n ) = (");
    for (int i=1;i<=tilts.size();i++)
    {
        covering_element[2*i-2] = tilts[i-1];
        covering_element[2*i-1] = phase[i-1];
        my_Printf(" %.10f %.10f",tilts[i-1],phase[i-1]);
    }

    _arearatio = Fcout(covering_element);
    my_Printf(" )\n Area ratio = %.3f  \n \n",Fcout(covering_element));

    if (writeimage)
    {
        if (tilts.size()>0)
            write_image_covering(covering_element, radius, region, tilts[tilts.size()-1]*radius, 200);
        else
            write_image_covering(covering_element, radius, region, radius*radius, 200);


    }
}
