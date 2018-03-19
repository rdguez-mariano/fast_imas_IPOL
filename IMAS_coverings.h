/**
  * @file IMAS_covering.h
  * @author Mariano Rodríguez
  * @date 2017
  * @brief handles IMAS coverings: either coverings proposed in the literature or optimal coverings as in Mariano Rodríguez, Julie Delon and Jean-Michel Morel.
  */
#ifndef IMAS_COVERING_H
#define IMAS_COVERING_H

#include <vector>
#include "mex_and_omp.h"
#include<fstream>


/**
 * @brief It stores all rotations to be simulated for an certain tilt
 */
struct tilt_simu {
    float	t;
    std::vector<float> rots;
};


class imasCoverings
{
protected:
    /**
     * @brief Global variable that stores tilts and rotations to be performed on image1
     * @code
    // Loop on tilts
    for (tt = 1; tt <= simu_details.size(); tt++)
    {
        // Loop on rotations.
        for ( int rr = 1; rr <= simu_details[tt-1].rots.size(); rr++ )
            {
                ... Do something here with it
            }
    }
    *@endcode
     */
    std::vector<tilt_simu> simu_details1;

    /**
     * @brief Global variable that stores tilts and rotations to be performed on image2
     * @code
    // Loop on tilts
    for (tt = 1; tt <= simu_details.size(); tt++)
    {
        // Loop on rotations.
        for ( int rr = 1; rr <= simu_details[tt-1].rots.size(); rr++ )
            {
                ... Do something here with it
            }
    }
    *@endcode
     */
    std::vector<tilt_simu> simu_details2;
    /**
     * @brief Global variable that stores the total amount of simulations to be performed on image1
     */
    int totsimu1;

    /**
     * @brief Global variable that stores the total amount of simulations to be performed on image2
     */
    int totsimu2;
    float _arearatio;

    static float transition_tilt(float t, float psi1, float s, float psi2);
    static float Fcout(const std::vector<float>& tilt_element);
    static void drawpoint(float * data, int width, int i, int j, float radius, float value);
    static void getdisks(float t,float psi, float r, float phistep, std::vector<float>& phivec, std::vector<float>& tupvec,std::vector<float>& tlowvec);
    static void drawborder(float *data, std::vector<float> phivec, std::vector<float>& tvec,float seenregion, int logseenregionpixels);
    static std::vector<float> scalarmult(std::vector<float> vec, float scalar);
    static void drawdisks(float * data, const std::vector<float>& covering_element, float r,float seenregion, int logseenregionpixels);

    void insertsimu(float t1,float r1, std::vector<tilt_simu>& simu_details);
    static std::vector<float> string2vec(std::string s);


public:
    static void write_image_covering(const std::vector<float>& covering_element, float r,float region,float seenregion, int logseenregionpixels);
    void loadsimulations2do();
    void loadsimulations2do(std::vector<float>& vec_simtilts1, std::vector<float>& vec_simrot1,std::vector<float>& vec_simtilts2, std::vector<float>& vec_simrot2);
    void load_optimal_simulations2do(const std::vector<float>& tilts, const std::vector<float>& phase);
    void loadsimulations2do(float covering_code,float radius, bool writeimage);

    imasCoverings()
    {
        _arearatio = 0.0f;
    }
    const std::vector<tilt_simu> getSimuDetails1()
    {
        return(simu_details1);
    }

    int getTotSimu1()
    {
        return(totsimu1);
    }
    const std::vector<tilt_simu> getSimuDetails2()
    {
        return(simu_details2);
    }
    int getTotSimu2()
    {
        return(totsimu2);
    }
    float getAreaRatio()
    {
        return(_arearatio);
    }
    ~imasCoverings()
    {
        simu_details1.clear();
        simu_details2.clear();
    }
};

#endif // IMAS_COVERING_H
