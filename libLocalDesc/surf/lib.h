/*
 Header of generic tools for SURF

 Copyright 2013: Edouard Oyallon, Julien Rabin

 Version for IPOL.
 */

#ifndef LIB
#define LIB
#include <math.h>
#include <sstream>
#include <string>

inline int fround(float flt);
inline double gaussian(double x, double y, double sig);


#ifndef M_PI
#define M_PI   3.14159265358979323846
#endif /* !M_PI */

// Type of the integral image
typedef long int INTEGRAL_IMAGE;

// Type for a normal image
typedef double REGULAR_IMAGE;

// Amount of considered angular regions
#define NUMBER_SECTOR 20

// Maximum octave
#define OCTAVE 4

// Maximum scale
#define INTERVAL 4

// Sampling of the image at each step
#define SAMPLE_IMAGE 2

// Size of the descriptor along x or y dimension
#define DESCRIPTOR_SIZE_1D 4

// Gaussian - should be computed as an array to be faster.
inline double gaussian(double x, double y, double sig)
{
	return 1/(2*M_PI*sig*sig)*exp( -(x*x+y*y)/(2*sig*sig));
}

// Round-off functions
inline int fround(REGULAR_IMAGE rgp) { return (int) (rgp+0.5f); }
inline int fround(float flt) { return (int) (flt+0.5f); }


// Absolute value
inline REGULAR_IMAGE absval(REGULAR_IMAGE x) { return ((x>0)?x:-x); }

// int->char
inline std::string convertIntToChar(int i)
{
    std::ostringstream oss;
    oss<<i;
    return oss.str();
}

#endif
