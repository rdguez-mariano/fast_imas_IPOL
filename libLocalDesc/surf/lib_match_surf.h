/*
 Lib match header for SURF
 
 Copyright 2013: Edouard Oyallon, Julien Rabin
 
 Version for IPOL.
 */


#ifndef LIB_MATCH_SURF
#define LIB_MATCH_SURF
#include "descriptor.h"
#include "libMatch/match.h"
//#include "lib/orsa.h"

#include <iostream>


struct MatchSurf {
    float x1, y1, scale1, angle1, x2, y2, scale2, angle2;
};


// Function to match two sets of descriptors
std::vector<MatchSurf>  matchDescriptor(listDescriptor * l1, listDescriptor * l2);

// Return the euclidean distance between 2 descriptors
float euclideanDistance(descriptor *a,descriptor* b);

// Ratio between two matches
//extern float surf_ratio;

#endif
