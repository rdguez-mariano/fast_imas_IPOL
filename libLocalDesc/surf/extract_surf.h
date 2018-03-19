/*
 Header file for SURF
 
 Copyright 2013: Edouard Oyallon, Julien Rabin
 
 Version for IPOL.
 */


#ifndef SURF
#define SURF
#include "integral.h"
#include "descriptor.h"
#include "keypoint.h"
#include <stdlib.h>
#endif

listDescriptor* extract_surf(float* img_double, int width, int height);
