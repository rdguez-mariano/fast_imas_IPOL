/*
 Header of the file for the keypoint detection
 
 Copyright 2013: Edouard Oyallon, Julien Rabin
 
 Version for IPOL.
 */

#ifndef KEYPOINT
#define KEYPOINT

#include <vector>
#include <math.h>

#include "lib.h"
#include "image.h"
#include "integral.h"
#include <sstream>
#include <iostream>
#include <fstream>
#include <ctime>


// Keypoint class
class keyPoint {
public:
	double x,y,scale,orientation;
	bool signLaplacian;
    // Constructor
	keyPoint(REGULAR_IMAGE x_, REGULAR_IMAGE y_, REGULAR_IMAGE scale_,REGULAR_IMAGE orientation_, bool signLaplacian_):x(x_),y(y_),scale(scale_),orientation(orientation_),signLaplacian(signLaplacian_){}
    keyPoint(){}
};

// List of keypoints
typedef  std::vector<keyPoint*> listKeyPoints;


#include "descriptor.h"

// Create a keypoint
// (i,j) are the coordinates of the keypoint, signLapl is the sign of the Laplacian, scale the box-size.
void addKeyPoint(imageIntegral* img,REGULAR_IMAGE i,REGULAR_IMAGE j,bool signLapl,REGULAR_IMAGE scale,listKeyPoints* listKeyPoints);
							 
// Compute the orientation of a keypoint
float getOrientation(imageIntegral* imgInt,int x,int y,int numberSector,REGULAR_IMAGE scale);

// Reject or interpolate the coordinate of a keypoint. This is necessary since there
// was a subsampling of the image.
bool interpolationScaleSpace(image** img,int x, int y, int i, REGULAR_IMAGE &x_, REGULAR_IMAGE &y_, REGULAR_IMAGE &s_, int sample, int octaveValue);


// Check if a point is a local maximum or not, and more than a given threshold.
inline bool isMaximum(image** imageStamp,int x,int y,int scale, float threshold)
{
	REGULAR_IMAGE tmp=(*(imageStamp[scale]))(x,y);
	
	if(tmp>threshold)
	{
        for(int j=-1+y;j<2+y;j++)
            for(int i=-1+x;i<2+x;i++) {
                if((*(imageStamp[scale-1]))(i,j)>=tmp)
                    return false;
                if((*(imageStamp[scale+1]))(i,j)>=tmp)
                    return false;
                if((x!=i || y!=j) && (*(imageStamp[scale]))(i,j)>=tmp)
                    return false;
            }
		return true;
	}						
	else
		return false;
}
#endif


