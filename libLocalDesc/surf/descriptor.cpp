/*
 Descriptor file for SURF
 
 Copyright 2013: Edouard Oyallon, Julien Rabin
 
 Version for IPOL.
 */

#include "descriptor.h"



// For a given list of keypoints, return the associated list of descriptors.
// Handle the integral image suppression.
listDescriptor* getDescriptor(imageIntegral* imgInt,listKeyPoints* lPC)
{
	listDescriptor* lD=new listDescriptor();
    // Compute descriptor from each keypoints
	for(int i=0;i<(int)lPC->size();i++)
		lD->push_back(makeDescriptor(imgInt, (*lPC)[i]));
	delete imgInt;/*MemCheck*/
	return lD;
}



// Create a descriptor in a squared domain of size 20*scale
descriptor* makeDescriptor(imageIntegral* imgInt,keyPoint* pC)
{
	REGULAR_IMAGE scale=pC->scale;
	descriptor* desc=new descriptor();
	// Divide in a 4x4 zone the space around the interest point

	// First compute the orientation.
	REGULAR_IMAGE cosP=cos(pC->orientation);
	REGULAR_IMAGE sinP=sin(pC->orientation);
	REGULAR_IMAGE norm=0,u,v,gauss,responseU,responseV,responseX,responseY;
	
	// Divide in 16 sectors the space around the interest point.
	for(int i=0;i<DESCRIPTOR_SIZE_1D;i++)
	{
	   for(int j=0;j<DESCRIPTOR_SIZE_1D;j++)
		{
			(desc->list[DESCRIPTOR_SIZE_1D*i+j]).sumDx=0;
			(desc->list[DESCRIPTOR_SIZE_1D*i+j]).sumAbsDx=0;
			(desc->list[DESCRIPTOR_SIZE_1D*i+j]).sumDy=0;
			(desc->list[DESCRIPTOR_SIZE_1D*i+j]).sumAbsDy=0;
			// Then each 4x4 is subsampled into a 5x5 zone
			for(int k=0;k<5;k++)
			{
				for(int l=0;l<5;l++)
				{
					// We pre compute Haar answers
					u=(pC->x+scale*(cosP*((i-2)*5+k+0.5)-sinP*((j-2)*5+l+0.5)));
					v=(pC->y+scale*(sinP*((i-2)*5+k+0.5)+cosP*((j-2)*5+l+0.5)));
					responseX=haarX(imgInt,u,v,fround(scale)); // (u,v) are already translated of 0.5, which means
                                                               // that there is no round-off to perform: one takes
                                                               // the integer part of the coordinates.
					responseY=haarY(imgInt,u,v,fround(scale));
					
					// Gaussian weight
					gauss=gaussian(((i-2)*5+k+0.5),((j-2)*5+l+0.5),3.3);
					
				    // Rotation of the axis
					//responseU = gauss*( -responseX*sinP + responseY*cosP);
					//responseV = gauss*(responseX*cosP + responseY*sinP);
                    responseU = gauss*(responseX*cosP +responseY*sinP);
					responseV = gauss*(-responseX*sinP + responseY*cosP);
                    
				    // The descriptors.
				    (desc->list[DESCRIPTOR_SIZE_1D*i+j]).sumDx+=responseU;
				    (desc->list[DESCRIPTOR_SIZE_1D*i+j]).sumAbsDx+=absval(responseU);
				    (desc->list[DESCRIPTOR_SIZE_1D*i+j]).sumDy+=responseV;
					(desc->list[DESCRIPTOR_SIZE_1D*i+j]).sumAbsDy+=absval(responseV);
					
				}
			}
			// Compute the norm of the vector
			norm+=(desc->list[DESCRIPTOR_SIZE_1D*i+j]).sumAbsDx*(desc->list[DESCRIPTOR_SIZE_1D*i+j]).sumAbsDx+(desc->list[DESCRIPTOR_SIZE_1D*i+j]).sumAbsDy*(desc->list[DESCRIPTOR_SIZE_1D*i+j]).sumAbsDy+((desc->list[DESCRIPTOR_SIZE_1D*i+j]).sumDx*(desc->list[DESCRIPTOR_SIZE_1D*i+j]).sumDx+(desc->list[DESCRIPTOR_SIZE_1D*i+j]).sumDy*(desc->list[DESCRIPTOR_SIZE_1D*i+j]).sumDy);

		}
	}
	// Normalization of the descriptors in order to improve invariance to contrast change
    // and whitening the descriptors.
	norm=sqrtf(norm);
	if(norm!=0)
	for(int i=0;i<DESCRIPTOR_SIZE_1D*DESCRIPTOR_SIZE_1D;i++)
	{
		(desc->list[i]).sumDx/=norm;
		(desc->list[i]).sumAbsDx/=norm;
		(desc->list[i]).sumDy/=norm;
		(desc->list[i]).sumAbsDy/=norm;	
	}
	desc->kP=new keyPoint(*pC);
	return desc;
}


