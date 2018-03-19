/*
    Keypoint extractor file for SURF
 
    Copyright 2013: Edouard Oyallon, Julien Rabin
 
    Version for IPOL.
*/

#include "extract_surf.h"
using namespace std;

// Should be execute as surf file1.png descriptors.txt
listDescriptor* extract_surf(float* img_double, int width, int height)
{


    // Threshold on the Hessian
    float threshold=1000;   

	//Input images in SURF IPOL format
    image* img=new image(width, height, img_double);
    
    // Normalize the image by setting the minimum value of the image to 0 and its maximum value to 256. Several technics like histogram equalization, linearization could be used. Standardization can not be used since the algorithm works with non negative integers.
    img->normalizeImage();

	// The list of key points.
	listKeyPoints* l=new listKeyPoints();
    listDescriptor* listDesc;
    
	// Keypoints detection and description
    listDesc=getKeyPoints(img,l,threshold);
		
	// Free memory
    /*MemCheck*/
	for(int i=0;i<(int)l->size();i++)
		delete((*l)[i]);
    delete l;
    delete img;
    return(listDesc);
} 
