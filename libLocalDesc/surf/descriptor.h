/*
 Header of the descriptor file for SURF
 
 Copyright 2013: Edouard Oyallon, Julien Rabin
 
 Version for IPOL.
 */



#ifndef DESCRIPTOR

#define DESCRIPTOR
#include <math.h>
#include "image.h"
#include "keypoint.h"
#include "lib.h"



// Class of the descriptor for the vector of SURF's detected keypoint.
class vectorDescriptor{
	public :
	REGULAR_IMAGE sumDx;
	REGULAR_IMAGE sumDy;
	REGULAR_IMAGE sumAbsDx;
	REGULAR_IMAGE sumAbsDy;
	// Initialization
	vectorDescriptor(REGULAR_IMAGE sumDx_,REGULAR_IMAGE sumDy_,REGULAR_IMAGE sumAbsDx_,REGULAR_IMAGE sumAbsDy_):sumDx(sumDx_),sumDy(sumDy_),sumAbsDx(sumAbsDx_),sumAbsDy(sumAbsDy_){}
	vectorDescriptor():sumDx(0),sumDy(0),sumAbsDx(0),sumAbsDy(0){}
};



// Descriptor of a keypoint, with its vector descriptor and several other additional informations.
class descriptor{
public:
    // The array is ordered as sum dx, sum dy, sum |dx|, sum |dy| for each cell in the array
    // of size 20s.
    vectorDescriptor* list;
    
	keyPoint *kP;// Keypoint pointer.

    descriptor(){list=new vectorDescriptor[16];kP = new keyPoint();}

	~descriptor(){/* MemCheck*/delete kP;delete[] list;}

	// Copy constructor
    descriptor(const descriptor & des){
        list=new vectorDescriptor[DESCRIPTOR_SIZE_1D*DESCRIPTOR_SIZE_1D];
        for(int i=0;i<DESCRIPTOR_SIZE_1D*DESCRIPTOR_SIZE_1D;i++)
            list[i]=(des.list)[i];
        kP=new keyPoint(*des.kP);}
	
};

// List of descriptor
typedef  std::vector<descriptor*> listDescriptor;

// This function creates the descriptors from list of key points and
// uses the integrale image
descriptor* makeDescriptor(imageIntegral* imgInt,keyPoint* pC);

// This function creates the list of descriptors
listDescriptor* getDescriptor(imageIntegral* imgInt,listKeyPoints* lPC);

// This function creates the list of keypoints
listDescriptor* getKeyPoints(image *img,listKeyPoints* lKP,float threshold);

#endif
