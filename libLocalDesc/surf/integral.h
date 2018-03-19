/*
 Header file of integral.cpp for SURF
 
 Copyright 2013: Edouard Oyallon, Julien Rabin
 
 Version for IPOL.
 */

#ifndef INTEGRAL
#define INTEGRAL

#include "image.h"


// Convolution by a square defined by the bottom-left (a,b) and top-right (c,d)
inline double squareConvolutionXY(imageIntegral* imgInt,int a,int b,int c,int d,int x,int y)
{
	int a1=x-a;
	int a2=y-b;
	int b1=a1-c;
	int b2=a2-d;
	return ((*imgInt)(b1,b2)+(*imgInt)(a1,a2)-(*imgInt)(b1,a2)-(*imgInt)(a1,b2));// Note: No L2-normalization is performed here.
}	



// Convolution by a box [-1,+1]
inline INTEGRAL_IMAGE haarX(imageIntegral* img,int x,int y,int lambda)
{
	
	return -(squareConvolutionXY(img,1,-lambda-1,-lambda-1,lambda*2+1, x, y)+
			squareConvolutionXY(img, 0,-lambda-1, lambda+1,lambda*2+1, x, y));
	
	
}

// Convolution by a box [-1;+1]
inline INTEGRAL_IMAGE haarY(imageIntegral* img,int x,int y,int lambda)
{
    return -(squareConvolutionXY(img, -lambda-1,1, 2*lambda+1,-lambda-1, x, y)+
		 squareConvolutionXY(img, -lambda-1,0, 2*lambda+1,lambda+1, x, y));
}

#endif