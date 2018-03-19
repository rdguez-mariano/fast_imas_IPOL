/*
 Implementation file of the image class for SURF

 Copyright 2013: Edouard Oyallon, Julien Rabin

 Version for IPOL.
 */


#include "image.h"



// Constructors
image::image(int x, int y, float* data):width(x),height(y)
{
    img=new REGULAR_IMAGE[width*height];
    for(int i=0;i<x;i++)
        for(int j=0;j<y;j++)
            img[width*j+i]=(double) data[width*j+i];
}
image::image(int x, int y, unsigned char* data):width(x),height(y)
{
    img=new REGULAR_IMAGE[width*height];
    for(int i=0;i<x;i++)
        for(int j=0;j<y;j++)
            img[width*j+i]=(int) data[width*j+i];
}
image::image(int x,int y):width(x),height(y)
{
	img=new REGULAR_IMAGE[width*height];
    memset(img,0,width*height*sizeof(REGULAR_IMAGE));
}
image::image(image* im):width(im->width),height(im->height) // not cloning the im input
{
	img=new REGULAR_IMAGE[width*height];
    memset(img,0,width*height*sizeof(REGULAR_IMAGE));
}

// Destructor
imageIntegral::~imageIntegral()
{
    delete[] img;
}


// Superpose two images.(might be a "friend" function either)
void image::printImagePara(  char fileName[],image* para)
{
	float* data=new float[3*width*height];
	float max,min;
	max=img[0];
	min=img[0];

	for(int i=0;i<width*height;i++)
	{
		if(min>img[i]) min=img[i];
        if(max<img[i]) max=img[i];
	}

	for(int i=0;i<width*height;i++)
			{
				float lambda=(img[i]-max)/(min-max);
				data[i]=fround(0*lambda+(1-lambda)*255);
				data[width*height+i]=fround(0*lambda+(1-lambda)*255);
				data[2*width*height+i]=fround(0*lambda+(1-lambda)*255);
				if((*para)(i%para->getWidth(),i/para->getWidth())!=0)
				{
					data[2*width*height+i]=0;
					data[0*width*height+i]=fround(0*lambda+(1-lambda)*130)+(*para)(i%para->getWidth(),i/para->getWidth());
					data[1*width*height+i]=0;
				}
			}

//	write_png_f32(fileName, data, width,height,3);
	delete[] data;
}

// Pad an image, and allocate some memory according to padding
image* image::padImage(int padding)
{
    image* img2=new image(width+2*padding,height+2*padding);
    int i0,j0;

    for(int i=-padding;i<width+padding;i++)
            for(int j=-padding;j<height+padding;j++)
            {
                i0=i;
                j0=j;
                if(i0<0)
                    i0=-i0;
                if(j0<0)
                    j0=-j0;
                i0=i0%(2*width);
                j0=j0%(2*height);
                if(i0>=width)
                    i0=2*width-i0-1;
                if(j0>=height)
                    j0=2*height-j0-1;

                (*img2)(i+padding,j+padding)=(*this)(i0,j0);
            }

    return img2;

}

// Constructor of the integral image
imageIntegral::imageIntegral(image* img_input)
{
    padding=312;// size descriptor * max size L = 4*0.4*195;
    width=img_input->getWidth()+2*padding;
    height=img_input->getHeight()+2*padding;

    image* img_padded=img_input->padImage(padding); // Pad the image

    img=new INTEGRAL_IMAGE[width*height];
    computeIntegralImage(img_padded);
    delete img_padded;
}


// Normalization
void image::normalizeImage()
{
    REGULAR_IMAGE min=img[0],max=img[0];

    for(int i=0;i<width;i++)
		for(int j=0;j<height;j++)
		{
			min=(img[j*width+i]<min)?img[j*width+i]:min;
			max=(img[j*width+i]>max)?img[j*width+i]:max;
		}

    for(int i=0;i<width;i++)
        for(int j=0;j<height;j++)
            img[j*width+i]=(int) 255*((img[j*width+i]-min)/(max-min));
}

// Return by reference the subsampled size
void image::getSampledImage(int& x,int& y,int sample)
{
    x=width/sample;
    y=height/sample;
}


// This functions computes the integral image. In order to avoir border effects,
// the image is firstly periodized by having computing
void imageIntegral::computeIntegralImage(image* image)
{
	// Initialization
    img[0]=(*image)(0,0);

    // First row
    for(int i=1;i<image->getWidth();i++)
		img[i]=img[i-1]+(*image)(i,0);

    // Recursion
	for(int j=1;j<image->getHeight();j++)
	{
		INTEGRAL_IMAGE h=0;
		for(int i=0;i<image->getWidth();i++)
		{
			h+=(*image)(i,j);
			img[i+width*j]=img[i+width*(j-1)]+h;
		}
	}
}
