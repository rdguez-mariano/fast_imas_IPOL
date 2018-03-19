/*
 Image class file for SURF

 Copyright 2013: Edouard Oyallon, Julien Rabin

 Version for IPOL.
 */

#ifndef IMAGE
#define IMAGE

#include "io_png/io_png.h"
//#include "extern_files_used/io_png.h"
#include "lib.h"
#include <string.h>

// Class that handles any images except the integral images
class image {
public:

    // Constructors & clone constructors
    image(int x, int y);
	image(int x, int y, unsigned char* data);
    image(int x, int y, float* data);
    image(image* im);

    // Destructor
    ~image(){ delete[] img; }

    // Accessor
	inline int getWidth() {return width;}
	inline int getHeight() {return height;}
    // Get methods and reference
	inline REGULAR_IMAGE operator()(int x, int y) const {return img[ y*width + x ];}
	inline REGULAR_IMAGE& operator()(int x, int y) {return img[ y*width+ x ];}

    // Print 3 images to show the matchs using ORSA.
	void printImagePara(  char fileName[],image* para);

    // Linearly sets the min and max value of an image to be 0 dans 255
    void normalizeImage();

    // Returns the sampled image size using references
    void getSampledImage(int& w,int& h,int sample);

    // Returns a padded image
    image* padImage(int padding);

private:
	int	width,height; // size of the images
	REGULAR_IMAGE*	img; // Array containing the image
};

// Special class for integral images
class imageIntegral{
public:
    // Constructor which computes the integral image.
    imageIntegral(image* im);
	~imageIntegral();

    // Compute the integral image
    void computeIntegralImage(image* img);
	inline INTEGRAL_IMAGE& operator()(int x, int y) {return img[width*(y+padding)+(x+padding)];} // setter
    inline INTEGRAL_IMAGE  operator()(int x, int y) const {return img[width*(y+padding)+(x+padding)];} // accessor


private:
    int	width,height;	// integral image real size
    INTEGRAL_IMAGE* img; // size
    int padding; // padding of the original image

};
#endif
