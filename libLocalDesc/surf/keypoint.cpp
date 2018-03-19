/*
 Keypoint file for SURF

 Copyright 2013: Edouard Oyallon, Julien Rabin

 Version for IPOL.
 */

#include "keypoint.h"
#include <iostream>
#include <fstream>

// Compute the list of descriptors and keypoints, for a given threshold
listDescriptor* getKeyPoints(image *img,listKeyPoints* lKP,float threshold)
{

	// Compute the integral image
    imageIntegral* imgInt=new imageIntegral(img);

    // Array of each Hessians and of each sign of the Laplacian
    image*  hessian[INTERVAL];
    image* signLaplacian[INTERVAL];

	// Auxiliary variables
	double Dxx,Dxy,Dyy;
	int intervalCounter,octaveCounter,x,y,w,h,xcoo,ycoo,lp1,l3,mlp1p2,lp1d2,l2p1,pow2,sample,l;
    double nxy,nxx;

    // For loop on octave
    for( octaveCounter=0;octaveCounter<OCTAVE;octaveCounter++)
	{
        pow2=pow(2,octaveCounter+1);

        sample=pow(SAMPLE_IMAGE,octaveCounter);// Sampling step

        img->getSampledImage(w,h,sample);// Build a sampled images filled in with 0

        // Memory initialization
        for(int i=0;i<INTERVAL;i++)
        {
            hessian[i]=new image(w,h);
            signLaplacian[i]=new image(w,h);
        }

        // For loop on intervals
        for( intervalCounter=0;intervalCounter<INTERVAL;intervalCounter++)
		{
            l=pow2*(intervalCounter+1)+1; // the "L" in the article.

            // These variables are precomputed to allow fast computations.
            // They correspond exactly to the Gamma of the formula given in the article for
            // the second order filters.
            lp1=-l+1;
            l3=3*l;
            lp1d2=(-l+1)/2;
            mlp1p2=(-l+1)/2-l;
            l2p1=2*l-1;

            nxx=sqrt(6*l*(2*l-1));// frobenius norm of the xx and yy filters
            nxy=sqrt(4*l*l);// frobenius of the xy filter.


            // These are the time consuming loops that compute the Hessian at each points.
            for( y=0;y<h;y++)
			{
				for( x=0;x<w;x++)
				{
                    // Sampling
                    xcoo=x*sample;
                    ycoo=y*sample;

                    // Second order filters
                    Dxx=squareConvolutionXY(imgInt,lp1,mlp1p2,l2p1,l3,xcoo,ycoo)-3*squareConvolutionXY(imgInt,lp1,lp1d2,l2p1,l,xcoo,ycoo);
                    Dxx/=nxx;

					Dyy=squareConvolutionXY(imgInt,mlp1p2,lp1,l3,l2p1,xcoo,ycoo)-3*squareConvolutionXY(imgInt,lp1d2,lp1,l,l2p1,xcoo,ycoo);
                    Dyy/=nxx;
					Dxy=squareConvolutionXY(imgInt,1,1,l,l,xcoo,ycoo)+squareConvolutionXY(imgInt,0,0,-l,-l,xcoo,ycoo)
						+squareConvolutionXY(imgInt,1,0,l,-l,xcoo,ycoo)+squareConvolutionXY(imgInt,0,1,-l,l,xcoo,ycoo);

                    Dxy/=nxy;

                    // Computation of the Hessian and Laplacian
                    (*hessian[intervalCounter])(x,y)= (Dxx*Dyy-0.8317*(Dxy*Dxy));
                    (*signLaplacian[intervalCounter])(x,y)=Dxx+Dyy>0;
				}
			}



		}

        REGULAR_IMAGE x_,y_,s_;

		// Detect keypoints
        for(intervalCounter=1;intervalCounter<INTERVAL-1;intervalCounter++)
		{
			l=(pow2*(intervalCounter+1)+1);
                // border points are removed
				for(int y=1;y<h-1;y++)
					for(int x=1 ; x<w-1 ; x++)
                        if(isMaximum(hessian, x, y, intervalCounter,threshold))
                        {
                            x_=x*sample;
                            y_=y*sample;
                            s_=0.4*(pow2*(intervalCounter+1)+2); // box size or scale
                            // Affine refinement is performed for a given octave and sampling
                            if( interpolationScaleSpace(hessian, x, y, intervalCounter, x_, y_, s_, sample,pow2) )
                                addKeyPoint(imgInt, x_, y_, (*(signLaplacian[intervalCounter]))(x,y),s_, lKP);
                        }
		}

        /* MemCheck*/
        for(int j=0;j<INTERVAL;j++)
        {
            delete hessian[j];
            delete signLaplacian[j];
        }

    }

    // Compute the descriptors
	return getDescriptor(imgInt,lKP);
}


// Create a keypoint and add it to the list of keypoints
void addKeyPoint(imageIntegral* img,REGULAR_IMAGE i,REGULAR_IMAGE j,bool signL,REGULAR_IMAGE scale,listKeyPoints* lKP)
{
		keyPoint* pt=new keyPoint(i,j,scale,getOrientation(img,  i,j,NUMBER_SECTOR,scale),signL);
		lKP->push_back(pt);
}


// Compute the orientation to assign to a keypoint
float getOrientation(imageIntegral* imgInt,int x,int y,int sectors,REGULAR_IMAGE scale)
{
	REGULAR_IMAGE haarResponseX[sectors];
	REGULAR_IMAGE haarResponseY[sectors];
	REGULAR_IMAGE haarResponseSectorX[sectors];
	REGULAR_IMAGE haarResponseSectorY[sectors];
    INTEGRAL_IMAGE answerX,answerY;
    REGULAR_IMAGE gauss;

    int theta;

    memset(haarResponseSectorX,0,sizeof(REGULAR_IMAGE)*sectors);
    memset(haarResponseSectorY,0,sizeof(REGULAR_IMAGE)*sectors);
    memset(haarResponseX,0,sizeof(REGULAR_IMAGE)*sectors);
    memset(haarResponseY,0,sizeof(REGULAR_IMAGE)*sectors);

	// Computation of the contribution of each angular sectors.
	for( int i = -6 ; i <= 6 ; i++ )
		for( int j = -6 ; j <= 6 ; j++ )
			if( i*i + j*j <= 36  )
			{

				 answerX=haarX(imgInt, x+scale*i,y+scale*j,fround(2*scale));
				 answerY=haarY(imgInt, x+scale*i,y+scale*j,fround(2*scale));

				// Associated angle
				theta=(int) (atan2(answerY,answerX)* sectors/(2*M_PI));
				theta=((theta>=0)?(theta):(theta+sectors));

				// Gaussian weight
                gauss=gaussian(i,j,2);

                // Cumulative answers
				haarResponseSectorX[theta]+=answerX*gauss;
				haarResponseSectorY[theta]+=answerY*gauss;
			}

	// Compute a windowed answer
	for(int i=0 ; i<sectors;i++)
		for(int j=-sectors/12;j<=sectors/12;j++)
			if(0<=i+j && i+j<sectors)
			{
				haarResponseX[i]+=haarResponseSectorX[i+j];
				haarResponseY[i]+=haarResponseSectorY[i+j];
			}
			// The answer can be on any cadrant of the unit circle
			else if( i+j < 0)
			{
				haarResponseX[i]+=haarResponseSectorX[sectors+i+j];
				haarResponseY[i]+=haarResponseSectorY[i+j+sectors];
			}

			else
			{
				haarResponseX[i]+=haarResponseSectorX[i+j-sectors];
				haarResponseY[i]+=haarResponseSectorY[i+j-sectors];
			}



	// Find out the maximum answer
	REGULAR_IMAGE max=haarResponseX[0]*haarResponseX[0]+haarResponseY[0]*haarResponseY[0];

	int t=0;
	for( int i=1 ; i<sectors ; i++ )
	{
		REGULAR_IMAGE norm=haarResponseX[i]*haarResponseX[i]+haarResponseY[i]*haarResponseY[i];
		t=((max<norm)?i:t);
		max=((max<norm)?norm:max);
	}


	// Return the angle ; better than atan which is not defined in pi/2
	return atan2(haarResponseY[t],haarResponseX[t]);
}



// Scale space interpolation as described in Lowe
bool interpolationScaleSpace(image** img,int x, int y, int i, REGULAR_IMAGE &x_, REGULAR_IMAGE &y_, REGULAR_IMAGE &s_, int sample, int octaveValue)
{
	//If we are outside the image...
	if(x<=0 || y<=0 || x>=img[i]->getWidth()-2 || y>=img[i]->getHeight()-2)
		return false;
	REGULAR_IMAGE mx,my,mi,dx,dy,di,dxx,dyy,dii,dxy,dxi,dyi;

    //Nabla X
	dx=((*(img[i]))(x+1,y)-(*(img[i]))(x-1,y))/2;
	dy=((*(img[i]))(x,y+1)-(*(img[i]))(x,y-1))/2;
	di=((*(img[i]))(x,y)-(*(img[i]))(x,y))/2;

    //Hessian X
	REGULAR_IMAGE a=(*(img[i]))(x,y);
	dxx=(*(img[i]))(x+1,y)+(*(img[i]))(x-1,y)-2*a;
	dyy=(*(img[i]))(x,y+1)+(*(img[i]))(x,y+1)-2*a;
	dii=((*(img[i-1]))(x,y)+(*(img[i+1]))(x,y)-2*a);

	dxy=((*(img[i]))(x+1,y+1)-(*(img[i]))(x+1,y-1)-(*(img[i]))(x-1,y+1)+(*(img[i]))(x-1,y-1))/4;
	dxi=((*(img[i+1]))(x+1,y)-(*(img[i+1]))(x-1,y)-(*(img[i-1]))(x+1,y)+(*(img[i-1]))(x-1,y))/4;
	dyi=((*(img[i+1]))(x,y+1)-(*(img[i+1]))(x,y-1)-(*(img[i-1]))(x,y+1)+(*(img[i-1]))(x,y-1))/4;

    // Det
	REGULAR_IMAGE det=dxx*dyy*dii-dxx*dyi*dyi-dyy*dxi*dxi+2*dxi*dyi*dxy-dii*dxy*dxy;

	if(det!=0) //Matrix must be inversible - maybe useless.
	{
		mx=-1/det*(dx*(dyy*dii-dyi*dyi)+dy*(dxi*dyi-dii*dxy)+di*(dxy*dyi-dyy*dxi));
		my=-1/det*(dx*(dxi*dyi-dii*dxy)+dy*(dxx*dii-dxi*dxi)+di*(dxy*dxi-dxx*dyi));
		mi=-1/det*(dx*(dxy*dyi-dyy*dxi)+dy*(dxy*dxi-dxx*dyi)+di*(dxx*dyy-dxy*dxy));

        // If the point is stable
        if(absval(mx)<1 && absval(my)<1 && absval(mi)<1)
        {

            x_=sample*(x+mx)+0.5;// Center the pixels value
            y_=sample*(y+my)+0.5;
            s_=0.4*(1+octaveValue*(i+mi+1));
                return true;
        }

    }
	return false;
}
