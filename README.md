# Fast Image Matching by Affine Simulation

Methods performing Image Matching by Affine Simulation (IMAS) attain affine invariance by applying a finite set of affine transforms to the images. They are based on Scale Invariant Image Matching (SIIM) methods like SIFT or SURF. The following SIIMs are available here:
- SIFT
- Root-SIFT
- HALF-SIFT
- HALF-Root-SIFT
- SURF
- BRISK (only with OpenCV)
- FREAK (only with OpenCV)
- ORB (only with OpenCV)
- BRIEF (only with OpenCV)
- AGAST (only with OpenCV)
- LATCH (only with OpenCV)
- LUCID (only with OpenCV)
- DAISY (only with OpenCV)
- AKAZE (only with OpenCV)

Some [LDAHash descriptors](https://cvlab.epfl.ch/research/detect/ldahash) are available. They are:
- DIF128
- LDA128
- DIF64
- LDA64

Also, those descriptors and matchers introduced in [Affine invariant image comparison under repetitive structures](https://rdguez-mariano.github.io/pages/acdesc) are now available (but still unoptimised). They are:
- AC
- AC-Q
- AC-W

Depending on the SIIM, we propose optimal sets of affine simulations as in [Covering the Space of Tilts](https://rdguez-mariano.github.io/pages/imas).

This version of IMAS is based on the concept of hyper-descriptors and their associated matchers. See [Fast Affine Invariant Image Matching](https://rdguez-mariano.github.io/pages/hyperdescriptors) for more information on this.

## Online demo

Some of these methods are available online at [IPOL](https://ipolcore.ipol.im/demo/clientApp/demo.html?id=225) for you to test with your own images.

*Remark:* The IPOL Journal does not accept OpenCV. So those descriptors only proposed by OpenCV are not available online. Neither are those descriptors and matchers proposed in [ICIP 2018](https://rdguez-mariano.github.io/pages/acdesc). On the other hand, USAC is available !

## Prerequisites
This source code is standalone, although there are two optional capabilities (OpenCV 3.2.0 and the USAC Filter) that require external libraries. If any compilation error arises, is probably due to some missing external libraries.

### Activating OpenCV
In order to use SIIM descriptors proposed by OpenCV you'll need to download, compile and install the OpenCV version 3.2.0.

Then be sure that the CMakeLists.txt file has the OpenCV flag set to ON (.e.g. `set(opencv ON)`) and modify the path in cmake variables to your OpenCV installation accordingly:
- `set (OpenCV_DIR "/path/to/opencv/share/OpenCV")`
- `set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "/path/to/opencv/share/OpenCV")`


### Deactivating OpenCV
Just be sure that the CMakeLists.txt file has the OpenCV flag set to OFF (.e.g. `set(opencv OFF)`).

### Activating USAC
If you want to use the [USAC](http://www.cs.unc.edu/~rraguram/usac/) algorithm you'll need to install some libraries first. On ubuntu you might want to type something like:
```bash
sudo apt-get update
sudo apt-get install libblas-dev liblapack-dev libconfig-dev libconfig++-dev
```

Then be sure that the CMakeLists.txt file has the USAC flag set to ON (.e.g. `set(USAC ON)`).

### Deactivating USAC
Just be sure that the CMakeLists.txt file has the USAC flag set to OFF (.e.g. `set(USAC OFF)`).

### Activating GDAL
If you want to read geospatial data you'll need to install the [GDAL](http://www.gdal.org/) library first. On ubuntu you might want to type something like:

```bash
sudo add-apt-repository ppa:ubuntugis/ppa
sudo apt-get update
sudo apt-get install libgdal
```

Then be sure that the CMakeLists.txt file has the GDAL flag set to ON (.e.g. `set(GDAL ON)`).

### Deactivating GDAL
Just be sure that the CMakeLists.txt file has the GDAL flag set to OFF (.e.g. `set(GDAL OFF)`).



### Activating LDAHash
[LDAHash descriptors](https://cvlab.epfl.ch/research/detect/ldahash) are available by setting the LDAHash flag set to ON in the CMakeLists.txt file (.e.g. `set(LDAHASH ON)`).

### Deactivating LDAHash
Just be sure that the CMakeLists.txt file has the LDAHash flag set to OFF (.e.g. `set(LDAHASH OFF)`).



## Compiling on Linux
```bash
mkdir -p build && cd build && cmake .. && make
```

## Getting Started
Input Arguments:
* "-im1 PATH/im1.png" Selects the query input image.
* "-im2 PATH/im2.png" Selects the target input image.
* "-im3 PATH/im3.png" Selects the a-contrario input image and activates the a-contrario Matcher. **(None by default)**
* "-max_keys_im3 VALUE_N" Sets the maximum number of keypoints to be used for the a-contrario Matcher to VALUE_N. **(All by default)**
* "-applyfilter VALUE_F" Selects the geometric filter to apply, the number VALUE_F stands for:
  - 1 -> ORSA Fundamental
  - 2 -> ORSA Homography **(Default)**
  - 3 -> USAC Fundamental
  - 4 -> USAC Homography
* "-desc VALUE_X" Selects the SIIM method. VALUE_X stands for:
  - 1 -> SIFT
  - 2 -> SURF
  - 11 -> Root-SIFT **(Default)**
  - 21 -> HALF-SIFT
  - 22 -> HALF-ROOT-SIFT
  - 3 -> BRISK
  - 4 -> BRIEF
  - 5 -> ORB
  - 6 -> DAISY
  - 7 -> AKAZE
  - 8 -> LATCH
  - 9 -> FREAK
  - 10 -> LUCID
  - 13 -> AGAST
  - 30 -> AC
  - 31 -> AC-W
  - 32 -> AC-Q
  - 41 -> DIF128
  - 42 -> LDA128
  - 43 -> DIF64
  - 44 -> LDA64
* "-covering VALUE_C" Selects the near optimal covering to be used. Available choices are: 1.4, 1.5, 1.6, 1.7, 1.8, 1.9 and 2. **(1.7 by default)**
* "-match_ratio VALUE_M" Sets the Nearest Neighbour Distance Ratio. VALUE_M is a real number between 0 and 1. **(0.6 for SURF and 0.8 for SIFT based)**
* "-filter_precision VALUE_P" Sets the precision threshold for ORSA or USAC. VALUE_P is normally in terms of pixels. **(3 pixels for Fundamental and 10 pixels for Homography)**
* "-filter_radius rho" It tells IMAS to use rho-hyperdescriptors **(4 pixels by default)**.
* "-fixed_area" Resizes input images to have areas of about 800*600. *This affects the position of matches and all output images*
* "-bigpanorama" Allows to recreate a panorama with no restrictions on size. The frame is computed automatically so as both target and the homography-transformed-query images fit in. *Wild homographies might cause big output panorama images.*
* "-framewidth VALUE_W" Sets the frame width around the target image for the panorama visualisation. The argument "-bigpanorama" overrides this action.
* "-eigen_threshold VALUE_ET" and "-tensor_eigen_threshold VALUE_TT" Controls thresholds for eliminating aberrant descriptors. **(Both set to 10 by default)**

For example, suppose we have two images (adam1.png and adam2.png) on which we want to apply Optimal-Affine-RootSIFT with the near optimal covering of 1.4. This is obtained by typing on bash the following:

```bash
./main -im1 adam1.png -im2 adam2.png -desc 11 -covering 1.4
```

### Reading Geospatial Data
In order to read geospatial data please use the following flags:
* "-im1_gdal PATH/im1.tif XOff YOff XSize YSize" Selects a patch from im1.tif as the query input image.
* "-im2_gdal PATH/im2.tif XOff YOff XSize YSize" Selects a patch from im2.tif as the target input image.

where  (XOff, YOff) are the coordinates of the top left corner of a patch whose width and height are respectively XSize and YSize.

***Remark:*** Use these flags instead of "-im1" and/or "-im2".

For example, the following code will use as query image an extracted 1000x1000 patch whose top-left corner lies on (23000, 5000).

```bash
./main -im1_gdal test.tif 23000 5000 1000 1000 -im2 adam2.png
```

### Output files
* **"covering.png"**. A representation in the Space of Tilts of the applied set of affine simulations. See [Covering the Space of Tilts](https://rdguez-mariano.github.io/pages/imas) for more information on those coverings.
* **"output_hori.png"**, **"output_vert.png"**. Shows resulting matches on images.
* **"output_hori_rich.png"**, **"output_vert_rich.png"**. Shows a representation of the scale and the tilt for each matched descriptor.
* **"data_matches.csv"**. The list of matches containing the following columns:
  - "x1, y1". Coordinates on the query image.  
  - "sigma1, angle1". Scale and angle of the matched query descriptor.
  - "t1_x, t1_y, theta1". Applied affine simulation on which the matched query descriptor was found.
  - "x2, y2". Coordinates on the target image.
  - "sigma2, angle2". Scale and angle of the matched target descriptor.
  - "t2_x, t2_y, theta2". Applied affine simulation on which the matched target descriptor was found.


Additionally, if an underlying homography has been identified by ORSA or USAC, then:
* **"panorama.png"**. Shows the query image, transformed by this homography, on the target image.


## IMAS on MATLAB
These methods are available in MATLAB through MEX. The connection is done by `IMAS_matlab.cpp`. A MATLAB function, `perform_IMAS.m`, reachable at [imas_analytics](https://github.com/rdguez-mariano/imas_analytics), is provided to automatically handle compilation and calls to the MEX function.


## Developer Documentation
This code comes with a doxygen documentation. It can be generated locally from the source code. You might choose to use its graphical front-end: [Doxywizard](https://www.stack.nl/~dimitri/doxygen/manual/doxywizard_usage.html).


## Authors

* **Mariano Rodríguez** - [web page](https://rdguez-mariano.github.io/)
* **Julie Delon** - [web page](https://delon.wp.imt.fr/)
* **Jean-Michel Morel** - [web page](https://sites.google.com/site/jeanmichelmorelcmlaenscachan/)
* **Rafael Grompone Von Gioi**

## Contributors

* **Thibaud Ehret** - [web page](http://perso.eleves.ens-rennes.fr/people/thibaud.ehret/index.html)

## License

This project is licensed under the BSD License - see the [LICENSE](LICENSE) file for details

## Acknowledgements

This project :
* was inspired by the [ASIFT Project](http://demo.ipol.im/demo/my_affine_sift/). Copyright (c) 2011, Guoshen Yu and Jean-Michel Morel / Distributed under the BSD license.
* might be linked to two patents:
  * Jean-Michel Morel and Guoshen Yu,  Method and device for the invariant affine recognition recognition of shapes. U.S. Patent 8687920.
  * David Lowe  "Method and apparatus for identifying scale invariant features in an image and use of same for locating an object in an image",  U.S. Patent 6,711,293.
* The source code in the subdirectory third_party comes from the Eigen library, which is [LGPL-licensed](see http://www.gnu.org/copyleft/lesser.html)
* calls libSimuTilts (all but digital_tilt.cpp), libOrsa, libMatch and libNumerics. Copyright (C) 2007-2010, Lionel Moisan, distributed under the BSD license.
* calls libSimuTilts/digital_tilt.cpp. Copyright (c) 2011, Guoshen Yu and Jean-Michel Morel / Distributed under the BSD license.
* can optionally call libUSAC. Copyright (c) 2012 University of North Carolina at Chapel Hill / See its [web page](http://www.cs.unc.edu/~rraguram/usac/) to see their specific associated licence.

## Github repository

<https://github.com/rdguez-mariano/fast_imas_IPOL>
