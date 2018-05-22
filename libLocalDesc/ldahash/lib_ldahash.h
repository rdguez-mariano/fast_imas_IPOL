/**
  * @file lib_ldahash.h
  * @author Mariano Rodríguez
  * @date 2018
  * @brief The IMAS algorithm implementation.
  * @warning This code was extracted and modified from https://cvlab.epfl.ch/research/detect/ldahash. Copyright (C) 2010 by Christoph Strecha
  */

#ifndef _LIB_LDAHASH_H
#define _LIB_LDAHASH_H

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sys/stat.h>

#include "hashpro.h"
#include "imas.h"
#include "libLocalDesc/sift/demo_lib_sift.h"
#include <xmmintrin.h>

#define BIN_WORD unsigned long long


struct ldadescriptor
{
BIN_WORD* ldadesc; //array
keypoint* sift_desc; //pointer
ldadescriptor(int nrdim, int method):dim(nrdim),method_id(method)
{
    ldadesc = new BIN_WORD[nrdim];
}
const int dim;
const int method_id;
};



typedef union F128
{
    __m128 pack;
  float f[4];
} F128;

/// a.b
float sseg_dot(const float* a, const float* b, int sz );
void sseg_matrix_vector_mul(const float* A, int ar, int ac, int ald, const float* b, float* c);

ldadescriptor* lda_describe_from_SIFT(keypoint & siftdesc, int method);
float lda_hamming_distance(ldadescriptor *k1,ldadescriptor *k2, float tdist);

#endif // _LIB_IMAS_H
