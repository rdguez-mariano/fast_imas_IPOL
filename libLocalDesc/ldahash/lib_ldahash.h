
/***************************************************************************
 *   Copyright (C) 2010 by Christoph Strecha   *
 *   christoph.strecha@epfl.ch   *
 ***************************************************************************/

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
#include "../sift/demo_lib_sift.h"
#include <xmmintrin.h>

#define BIN_WORD unsigned long long


struct ldadescriptor
{
BIN_WORD* ldadesc; //array
keypoint* sift_desc; //pointer
ldadescriptor(int nrdim, int method);
int get_method_id() {return(method_id);}
int get_dim(){return(dim);}
private:
int dim;
int method_id;
};

ldadescriptor::ldadescriptor(int nrdim, int method)
{
    dim = nrdim;
    ldadesc = new BIN_WORD[nrdim];
    method_id = method;
}

typedef union F128
{
    __m128 pack;
  float f[4];
} F128;

/// a.b
float sseg_dot(const float* a, const float* b, int sz );
void sseg_matrix_vector_mul(const float* A, int ar, int ac, int ald, const float* b, float* c);

#endif // _LIB_IMAS_H
