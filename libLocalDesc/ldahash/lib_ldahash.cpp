/**
  * @file lib_ldahash.cpp
  * @author Mariano Rodríguez
  * @date 2018
  * @brief The IMAS algorithm implementation.
  * @warning This code was extracted and modified from https://cvlab.epfl.ch/research/detect/ldahash. Copyright (C) 2010 by Christoph Strecha
  */

#include "lib_ldahash.h"


using namespace std;

/// a.b
float sseg_dot(const float* a, const float* b, int sz )
{
    int ksimdlen = sz/4*4;
    __m128 xmm_a, xmm_b;
    F128   xmm_s;
    float sum;
    int j;
    xmm_s.pack = _mm_set1_ps(0.0);
    for( j=0; j<ksimdlen; j+=4 ) {
        xmm_a = _mm_loadu_ps((float*)(a+j));
        xmm_b = _mm_loadu_ps((float*)(b+j));
        xmm_s.pack = _mm_add_ps(xmm_s.pack, _mm_mul_ps(xmm_a,xmm_b));
    }
    sum = xmm_s.f[0]+xmm_s.f[1]+xmm_s.f[2]+xmm_s.f[3];
    for(; j<sz; j++ ) sum+=a[j]*b[j];
    return sum;
}

/// c = Ab
/// A   : matrix
/// ar  : # rows of A
/// ald : # columns of A -> leading dimension as in blas
/// ac  : size of the active part in the row
/// b   : vector with ac size
/// c   : resulting vector with ac size

void sseg_matrix_vector_mul(const float* A, int ar, int ac, int ald, const float* b, float* c)
{
    for( int r=0; r<ar; r++ )
        c[r] = sseg_dot(A+r*ald, b, ac);
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ldadescriptor* lda_describe_from_SIFT(keypoint & siftdesc, int method)
{

    assert(siftdesc.veclength==128);
    BIN_WORD *singleBitWord = new BIN_WORD[64];

    // compute words with particular bit turned on
    // singleBitWord[0] :  000000...000001  <=> 1 = 2^0
    // singleBitWord[1] :  000000...000010  <=> 2 = 2^1
    // singleBitWord[2] :  000000...000100  <=> 4 = 2^2
    // singleBitWord[3] :  000000...001000  <=> 8 = 2^3

    singleBitWord[0] = 1;
    for (int i=1; i < 64; i++)
    {
        singleBitWord[i] = singleBitWord[0] << i;
    }

    ldadescriptor* ldadesc = 0;

    int nrDim;
    if(method == IMAS_DIF128)   {nrDim=2; }
    if(method == IMAS_LDA128)   {nrDim=2; }
    if(method == IMAS_DIF64)    {nrDim=1; }
    if(method == IMAS_LDA64)    {nrDim=1; }

    BIN_WORD b;
    float provec[128];

    switch (method) {

    case IMAS_DIF128 : {
        sseg_matrix_vector_mul(Adif128, 128, 128, 128, siftdesc.vec, provec);
        b = 0;
        for(int k=0; k < 64; k++){
            if(provec[k] + tdif128[k] <= 0.0) b |= singleBitWord[k];
        }
        ldadesc = new ldadescriptor(nrDim,method);
        ldadesc->ldadesc[0] = b;
        b = 0;
        for(int k=0; k < 64; k++){
            if(provec[k+64] + tdif128[k+64] <= 0.0) b |= singleBitWord[k];
        }
        ldadesc->ldadesc[1] = b;
        ldadesc->sift_desc = &siftdesc;

        break;
    }
    case IMAS_LDA128 : {
        sseg_matrix_vector_mul(Alda128, 128, 128, 128, siftdesc.vec, provec);
        b = 0;
        for(int k=0; k < 64; k++){
            if(provec[k] + tlda128[k] <= 0.0) b |= singleBitWord[k];
        }
        ldadesc = new ldadescriptor(nrDim,method);
        ldadesc->ldadesc[0] = b;
        b = 0;
        for(int k=0; k < 64; k++){
            if(provec[k+64] + tlda128[k+64] <= 0.0) b |= singleBitWord[k];
        }
        ldadesc->ldadesc[1] = b;
        ldadesc->sift_desc = &siftdesc;

        break;
    }
    case IMAS_DIF64 : {
        sseg_matrix_vector_mul(Adif64, 64, 128, 128, siftdesc.vec, provec);
        b = 0;
        for(int k=0; k < 64; k++) {
            if(provec[k] + tdif64[k]  <= 0.0) b |= singleBitWord[k];
        }

        ldadesc = new ldadescriptor(nrDim,method);
        ldadesc->sift_desc = &siftdesc;
        ldadesc->ldadesc[0] = b;
        break;
    }
    case IMAS_LDA64 : {
        sseg_matrix_vector_mul(Alda64, 64, 128, 128, siftdesc.vec, provec);
        b = 0;
        for(int k=0; k < 64; k++) {
            if(provec[k] + tlda64[k]  <= 0.0) b |= singleBitWord[k];
        }

        ldadesc = new ldadescriptor(nrDim,method);
        ldadesc->sift_desc = &siftdesc;
        ldadesc->ldadesc[0] = b;
        break;
    }
    }

    return(ldadesc);
}



float lda_hamming_distance(ldadescriptor *k1,ldadescriptor *k2, float tdist)
{
    float dist = __builtin_popcountll(k2->ldadesc[0] ^ k1->ldadesc[0]);
    for(int j = 1; (j < k1->dim)&&(dist<tdist); j++)
    {
        dist += __builtin_popcountll(k2->ldadesc[j] ^ k1->ldadesc[j]);
    }
    return(dist);
}

