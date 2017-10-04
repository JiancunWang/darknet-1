#ifndef CNN_H
#define CNN_H

#include<hls_stream.h>
//#define M 16
//#define K 27
//#define N 600
#define array_size(x,y) x*y

// #include "typedefs.h"
//#include "training_data.h"

// The K_CONST value: number of nearest neighbors
//#define K_CONST 3

// Top function for synthesis
void dut(
    hls::stream<float> &cnn_in,
    hls::stream<float> &cnn_out
);

// Top function for digit recognition

// Given the testing instance and a (new) training instance,
// this function is expected to maintain/update an array of
// K minimum distances per training set

// void cnn_compute( int M, int N, int K, float A[M*K], float B[N*K], float &C[M*N*K] );
void cnn_compute( float A[array_size(27,16)], float B[array_size(27,600)], float C[array_size(16,600)]) ;
void init_weights_hw(float Weights[16][27]);
void cnn_compute_kernel(float W[16][27], float b_buf[27][600], float c_buf[16][600]);
#endif
