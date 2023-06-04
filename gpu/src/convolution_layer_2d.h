/*
------------------DR VASILIOS KELEFOURAS-----------------------------------------------------
------------------COMP3001 ------------------------------------------------------------------
------------------PARALLEL PROGAMMING MODULE-------------------------------------------------
------------------UNIVERSITY OF PLYMOUTH, SCHOOL OF ENGINEERING, COMPUTING AND MATHEMATICS---
*/

#ifndef CONVOLUTION_LAYER_CONVOLUTION_LAYER_2D_H
#define CONVOLUTION_LAYER_CONVOLUTION_LAYER_2D_H

#endif //CONVOLUTION_LAYER_CONVOLUTION_LAYER_2D_H

#include <stdio.h>
#include <time.h>
#include <pmmintrin.h>
#include <process.h>
#include <chrono>
#include <iostream>
#include <immintrin.h>


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <emmintrin.h>
#include <limits.h>
#include <pmmintrin.h>
#include <immintrin.h>
#include <stdint.h>	/* for uint64 definition */
#include <omp.h>

//gpu includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"




float Relu_float(const float temp);
__device__ float Relu_float_kernel(const float temp);

int unoptimized_layer_FP(const float* in, const float* filter, const float* bias_array, float* out_to_compare_with);

int optimised_layer_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);


__global__ void optimised_layer_bmyx_FP_kernel(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
__global__ void optimised_layer_bmy_FP_kernel(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);


//int optimized_layer_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
//int optimized_layer_FP_ver2(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
//int optimized_layer_FP_ver3(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
//int optimized_layer_FP_ver4(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
//int optimized_layer_FP_ver5(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);







//debugging routines
void show_32_bit_in_AVX_register(__m256i temp);


extern unsigned int Input_Output_batch_dim;
extern unsigned int Input_X_dim;
extern unsigned int Input_Y_dim;
extern unsigned int Input_depth_dim;

extern unsigned int Stride_X_dim;
extern unsigned int Stride_Y_dim;
extern unsigned int Stride_Z_dim;

//output dimensions
extern unsigned int Output_X_dim;
extern unsigned int Output_Y_dim;
extern unsigned int Output_depth_dim;
//output batch == input batch

//mask dimensions
extern unsigned int Mask_X_dim;
extern unsigned int Mask_Y_dim;
extern unsigned int Mask_Z_dim;

extern unsigned int TILE;

extern float Scale;
extern unsigned int M0_by_n;
extern unsigned char Zero_point;
extern __m256i M0_by_n_vector;
extern __m256 Scale_vector;


