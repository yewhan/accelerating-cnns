/*
------------------DR VASILIOS KELEFOURAS-----------------------------------------------------
------------------UNIVERSITY OF PLYMOUTH, SCHOOL OF ENGINEERING, COMPUTING AND MATHEMATICS---

Altered by Euan Hughes

*/

#ifndef CONVOLUTION_LAYER_CONVOLUTION_LAYER_2D_H
#define CONVOLUTION_LAYER_CONVOLUTION_LAYER_2D_H

#endif //CONVOLUTION_LAYER_CONVOLUTION_LAYER_2D_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <immintrin.h>
#include <stdint.h>	/* for uint64 definition */
#include <sched.h>
#include <pthread.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <omp.h>



float Relu_float(const float temp);


int unoptimized_layer_FP(const float* in, const float* filter, const float* bias_array, float* out_to_compare_with);

int optimised_layerv1_arraycopying_vectorised_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layerv2_unroll_x2_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layerv3_unroll_x4_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layerv4_unroll_m16_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layerv5_register_pressure_d_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layerv6_register_pressure_x_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layerv7_strength_reduction_d_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layerv8_loop_tiling_m_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layerv9_unroll_d2_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layerv10_unroll_d4_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layerv11_unroll_d8_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layerv12_ops_outside_loop_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layerv13_arraycopying_sign_unsigned_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layerv14_omp_2blocks_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layerv15_omp_1block_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);



// Quantisation functions
int Relu_int(const int temp);

int unoptimized_layer_Char(const unsigned char* in_Char, const signed char* filter_Char, const int* bias_array_Int, unsigned char* out_to_compare_with_Char);
int optimised_layerv1_vectorised_Char(const unsigned char* in_Char, const signed char* filter_Char, const int* bias_array_Int, unsigned char* out_to_compare_with_Char);
int optimised_layerv1_arraycopying_vectorised_Char(const unsigned char* in_Char, const signed char* filter_Char, const int* bias_array_Int, unsigned char* out_to_compare_with_Char);







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


