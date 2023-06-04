#ifndef CONVOLUTION_LAYER_QUANT_CONV_LAYER_2D_H
#define CONVOLUTION_LAYER_QUANT_CONV_LAYER_2D_H

#endif //CONVOLUTION_LAYER_QUANT_CONV_LAYER_2D_H

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


int profile_Char(const unsigned char* in_Char, const signed char* filter_Char, const int* bias_array_Int, unsigned char* out_to_compare_with_Char);


int opt_m2x4_deep_AVX_d64_Char(const unsigned char* in_Char, const signed char* filter_Char, const int* bias_array_Int, unsigned char* out_to_compare_with_Char);
int opt_m2x4_deep_AVX_d32_Char(const unsigned char* in_Char, const signed char* filter_Char, const int* bias_array_Int, unsigned char* out_to_compare_with_Char);
int opt_m2x4_deep_SSE_d16_Char(const unsigned char* in_Char, const signed char* filter_Char, const int* bias_array_Int, unsigned char* out_to_compare_with_Char);


int opt_m2x4_deep_d64_m_fallback_Char(const unsigned char* in_Char, const signed char* filter_Char, const int* bias_array_Int, unsigned char* out_to_compare_with_Char);
int opt_m2x4_d8_Char(const unsigned char* in_Char, const signed char* filter_Char, const int* bias_array_Int, unsigned char* out_to_compare_with_Char);
int opt_m4x2_Char(const unsigned char* in_Char, const signed char* filter_Char, const int* bias_array_Int, unsigned char* out_to_compare_with_Char);
int opt_m2x4_Char(const unsigned char* in_Char, const signed char* filter_Char, const int* bias_array_Int, unsigned char* out_to_compare_with_Char);
int opt_m2x2_Char(const unsigned char* in_Char, const signed char* filter_Char, const int* bias_array_Int, unsigned char* out_to_compare_with_Char);
int opt_m3x3_Char(const unsigned char* in_Char, const signed char* filter_Char, const int* bias_array_Int, unsigned char* out_to_compare_with_Char);



// iterative opts
int optimised_layerv1_vectorised_Char(const unsigned char* in_Char, const signed char* filter_Char, const int* bias_array_Int, unsigned char* out_to_compare_with_Char);
int optimised_layerv1_arraycopying_vectorised_Char(const unsigned char* in_Char, const signed char* filter_Char, const int* bias_array_Int, unsigned char* out_to_compare_with_Char);
int optimised_layerv2_unroll_x2_Char(const unsigned char* in_Char, const signed char* filter_Char, const int* bias_array_Int, unsigned char* out_to_compare_with_Char);
int optimised_layerv3_unroll_m2_Char(const unsigned char* in_Char, const signed char* filter_Char, const int* bias_array_Int, unsigned char* out_to_compare_with_Char);
int optimised_layerv4_general_register_pressure_d_Char(const unsigned char* in_Char, const signed char* filter_Char, const int* bias_array_Int, unsigned char* out_to_compare_with_Char);
int optimised_layerv5_loop_tiling_d_Char(const unsigned char* in_Char, const signed char* filter_Char, const int* bias_array_Int, unsigned char* out_to_compare_with_Char);


// ***** unoptimised functions *****
int unoptimized_layer_Char(const unsigned char* in_Char, const signed char* filter_Char, const int* bias_array_Int, unsigned char* out_to_compare_with_Char);

int Relu_int(const int temp);



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