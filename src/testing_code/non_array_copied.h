#ifndef NON_AC_CONVOLUTION_LAYER_CONVOLUTION_LAYER_2D_H
#define NON_AC_CONVOLUTION_LAYER_CONVOLUTION_LAYER_2D_H

#endif //CONVOLUTION_LAYER_CONVOLUTION_LAYER_2D_H

#include "../convolution_layer_2D.h"

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




// vectorised d
int optimised_layer_v1_vectorised_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v1_vectorised_opt_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);

int optimised_layer_v2_unroll_x2_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v2_unroll_x4_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v2_unroll_x4_hadd_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v2_unroll_x8_hadd_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v2_unroll_m2_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v2_unroll_m4_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v2_unroll_m4_hadd_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v2_unroll_m8_hadd_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v2_unroll_x2m2_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v2_unroll_x2m2_hadd_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v2_unroll_x4m2_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v2_unroll_x4m2_hadd_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v2_unroll_x2m4_hadd_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v2_unroll_x4m2_hadd_register_pressure_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v2_unroll_x2m4_hadd_register_pressure_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v2_unroll_x3m3_hadd_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v2_unroll_x3m3_hadd_opt_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);

int optimised_layer_v3_x3m3_unroll_d16_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v3_x3m3_unroll_d16_v2_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v3_x2m4_unroll_d16_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v3_x4m2_unroll_d16_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v3_x2m2_unroll_d16_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v3_x2m2_unroll_d16_opt_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v3_x3m3_unroll_d32_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v3_x3m3_unroll_d32_v2_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v3_x2m4_unroll_d32_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v3_x4m2_unroll_d32_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v3_x2m2_unroll_d32_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v3_x2m2_unroll_d32_opt_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);

int optimised_layer_v4_x3m3_tiled_y_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v4_x3m3_tiled_x_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v4_x3m3_tiled_d_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v4_x2m4_tiled_x_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v4_x2m4_tiled_m_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v4_x2m4_tiled_m_loop_interchange_m_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v4_x2m4_tiled_d_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v4_x2m4_tiled_x_m_loop_interchange_m_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);

int optimised_layer_v5_x2m4_loop_interchange_m_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v5_x3m3_loop_interchange_m_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);

int optimised_layer_v6_x2m4_left_shift_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v6_x2m4_left_shift_opt_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v6_x2m4_left_shift_register_pressure_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v6_x2m4_left_shift_register_pressure_opt_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v6_x3m3_left_shift_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v6_x3m3_left_shift_register_pressure_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v6_x3m3_left_shift_register_pressure_opt_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);

int optimised_layer_v7_x2m4_ops_outside_loop_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v7_x3m3_ops_outside_loop_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);

int optimised_layer_v8_x2m4_inline_relu_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v8_x3m3_inline_relu_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);

int optimised_layer_v9_x2m4_omp_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v9_x2m4_omp_collapsed_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v9_x3m3_omp_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v9_x3m3_omp_collapsed_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);

int optimised_layer_v10_x2m4_const_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v10_x3m3_const_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);

int optimised_layer_v11_x2m4_var_declaration_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v11_x3m3_var_declaration_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);

int optimised_layer_v12_x2m4_fmadd_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v12_x3m3_fmadd_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);

int optimised_layer_v13_x2m4_loop_tiling_m_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v13_x3m3_loop_tiling_m_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);

int optimised_layer_v14_x2m4_cleaned_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v14_x3m3_cleaned_FP_AC(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v14_x4m2_cleaned_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);

int optimised_layer_v15_x4m2_edge_cases_AC(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v15_x3m3_edge_cases_AC(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);




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