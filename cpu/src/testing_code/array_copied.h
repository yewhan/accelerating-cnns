#ifndef AC_CONVOLUTION_LAYER_CONVOLUTION_LAYER_2D_H
#define AC_CONVOLUTION_LAYER_CONVOLUTION_LAYER_2D_H

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



// vectorised m, AKA array copying functions
int optimised_layer_v1_AC_vectorised_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);

int optimised_layer_v2_AC_unroll_x2_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v2_AC_unroll_x4_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v2_AC_unroll_x8_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v2_AC_unroll_m16_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v2_AC_unroll_m32_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v2_AC_unroll_m64_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v2_AC_unroll_x4m16_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v2_AC_unroll_x2m32_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v2_AC_unroll_x3m24_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);

int optimised_layer_v3_AC_x2m32_unroll_d2_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v3_AC_x2m32_unroll_d4_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v3_AC_x2m32_unroll_d8_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v3_AC_x3m24_unroll_d2_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v3_AC_x3m24_unroll_d4_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v3_AC_x3m24_unroll_d8_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);

int optimised_layer_v4_AC_x2m32d8_loop_interchange_m_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v4_AC_x3m24d8_loop_interchange_m_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);

int optimised_layer_v5_AC_x2m32d8_left_shift_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v5_AC_x2m32d8_left_shift_opt_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v5_AC_x2m32d8_left_shift_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v5_AC_x3m24d8_left_shift_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v5_AC_x3m24d8_left_shift_opt_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);

int optimised_layer_v6_AC_x2m32d8_ops_outside_loop_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v6_AC_x3m24d8_ops_outside_loop_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v6_AC_x3m24_ops_outside_loop_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);

int optimised_layer_v7_AC_x2m32d8_omp_1block_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v7_AC_x2m32d8_omp_1block_loop_interchange_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v7_AC_x2m32d8_omp_2block_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v7_AC_x3m24d8_omp_1block_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v7_AC_x3m24d8_omp_1block_loop_interchange_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v7_AC_x3m24d8_omp_2block_loop_interchange_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);

int optimised_layer_v8_AC_x2m32d8_const_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
// int optimised_layer_v8_AC_x2m32d8_const2_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v8_AC_x3m24d8_const_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);

int optimised_layer_v9_AC_x2m32d8_var_declaration_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v9_AC_x3m24d8_var_declaration_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);

int optimised_layer_v10_AC_x2m32d8_subexpr_elim_d_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v10_AC_x2m32d8_subexpr_elim_x_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v10_AC_x3m24d8_subexpr_elim_d_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);

int optimised_layer_v11_AC_x2m32d8_AC_loop_opt_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v11_AC_x4m16d8_AC_loop_opt_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);

int optimised_layer_v12_AC_x2m32d8_loop_interchange_m_y_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v12_AC_x2m32d8_loop_interchange_m_x_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v12_AC_x4m16d8_loop_interchange_m_y_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v12_AC_x4m16d8_loop_interchange_m_x_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);

int optimised_layer_v13_AC_x4m16d8_broadcast_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);

int optimised_layer_v14_AC_x4m16d8_cleaned_code_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);

int optimised_layer_v15_AC_x4m16d8_tiled_m_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);

int optimised_layer_v16_AC_x4m16d8_edge_cases_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v16_AC_x4m16d8_tiled_edge_cases_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v16_AC_x4m16d8_edge_cases_interchange_x_m_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v16_AC_x4m16d8_tiled_edge_cases_interchange_x_m_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);





int optimised_layer_v4_AC_unroll_m16_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v5_AC_register_pressure_d_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v6_AC_register_pressure_x_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v7_AC_strength_reduction_d_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v8_AC_loop_tiling_m_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v9_AC_unroll_d2_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v10_AC_unroll_d4_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v11_AC_unroll_d8_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v12_AC_ops_outside_loop_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v13_AC_sign_unsigned_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v14_AC_omp_2blocks_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);
int optimised_layer_v15_AC_omp_1block_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP);



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