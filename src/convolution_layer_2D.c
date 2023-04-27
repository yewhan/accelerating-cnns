#include "convolution_layer_2D.h"
#include <xmmintrin.h>

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))




// loop copying applied, to enable vectorisation
// avx instructions applied
// moved bias load outside of x to m loop
// 10 GFLOPS, 5x speedup from unopt using -O3
int optimised_layer_v1_AC_vectorised_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  __m256 bias, temp, s, w;

  unsigned int filter_FP_length = Output_depth_dim * Mask_Y_dim * Mask_X_dim * Input_depth_dim;
  float* filter_FP_copy = (float*)_mm_malloc(filter_FP_length * sizeof(float), 64);
  if (filter_FP_copy == NULL) {
    printf("\nerror with malloc allocating filter array copy");
    exit(EXIT_FAILURE);
  }

  // array copying - filter_FP into form usable for vectorising m loop
  for (unsigned int y = 0; y < Mask_Y_dim; y++) {
    for (unsigned int x = 0; x < Mask_X_dim; x++) {
      for (unsigned int d = 0; d < Input_depth_dim; d++) {
        for (unsigned int m = 0; m < Output_depth_dim; m++) {
          unsigned long long int old_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
            + y * Mask_X_dim * Input_depth_dim
            + x * Input_depth_dim
            + d;
              
          unsigned long long int new_subscript = y * Mask_X_dim * Input_depth_dim * Output_depth_dim
            + x * Input_depth_dim * Output_depth_dim
            + d * Output_depth_dim
            + m;

          filter_FP_copy[new_subscript] = filter_FP[old_subscript];
        }
      }
    }
  }


  // main loop body
  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int m = 0; m < Output_depth_dim; m+=8) { //channels
      bias = _mm256_load_ps(&bias_array_FP[m]);

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (unsigned int x = 0; x < Output_X_dim; x++) {	//Output Width
          temp = _mm256_setzero_ps();
          // bias = bias_array_FP[m];
          // temp = 0.0f;
          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d++) {

                unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + d;

                unsigned long long int filter_subscript = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + m;

                
                s = _mm256_set1_ps(in_FP[in_subscript]);
                w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                temp = _mm256_fmadd_ps(s, w, temp);
                // float s = in_FP[in_subscript];
                // float w = filter_FP[filter_subscript];
                // temp = temp + s * w;
              }
            }
          }


          unsigned long long int out_subscript = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + m;

          
          temp = _mm256_add_ps(temp, bias);
          temp = _mm256_max_ps(temp, _mm256_setzero_ps()); // merge Relu layer via native AVX intrinsics
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript], temp);
          // temp += bias;
          // out_to_compare_with_FP[out_subscript] = Relu_float(temp);

        }
      }
    }
  }
  printf("\n from optimised_layer_v1 %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  _mm_free(filter_FP_copy);
  return 0;
}


// register blocking/ loop unroll x by factor of 2 (x+=2)
// 18 GFLOPS, 1.8x speedup from v1
int optimised_layer_v2_AC_unroll_x2_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  __m256 bias, temp, temp2, s, s2, w;

  unsigned int filter_FP_length = Output_depth_dim * Mask_Y_dim * Mask_X_dim * Input_depth_dim;
  float* filter_FP_copy = (float*)_mm_malloc(filter_FP_length * sizeof(float), 64);
  if (filter_FP_copy == NULL) {
    printf("\nerror with malloc allocating filter array copy");
    exit(EXIT_FAILURE);
  }

  // array copying - filter_FP into form usable for vectorising m loop
  for (unsigned int y = 0; y < Mask_Y_dim; y++) {
    for (unsigned int x = 0; x < Mask_X_dim; x++) {
      for (unsigned int d = 0; d < Input_depth_dim; d++) {
        for (unsigned int m = 0; m < Output_depth_dim; m++) {
          unsigned long long int old_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
            + y * Mask_X_dim * Input_depth_dim
            + x * Input_depth_dim
            + d;
              
          unsigned long long int new_subscript = y * Mask_X_dim * Input_depth_dim * Output_depth_dim
            + x * Input_depth_dim * Output_depth_dim
            + d * Output_depth_dim
            + m;

          filter_FP_copy[new_subscript] = filter_FP[old_subscript];
        }
      }
    }
  }


  // main loop body
  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int m = 0; m < Output_depth_dim; m+=8) { //channels
      bias = _mm256_load_ps(&bias_array_FP[m]);

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (unsigned int x = 0; x < Output_X_dim; x+=2) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          // bias = bias_array_FP[m];
          // temp = 0.0f;
          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d++) {

                unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + d;
                unsigned long long int in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + d;

                unsigned long long int filter_subscript = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + m;

                
                s = _mm256_set1_ps(in_FP[in_subscript]);
                s2 = _mm256_set1_ps(in_FP[in_subscript2]);

                w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);

                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                // float s = in_FP[in_subscript];
                // float w = filter_FP[filter_subscript];
                // temp = temp + s * w;
              }
            }
          }


          unsigned long long int out_subscript = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + m;

          unsigned long long int out_subscript2 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + m;

          
          temp = _mm256_add_ps(temp, bias);
          temp = _mm256_max_ps(temp, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript], temp);

          temp2 = _mm256_add_ps(temp2, bias);
          temp2 = _mm256_max_ps(temp2, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript2], temp2);
          // temp += bias;
          // out_to_compare_with_FP[out_subscript] = Relu_float(temp);

        }
      }
    }
  }
  printf("\n from optimised_layer_v2 %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  _mm_free(filter_FP_copy);
  return 0;
}


// register block/ unroll x by factor of 2 again (x+=4)
// 34 GFLOPS, 1.88x speedup from v2
int optimised_layer_v3_AC_unroll_x4_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  __m256 bias, temp, temp2, temp3, temp4, s, s2, s3, s4, w;

  unsigned int filter_FP_length = Output_depth_dim * Mask_Y_dim * Mask_X_dim * Input_depth_dim;
  float* filter_FP_copy = (float*)_mm_malloc(filter_FP_length * sizeof(float), 64);
  if (filter_FP_copy == NULL) {
    printf("\nerror with malloc allocating filter array copy");
    exit(EXIT_FAILURE);
  }

  // array copying - filter_FP into form usable for vectorising m loop
  for (unsigned int y = 0; y < Mask_Y_dim; y++) {
    for (unsigned int x = 0; x < Mask_X_dim; x++) {
      for (unsigned int d = 0; d < Input_depth_dim; d++) {
        for (unsigned int m = 0; m < Output_depth_dim; m++) {
          unsigned long long int old_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
            + y * Mask_X_dim * Input_depth_dim
            + x * Input_depth_dim
            + d;
              
          unsigned long long int new_subscript = y * Mask_X_dim * Input_depth_dim * Output_depth_dim
            + x * Input_depth_dim * Output_depth_dim
            + d * Output_depth_dim
            + m;

          filter_FP_copy[new_subscript] = filter_FP[old_subscript];
        }
      }
    }
  }


  // main loop body
  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int m = 0; m < Output_depth_dim; m+=8) { //channels
      bias = _mm256_load_ps(&bias_array_FP[m]);

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (unsigned int x = 0; x < Output_X_dim; x+=4) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          temp4 = _mm256_setzero_ps();
          // bias = bias_array_FP[m];
          // temp = 0.0f;
          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d++) {

                unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + d;
                unsigned long long int in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + d;
                unsigned long long int in_subscript3 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+2) * Stride_X_dim + off_x) * Input_depth_dim
                  + d;
                unsigned long long int in_subscript4 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+3) * Stride_X_dim + off_x) * Input_depth_dim
                  + d;

                unsigned long long int filter_subscript = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + m;
                

                s = _mm256_set1_ps(in_FP[in_subscript]); // in_FP[b][y+off_y][x+off_x][d]
                s2 = _mm256_set1_ps(in_FP[in_subscript2]);
                s3 = _mm256_set1_ps(in_FP[in_subscript3]);
                s4 = _mm256_set1_ps(in_FP[in_subscript4]);

                w = _mm256_load_ps(&filter_FP_copy[filter_subscript]); // filter_FP[off_y][off_x][d][m]

                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s3, w, temp3);
                temp4 = _mm256_fmadd_ps(s4, w, temp4);
                // float s = in_FP[in_subscript];
                // float w = filter_FP[filter_subscript];
                // temp = temp + s * w;
              }
            }
          }


          unsigned long long int out_subscript = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + m;
          unsigned long long int out_subscript2 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + m;
          unsigned long long int out_subscript3 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+2) * Output_depth_dim
            + m;
          unsigned long long int out_subscript4 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+3) * Output_depth_dim
            + m;

          
          temp = _mm256_add_ps(temp, bias);
          temp = _mm256_max_ps(temp, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript], temp);

          temp2 = _mm256_add_ps(temp2, bias);
          temp2 = _mm256_max_ps(temp2, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript2], temp2);

          temp3 = _mm256_add_ps(temp3, bias);
          temp3 = _mm256_max_ps(temp3, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript3], temp3);

          temp4 = _mm256_add_ps(temp4, bias);
          temp4 = _mm256_max_ps(temp4, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript4], temp4);
          // temp += bias;
          // out_to_compare_with_FP[out_subscript] = Relu_float(temp);

        }
      }
    }
  }
  printf("\n from optimised_layer_v3 %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  _mm_free(filter_FP_copy);
  return 0;
}


// register block/ unroll m by factor of 2 (m+=16)
// 61 GFLOPS, 1.79x speedup from v3
int optimised_layer_v4_AC_unroll_m16_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  __m256 bias, bias2, temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8, s, s2, s3, s4, w, w2;

  unsigned int filter_FP_length = Output_depth_dim * Mask_Y_dim * Mask_X_dim * Input_depth_dim;
  float* filter_FP_copy = (float*)_mm_malloc(filter_FP_length * sizeof(float), 64);
  if (filter_FP_copy == NULL) {
    printf("\nerror with malloc allocating filter array copy");
    exit(EXIT_FAILURE);
  }

  // array copying - filter_FP into form usable for vectorising m loop
  for (unsigned int y = 0; y < Mask_Y_dim; y++) {
    for (unsigned int x = 0; x < Mask_X_dim; x++) {
      for (unsigned int d = 0; d < Input_depth_dim; d++) {
        for (unsigned int m = 0; m < Output_depth_dim; m++) {
          unsigned long long int old_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
            + y * Mask_X_dim * Input_depth_dim
            + x * Input_depth_dim
            + d;
              
          unsigned long long int new_subscript = y * Mask_X_dim * Input_depth_dim * Output_depth_dim
            + x * Input_depth_dim * Output_depth_dim
            + d * Output_depth_dim
            + m;

          filter_FP_copy[new_subscript] = filter_FP[old_subscript];
        }
      }
    }
  }


  // main loop body
  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int m = 0; m < Output_depth_dim; m+=16) { //channels
      bias = _mm256_load_ps(&bias_array_FP[m]);
      bias2 = _mm256_load_ps( &bias_array_FP[m+8]);

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (unsigned int x = 0; x < Output_X_dim; x+=4) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          temp4 = _mm256_setzero_ps();
          temp5 = _mm256_setzero_ps();
          temp6 = _mm256_setzero_ps();
          temp7 = _mm256_setzero_ps();
          temp8 = _mm256_setzero_ps();
          // bias = bias_array_FP[m];
          // temp = 0.0f;
          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d++) {

                unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + d;
                unsigned long long int in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + d;
                unsigned long long int in_subscript3 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+2) * Stride_X_dim + off_x) * Input_depth_dim
                  + d;
                unsigned long long int in_subscript4 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+3) * Stride_X_dim + off_x) * Input_depth_dim
                  + d;

                unsigned long long int filter_subscript = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + m;
                unsigned long long int filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + (m+8);
                

                s = _mm256_set1_ps(in_FP[in_subscript]); // in_FP[b][y+off_y][x+off_x][d]
                s2 = _mm256_set1_ps(in_FP[in_subscript2]);
                s3 = _mm256_set1_ps(in_FP[in_subscript3]);
                s4 = _mm256_set1_ps(in_FP[in_subscript4]);

                w = _mm256_load_ps(&filter_FP_copy[filter_subscript]); // filter_FP[off_y][off_x][d][m]
                w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript2]);

                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s3, w, temp3);
                temp4 = _mm256_fmadd_ps(s4, w, temp4);
                temp5 = _mm256_fmadd_ps(s, w2, temp5);
                temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                temp8 = _mm256_fmadd_ps(s4, w2, temp8);
                // float s = in_FP[in_subscript];
                // float w = filter_FP[filter_subscript];
                // temp = temp + s * w;
              }
            }
          }


          unsigned long long int out_subscript = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + m;
          unsigned long long int out_subscript2 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + m;
          unsigned long long int out_subscript3 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+2) * Output_depth_dim
            + m;
          unsigned long long int out_subscript4 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+3) * Output_depth_dim
            + m;

          unsigned long long int out_subscript5 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+8);
          unsigned long long int out_subscript6 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+8);
          unsigned long long int out_subscript7 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+2) * Output_depth_dim
            + (m+8);
          unsigned long long int out_subscript8 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+3) * Output_depth_dim
            + (m+8);

          
          temp = _mm256_add_ps(temp, bias);
          temp = _mm256_max_ps(temp, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript], temp);

          temp2 = _mm256_add_ps(temp2, bias);
          temp2 = _mm256_max_ps(temp2, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript2], temp2);

          temp3 = _mm256_add_ps(temp3, bias);
          temp3 = _mm256_max_ps(temp3, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript3], temp3);

          temp4 = _mm256_add_ps(temp4, bias);
          temp4 = _mm256_max_ps(temp4, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript4], temp4); 

          temp5 = _mm256_add_ps(temp5, bias2);
          temp5 = _mm256_max_ps(temp5, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript5], temp5);

          temp6 = _mm256_add_ps(temp6, bias2);
          temp6 = _mm256_max_ps(temp6, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript6], temp6);

          temp7 = _mm256_add_ps(temp7, bias2);
          temp7 = _mm256_max_ps(temp7, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript7], temp7);

          temp8 = _mm256_add_ps(temp8, bias2);
          temp8 = _mm256_max_ps(temp8, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript8], temp8);
          // temp += bias;
          // out_to_compare_with_FP[out_subscript] = Relu_float(temp);

        }
      }
    }
  }
  printf("\n from optimised_layer_v4 %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  _mm_free(filter_FP_copy);
  return 0;
}


// reduce register pressure on d loop, as v4 uses 20 general purpose registers, now uses 16
// also strength reduction on d loop (spent less time computing values)
// 64 GFLOPS, 1.05x speedup from v4
int optimised_layer_v5_AC_register_pressure_d_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  __m256 bias, bias2, temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8, s, s2, s3, s4, w, w2;

  unsigned int filter_FP_length = Output_depth_dim * Mask_Y_dim * Mask_X_dim * Input_depth_dim;
  float* filter_FP_copy = (float*)_mm_malloc(filter_FP_length * sizeof(float), 64);
  if (filter_FP_copy == NULL) {
    printf("\nerror with malloc allocating filter array copy");
    exit(EXIT_FAILURE);
  }

  // array copying - filter_FP into form usable for vectorising m loop
  for (unsigned int y = 0; y < Mask_Y_dim; y++) {
    for (unsigned int x = 0; x < Mask_X_dim; x++) {
      for (unsigned int d = 0; d < Input_depth_dim; d++) {
        for (unsigned int m = 0; m < Output_depth_dim; m++) {
          unsigned long long int old_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
            + y * Mask_X_dim * Input_depth_dim
            + x * Input_depth_dim
            + d;
              
          unsigned long long int new_subscript = y * Mask_X_dim * Input_depth_dim * Output_depth_dim
            + x * Input_depth_dim * Output_depth_dim
            + d * Output_depth_dim
            + m;

          filter_FP_copy[new_subscript] = filter_FP[old_subscript];
        }
      }
    }
  }


  // main loop body
  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int m = 0; m < Output_depth_dim; m+=16) { //channels
      bias = _mm256_load_ps(&bias_array_FP[m]);
      bias2 = _mm256_load_ps( &bias_array_FP[m+8]);

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (unsigned int x = 0; x < Output_X_dim; x+=4) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          temp4 = _mm256_setzero_ps();
          temp5 = _mm256_setzero_ps();
          temp6 = _mm256_setzero_ps();
          temp7 = _mm256_setzero_ps();
          temp8 = _mm256_setzero_ps();
          // bias = bias_array_FP[m];
          // temp = 0.0f;
          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d++) {

                unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + d;
                // unsigned long long int in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                //   + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                //   + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                //   + d;
                // unsigned long long int in_subscript3 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                //   + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                //   + ((x+2) * Stride_X_dim + off_x) * Input_depth_dim
                //   + d;
                // unsigned long long int in_subscript4 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                //   + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                //   + ((x+3) * Stride_X_dim + off_x) * Input_depth_dim
                //   + d;

                unsigned long long int filter_subscript = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + m;
                // unsigned long long int filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                //   + off_x * Input_depth_dim * Output_depth_dim
                //   + d * Output_depth_dim
                //   + (m+8);
                

                s = _mm256_set1_ps(in_FP[in_subscript]); // in_FP[b][y+off_y][x+off_x][d]
                s2 = _mm256_set1_ps(in_FP[in_subscript + Input_depth_dim]); // x+1
                s3 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 2)]); // x+2 
                s4 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 3)]); // x+3

                w = _mm256_load_ps(&filter_FP_copy[filter_subscript]); // filter_FP[off_y][off_x][d][m]
                w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);

                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s3, w, temp3);
                temp4 = _mm256_fmadd_ps(s4, w, temp4);
                temp5 = _mm256_fmadd_ps(s, w2, temp5);
                temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                temp8 = _mm256_fmadd_ps(s4, w2, temp8);
                // float s = in_FP[in_subscript];
                // float w = filter_FP[filter_subscript];
                // temp = temp + s * w;
              }
            }
          }


          unsigned long long int out_subscript = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + m;
          unsigned long long int out_subscript2 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + m;
          unsigned long long int out_subscript3 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+2) * Output_depth_dim
            + m;
          unsigned long long int out_subscript4 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+3) * Output_depth_dim
            + m;

          unsigned long long int out_subscript5 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+8);
          unsigned long long int out_subscript6 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+8);
          unsigned long long int out_subscript7 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+2) * Output_depth_dim
            + (m+8);
          unsigned long long int out_subscript8 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+3) * Output_depth_dim
            + (m+8);

          
          temp = _mm256_add_ps(temp, bias);
          temp = _mm256_max_ps(temp, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript], temp);

          temp2 = _mm256_add_ps(temp2, bias);
          temp2 = _mm256_max_ps(temp2, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript2], temp2);

          temp3 = _mm256_add_ps(temp3, bias);
          temp3 = _mm256_max_ps(temp3, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript3], temp3);

          temp4 = _mm256_add_ps(temp4, bias);
          temp4 = _mm256_max_ps(temp4, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript4], temp4); 

          temp5 = _mm256_add_ps(temp5, bias2);
          temp5 = _mm256_max_ps(temp5, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript5], temp5);

          temp6 = _mm256_add_ps(temp6, bias2);
          temp6 = _mm256_max_ps(temp6, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript6], temp6);

          temp7 = _mm256_add_ps(temp7, bias2);
          temp7 = _mm256_max_ps(temp7, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript7], temp7);

          temp8 = _mm256_add_ps(temp8, bias2);
          temp8 = _mm256_max_ps(temp8, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript8], temp8);
          // temp += bias;
          // out_to_compare_with_FP[out_subscript] = Relu_float(temp);

        }
      }
    }
  }
  printf("\n from optimised_layer_v5 %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  _mm_free(filter_FP_copy);
  return 0;
}


// reduce register pressure on x loop, as v5 uses 15, v6 uses 8 (shouldn't have huge impact)
// also strength reduction on x
// 64 GFLOPS, no noticable difference from v5, but worth keeping
int optimised_layer_v6_AC_register_pressure_x_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  __m256 bias, bias2, temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8, s, s2, s3, s4, w, w2;

  unsigned int filter_FP_length = Output_depth_dim * Mask_Y_dim * Mask_X_dim * Input_depth_dim;
  float* filter_FP_copy = (float*)_mm_malloc(filter_FP_length * sizeof(float), 64);
  if (filter_FP_copy == NULL) {
    printf("\nerror with malloc allocating filter array copy");
    exit(EXIT_FAILURE);
  }

  // array copying - filter_FP into form usable for vectorising m loop
  for (unsigned int y = 0; y < Mask_Y_dim; y++) {
    for (unsigned int x = 0; x < Mask_X_dim; x++) {
      for (unsigned int d = 0; d < Input_depth_dim; d++) {
        for (unsigned int m = 0; m < Output_depth_dim; m++) {
          unsigned long long int old_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
            + y * Mask_X_dim * Input_depth_dim
            + x * Input_depth_dim
            + d;
              
          unsigned long long int new_subscript = y * Mask_X_dim * Input_depth_dim * Output_depth_dim
            + x * Input_depth_dim * Output_depth_dim
            + d * Output_depth_dim
            + m;

          filter_FP_copy[new_subscript] = filter_FP[old_subscript];
        }
      }
    }
  }


  // main loop body
  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int m = 0; m < Output_depth_dim; m+=16) { //channels
      bias = _mm256_load_ps(&bias_array_FP[m]);
      bias2 = _mm256_load_ps( &bias_array_FP[m+8]);

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (unsigned int x = 0; x < Output_X_dim; x+=4) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          temp4 = _mm256_setzero_ps();
          temp5 = _mm256_setzero_ps();
          temp6 = _mm256_setzero_ps();
          temp7 = _mm256_setzero_ps();
          temp8 = _mm256_setzero_ps();
          // bias = bias_array_FP[m];
          // temp = 0.0f;
          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d++) {

                unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + d;
                // unsigned long long int in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                //   + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                //   + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                //   + d;
                // unsigned long long int in_subscript3 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                //   + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                //   + ((x+2) * Stride_X_dim + off_x) * Input_depth_dim
                //   + d;
                // unsigned long long int in_subscript4 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                //   + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                //   + ((x+3) * Stride_X_dim + off_x) * Input_depth_dim
                //   + d;

                unsigned long long int filter_subscript = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + m;
                // unsigned long long int filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                //   + off_x * Input_depth_dim * Output_depth_dim
                //   + d * Output_depth_dim
                //   + (m+8);
                

                s = _mm256_set1_ps(in_FP[in_subscript]); // in_FP[b][y+off_y][x+off_x][d]
                s2 = _mm256_set1_ps(in_FP[in_subscript + Input_depth_dim]); // x+1
                s3 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 2)]); // x+2 
                s4 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 3)]); // x+3

                w = _mm256_load_ps(&filter_FP_copy[filter_subscript]); // filter_FP[off_y][off_x][d][m]
                w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);

                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s3, w, temp3);
                temp4 = _mm256_fmadd_ps(s4, w, temp4);
                temp5 = _mm256_fmadd_ps(s, w2, temp5);
                temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                temp8 = _mm256_fmadd_ps(s4, w2, temp8);
                // float s = in_FP[in_subscript];
                // float w = filter_FP[filter_subscript];
                // temp = temp + s * w;
              }
            }
          }


          unsigned long long int out_subscript = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + m;
          // unsigned long long int out_subscript2 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
          //   y * (Output_depth_dim * Output_X_dim) +
          //   (x+1) * Output_depth_dim
          //   + m;
          // unsigned long long int out_subscript3 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
          //   y * (Output_depth_dim * Output_X_dim) +
          //   (x+2) * Output_depth_dim
          //   + m;
          // unsigned long long int out_subscript4 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
          //   y * (Output_depth_dim * Output_X_dim) +
          //   (x+3) * Output_depth_dim
          //   + m;

          // unsigned long long int out_subscript5 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
          //   y * (Output_depth_dim * Output_X_dim) +
          //   x * Output_depth_dim
          //   + (m+8);
          // unsigned long long int out_subscript6 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
          //   y * (Output_depth_dim * Output_X_dim) +
          //   (x+1) * Output_depth_dim
          //   + (m+8);
          // unsigned long long int out_subscript7 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
          //   y * (Output_depth_dim * Output_X_dim) +
          //   (x+2) * Output_depth_dim
          //   + (m+8);
          // unsigned long long int out_subscript8 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
          //   y * (Output_depth_dim * Output_X_dim) +
          //   (x+3) * Output_depth_dim
          //   + (m+8);

          
          temp = _mm256_add_ps(temp, bias);
          temp = _mm256_max_ps(temp, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript], temp);

          temp2 = _mm256_add_ps(temp2, bias);
          temp2 = _mm256_max_ps(temp2, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript + Output_depth_dim], temp2); // x+1

          temp3 = _mm256_add_ps(temp3, bias);
          temp3 = _mm256_max_ps(temp3, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript + (Output_depth_dim * 2)], temp3); // x+2

          temp4 = _mm256_add_ps(temp4, bias);
          temp4 = _mm256_max_ps(temp4, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript + (Output_depth_dim * 3)], temp4); // x+3

          temp5 = _mm256_add_ps(temp5, bias2);
          temp5 = _mm256_max_ps(temp5, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript + 8], temp5); // m+8

          temp6 = _mm256_add_ps(temp6, bias2);
          temp6 = _mm256_max_ps(temp6, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript + Output_depth_dim + 8], temp6); // m+8, x+1

          temp7 = _mm256_add_ps(temp7, bias2);
          temp7 = _mm256_max_ps(temp7, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript + (Output_depth_dim * 2) + 8], temp7); // m+8, x+2

          temp8 = _mm256_add_ps(temp8, bias2);
          temp8 = _mm256_max_ps(temp8, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript + (Output_depth_dim * 3) + 8], temp8); // m+8, x+3
          // temp += bias;
          // out_to_compare_with_FP[out_subscript] = Relu_float(temp);

        }
      }
    }
  }
  printf("\n from optimised_layer_v6 %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  _mm_free(filter_FP_copy);
  return 0;
}


// replace mul with shift ops in d loop
// IN CURRENT FORM DOESN'T WORK WHEN OUT/INPUT DEPTHS ARE CHANGED
// 63 GFLOPS, 1.02x SLOWDOWN from v6, maybe revisit later
int optimised_layer_v7_AC_strength_reduction_d_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  #define output_depth_lshift 7   // x * output_depth_dim == x << 7
  #define input_depth_lshift 8    // x * input_depth_dim == x << 8
  #define in_out_depth_lshift 15  // x * input_depth_dim * output_depth_dim == x << 15

  __m256 bias, bias2, temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8, s, s2, s3, s4, w, w2;

  unsigned int filter_FP_length = Output_depth_dim * Mask_Y_dim * Mask_X_dim * Input_depth_dim;
  float* filter_FP_copy = (float*)_mm_malloc(filter_FP_length * sizeof(float), 64);
  if (filter_FP_copy == NULL) {
    printf("\nerror with malloc allocating filter array copy");
    exit(EXIT_FAILURE);
  }

  // array copying - filter_FP into form usable for vectorising m loop
  for (unsigned int y = 0; y < Mask_Y_dim; y++) {
    for (unsigned int x = 0; x < Mask_X_dim; x++) {
      for (unsigned int d = 0; d < Input_depth_dim; d++) {
        for (unsigned int m = 0; m < Output_depth_dim; m++) {
          unsigned long long int old_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
            + y * Mask_X_dim * Input_depth_dim
            + x * Input_depth_dim
            + d;
              
          unsigned long long int new_subscript = y * Mask_X_dim * Input_depth_dim * Output_depth_dim
            + x * Input_depth_dim * Output_depth_dim
            + d * Output_depth_dim
            + m;

          filter_FP_copy[new_subscript] = filter_FP[old_subscript];
        }
      }
    }
  }


  // main loop body
  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int m = 0; m < Output_depth_dim; m+=16) { //channels
      bias = _mm256_load_ps(&bias_array_FP[m]);
      bias2 = _mm256_load_ps( &bias_array_FP[m+8]);

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (unsigned int x = 0; x < Output_X_dim; x+=4) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          temp4 = _mm256_setzero_ps();
          temp5 = _mm256_setzero_ps();
          temp6 = _mm256_setzero_ps();
          temp7 = _mm256_setzero_ps();
          temp8 = _mm256_setzero_ps();
          // bias = bias_array_FP[m];
          // temp = 0.0f;
          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d++) {

                unsigned long long int in_subscript = b * ((Input_Y_dim * Input_X_dim << input_depth_lshift))
                  + ((y * Stride_Y_dim + off_y) * Input_X_dim << input_depth_lshift)
                  + ((x * Stride_X_dim + off_x) << input_depth_lshift)
                  + d;


                unsigned long long int filter_subscript = (off_y * Mask_X_dim << in_out_depth_lshift)
                  + (off_x << in_out_depth_lshift)
                  + (d << output_depth_lshift)
                  + m;

                

                s = _mm256_set1_ps(in_FP[in_subscript]); // in_FP[b][y+off_y][x+off_x][d]
                s2 = _mm256_set1_ps(in_FP[in_subscript + Input_depth_dim]); // x+1
                s3 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 2)]); // x+2 
                s4 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 3)]); // x+3

                w = _mm256_load_ps(&filter_FP_copy[filter_subscript]); // filter_FP[off_y][off_x][d][m]
                w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);

                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s3, w, temp3);
                temp4 = _mm256_fmadd_ps(s4, w, temp4);
                temp5 = _mm256_fmadd_ps(s, w2, temp5);
                temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                temp8 = _mm256_fmadd_ps(s4, w2, temp8);
                // float s = in_FP[in_subscript];
                // float w = filter_FP[filter_subscript];
                // temp = temp + s * w;
              }
            }
          }


          unsigned long long int out_subscript = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + m;

          
          temp = _mm256_add_ps(temp, bias);
          temp = _mm256_max_ps(temp, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript], temp);

          temp2 = _mm256_add_ps(temp2, bias);
          temp2 = _mm256_max_ps(temp2, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript + Output_depth_dim], temp2); // x+1

          temp3 = _mm256_add_ps(temp3, bias);
          temp3 = _mm256_max_ps(temp3, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript + (Output_depth_dim * 2)], temp3); // x+2

          temp4 = _mm256_add_ps(temp4, bias);
          temp4 = _mm256_max_ps(temp4, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript + (Output_depth_dim * 3)], temp4); // x+3

          temp5 = _mm256_add_ps(temp5, bias2);
          temp5 = _mm256_max_ps(temp5, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript + 8], temp5); // m+8

          temp6 = _mm256_add_ps(temp6, bias2);
          temp6 = _mm256_max_ps(temp6, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript + Output_depth_dim + 8], temp6); // m+8, x+1

          temp7 = _mm256_add_ps(temp7, bias2);
          temp7 = _mm256_max_ps(temp7, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript + (Output_depth_dim * 2) + 8], temp7); // m+8, x+2

          temp8 = _mm256_add_ps(temp8, bias2);
          temp8 = _mm256_max_ps(temp8, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript + (Output_depth_dim * 3) + 8], temp8); // m+8, x+3
          // temp += bias;
          // out_to_compare_with_FP[out_subscript] = Relu_float(temp);

        }
      }
    }
  }
  printf("\n from optimised_layer_v7 %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  _mm_free(filter_FP_copy);
  return 0;
}


// loop tiling on m loop
// 65 GFLOPS, 1.02x speedup from v6 when m = 128(low), results in less cache misses so perf scales with m
int optimised_layer_v8_AC_loop_tiling_m_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  #define m_tile 16

  __m256 bias, bias2, temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8, s, s2, s3, s4, w, w2;

  unsigned int filter_FP_length = Output_depth_dim * Mask_Y_dim * Mask_X_dim * Input_depth_dim;
  float* filter_FP_copy = (float*)_mm_malloc(filter_FP_length * sizeof(float), 64);
  if (filter_FP_copy == NULL) {
    printf("\nerror with malloc allocating filter array copy");
    exit(EXIT_FAILURE);
  }

  // array copying - filter_FP into form usable for vectorising m loop
  for (unsigned int y = 0; y < Mask_Y_dim; y++) {
    for (unsigned int x = 0; x < Mask_X_dim; x++) {
      for (unsigned int d = 0; d < Input_depth_dim; d++) {
        for (unsigned int m = 0; m < Output_depth_dim; m++) {
          unsigned long long int old_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
            + y * Mask_X_dim * Input_depth_dim
            + x * Input_depth_dim
            + d;
              
          unsigned long long int new_subscript = y * Mask_X_dim * Input_depth_dim * Output_depth_dim
            + x * Input_depth_dim * Output_depth_dim
            + d * Output_depth_dim
            + m;

          filter_FP_copy[new_subscript] = filter_FP[old_subscript];
        }
      }
    }
  }


  // main loop body
  for (unsigned int mm = 0; mm < Output_depth_dim; mm += m_tile) {
    for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
      for (unsigned int m = mm; m < mm + m_tile; m+=16) { //channels
        bias = _mm256_load_ps(&bias_array_FP[m]);
        bias2 = _mm256_load_ps( &bias_array_FP[m+8]);

        for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
          for (unsigned int x = 0; x < Output_X_dim; x+=4) {	//Output Width
            temp = _mm256_setzero_ps();
            temp2 = _mm256_setzero_ps();
            temp3 = _mm256_setzero_ps();
            temp4 = _mm256_setzero_ps();
            temp5 = _mm256_setzero_ps();
            temp6 = _mm256_setzero_ps();
            temp7 = _mm256_setzero_ps();
            temp8 = _mm256_setzero_ps();
            // bias = bias_array_FP[m];
            // temp = 0.0f;
            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                for (unsigned int d = 0; d < Input_depth_dim; d++) {

                  unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                    + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                    + (x * Stride_X_dim + off_x) * Input_depth_dim
                    + d;

                  unsigned long long int filter_subscript = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                    + off_x * Input_depth_dim * Output_depth_dim
                    + d * Output_depth_dim
                    + m;
                  

                  s = _mm256_set1_ps(in_FP[in_subscript]); // in_FP[b][y+off_y][x+off_x][d]
                  s2 = _mm256_set1_ps(in_FP[in_subscript + Input_depth_dim]); // x+1
                  s3 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 2)]); // x+2 
                  s4 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 3)]); // x+3

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript]); // filter_FP[off_y][off_x][d][m]
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);
                  // float s = in_FP[in_subscript];
                  // float w = filter_FP[filter_subscript];
                  // temp = temp + s * w;
                }
              }
            }


            unsigned long long int out_subscript = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
              y * (Output_depth_dim * Output_X_dim) +
              x * Output_depth_dim
              + m;

            
            temp = _mm256_add_ps(temp, bias);
            temp = _mm256_max_ps(temp, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript], temp);

            temp2 = _mm256_add_ps(temp2, bias);
            temp2 = _mm256_max_ps(temp2, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + Output_depth_dim], temp2); // x+1

            temp3 = _mm256_add_ps(temp3, bias);
            temp3 = _mm256_max_ps(temp3, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + (Output_depth_dim * 2)], temp3); // x+2

            temp4 = _mm256_add_ps(temp4, bias);
            temp4 = _mm256_max_ps(temp4, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + (Output_depth_dim * 3)], temp4); // x+3

            temp5 = _mm256_add_ps(temp5, bias2);
            temp5 = _mm256_max_ps(temp5, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + 8], temp5); // m+8

            temp6 = _mm256_add_ps(temp6, bias2);
            temp6 = _mm256_max_ps(temp6, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + Output_depth_dim + 8], temp6); // m+8, x+1

            temp7 = _mm256_add_ps(temp7, bias2);
            temp7 = _mm256_max_ps(temp7, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + (Output_depth_dim * 2) + 8], temp7); // m+8, x+2

            temp8 = _mm256_add_ps(temp8, bias2);
            temp8 = _mm256_max_ps(temp8, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + (Output_depth_dim * 3) + 8], temp8); // m+8, x+3
            // temp += bias;
            // out_to_compare_with_FP[out_subscript] = Relu_float(temp);

          }
        }
      }
    }
  }
  printf("\n from optimised_layer_v8 %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  _mm_free(filter_FP_copy);
  return 0;
}


// unroll d loop by 2
// 66 GFLOPS, 1.02x speedup from v8
int optimised_layer_v9_AC_unroll_d2_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  #define m_tile 16

  __m256 bias, bias2, temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8, s, s2, s3, s4, w, w2;

  unsigned int filter_FP_length = Output_depth_dim * Mask_Y_dim * Mask_X_dim * Input_depth_dim;
  float* filter_FP_copy = (float*)_mm_malloc(filter_FP_length * sizeof(float), 64);
  if (filter_FP_copy == NULL) {
    printf("\nerror with malloc allocating filter array copy");
    exit(EXIT_FAILURE);
  }

  // array copying - filter_FP into form usable for vectorising m loop
  for (unsigned int y = 0; y < Mask_Y_dim; y++) {
    for (unsigned int x = 0; x < Mask_X_dim; x++) {
      for (unsigned int d = 0; d < Input_depth_dim; d++) {
        for (unsigned int m = 0; m < Output_depth_dim; m++) {
          unsigned long long int old_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
            + y * Mask_X_dim * Input_depth_dim
            + x * Input_depth_dim
            + d;
              
          unsigned long long int new_subscript = y * Mask_X_dim * Input_depth_dim * Output_depth_dim
            + x * Input_depth_dim * Output_depth_dim
            + d * Output_depth_dim
            + m;

          filter_FP_copy[new_subscript] = filter_FP[old_subscript];
        }
      }
    }
  }


  // main loop body
  for (unsigned int mm = 0; mm < Output_depth_dim; mm += m_tile) {
    for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
      for (unsigned int m = mm; m < mm + m_tile; m+=16) { //channels
        bias = _mm256_load_ps(&bias_array_FP[m]);
        bias2 = _mm256_load_ps( &bias_array_FP[m+8]);

        for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
          for (unsigned int x = 0; x < Output_X_dim; x+=4) {	//Output Width
            temp = _mm256_setzero_ps();
            temp2 = _mm256_setzero_ps();
            temp3 = _mm256_setzero_ps();
            temp4 = _mm256_setzero_ps();
            temp5 = _mm256_setzero_ps();
            temp6 = _mm256_setzero_ps();
            temp7 = _mm256_setzero_ps();
            temp8 = _mm256_setzero_ps();
            // bias = bias_array_FP[m];
            // temp = 0.0f;
            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                for (unsigned int d = 0; d < Input_depth_dim; d+=2) {

                  unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                    + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                    + (x * Stride_X_dim + off_x) * Input_depth_dim
                    + d;

                  unsigned long long int filter_subscript = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                    + off_x * Input_depth_dim * Output_depth_dim
                    + d * Output_depth_dim
                    + m;


                  s = _mm256_set1_ps(in_FP[in_subscript]);        // d, x
                  s2 = _mm256_set1_ps(in_FP[in_subscript + Input_depth_dim]); // d, x+1
                  s3 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 2)]); // d, x+2
                  s4 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 3)]); // d, x+3

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);      // d, m
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]); // d, m+8

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                  // __________________________________________________________________________
                  // |
                  // |
                  // v d+1

                  s = _mm256_set1_ps(in_FP[in_subscript + 1]);        // d+1, x
                  s2 = _mm256_set1_ps(in_FP[in_subscript + Input_depth_dim + 1]); // d+1, x+1
                  s3 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 2) + 1]); // d+1, x+2
                  s4 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 3) + 1]); // d+1, x+3

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript + Output_depth_dim]);      // d+1, m
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + Output_depth_dim + 8]); // d+1, m+8

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);
                }
              }
            }


            unsigned long long int out_subscript = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
              y * (Output_depth_dim * Output_X_dim) +
              x * Output_depth_dim
              + m;

            
            temp = _mm256_add_ps(temp, bias);
            temp = _mm256_max_ps(temp, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript], temp);

            temp2 = _mm256_add_ps(temp2, bias);
            temp2 = _mm256_max_ps(temp2, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + Output_depth_dim], temp2); // x+1

            temp3 = _mm256_add_ps(temp3, bias);
            temp3 = _mm256_max_ps(temp3, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + (Output_depth_dim * 2)], temp3); // x+2

            temp4 = _mm256_add_ps(temp4, bias);
            temp4 = _mm256_max_ps(temp4, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + (Output_depth_dim * 3)], temp4); // x+3

            temp5 = _mm256_add_ps(temp5, bias2);
            temp5 = _mm256_max_ps(temp5, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + 8], temp5); // m+8

            temp6 = _mm256_add_ps(temp6, bias2);
            temp6 = _mm256_max_ps(temp6, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + Output_depth_dim + 8], temp6); // m+8, x+1

            temp7 = _mm256_add_ps(temp7, bias2);
            temp7 = _mm256_max_ps(temp7, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + (Output_depth_dim * 2) + 8], temp7); // m+8, x+2

            temp8 = _mm256_add_ps(temp8, bias2);
            temp8 = _mm256_max_ps(temp8, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + (Output_depth_dim * 3) + 8], temp8); // m+8, x+3
            // temp += bias;
            // out_to_compare_with_FP[out_subscript] = Relu_float(temp);

          }
        }
      }
    }
  }
  printf("\n from optimised_layer_v9 %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  _mm_free(filter_FP_copy);
  return 0;
}


// unroll d loop by 4
// 65 GFLOPS, 1.02x slowdown from v9?
int optimised_layer_v10_AC_unroll_d4_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  #define m_tile 16

  __m256 bias, bias2, temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8, s, s2, s3, s4, w, w2;

  unsigned int filter_FP_length = Output_depth_dim * Mask_Y_dim * Mask_X_dim * Input_depth_dim;
  float* filter_FP_copy = (float*)_mm_malloc(filter_FP_length * sizeof(float), 64);
  if (filter_FP_copy == NULL) {
    printf("\nerror with malloc allocating filter array copy");
    exit(EXIT_FAILURE);
  }

  // array copying - filter_FP into form usable for vectorising m loop
  for (unsigned int y = 0; y < Mask_Y_dim; y++) {
    for (unsigned int x = 0; x < Mask_X_dim; x++) {
      for (unsigned int d = 0; d < Input_depth_dim; d++) {
        for (unsigned int m = 0; m < Output_depth_dim; m++) {
          unsigned long long int old_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
            + y * Mask_X_dim * Input_depth_dim
            + x * Input_depth_dim
            + d;
              
          unsigned long long int new_subscript = y * Mask_X_dim * Input_depth_dim * Output_depth_dim
            + x * Input_depth_dim * Output_depth_dim
            + d * Output_depth_dim
            + m;

          filter_FP_copy[new_subscript] = filter_FP[old_subscript];
        }
      }
    }
  }


  // main loop body
  for (unsigned int mm = 0; mm < Output_depth_dim; mm += m_tile) {
    for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
      for (unsigned int m = mm; m < mm + m_tile; m+=16) { //channels
        bias = _mm256_load_ps(&bias_array_FP[m]);
        bias2 = _mm256_load_ps( &bias_array_FP[m+8]);

        for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
          for (unsigned int x = 0; x < Output_X_dim; x+=4) {	//Output Width
            temp = _mm256_setzero_ps();
            temp2 = _mm256_setzero_ps();
            temp3 = _mm256_setzero_ps();
            temp4 = _mm256_setzero_ps();
            temp5 = _mm256_setzero_ps();
            temp6 = _mm256_setzero_ps();
            temp7 = _mm256_setzero_ps();
            temp8 = _mm256_setzero_ps();
            // bias = bias_array_FP[m];
            // temp = 0.0f;
            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                for (unsigned int d = 0; d < Input_depth_dim; d+=4) {

                  unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                    + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                    + (x * Stride_X_dim + off_x) * Input_depth_dim
                    + d;

                  unsigned long long int filter_subscript = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                    + off_x * Input_depth_dim * Output_depth_dim
                    + d * Output_depth_dim
                    + m;


                  s = _mm256_set1_ps(in_FP[in_subscript]);        // d, x
                  s2 = _mm256_set1_ps(in_FP[in_subscript + Input_depth_dim]); // d, x+1
                  s3 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 2)]); // d, x+2
                  s4 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 3)]); // d, x+3

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);      // d, m
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]); // d, m+8

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);#


                  // __________________________________________________________________________
                  // |
                  // |
                  // v d+1

                  s = _mm256_set1_ps(in_FP[in_subscript + 1]);        // d+1, x
                  s2 = _mm256_set1_ps(in_FP[in_subscript + Input_depth_dim + 1]); // d+1, x+1
                  s3 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 2) + 1]); // d+1, x+2
                  s4 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 3) + 1]); // d+1, x+3

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript + Output_depth_dim]);      // d+1, m
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + Output_depth_dim + 8]); // d+1, m+8

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                  // __________________________________________________________________________
                  // |
                  // |
                  // v d+2

                  s = _mm256_set1_ps(in_FP[in_subscript + 2]);        // d+2, x
                  s2 = _mm256_set1_ps(in_FP[in_subscript + Input_depth_dim + 2]); // d+2, x+1
                  s3 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 2) + 2]); // d+2, x+2
                  s4 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 3) + 2]); // d+2, x+3

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 2)]);      // d+2, m
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 2) + 8]); // d+2, m+8

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                  // __________________________________________________________________________
                  // |
                  // |
                  // v d+3

                  s = _mm256_set1_ps(in_FP[in_subscript + 3]);        // d+3, x
                  s2 = _mm256_set1_ps(in_FP[in_subscript + Input_depth_dim + 3]); // d+3, x+1
                  s3 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 2) + 3]); // d+3, x+2
                  s4 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 3) + 3]); // d+3, x+3

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 3)]);      // d+3, m
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 3) + 8]); // d+3, m+8

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);
                }
              }
            }


            unsigned long long int out_subscript = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
              y * (Output_depth_dim * Output_X_dim) +
              x * Output_depth_dim
              + m;

            
            temp = _mm256_add_ps(temp, bias);
            temp = _mm256_max_ps(temp, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript], temp);

            temp2 = _mm256_add_ps(temp2, bias);
            temp2 = _mm256_max_ps(temp2, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + Output_depth_dim], temp2); // x+1

            temp3 = _mm256_add_ps(temp3, bias);
            temp3 = _mm256_max_ps(temp3, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + (Output_depth_dim * 2)], temp3); // x+2

            temp4 = _mm256_add_ps(temp4, bias);
            temp4 = _mm256_max_ps(temp4, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + (Output_depth_dim * 3)], temp4); // x+3

            temp5 = _mm256_add_ps(temp5, bias2);
            temp5 = _mm256_max_ps(temp5, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + 8], temp5); // m+8

            temp6 = _mm256_add_ps(temp6, bias2);
            temp6 = _mm256_max_ps(temp6, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + Output_depth_dim + 8], temp6); // m+8, x+1

            temp7 = _mm256_add_ps(temp7, bias2);
            temp7 = _mm256_max_ps(temp7, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + (Output_depth_dim * 2) + 8], temp7); // m+8, x+2

            temp8 = _mm256_add_ps(temp8, bias2);
            temp8 = _mm256_max_ps(temp8, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + (Output_depth_dim * 3) + 8], temp8); // m+8, x+3
            // temp += bias;
            // out_to_compare_with_FP[out_subscript] = Relu_float(temp);

          }
        }
      }
    }
  }
  printf("\n from optimised_layer_v10 %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  _mm_free(filter_FP_copy);
  return 0;
}


// unroll d loop by 8
// 76 GFLOPS, 1.15x speedup from v9
int optimised_layer_v11_AC_unroll_d8_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  #define m_tile 16

  __m256 bias, bias2, temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8, s, s2, s3, s4, w, w2;

  unsigned int filter_FP_length = Output_depth_dim * Mask_Y_dim * Mask_X_dim * Input_depth_dim;
  float* filter_FP_copy = (float*)_mm_malloc(filter_FP_length * sizeof(float), 64);
  if (filter_FP_copy == NULL) {
    printf("\nerror with malloc allocating filter array copy");
    exit(EXIT_FAILURE);
  }

  // array copying - filter_FP into form usable for vectorising m loop
  for (unsigned int y = 0; y < Mask_Y_dim; y++) {
    for (unsigned int x = 0; x < Mask_X_dim; x++) {
      for (unsigned int d = 0; d < Input_depth_dim; d++) {
        for (unsigned int m = 0; m < Output_depth_dim; m++) {
          unsigned long long int old_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
            + y * Mask_X_dim * Input_depth_dim
            + x * Input_depth_dim
            + d;
              
          unsigned long long int new_subscript = y * Mask_X_dim * Input_depth_dim * Output_depth_dim
            + x * Input_depth_dim * Output_depth_dim
            + d * Output_depth_dim
            + m;

          filter_FP_copy[new_subscript] = filter_FP[old_subscript];
        }
      }
    }
  }


  // main loop body
  for (unsigned int mm = 0; mm < Output_depth_dim; mm += m_tile) {
    for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
      for (unsigned int m = mm; m < mm + m_tile; m+=16) { //channels
        bias = _mm256_load_ps(&bias_array_FP[m]);
        bias2 = _mm256_load_ps( &bias_array_FP[m+8]);

        for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
          for (unsigned int x = 0; x < Output_X_dim; x+=4) {	//Output Width
            temp = _mm256_setzero_ps();
            temp2 = _mm256_setzero_ps();
            temp3 = _mm256_setzero_ps();
            temp4 = _mm256_setzero_ps();
            temp5 = _mm256_setzero_ps();
            temp6 = _mm256_setzero_ps();
            temp7 = _mm256_setzero_ps();
            temp8 = _mm256_setzero_ps();
            // bias = bias_array_FP[m];
            // temp = 0.0f;
            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                  unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                    + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                    + (x * Stride_X_dim + off_x) * Input_depth_dim
                    + d;

                  unsigned long long int filter_subscript = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                    + off_x * Input_depth_dim * Output_depth_dim
                    + d * Output_depth_dim
                    + m;


                  s = _mm256_set1_ps(in_FP[in_subscript]);        // d, x
                  s2 = _mm256_set1_ps(in_FP[in_subscript + Input_depth_dim]); // d, x+1
                  s3 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 2)]); // d, x+2
                  s4 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 3)]); // d, x+3

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);      // d, m
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]); // d, m+8

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                  // __________________________________________________________________________
                  // |
                  // |
                  // v d+1

                  s = _mm256_set1_ps(in_FP[in_subscript + 1]);        // d+1, x
                  s2 = _mm256_set1_ps(in_FP[in_subscript + Input_depth_dim + 1]); // d+1, x+1
                  s3 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 2) + 1]); // d+1, x+2
                  s4 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 3) + 1]); // d+1, x+3

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript + Output_depth_dim]);      // d+1, m
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + Output_depth_dim + 8]); // d+1, m+8

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                  // __________________________________________________________________________
                  // |
                  // |
                  // v d+2

                  s = _mm256_set1_ps(in_FP[in_subscript + 2]);        // d+2, x
                  s2 = _mm256_set1_ps(in_FP[in_subscript + Input_depth_dim + 2]); // d+2, x+1
                  s3 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 2) + 2]); // d+2, x+2
                  s4 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 3) + 2]); // d+2, x+3

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 2)]);      // d+2, m
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 2) + 8]); // d+2, m+8

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                  // __________________________________________________________________________
                  // |
                  // |
                  // v d+3

                  s = _mm256_set1_ps(in_FP[in_subscript + 3]);        // d+3, x
                  s2 = _mm256_set1_ps(in_FP[in_subscript + Input_depth_dim + 3]); // d+3, x+1
                  s3 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 2) + 3]); // d+3, x+2
                  s4 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 3) + 3]); // d+3, x+3

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 3)]);      // d+3, m
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 3) + 8]); // d+3, m+8

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                  // __________________________________________________________________________
                  // |
                  // |
                  // v d+4

                  s = _mm256_set1_ps(in_FP[in_subscript + 4]);        // d+4, x
                  s2 = _mm256_set1_ps(in_FP[in_subscript + Input_depth_dim + 4]); // d+4, x+1
                  s3 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 2) + 4]); // d+4, x+2
                  s4 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 3) + 4]); // d+4, x+3

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 4)]);      // d+4, m
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 4) + 8]); // d+4, m+8

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                  // __________________________________________________________________________
                  // |
                  // |
                  // v d+5

                  s = _mm256_set1_ps(in_FP[in_subscript + 5]);        // d+5, x
                  s2 = _mm256_set1_ps(in_FP[in_subscript + Input_depth_dim + 5]); // d+5, x+1
                  s3 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 2) + 5]); // d+5, x+2
                  s4 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 3) + 5]); // d+5, x+3

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 5)]);      // d+5, m
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 5) + 8]); // d+5, m+8

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                  // __________________________________________________________________________
                  // |
                  // |
                  // v d+6

                  s = _mm256_set1_ps(in_FP[in_subscript + 6]);        // d+6, x
                  s2 = _mm256_set1_ps(in_FP[in_subscript + Input_depth_dim + 6]); // d+6, x+1
                  s3 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 2) + 6]); // d+6, x+2
                  s4 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 3) + 6]); // d+6, x+3

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 6)]);      // d+6, m
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 6) + 8]); // d+6, m+8

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                  // __________________________________________________________________________
                  // |
                  // |
                  // v d+7

                  s = _mm256_set1_ps(in_FP[in_subscript + 7]);        // d+7, x
                  s2 = _mm256_set1_ps(in_FP[in_subscript + Input_depth_dim + 7]); // d+7, x+1
                  s3 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 2) + 7]); // d+7, x+2
                  s4 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 3) + 7]); // d+7, x+3

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 7)]);      // d+7, m
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 7) + 8]); // d+7, m+8

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);

                  // float s = in_FP[in_subscript];
                  // float w = filter_FP[filter_subscript];
                  // temp = temp + s * w;
                }
              }
            }


            unsigned long long int out_subscript = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
              y * (Output_depth_dim * Output_X_dim) +
              x * Output_depth_dim
              + m;

            
            temp = _mm256_add_ps(temp, bias);
            temp = _mm256_max_ps(temp, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript], temp);

            temp2 = _mm256_add_ps(temp2, bias);
            temp2 = _mm256_max_ps(temp2, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + Output_depth_dim], temp2); // x+1

            temp3 = _mm256_add_ps(temp3, bias);
            temp3 = _mm256_max_ps(temp3, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + (Output_depth_dim * 2)], temp3); // x+2

            temp4 = _mm256_add_ps(temp4, bias);
            temp4 = _mm256_max_ps(temp4, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + (Output_depth_dim * 3)], temp4); // x+3

            temp5 = _mm256_add_ps(temp5, bias2);
            temp5 = _mm256_max_ps(temp5, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + 8], temp5); // m+8

            temp6 = _mm256_add_ps(temp6, bias2);
            temp6 = _mm256_max_ps(temp6, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + Output_depth_dim + 8], temp6); // m+8, x+1

            temp7 = _mm256_add_ps(temp7, bias2);
            temp7 = _mm256_max_ps(temp7, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + (Output_depth_dim * 2) + 8], temp7); // m+8, x+2

            temp8 = _mm256_add_ps(temp8, bias2);
            temp8 = _mm256_max_ps(temp8, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + (Output_depth_dim * 3) + 8], temp8); // m+8, x+3
            // temp += bias;
            // out_to_compare_with_FP[out_subscript] = Relu_float(temp);

          }
        }
      }
    }
  }
  printf("\n from optimised_layer_v11 %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  _mm_free(filter_FP_copy);
  return 0;
}


// move operations outside of d loop, general registers used in d loop = 16
// 81 GFLOPS, 1.07x speedup from v11
int optimised_layer_v12_AC_ops_outside_loop_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  #define m_tile 16

  __m256 bias, bias2, temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8, s, s2, s3, s4, w, w2;

  unsigned int yx_in_depth = Input_Y_dim * Input_X_dim * Input_depth_dim;
  unsigned int x_in_depth = Input_X_dim * Input_depth_dim;

  unsigned int mask_in_out_depth = Mask_X_dim * Input_depth_dim * Output_depth_dim;
  // unsigned int in_out_depth = Input_depth_dim * Output_depth_dim;

  unsigned int filter_FP_length = Output_depth_dim * Mask_Y_dim * Mask_X_dim * Input_depth_dim;
  float* filter_FP_copy = (float*)_mm_malloc(filter_FP_length * sizeof(float), 64);
  if (filter_FP_copy == NULL) {
    printf("\nerror with malloc allocating filter array copy");
    exit(EXIT_FAILURE);
  }

  // array copying - filter_FP into form usable for vectorising m loop
  for (unsigned int y = 0; y < Mask_Y_dim; y++) {
    for (unsigned int x = 0; x < Mask_X_dim; x++) {
      for (unsigned int d = 0; d < Input_depth_dim; d++) {
        for (unsigned int m = 0; m < Output_depth_dim; m++) {
          unsigned long long int old_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
            + y * Mask_X_dim * Input_depth_dim
            + x * Input_depth_dim
            + d;
              
          unsigned long long int new_subscript = y * Mask_X_dim * Input_depth_dim * Output_depth_dim
            + x * Input_depth_dim * Output_depth_dim
            + d * Output_depth_dim
            + m;

          filter_FP_copy[new_subscript] = filter_FP[old_subscript];
        }
      }
    }
  }


  // main loop body
  for (unsigned int mm = 0; mm < Output_depth_dim; mm += m_tile) {
    for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
      for (unsigned int m = mm; m < mm + m_tile; m+=16) { //channels
        bias = _mm256_load_ps(&bias_array_FP[m]);
        bias2 = _mm256_load_ps( &bias_array_FP[m+8]);

        for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
          for (unsigned int x = 0; x < Output_X_dim; x+=4) {	//Output Width
            temp = _mm256_setzero_ps();
            temp2 = _mm256_setzero_ps();
            temp3 = _mm256_setzero_ps();
            temp4 = _mm256_setzero_ps();
            temp5 = _mm256_setzero_ps();
            temp6 = _mm256_setzero_ps();
            temp7 = _mm256_setzero_ps();
            temp8 = _mm256_setzero_ps();
            // bias = bias_array_FP[m];
            // temp = 0.0f;
            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                  unsigned long long int in_subscript = b * yx_in_depth
                    + (y * Stride_Y_dim + off_y) * x_in_depth
                    + (x * Stride_X_dim + off_x) * Input_depth_dim
                    + d;

                  unsigned long long int filter_subscript = off_y * mask_in_out_depth
                    + off_x * Input_depth_dim * Output_depth_dim
                    + d * Output_depth_dim
                    + m;


                  s = _mm256_set1_ps(in_FP[in_subscript]);        // d, x
                  s2 = _mm256_set1_ps(in_FP[in_subscript + Input_depth_dim]); // d, x+1
                  s3 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 2)]); // d, x+2
                  s4 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 3)]); // d, x+3

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);      // d, m
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]); // d, m+8

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                  // __________________________________________________________________________
                  // |
                  // |
                  // v d+1

                  s = _mm256_set1_ps(in_FP[in_subscript + 1]);        // d+1, x
                  s2 = _mm256_set1_ps(in_FP[in_subscript + Input_depth_dim + 1]); // d+1, x+1
                  s3 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 2) + 1]); // d+1, x+2
                  s4 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 3) + 1]); // d+1, x+3

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript + Output_depth_dim]);      // d+1, m
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + Output_depth_dim + 8]); // d+1, m+8

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                  // __________________________________________________________________________
                  // |
                  // |
                  // v d+2

                  s = _mm256_set1_ps(in_FP[in_subscript + 2]);        // d+2, x
                  s2 = _mm256_set1_ps(in_FP[in_subscript + Input_depth_dim + 2]); // d+2, x+1
                  s3 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 2) + 2]); // d+2, x+2
                  s4 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 3) + 2]); // d+2, x+3

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 2)]);      // d+2, m
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 2) + 8]); // d+2, m+8

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                  // __________________________________________________________________________
                  // |
                  // |
                  // v d+3

                  s = _mm256_set1_ps(in_FP[in_subscript + 3]);        // d+3, x
                  s2 = _mm256_set1_ps(in_FP[in_subscript + Input_depth_dim + 3]); // d+3, x+1
                  s3 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 2) + 3]); // d+3, x+2
                  s4 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 3) + 3]); // d+3, x+3

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 3)]);      // d+3, m
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 3) + 8]); // d+3, m+8

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                  // __________________________________________________________________________
                  // |
                  // |
                  // v d+4

                  s = _mm256_set1_ps(in_FP[in_subscript + 4]);        // d+4, x
                  s2 = _mm256_set1_ps(in_FP[in_subscript + Input_depth_dim + 4]); // d+4, x+1
                  s3 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 2) + 4]); // d+4, x+2
                  s4 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 3) + 4]); // d+4, x+3

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 4)]);      // d+4, m
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 4) + 8]); // d+4, m+8

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                  // __________________________________________________________________________
                  // |
                  // |
                  // v d+5

                  s = _mm256_set1_ps(in_FP[in_subscript + 5]);        // d+5, x
                  s2 = _mm256_set1_ps(in_FP[in_subscript + Input_depth_dim + 5]); // d+5, x+1
                  s3 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 2) + 5]); // d+5, x+2
                  s4 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 3) + 5]); // d+5, x+3

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 5)]);      // d+5, m
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 5) + 8]); // d+5, m+8

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                  // __________________________________________________________________________
                  // |
                  // |
                  // v d+6

                  s = _mm256_set1_ps(in_FP[in_subscript + 6]);        // d+6, x
                  s2 = _mm256_set1_ps(in_FP[in_subscript + Input_depth_dim + 6]); // d+6, x+1
                  s3 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 2) + 6]); // d+6, x+2
                  s4 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 3) + 6]); // d+6, x+3

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 6)]);      // d+6, m
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 6) + 8]); // d+6, m+8

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                  // __________________________________________________________________________
                  // |
                  // |
                  // v d+7

                  s = _mm256_set1_ps(in_FP[in_subscript + 7]);        // d+7, x
                  s2 = _mm256_set1_ps(in_FP[in_subscript + Input_depth_dim + 7]); // d+7, x+1
                  s3 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 2) + 7]); // d+7, x+2
                  s4 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 3) + 7]); // d+7, x+3

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 7)]);      // d+7, m
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 7) + 8]); // d+7, m+8

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);

                  // float s = in_FP[in_subscript];
                  // float w = filter_FP[filter_subscript];
                  // temp = temp + s * w;
                }
              }
            }


            unsigned long long int out_subscript = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
              y * (Output_depth_dim * Output_X_dim) +
              x * Output_depth_dim
              + m;

            
            temp = _mm256_add_ps(temp, bias);
            temp = _mm256_max_ps(temp, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript], temp);

            temp2 = _mm256_add_ps(temp2, bias);
            temp2 = _mm256_max_ps(temp2, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + Output_depth_dim], temp2); // x+1

            temp3 = _mm256_add_ps(temp3, bias);
            temp3 = _mm256_max_ps(temp3, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + (Output_depth_dim * 2)], temp3); // x+2

            temp4 = _mm256_add_ps(temp4, bias);
            temp4 = _mm256_max_ps(temp4, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + (Output_depth_dim * 3)], temp4); // x+3

            temp5 = _mm256_add_ps(temp5, bias2);
            temp5 = _mm256_max_ps(temp5, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + 8], temp5); // m+8

            temp6 = _mm256_add_ps(temp6, bias2);
            temp6 = _mm256_max_ps(temp6, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + Output_depth_dim + 8], temp6); // m+8, x+1

            temp7 = _mm256_add_ps(temp7, bias2);
            temp7 = _mm256_max_ps(temp7, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + (Output_depth_dim * 2) + 8], temp7); // m+8, x+2

            temp8 = _mm256_add_ps(temp8, bias2);
            temp8 = _mm256_max_ps(temp8, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + (Output_depth_dim * 3) + 8], temp8); // m+8, x+3
            // temp += bias;
            // out_to_compare_with_FP[out_subscript] = Relu_float(temp);

          }
        }
      }
    }
  }
  printf("\n from optimised_layer_v12 %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  _mm_free(filter_FP_copy);
  return 0;
}


// change ints to unsigned ints in array copying loop to prevent expensive conversion instructions
// 82 GFLOPS, 1.01x speedup from v12, likely within margin of error due to small size of array copying loop
int optimised_layer_v13_AC_sign_unsigned_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  #define m_tile 16

  __m256 bias, bias2, temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8, s, s2, s3, s4, w, w2;

  unsigned int yx_in_depth = Input_Y_dim * Input_X_dim * Input_depth_dim;
  unsigned int x_in_depth = Input_X_dim * Input_depth_dim;

  unsigned int mask_in_out_depth = Mask_X_dim * Input_depth_dim * Output_depth_dim;
  // unsigned int in_out_depth = Input_depth_dim * Output_depth_dim;

  unsigned int filter_FP_length = Output_depth_dim * Mask_Y_dim * Mask_X_dim * Input_depth_dim;
  float* filter_FP_copy = (float*)_mm_malloc(filter_FP_length * sizeof(float), 64);
  if (filter_FP_copy == NULL) {
    printf("\nerror with malloc allocating filter array copy");
    exit(EXIT_FAILURE);
  }

  // array copying - filter_FP into form usable for vectorising m loop
  for (unsigned int y = 0; y < Mask_Y_dim; y++) {
    for (unsigned int x = 0; x < Mask_X_dim; x++) {
      for (unsigned int d = 0; d < Input_depth_dim; d++) {
        for (unsigned int m = 0; m < Output_depth_dim; m++) {
          unsigned long long int old_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
            + y * Mask_X_dim * Input_depth_dim
            + x * Input_depth_dim
            + d;
              
          unsigned long long int new_subscript = y * Mask_X_dim * Input_depth_dim * Output_depth_dim
            + x * Input_depth_dim * Output_depth_dim
            + d * Output_depth_dim
            + m;

          filter_FP_copy[new_subscript] = filter_FP[old_subscript];
        }
      }
    }
  }


  // main loop body
  for (unsigned int mm = 0; mm < Output_depth_dim; mm += m_tile) {
    for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
      for (unsigned int m = mm; m < mm + m_tile; m+=16) { //channels
        bias = _mm256_load_ps(&bias_array_FP[m]);
        bias2 = _mm256_load_ps( &bias_array_FP[m+8]);

        for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
          for (unsigned int x = 0; x < Output_X_dim; x+=4) {	//Output Width
            temp = _mm256_setzero_ps();
            temp2 = _mm256_setzero_ps();
            temp3 = _mm256_setzero_ps();
            temp4 = _mm256_setzero_ps();
            temp5 = _mm256_setzero_ps();
            temp6 = _mm256_setzero_ps();
            temp7 = _mm256_setzero_ps();
            temp8 = _mm256_setzero_ps();
            // bias = bias_array_FP[m];
            // temp = 0.0f;
            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                  unsigned long long int in_subscript = b * yx_in_depth
                    + (y * Stride_Y_dim + off_y) * x_in_depth
                    + (x * Stride_X_dim + off_x) * Input_depth_dim
                    + d;

                  unsigned long long int filter_subscript = off_y * mask_in_out_depth
                    + off_x * Input_depth_dim * Output_depth_dim
                    + d * Output_depth_dim
                    + m;


                  s = _mm256_set1_ps(in_FP[in_subscript]);        // d, x
                  s2 = _mm256_set1_ps(in_FP[in_subscript + Input_depth_dim]); // d, x+1
                  s3 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 2)]); // d, x+2
                  s4 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 3)]); // d, x+3

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);      // d, m
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]); // d, m+8

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                  // __________________________________________________________________________
                  // |
                  // |
                  // v d+1

                  s = _mm256_set1_ps(in_FP[in_subscript + 1]);        // d+1, x
                  s2 = _mm256_set1_ps(in_FP[in_subscript + Input_depth_dim + 1]); // d+1, x+1
                  s3 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 2) + 1]); // d+1, x+2
                  s4 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 3) + 1]); // d+1, x+3

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript + Output_depth_dim]);      // d+1, m
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + Output_depth_dim + 8]); // d+1, m+8

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                  // __________________________________________________________________________
                  // |
                  // |
                  // v d+2

                  s = _mm256_set1_ps(in_FP[in_subscript + 2]);        // d+2, x
                  s2 = _mm256_set1_ps(in_FP[in_subscript + Input_depth_dim + 2]); // d+2, x+1
                  s3 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 2) + 2]); // d+2, x+2
                  s4 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 3) + 2]); // d+2, x+3

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 2)]);      // d+2, m
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 2) + 8]); // d+2, m+8

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                  // __________________________________________________________________________
                  // |
                  // |
                  // v d+3

                  s = _mm256_set1_ps(in_FP[in_subscript + 3]);        // d+3, x
                  s2 = _mm256_set1_ps(in_FP[in_subscript + Input_depth_dim + 3]); // d+3, x+1
                  s3 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 2) + 3]); // d+3, x+2
                  s4 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 3) + 3]); // d+3, x+3

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 3)]);      // d+3, m
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 3) + 8]); // d+3, m+8

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                  // __________________________________________________________________________
                  // |
                  // |
                  // v d+4

                  s = _mm256_set1_ps(in_FP[in_subscript + 4]);        // d+4, x
                  s2 = _mm256_set1_ps(in_FP[in_subscript + Input_depth_dim + 4]); // d+4, x+1
                  s3 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 2) + 4]); // d+4, x+2
                  s4 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 3) + 4]); // d+4, x+3

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 4)]);      // d+4, m
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 4) + 8]); // d+4, m+8

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                  // __________________________________________________________________________
                  // |
                  // |
                  // v d+5

                  s = _mm256_set1_ps(in_FP[in_subscript + 5]);        // d+5, x
                  s2 = _mm256_set1_ps(in_FP[in_subscript + Input_depth_dim + 5]); // d+5, x+1
                  s3 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 2) + 5]); // d+5, x+2
                  s4 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 3) + 5]); // d+5, x+3

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 5)]);      // d+5, m
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 5) + 8]); // d+5, m+8

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                  // __________________________________________________________________________
                  // |
                  // |
                  // v d+6

                  s = _mm256_set1_ps(in_FP[in_subscript + 6]);        // d+6, x
                  s2 = _mm256_set1_ps(in_FP[in_subscript + Input_depth_dim + 6]); // d+6, x+1
                  s3 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 2) + 6]); // d+6, x+2
                  s4 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 3) + 6]); // d+6, x+3

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 6)]);      // d+6, m
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 6) + 8]); // d+6, m+8

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                  // __________________________________________________________________________
                  // |
                  // |
                  // v d+7

                  s = _mm256_set1_ps(in_FP[in_subscript + 7]);        // d+7, x
                  s2 = _mm256_set1_ps(in_FP[in_subscript + Input_depth_dim + 7]); // d+7, x+1
                  s3 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 2) + 7]); // d+7, x+2
                  s4 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 3) + 7]); // d+7, x+3

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 7)]);      // d+7, m
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 7) + 8]); // d+7, m+8

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);

                  // float s = in_FP[in_subscript];
                  // float w = filter_FP[filter_subscript];
                  // temp = temp + s * w;
                }
              }
            }


            unsigned long long int out_subscript = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
              y * (Output_depth_dim * Output_X_dim) +
              x * Output_depth_dim
              + m;

            
            temp = _mm256_add_ps(temp, bias);
            temp = _mm256_max_ps(temp, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript], temp);

            temp2 = _mm256_add_ps(temp2, bias);
            temp2 = _mm256_max_ps(temp2, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + Output_depth_dim], temp2); // x+1

            temp3 = _mm256_add_ps(temp3, bias);
            temp3 = _mm256_max_ps(temp3, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + (Output_depth_dim * 2)], temp3); // x+2

            temp4 = _mm256_add_ps(temp4, bias);
            temp4 = _mm256_max_ps(temp4, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + (Output_depth_dim * 3)], temp4); // x+3

            temp5 = _mm256_add_ps(temp5, bias2);
            temp5 = _mm256_max_ps(temp5, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + 8], temp5); // m+8

            temp6 = _mm256_add_ps(temp6, bias2);
            temp6 = _mm256_max_ps(temp6, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + Output_depth_dim + 8], temp6); // m+8, x+1

            temp7 = _mm256_add_ps(temp7, bias2);
            temp7 = _mm256_max_ps(temp7, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + (Output_depth_dim * 2) + 8], temp7); // m+8, x+2

            temp8 = _mm256_add_ps(temp8, bias2);
            temp8 = _mm256_max_ps(temp8, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + (Output_depth_dim * 3) + 8], temp8); // m+8, x+3
            // temp += bias;
            // out_to_compare_with_FP[out_subscript] = Relu_float(temp);

          }
        }
      }
    }
  }
  printf("\n from optimised_layer_v13 %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  _mm_free(filter_FP_copy);
  return 0;
}


// make use of OpenMP API, 2 parallel calls
// 483 GFLOPS, 5.85x speedup from v13
int optimised_layer_v14_AC_omp_2blocks_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  #define m_tile 16

  __m256 bias, bias2, temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8, s, s2, s3, s4, w, w2;

  unsigned int yx_in_depth = Input_Y_dim * Input_X_dim * Input_depth_dim;
  unsigned int x_in_depth = Input_X_dim * Input_depth_dim;

  unsigned int mask_in_out_depth = Mask_X_dim * Input_depth_dim * Output_depth_dim;
  // unsigned int in_out_depth = Input_depth_dim * Output_depth_dim;

  unsigned int filter_FP_length = Output_depth_dim * Mask_Y_dim * Mask_X_dim * Input_depth_dim;
  float* filter_FP_copy = (float*)_mm_malloc(filter_FP_length * sizeof(float), 64);
  if (filter_FP_copy == NULL) {
    printf("\nerror with malloc allocating filter array copy");
    exit(EXIT_FAILURE);
  }

  // array copying - filter_FP into form usable for vectorising m loop
  #pragma omp parallel for collapse(3) schedule(static)
    for (unsigned int y = 0; y < Mask_Y_dim; y++) {
      for (unsigned int x = 0; x < Mask_X_dim; x++) {
        for (unsigned int d = 0; d < Input_depth_dim; d++) {
          for (unsigned int m = 0; m < Output_depth_dim; m++) {
            unsigned long long int old_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
              + y * Mask_X_dim * Input_depth_dim
              + x * Input_depth_dim
              + d;
                
            unsigned long long int new_subscript = y * Mask_X_dim * Input_depth_dim * Output_depth_dim
              + x * Input_depth_dim * Output_depth_dim
              + d * Output_depth_dim
              + m;

            filter_FP_copy[new_subscript] = filter_FP[old_subscript];
          }
        }
      }
    }


  // main loop body
  #pragma omp parallel for private(bias, bias2, temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8, s, s2, s3, s4, w, w2)\
    default(shared) collapse(5) schedule(static) 
  for (unsigned int mm = 0; mm < Output_depth_dim; mm += m_tile) {
    for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
      for (unsigned int m = mm; m < mm + m_tile; m+=16) { //channels
        bias = _mm256_load_ps(&bias_array_FP[m]);
        bias2 = _mm256_load_ps( &bias_array_FP[m+8]);

        for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
          for (unsigned int x = 0; x < Output_X_dim; x+=4) {	//Output Width
            temp = _mm256_setzero_ps();
            temp2 = _mm256_setzero_ps();
            temp3 = _mm256_setzero_ps();
            temp4 = _mm256_setzero_ps();
            temp5 = _mm256_setzero_ps();
            temp6 = _mm256_setzero_ps();
            temp7 = _mm256_setzero_ps();
            temp8 = _mm256_setzero_ps();
            // bias = bias_array_FP[m];
            // temp = 0.0f;
            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                  unsigned long long int in_subscript = b * yx_in_depth
                    + (y * Stride_Y_dim + off_y) * x_in_depth
                    + (x * Stride_X_dim + off_x) * Input_depth_dim
                    + d;

                  unsigned long long int filter_subscript = off_y * mask_in_out_depth
                    + off_x * Input_depth_dim * Output_depth_dim
                    + d * Output_depth_dim
                    + m;


                  s = _mm256_set1_ps(in_FP[in_subscript]);        // d, x
                  s2 = _mm256_set1_ps(in_FP[in_subscript + Input_depth_dim]); // d, x+1
                  s3 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 2)]); // d, x+2
                  s4 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 3)]); // d, x+3

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);      // d, m
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]); // d, m+8

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                  // __________________________________________________________________________
                  // |
                  // |
                  // v d+1

                  s = _mm256_set1_ps(in_FP[in_subscript + 1]);        // d+1, x
                  s2 = _mm256_set1_ps(in_FP[in_subscript + Input_depth_dim + 1]); // d+1, x+1
                  s3 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 2) + 1]); // d+1, x+2
                  s4 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 3) + 1]); // d+1, x+3

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript + Output_depth_dim]);      // d+1, m
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + Output_depth_dim + 8]); // d+1, m+8

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                  // __________________________________________________________________________
                  // |
                  // |
                  // v d+2

                  s = _mm256_set1_ps(in_FP[in_subscript + 2]);        // d+2, x
                  s2 = _mm256_set1_ps(in_FP[in_subscript + Input_depth_dim + 2]); // d+2, x+1
                  s3 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 2) + 2]); // d+2, x+2
                  s4 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 3) + 2]); // d+2, x+3

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 2)]);      // d+2, m
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 2) + 8]); // d+2, m+8

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                  // __________________________________________________________________________
                  // |
                  // |
                  // v d+3

                  s = _mm256_set1_ps(in_FP[in_subscript + 3]);        // d+3, x
                  s2 = _mm256_set1_ps(in_FP[in_subscript + Input_depth_dim + 3]); // d+3, x+1
                  s3 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 2) + 3]); // d+3, x+2
                  s4 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 3) + 3]); // d+3, x+3

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 3)]);      // d+3, m
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 3) + 8]); // d+3, m+8

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                  // __________________________________________________________________________
                  // |
                  // |
                  // v d+4

                  s = _mm256_set1_ps(in_FP[in_subscript + 4]);        // d+4, x
                  s2 = _mm256_set1_ps(in_FP[in_subscript + Input_depth_dim + 4]); // d+4, x+1
                  s3 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 2) + 4]); // d+4, x+2
                  s4 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 3) + 4]); // d+4, x+3

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 4)]);      // d+4, m
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 4) + 8]); // d+4, m+8

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                  // __________________________________________________________________________
                  // |
                  // |
                  // v d+5

                  s = _mm256_set1_ps(in_FP[in_subscript + 5]);        // d+5, x
                  s2 = _mm256_set1_ps(in_FP[in_subscript + Input_depth_dim + 5]); // d+5, x+1
                  s3 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 2) + 5]); // d+5, x+2
                  s4 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 3) + 5]); // d+5, x+3

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 5)]);      // d+5, m
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 5) + 8]); // d+5, m+8

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                  // __________________________________________________________________________
                  // |
                  // |
                  // v d+6

                  s = _mm256_set1_ps(in_FP[in_subscript + 6]);        // d+6, x
                  s2 = _mm256_set1_ps(in_FP[in_subscript + Input_depth_dim + 6]); // d+6, x+1
                  s3 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 2) + 6]); // d+6, x+2
                  s4 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 3) + 6]); // d+6, x+3

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 6)]);      // d+6, m
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 6) + 8]); // d+6, m+8

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                  // __________________________________________________________________________
                  // |
                  // |
                  // v d+7

                  s = _mm256_set1_ps(in_FP[in_subscript + 7]);        // d+7, x
                  s2 = _mm256_set1_ps(in_FP[in_subscript + Input_depth_dim + 7]); // d+7, x+1
                  s3 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 2) + 7]); // d+7, x+2
                  s4 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 3) + 7]); // d+7, x+3

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 7)]);      // d+7, m
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 7) + 8]); // d+7, m+8

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);

                  // float s = in_FP[in_subscript];
                  // float w = filter_FP[filter_subscript];
                  // temp = temp + s * w;
                }
              }
            }


            unsigned long long int out_subscript = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
              y * (Output_depth_dim * Output_X_dim) +
              x * Output_depth_dim
              + m;

            
            temp = _mm256_add_ps(temp, bias);
            temp = _mm256_max_ps(temp, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript], temp);

            temp2 = _mm256_add_ps(temp2, bias);
            temp2 = _mm256_max_ps(temp2, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + Output_depth_dim], temp2); // x+1

            temp3 = _mm256_add_ps(temp3, bias);
            temp3 = _mm256_max_ps(temp3, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + (Output_depth_dim * 2)], temp3); // x+2

            temp4 = _mm256_add_ps(temp4, bias);
            temp4 = _mm256_max_ps(temp4, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + (Output_depth_dim * 3)], temp4); // x+3

            temp5 = _mm256_add_ps(temp5, bias2);
            temp5 = _mm256_max_ps(temp5, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + 8], temp5); // m+8

            temp6 = _mm256_add_ps(temp6, bias2);
            temp6 = _mm256_max_ps(temp6, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + Output_depth_dim + 8], temp6); // m+8, x+1

            temp7 = _mm256_add_ps(temp7, bias2);
            temp7 = _mm256_max_ps(temp7, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + (Output_depth_dim * 2) + 8], temp7); // m+8, x+2

            temp8 = _mm256_add_ps(temp8, bias2);
            temp8 = _mm256_max_ps(temp8, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + (Output_depth_dim * 3) + 8], temp8); // m+8, x+3
            // temp += bias;
            // out_to_compare_with_FP[out_subscript] = Relu_float(temp);

          }
        }
      }
    }
  }
  printf("\n from optimised_layer_v14 %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  _mm_free(filter_FP_copy);
  return 0;
}


// make use of OpenMP API, 1 parallel call
// 456 GFLOPS, 1.06x slowdown from v14, not expected
int optimised_layer_v15_AC_omp_1block_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  #define m_tile 16

  __m256 bias, bias2, temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8, s, s2, s3, s4, w, w2;

  unsigned int yx_in_depth = Input_Y_dim * Input_X_dim * Input_depth_dim;
  unsigned int x_in_depth = Input_X_dim * Input_depth_dim;

  unsigned int mask_in_out_depth = Mask_X_dim * Input_depth_dim * Output_depth_dim;
  // unsigned int in_out_depth = Input_depth_dim * Output_depth_dim;

  unsigned int filter_FP_length = Output_depth_dim * Mask_Y_dim * Mask_X_dim * Input_depth_dim;
  float* filter_FP_copy = (float*)_mm_malloc(filter_FP_length * sizeof(float), 64);
  if (filter_FP_copy == NULL) {
    printf("\nerror with malloc allocating filter array copy");
    exit(EXIT_FAILURE);
  }
  
  #pragma omp parallel
  {

    // array copying - filter_FP into form usable for vectorising m loop
    #pragma omp for collapse(3) schedule(static)
      for (unsigned int y = 0; y < Mask_Y_dim; y++) {
        for (unsigned int x = 0; x < Mask_X_dim; x++) {
          for (unsigned int d = 0; d < Input_depth_dim; d++) {
            for (unsigned int m = 0; m < Output_depth_dim; m++) {
              unsigned long long int old_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                + y * Mask_X_dim * Input_depth_dim
                + x * Input_depth_dim
                + d;
                  
              unsigned long long int new_subscript = y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                + x * Input_depth_dim * Output_depth_dim
                + d * Output_depth_dim
                + m;

              filter_FP_copy[new_subscript] = filter_FP[old_subscript];
            }
          }
        }
      }
    #pragma omp barrier
    // main loop body
    #pragma omp for private(bias, bias2, temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8, s, s2, s3, s4, w, w2)\
     collapse(5) schedule(static) 
    for (unsigned int mm = 0; mm < Output_depth_dim; mm += m_tile) {  // loop tiling not really worth here since there should be at least 2 loops between the tiling and the tiled loop
                                                                      // experiment with moving loops around, b,y,m / others
                                                                      // b should be the outermost loop no matter what???
      for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
        for (unsigned int m = mm; m < mm + m_tile; m+=16) { //channels
          bias = _mm256_load_ps(&bias_array_FP[m]);
          bias2 = _mm256_load_ps( &bias_array_FP[m+8]);

          for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
            for (unsigned int x = 0; x < Output_X_dim; x+=4) {	//Output Width
              temp = _mm256_setzero_ps();
              temp2 = _mm256_setzero_ps();
              temp3 = _mm256_setzero_ps();
              temp4 = _mm256_setzero_ps();
              temp5 = _mm256_setzero_ps();
              temp6 = _mm256_setzero_ps();
              temp7 = _mm256_setzero_ps();
              temp8 = _mm256_setzero_ps();
              // bias = bias_array_FP[m];
              // temp = 0.0f;
              for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
                for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                  for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                    unsigned long long int in_subscript = b * yx_in_depth
                      + (y * Stride_Y_dim + off_y) * x_in_depth
                      + (x * Stride_X_dim + off_x) * Input_depth_dim
                      + d;

                    unsigned long long int filter_subscript = off_y * mask_in_out_depth
                      + off_x * Input_depth_dim * Output_depth_dim
                      + d * Output_depth_dim
                      + m;


                    s = _mm256_set1_ps(in_FP[in_subscript]);        // d, x
                    s2 = _mm256_set1_ps(in_FP[in_subscript + Input_depth_dim]); // d, x+1
                    s3 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 2)]); // d, x+2
                    s4 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 3)]); // d, x+3

                    w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);      // d, m
                    w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]); // d, m+8

                    temp = _mm256_fmadd_ps(s, w, temp);
                    temp2 = _mm256_fmadd_ps(s2, w, temp2);
                    temp3 = _mm256_fmadd_ps(s3, w, temp3);
                    temp4 = _mm256_fmadd_ps(s4, w, temp4);
                    temp5 = _mm256_fmadd_ps(s, w2, temp5);
                    temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                    temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                    temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                    // __________________________________________________________________________
                    // |
                    // |
                    // v d+1

                    s = _mm256_set1_ps(in_FP[in_subscript + 1]);        // d+1, x
                    s2 = _mm256_set1_ps(in_FP[in_subscript + Input_depth_dim + 1]); // d+1, x+1
                    s3 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 2) + 1]); // d+1, x+2
                    s4 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 3) + 1]); // d+1, x+3

                    w = _mm256_load_ps(&filter_FP_copy[filter_subscript + Output_depth_dim]);      // d+1, m
                    w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + Output_depth_dim + 8]); // d+1, m+8

                    temp = _mm256_fmadd_ps(s, w, temp);
                    temp2 = _mm256_fmadd_ps(s2, w, temp2);
                    temp3 = _mm256_fmadd_ps(s3, w, temp3);
                    temp4 = _mm256_fmadd_ps(s4, w, temp4);
                    temp5 = _mm256_fmadd_ps(s, w2, temp5);
                    temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                    temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                    temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                    // __________________________________________________________________________
                    // |
                    // |
                    // v d+2

                    s = _mm256_set1_ps(in_FP[in_subscript + 2]);        // d+2, x
                    s2 = _mm256_set1_ps(in_FP[in_subscript + Input_depth_dim + 2]); // d+2, x+1
                    s3 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 2) + 2]); // d+2, x+2
                    s4 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 3) + 2]); // d+2, x+3

                    w = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 2)]);      // d+2, m
                    w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 2) + 8]); // d+2, m+8

                    temp = _mm256_fmadd_ps(s, w, temp);
                    temp2 = _mm256_fmadd_ps(s2, w, temp2);
                    temp3 = _mm256_fmadd_ps(s3, w, temp3);
                    temp4 = _mm256_fmadd_ps(s4, w, temp4);
                    temp5 = _mm256_fmadd_ps(s, w2, temp5);
                    temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                    temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                    temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                    // __________________________________________________________________________
                    // |
                    // |
                    // v d+3

                    s = _mm256_set1_ps(in_FP[in_subscript + 3]);        // d+3, x
                    s2 = _mm256_set1_ps(in_FP[in_subscript + Input_depth_dim + 3]); // d+3, x+1
                    s3 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 2) + 3]); // d+3, x+2
                    s4 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 3) + 3]); // d+3, x+3

                    w = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 3)]);      // d+3, m
                    w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 3) + 8]); // d+3, m+8

                    temp = _mm256_fmadd_ps(s, w, temp);
                    temp2 = _mm256_fmadd_ps(s2, w, temp2);
                    temp3 = _mm256_fmadd_ps(s3, w, temp3);
                    temp4 = _mm256_fmadd_ps(s4, w, temp4);
                    temp5 = _mm256_fmadd_ps(s, w2, temp5);
                    temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                    temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                    temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                    // __________________________________________________________________________
                    // |
                    // |
                    // v d+4

                    s = _mm256_set1_ps(in_FP[in_subscript + 4]);        // d+4, x
                    s2 = _mm256_set1_ps(in_FP[in_subscript + Input_depth_dim + 4]); // d+4, x+1
                    s3 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 2) + 4]); // d+4, x+2
                    s4 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 3) + 4]); // d+4, x+3

                    w = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 4)]);      // d+4, m
                    w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 4) + 8]); // d+4, m+8

                    temp = _mm256_fmadd_ps(s, w, temp);
                    temp2 = _mm256_fmadd_ps(s2, w, temp2);
                    temp3 = _mm256_fmadd_ps(s3, w, temp3);
                    temp4 = _mm256_fmadd_ps(s4, w, temp4);
                    temp5 = _mm256_fmadd_ps(s, w2, temp5);
                    temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                    temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                    temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                    // __________________________________________________________________________
                    // |
                    // |
                    // v d+5

                    s = _mm256_set1_ps(in_FP[in_subscript + 5]);        // d+5, x
                    s2 = _mm256_set1_ps(in_FP[in_subscript + Input_depth_dim + 5]); // d+5, x+1
                    s3 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 2) + 5]); // d+5, x+2
                    s4 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 3) + 5]); // d+5, x+3

                    w = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 5)]);      // d+5, m
                    w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 5) + 8]); // d+5, m+8

                    temp = _mm256_fmadd_ps(s, w, temp);
                    temp2 = _mm256_fmadd_ps(s2, w, temp2);
                    temp3 = _mm256_fmadd_ps(s3, w, temp3);
                    temp4 = _mm256_fmadd_ps(s4, w, temp4);
                    temp5 = _mm256_fmadd_ps(s, w2, temp5);
                    temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                    temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                    temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                    // __________________________________________________________________________
                    // |
                    // |
                    // v d+6

                    s = _mm256_set1_ps(in_FP[in_subscript + 6]);        // d+6, x
                    s2 = _mm256_set1_ps(in_FP[in_subscript + Input_depth_dim + 6]); // d+6, x+1
                    s3 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 2) + 6]); // d+6, x+2
                    s4 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 3) + 6]); // d+6, x+3

                    w = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 6)]);      // d+6, m
                    w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 6) + 8]); // d+6, m+8

                    temp = _mm256_fmadd_ps(s, w, temp);
                    temp2 = _mm256_fmadd_ps(s2, w, temp2);
                    temp3 = _mm256_fmadd_ps(s3, w, temp3);
                    temp4 = _mm256_fmadd_ps(s4, w, temp4);
                    temp5 = _mm256_fmadd_ps(s, w2, temp5);
                    temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                    temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                    temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                    // __________________________________________________________________________
                    // |
                    // |
                    // v d+7

                    s = _mm256_set1_ps(in_FP[in_subscript + 7]);        // d+7, x
                    s2 = _mm256_set1_ps(in_FP[in_subscript + Input_depth_dim + 7]); // d+7, x+1
                    s3 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 2) + 7]); // d+7, x+2
                    s4 = _mm256_set1_ps(in_FP[in_subscript + (Input_depth_dim * 3) + 7]); // d+7, x+3

                    w = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 7)]);      // d+7, m
                    w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + (Output_depth_dim * 7) + 8]); // d+7, m+8

                    temp = _mm256_fmadd_ps(s, w, temp);
                    temp2 = _mm256_fmadd_ps(s2, w, temp2);
                    temp3 = _mm256_fmadd_ps(s3, w, temp3);
                    temp4 = _mm256_fmadd_ps(s4, w, temp4);
                    temp5 = _mm256_fmadd_ps(s, w2, temp5);
                    temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                    temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                    temp8 = _mm256_fmadd_ps(s4, w2, temp8);

                    // float s = in_FP[in_subscript];
                    // float w = filter_FP[filter_subscript];
                    // temp = temp + s * w;
                  }
                }
              }


              unsigned long long int out_subscript = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
                y * (Output_depth_dim * Output_X_dim) +
                x * Output_depth_dim
                + m;

              
              temp = _mm256_add_ps(temp, bias);
              temp = _mm256_max_ps(temp, _mm256_setzero_ps());
              _mm256_store_ps(&out_to_compare_with_FP[out_subscript], temp);

              temp2 = _mm256_add_ps(temp2, bias);
              temp2 = _mm256_max_ps(temp2, _mm256_setzero_ps());
              _mm256_store_ps(&out_to_compare_with_FP[out_subscript + Output_depth_dim], temp2); // x+1

              temp3 = _mm256_add_ps(temp3, bias);
              temp3 = _mm256_max_ps(temp3, _mm256_setzero_ps());
              _mm256_store_ps(&out_to_compare_with_FP[out_subscript + (Output_depth_dim * 2)], temp3); // x+2

              temp4 = _mm256_add_ps(temp4, bias);
              temp4 = _mm256_max_ps(temp4, _mm256_setzero_ps());
              _mm256_store_ps(&out_to_compare_with_FP[out_subscript + (Output_depth_dim * 3)], temp4); // x+3

              temp5 = _mm256_add_ps(temp5, bias2);
              temp5 = _mm256_max_ps(temp5, _mm256_setzero_ps());
              _mm256_store_ps(&out_to_compare_with_FP[out_subscript + 8], temp5); // m+8

              temp6 = _mm256_add_ps(temp6, bias2);
              temp6 = _mm256_max_ps(temp6, _mm256_setzero_ps());
              _mm256_store_ps(&out_to_compare_with_FP[out_subscript + Output_depth_dim + 8], temp6); // m+8, x+1

              temp7 = _mm256_add_ps(temp7, bias2);
              temp7 = _mm256_max_ps(temp7, _mm256_setzero_ps());
              _mm256_store_ps(&out_to_compare_with_FP[out_subscript + (Output_depth_dim * 2) + 8], temp7); // m+8, x+2

              temp8 = _mm256_add_ps(temp8, bias2);
              temp8 = _mm256_max_ps(temp8, _mm256_setzero_ps());
              _mm256_store_ps(&out_to_compare_with_FP[out_subscript + (Output_depth_dim * 3) + 8], temp8); // m+8, x+3
              // temp += bias;
              // out_to_compare_with_FP[out_subscript] = Relu_float(temp);

            }
          }
        }
      }
    }
  }
  printf("\n from optimised_layer_v15 %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  _mm_free(filter_FP_copy);
  return 0;
}


// naming convention optimised_layer_vX_AC_OptApplied_FP/Char



// 2 GFLOPS
int unoptimized_layer_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {

  float temp, bias;

  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int m = 0; m < Output_depth_dim; m++) { //channels
      for (unsigned int od = 0; od < 1; od++) {
        for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
          for (unsigned int x = 0; x < Output_X_dim; x++) {	//Output Width
            bias = bias_array_FP[m];
            temp = 0.0f;
            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                for (unsigned int d = 0; d < Input_depth_dim; d++) {

                  unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                    + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                    + (x * Stride_X_dim + off_x) * Input_depth_dim
                    + d;

                  unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                    + off_y * Mask_X_dim * Input_depth_dim
                    + off_x * Input_depth_dim
                    + d;

                  float s = in_FP[in_subscript];
                  float w = filter_FP[filter_subscript];
                  temp = temp + s * w;


                }
              }
            }


            unsigned long long int out_subscript = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
              y * (Output_depth_dim * Output_X_dim) +
              x * Output_depth_dim
              + m;

            temp += bias;
            out_to_compare_with_FP[out_subscript] = Relu_float(temp);

          }
        }
      }
    }
  }

  printf("\n from unopt %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


float Relu_float(const float temp) {

  // return temp > 0.0f ? temp : 0.0f;
  if (temp < 0.0f)
    return 0.0f;
  else
    return temp;

}