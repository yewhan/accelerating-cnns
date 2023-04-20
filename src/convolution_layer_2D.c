#include "convolution_layer_2D.h"

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))




// loop copying applied, to enable vectorisation
// avx instructions applied
// moved bias load outside of x to m loop
// 10 GFLOPS, 5x speedup from unopt using -O3
int optimised_layerv1_arraycopying_vectorised(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  __m256 bias, temp, s, w;

  unsigned int filter_FP_length = Output_depth_dim * Mask_Y_dim * Mask_X_dim * Input_depth_dim;
  float* filter_FP_copy = (float*)_mm_malloc(filter_FP_length * sizeof(float), 64);
  if (filter_FP_copy == NULL) {
    printf("\nerror with malloc allocating filter array copy");
    exit(EXIT_FAILURE);
  }

  // array copying - filter_FP into form usable for vectorising m loop
  for (int m = 0; m < Output_depth_dim; m += 8) {
    for (int y = 0; y < Mask_Y_dim; y++) {
      for (int x = 0; x < Mask_X_dim; x++) {
        for (int d = 0; d < Input_depth_dim; d++) {
          for (int mm = m; mm < m + 8; mm++) {
            unsigned long long int old_subscript = mm * Mask_Y_dim * Mask_X_dim * Input_depth_dim
              + y * Mask_X_dim * Input_depth_dim
              + x * Input_depth_dim
              + d;
              
            unsigned long long int new_subscript = y * Mask_X_dim * Input_depth_dim * Output_depth_dim
              + x * Input_depth_dim * Output_depth_dim
              + d * Output_depth_dim
              + mm;

            filter_FP_copy[new_subscript] = filter_FP[old_subscript];
          }
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
int optimised_layerv2_unroll_x2(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  __m256 bias, temp, temp2, s, s2, w;

  unsigned int filter_FP_length = Output_depth_dim * Mask_Y_dim * Mask_X_dim * Input_depth_dim;
  float* filter_FP_copy = (float*)_mm_malloc(filter_FP_length * sizeof(float), 64);
  if (filter_FP_copy == NULL) {
    printf("\nerror with malloc allocating filter array copy");
    exit(EXIT_FAILURE);
  }

  // array copying - filter_FP into form usable for vectorising m loop
  for (int m = 0; m < Output_depth_dim; m += 8) {
    for (int y = 0; y < Mask_Y_dim; y++) {
      for (int x = 0; x < Mask_X_dim; x++) {
        for (int d = 0; d < Input_depth_dim; d++) {
          for (int mm = m; mm < m + 8; mm++) {
            unsigned long long int old_subscript = mm * Mask_Y_dim * Mask_X_dim * Input_depth_dim
              + y * Mask_X_dim * Input_depth_dim
              + x * Input_depth_dim
              + d;
              
            unsigned long long int new_subscript = y * Mask_X_dim * Input_depth_dim * Output_depth_dim
              + x * Input_depth_dim * Output_depth_dim
              + d * Output_depth_dim
              + mm;

            filter_FP_copy[new_subscript] = filter_FP[old_subscript];
          }
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
int optimised_layerv3_unroll_x4(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  __m256 bias, temp, temp2, temp3, temp4, s, s2, s3, s4, w;

  unsigned int filter_FP_length = Output_depth_dim * Mask_Y_dim * Mask_X_dim * Input_depth_dim;
  float* filter_FP_copy = (float*)_mm_malloc(filter_FP_length * sizeof(float), 64);
  if (filter_FP_copy == NULL) {
    printf("\nerror with malloc allocating filter array copy");
    exit(EXIT_FAILURE);
  }

  // array copying - filter_FP into form usable for vectorising m loop

  for (int m = 0; m < Output_depth_dim; m += 8) {
    for (int y = 0; y < Mask_Y_dim; y++) {
      for (int x = 0; x < Mask_X_dim; x++) {
        for (int d = 0; d < Input_depth_dim; d++) {
          for (int mm = m; mm < m + 8; mm++) {
            unsigned long long int old_subscript = mm * Mask_Y_dim * Mask_X_dim * Input_depth_dim
              + y * Mask_X_dim * Input_depth_dim
              + x * Input_depth_dim
              + d;
              
            unsigned long long int new_subscript = y * Mask_X_dim * Input_depth_dim * Output_depth_dim
              + x * Input_depth_dim * Output_depth_dim
              + d * Output_depth_dim
              + mm;

            filter_FP_copy[new_subscript] = filter_FP[old_subscript];
          }
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
int optimised_layerv4_unroll_m16(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  __m256 bias, bias2, temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8, s, s2, s3, s4, w, w2;

  unsigned int filter_FP_length = Output_depth_dim * Mask_Y_dim * Mask_X_dim * Input_depth_dim;
  float* filter_FP_copy = (float*)_mm_malloc(filter_FP_length * sizeof(float), 64);
  if (filter_FP_copy == NULL) {
    printf("\nerror with malloc allocating filter array copy");
    exit(EXIT_FAILURE);
  }

  // array copying - filter_FP into form usable for vectorising m loop

  for (int m = 0; m < Output_depth_dim; m += 8) {
    for (int y = 0; y < Mask_Y_dim; y++) {
      for (int x = 0; x < Mask_X_dim; x++) {
        for (int d = 0; d < Input_depth_dim; d++) {
          for (int mm = m; mm < m + 8; mm++) {
            unsigned long long int old_subscript = mm * Mask_Y_dim * Mask_X_dim * Input_depth_dim
              + y * Mask_X_dim * Input_depth_dim
              + x * Input_depth_dim
              + d;
              
            unsigned long long int new_subscript = y * Mask_X_dim * Input_depth_dim * Output_depth_dim
              + x * Input_depth_dim * Output_depth_dim
              + d * Output_depth_dim
              + mm;

            filter_FP_copy[new_subscript] = filter_FP[old_subscript];
          }
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
int optimised_layerv5_strength_reduction(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  __m256 bias, bias2, temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8, s, s2, s3, s4, w, w2;

  unsigned int filter_FP_length = Output_depth_dim * Mask_Y_dim * Mask_X_dim * Input_depth_dim;
  float* filter_FP_copy = (float*)_mm_malloc(filter_FP_length * sizeof(float), 64);
  if (filter_FP_copy == NULL) {
    printf("\nerror with malloc allocating filter array copy");
    exit(EXIT_FAILURE);
  }

  // array copying - filter_FP into form usable for vectorising m loop

  for (int m = 0; m < Output_depth_dim; m += 8) {
    for (int y = 0; y < Mask_Y_dim; y++) {
      for (int x = 0; x < Mask_X_dim; x++) {
        for (int d = 0; d < Input_depth_dim; d++) {
          for (int mm = m; mm < m + 8; mm++) {
            unsigned long long int old_subscript = mm * Mask_Y_dim * Mask_X_dim * Input_depth_dim
              + y * Mask_X_dim * Input_depth_dim
              + x * Input_depth_dim
              + d;
              
            unsigned long long int new_subscript = y * Mask_X_dim * Input_depth_dim * Output_depth_dim
              + x * Input_depth_dim * Output_depth_dim
              + d * Output_depth_dim
              + mm;

            filter_FP_copy[new_subscript] = filter_FP[old_subscript];
          }
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