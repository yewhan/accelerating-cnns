#include "convolution_layer_2D.h"

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))




// loop copying applied, to enable vectorisation
// avx instructions applied
// moved bias load outside of x to m loop
// 10 GFLOPS, 5x speedup from unopt using -O3
int optimised_layer_arraycopying_vectorised(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
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
int optimised_layerv2_unroll_x(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
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
  LIKWID_MARKER_STOP("foo");
  LIKWID_MARKER_CLOSE;
  return 0;

}


float Relu_float(const float temp) {

  // return temp > 0.0f ? temp : 0.0f;
  if (temp < 0.0f)
    return 0.0f;
  else
    return temp;

}