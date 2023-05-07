#include "../convolution_layer_2D.h"

// using array copying

// start from scratch



// loop copying applied, to enable vectorisation on m loop (means 8 outputs calculated at once)
// avx instructions applied
// moved bias load outside of x to m loop
// ~10 GFLOPS, 5x speedup from unopt using -O3
int optimised_layer_v1_AC_vectorised_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {

  __m256 bias, temp;

  unsigned int filter_FP_length = Output_depth_dim * Mask_Y_dim * Mask_X_dim * Input_depth_dim;
  float* filter_FP_copy = (float*)_mm_malloc(filter_FP_length * sizeof(float), 64);
  if (filter_FP_copy == NULL) {
    printf("\nerror with malloc allocating filter array copy");
    exit(EXIT_FAILURE);
  }

  // array copying - filter_FP into form usable for vectorising m loop
  unsigned long long int new_subscript = 0;
  for (unsigned int y = 0; y < Mask_Y_dim; y++) {
    for (unsigned int x = 0; x < Mask_X_dim; x++) {
      for (unsigned int d = 0; d < Input_depth_dim; d++) {
        for (unsigned int m = 0; m < Output_depth_dim; m++) {
          unsigned long long int old_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
            + y * Mask_X_dim * Input_depth_dim
            + x * Input_depth_dim
            + d;
              
          // unsigned long long int new_subscript = y * Mask_X_dim * Input_depth_dim * Output_depth_dim
          //   + x * Input_depth_dim * Output_depth_dim
          //   + d * Output_depth_dim
          //   + m;

          filter_FP_copy[new_subscript] = filter_FP[old_subscript];
          new_subscript++;
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

                
                __m256 s = _mm256_set1_ps(in_FP[in_subscript]);
                __m256 w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
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
  printf("\n from optv1 AC %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  _mm_free(filter_FP_copy);
  return 0;
}






// v2 - unroll x/ m loop



// unrolled x by 2
// ~18 GFLOPS
int optimised_layer_v2_AC_unroll_x2_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  __m256 bias, temp, temp2;

  unsigned int filter_FP_length = Output_depth_dim * Mask_Y_dim * Mask_X_dim * Input_depth_dim;
  float* filter_FP_copy = (float*)_mm_malloc(filter_FP_length * sizeof(float), 64);
  if (filter_FP_copy == NULL) {
    printf("\nerror with malloc allocating filter array copy");
    exit(EXIT_FAILURE);
  }

  // array copying - filter_FP into form usable for vectorising m loop
  unsigned long long int new_subscript = 0;
  for (unsigned int y = 0; y < Mask_Y_dim; y++) {
    for (unsigned int x = 0; x < Mask_X_dim; x++) {
      for (unsigned int d = 0; d < Input_depth_dim; d++) {
        for (unsigned int m = 0; m < Output_depth_dim; m++) {
          unsigned long long int old_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
            + y * Mask_X_dim * Input_depth_dim
            + x * Input_depth_dim
            + d;
              
          // unsigned long long int new_subscript = y * Mask_X_dim * Input_depth_dim * Output_depth_dim
          //   + x * Input_depth_dim * Output_depth_dim
          //   + d * Output_depth_dim
          //   + m;

          filter_FP_copy[new_subscript] = filter_FP[old_subscript];
          new_subscript++;
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

                
                __m256 s = _mm256_set1_ps(in_FP[in_subscript]);
                __m256 s2 = _mm256_set1_ps(in_FP[in_subscript2]);

                __m256 w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);

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
  printf("\n from AC optv2 x2 %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  _mm_free(filter_FP_copy);
  return 0;
}


// unrolled x by 4
// ~34 GFLOPS
int optimised_layer_v2_AC_unroll_x4_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  __m256 bias, temp, temp2, temp3, temp4;

  unsigned int filter_FP_length = Output_depth_dim * Mask_Y_dim * Mask_X_dim * Input_depth_dim;
  float* filter_FP_copy = (float*)_mm_malloc(filter_FP_length * sizeof(float), 64);
  if (filter_FP_copy == NULL) {
    printf("\nerror with malloc allocating filter array copy");
    exit(EXIT_FAILURE);
  }

  // array copying - filter_FP into form usable for vectorising m loop
  unsigned long long int new_subscript = 0;
  for (unsigned int y = 0; y < Mask_Y_dim; y++) {
    for (unsigned int x = 0; x < Mask_X_dim; x++) {
      for (unsigned int d = 0; d < Input_depth_dim; d++) {
        for (unsigned int m = 0; m < Output_depth_dim; m++) {
          unsigned long long int old_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
            + y * Mask_X_dim * Input_depth_dim
            + x * Input_depth_dim
            + d;
              
          // unsigned long long int new_subscript = y * Mask_X_dim * Input_depth_dim * Output_depth_dim
          //   + x * Input_depth_dim * Output_depth_dim
          //   + d * Output_depth_dim
          //   + m;

          filter_FP_copy[new_subscript] = filter_FP[old_subscript];
          new_subscript++;
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

                
                __m256 s = _mm256_set1_ps(in_FP[in_subscript]);
                __m256 s2 = _mm256_set1_ps(in_FP[in_subscript2]);
                __m256 s3 = _mm256_set1_ps(in_FP[in_subscript3]);
                __m256 s4 = _mm256_set1_ps(in_FP[in_subscript4]);

                __m256 w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);

                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s3, w, temp3);
                temp4 = _mm256_fmadd_ps(s4, w, temp4);
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

        }
      }
    }
  }
  printf("\n from AC optv2 x4 %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  _mm_free(filter_FP_copy);
  return 0;
}


// unrolled x by 8
// ~55 GFLOPS
int optimised_layer_v2_AC_unroll_x8_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  __m256 bias, temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

  unsigned int filter_FP_length = Output_depth_dim * Mask_Y_dim * Mask_X_dim * Input_depth_dim;
  float* filter_FP_copy = (float*)_mm_malloc(filter_FP_length * sizeof(float), 64);
  if (filter_FP_copy == NULL) {
    printf("\nerror with malloc allocating filter array copy");
    exit(EXIT_FAILURE);
  }

  // array copying - filter_FP into form usable for vectorising m loop
  unsigned long long int new_subscript = 0;
  for (unsigned int y = 0; y < Mask_Y_dim; y++) {
    for (unsigned int x = 0; x < Mask_X_dim; x++) {
      for (unsigned int d = 0; d < Input_depth_dim; d++) {
        for (unsigned int m = 0; m < Output_depth_dim; m++) {
          unsigned long long int old_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
            + y * Mask_X_dim * Input_depth_dim
            + x * Input_depth_dim
            + d;
              
          // unsigned long long int new_subscript = y * Mask_X_dim * Input_depth_dim * Output_depth_dim
          //   + x * Input_depth_dim * Output_depth_dim
          //   + d * Output_depth_dim
          //   + m;

          filter_FP_copy[new_subscript] = filter_FP[old_subscript];
          new_subscript++;
        }
      }
    }
  }


  // main loop body
  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int m = 0; m < Output_depth_dim; m+=8) { //channels
      bias = _mm256_load_ps(&bias_array_FP[m]);

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (unsigned int x = 0; x < Output_X_dim; x+=8) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          temp4 = _mm256_setzero_ps();
          temp5 = _mm256_setzero_ps();
          temp6 = _mm256_setzero_ps();
          temp7 = _mm256_setzero_ps();
          temp8 = _mm256_setzero_ps();

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
                unsigned long long int in_subscript5 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+4) * Stride_X_dim + off_x) * Input_depth_dim
                  + d;
                unsigned long long int in_subscript6 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+5) * Stride_X_dim + off_x) * Input_depth_dim
                  + d;
                unsigned long long int in_subscript7 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+6) * Stride_X_dim + off_x) * Input_depth_dim
                  + d;
                unsigned long long int in_subscript8 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+7) * Stride_X_dim + off_x) * Input_depth_dim
                  + d;
                

                unsigned long long int filter_subscript = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + m;

                
                __m256 s = _mm256_set1_ps(in_FP[in_subscript]);
                __m256 s2 = _mm256_set1_ps(in_FP[in_subscript2]);
                __m256 s3 = _mm256_set1_ps(in_FP[in_subscript3]);
                __m256 s4 = _mm256_set1_ps(in_FP[in_subscript4]);
                __m256 s5 = _mm256_set1_ps(in_FP[in_subscript5]);
                __m256 s6 = _mm256_set1_ps(in_FP[in_subscript6]);
                __m256 s7 = _mm256_set1_ps(in_FP[in_subscript7]);
                __m256 s8 = _mm256_set1_ps(in_FP[in_subscript8]);


                __m256 w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);

                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s3, w, temp3);
                temp4 = _mm256_fmadd_ps(s4, w, temp4);
                temp5 = _mm256_fmadd_ps(s5, w, temp5);
                temp6 = _mm256_fmadd_ps(s6, w, temp6);
                temp7 = _mm256_fmadd_ps(s7, w, temp7);
                temp8 = _mm256_fmadd_ps(s8, w, temp8);
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
            (x+4) * Output_depth_dim
            + m;
          unsigned long long int out_subscript6 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+5) * Output_depth_dim
            + m;
          unsigned long long int out_subscript7 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+6) * Output_depth_dim
            + m;
          unsigned long long int out_subscript8 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+7) * Output_depth_dim
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

          temp5 = _mm256_add_ps(temp5, bias);
          temp5 = _mm256_max_ps(temp5, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript5], temp5);

          temp6 = _mm256_add_ps(temp6, bias);
          temp6 = _mm256_max_ps(temp6, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript6], temp6);

          temp7 = _mm256_add_ps(temp7, bias);
          temp7 = _mm256_max_ps(temp7, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript7], temp7);

          temp8 = _mm256_add_ps(temp8, bias);
          temp8 = _mm256_max_ps(temp8, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript8], temp8);

        }
      }
    }
  }
  printf("\n from AC optv2 x8 %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  _mm_free(filter_FP_copy);
  return 0;
}


// unrolled m by factor of 2 (m16)
// ~19 GFLOPS
int optimised_layer_v2_AC_unroll_m16_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {

  __m256 bias, bias2, temp, temp2;

  unsigned int filter_FP_length = Output_depth_dim * Mask_Y_dim * Mask_X_dim * Input_depth_dim;
  float* filter_FP_copy = (float*)_mm_malloc(filter_FP_length * sizeof(float), 64);
  if (filter_FP_copy == NULL) {
    printf("\nerror with malloc allocating filter array copy");
    exit(EXIT_FAILURE);
  }

  // array copying - filter_FP into form usable for vectorising m loop
  unsigned long long int new_subscript = 0;
  for (unsigned int y = 0; y < Mask_Y_dim; y++) {
    for (unsigned int x = 0; x < Mask_X_dim; x++) {
      for (unsigned int d = 0; d < Input_depth_dim; d++) {
        for (unsigned int m = 0; m < Output_depth_dim; m++) {
          unsigned long long int old_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
            + y * Mask_X_dim * Input_depth_dim
            + x * Input_depth_dim
            + d;
              
          // unsigned long long int new_subscript = y * Mask_X_dim * Input_depth_dim * Output_depth_dim
          //   + x * Input_depth_dim * Output_depth_dim
          //   + d * Output_depth_dim
          //   + m;

          filter_FP_copy[new_subscript] = filter_FP[old_subscript];
          new_subscript++;
        }
      }
    }
  }

  // main loop body
  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int m = 0; m < Output_depth_dim; m+=16) { //channels
      bias = _mm256_load_ps(&bias_array_FP[m]);
      bias2 = _mm256_load_ps(&bias_array_FP[m+8]);

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (unsigned int x = 0; x < Output_X_dim; x++) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();

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
                unsigned long long int filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + (m+8);

                
                __m256 s = _mm256_set1_ps(in_FP[in_subscript]);

                __m256 w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                __m256 w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript2]);
                
                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s, w2, temp2);
              }
            }
          }


          unsigned long long int out_subscript = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + m;
          unsigned long long int out_subscript2 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+8);

          
          temp = _mm256_add_ps(temp, bias);
          temp = _mm256_max_ps(temp, _mm256_setzero_ps()); // merge Relu layer via native AVX intrinsics
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript], temp);

          temp2 = _mm256_add_ps(temp2, bias2);
          temp2 = _mm256_max_ps(temp2, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript2], temp2);
        }
      }
    }
  }
  printf("\n from AC optv2 m16 %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  _mm_free(filter_FP_copy);
  return 0;
}


// unrolled m by factor of 4 (m32)
// ~35 GFLOPS
int optimised_layer_v2_AC_unroll_m32_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {

  __m256 bias, bias2, bias3, bias4, temp, temp2, temp3, temp4;

  unsigned int filter_FP_length = Output_depth_dim * Mask_Y_dim * Mask_X_dim * Input_depth_dim;
  float* filter_FP_copy = (float*)_mm_malloc(filter_FP_length * sizeof(float), 64);
  if (filter_FP_copy == NULL) {
    printf("\nerror with malloc allocating filter array copy");
    exit(EXIT_FAILURE);
  }

  // array copying - filter_FP into form usable for vectorising m loop
  unsigned long long int new_subscript = 0;
  for (unsigned int y = 0; y < Mask_Y_dim; y++) {
    for (unsigned int x = 0; x < Mask_X_dim; x++) {
      for (unsigned int d = 0; d < Input_depth_dim; d++) {
        for (unsigned int m = 0; m < Output_depth_dim; m++) {
          unsigned long long int old_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
            + y * Mask_X_dim * Input_depth_dim
            + x * Input_depth_dim
            + d;
              
          // unsigned long long int new_subscript = y * Mask_X_dim * Input_depth_dim * Output_depth_dim
          //   + x * Input_depth_dim * Output_depth_dim
          //   + d * Output_depth_dim
          //   + m;

          filter_FP_copy[new_subscript] = filter_FP[old_subscript];
          new_subscript++;
        }
      }
    }
  }

  // main loop body
  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int m = 0; m < Output_depth_dim; m+=32) { //channels
      bias = _mm256_load_ps(&bias_array_FP[m]);
      bias2 = _mm256_load_ps(&bias_array_FP[m+8]);
      bias3 = _mm256_load_ps(&bias_array_FP[m+16]);
      bias4 = _mm256_load_ps(&bias_array_FP[m+24]);

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (unsigned int x = 0; x < Output_X_dim; x++) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          temp4 = _mm256_setzero_ps();

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
                unsigned long long int filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + (m+8);
                unsigned long long int filter_subscript3 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + (m+16);
                unsigned long long int filter_subscript4 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + (m+24);
                
                __m256 s = _mm256_set1_ps(in_FP[in_subscript]);

                __m256 w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                __m256 w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript2]);
                __m256 w3 = _mm256_load_ps(&filter_FP_copy[filter_subscript3]);
                __m256 w4 = _mm256_load_ps(&filter_FP_copy[filter_subscript4]);
                
                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s, w2, temp2);
                temp3 = _mm256_fmadd_ps(s, w3, temp3);
                temp4 = _mm256_fmadd_ps(s, w4, temp4);
              }
            }
          }


          unsigned long long int out_subscript = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + m;
          unsigned long long int out_subscript2 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+8);
          unsigned long long int out_subscript3 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+16);
          unsigned long long int out_subscript4 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+24);
          
          
          temp = _mm256_add_ps(temp, bias);
          temp = _mm256_max_ps(temp, _mm256_setzero_ps()); // merge Relu layer via native AVX intrinsics
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript], temp);

          temp2 = _mm256_add_ps(temp2, bias2);
          temp2 = _mm256_max_ps(temp2, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript2], temp2);

          temp3 = _mm256_add_ps(temp3, bias3);
          temp3 = _mm256_max_ps(temp3, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript3], temp3);

          temp4 = _mm256_add_ps(temp4, bias4);
          temp4 = _mm256_max_ps(temp4, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript4], temp4);
        }
      }
    }
  }
  printf("\n from AC optv2 m32 %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  _mm_free(filter_FP_copy);
  return 0;
}


// unrolled m by factor of 8 (m64)
// ~33 GFLOPS, 
int optimised_layer_v2_AC_unroll_m64_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {

  __m256 bias, bias2, bias3, bias4, bias5, bias6, bias7, bias8, temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

  unsigned int filter_FP_length = Output_depth_dim * Mask_Y_dim * Mask_X_dim * Input_depth_dim;
  float* filter_FP_copy = (float*)_mm_malloc(filter_FP_length * sizeof(float), 64);
  if (filter_FP_copy == NULL) {
    printf("\nerror with malloc allocating filter array copy");
    exit(EXIT_FAILURE);
  }

  // array copying - filter_FP into form usable for vectorising m loop
  unsigned long long int new_subscript = 0;
  for (unsigned int y = 0; y < Mask_Y_dim; y++) {
    for (unsigned int x = 0; x < Mask_X_dim; x++) {
      for (unsigned int d = 0; d < Input_depth_dim; d++) {
        for (unsigned int m = 0; m < Output_depth_dim; m++) {
          unsigned long long int old_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
            + y * Mask_X_dim * Input_depth_dim
            + x * Input_depth_dim
            + d;
              
          // unsigned long long int new_subscript = y * Mask_X_dim * Input_depth_dim * Output_depth_dim
          //   + x * Input_depth_dim * Output_depth_dim
          //   + d * Output_depth_dim
          //   + m;

          filter_FP_copy[new_subscript] = filter_FP[old_subscript];
          new_subscript++;
        }
      }
    }
  }

  // main loop body
  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int m = 0; m < Output_depth_dim; m+=64) { //channels
      bias = _mm256_load_ps(&bias_array_FP[m]);
      bias2 = _mm256_load_ps(&bias_array_FP[m+8]);
      bias3 = _mm256_load_ps(&bias_array_FP[m+16]);
      bias4 = _mm256_load_ps(&bias_array_FP[m+24]);
      bias5 = _mm256_load_ps(&bias_array_FP[m+32]);
      bias6 = _mm256_load_ps(&bias_array_FP[m+40]);
      bias7 = _mm256_load_ps(&bias_array_FP[m+48]);
      bias8 = _mm256_load_ps(&bias_array_FP[m+56]);

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (unsigned int x = 0; x < Output_X_dim; x++) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          temp4 = _mm256_setzero_ps();
          temp5 = _mm256_setzero_ps();
          temp6 = _mm256_setzero_ps();
          temp7 = _mm256_setzero_ps();
          temp8 = _mm256_setzero_ps();

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
                unsigned long long int filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + (m+8);
                unsigned long long int filter_subscript3 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + (m+16);
                unsigned long long int filter_subscript4 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + (m+24);
                unsigned long long int filter_subscript5 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + (m+32);
                unsigned long long int filter_subscript6 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + (m+40);
                unsigned long long int filter_subscript7 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + (m+48);
                unsigned long long int filter_subscript8 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + (m+56);
                

                __m256 s = _mm256_set1_ps(in_FP[in_subscript]);

                __m256 w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                __m256 w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript2]);
                __m256 w3 = _mm256_load_ps(&filter_FP_copy[filter_subscript3]);
                __m256 w4 = _mm256_load_ps(&filter_FP_copy[filter_subscript4]);
                __m256 w5 = _mm256_load_ps(&filter_FP_copy[filter_subscript5]);
                __m256 w6 = _mm256_load_ps(&filter_FP_copy[filter_subscript6]);
                __m256 w7 = _mm256_load_ps(&filter_FP_copy[filter_subscript7]);
                __m256 w8 = _mm256_load_ps(&filter_FP_copy[filter_subscript8]);
                
                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s, w2, temp2);
                temp3 = _mm256_fmadd_ps(s, w3, temp3);
                temp4 = _mm256_fmadd_ps(s, w4, temp4);
                temp5 = _mm256_fmadd_ps(s, w5, temp5);
                temp6 = _mm256_fmadd_ps(s, w6, temp6);
                temp7 = _mm256_fmadd_ps(s, w7, temp7);
                temp8 = _mm256_fmadd_ps(s, w8, temp8);
              }
            }
          }


          unsigned long long int out_subscript = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + m;
          unsigned long long int out_subscript2 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+8);
          unsigned long long int out_subscript3 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+16);
          unsigned long long int out_subscript4 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+24);
          unsigned long long int out_subscript5 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+32);
          unsigned long long int out_subscript6 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+40);
          unsigned long long int out_subscript7 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+48);
          unsigned long long int out_subscript8 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+56);
          
          
          temp = _mm256_add_ps(temp, bias);
          temp = _mm256_max_ps(temp, _mm256_setzero_ps()); // merge Relu layer via native AVX intrinsics
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript], temp);

          temp2 = _mm256_add_ps(temp2, bias2);
          temp2 = _mm256_max_ps(temp2, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript2], temp2);

          temp3 = _mm256_add_ps(temp3, bias3);
          temp3 = _mm256_max_ps(temp3, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript3], temp3);

          temp4 = _mm256_add_ps(temp4, bias4);
          temp4 = _mm256_max_ps(temp4, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript4], temp4);

          temp5 = _mm256_add_ps(temp5, bias5);
          temp5 = _mm256_max_ps(temp5, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript5], temp5);

          temp6 = _mm256_add_ps(temp6, bias6);
          temp6 = _mm256_max_ps(temp6, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript6], temp6);

          temp7 = _mm256_add_ps(temp7, bias7);
          temp7 = _mm256_max_ps(temp7, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript7], temp7);

          temp8 = _mm256_add_ps(temp8, bias8);
          temp8 = _mm256_max_ps(temp8, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript8], temp8);
        }
      }
    }
  }
  printf("\n from AC optv2 m64 %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  _mm_free(filter_FP_copy);
  return 0;
}


// unrolled m by factor of 2 (m16), x by 4
// ~61 GFLOPS
int optimised_layer_v2_AC_unroll_x4m16_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {

  __m256 bias, bias2, temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

  unsigned int filter_FP_length = Output_depth_dim * Mask_Y_dim * Mask_X_dim * Input_depth_dim;
  float* filter_FP_copy = (float*)_mm_malloc(filter_FP_length * sizeof(float), 64);
  if (filter_FP_copy == NULL) {
    printf("\nerror with malloc allocating filter array copy");
    exit(EXIT_FAILURE);
  }

  // array copying - filter_FP into form usable for vectorising m loop
  unsigned long long int new_subscript = 0;
  for (unsigned int y = 0; y < Mask_Y_dim; y++) {
    for (unsigned int x = 0; x < Mask_X_dim; x++) {
      for (unsigned int d = 0; d < Input_depth_dim; d++) {
        for (unsigned int m = 0; m < Output_depth_dim; m++) {
          unsigned long long int old_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
            + y * Mask_X_dim * Input_depth_dim
            + x * Input_depth_dim
            + d;
              
          // unsigned long long int new_subscript = y * Mask_X_dim * Input_depth_dim * Output_depth_dim
          //   + x * Input_depth_dim * Output_depth_dim
          //   + d * Output_depth_dim
          //   + m;

          filter_FP_copy[new_subscript] = filter_FP[old_subscript];
          new_subscript++;
        }
      }
    }
  }

  // main loop body
  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int m = 0; m < Output_depth_dim; m+=16) { //channels
      bias = _mm256_load_ps(&bias_array_FP[m]);
      bias2 = _mm256_load_ps(&bias_array_FP[m+8]);

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

                
                __m256 s = _mm256_set1_ps(in_FP[in_subscript]);
                __m256 s2 = _mm256_set1_ps(in_FP[in_subscript2]);
                __m256 s3 = _mm256_set1_ps(in_FP[in_subscript3]);
                __m256 s4 = _mm256_set1_ps(in_FP[in_subscript4]);

                __m256 w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                __m256 w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript2]);
                
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
          temp = _mm256_max_ps(temp, _mm256_setzero_ps()); // merge Relu layer via native AVX intrinsics
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
        }
      }
    }
  }
  printf("\n from AC optv2 m16 x4 %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  _mm_free(filter_FP_copy);
  return 0;
}


// unrolled m by factor of 4 (m32), x by 2
// ~61 GFLOPS
int optimised_layer_v2_AC_unroll_x2m32_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {

  __m256 bias, bias2, bias3, bias4, temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

  unsigned int filter_FP_length = Output_depth_dim * Mask_Y_dim * Mask_X_dim * Input_depth_dim;
  float* filter_FP_copy = (float*)_mm_malloc(filter_FP_length * sizeof(float), 64);
  if (filter_FP_copy == NULL) {
    printf("\nerror with malloc allocating filter array copy");
    exit(EXIT_FAILURE);
  }

  // array copying - filter_FP into form usable for vectorising m loop
  unsigned long long int new_subscript = 0;
  for (unsigned int y = 0; y < Mask_Y_dim; y++) {
    for (unsigned int x = 0; x < Mask_X_dim; x++) {
      for (unsigned int d = 0; d < Input_depth_dim; d++) {
        for (unsigned int m = 0; m < Output_depth_dim; m++) {
          unsigned long long int old_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
            + y * Mask_X_dim * Input_depth_dim
            + x * Input_depth_dim
            + d;
              
          // unsigned long long int new_subscript = y * Mask_X_dim * Input_depth_dim * Output_depth_dim
          //   + x * Input_depth_dim * Output_depth_dim
          //   + d * Output_depth_dim
          //   + m;

          filter_FP_copy[new_subscript] = filter_FP[old_subscript];
          new_subscript++;
        }
      }
    }
  }

  // main loop body
  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int m = 0; m < Output_depth_dim; m+=32) { //channels
      bias = _mm256_load_ps(&bias_array_FP[m]);
      bias2 = _mm256_load_ps(&bias_array_FP[m+8]);
      bias3 = _mm256_load_ps(&bias_array_FP[m+16]);
      bias4 = _mm256_load_ps(&bias_array_FP[m+24]);

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (unsigned int x = 0; x < Output_X_dim; x+=2) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          temp4 = _mm256_setzero_ps();
          temp5 = _mm256_setzero_ps();
          temp6 = _mm256_setzero_ps();
          temp7 = _mm256_setzero_ps();
          temp8 = _mm256_setzero_ps();

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
                unsigned long long int filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + (m+8);
                unsigned long long int filter_subscript3 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + (m+16);
                unsigned long long int filter_subscript4 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + (m+24);
                
                __m256 s = _mm256_set1_ps(in_FP[in_subscript]);
                __m256 s2 = _mm256_set1_ps(in_FP[in_subscript2]);

                __m256 w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                __m256 w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript2]);
                __m256 w3 = _mm256_load_ps(&filter_FP_copy[filter_subscript3]);
                __m256 w4 = _mm256_load_ps(&filter_FP_copy[filter_subscript4]);


                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s, w2, temp3);
                temp4 = _mm256_fmadd_ps(s2, w2, temp4);
                temp5 = _mm256_fmadd_ps(s, w3, temp5);
                temp6 = _mm256_fmadd_ps(s2, w3, temp6);
                temp7 = _mm256_fmadd_ps(s, w4, temp7);
                temp8 = _mm256_fmadd_ps(s2, w4, temp8);
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
            x * Output_depth_dim
            + (m+8);
          unsigned long long int out_subscript4 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+8);
          unsigned long long int out_subscript5 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+16);
          unsigned long long int out_subscript6 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+16);
          unsigned long long int out_subscript7 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+24);
          unsigned long long int out_subscript8 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+24);

          
          temp = _mm256_add_ps(temp, bias);
          temp = _mm256_max_ps(temp, _mm256_setzero_ps()); // merge Relu layer via native AVX intrinsics
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript], temp);

          temp2 = _mm256_add_ps(temp2, bias);
          temp2 = _mm256_max_ps(temp2, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript2], temp2);
          
          temp3 = _mm256_add_ps(temp3, bias2);
          temp3 = _mm256_max_ps(temp3, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript3], temp3);
          
          temp4 = _mm256_add_ps(temp4, bias2);
          temp4 = _mm256_max_ps(temp4, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript4], temp4);
          
          temp5 = _mm256_add_ps(temp5, bias3);
          temp5 = _mm256_max_ps(temp5, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript5], temp5);
          
          temp6 = _mm256_add_ps(temp6, bias3);
          temp6 = _mm256_max_ps(temp6, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript6], temp6);
          
          temp7 = _mm256_add_ps(temp7, bias4);
          temp7 = _mm256_max_ps(temp7, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript7], temp7);
          
          temp8 = _mm256_add_ps(temp8, bias4);
          temp8 = _mm256_max_ps(temp8, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript8], temp8);
        }
      }
    }
  }
  printf("\n from AC optv2 m32 x2 %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  _mm_free(filter_FP_copy);
  return 0;
}


// unrolled m by factor of 4 (m32), x by 2
// ~65 GFLOPS
int optimised_layer_v2_AC_unroll_x3m24_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {

  __m256 bias, bias2, bias3, temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9;
  unsigned int m, x;

  int m_bound = (Output_depth_dim/ 24) * 24;
  int x_bound = (Output_X_dim/ 3) * 3; 

  unsigned int filter_FP_length = Output_depth_dim * Mask_Y_dim * Mask_X_dim * Input_depth_dim;
  float* filter_FP_copy = (float*)_mm_malloc(filter_FP_length * sizeof(float), 64);
  if (filter_FP_copy == NULL) {
    printf("\nerror with malloc allocating filter array copy");
    exit(EXIT_FAILURE);
  }

  // array copying - filter_FP into form usable for vectorising m loop
  unsigned long long int new_subscript = 0;
  for (unsigned int y = 0; y < Mask_Y_dim; y++) {
    for (unsigned int x = 0; x < Mask_X_dim; x++) {
      for (unsigned int d = 0; d < Input_depth_dim; d++) {
        for (unsigned int m = 0; m < Output_depth_dim; m++) {
          unsigned long long int old_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
            + y * Mask_X_dim * Input_depth_dim
            + x * Input_depth_dim
            + d;
              
          // unsigned long long int new_subscript = y * Mask_X_dim * Input_depth_dim * Output_depth_dim
          //   + x * Input_depth_dim * Output_depth_dim
          //   + d * Output_depth_dim
          //   + m;

          filter_FP_copy[new_subscript] = filter_FP[old_subscript];
          new_subscript++;
        }
      }
    }
  }

  // main loop body
  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (m = 0; m < m_bound; m+=24) { //channels
      bias = _mm256_load_ps(&bias_array_FP[m]);
      bias2 = _mm256_load_ps(&bias_array_FP[m+8]);
      bias3 = _mm256_load_ps(&bias_array_FP[m+16]);

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (x = 0; x < x_bound; x+=3) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          temp4 = _mm256_setzero_ps();
          temp5 = _mm256_setzero_ps();
          temp6 = _mm256_setzero_ps();
          temp7 = _mm256_setzero_ps();
          temp8 = _mm256_setzero_ps();
          temp9 = _mm256_setzero_ps();

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

                unsigned long long int filter_subscript = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + m;
                unsigned long long int filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + (m+8);
                unsigned long long int filter_subscript3 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + (m+16);
                
                __m256 s = _mm256_set1_ps(in_FP[in_subscript]);
                __m256 s2 = _mm256_set1_ps(in_FP[in_subscript2]);
                __m256 s3 = _mm256_set1_ps(in_FP[in_subscript3]);

                __m256 w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                __m256 w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript2]);
                __m256 w3 = _mm256_load_ps(&filter_FP_copy[filter_subscript3]);


                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s3, w, temp3);
                temp4 = _mm256_fmadd_ps(s, w2, temp4);
                temp5 = _mm256_fmadd_ps(s2, w2, temp5);
                temp6 = _mm256_fmadd_ps(s3, w2, temp6);
                temp7 = _mm256_fmadd_ps(s, w3, temp7);
                temp8 = _mm256_fmadd_ps(s2, w3, temp8);
                temp9 = _mm256_fmadd_ps(s3, w3, temp9);
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
            x * Output_depth_dim
            + (m+8);
          unsigned long long int out_subscript5 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+8);
          unsigned long long int out_subscript6 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+2) * Output_depth_dim
            + (m+8);
          unsigned long long int out_subscript7 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+16);
          unsigned long long int out_subscript8 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+16);
          unsigned long long int out_subscript9 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+2) * Output_depth_dim
            + (m+16);

          
          temp = _mm256_add_ps(temp, bias);
          temp = _mm256_max_ps(temp, _mm256_setzero_ps()); // merge Relu layer via native AVX intrinsics
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript], temp);

          temp2 = _mm256_add_ps(temp2, bias);
          temp2 = _mm256_max_ps(temp2, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript2], temp2);
          
          temp3 = _mm256_add_ps(temp3, bias);
          temp3 = _mm256_max_ps(temp3, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript3], temp3);
          
          temp4 = _mm256_add_ps(temp4, bias2);
          temp4 = _mm256_max_ps(temp4, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript4], temp4);
          
          temp5 = _mm256_add_ps(temp5, bias2);
          temp5 = _mm256_max_ps(temp5, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript5], temp5);
          
          temp6 = _mm256_add_ps(temp6, bias2);
          temp6 = _mm256_max_ps(temp6, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript6], temp6);
          
          temp7 = _mm256_add_ps(temp7, bias3);
          temp7 = _mm256_max_ps(temp7, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript7], temp7);
          
          temp8 = _mm256_add_ps(temp8, bias3);
          temp8 = _mm256_max_ps(temp8, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript8], temp8);
                    
          temp9 = _mm256_add_ps(temp9, bias3);
          temp9 = _mm256_max_ps(temp9, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript9], temp9);
        }
        // overflow/ fallback x loop
        for (; x < Output_X_dim; x++) {
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();

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
                unsigned long long int filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + (m+8);
                unsigned long long int filter_subscript3 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + (m+16);
                
                __m256 s = _mm256_set1_ps(in_FP[in_subscript]);

                __m256 w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                __m256 w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript2]);
                __m256 w3 = _mm256_load_ps(&filter_FP_copy[filter_subscript3]);


                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s, w2, temp2);
                temp3 = _mm256_fmadd_ps(s, w3, temp3);
              }
            }
          }
          unsigned long long int out_subscript = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + m;
          unsigned long long int out_subscript2 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+8);
          unsigned long long int out_subscript3 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+16);

          temp = _mm256_add_ps(temp, bias);
          temp = _mm256_max_ps(temp, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript], temp);

          temp2 = _mm256_add_ps(temp2, bias2);
          temp2 = _mm256_max_ps(temp2, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript2], temp2);
          
          temp3 = _mm256_add_ps(temp3, bias3);
          temp3 = _mm256_max_ps(temp3, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript3], temp3);

        }
      }
    }
    for (; m < Output_depth_dim; m+=8) {
      bias = _mm256_load_ps(&bias_array_FP[m]);

      for (unsigned int y = 0; y < Output_Y_dim; y++) {
        for (x = 0; x < Output_X_dim; x+=4) {
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          temp4 = _mm256_setzero_ps();

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

                unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;

                __m256 s = _mm256_loadu_ps(&in_FP[in_subscript]);
                __m256 s2 = _mm256_loadu_ps(&in_FP[in_subscript2]);
                __m256 s3 = _mm256_loadu_ps(&in_FP[in_subscript3]);
                __m256 s4 = _mm256_loadu_ps(&in_FP[in_subscript4]);

                __m256 w = _mm256_loadu_ps(&filter_FP_copy[filter_subscript]);

                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s3, w, temp3);
                temp4 = _mm256_fmadd_ps(s4, w, temp4);
                
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
        }
      }
    }
  }
  printf("\n from AC optv2 m24 x3 %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  _mm_free(filter_FP_copy);
  return 0;
}






// v3 - unroll d loop


// unrolled d by factor of 2
// ~53 GFLOPS
int optimised_layer_v3_AC_x2m32_unroll_d2_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {

  __m256 bias, bias2, bias3, bias4, temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

  unsigned int filter_FP_length = Output_depth_dim * Mask_Y_dim * Mask_X_dim * Input_depth_dim;
  float* filter_FP_copy = (float*)_mm_malloc(filter_FP_length * sizeof(float), 64);
  if (filter_FP_copy == NULL) {
    printf("\nerror with malloc allocating filter array copy");
    exit(EXIT_FAILURE);
  }

  // array copying - filter_FP into form usable for vectorising m loop
  unsigned long long int new_subscript = 0;
  for (unsigned int y = 0; y < Mask_Y_dim; y++) {
    for (unsigned int x = 0; x < Mask_X_dim; x++) {
      for (unsigned int d = 0; d < Input_depth_dim; d++) {
        for (unsigned int m = 0; m < Output_depth_dim; m++) {
          unsigned long long int old_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
            + y * Mask_X_dim * Input_depth_dim
            + x * Input_depth_dim
            + d;
              
          // unsigned long long int new_subscript = y * Mask_X_dim * Input_depth_dim * Output_depth_dim
          //   + x * Input_depth_dim * Output_depth_dim
          //   + d * Output_depth_dim
          //   + m;

          filter_FP_copy[new_subscript] = filter_FP[old_subscript];
          new_subscript++;
        }
      }
    }
  }

  // main loop body
  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int m = 0; m < Output_depth_dim; m+=32) { //channels
      bias = _mm256_load_ps(&bias_array_FP[m]);
      bias2 = _mm256_load_ps(&bias_array_FP[m+8]);
      bias3 = _mm256_load_ps(&bias_array_FP[m+16]);
      bias4 = _mm256_load_ps(&bias_array_FP[m+24]);

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (unsigned int x = 0; x < Output_X_dim; x+=2) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          temp4 = _mm256_setzero_ps();
          temp5 = _mm256_setzero_ps();
          temp6 = _mm256_setzero_ps();
          temp7 = _mm256_setzero_ps();
          temp8 = _mm256_setzero_ps();

          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=2) {

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
                unsigned long long int filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + (m+8);
                unsigned long long int filter_subscript3 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + (m+16);
                unsigned long long int filter_subscript4 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + (m+24);
                
                __m256 s = _mm256_set1_ps(in_FP[in_subscript]);
                __m256 s2 = _mm256_set1_ps(in_FP[in_subscript2]);

                __m256 w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                __m256 w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript2]);
                __m256 w3 = _mm256_load_ps(&filter_FP_copy[filter_subscript3]);
                __m256 w4 = _mm256_load_ps(&filter_FP_copy[filter_subscript4]);


                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s, w2, temp3);
                temp4 = _mm256_fmadd_ps(s2, w2, temp4);
                temp5 = _mm256_fmadd_ps(s, w3, temp5);
                temp6 = _mm256_fmadd_ps(s2, w3, temp6);
                temp7 = _mm256_fmadd_ps(s, w4, temp7);
                temp8 = _mm256_fmadd_ps(s2, w4, temp8);


                // d+1 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+1);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+1);

                filter_subscript = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+1) * Output_depth_dim
                  + m;
                filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+1) * Output_depth_dim
                  + (m+8);
                filter_subscript3 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+1) * Output_depth_dim
                  + (m+16);
                filter_subscript4 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+1) * Output_depth_dim
                  + (m+24);
                
                s = _mm256_set1_ps(in_FP[in_subscript]);
                s2 = _mm256_set1_ps(in_FP[in_subscript2]);

                w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript2]);
                w3 = _mm256_load_ps(&filter_FP_copy[filter_subscript3]);
                w4 = _mm256_load_ps(&filter_FP_copy[filter_subscript4]);


                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s, w2, temp3);
                temp4 = _mm256_fmadd_ps(s2, w2, temp4);
                temp5 = _mm256_fmadd_ps(s, w3, temp5);
                temp6 = _mm256_fmadd_ps(s2, w3, temp6);
                temp7 = _mm256_fmadd_ps(s, w4, temp7);
                temp8 = _mm256_fmadd_ps(s2, w4, temp8);
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
            x * Output_depth_dim
            + (m+8);
          unsigned long long int out_subscript4 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+8);
          unsigned long long int out_subscript5 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+16);
          unsigned long long int out_subscript6 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+16);
          unsigned long long int out_subscript7 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+24);
          unsigned long long int out_subscript8 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+24);

          
          temp = _mm256_add_ps(temp, bias);
          temp = _mm256_max_ps(temp, _mm256_setzero_ps()); // merge Relu layer via native AVX intrinsics
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript], temp);

          temp2 = _mm256_add_ps(temp2, bias);
          temp2 = _mm256_max_ps(temp2, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript2], temp2);
          
          temp3 = _mm256_add_ps(temp3, bias2);
          temp3 = _mm256_max_ps(temp3, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript3], temp3);
          
          temp4 = _mm256_add_ps(temp4, bias2);
          temp4 = _mm256_max_ps(temp4, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript4], temp4);
          
          temp5 = _mm256_add_ps(temp5, bias3);
          temp5 = _mm256_max_ps(temp5, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript5], temp5);
          
          temp6 = _mm256_add_ps(temp6, bias3);
          temp6 = _mm256_max_ps(temp6, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript6], temp6);
          
          temp7 = _mm256_add_ps(temp7, bias4);
          temp7 = _mm256_max_ps(temp7, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript7], temp7);
          
          temp8 = _mm256_add_ps(temp8, bias4);
          temp8 = _mm256_max_ps(temp8, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript8], temp8);
        }
      }
    }
  }
  printf("\n from AC optv3 m32 x2 - d2 %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  _mm_free(filter_FP_copy);
  return 0;
}


// unrolled d by factor of 4
// ~44 GFLOPS
int optimised_layer_v3_AC_x2m32_unroll_d4_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {

  __m256 bias, bias2, bias3, bias4, temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

  unsigned int filter_FP_length = Output_depth_dim * Mask_Y_dim * Mask_X_dim * Input_depth_dim;
  float* filter_FP_copy = (float*)_mm_malloc(filter_FP_length * sizeof(float), 64);
  if (filter_FP_copy == NULL) {
    printf("\nerror with malloc allocating filter array copy");
    exit(EXIT_FAILURE);
  }

  // array copying - filter_FP into form usable for vectorising m loop
  unsigned long long int new_subscript = 0;
  for (unsigned int y = 0; y < Mask_Y_dim; y++) {
    for (unsigned int x = 0; x < Mask_X_dim; x++) {
      for (unsigned int d = 0; d < Input_depth_dim; d++) {
        for (unsigned int m = 0; m < Output_depth_dim; m++) {
          unsigned long long int old_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
            + y * Mask_X_dim * Input_depth_dim
            + x * Input_depth_dim
            + d;
              
          // unsigned long long int new_subscript = y * Mask_X_dim * Input_depth_dim * Output_depth_dim
          //   + x * Input_depth_dim * Output_depth_dim
          //   + d * Output_depth_dim
          //   + m;

          filter_FP_copy[new_subscript] = filter_FP[old_subscript];
          new_subscript++;
        }
      }
    }
  }

  // main loop body
  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int m = 0; m < Output_depth_dim; m+=32) { //channels
      bias = _mm256_load_ps(&bias_array_FP[m]);
      bias2 = _mm256_load_ps(&bias_array_FP[m+8]);
      bias3 = _mm256_load_ps(&bias_array_FP[m+16]);
      bias4 = _mm256_load_ps(&bias_array_FP[m+24]);

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (unsigned int x = 0; x < Output_X_dim; x+=2) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          temp4 = _mm256_setzero_ps();
          temp5 = _mm256_setzero_ps();
          temp6 = _mm256_setzero_ps();
          temp7 = _mm256_setzero_ps();
          temp8 = _mm256_setzero_ps();

          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=4) {

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
                unsigned long long int filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + (m+8);
                unsigned long long int filter_subscript3 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + (m+16);
                unsigned long long int filter_subscript4 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + (m+24);
                
                __m256 s = _mm256_set1_ps(in_FP[in_subscript]);
                __m256 s2 = _mm256_set1_ps(in_FP[in_subscript2]);

                __m256 w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                __m256 w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript2]);
                __m256 w3 = _mm256_load_ps(&filter_FP_copy[filter_subscript3]);
                __m256 w4 = _mm256_load_ps(&filter_FP_copy[filter_subscript4]);


                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s, w2, temp3);
                temp4 = _mm256_fmadd_ps(s2, w2, temp4);
                temp5 = _mm256_fmadd_ps(s, w3, temp5);
                temp6 = _mm256_fmadd_ps(s2, w3, temp6);
                temp7 = _mm256_fmadd_ps(s, w4, temp7);
                temp8 = _mm256_fmadd_ps(s2, w4, temp8);


                // d+1 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+1);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+1);

                filter_subscript = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+1) * Output_depth_dim
                  + m;
                filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+1) * Output_depth_dim
                  + (m+8);
                filter_subscript3 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+1) * Output_depth_dim
                  + (m+16);
                filter_subscript4 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+1) * Output_depth_dim
                  + (m+24);
                
                s = _mm256_set1_ps(in_FP[in_subscript]);
                s2 = _mm256_set1_ps(in_FP[in_subscript2]);

                w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript2]);
                w3 = _mm256_load_ps(&filter_FP_copy[filter_subscript3]);
                w4 = _mm256_load_ps(&filter_FP_copy[filter_subscript4]);


                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s, w2, temp3);
                temp4 = _mm256_fmadd_ps(s2, w2, temp4);
                temp5 = _mm256_fmadd_ps(s, w3, temp5);
                temp6 = _mm256_fmadd_ps(s2, w3, temp6);
                temp7 = _mm256_fmadd_ps(s, w4, temp7);
                temp8 = _mm256_fmadd_ps(s2, w4, temp8);


                // d+2 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+2);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+2);

                filter_subscript = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+2) * Output_depth_dim
                  + m;
                filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+2) * Output_depth_dim
                  + (m+8);
                filter_subscript3 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+2) * Output_depth_dim
                  + (m+16);
                filter_subscript4 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+2) * Output_depth_dim
                  + (m+24);
                
                s = _mm256_set1_ps(in_FP[in_subscript]);
                s2 = _mm256_set1_ps(in_FP[in_subscript2]);

                w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript2]);
                w3 = _mm256_load_ps(&filter_FP_copy[filter_subscript3]);
                w4 = _mm256_load_ps(&filter_FP_copy[filter_subscript4]);


                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s, w2, temp3);
                temp4 = _mm256_fmadd_ps(s2, w2, temp4);
                temp5 = _mm256_fmadd_ps(s, w3, temp5);
                temp6 = _mm256_fmadd_ps(s2, w3, temp6);
                temp7 = _mm256_fmadd_ps(s, w4, temp7);
                temp8 = _mm256_fmadd_ps(s2, w4, temp8);


                // d+3 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+3);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+3);

                filter_subscript = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+3) * Output_depth_dim
                  + m;
                filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+3) * Output_depth_dim
                  + (m+8);
                filter_subscript3 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+3) * Output_depth_dim
                  + (m+16);
                filter_subscript4 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+3) * Output_depth_dim
                  + (m+24);
                
                s = _mm256_set1_ps(in_FP[in_subscript]);
                s2 = _mm256_set1_ps(in_FP[in_subscript2]);

                w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript2]);
                w3 = _mm256_load_ps(&filter_FP_copy[filter_subscript3]);
                w4 = _mm256_load_ps(&filter_FP_copy[filter_subscript4]);


                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s, w2, temp3);
                temp4 = _mm256_fmadd_ps(s2, w2, temp4);
                temp5 = _mm256_fmadd_ps(s, w3, temp5);
                temp6 = _mm256_fmadd_ps(s2, w3, temp6);
                temp7 = _mm256_fmadd_ps(s, w4, temp7);
                temp8 = _mm256_fmadd_ps(s2, w4, temp8);
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
            x * Output_depth_dim
            + (m+8);
          unsigned long long int out_subscript4 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+8);
          unsigned long long int out_subscript5 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+16);
          unsigned long long int out_subscript6 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+16);
          unsigned long long int out_subscript7 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+24);
          unsigned long long int out_subscript8 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+24);

          
          temp = _mm256_add_ps(temp, bias);
          temp = _mm256_max_ps(temp, _mm256_setzero_ps()); // merge Relu layer via native AVX intrinsics
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript], temp);

          temp2 = _mm256_add_ps(temp2, bias);
          temp2 = _mm256_max_ps(temp2, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript2], temp2);
          
          temp3 = _mm256_add_ps(temp3, bias2);
          temp3 = _mm256_max_ps(temp3, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript3], temp3);
          
          temp4 = _mm256_add_ps(temp4, bias2);
          temp4 = _mm256_max_ps(temp4, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript4], temp4);
          
          temp5 = _mm256_add_ps(temp5, bias3);
          temp5 = _mm256_max_ps(temp5, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript5], temp5);
          
          temp6 = _mm256_add_ps(temp6, bias3);
          temp6 = _mm256_max_ps(temp6, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript6], temp6);
          
          temp7 = _mm256_add_ps(temp7, bias4);
          temp7 = _mm256_max_ps(temp7, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript7], temp7);
          
          temp8 = _mm256_add_ps(temp8, bias4);
          temp8 = _mm256_max_ps(temp8, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript8], temp8);
        }
      }
    }
  }
  printf("\n from AC optv3 m32 x2 - d4 %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  _mm_free(filter_FP_copy);
  return 0;
}


// unrolled d by factor of 8
// ~64 GFLOPS
int optimised_layer_v3_AC_x2m32_unroll_d8_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {

  __m256 bias, bias2, bias3, bias4, temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

  unsigned int filter_FP_length = Output_depth_dim * Mask_Y_dim * Mask_X_dim * Input_depth_dim;
  float* filter_FP_copy = (float*)_mm_malloc(filter_FP_length * sizeof(float), 64);
  if (filter_FP_copy == NULL) {
    printf("\nerror with malloc allocating filter array copy");
    exit(EXIT_FAILURE);
  }

  // array copying - filter_FP into form usable for vectorising m loop
  unsigned long long int new_subscript = 0;
  for (unsigned int y = 0; y < Mask_Y_dim; y++) {
    for (unsigned int x = 0; x < Mask_X_dim; x++) {
      for (unsigned int d = 0; d < Input_depth_dim; d++) {
        for (unsigned int m = 0; m < Output_depth_dim; m++) {
          unsigned long long int old_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
            + y * Mask_X_dim * Input_depth_dim
            + x * Input_depth_dim
            + d;
              
          // unsigned long long int new_subscript = y * Mask_X_dim * Input_depth_dim * Output_depth_dim
          //   + x * Input_depth_dim * Output_depth_dim
          //   + d * Output_depth_dim
          //   + m;

          filter_FP_copy[new_subscript] = filter_FP[old_subscript];
          new_subscript++;
        }
      }
    }
  }

  // main loop body
  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int m = 0; m < Output_depth_dim; m+=32) { //channels
      bias = _mm256_load_ps(&bias_array_FP[m]);
      bias2 = _mm256_load_ps(&bias_array_FP[m+8]);
      bias3 = _mm256_load_ps(&bias_array_FP[m+16]);
      bias4 = _mm256_load_ps(&bias_array_FP[m+24]);

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (unsigned int x = 0; x < Output_X_dim; x+=2) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          temp4 = _mm256_setzero_ps();
          temp5 = _mm256_setzero_ps();
          temp6 = _mm256_setzero_ps();
          temp7 = _mm256_setzero_ps();
          temp8 = _mm256_setzero_ps();

          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

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
                unsigned long long int filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + (m+8);
                unsigned long long int filter_subscript3 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + (m+16);
                unsigned long long int filter_subscript4 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + (m+24);
                
                __m256 s = _mm256_set1_ps(in_FP[in_subscript]);
                __m256 s2 = _mm256_set1_ps(in_FP[in_subscript2]);

                __m256 w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                __m256 w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript2]);
                __m256 w3 = _mm256_load_ps(&filter_FP_copy[filter_subscript3]);
                __m256 w4 = _mm256_load_ps(&filter_FP_copy[filter_subscript4]);


                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s, w2, temp3);
                temp4 = _mm256_fmadd_ps(s2, w2, temp4);
                temp5 = _mm256_fmadd_ps(s, w3, temp5);
                temp6 = _mm256_fmadd_ps(s2, w3, temp6);
                temp7 = _mm256_fmadd_ps(s, w4, temp7);
                temp8 = _mm256_fmadd_ps(s2, w4, temp8);


                // d+1 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+1);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+1);

                filter_subscript = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+1) * Output_depth_dim
                  + m;
                filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+1) * Output_depth_dim
                  + (m+8);
                filter_subscript3 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+1) * Output_depth_dim
                  + (m+16);
                filter_subscript4 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+1) * Output_depth_dim
                  + (m+24);
                
                s = _mm256_set1_ps(in_FP[in_subscript]);
                s2 = _mm256_set1_ps(in_FP[in_subscript2]);

                w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript2]);
                w3 = _mm256_load_ps(&filter_FP_copy[filter_subscript3]);
                w4 = _mm256_load_ps(&filter_FP_copy[filter_subscript4]);


                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s, w2, temp3);
                temp4 = _mm256_fmadd_ps(s2, w2, temp4);
                temp5 = _mm256_fmadd_ps(s, w3, temp5);
                temp6 = _mm256_fmadd_ps(s2, w3, temp6);
                temp7 = _mm256_fmadd_ps(s, w4, temp7);
                temp8 = _mm256_fmadd_ps(s2, w4, temp8);


                // d+2 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+2);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+2);

                filter_subscript = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+2) * Output_depth_dim
                  + m;
                filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+2) * Output_depth_dim
                  + (m+8);
                filter_subscript3 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+2) * Output_depth_dim
                  + (m+16);
                filter_subscript4 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+2) * Output_depth_dim
                  + (m+24);
                
                s = _mm256_set1_ps(in_FP[in_subscript]);
                s2 = _mm256_set1_ps(in_FP[in_subscript2]);

                w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript2]);
                w3 = _mm256_load_ps(&filter_FP_copy[filter_subscript3]);
                w4 = _mm256_load_ps(&filter_FP_copy[filter_subscript4]);


                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s, w2, temp3);
                temp4 = _mm256_fmadd_ps(s2, w2, temp4);
                temp5 = _mm256_fmadd_ps(s, w3, temp5);
                temp6 = _mm256_fmadd_ps(s2, w3, temp6);
                temp7 = _mm256_fmadd_ps(s, w4, temp7);
                temp8 = _mm256_fmadd_ps(s2, w4, temp8);


                // d+3 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+3);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+3);

                filter_subscript = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+3) * Output_depth_dim
                  + m;
                filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+3) * Output_depth_dim
                  + (m+8);
                filter_subscript3 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+3) * Output_depth_dim
                  + (m+16);
                filter_subscript4 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+3) * Output_depth_dim
                  + (m+24);
                
                s = _mm256_set1_ps(in_FP[in_subscript]);
                s2 = _mm256_set1_ps(in_FP[in_subscript2]);

                w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript2]);
                w3 = _mm256_load_ps(&filter_FP_copy[filter_subscript3]);
                w4 = _mm256_load_ps(&filter_FP_copy[filter_subscript4]);


                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s, w2, temp3);
                temp4 = _mm256_fmadd_ps(s2, w2, temp4);
                temp5 = _mm256_fmadd_ps(s, w3, temp5);
                temp6 = _mm256_fmadd_ps(s2, w3, temp6);
                temp7 = _mm256_fmadd_ps(s, w4, temp7);
                temp8 = _mm256_fmadd_ps(s2, w4, temp8);


                // d+4 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+4);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+4);

                filter_subscript = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+4) * Output_depth_dim
                  + m;
                filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+4) * Output_depth_dim
                  + (m+8);
                filter_subscript3 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+4) * Output_depth_dim
                  + (m+16);
                filter_subscript4 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+4) * Output_depth_dim
                  + (m+24);
                
                s = _mm256_set1_ps(in_FP[in_subscript]);
                s2 = _mm256_set1_ps(in_FP[in_subscript2]);

                w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript2]);
                w3 = _mm256_load_ps(&filter_FP_copy[filter_subscript3]);
                w4 = _mm256_load_ps(&filter_FP_copy[filter_subscript4]);


                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s, w2, temp3);
                temp4 = _mm256_fmadd_ps(s2, w2, temp4);
                temp5 = _mm256_fmadd_ps(s, w3, temp5);
                temp6 = _mm256_fmadd_ps(s2, w3, temp6);
                temp7 = _mm256_fmadd_ps(s, w4, temp7);
                temp8 = _mm256_fmadd_ps(s2, w4, temp8);


                // d+5 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+5);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+5);

                filter_subscript = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+5) * Output_depth_dim
                  + m;
                filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+5) * Output_depth_dim
                  + (m+8);
                filter_subscript3 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+5) * Output_depth_dim
                  + (m+16);
                filter_subscript4 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+5) * Output_depth_dim
                  + (m+24);
                
                s = _mm256_set1_ps(in_FP[in_subscript]);
                s2 = _mm256_set1_ps(in_FP[in_subscript2]);

                w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript2]);
                w3 = _mm256_load_ps(&filter_FP_copy[filter_subscript3]);
                w4 = _mm256_load_ps(&filter_FP_copy[filter_subscript4]);


                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s, w2, temp3);
                temp4 = _mm256_fmadd_ps(s2, w2, temp4);
                temp5 = _mm256_fmadd_ps(s, w3, temp5);
                temp6 = _mm256_fmadd_ps(s2, w3, temp6);
                temp7 = _mm256_fmadd_ps(s, w4, temp7);
                temp8 = _mm256_fmadd_ps(s2, w4, temp8);


                // d+6 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+6);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+6);

                filter_subscript = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+6) * Output_depth_dim
                  + m;
                filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+6) * Output_depth_dim
                  + (m+8);
                filter_subscript3 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+6) * Output_depth_dim
                  + (m+16);
                filter_subscript4 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+6) * Output_depth_dim
                  + (m+24);
                
                s = _mm256_set1_ps(in_FP[in_subscript]);
                s2 = _mm256_set1_ps(in_FP[in_subscript2]);

                w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript2]);
                w3 = _mm256_load_ps(&filter_FP_copy[filter_subscript3]);
                w4 = _mm256_load_ps(&filter_FP_copy[filter_subscript4]);


                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s, w2, temp3);
                temp4 = _mm256_fmadd_ps(s2, w2, temp4);
                temp5 = _mm256_fmadd_ps(s, w3, temp5);
                temp6 = _mm256_fmadd_ps(s2, w3, temp6);
                temp7 = _mm256_fmadd_ps(s, w4, temp7);
                temp8 = _mm256_fmadd_ps(s2, w4, temp8);


                // d+7 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+7);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+7);

                filter_subscript = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+7) * Output_depth_dim
                  + m;
                filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+7) * Output_depth_dim
                  + (m+8);
                filter_subscript3 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+7) * Output_depth_dim
                  + (m+16);
                filter_subscript4 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+7) * Output_depth_dim
                  + (m+24);
                
                s = _mm256_set1_ps(in_FP[in_subscript]);
                s2 = _mm256_set1_ps(in_FP[in_subscript2]);

                w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript2]);
                w3 = _mm256_load_ps(&filter_FP_copy[filter_subscript3]);
                w4 = _mm256_load_ps(&filter_FP_copy[filter_subscript4]);


                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s, w2, temp3);
                temp4 = _mm256_fmadd_ps(s2, w2, temp4);
                temp5 = _mm256_fmadd_ps(s, w3, temp5);
                temp6 = _mm256_fmadd_ps(s2, w3, temp6);
                temp7 = _mm256_fmadd_ps(s, w4, temp7);
                temp8 = _mm256_fmadd_ps(s2, w4, temp8);
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
            x * Output_depth_dim
            + (m+8);
          unsigned long long int out_subscript4 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+8);
          unsigned long long int out_subscript5 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+16);
          unsigned long long int out_subscript6 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+16);
          unsigned long long int out_subscript7 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+24);
          unsigned long long int out_subscript8 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+24);

          
          temp = _mm256_add_ps(temp, bias);
          temp = _mm256_max_ps(temp, _mm256_setzero_ps()); // merge Relu layer via native AVX intrinsics
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript], temp);

          temp2 = _mm256_add_ps(temp2, bias);
          temp2 = _mm256_max_ps(temp2, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript2], temp2);
          
          temp3 = _mm256_add_ps(temp3, bias2);
          temp3 = _mm256_max_ps(temp3, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript3], temp3);
          
          temp4 = _mm256_add_ps(temp4, bias2);
          temp4 = _mm256_max_ps(temp4, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript4], temp4);
          
          temp5 = _mm256_add_ps(temp5, bias3);
          temp5 = _mm256_max_ps(temp5, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript5], temp5);
          
          temp6 = _mm256_add_ps(temp6, bias3);
          temp6 = _mm256_max_ps(temp6, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript6], temp6);
          
          temp7 = _mm256_add_ps(temp7, bias4);
          temp7 = _mm256_max_ps(temp7, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript7], temp7);
          
          temp8 = _mm256_add_ps(temp8, bias4);
          temp8 = _mm256_max_ps(temp8, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript8], temp8);
        }
      }
    }
  }
  printf("\n from AC optv3 m32 x2 - d8 %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  _mm_free(filter_FP_copy);
  return 0;
}


// unrolled d by factor of 2
// ~54 GFLOPS
int optimised_layer_v2_AC_x3m24_unroll_d2_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {

  __m256 bias, bias2, bias3, temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9;
  unsigned int m, x;

  int m_bound = (Output_depth_dim/ 24) * 24;
  int x_bound = (Output_X_dim/ 3) * 3; 

  unsigned int filter_FP_length = Output_depth_dim * Mask_Y_dim * Mask_X_dim * Input_depth_dim;
  float* filter_FP_copy = (float*)_mm_malloc(filter_FP_length * sizeof(float), 64);
  if (filter_FP_copy == NULL) {
    printf("\nerror with malloc allocating filter array copy");
    exit(EXIT_FAILURE);
  }

  // array copying - filter_FP into form usable for vectorising m loop
  unsigned long long int new_subscript = 0;
  for (unsigned int y = 0; y < Mask_Y_dim; y++) {
    for (unsigned int x = 0; x < Mask_X_dim; x++) {
      for (unsigned int d = 0; d < Input_depth_dim; d++) {
        for (unsigned int m = 0; m < Output_depth_dim; m++) {
          unsigned long long int old_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
            + y * Mask_X_dim * Input_depth_dim
            + x * Input_depth_dim
            + d;
              
          // unsigned long long int new_subscript = y * Mask_X_dim * Input_depth_dim * Output_depth_dim
          //   + x * Input_depth_dim * Output_depth_dim
          //   + d * Output_depth_dim
          //   + m;

          filter_FP_copy[new_subscript] = filter_FP[old_subscript];
          new_subscript++;
        }
      }
    }
  }

  // main loop body
  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (m = 0; m < m_bound; m+=24) { //channels
      bias = _mm256_load_ps(&bias_array_FP[m]);
      bias2 = _mm256_load_ps(&bias_array_FP[m+8]);
      bias3 = _mm256_load_ps(&bias_array_FP[m+16]);

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (x = 0; x < x_bound; x+=3) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          temp4 = _mm256_setzero_ps();
          temp5 = _mm256_setzero_ps();
          temp6 = _mm256_setzero_ps();
          temp7 = _mm256_setzero_ps();
          temp8 = _mm256_setzero_ps();
          temp9 = _mm256_setzero_ps();

          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=2) {

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

                unsigned long long int filter_subscript = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + m;
                unsigned long long int filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + (m+8);
                unsigned long long int filter_subscript3 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + (m+16);
                
                __m256 s = _mm256_set1_ps(in_FP[in_subscript]);
                __m256 s2 = _mm256_set1_ps(in_FP[in_subscript2]);
                __m256 s3 = _mm256_set1_ps(in_FP[in_subscript3]);

                __m256 w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                __m256 w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript2]);
                __m256 w3 = _mm256_load_ps(&filter_FP_copy[filter_subscript3]);


                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s3, w, temp3);
                temp4 = _mm256_fmadd_ps(s, w2, temp4);
                temp5 = _mm256_fmadd_ps(s2, w2, temp5);
                temp6 = _mm256_fmadd_ps(s3, w2, temp6);
                temp7 = _mm256_fmadd_ps(s, w3, temp7);
                temp8 = _mm256_fmadd_ps(s2, w3, temp8);
                temp9 = _mm256_fmadd_ps(s3, w3, temp9);


                // d+1 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+1);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+1);
                in_subscript3 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+2) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+1);

                filter_subscript = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+1) * Output_depth_dim
                  + m;
                filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+1) * Output_depth_dim
                  + (m+8);
                filter_subscript3 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+1) * Output_depth_dim
                  + (m+16);
                
                s = _mm256_set1_ps(in_FP[in_subscript]);
                s2 = _mm256_set1_ps(in_FP[in_subscript2]);
                s3 = _mm256_set1_ps(in_FP[in_subscript3]);

                w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript2]);
                w3 = _mm256_load_ps(&filter_FP_copy[filter_subscript3]);


                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s3, w, temp3);
                temp4 = _mm256_fmadd_ps(s, w2, temp4);
                temp5 = _mm256_fmadd_ps(s2, w2, temp5);
                temp6 = _mm256_fmadd_ps(s3, w2, temp6);
                temp7 = _mm256_fmadd_ps(s, w3, temp7);
                temp8 = _mm256_fmadd_ps(s2, w3, temp8);
                temp9 = _mm256_fmadd_ps(s3, w3, temp9);
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
            x * Output_depth_dim
            + (m+8);
          unsigned long long int out_subscript5 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+8);
          unsigned long long int out_subscript6 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+2) * Output_depth_dim
            + (m+8);
          unsigned long long int out_subscript7 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+16);
          unsigned long long int out_subscript8 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+16);
          unsigned long long int out_subscript9 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+2) * Output_depth_dim
            + (m+16);

          
          temp = _mm256_add_ps(temp, bias);
          temp = _mm256_max_ps(temp, _mm256_setzero_ps()); // merge Relu layer via native AVX intrinsics
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript], temp);

          temp2 = _mm256_add_ps(temp2, bias);
          temp2 = _mm256_max_ps(temp2, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript2], temp2);
          
          temp3 = _mm256_add_ps(temp3, bias);
          temp3 = _mm256_max_ps(temp3, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript3], temp3);
          
          temp4 = _mm256_add_ps(temp4, bias2);
          temp4 = _mm256_max_ps(temp4, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript4], temp4);
          
          temp5 = _mm256_add_ps(temp5, bias2);
          temp5 = _mm256_max_ps(temp5, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript5], temp5);
          
          temp6 = _mm256_add_ps(temp6, bias2);
          temp6 = _mm256_max_ps(temp6, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript6], temp6);
          
          temp7 = _mm256_add_ps(temp7, bias3);
          temp7 = _mm256_max_ps(temp7, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript7], temp7);
          
          temp8 = _mm256_add_ps(temp8, bias3);
          temp8 = _mm256_max_ps(temp8, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript8], temp8);
                    
          temp9 = _mm256_add_ps(temp9, bias3);
          temp9 = _mm256_max_ps(temp9, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript9], temp9);
        }
        // overflow/ fallback x loop
        for (; x < Output_X_dim; x++) {
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();

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
                unsigned long long int filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + (m+8);
                unsigned long long int filter_subscript3 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + (m+16);
                
                __m256 s = _mm256_set1_ps(in_FP[in_subscript]);

                __m256 w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                __m256 w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript2]);
                __m256 w3 = _mm256_load_ps(&filter_FP_copy[filter_subscript3]);


                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s, w2, temp2);
                temp3 = _mm256_fmadd_ps(s, w3, temp3);


                // d+1 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+1);

                filter_subscript = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+1) * Output_depth_dim
                  + m;
                filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+1) * Output_depth_dim
                  + (m+8);
                filter_subscript3 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+1) * Output_depth_dim
                  + (m+16);
                
                s = _mm256_set1_ps(in_FP[in_subscript]);

                w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript2]);
                w3 = _mm256_load_ps(&filter_FP_copy[filter_subscript3]);


                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s, w2, temp2);
                temp3 = _mm256_fmadd_ps(s, w3, temp3);
              }
            }
          }
          unsigned long long int out_subscript = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + m;
          unsigned long long int out_subscript2 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+8);
          unsigned long long int out_subscript3 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+16);

          temp = _mm256_add_ps(temp, bias);
          temp = _mm256_max_ps(temp, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript], temp);

          temp2 = _mm256_add_ps(temp2, bias2);
          temp2 = _mm256_max_ps(temp2, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript2], temp2);
          
          temp3 = _mm256_add_ps(temp3, bias3);
          temp3 = _mm256_max_ps(temp3, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript3], temp3);

        }
      }
    }
    for (; m < Output_depth_dim; m+=8) {
      bias = _mm256_load_ps(&bias_array_FP[m]);

      for (unsigned int y = 0; y < Output_Y_dim; y++) {
        for (x = 0; x < Output_X_dim; x+=4) {
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          temp4 = _mm256_setzero_ps();

          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=2) {

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

                unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;

                __m256 s = _mm256_loadu_ps(&in_FP[in_subscript]);
                __m256 s2 = _mm256_loadu_ps(&in_FP[in_subscript2]);
                __m256 s3 = _mm256_loadu_ps(&in_FP[in_subscript3]);
                __m256 s4 = _mm256_loadu_ps(&in_FP[in_subscript4]);

                __m256 w = _mm256_loadu_ps(&filter_FP_copy[filter_subscript]);

                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s3, w, temp3);
                temp4 = _mm256_fmadd_ps(s4, w, temp4);
                

                // d+1 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+1);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+1);
                in_subscript3 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+2) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+1);
                in_subscript4 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+3) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+1);

                filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+1);

                s = _mm256_loadu_ps(&in_FP[in_subscript]);
                s2 = _mm256_loadu_ps(&in_FP[in_subscript2]);
                s3 = _mm256_loadu_ps(&in_FP[in_subscript3]);
                s4 = _mm256_loadu_ps(&in_FP[in_subscript4]);

                w = _mm256_loadu_ps(&filter_FP_copy[filter_subscript]);

                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s3, w, temp3);
                temp4 = _mm256_fmadd_ps(s4, w, temp4);
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
        }
      }
    }
  }
  printf("\n from AC optv3 m24 x3 - d2 %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  _mm_free(filter_FP_copy);
  return 0;
}


// unrolled d by factor of 4
// ~63 GFLOPS
int optimised_layer_v2_AC_x3m24_unroll_d4_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {

  __m256 bias, bias2, bias3, temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9;
  unsigned int m, x;

  int m_bound = (Output_depth_dim/ 24) * 24;
  int x_bound = (Output_X_dim/ 3) * 3; 

  unsigned int filter_FP_length = Output_depth_dim * Mask_Y_dim * Mask_X_dim * Input_depth_dim;
  float* filter_FP_copy = (float*)_mm_malloc(filter_FP_length * sizeof(float), 64);
  if (filter_FP_copy == NULL) {
    printf("\nerror with malloc allocating filter array copy");
    exit(EXIT_FAILURE);
  }

  // array copying - filter_FP into form usable for vectorising m loop
  unsigned long long int new_subscript = 0;
  for (unsigned int y = 0; y < Mask_Y_dim; y++) {
    for (unsigned int x = 0; x < Mask_X_dim; x++) {
      for (unsigned int d = 0; d < Input_depth_dim; d++) {
        for (unsigned int m = 0; m < Output_depth_dim; m++) {
          unsigned long long int old_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
            + y * Mask_X_dim * Input_depth_dim
            + x * Input_depth_dim
            + d;
              
          // unsigned long long int new_subscript = y * Mask_X_dim * Input_depth_dim * Output_depth_dim
          //   + x * Input_depth_dim * Output_depth_dim
          //   + d * Output_depth_dim
          //   + m;

          filter_FP_copy[new_subscript] = filter_FP[old_subscript];
          new_subscript++;
        }
      }
    }
  }

  // main loop body
  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (m = 0; m < m_bound; m+=24) { //channels
      bias = _mm256_load_ps(&bias_array_FP[m]);
      bias2 = _mm256_load_ps(&bias_array_FP[m+8]);
      bias3 = _mm256_load_ps(&bias_array_FP[m+16]);

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (x = 0; x < x_bound; x+=3) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          temp4 = _mm256_setzero_ps();
          temp5 = _mm256_setzero_ps();
          temp6 = _mm256_setzero_ps();
          temp7 = _mm256_setzero_ps();
          temp8 = _mm256_setzero_ps();
          temp9 = _mm256_setzero_ps();

          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=4) {

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

                unsigned long long int filter_subscript = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + m;
                unsigned long long int filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + (m+8);
                unsigned long long int filter_subscript3 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + (m+16);
                
                __m256 s = _mm256_set1_ps(in_FP[in_subscript]);
                __m256 s2 = _mm256_set1_ps(in_FP[in_subscript2]);
                __m256 s3 = _mm256_set1_ps(in_FP[in_subscript3]);

                __m256 w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                __m256 w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript2]);
                __m256 w3 = _mm256_load_ps(&filter_FP_copy[filter_subscript3]);


                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s3, w, temp3);
                temp4 = _mm256_fmadd_ps(s, w2, temp4);
                temp5 = _mm256_fmadd_ps(s2, w2, temp5);
                temp6 = _mm256_fmadd_ps(s3, w2, temp6);
                temp7 = _mm256_fmadd_ps(s, w3, temp7);
                temp8 = _mm256_fmadd_ps(s2, w3, temp8);
                temp9 = _mm256_fmadd_ps(s3, w3, temp9);


                // d+1 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+1);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+1);
                in_subscript3 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+2) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+1);

                filter_subscript = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+1) * Output_depth_dim
                  + m;
                filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+1) * Output_depth_dim
                  + (m+8);
                filter_subscript3 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+1) * Output_depth_dim
                  + (m+16);
                
                s = _mm256_set1_ps(in_FP[in_subscript]);
                s2 = _mm256_set1_ps(in_FP[in_subscript2]);
                s3 = _mm256_set1_ps(in_FP[in_subscript3]);

                w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript2]);
                w3 = _mm256_load_ps(&filter_FP_copy[filter_subscript3]);


                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s3, w, temp3);
                temp4 = _mm256_fmadd_ps(s, w2, temp4);
                temp5 = _mm256_fmadd_ps(s2, w2, temp5);
                temp6 = _mm256_fmadd_ps(s3, w2, temp6);
                temp7 = _mm256_fmadd_ps(s, w3, temp7);
                temp8 = _mm256_fmadd_ps(s2, w3, temp8);
                temp9 = _mm256_fmadd_ps(s3, w3, temp9);


                // d+2 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+2);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+2);
                in_subscript3 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+2) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+2);

                filter_subscript = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+2) * Output_depth_dim
                  + m;
                filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+2) * Output_depth_dim
                  + (m+8);
                filter_subscript3 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+2) * Output_depth_dim
                  + (m+16);
                
                s = _mm256_set1_ps(in_FP[in_subscript]);
                s2 = _mm256_set1_ps(in_FP[in_subscript2]);
                s3 = _mm256_set1_ps(in_FP[in_subscript3]);

                w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript2]);
                w3 = _mm256_load_ps(&filter_FP_copy[filter_subscript3]);


                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s3, w, temp3);
                temp4 = _mm256_fmadd_ps(s, w2, temp4);
                temp5 = _mm256_fmadd_ps(s2, w2, temp5);
                temp6 = _mm256_fmadd_ps(s3, w2, temp6);
                temp7 = _mm256_fmadd_ps(s, w3, temp7);
                temp8 = _mm256_fmadd_ps(s2, w3, temp8);
                temp9 = _mm256_fmadd_ps(s3, w3, temp9);


                // d+3 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+3);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+3);
                in_subscript3 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+2) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+3);

                filter_subscript = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+3) * Output_depth_dim
                  + m;
                filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+3) * Output_depth_dim
                  + (m+8);
                filter_subscript3 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+3) * Output_depth_dim
                  + (m+16);
                
                s = _mm256_set1_ps(in_FP[in_subscript]);
                s2 = _mm256_set1_ps(in_FP[in_subscript2]);
                s3 = _mm256_set1_ps(in_FP[in_subscript3]);

                w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript2]);
                w3 = _mm256_load_ps(&filter_FP_copy[filter_subscript3]);


                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s3, w, temp3);
                temp4 = _mm256_fmadd_ps(s, w2, temp4);
                temp5 = _mm256_fmadd_ps(s2, w2, temp5);
                temp6 = _mm256_fmadd_ps(s3, w2, temp6);
                temp7 = _mm256_fmadd_ps(s, w3, temp7);
                temp8 = _mm256_fmadd_ps(s2, w3, temp8);
                temp9 = _mm256_fmadd_ps(s3, w3, temp9);
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
            x * Output_depth_dim
            + (m+8);
          unsigned long long int out_subscript5 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+8);
          unsigned long long int out_subscript6 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+2) * Output_depth_dim
            + (m+8);
          unsigned long long int out_subscript7 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+16);
          unsigned long long int out_subscript8 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+16);
          unsigned long long int out_subscript9 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+2) * Output_depth_dim
            + (m+16);

          
          temp = _mm256_add_ps(temp, bias);
          temp = _mm256_max_ps(temp, _mm256_setzero_ps()); // merge Relu layer via native AVX intrinsics
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript], temp);

          temp2 = _mm256_add_ps(temp2, bias);
          temp2 = _mm256_max_ps(temp2, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript2], temp2);
          
          temp3 = _mm256_add_ps(temp3, bias);
          temp3 = _mm256_max_ps(temp3, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript3], temp3);
          
          temp4 = _mm256_add_ps(temp4, bias2);
          temp4 = _mm256_max_ps(temp4, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript4], temp4);
          
          temp5 = _mm256_add_ps(temp5, bias2);
          temp5 = _mm256_max_ps(temp5, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript5], temp5);
          
          temp6 = _mm256_add_ps(temp6, bias2);
          temp6 = _mm256_max_ps(temp6, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript6], temp6);
          
          temp7 = _mm256_add_ps(temp7, bias3);
          temp7 = _mm256_max_ps(temp7, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript7], temp7);
          
          temp8 = _mm256_add_ps(temp8, bias3);
          temp8 = _mm256_max_ps(temp8, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript8], temp8);
                    
          temp9 = _mm256_add_ps(temp9, bias3);
          temp9 = _mm256_max_ps(temp9, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript9], temp9);
        }
        // overflow/ fallback x loop
        for (; x < Output_X_dim; x++) {
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();

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
                unsigned long long int filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + (m+8);
                unsigned long long int filter_subscript3 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + (m+16);
                
                __m256 s = _mm256_set1_ps(in_FP[in_subscript]);

                __m256 w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                __m256 w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript2]);
                __m256 w3 = _mm256_load_ps(&filter_FP_copy[filter_subscript3]);


                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s, w2, temp2);
                temp3 = _mm256_fmadd_ps(s, w3, temp3);


                // d+1 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+1);

                filter_subscript = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+1) * Output_depth_dim
                  + m;
                filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+1) * Output_depth_dim
                  + (m+8);
                filter_subscript3 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+1) * Output_depth_dim
                  + (m+16);
                
                s = _mm256_set1_ps(in_FP[in_subscript]);

                w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript2]);
                w3 = _mm256_load_ps(&filter_FP_copy[filter_subscript3]);


                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s, w2, temp2);
                temp3 = _mm256_fmadd_ps(s, w3, temp3);


                // d+2 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+2);

                filter_subscript = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+2) * Output_depth_dim
                  + m;
                filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+2) * Output_depth_dim
                  + (m+8);
                filter_subscript3 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+2) * Output_depth_dim
                  + (m+16);
                
                s = _mm256_set1_ps(in_FP[in_subscript]);

                w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript2]);
                w3 = _mm256_load_ps(&filter_FP_copy[filter_subscript3]);


                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s, w2, temp2);
                temp3 = _mm256_fmadd_ps(s, w3, temp3);


                // d+3 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+3);

                filter_subscript = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+3) * Output_depth_dim
                  + m;
                filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+3) * Output_depth_dim
                  + (m+8);
                filter_subscript3 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+3) * Output_depth_dim
                  + (m+16);
                
                s = _mm256_set1_ps(in_FP[in_subscript]);

                w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript2]);
                w3 = _mm256_load_ps(&filter_FP_copy[filter_subscript3]);


                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s, w2, temp2);
                temp3 = _mm256_fmadd_ps(s, w3, temp3);
              }
            }
          }
          unsigned long long int out_subscript = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + m;
          unsigned long long int out_subscript2 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+8);
          unsigned long long int out_subscript3 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+16);

          temp = _mm256_add_ps(temp, bias);
          temp = _mm256_max_ps(temp, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript], temp);

          temp2 = _mm256_add_ps(temp2, bias2);
          temp2 = _mm256_max_ps(temp2, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript2], temp2);
          
          temp3 = _mm256_add_ps(temp3, bias3);
          temp3 = _mm256_max_ps(temp3, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript3], temp3);

        }
      }
    }
    for (; m < Output_depth_dim; m+=8) {
      bias = _mm256_load_ps(&bias_array_FP[m]);

      for (unsigned int y = 0; y < Output_Y_dim; y++) {
        for (x = 0; x < Output_X_dim; x+=4) {
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          temp4 = _mm256_setzero_ps();

          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=4) {

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

                unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;

                __m256 s = _mm256_loadu_ps(&in_FP[in_subscript]);
                __m256 s2 = _mm256_loadu_ps(&in_FP[in_subscript2]);
                __m256 s3 = _mm256_loadu_ps(&in_FP[in_subscript3]);
                __m256 s4 = _mm256_loadu_ps(&in_FP[in_subscript4]);

                __m256 w = _mm256_loadu_ps(&filter_FP_copy[filter_subscript]);

                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s3, w, temp3);
                temp4 = _mm256_fmadd_ps(s4, w, temp4);
                

                // d+1 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+1);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+1);
                in_subscript3 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+2) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+1);
                in_subscript4 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+3) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+1);

                filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+1);

                s = _mm256_loadu_ps(&in_FP[in_subscript]);
                s2 = _mm256_loadu_ps(&in_FP[in_subscript2]);
                s3 = _mm256_loadu_ps(&in_FP[in_subscript3]);
                s4 = _mm256_loadu_ps(&in_FP[in_subscript4]);

                w = _mm256_loadu_ps(&filter_FP_copy[filter_subscript]);

                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s3, w, temp3);
                temp4 = _mm256_fmadd_ps(s4, w, temp4);
                

                // d+2 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+2);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+2);
                in_subscript3 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+2) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+2);
                in_subscript4 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+3) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+2);

                filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+2);

                s = _mm256_loadu_ps(&in_FP[in_subscript]);
                s2 = _mm256_loadu_ps(&in_FP[in_subscript2]);
                s3 = _mm256_loadu_ps(&in_FP[in_subscript3]);
                s4 = _mm256_loadu_ps(&in_FP[in_subscript4]);

                w = _mm256_loadu_ps(&filter_FP_copy[filter_subscript]);

                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s3, w, temp3);
                temp4 = _mm256_fmadd_ps(s4, w, temp4);
                

                // d+3 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+3);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+3);
                in_subscript3 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+2) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+3);
                in_subscript4 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+3) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+3);

                filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+3);

                s = _mm256_loadu_ps(&in_FP[in_subscript]);
                s2 = _mm256_loadu_ps(&in_FP[in_subscript2]);
                s3 = _mm256_loadu_ps(&in_FP[in_subscript3]);
                s4 = _mm256_loadu_ps(&in_FP[in_subscript4]);

                w = _mm256_loadu_ps(&filter_FP_copy[filter_subscript]);

                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s3, w, temp3);
                temp4 = _mm256_fmadd_ps(s4, w, temp4);
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
        }
      }
    }
  }
  printf("\n from AC optv3 m24 x3 - d4 %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  _mm_free(filter_FP_copy);
  return 0;
}


// unrolled d by factor of 8
// ~67 GFLOPS
int optimised_layer_v2_AC_x3m24_unroll_d8_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {

  __m256 bias, bias2, bias3, temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9;
  unsigned int m, x;

  int m_bound = (Output_depth_dim/ 24) * 24;
  int x_bound = (Output_X_dim/ 3) * 3; 

  unsigned int filter_FP_length = Output_depth_dim * Mask_Y_dim * Mask_X_dim * Input_depth_dim;
  float* filter_FP_copy = (float*)_mm_malloc(filter_FP_length * sizeof(float), 64);
  if (filter_FP_copy == NULL) {
    printf("\nerror with malloc allocating filter array copy");
    exit(EXIT_FAILURE);
  }

  // array copying - filter_FP into form usable for vectorising m loop
  unsigned long long int new_subscript = 0;
  for (unsigned int y = 0; y < Mask_Y_dim; y++) {
    for (unsigned int x = 0; x < Mask_X_dim; x++) {
      for (unsigned int d = 0; d < Input_depth_dim; d++) {
        for (unsigned int m = 0; m < Output_depth_dim; m++) {
          unsigned long long int old_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
            + y * Mask_X_dim * Input_depth_dim
            + x * Input_depth_dim
            + d;
              
          // unsigned long long int new_subscript = y * Mask_X_dim * Input_depth_dim * Output_depth_dim
          //   + x * Input_depth_dim * Output_depth_dim
          //   + d * Output_depth_dim
          //   + m;

          filter_FP_copy[new_subscript] = filter_FP[old_subscript];
          new_subscript++;
        }
      }
    }
  }

  // main loop body
  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (m = 0; m < m_bound; m+=24) { //channels
      bias = _mm256_load_ps(&bias_array_FP[m]);
      bias2 = _mm256_load_ps(&bias_array_FP[m+8]);
      bias3 = _mm256_load_ps(&bias_array_FP[m+16]);

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (x = 0; x < x_bound; x+=3) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          temp4 = _mm256_setzero_ps();
          temp5 = _mm256_setzero_ps();
          temp6 = _mm256_setzero_ps();
          temp7 = _mm256_setzero_ps();
          temp8 = _mm256_setzero_ps();
          temp9 = _mm256_setzero_ps();

          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

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

                unsigned long long int filter_subscript = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + m;
                unsigned long long int filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + (m+8);
                unsigned long long int filter_subscript3 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + (m+16);
                
                __m256 s = _mm256_set1_ps(in_FP[in_subscript]);
                __m256 s2 = _mm256_set1_ps(in_FP[in_subscript2]);
                __m256 s3 = _mm256_set1_ps(in_FP[in_subscript3]);

                __m256 w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                __m256 w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript2]);
                __m256 w3 = _mm256_load_ps(&filter_FP_copy[filter_subscript3]);


                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s3, w, temp3);
                temp4 = _mm256_fmadd_ps(s, w2, temp4);
                temp5 = _mm256_fmadd_ps(s2, w2, temp5);
                temp6 = _mm256_fmadd_ps(s3, w2, temp6);
                temp7 = _mm256_fmadd_ps(s, w3, temp7);
                temp8 = _mm256_fmadd_ps(s2, w3, temp8);
                temp9 = _mm256_fmadd_ps(s3, w3, temp9);


                // d+1 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+1);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+1);
                in_subscript3 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+2) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+1);

                filter_subscript = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+1) * Output_depth_dim
                  + m;
                filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+1) * Output_depth_dim
                  + (m+8);
                filter_subscript3 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+1) * Output_depth_dim
                  + (m+16);
                
                s = _mm256_set1_ps(in_FP[in_subscript]);
                s2 = _mm256_set1_ps(in_FP[in_subscript2]);
                s3 = _mm256_set1_ps(in_FP[in_subscript3]);

                w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript2]);
                w3 = _mm256_load_ps(&filter_FP_copy[filter_subscript3]);


                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s3, w, temp3);
                temp4 = _mm256_fmadd_ps(s, w2, temp4);
                temp5 = _mm256_fmadd_ps(s2, w2, temp5);
                temp6 = _mm256_fmadd_ps(s3, w2, temp6);
                temp7 = _mm256_fmadd_ps(s, w3, temp7);
                temp8 = _mm256_fmadd_ps(s2, w3, temp8);
                temp9 = _mm256_fmadd_ps(s3, w3, temp9);


                // d+2 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+2);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+2);
                in_subscript3 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+2) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+2);

                filter_subscript = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+2) * Output_depth_dim
                  + m;
                filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+2) * Output_depth_dim
                  + (m+8);
                filter_subscript3 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+2) * Output_depth_dim
                  + (m+16);
                
                s = _mm256_set1_ps(in_FP[in_subscript]);
                s2 = _mm256_set1_ps(in_FP[in_subscript2]);
                s3 = _mm256_set1_ps(in_FP[in_subscript3]);

                w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript2]);
                w3 = _mm256_load_ps(&filter_FP_copy[filter_subscript3]);


                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s3, w, temp3);
                temp4 = _mm256_fmadd_ps(s, w2, temp4);
                temp5 = _mm256_fmadd_ps(s2, w2, temp5);
                temp6 = _mm256_fmadd_ps(s3, w2, temp6);
                temp7 = _mm256_fmadd_ps(s, w3, temp7);
                temp8 = _mm256_fmadd_ps(s2, w3, temp8);
                temp9 = _mm256_fmadd_ps(s3, w3, temp9);


                // d+3 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+3);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+3);
                in_subscript3 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+2) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+3);

                filter_subscript = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+3) * Output_depth_dim
                  + m;
                filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+3) * Output_depth_dim
                  + (m+8);
                filter_subscript3 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+3) * Output_depth_dim
                  + (m+16);
                
                s = _mm256_set1_ps(in_FP[in_subscript]);
                s2 = _mm256_set1_ps(in_FP[in_subscript2]);
                s3 = _mm256_set1_ps(in_FP[in_subscript3]);

                w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript2]);
                w3 = _mm256_load_ps(&filter_FP_copy[filter_subscript3]);


                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s3, w, temp3);
                temp4 = _mm256_fmadd_ps(s, w2, temp4);
                temp5 = _mm256_fmadd_ps(s2, w2, temp5);
                temp6 = _mm256_fmadd_ps(s3, w2, temp6);
                temp7 = _mm256_fmadd_ps(s, w3, temp7);
                temp8 = _mm256_fmadd_ps(s2, w3, temp8);
                temp9 = _mm256_fmadd_ps(s3, w3, temp9);


                // d+4 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+4);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+4);
                in_subscript3 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+2) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+4);

                filter_subscript = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+4) * Output_depth_dim
                  + m;
                filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+4) * Output_depth_dim
                  + (m+8);
                filter_subscript3 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+4) * Output_depth_dim
                  + (m+16);
                
                s = _mm256_set1_ps(in_FP[in_subscript]);
                s2 = _mm256_set1_ps(in_FP[in_subscript2]);
                s3 = _mm256_set1_ps(in_FP[in_subscript3]);

                w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript2]);
                w3 = _mm256_load_ps(&filter_FP_copy[filter_subscript3]);


                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s3, w, temp3);
                temp4 = _mm256_fmadd_ps(s, w2, temp4);
                temp5 = _mm256_fmadd_ps(s2, w2, temp5);
                temp6 = _mm256_fmadd_ps(s3, w2, temp6);
                temp7 = _mm256_fmadd_ps(s, w3, temp7);
                temp8 = _mm256_fmadd_ps(s2, w3, temp8);
                temp9 = _mm256_fmadd_ps(s3, w3, temp9);


                // d+5 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+5);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+5);
                in_subscript3 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+2) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+5);

                filter_subscript = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+5) * Output_depth_dim
                  + m;
                filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+5) * Output_depth_dim
                  + (m+8);
                filter_subscript3 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+5) * Output_depth_dim
                  + (m+16);
                
                s = _mm256_set1_ps(in_FP[in_subscript]);
                s2 = _mm256_set1_ps(in_FP[in_subscript2]);
                s3 = _mm256_set1_ps(in_FP[in_subscript3]);

                w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript2]);
                w3 = _mm256_load_ps(&filter_FP_copy[filter_subscript3]);


                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s3, w, temp3);
                temp4 = _mm256_fmadd_ps(s, w2, temp4);
                temp5 = _mm256_fmadd_ps(s2, w2, temp5);
                temp6 = _mm256_fmadd_ps(s3, w2, temp6);
                temp7 = _mm256_fmadd_ps(s, w3, temp7);
                temp8 = _mm256_fmadd_ps(s2, w3, temp8);
                temp9 = _mm256_fmadd_ps(s3, w3, temp9);


                // d+6 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+6);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+6);
                in_subscript3 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+2) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+6);

                filter_subscript = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+6) * Output_depth_dim
                  + m;
                filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+6) * Output_depth_dim
                  + (m+8);
                filter_subscript3 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+6) * Output_depth_dim
                  + (m+16);
                
                s = _mm256_set1_ps(in_FP[in_subscript]);
                s2 = _mm256_set1_ps(in_FP[in_subscript2]);
                s3 = _mm256_set1_ps(in_FP[in_subscript3]);

                w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript2]);
                w3 = _mm256_load_ps(&filter_FP_copy[filter_subscript3]);


                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s3, w, temp3);
                temp4 = _mm256_fmadd_ps(s, w2, temp4);
                temp5 = _mm256_fmadd_ps(s2, w2, temp5);
                temp6 = _mm256_fmadd_ps(s3, w2, temp6);
                temp7 = _mm256_fmadd_ps(s, w3, temp7);
                temp8 = _mm256_fmadd_ps(s2, w3, temp8);
                temp9 = _mm256_fmadd_ps(s3, w3, temp9);


                // d+7 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+7);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+7);
                in_subscript3 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+2) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+7);

                filter_subscript = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+7) * Output_depth_dim
                  + m;
                filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+7) * Output_depth_dim
                  + (m+8);
                filter_subscript3 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+7) * Output_depth_dim
                  + (m+16);
                
                s = _mm256_set1_ps(in_FP[in_subscript]);
                s2 = _mm256_set1_ps(in_FP[in_subscript2]);
                s3 = _mm256_set1_ps(in_FP[in_subscript3]);

                w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript2]);
                w3 = _mm256_load_ps(&filter_FP_copy[filter_subscript3]);


                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s3, w, temp3);
                temp4 = _mm256_fmadd_ps(s, w2, temp4);
                temp5 = _mm256_fmadd_ps(s2, w2, temp5);
                temp6 = _mm256_fmadd_ps(s3, w2, temp6);
                temp7 = _mm256_fmadd_ps(s, w3, temp7);
                temp8 = _mm256_fmadd_ps(s2, w3, temp8);
                temp9 = _mm256_fmadd_ps(s3, w3, temp9);
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
            x * Output_depth_dim
            + (m+8);
          unsigned long long int out_subscript5 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+8);
          unsigned long long int out_subscript6 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+2) * Output_depth_dim
            + (m+8);
          unsigned long long int out_subscript7 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+16);
          unsigned long long int out_subscript8 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+16);
          unsigned long long int out_subscript9 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+2) * Output_depth_dim
            + (m+16);

          
          temp = _mm256_add_ps(temp, bias);
          temp = _mm256_max_ps(temp, _mm256_setzero_ps()); // merge Relu layer via native AVX intrinsics
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript], temp);

          temp2 = _mm256_add_ps(temp2, bias);
          temp2 = _mm256_max_ps(temp2, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript2], temp2);
          
          temp3 = _mm256_add_ps(temp3, bias);
          temp3 = _mm256_max_ps(temp3, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript3], temp3);
          
          temp4 = _mm256_add_ps(temp4, bias2);
          temp4 = _mm256_max_ps(temp4, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript4], temp4);
          
          temp5 = _mm256_add_ps(temp5, bias2);
          temp5 = _mm256_max_ps(temp5, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript5], temp5);
          
          temp6 = _mm256_add_ps(temp6, bias2);
          temp6 = _mm256_max_ps(temp6, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript6], temp6);
          
          temp7 = _mm256_add_ps(temp7, bias3);
          temp7 = _mm256_max_ps(temp7, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript7], temp7);
          
          temp8 = _mm256_add_ps(temp8, bias3);
          temp8 = _mm256_max_ps(temp8, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript8], temp8);
                    
          temp9 = _mm256_add_ps(temp9, bias3);
          temp9 = _mm256_max_ps(temp9, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript9], temp9);
        }
        // overflow/ fallback x loop
        for (; x < Output_X_dim; x++) {
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();

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
                unsigned long long int filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + (m+8);
                unsigned long long int filter_subscript3 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + d * Output_depth_dim
                  + (m+16);
                
                __m256 s = _mm256_set1_ps(in_FP[in_subscript]);

                __m256 w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                __m256 w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript2]);
                __m256 w3 = _mm256_load_ps(&filter_FP_copy[filter_subscript3]);


                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s, w2, temp2);
                temp3 = _mm256_fmadd_ps(s, w3, temp3);


                // d+1 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+1);

                filter_subscript = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+1) * Output_depth_dim
                  + m;
                filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+1) * Output_depth_dim
                  + (m+8);
                filter_subscript3 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+1) * Output_depth_dim
                  + (m+16);
                
                s = _mm256_set1_ps(in_FP[in_subscript]);

                w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript2]);
                w3 = _mm256_load_ps(&filter_FP_copy[filter_subscript3]);


                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s, w2, temp2);
                temp3 = _mm256_fmadd_ps(s, w3, temp3);


                // d+2 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+2);

                filter_subscript = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+2) * Output_depth_dim
                  + m;
                filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+2) * Output_depth_dim
                  + (m+8);
                filter_subscript3 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+2) * Output_depth_dim
                  + (m+16);
                
                s = _mm256_set1_ps(in_FP[in_subscript]);

                w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript2]);
                w3 = _mm256_load_ps(&filter_FP_copy[filter_subscript3]);


                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s, w2, temp2);
                temp3 = _mm256_fmadd_ps(s, w3, temp3);


                // d+3 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+3);

                filter_subscript = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+3) * Output_depth_dim
                  + m;
                filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+3) * Output_depth_dim
                  + (m+8);
                filter_subscript3 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+3) * Output_depth_dim
                  + (m+16);
                
                s = _mm256_set1_ps(in_FP[in_subscript]);

                w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript2]);
                w3 = _mm256_load_ps(&filter_FP_copy[filter_subscript3]);


                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s, w2, temp2);
                temp3 = _mm256_fmadd_ps(s, w3, temp3);


                // d+4 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+4);

                filter_subscript = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+4) * Output_depth_dim
                  + m;
                filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+4) * Output_depth_dim
                  + (m+8);
                filter_subscript3 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+4) * Output_depth_dim
                  + (m+16);
                
                s = _mm256_set1_ps(in_FP[in_subscript]);

                w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript2]);
                w3 = _mm256_load_ps(&filter_FP_copy[filter_subscript3]);


                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s, w2, temp2);
                temp3 = _mm256_fmadd_ps(s, w3, temp3);


                // d+5 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+5);

                filter_subscript = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+5) * Output_depth_dim
                  + m;
                filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+5) * Output_depth_dim
                  + (m+8);
                filter_subscript3 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+5) * Output_depth_dim
                  + (m+16);
                
                s = _mm256_set1_ps(in_FP[in_subscript]);

                w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript2]);
                w3 = _mm256_load_ps(&filter_FP_copy[filter_subscript3]);


                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s, w2, temp2);
                temp3 = _mm256_fmadd_ps(s, w3, temp3);


                // d+6 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+6);

                filter_subscript = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+6) * Output_depth_dim
                  + m;
                filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+6) * Output_depth_dim
                  + (m+8);
                filter_subscript3 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+6) * Output_depth_dim
                  + (m+16);
                
                s = _mm256_set1_ps(in_FP[in_subscript]);

                w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript2]);
                w3 = _mm256_load_ps(&filter_FP_copy[filter_subscript3]);


                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s, w2, temp2);
                temp3 = _mm256_fmadd_ps(s, w3, temp3);


                // d+7 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+7);

                filter_subscript = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+7) * Output_depth_dim
                  + m;
                filter_subscript2 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+7) * Output_depth_dim
                  + (m+8);
                filter_subscript3 = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                  + off_x * Input_depth_dim * Output_depth_dim
                  + (d+7) * Output_depth_dim
                  + (m+16);
                
                s = _mm256_set1_ps(in_FP[in_subscript]);

                w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript2]);
                w3 = _mm256_load_ps(&filter_FP_copy[filter_subscript3]);


                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s, w2, temp2);
                temp3 = _mm256_fmadd_ps(s, w3, temp3);
              }
            }
          }
          unsigned long long int out_subscript = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + m;
          unsigned long long int out_subscript2 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+8);
          unsigned long long int out_subscript3 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+16);

          temp = _mm256_add_ps(temp, bias);
          temp = _mm256_max_ps(temp, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript], temp);

          temp2 = _mm256_add_ps(temp2, bias2);
          temp2 = _mm256_max_ps(temp2, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript2], temp2);
          
          temp3 = _mm256_add_ps(temp3, bias3);
          temp3 = _mm256_max_ps(temp3, _mm256_setzero_ps());
          _mm256_store_ps(&out_to_compare_with_FP[out_subscript3], temp3);

        }
      }
    }
    for (; m < Output_depth_dim; m+=8) {
      bias = _mm256_load_ps(&bias_array_FP[m]);

      for (unsigned int y = 0; y < Output_Y_dim; y++) {
        for (x = 0; x < Output_X_dim; x+=4) {
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          temp4 = _mm256_setzero_ps();

          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

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

                unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;

                __m256 s = _mm256_loadu_ps(&in_FP[in_subscript]);
                __m256 s2 = _mm256_loadu_ps(&in_FP[in_subscript2]);
                __m256 s3 = _mm256_loadu_ps(&in_FP[in_subscript3]);
                __m256 s4 = _mm256_loadu_ps(&in_FP[in_subscript4]);

                __m256 w = _mm256_loadu_ps(&filter_FP_copy[filter_subscript]);

                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s3, w, temp3);
                temp4 = _mm256_fmadd_ps(s4, w, temp4);
                

                // d+1 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+1);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+1);
                in_subscript3 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+2) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+1);
                in_subscript4 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+3) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+1);

                filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+1);

                s = _mm256_loadu_ps(&in_FP[in_subscript]);
                s2 = _mm256_loadu_ps(&in_FP[in_subscript2]);
                s3 = _mm256_loadu_ps(&in_FP[in_subscript3]);
                s4 = _mm256_loadu_ps(&in_FP[in_subscript4]);

                w = _mm256_loadu_ps(&filter_FP_copy[filter_subscript]);

                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s3, w, temp3);
                temp4 = _mm256_fmadd_ps(s4, w, temp4);
                

                // d+2 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+2);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+2);
                in_subscript3 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+2) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+2);
                in_subscript4 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+3) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+2);

                filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+2);

                s = _mm256_loadu_ps(&in_FP[in_subscript]);
                s2 = _mm256_loadu_ps(&in_FP[in_subscript2]);
                s3 = _mm256_loadu_ps(&in_FP[in_subscript3]);
                s4 = _mm256_loadu_ps(&in_FP[in_subscript4]);

                w = _mm256_loadu_ps(&filter_FP_copy[filter_subscript]);

                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s3, w, temp3);
                temp4 = _mm256_fmadd_ps(s4, w, temp4);
                

                // d+3 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+3);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+3);
                in_subscript3 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+2) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+3);
                in_subscript4 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+3) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+3);

                filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+3);

                s = _mm256_loadu_ps(&in_FP[in_subscript]);
                s2 = _mm256_loadu_ps(&in_FP[in_subscript2]);
                s3 = _mm256_loadu_ps(&in_FP[in_subscript3]);
                s4 = _mm256_loadu_ps(&in_FP[in_subscript4]);

                w = _mm256_loadu_ps(&filter_FP_copy[filter_subscript]);

                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s3, w, temp3);
                temp4 = _mm256_fmadd_ps(s4, w, temp4);
                

                // d+4 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+4);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+4);
                in_subscript3 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+2) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+4);
                in_subscript4 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+3) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+4);

                filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+4);

                s = _mm256_loadu_ps(&in_FP[in_subscript]);
                s2 = _mm256_loadu_ps(&in_FP[in_subscript2]);
                s3 = _mm256_loadu_ps(&in_FP[in_subscript3]);
                s4 = _mm256_loadu_ps(&in_FP[in_subscript4]);

                w = _mm256_loadu_ps(&filter_FP_copy[filter_subscript]);

                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s3, w, temp3);
                temp4 = _mm256_fmadd_ps(s4, w, temp4);
                

                // d+5 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+5);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+5);
                in_subscript3 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+2) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+5);
                in_subscript4 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+3) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+5);

                filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+5);

                s = _mm256_loadu_ps(&in_FP[in_subscript]);
                s2 = _mm256_loadu_ps(&in_FP[in_subscript2]);
                s3 = _mm256_loadu_ps(&in_FP[in_subscript3]);
                s4 = _mm256_loadu_ps(&in_FP[in_subscript4]);

                w = _mm256_loadu_ps(&filter_FP_copy[filter_subscript]);

                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s3, w, temp3);
                temp4 = _mm256_fmadd_ps(s4, w, temp4);
                

                // d+6 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+6);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+6);
                in_subscript3 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+2) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+6);
                in_subscript4 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+3) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+6);

                filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+6);

                s = _mm256_loadu_ps(&in_FP[in_subscript]);
                s2 = _mm256_loadu_ps(&in_FP[in_subscript2]);
                s3 = _mm256_loadu_ps(&in_FP[in_subscript3]);
                s4 = _mm256_loadu_ps(&in_FP[in_subscript4]);

                w = _mm256_loadu_ps(&filter_FP_copy[filter_subscript]);

                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s3, w, temp3);
                temp4 = _mm256_fmadd_ps(s4, w, temp4);
                

                // d+7 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+7);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+7);
                in_subscript3 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+2) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+7);
                in_subscript4 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+3) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+7);

                filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+7);

                s = _mm256_loadu_ps(&in_FP[in_subscript]);
                s2 = _mm256_loadu_ps(&in_FP[in_subscript2]);
                s3 = _mm256_loadu_ps(&in_FP[in_subscript3]);
                s4 = _mm256_loadu_ps(&in_FP[in_subscript4]);

                w = _mm256_loadu_ps(&filter_FP_copy[filter_subscript]);

                temp = _mm256_fmadd_ps(s, w, temp);
                temp2 = _mm256_fmadd_ps(s2, w, temp2);
                temp3 = _mm256_fmadd_ps(s3, w, temp3);
                temp4 = _mm256_fmadd_ps(s4, w, temp4);
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
        }
      }
    }
  }
  printf("\n from AC optv3 m24 x3 - d8 %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
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
  unsigned long long int new_subscript = 0;
  for (unsigned int y = 0; y < Mask_Y_dim; y++) {
    for (unsigned int x = 0; x < Mask_X_dim; x++) {
      for (unsigned int d = 0; d < Input_depth_dim; d++) {
        for (unsigned int m = 0; m < Output_depth_dim; m++) {
          unsigned long long int old_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
            + y * Mask_X_dim * Input_depth_dim
            + x * Input_depth_dim
            + d;
              
          // unsigned long long int new_subscript = y * Mask_X_dim * Input_depth_dim * Output_depth_dim
          //   + x * Input_depth_dim * Output_depth_dim
          //   + d * Output_depth_dim
          //   + m;

          filter_FP_copy[new_subscript] = filter_FP[old_subscript];
          new_subscript++;
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
  unsigned long long int new_subscript = 0;
  for (unsigned int y = 0; y < Mask_Y_dim; y++) {
    for (unsigned int x = 0; x < Mask_X_dim; x++) {
      for (unsigned int d = 0; d < Input_depth_dim; d++) {
        for (unsigned int m = 0; m < Output_depth_dim; m++) {
          unsigned long long int old_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
            + y * Mask_X_dim * Input_depth_dim
            + x * Input_depth_dim
            + d;
              
          // unsigned long long int new_subscript = y * Mask_X_dim * Input_depth_dim * Output_depth_dim
          //   + x * Input_depth_dim * Output_depth_dim
          //   + d * Output_depth_dim
          //   + m;

          filter_FP_copy[new_subscript] = filter_FP[old_subscript];
          new_subscript++;
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
  unsigned long long int new_subscript = 0;
  for (unsigned int y = 0; y < Mask_Y_dim; y++) {
    for (unsigned int x = 0; x < Mask_X_dim; x++) {
      for (unsigned int d = 0; d < Input_depth_dim; d++) {
        for (unsigned int m = 0; m < Output_depth_dim; m++) {
          unsigned long long int old_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
            + y * Mask_X_dim * Input_depth_dim
            + x * Input_depth_dim
            + d;
              
          // unsigned long long int new_subscript = y * Mask_X_dim * Input_depth_dim * Output_depth_dim
          //   + x * Input_depth_dim * Output_depth_dim
          //   + d * Output_depth_dim
          //   + m;

          filter_FP_copy[new_subscript] = filter_FP[old_subscript];
          new_subscript++;
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
  unsigned long long int new_subscript = 0;
  for (unsigned int y = 0; y < Mask_Y_dim; y++) {
    for (unsigned int x = 0; x < Mask_X_dim; x++) {
      for (unsigned int d = 0; d < Input_depth_dim; d++) {
        for (unsigned int m = 0; m < Output_depth_dim; m++) {
          unsigned long long int old_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
            + y * Mask_X_dim * Input_depth_dim
            + x * Input_depth_dim
            + d;
              
          // unsigned long long int new_subscript = y * Mask_X_dim * Input_depth_dim * Output_depth_dim
          //   + x * Input_depth_dim * Output_depth_dim
          //   + d * Output_depth_dim
          //   + m;

          filter_FP_copy[new_subscript] = filter_FP[old_subscript];
          new_subscript++;
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
  unsigned long long int new_subscript = 0;
  for (unsigned int y = 0; y < Mask_Y_dim; y++) {
    for (unsigned int x = 0; x < Mask_X_dim; x++) {
      for (unsigned int d = 0; d < Input_depth_dim; d++) {
        for (unsigned int m = 0; m < Output_depth_dim; m++) {
          unsigned long long int old_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
            + y * Mask_X_dim * Input_depth_dim
            + x * Input_depth_dim
            + d;
              
          // unsigned long long int new_subscript = y * Mask_X_dim * Input_depth_dim * Output_depth_dim
          //   + x * Input_depth_dim * Output_depth_dim
          //   + d * Output_depth_dim
          //   + m;

          filter_FP_copy[new_subscript] = filter_FP[old_subscript];
          new_subscript++;
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
  unsigned long long int new_subscript = 0;
  for (unsigned int y = 0; y < Mask_Y_dim; y++) {
    for (unsigned int x = 0; x < Mask_X_dim; x++) {
      for (unsigned int d = 0; d < Input_depth_dim; d++) {
        for (unsigned int m = 0; m < Output_depth_dim; m++) {
          unsigned long long int old_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
            + y * Mask_X_dim * Input_depth_dim
            + x * Input_depth_dim
            + d;
              
          // unsigned long long int new_subscript = y * Mask_X_dim * Input_depth_dim * Output_depth_dim
          //   + x * Input_depth_dim * Output_depth_dim
          //   + d * Output_depth_dim
          //   + m;

          filter_FP_copy[new_subscript] = filter_FP[old_subscript];
          new_subscript++;
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
  unsigned long long int new_subscript = 0;
  for (unsigned int y = 0; y < Mask_Y_dim; y++) {
    for (unsigned int x = 0; x < Mask_X_dim; x++) {
      for (unsigned int d = 0; d < Input_depth_dim; d++) {
        for (unsigned int m = 0; m < Output_depth_dim; m++) {
          unsigned long long int old_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
            + y * Mask_X_dim * Input_depth_dim
            + x * Input_depth_dim
            + d;
              
          // unsigned long long int new_subscript = y * Mask_X_dim * Input_depth_dim * Output_depth_dim
          //   + x * Input_depth_dim * Output_depth_dim
          //   + d * Output_depth_dim
          //   + m;

          filter_FP_copy[new_subscript] = filter_FP[old_subscript];
          new_subscript++;
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
  unsigned long long int new_subscript = 0;
  for (unsigned int y = 0; y < Mask_Y_dim; y++) {
    for (unsigned int x = 0; x < Mask_X_dim; x++) {
      for (unsigned int d = 0; d < Input_depth_dim; d++) {
        for (unsigned int m = 0; m < Output_depth_dim; m++) {
          unsigned long long int old_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
            + y * Mask_X_dim * Input_depth_dim
            + x * Input_depth_dim
            + d;
              
          // unsigned long long int new_subscript = y * Mask_X_dim * Input_depth_dim * Output_depth_dim
          //   + x * Input_depth_dim * Output_depth_dim
          //   + d * Output_depth_dim
          //   + m;

          filter_FP_copy[new_subscript] = filter_FP[old_subscript];
          new_subscript++;
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
  unsigned long long int new_subscript = 0;
  for (unsigned int y = 0; y < Mask_Y_dim; y++) {
    for (unsigned int x = 0; x < Mask_X_dim; x++) {
      for (unsigned int d = 0; d < Input_depth_dim; d++) {
        for (unsigned int m = 0; m < Output_depth_dim; m++) {
          unsigned long long int old_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
            + y * Mask_X_dim * Input_depth_dim
            + x * Input_depth_dim
            + d;
              
          // unsigned long long int new_subscript = y * Mask_X_dim * Input_depth_dim * Output_depth_dim
          //   + x * Input_depth_dim * Output_depth_dim
          //   + d * Output_depth_dim
          //   + m;

          filter_FP_copy[new_subscript] = filter_FP[old_subscript];
          new_subscript++;
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
  unsigned long long int new_subscript = 0;
  for (unsigned int y = 0; y < Mask_Y_dim; y++) {
    for (unsigned int x = 0; x < Mask_X_dim; x++) {
      for (unsigned int d = 0; d < Input_depth_dim; d++) {
        for (unsigned int m = 0; m < Output_depth_dim; m++) {
          unsigned long long int old_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
            + y * Mask_X_dim * Input_depth_dim
            + x * Input_depth_dim
            + d;
              
          // unsigned long long int new_subscript = y * Mask_X_dim * Input_depth_dim * Output_depth_dim
          //   + x * Input_depth_dim * Output_depth_dim
          //   + d * Output_depth_dim
          //   + m;

          filter_FP_copy[new_subscript] = filter_FP[old_subscript];
          new_subscript++;
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
