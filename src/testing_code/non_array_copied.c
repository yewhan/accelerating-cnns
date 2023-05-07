#include "../convolution_layer_2D.h"

// vectorised d loop, using hadd instructions
// 23 GFLOPS
int optimised_layer_v1_vectorised_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  float bias;
  __m256 temp;

  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int m = 0; m < Output_depth_dim; m++) { //channels
      bias = bias_array_FP[m];

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (unsigned int x = 0; x < Output_X_dim; x++) {	//Output Width
          temp = _mm256_setzero_ps();
          // temp = 0.0f;

          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + d;

                unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;

                __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]);
                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));


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

          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);

        }
      }
    }
  }

  printf("\n from optv1 hadd %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// vectorised d loop, optimised hadd
// 23 GFLOPS, compare again after all optimisations applied
int optimised_layer_v1_vectorised_opt_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  float bias;
  __m256 temp;

  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int m = 0; m < Output_depth_dim; m++) { //channels
      bias = bias_array_FP[m];

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (unsigned int x = 0; x < Output_X_dim; x++) {	//Output Width
          temp = _mm256_setzero_ps();
          // temp = 0.0f;

          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + d;

                unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;

                __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]);
                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));


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


          __m128 sseLo = _mm256_castps256_ps128(temp);
          __m128 sseHi = _mm256_extractf128_ps(temp, 1);
          sseLo = _mm_add_ps(sseLo, sseHi);

          __m128 sseShuf = _mm_movehdup_ps(sseLo);
          __m128 sseSum = _mm_add_ps(sseLo, sseShuf);
          sseShuf = _mm_movehl_ps(sseShuf, sseSum);
          sseSum = _mm_add_ss(sseSum, sseShuf);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);

        }
      }
    }
  }

  printf("\n from optv1 hadd %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}






// v2 - unroll x/ m loop


// unrolled x by 2
// ~42 GFLOPS
int optimised_layer_v2_unroll_x2_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  float bias;
  __m256 temp, temp2;

  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int m = 0; m < Output_depth_dim; m++) { //channels
      bias = bias_array_FP[m];

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (unsigned int x = 0; x < Output_X_dim; x+=2) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          // temp = 0.0f;

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


                unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;

                __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                __m256 s2 = _mm256_load_ps(&in_FP[in_subscript2]);

                __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));


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

          __m128 sseLo = _mm256_castps256_ps128(temp);
          __m128 sseHi = _mm256_extractf128_ps(temp, 1);
          sseLo = _mm_add_ps(sseLo, sseHi);

          __m128 sseShuf = _mm_movehdup_ps(sseLo);
          __m128 sseSum = _mm_add_ps(sseLo, sseShuf);
          sseShuf = _mm_movehl_ps(sseShuf, sseSum);
          sseSum = _mm_add_ss(sseSum, sseShuf);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          __m128 sseLo2 = _mm256_castps256_ps128(temp2);
          __m128 sseHi2 = _mm256_extractf128_ps(temp2, 1);
          sseLo2 = _mm_add_ps(sseLo2, sseHi2);

          __m128 sseShuf2 = _mm_movehdup_ps(sseLo2);
          __m128 sseSum2 = _mm_add_ps(sseLo2, sseShuf2);
          sseShuf2 = _mm_movehl_ps(sseShuf2, sseSum2);
          sseSum2 = _mm_add_ss(sseSum2, sseShuf2);
          
          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);




        }
      }
    }
  }

  printf("\n from optv2 x2 %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// unrolled x by 4
// ~60 GFLOPS
int optimised_layer_v2_unroll_x4_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  float bias;
  __m256 temp, temp2, temp3, temp4;

  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int m = 0; m < Output_depth_dim; m++) { //channels
      bias = bias_array_FP[m];

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (unsigned int x = 0; x < Output_X_dim; x+=4) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          temp4 = _mm256_setzero_ps();
          // temp = 0.0f;

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

                __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                __m256 s2 = _mm256_load_ps(&in_FP[in_subscript2]);
                __m256 s3 = _mm256_load_ps(&in_FP[in_subscript3]);
                __m256 s4 = _mm256_load_ps(&in_FP[in_subscript4]);

                __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s4, w));


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

          __m128 sseLo = _mm256_castps256_ps128(temp);
          __m128 sseHi = _mm256_extractf128_ps(temp, 1);
          sseLo = _mm_add_ps(sseLo, sseHi);

          __m128 sseShuf = _mm_movehdup_ps(sseLo);
          __m128 sseSum = _mm_add_ps(sseLo, sseShuf);
          sseShuf = _mm_movehl_ps(sseShuf, sseSum);
          sseSum = _mm_add_ss(sseSum, sseShuf);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          __m128 sseLo2 = _mm256_castps256_ps128(temp2);
          __m128 sseHi2 = _mm256_extractf128_ps(temp2, 1);
          sseLo2 = _mm_add_ps(sseLo2, sseHi2);

          __m128 sseShuf2 = _mm_movehdup_ps(sseLo2);
          __m128 sseSum2 = _mm_add_ps(sseLo2, sseShuf2);
          sseShuf2 = _mm_movehl_ps(sseShuf2, sseSum2);
          sseSum2 = _mm_add_ss(sseSum2, sseShuf2);
          
          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          __m128 sseLo3 = _mm256_castps256_ps128(temp3);
          __m128 sseHi3 = _mm256_extractf128_ps(temp3, 1);
          sseLo3 = _mm_add_ps(sseLo3, sseHi3);

          __m128 sseShuf3 = _mm_movehdup_ps(sseLo3);
          __m128 sseSum3 = _mm_add_ps(sseLo3, sseShuf3);
          sseShuf3 = _mm_movehl_ps(sseShuf3, sseSum3);
          sseSum3 = _mm_add_ss(sseSum3, sseShuf3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


          __m128 sseLo4 = _mm256_castps256_ps128(temp4);
          __m128 sseHi4 = _mm256_extractf128_ps(temp4, 1);
          sseLo4 = _mm_add_ps(sseLo4, sseHi4);

          __m128 sseShuf4 = _mm_movehdup_ps(sseLo4);
          __m128 sseSum4 = _mm_add_ps(sseLo4, sseShuf4);
          sseShuf4 = _mm_movehl_ps(sseShuf4, sseSum4);
          sseSum4 = _mm_add_ss(sseSum4, sseShuf4);
          
          float sum4 = _mm_cvtss_f32(sseSum4);

          sum4 += bias;
          out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);

        }
      }
    }
  }

  printf("\n from optv2 x4 %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// same as above, but using hadd to reduce simd register pressure in x loop
// ~63 GFLOPS
int optimised_layer_v2_unroll_x4_hadd_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  float bias;
  __m256 temp, temp2, temp3, temp4;

  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int m = 0; m < Output_depth_dim; m++) { //channels
      bias = bias_array_FP[m];

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (unsigned int x = 0; x < Output_X_dim; x+=4) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          temp4 = _mm256_setzero_ps();
          // temp = 0.0f;

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

                __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                __m256 s2 = _mm256_load_ps(&in_FP[in_subscript2]);
                __m256 s3 = _mm256_load_ps(&in_FP[in_subscript3]);
                __m256 s4 = _mm256_load_ps(&in_FP[in_subscript4]);

                __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s4, w));


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


          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
          __m128 tempLo3 = _mm256_castps256_ps128(temp3);
          __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
          __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


          temp4 = _mm256_hadd_ps(temp4, temp4);
          temp4 = _mm256_hadd_ps(temp4, temp4);
          __m128 tempLo4 = _mm256_castps256_ps128(temp4);
          __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
          __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

          float sum4 = _mm_cvtss_f32(sseSum4);

          sum4 += bias;
          out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);
          
        }
      }
    }
  }

  printf("\n from optv2 x4 hadd %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// same as above, but x unrolled by 8
// ~63 GFLOPS, increased register pressure in x loop potentially offset unroll perf, not worth
int optimised_layer_v2_unroll_x8_hadd_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  float bias;
  __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int m = 0; m < Output_depth_dim; m++) { //channels
      bias = bias_array_FP[m];

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

          // temp = 0.0f;

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
                  +d;


                unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;

                __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                __m256 s2 = _mm256_load_ps(&in_FP[in_subscript2]);
                __m256 s3 = _mm256_load_ps(&in_FP[in_subscript3]);
                __m256 s4 = _mm256_load_ps(&in_FP[in_subscript4]);
                __m256 s5 = _mm256_load_ps(&in_FP[in_subscript5]);
                __m256 s6 = _mm256_load_ps(&in_FP[in_subscript6]);
                __m256 s7 = _mm256_load_ps(&in_FP[in_subscript7]);
                __m256 s8 = _mm256_load_ps(&in_FP[in_subscript8]);

                __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s4, w));
                temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s5, w));
                temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s6, w));
                temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s7, w));
                temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s8, w));


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


          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
          __m128 tempLo3 = _mm256_castps256_ps128(temp3);
          __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
          __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


          temp4 = _mm256_hadd_ps(temp4, temp4);
          temp4 = _mm256_hadd_ps(temp4, temp4);
          __m128 tempLo4 = _mm256_castps256_ps128(temp4);
          __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
          __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

          float sum4 = _mm_cvtss_f32(sseSum4);

          sum4 += bias;
          out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);


          temp5 = _mm256_hadd_ps(temp5, temp5);
          temp5 = _mm256_hadd_ps(temp5, temp5);
          __m128 tempLo5 = _mm256_castps256_ps128(temp5);
          __m128 tempHi5 = _mm256_extractf128_ps(temp5, 1);
          __m128 sseSum5 = _mm_add_ps(tempLo5, tempHi5);

          float sum5 = _mm_cvtss_f32(sseSum5);

          sum5 += bias;
          out_to_compare_with_FP[out_subscript5] = Relu_float(sum5);


          temp6 = _mm256_hadd_ps(temp6, temp6);
          temp6 = _mm256_hadd_ps(temp6, temp6);
          __m128 tempLo6 = _mm256_castps256_ps128(temp6);
          __m128 tempHi6 = _mm256_extractf128_ps(temp6, 1);
          __m128 sseSum6 = _mm_add_ps(tempLo6, tempHi6);

          float sum6 = _mm_cvtss_f32(sseSum6);

          sum6 += bias;
          out_to_compare_with_FP[out_subscript6] = Relu_float(sum6);


          temp7 = _mm256_hadd_ps(temp7, temp7);
          temp7 = _mm256_hadd_ps(temp7, temp7);
          __m128 tempLo7 = _mm256_castps256_ps128(temp7);
          __m128 tempHi7 = _mm256_extractf128_ps(temp7, 1);
          __m128 sseSum7 = _mm_add_ps(tempLo7, tempHi7);

          float sum7 = _mm_cvtss_f32(sseSum7);

          sum7 += bias;
          out_to_compare_with_FP[out_subscript7] = Relu_float(sum7);


          temp8 = _mm256_hadd_ps(temp8, temp8);
          temp8 = _mm256_hadd_ps(temp8, temp8);
          __m128 tempLo8 = _mm256_castps256_ps128(temp8);
          __m128 tempHi8 = _mm256_extractf128_ps(temp8, 1);
          __m128 sseSum8 = _mm_add_ps(tempLo8, tempHi8);

          float sum8 = _mm_cvtss_f32(sseSum8);

          sum8 += bias;
          out_to_compare_with_FP[out_subscript8] = Relu_float(sum8);
          
        }
      }
    }
  }

  printf("\n from optv2 x8 hadd %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// unrolled m loop by 2
// ~42 GFLOPS
int optimised_layer_v2_unroll_m2_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  float bias, bias2;
  __m256 temp, temp2;

  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int m = 0; m < Output_depth_dim; m+=2) { //channels
      bias = bias_array_FP[m];
      bias2 = bias_array_FP[m+1];

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (unsigned int x = 0; x < Output_X_dim; x++) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          // temp = 0.0f;

          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + d;

                unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;

                __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]);
                __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s, w2));


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
            x * Output_depth_dim
            + (m+1);


          __m128 sseLo = _mm256_castps256_ps128(temp);
          __m128 sseHi = _mm256_extractf128_ps(temp, 1);
          sseLo = _mm_add_ps(sseLo, sseHi);

          __m128 sseShuf = _mm_movehdup_ps(sseLo);
          __m128 sseSum = _mm_add_ps(sseLo, sseShuf);
          sseShuf = _mm_movehl_ps(sseShuf, sseSum);
          sseSum = _mm_add_ss(sseSum, sseShuf);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          __m128 sseLo2 = _mm256_castps256_ps128(temp2);
          __m128 sseHi2 = _mm256_extractf128_ps(temp2, 1);
          sseLo2 = _mm_add_ps(sseLo2, sseHi2);

          __m128 sseShuf2 = _mm_movehdup_ps(sseLo2);
          __m128 sseSum2 = _mm_add_ps(sseLo2, sseShuf2);
          sseShuf2 = _mm_movehl_ps(sseShuf2, sseSum2);
          sseSum2 = _mm_add_ss(sseSum2, sseShuf2);
          
          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias2;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);

        }
      }
    }
  }

  printf("\n from optv2 m2 %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// unrolled m loop by 4
// ~50 GFLOPS, worse performance than x4? maybe because unrolling inner loops is more beneficial?
int optimised_layer_v2_unroll_m4_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  float bias, bias2, bias3, bias4;
  __m256 temp, temp2, temp3, temp4;

  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int m = 0; m < Output_depth_dim; m+=4) { //channels
      bias = bias_array_FP[m];
      bias2 = bias_array_FP[m+1];
      bias3 = bias_array_FP[m+2];
      bias4 = bias_array_FP[m+3];


      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (unsigned int x = 0; x < Output_X_dim; x++) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          temp4 = _mm256_setzero_ps();
          // temp = 0.0f;

          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + d;

                unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript4 = (m+3) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;

                __m256 s = _mm256_load_ps(&in_FP[in_subscript]);

                __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]);
                __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);
                __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript3]);
                __m256 w4 = _mm256_load_ps(&filter_FP[filter_subscript4]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s, w2));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w3));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s, w4));


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
            x * Output_depth_dim
            + (m+1);
          unsigned long long int out_subscript3 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+2);
          unsigned long long int out_subscript4 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+3);


          __m128 sseLo = _mm256_castps256_ps128(temp);
          __m128 sseHi = _mm256_extractf128_ps(temp, 1);
          sseLo = _mm_add_ps(sseLo, sseHi);

          __m128 sseShuf = _mm_movehdup_ps(sseLo);
          __m128 sseSum = _mm_add_ps(sseLo, sseShuf);
          sseShuf = _mm_movehl_ps(sseShuf, sseSum);
          sseSum = _mm_add_ss(sseSum, sseShuf);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          __m128 sseLo2 = _mm256_castps256_ps128(temp2);
          __m128 sseHi2 = _mm256_extractf128_ps(temp2, 1);
          sseLo2 = _mm_add_ps(sseLo2, sseHi2);

          __m128 sseShuf2 = _mm_movehdup_ps(sseLo2);
          __m128 sseSum2 = _mm_add_ps(sseLo2, sseShuf2);
          sseShuf2 = _mm_movehl_ps(sseShuf2, sseSum2);
          sseSum2 = _mm_add_ss(sseSum2, sseShuf2);
          
          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias2;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          __m128 sseLo3 = _mm256_castps256_ps128(temp3);
          __m128 sseHi3 = _mm256_extractf128_ps(temp3, 1);
          sseLo3 = _mm_add_ps(sseLo3, sseHi3);

          __m128 sseShuf3 = _mm_movehdup_ps(sseLo3);
          __m128 sseSum3 = _mm_add_ps(sseLo3, sseShuf3);
          sseShuf3 = _mm_movehl_ps(sseShuf3, sseSum3);
          sseSum3 = _mm_add_ss(sseSum3, sseShuf3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias3;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


          __m128 sseLo4 = _mm256_castps256_ps128(temp4);
          __m128 sseHi4 = _mm256_extractf128_ps(temp4, 1);
          sseLo4 = _mm_add_ps(sseLo4, sseHi4);

          __m128 sseShuf4 = _mm_movehdup_ps(sseLo4);
          __m128 sseSum4 = _mm_add_ps(sseLo4, sseShuf4);
          sseShuf4 = _mm_movehl_ps(sseShuf4, sseSum4);
          sseSum4 = _mm_add_ss(sseSum4, sseShuf4);
          
          float sum4 = _mm_cvtss_f32(sseSum4);

          sum4 += bias4;
          out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);

        }
      }
    }
  }

  printf("\n from optv2 m4 %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// same as above, but using hadd to reduce simd register pressure in x loop
// ~50 GFLOPS
int optimised_layer_v2_unroll_m4_hadd_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  float bias, bias2, bias3, bias4;
  __m256 temp, temp2, temp3, temp4;

  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int m = 0; m < Output_depth_dim; m+=4) { //channels
      bias = bias_array_FP[m];
      bias2 = bias_array_FP[m+1];
      bias3 = bias_array_FP[m+2];
      bias4 = bias_array_FP[m+3];


      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (unsigned int x = 0; x < Output_X_dim; x++) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          temp4 = _mm256_setzero_ps();
          // temp = 0.0f;

          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + d;

                unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript4 = (m+3) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;

                __m256 s = _mm256_load_ps(&in_FP[in_subscript]);

                __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]);
                __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);
                __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript3]);
                __m256 w4 = _mm256_load_ps(&filter_FP[filter_subscript4]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s, w2));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w3));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s, w4));


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
            x * Output_depth_dim
            + (m+1);
          unsigned long long int out_subscript3 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+2);
          unsigned long long int out_subscript4 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+3);


          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias2;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
          __m128 tempLo3 = _mm256_castps256_ps128(temp3);
          __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
          __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias3;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


          temp4 = _mm256_hadd_ps(temp4, temp4);
          temp4 = _mm256_hadd_ps(temp4, temp4);
          __m128 tempLo4 = _mm256_castps256_ps128(temp4);
          __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
          __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

          float sum4 = _mm_cvtss_f32(sseSum4);

          sum4 += bias4;
          out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);

        }
      }
    }
  }

  printf("\n from optv2 m4 hadd %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// same as above, but using hadd to reduce simd register pressure in x loop
// ~56 GFLOPS
int optimised_layer_v2_unroll_m8_hadd_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  float bias, bias2, bias3, bias4, bias5, bias6, bias7, bias8;
  __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int m = 0; m < Output_depth_dim; m+=8) { //channels
      bias = bias_array_FP[m];
      bias2 = bias_array_FP[m+1];
      bias3 = bias_array_FP[m+2];
      bias4 = bias_array_FP[m+3];
      bias5 = bias_array_FP[m+4];
      bias6 = bias_array_FP[m+5];
      bias7 = bias_array_FP[m+6];
      bias8 = bias_array_FP[m+7];

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
          // temp = 0.0f;

          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + d;

                unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript4 = (m+3) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript5 = (m+4) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript6 = (m+5) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript7 = (m+6) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript8 = (m+7) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;

                __m256 s = _mm256_load_ps(&in_FP[in_subscript]);

                __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]);
                __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);
                __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript3]);
                __m256 w4 = _mm256_load_ps(&filter_FP[filter_subscript4]);
                __m256 w5 = _mm256_load_ps(&filter_FP[filter_subscript5]);
                __m256 w6 = _mm256_load_ps(&filter_FP[filter_subscript6]);
                __m256 w7 = _mm256_load_ps(&filter_FP[filter_subscript7]);
                __m256 w8 = _mm256_load_ps(&filter_FP[filter_subscript8]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s, w2));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w3));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s, w4));
                temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s, w5));
                temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s, w6));
                temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w7));
                temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s, w8));


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
            x * Output_depth_dim
            + (m+1);
          unsigned long long int out_subscript3 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+2);
          unsigned long long int out_subscript4 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+3);
          unsigned long long int out_subscript5 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+4);
          unsigned long long int out_subscript6 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+5);
          unsigned long long int out_subscript7 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+6);
          unsigned long long int out_subscript8 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+7);


          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias2;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
          __m128 tempLo3 = _mm256_castps256_ps128(temp3);
          __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
          __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias3;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


          temp4 = _mm256_hadd_ps(temp4, temp4);
          temp4 = _mm256_hadd_ps(temp4, temp4);
          __m128 tempLo4 = _mm256_castps256_ps128(temp4);
          __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
          __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

          float sum4 = _mm_cvtss_f32(sseSum4);

          sum4 += bias4;
          out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);


          temp5 = _mm256_hadd_ps(temp5, temp5);
          temp5 = _mm256_hadd_ps(temp5, temp5);
          __m128 tempLo5 = _mm256_castps256_ps128(temp5);
          __m128 tempHi5 = _mm256_extractf128_ps(temp5, 1);
          __m128 sseSum5 = _mm_add_ps(tempLo5, tempHi5);

          float sum5 = _mm_cvtss_f32(sseSum5);

          sum5 += bias5;
          out_to_compare_with_FP[out_subscript5] = Relu_float(sum5);


          temp6 = _mm256_hadd_ps(temp6, temp6);
          temp6 = _mm256_hadd_ps(temp6, temp6);
          __m128 tempLo6 = _mm256_castps256_ps128(temp6);
          __m128 tempHi6 = _mm256_extractf128_ps(temp6, 1);
          __m128 sseSum6 = _mm_add_ps(tempLo6, tempHi6);

          float sum6 = _mm_cvtss_f32(sseSum6);

          sum6 += bias6;
          out_to_compare_with_FP[out_subscript6] = Relu_float(sum6);


          temp7 = _mm256_hadd_ps(temp7, temp7);
          temp7 = _mm256_hadd_ps(temp7, temp7);
          __m128 tempLo7 = _mm256_castps256_ps128(temp7);
          __m128 tempHi7 = _mm256_extractf128_ps(temp7, 1);
          __m128 sseSum7 = _mm_add_ps(tempLo7, tempHi7);

          float sum7 = _mm_cvtss_f32(sseSum7);

          sum7 += bias7;
          out_to_compare_with_FP[out_subscript7] = Relu_float(sum7);


          temp8 = _mm256_hadd_ps(temp8, temp8);
          temp8 = _mm256_hadd_ps(temp8, temp8);
          __m128 tempLo8 = _mm256_castps256_ps128(temp8);
          __m128 tempHi8 = _mm256_extractf128_ps(temp8, 1);
          __m128 sseSum8 = _mm_add_ps(tempLo8, tempHi8);

          float sum8 = _mm_cvtss_f32(sseSum8);

          sum8 += bias8;
          out_to_compare_with_FP[out_subscript8] = Relu_float(sum8);
        }
      }
    }
  }

  printf("\n from optv2 m8 hadd %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// unrolled x by 2, m by 2
// 57~ GFLOPS, 
int optimised_layer_v2_unroll_x2m2_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  float bias, bias2;
  __m256 temp, temp2, temp3, temp4;

  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int m = 0; m < Output_depth_dim; m+=2) { //channels
      bias = bias_array_FP[m];
      bias2 = bias_array_FP[m+1];

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (unsigned int x = 0; x < Output_X_dim; x+=2) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          temp4 = _mm256_setzero_ps();
          // temp = 0.0f;

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


                unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;

                __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                __m256 s2 = _mm256_load_ps(&in_FP[in_subscript2]);

                __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]);
                __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w2));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s2, w2));



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
            x * Output_depth_dim
            + (m+1);
          unsigned long long int out_subscript4 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+1);



          __m128 sseLo = _mm256_castps256_ps128(temp);
          __m128 sseHi = _mm256_extractf128_ps(temp, 1);
          sseLo = _mm_add_ps(sseLo, sseHi);

          __m128 sseShuf = _mm_movehdup_ps(sseLo);
          __m128 sseSum = _mm_add_ps(sseLo, sseShuf);
          sseShuf = _mm_movehl_ps(sseShuf, sseSum);
          sseSum = _mm_add_ss(sseSum, sseShuf);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          __m128 sseLo2 = _mm256_castps256_ps128(temp2);
          __m128 sseHi2 = _mm256_extractf128_ps(temp2, 1);
          sseLo2 = _mm_add_ps(sseLo2, sseHi2);

          __m128 sseShuf2 = _mm_movehdup_ps(sseLo2);
          __m128 sseSum2 = _mm_add_ps(sseLo2, sseShuf2);
          sseShuf2 = _mm_movehl_ps(sseShuf2, sseSum2);
          sseSum2 = _mm_add_ss(sseSum2, sseShuf2);
          
          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          __m128 sseLo3 = _mm256_castps256_ps128(temp3);
          __m128 sseHi3 = _mm256_extractf128_ps(temp3, 1);
          sseLo3 = _mm_add_ps(sseLo3, sseHi3);

          __m128 sseShuf3 = _mm_movehdup_ps(sseLo3);
          __m128 sseSum3 = _mm_add_ps(sseLo3, sseShuf3);
          sseShuf3 = _mm_movehl_ps(sseShuf3, sseSum3);
          sseSum3 = _mm_add_ss(sseSum3, sseShuf3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias2;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


          __m128 sseLo4 = _mm256_castps256_ps128(temp4);
          __m128 sseHi4 = _mm256_extractf128_ps(temp4, 1);
          sseLo4 = _mm_add_ps(sseLo4, sseHi4);

          __m128 sseShuf4 = _mm_movehdup_ps(sseLo4);
          __m128 sseSum4 = _mm_add_ps(sseLo4, sseShuf4);
          sseShuf4 = _mm_movehl_ps(sseShuf4, sseSum4);
          sseSum4 = _mm_add_ss(sseSum4, sseShuf4);
          
          float sum4 = _mm_cvtss_f32(sseSum4);

          sum4 += bias2;
          out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);

        }
      }
    }
  }

  printf("\n from optv2 x2 m2 %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// same as above, but using hadd to reduce simd register pressure in x loop
// ~57 GFLOPS, 
int optimised_layer_v2_unroll_x2m2_hadd_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  float bias, bias2;
  __m256 temp, temp2, temp3, temp4;

  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int m = 0; m < Output_depth_dim; m+=2) { //channels
      bias = bias_array_FP[m];
      bias2 = bias_array_FP[m+1];

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (unsigned int x = 0; x < Output_X_dim; x+=2) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          temp4 = _mm256_setzero_ps();
          // temp = 0.0f;

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


                unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;

                __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                __m256 s2 = _mm256_load_ps(&in_FP[in_subscript2]);

                __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]);
                __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w2));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s2, w2));



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
            x * Output_depth_dim
            + (m+1);
          unsigned long long int out_subscript4 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+1);



          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
          __m128 tempLo3 = _mm256_castps256_ps128(temp3);
          __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
          __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias2;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


          temp4 = _mm256_hadd_ps(temp4, temp4);
          temp4 = _mm256_hadd_ps(temp4, temp4);
          __m128 tempLo4 = _mm256_castps256_ps128(temp4);
          __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
          __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

          float sum4 = _mm_cvtss_f32(sseSum4);

          sum4 += bias2;
          out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);

        }
      }
    }
  }

  printf("\n from optv2 x2 m2 hadd %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// unrolled x by 4, m by 2
// ~72 GFLOPS, f.p. error at epsilon = 0.001, 0.01 fine
int optimised_layer_v2_unroll_x4m2_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  float bias, bias2;
  __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int m = 0; m < Output_depth_dim; m+=2) { //channels
      bias = bias_array_FP[m];
      bias2 = bias_array_FP[m+1];

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
          // temp = 0.0f;

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
                unsigned long long int filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;

                __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                __m256 s2 = _mm256_load_ps(&in_FP[in_subscript2]);
                __m256 s3 = _mm256_load_ps(&in_FP[in_subscript3]);
                __m256 s4 = _mm256_load_ps(&in_FP[in_subscript4]);

                __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]);
                __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s4, w));
                temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s, w2));
                temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s2, w2));
                temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s3, w2));
                temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s4, w2));


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
            + (m+1);
          unsigned long long int out_subscript6 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+1);
          unsigned long long int out_subscript7 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+2) * Output_depth_dim
            + (m+1);
          unsigned long long int out_subscript8 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+3) * Output_depth_dim
            + (m+1);



          __m128 sseLo = _mm256_castps256_ps128(temp);
          __m128 sseHi = _mm256_extractf128_ps(temp, 1);
          sseLo = _mm_add_ps(sseLo, sseHi);

          __m128 sseShuf = _mm_movehdup_ps(sseLo);
          __m128 sseSum = _mm_add_ps(sseLo, sseShuf);
          sseShuf = _mm_movehl_ps(sseShuf, sseSum);
          sseSum = _mm_add_ss(sseSum, sseShuf);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          __m128 sseLo2 = _mm256_castps256_ps128(temp2);
          __m128 sseHi2 = _mm256_extractf128_ps(temp2, 1);
          sseLo2 = _mm_add_ps(sseLo2, sseHi2);

          __m128 sseShuf2 = _mm_movehdup_ps(sseLo2);
          __m128 sseSum2 = _mm_add_ps(sseLo2, sseShuf2);
          sseShuf2 = _mm_movehl_ps(sseShuf2, sseSum2);
          sseSum2 = _mm_add_ss(sseSum2, sseShuf2);
          
          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          __m128 sseLo3 = _mm256_castps256_ps128(temp3);
          __m128 sseHi3 = _mm256_extractf128_ps(temp3, 1);
          sseLo3 = _mm_add_ps(sseLo3, sseHi3);

          __m128 sseShuf3 = _mm_movehdup_ps(sseLo3);
          __m128 sseSum3 = _mm_add_ps(sseLo3, sseShuf3);
          sseShuf3 = _mm_movehl_ps(sseShuf3, sseSum3);
          sseSum3 = _mm_add_ss(sseSum3, sseShuf3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


          __m128 sseLo4 = _mm256_castps256_ps128(temp4);
          __m128 sseHi4 = _mm256_extractf128_ps(temp4, 1);
          sseLo4 = _mm_add_ps(sseLo4, sseHi4);

          __m128 sseShuf4 = _mm_movehdup_ps(sseLo4);
          __m128 sseSum4 = _mm_add_ps(sseLo4, sseShuf4);
          sseShuf4 = _mm_movehl_ps(sseShuf4, sseSum4);
          sseSum4 = _mm_add_ss(sseSum4, sseShuf4);
          
          float sum4 = _mm_cvtss_f32(sseSum4);

          sum4 += bias;
          out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);




          __m128 sseLo5 = _mm256_castps256_ps128(temp5);
          __m128 sseHi5 = _mm256_extractf128_ps(temp5, 1);
          sseLo5 = _mm_add_ps(sseLo5, sseHi5);

          __m128 sseShuf5 = _mm_movehdup_ps(sseLo5);
          __m128 sseSum5 = _mm_add_ps(sseLo5, sseShuf5);
          sseShuf5 = _mm_movehl_ps(sseShuf5, sseSum5);
          sseSum5 = _mm_add_ss(sseSum5, sseShuf5);

          float sum5 = _mm_cvtss_f32(sseSum5);

          sum5 += bias2;
          out_to_compare_with_FP[out_subscript5] = Relu_float(sum5);


          __m128 sseLo6 = _mm256_castps256_ps128(temp6);
          __m128 sseHi6 = _mm256_extractf128_ps(temp6, 1);
          sseLo6 = _mm_add_ps(sseLo6, sseHi6);

          __m128 sseShuf6 = _mm_movehdup_ps(sseLo6);
          __m128 sseSum6 = _mm_add_ps(sseLo6, sseShuf6);
          sseShuf6 = _mm_movehl_ps(sseShuf6, sseSum6);
          sseSum6 = _mm_add_ss(sseSum6, sseShuf6);

          float sum6 = _mm_cvtss_f32(sseSum6);

          sum5 += bias2;
          out_to_compare_with_FP[out_subscript6] = Relu_float(sum6);


          __m128 sseLo7 = _mm256_castps256_ps128(temp7);
          __m128 sseHi7 = _mm256_extractf128_ps(temp7, 1);
          sseLo7 = _mm_add_ps(sseLo7, sseHi7);

          __m128 sseShuf7 = _mm_movehdup_ps(sseLo7);
          __m128 sseSum7 = _mm_add_ps(sseLo7, sseShuf7);
          sseShuf7 = _mm_movehl_ps(sseShuf7, sseSum7);
          sseSum7 = _mm_add_ss(sseSum7, sseShuf7);

          float sum7 = _mm_cvtss_f32(sseSum7);

          sum7 += bias2;
          out_to_compare_with_FP[out_subscript7] = Relu_float(sum7);


          __m128 sseLo8 = _mm256_castps256_ps128(temp8);
          __m128 sseHi8 = _mm256_extractf128_ps(temp8, 1);
          sseLo8 = _mm_add_ps(sseLo8, sseHi8);

          __m128 sseShuf8 = _mm_movehdup_ps(sseLo8);
          __m128 sseSum8 = _mm_add_ps(sseLo8, sseShuf8);
          sseShuf8 = _mm_movehl_ps(sseShuf8, sseSum8);
          sseSum8 = _mm_add_ss(sseSum8, sseShuf8);

          float sum8 = _mm_cvtss_f32(sseSum8);

          sum8 += bias2;
          out_to_compare_with_FP[out_subscript8] = Relu_float(sum8);
        }
      }
    }
  }

  printf("\n from optv2 x4 m2 %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// same as above, but using hadd to reduce simd register pressure in x loop
// ~72 GFLOPS
int optimised_layer_v2_unroll_x4m2_hadd_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  float bias, bias2;
  __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int m = 0; m < Output_depth_dim; m+=2) { //channels
      bias = bias_array_FP[m];
      bias2 = bias_array_FP[m+1];

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
          // temp = 0.0f;

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
                unsigned long long int filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;

                __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                __m256 s2 = _mm256_load_ps(&in_FP[in_subscript2]);
                __m256 s3 = _mm256_load_ps(&in_FP[in_subscript3]);
                __m256 s4 = _mm256_load_ps(&in_FP[in_subscript4]);

                __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]);
                __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s4, w));
                temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s, w2));
                temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s2, w2));
                temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s3, w2));
                temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s4, w2));


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
            + (m+1);
          unsigned long long int out_subscript6 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+1);
          unsigned long long int out_subscript7 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+2) * Output_depth_dim
            + (m+1);
          unsigned long long int out_subscript8 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+3) * Output_depth_dim
            + (m+1);



          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
          __m128 tempLo3 = _mm256_castps256_ps128(temp3);
          __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
          __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


          temp4 = _mm256_hadd_ps(temp4, temp4);
          temp4 = _mm256_hadd_ps(temp4, temp4);
          __m128 tempLo4 = _mm256_castps256_ps128(temp4);
          __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
          __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

          float sum4 = _mm_cvtss_f32(sseSum4);

          sum4 += bias;
          out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);


          temp5 = _mm256_hadd_ps(temp5, temp5);
          temp5 = _mm256_hadd_ps(temp5, temp5);
          __m128 tempLo5 = _mm256_castps256_ps128(temp5);
          __m128 tempHi5 = _mm256_extractf128_ps(temp5, 1);
          __m128 sseSum5 = _mm_add_ps(tempLo5, tempHi5);

          float sum5 = _mm_cvtss_f32(sseSum5);

          sum5 += bias2;
          out_to_compare_with_FP[out_subscript5] = Relu_float(sum5);


          temp6 = _mm256_hadd_ps(temp6, temp6);
          temp6 = _mm256_hadd_ps(temp6, temp6);
          __m128 tempLo6 = _mm256_castps256_ps128(temp6);
          __m128 tempHi6 = _mm256_extractf128_ps(temp6, 1);
          __m128 sseSum6 = _mm_add_ps(tempLo6, tempHi6);

          float sum6 = _mm_cvtss_f32(sseSum6);

          sum6 += bias2;
          out_to_compare_with_FP[out_subscript6] = Relu_float(sum6);


          temp7 = _mm256_hadd_ps(temp7, temp7);
          temp7 = _mm256_hadd_ps(temp7, temp7);
          __m128 tempLo7 = _mm256_castps256_ps128(temp7);
          __m128 tempHi7 = _mm256_extractf128_ps(temp7, 1);
          __m128 sseSum7 = _mm_add_ps(tempLo7, tempHi7);

          float sum7 = _mm_cvtss_f32(sseSum7);

          sum7 += bias2;
          out_to_compare_with_FP[out_subscript7] = Relu_float(sum7);


          temp8 = _mm256_hadd_ps(temp8, temp8);
          temp8 = _mm256_hadd_ps(temp8, temp8);
          __m128 tempLo8 = _mm256_castps256_ps128(temp8);
          __m128 tempHi8 = _mm256_extractf128_ps(temp8, 1);
          __m128 sseSum8 = _mm_add_ps(tempLo8, tempHi8);

          float sum8 = _mm_cvtss_f32(sseSum8);

          sum8 += bias2;
          out_to_compare_with_FP[out_subscript8] = Relu_float(sum8);
        }
      }
    }
  }

  printf("\n from optv2 x4 m2 hadd %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// unrolled x by 2, m by 4. using hadd to reduce simd register pressure
// ~73 GFLOPS
int optimised_layer_v2_unroll_x2m4_hadd_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  float bias, bias2, bias3, bias4;
  __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int m = 0; m < Output_depth_dim; m+=4) { //channels
      bias = bias_array_FP[m];
      bias2 = bias_array_FP[m+1];
      bias3 = bias_array_FP[m+2];
      bias4 = bias_array_FP[m+3];

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
          // temp = 0.0f;

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


                unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript4 = (m+3) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;

                __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                __m256 s2 = _mm256_load_ps(&in_FP[in_subscript2]);

                __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]);
                __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);
                __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript3]);
                __m256 w4 = _mm256_load_ps(&filter_FP[filter_subscript4]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w2));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s2, w2));
                temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s, w3));
                temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s2, w3));
                temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w4));
                temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w4));


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
            x * Output_depth_dim
            + (m+1);
          unsigned long long int out_subscript4 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+1);
          unsigned long long int out_subscript5 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+2);
          unsigned long long int out_subscript6 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+2);
          unsigned long long int out_subscript7 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+3);
          unsigned long long int out_subscript8 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+3);



          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
          __m128 tempLo3 = _mm256_castps256_ps128(temp3);
          __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
          __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias2;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


          temp4 = _mm256_hadd_ps(temp4, temp4);
          temp4 = _mm256_hadd_ps(temp4, temp4);
          __m128 tempLo4 = _mm256_castps256_ps128(temp4);
          __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
          __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

          float sum4 = _mm_cvtss_f32(sseSum4);

          sum4 += bias2;
          out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);


          temp5 = _mm256_hadd_ps(temp5, temp5);
          temp5 = _mm256_hadd_ps(temp5, temp5);
          __m128 tempLo5 = _mm256_castps256_ps128(temp5);
          __m128 tempHi5 = _mm256_extractf128_ps(temp5, 1);
          __m128 sseSum5 = _mm_add_ps(tempLo5, tempHi5);

          float sum5 = _mm_cvtss_f32(sseSum5);

          sum5 += bias3;
          out_to_compare_with_FP[out_subscript5] = Relu_float(sum5);


          temp6 = _mm256_hadd_ps(temp6, temp6);
          temp6 = _mm256_hadd_ps(temp6, temp6);
          __m128 tempLo6 = _mm256_castps256_ps128(temp6);
          __m128 tempHi6 = _mm256_extractf128_ps(temp6, 1);
          __m128 sseSum6 = _mm_add_ps(tempLo6, tempHi6);

          float sum6 = _mm_cvtss_f32(sseSum6);

          sum6 += bias3;
          out_to_compare_with_FP[out_subscript6] = Relu_float(sum6);


          temp7 = _mm256_hadd_ps(temp7, temp7);
          temp7 = _mm256_hadd_ps(temp7, temp7);
          __m128 tempLo7 = _mm256_castps256_ps128(temp7);
          __m128 tempHi7 = _mm256_extractf128_ps(temp7, 1);
          __m128 sseSum7 = _mm_add_ps(tempLo7, tempHi7);

          float sum7 = _mm_cvtss_f32(sseSum7);

          sum7 += bias4;
          out_to_compare_with_FP[out_subscript7] = Relu_float(sum7);


          temp8 = _mm256_hadd_ps(temp8, temp8);
          temp8 = _mm256_hadd_ps(temp8, temp8);
          __m128 tempLo8 = _mm256_castps256_ps128(temp8);
          __m128 tempHi8 = _mm256_extractf128_ps(temp8, 1);
          __m128 sseSum8 = _mm_add_ps(tempLo8, tempHi8);

          float sum8 = _mm_cvtss_f32(sseSum8);

          sum8 += bias4;
          out_to_compare_with_FP[out_subscript8] = Relu_float(sum8);
        }
      }
    }
  }

  printf("\n from optv2 x2 m4 hadd %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// reduce simd register pressure in x loop (reusing tempLo, tempHi, sseSum & tempLo2, tempHi2, sseSum2)
// ~71/72 GFLOPS, likely similar/ same asm as 2 above
int optimised_layer_v2_unroll_x4m2_hadd_register_pressure_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  float bias, bias2;
  __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int m = 0; m < Output_depth_dim; m+=2) { //channels
      bias = bias_array_FP[m];
      bias2 = bias_array_FP[m+1];

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
          // temp = 0.0f;

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
                unsigned long long int filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;

                __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                __m256 s2 = _mm256_load_ps(&in_FP[in_subscript2]);
                __m256 s3 = _mm256_load_ps(&in_FP[in_subscript3]);
                __m256 s4 = _mm256_load_ps(&in_FP[in_subscript4]);

                __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]);
                __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s4, w));
                temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s, w2));
                temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s2, w2));
                temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s3, w2));
                temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s4, w2));


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
            + (m+1);
          unsigned long long int out_subscript6 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+1);
          unsigned long long int out_subscript7 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+2) * Output_depth_dim
            + (m+1);
          unsigned long long int out_subscript8 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+3) * Output_depth_dim
            + (m+1);



          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
           tempLo = _mm256_castps256_ps128(temp3);
           tempHi = _mm256_extractf128_ps(temp3, 1);
           sseSum = _mm_add_ps(tempLo, tempHi);

          float sum3 = _mm_cvtss_f32(sseSum);

          sum3 += bias;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


          temp4 = _mm256_hadd_ps(temp4, temp4);
          temp4 = _mm256_hadd_ps(temp4, temp4);
           tempLo2 = _mm256_castps256_ps128(temp4);
           tempHi2 = _mm256_extractf128_ps(temp4, 1);
           sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum4 = _mm_cvtss_f32(sseSum2);

          sum4 += bias;
          out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);


          temp5 = _mm256_hadd_ps(temp5, temp5);
          temp5 = _mm256_hadd_ps(temp5, temp5);
           tempLo = _mm256_castps256_ps128(temp5);
           tempHi = _mm256_extractf128_ps(temp5, 1);
           sseSum = _mm_add_ps(tempLo, tempHi);

          float sum5 = _mm_cvtss_f32(sseSum);

          sum5 += bias2;
          out_to_compare_with_FP[out_subscript5] = Relu_float(sum5);


          temp6 = _mm256_hadd_ps(temp6, temp6);
          temp6 = _mm256_hadd_ps(temp6, temp6);
           tempLo2 = _mm256_castps256_ps128(temp6);
           tempHi2 = _mm256_extractf128_ps(temp6, 1);
           sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum6 = _mm_cvtss_f32(sseSum2);

          sum6 += bias2;
          out_to_compare_with_FP[out_subscript6] = Relu_float(sum6);


          temp7 = _mm256_hadd_ps(temp7, temp7);
          temp7 = _mm256_hadd_ps(temp7, temp7);
           tempLo = _mm256_castps256_ps128(temp7);
           tempHi = _mm256_extractf128_ps(temp7, 1);
           sseSum = _mm_add_ps(tempLo, tempHi);

          float sum7 = _mm_cvtss_f32(sseSum);

          sum7 += bias2;
          out_to_compare_with_FP[out_subscript7] = Relu_float(sum7);


          temp8 = _mm256_hadd_ps(temp8, temp8);
          temp8 = _mm256_hadd_ps(temp8, temp8);
           tempLo2 = _mm256_castps256_ps128(temp8);
           tempHi2 = _mm256_extractf128_ps(temp8, 1);
           sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum8 = _mm_cvtss_f32(sseSum2);

          sum8 += bias2;
          out_to_compare_with_FP[out_subscript8] = Relu_float(sum8);
        }
      }
    }
  }

  printf("\n from optv2 x4 m2 hadd register pressure %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// reduce simd register pressure in x loop
// ~72 GFLOPS, likely similar/ same asm as 2 above
int optimised_layer_v2_unroll_x2m4_hadd_register_pressure_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  float bias, bias2, bias3, bias4;
  __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int m = 0; m < Output_depth_dim; m+=4) { //channels
      bias = bias_array_FP[m];
      bias2 = bias_array_FP[m+1];
      bias3 = bias_array_FP[m+2];
      bias4 = bias_array_FP[m+3];

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
          // temp = 0.0f;

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


                unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript4 = (m+3) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;

                __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                __m256 s2 = _mm256_load_ps(&in_FP[in_subscript2]);

                __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]);
                __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);
                __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript3]);
                __m256 w4 = _mm256_load_ps(&filter_FP[filter_subscript4]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w2));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s2, w2));
                temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s, w3));
                temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s2, w3));
                temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w4));
                temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w4));


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
            x * Output_depth_dim
            + (m+1);
          unsigned long long int out_subscript4 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+1);
          unsigned long long int out_subscript5 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+2);
          unsigned long long int out_subscript6 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+2);
          unsigned long long int out_subscript7 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+3);
          unsigned long long int out_subscript8 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+3);



          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
           tempLo = _mm256_castps256_ps128(temp3);
           tempHi = _mm256_extractf128_ps(temp3, 1);
           sseSum = _mm_add_ps(tempLo, tempHi);

          float sum3 = _mm_cvtss_f32(sseSum);

          sum3 += bias2;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


          temp4 = _mm256_hadd_ps(temp4, temp4);
          temp4 = _mm256_hadd_ps(temp4, temp4);
           tempLo2 = _mm256_castps256_ps128(temp4);
           tempHi2 = _mm256_extractf128_ps(temp4, 1);
           sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum4 = _mm_cvtss_f32(sseSum2);

          sum4 += bias2;
          out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);


          temp5 = _mm256_hadd_ps(temp5, temp5);
          temp5 = _mm256_hadd_ps(temp5, temp5);
           tempLo = _mm256_castps256_ps128(temp5);
           tempHi = _mm256_extractf128_ps(temp5, 1);
           sseSum = _mm_add_ps(tempLo, tempHi);

          float sum5 = _mm_cvtss_f32(sseSum);

          sum5 += bias3;
          out_to_compare_with_FP[out_subscript5] = Relu_float(sum5);


          temp6 = _mm256_hadd_ps(temp6, temp6);
          temp6 = _mm256_hadd_ps(temp6, temp6);
           tempLo2 = _mm256_castps256_ps128(temp6);
           tempHi2 = _mm256_extractf128_ps(temp6, 1);
           sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum6 = _mm_cvtss_f32(sseSum2);

          sum6 += bias3;
          out_to_compare_with_FP[out_subscript6] = Relu_float(sum6);


          temp7 = _mm256_hadd_ps(temp7, temp7);
          temp7 = _mm256_hadd_ps(temp7, temp7);
           tempLo = _mm256_castps256_ps128(temp7);
           tempHi = _mm256_extractf128_ps(temp7, 1);
           sseSum = _mm_add_ps(tempLo, tempHi);

          float sum7 = _mm_cvtss_f32(sseSum);

          sum7 += bias4;
          out_to_compare_with_FP[out_subscript7] = Relu_float(sum7);


          temp8 = _mm256_hadd_ps(temp8, temp8);
          temp8 = _mm256_hadd_ps(temp8, temp8);
           tempLo2 = _mm256_castps256_ps128(temp8);
           tempHi2 = _mm256_extractf128_ps(temp8, 1);
           sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum8 = _mm_cvtss_f32(sseSum2);

          sum8 += bias4;
          out_to_compare_with_FP[out_subscript8] = Relu_float(sum8);
        }
      }
    }
  }

  printf("\n from optv2 x2 m4 hadd - register pressure %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// unrolled x by 3, m by 3 to get most use of simd registers in d loop (3x3 = 15). also have to make sure to stay in loop bounds
// ~74 GFLOPS, extra perf could be offset by increased code size, cmp instructions and worse performing fallback loops?
int optimised_layer_v2_unroll_x3m3_hadd_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  float bias, bias2, bias3;
  __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9;
  // Calculate loop bounds for unrolled loops
  int m_bound = (Output_depth_dim/ 3) * 3;
  int x_bound = (Output_X_dim/ 3) * 3; 

  unsigned int m, x;


  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (m = 0; m < m_bound; m+=3) { //channels
      bias = bias_array_FP[m];
      bias2 = bias_array_FP[m+1];
      bias3 = bias_array_FP[m+2];

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
          // temp = 0.0f;

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


                unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;

                __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                __m256 s2 = _mm256_load_ps(&in_FP[in_subscript2]);
                __m256 s3 = _mm256_load_ps(&in_FP[in_subscript3]);

                __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]); 
                __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);
                __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript3]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s, w2));
                temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s2, w2));
                temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s3, w2));
                temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w3));
                temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w3));
                temp9 = _mm256_add_ps(temp9, _mm256_mul_ps(s3, w3));


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
            x * Output_depth_dim
            + (m+1);
          unsigned long long int out_subscript5 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+1);
          unsigned long long int out_subscript6 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+2) * Output_depth_dim
            + (m+1);
          unsigned long long int out_subscript7 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+2);
          unsigned long long int out_subscript8 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+2);
          unsigned long long int out_subscript9 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+2) * Output_depth_dim
            + (m+2);



          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
          __m128 tempLo3 = _mm256_castps256_ps128(temp3);
          __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
          __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


          temp4 = _mm256_hadd_ps(temp4, temp4);
          temp4 = _mm256_hadd_ps(temp4, temp4);
          __m128 tempLo4 = _mm256_castps256_ps128(temp4);
          __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
          __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

          float sum4 = _mm_cvtss_f32(sseSum4);

          sum4 += bias2;
          out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);


          temp5 = _mm256_hadd_ps(temp5, temp5);
          temp5 = _mm256_hadd_ps(temp5, temp5);
          __m128 tempLo5 = _mm256_castps256_ps128(temp5);
          __m128 tempHi5 = _mm256_extractf128_ps(temp5, 1);
          __m128 sseSum5 = _mm_add_ps(tempLo5, tempHi5);

          float sum5 = _mm_cvtss_f32(sseSum5);

          sum5 += bias2;
          out_to_compare_with_FP[out_subscript5] = Relu_float(sum5);


          temp6 = _mm256_hadd_ps(temp6, temp6);
          temp6 = _mm256_hadd_ps(temp6, temp6);
          __m128 tempLo6 = _mm256_castps256_ps128(temp6);
          __m128 tempHi6 = _mm256_extractf128_ps(temp6, 1);
          __m128 sseSum6 = _mm_add_ps(tempLo6, tempHi6);

          float sum6 = _mm_cvtss_f32(sseSum6);

          sum6 += bias2;
          out_to_compare_with_FP[out_subscript6] = Relu_float(sum6);


          temp7 = _mm256_hadd_ps(temp7, temp7);
          temp7 = _mm256_hadd_ps(temp7, temp7);
          __m128 tempLo7 = _mm256_castps256_ps128(temp7);
          __m128 tempHi7 = _mm256_extractf128_ps(temp7, 1);
          __m128 sseSum7 = _mm_add_ps(tempLo7, tempHi7);

          float sum7 = _mm_cvtss_f32(sseSum7);

          sum7 += bias3;
          out_to_compare_with_FP[out_subscript7] = Relu_float(sum7);


          temp8 = _mm256_hadd_ps(temp8, temp8);
          temp8 = _mm256_hadd_ps(temp8, temp8);
          __m128 tempLo8 = _mm256_castps256_ps128(temp8);
          __m128 tempHi8 = _mm256_extractf128_ps(temp8, 1);
          __m128 sseSum8 = _mm_add_ps(tempLo8, tempHi8);

          float sum8 = _mm_cvtss_f32(sseSum8);

          sum8 += bias3;
          out_to_compare_with_FP[out_subscript8] = Relu_float(sum8);


          temp9 = _mm256_hadd_ps(temp9, temp9);
          temp9 = _mm256_hadd_ps(temp9, temp9);
          __m128 tempLo9 = _mm256_castps256_ps128(temp9);
          __m128 tempHi9 = _mm256_extractf128_ps(temp9, 1);
          __m128 sseSum9 = _mm_add_ps(tempLo9, tempHi9);

          float sum9 = _mm_cvtss_f32(sseSum9);

          sum9 += bias3;
          out_to_compare_with_FP[out_subscript9] = Relu_float(sum9);
        }
        // overflow/ fallback x loop
        for (; x < Output_X_dim; x++) {	//Output Width
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

                unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;

                __m256 s = _mm256_loadu_ps(&in_FP[in_subscript]);

                __m256 w = _mm256_loadu_ps(&filter_FP[filter_subscript]);
                __m256 w2 = _mm256_loadu_ps(&filter_FP[filter_subscript2]);
                __m256 w3 = _mm256_loadu_ps(&filter_FP[filter_subscript3]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s, w2));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w3));

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
            x * Output_depth_dim
            + (m+1);
          unsigned long long int out_subscript3 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+2);

          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias2;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
          __m128 tempLo3 = _mm256_castps256_ps128(temp3);
          __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
          __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias3;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);

        }
      }
    }
    for (; m < Output_depth_dim; m++) {
      bias = bias_array_FP[m];

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (unsigned int x = 0; x < Output_X_dim; x++) {	//Output Width
          temp = _mm256_setzero_ps();

          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + d;

                unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;

                __m256 s = _mm256_loadu_ps(&in_FP[in_subscript]);
                __m256 w = _mm256_loadu_ps(&filter_FP[filter_subscript]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
              }
            }
          }


          unsigned long long int out_subscript = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + m;

          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);

        }
      }
    }
  }

  printf("\n from optv2 x3 m3 hadd %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// optimised fallback loops of above
// ~76 GFLOPS
int optimised_layer_v2_unroll_x3m3_hadd_opt_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  float bias, bias2, bias3;
  __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9;
  // Calculate loop bounds for unrolled loops
  int m_bound = (Output_depth_dim/ 3) * 3;
  int x_bound = (Output_X_dim/ 3) * 3; 

  unsigned int m, x;


  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (m = 0; m < m_bound; m+=3) { //channels
      bias = bias_array_FP[m];
      bias2 = bias_array_FP[m+1];
      bias3 = bias_array_FP[m+2];

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
          // temp = 0.0f;

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


                unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;

                __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                __m256 s2 = _mm256_load_ps(&in_FP[in_subscript2]);
                __m256 s3 = _mm256_load_ps(&in_FP[in_subscript3]);

                __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]); 
                __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);
                __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript3]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s, w2));
                temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s2, w2));
                temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s3, w2));
                temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w3));
                temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w3));
                temp9 = _mm256_add_ps(temp9, _mm256_mul_ps(s3, w3));


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
            x * Output_depth_dim
            + (m+1);
          unsigned long long int out_subscript5 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+1);
          unsigned long long int out_subscript6 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+2) * Output_depth_dim
            + (m+1);
          unsigned long long int out_subscript7 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+2);
          unsigned long long int out_subscript8 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+2);
          unsigned long long int out_subscript9 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+2) * Output_depth_dim
            + (m+2);



          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
          __m128 tempLo3 = _mm256_castps256_ps128(temp3);
          __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
          __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


          temp4 = _mm256_hadd_ps(temp4, temp4);
          temp4 = _mm256_hadd_ps(temp4, temp4);
          __m128 tempLo4 = _mm256_castps256_ps128(temp4);
          __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
          __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

          float sum4 = _mm_cvtss_f32(sseSum4);

          sum4 += bias2;
          out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);


          temp5 = _mm256_hadd_ps(temp5, temp5);
          temp5 = _mm256_hadd_ps(temp5, temp5);
          __m128 tempLo5 = _mm256_castps256_ps128(temp5);
          __m128 tempHi5 = _mm256_extractf128_ps(temp5, 1);
          __m128 sseSum5 = _mm_add_ps(tempLo5, tempHi5);

          float sum5 = _mm_cvtss_f32(sseSum5);

          sum5 += bias2;
          out_to_compare_with_FP[out_subscript5] = Relu_float(sum5);


          temp6 = _mm256_hadd_ps(temp6, temp6);
          temp6 = _mm256_hadd_ps(temp6, temp6);
          __m128 tempLo6 = _mm256_castps256_ps128(temp6);
          __m128 tempHi6 = _mm256_extractf128_ps(temp6, 1);
          __m128 sseSum6 = _mm_add_ps(tempLo6, tempHi6);

          float sum6 = _mm_cvtss_f32(sseSum6);

          sum6 += bias2;
          out_to_compare_with_FP[out_subscript6] = Relu_float(sum6);


          temp7 = _mm256_hadd_ps(temp7, temp7);
          temp7 = _mm256_hadd_ps(temp7, temp7);
          __m128 tempLo7 = _mm256_castps256_ps128(temp7);
          __m128 tempHi7 = _mm256_extractf128_ps(temp7, 1);
          __m128 sseSum7 = _mm_add_ps(tempLo7, tempHi7);

          float sum7 = _mm_cvtss_f32(sseSum7);

          sum7 += bias3;
          out_to_compare_with_FP[out_subscript7] = Relu_float(sum7);


          temp8 = _mm256_hadd_ps(temp8, temp8);
          temp8 = _mm256_hadd_ps(temp8, temp8);
          __m128 tempLo8 = _mm256_castps256_ps128(temp8);
          __m128 tempHi8 = _mm256_extractf128_ps(temp8, 1);
          __m128 sseSum8 = _mm_add_ps(tempLo8, tempHi8);

          float sum8 = _mm_cvtss_f32(sseSum8);

          sum8 += bias3;
          out_to_compare_with_FP[out_subscript8] = Relu_float(sum8);


          temp9 = _mm256_hadd_ps(temp9, temp9);
          temp9 = _mm256_hadd_ps(temp9, temp9);
          __m128 tempLo9 = _mm256_castps256_ps128(temp9);
          __m128 tempHi9 = _mm256_extractf128_ps(temp9, 1);
          __m128 sseSum9 = _mm_add_ps(tempLo9, tempHi9);

          float sum9 = _mm_cvtss_f32(sseSum9);

          sum9 += bias3;
          out_to_compare_with_FP[out_subscript9] = Relu_float(sum9);
        }
        // overflow/ fallback x loop
        for (; x < Output_X_dim; x++) {	//Output Width
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

                unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;

                __m256 s = _mm256_loadu_ps(&in_FP[in_subscript]);

                __m256 w = _mm256_loadu_ps(&filter_FP[filter_subscript]);
                __m256 w2 = _mm256_loadu_ps(&filter_FP[filter_subscript2]);
                __m256 w3 = _mm256_loadu_ps(&filter_FP[filter_subscript3]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s, w2));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w3));

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
            x * Output_depth_dim
            + (m+1);
          unsigned long long int out_subscript3 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+2);

          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias2;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
          __m128 tempLo3 = _mm256_castps256_ps128(temp3);
          __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
          __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias3;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);

        }
      }
    }
    for (; m < Output_depth_dim; m++) {
      bias = bias_array_FP[m];

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (unsigned int x = 0; x < Output_X_dim; x+=4) {	//Output Width
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

                __m256 w = _mm256_loadu_ps(&filter_FP[filter_subscript]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s4, w));
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

          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
          __m128 tempLo3 = _mm256_castps256_ps128(temp3);
          __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
          __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


          temp4 = _mm256_hadd_ps(temp4, temp4);
          temp4 = _mm256_hadd_ps(temp4, temp4);
          __m128 tempLo4 = _mm256_castps256_ps128(temp4);
          __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
          __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

          float sum4 = _mm_cvtss_f32(sseSum4);

          sum4 += bias;
          out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);

        }
      }
    }
  }

  printf("\n from optv2 x3 m3 hadd opt %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}






// v3 - unroll d loop
// ***** using optimised_layer_v2_unroll_x3m3_hadd_opt_FP() as base for v3 *****


// unroll d by factor of 2 (d16)
// ~58 GFLOPS, maybe overflow instruction cache?
int optimised_layer_v3_x3m3_unroll_d16_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  float bias, bias2, bias3;
  __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9;
  // Calculate loop bounds for unrolled loops
  int m_bound = (Output_depth_dim/ 3) * 3;
  int x_bound = (Output_X_dim/ 3) * 3; 

  unsigned int m, x;


  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (m = 0; m < m_bound; m+=3) { //channels
      bias = bias_array_FP[m];
      bias2 = bias_array_FP[m+1];
      bias3 = bias_array_FP[m+2];

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
          // temp = 0.0f;

          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=16) {

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


                unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;

                __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                __m256 s2 = _mm256_load_ps(&in_FP[in_subscript2]);
                __m256 s3 = _mm256_load_ps(&in_FP[in_subscript3]);

                __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]); 
                __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);
                __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript3]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s, w2));
                temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s2, w2));
                temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s3, w2));
                temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w3));
                temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w3));
                temp9 = _mm256_add_ps(temp9, _mm256_mul_ps(s3, w3));


                // d+8 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+8);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+8);
                in_subscript3 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+2) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+8);


                filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+8);
                filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+8);
                filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+8);

                s = _mm256_load_ps(&in_FP[in_subscript]);
                s2 = _mm256_load_ps(&in_FP[in_subscript2]);
                s3 = _mm256_load_ps(&in_FP[in_subscript3]);

                w = _mm256_load_ps(&filter_FP[filter_subscript]); 
                w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);
                w3 = _mm256_load_ps(&filter_FP[filter_subscript3]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s, w2));
                temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s2, w2));
                temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s3, w2));
                temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w3));
                temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w3));
                temp9 = _mm256_add_ps(temp9, _mm256_mul_ps(s3, w3));

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
            + (m+1);
          unsigned long long int out_subscript5 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+1);
          unsigned long long int out_subscript6 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+2) * Output_depth_dim
            + (m+1);
          unsigned long long int out_subscript7 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+2);
          unsigned long long int out_subscript8 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+2);
          unsigned long long int out_subscript9 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+2) * Output_depth_dim
            + (m+2);



          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
          __m128 tempLo3 = _mm256_castps256_ps128(temp3);
          __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
          __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


          temp4 = _mm256_hadd_ps(temp4, temp4);
          temp4 = _mm256_hadd_ps(temp4, temp4);
          __m128 tempLo4 = _mm256_castps256_ps128(temp4);
          __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
          __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

          float sum4 = _mm_cvtss_f32(sseSum4);

          sum4 += bias2;
          out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);


          temp5 = _mm256_hadd_ps(temp5, temp5);
          temp5 = _mm256_hadd_ps(temp5, temp5);
          __m128 tempLo5 = _mm256_castps256_ps128(temp5);
          __m128 tempHi5 = _mm256_extractf128_ps(temp5, 1);
          __m128 sseSum5 = _mm_add_ps(tempLo5, tempHi5);

          float sum5 = _mm_cvtss_f32(sseSum5);

          sum5 += bias2;
          out_to_compare_with_FP[out_subscript5] = Relu_float(sum5);


          temp6 = _mm256_hadd_ps(temp6, temp6);
          temp6 = _mm256_hadd_ps(temp6, temp6);
          __m128 tempLo6 = _mm256_castps256_ps128(temp6);
          __m128 tempHi6 = _mm256_extractf128_ps(temp6, 1);
          __m128 sseSum6 = _mm_add_ps(tempLo6, tempHi6);

          float sum6 = _mm_cvtss_f32(sseSum6);

          sum6 += bias2;
          out_to_compare_with_FP[out_subscript6] = Relu_float(sum6);


          temp7 = _mm256_hadd_ps(temp7, temp7);
          temp7 = _mm256_hadd_ps(temp7, temp7);
          __m128 tempLo7 = _mm256_castps256_ps128(temp7);
          __m128 tempHi7 = _mm256_extractf128_ps(temp7, 1);
          __m128 sseSum7 = _mm_add_ps(tempLo7, tempHi7);

          float sum7 = _mm_cvtss_f32(sseSum7);

          sum7 += bias3;
          out_to_compare_with_FP[out_subscript7] = Relu_float(sum7);


          temp8 = _mm256_hadd_ps(temp8, temp8);
          temp8 = _mm256_hadd_ps(temp8, temp8);
          __m128 tempLo8 = _mm256_castps256_ps128(temp8);
          __m128 tempHi8 = _mm256_extractf128_ps(temp8, 1);
          __m128 sseSum8 = _mm_add_ps(tempLo8, tempHi8);

          float sum8 = _mm_cvtss_f32(sseSum8);

          sum8 += bias3;
          out_to_compare_with_FP[out_subscript8] = Relu_float(sum8);


          temp9 = _mm256_hadd_ps(temp9, temp9);
          temp9 = _mm256_hadd_ps(temp9, temp9);
          __m128 tempLo9 = _mm256_castps256_ps128(temp9);
          __m128 tempHi9 = _mm256_extractf128_ps(temp9, 1);
          __m128 sseSum9 = _mm_add_ps(tempLo9, tempHi9);

          float sum9 = _mm_cvtss_f32(sseSum9);

          sum9 += bias3;
          out_to_compare_with_FP[out_subscript9] = Relu_float(sum9);
        }
        // overflow/ fallback x loop
        for (; x < Output_X_dim; x++) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=16) {

                unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + d;

                unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;

                __m256 s = _mm256_loadu_ps(&in_FP[in_subscript]);

                __m256 w = _mm256_loadu_ps(&filter_FP[filter_subscript]);
                __m256 w2 = _mm256_loadu_ps(&filter_FP[filter_subscript2]);
                __m256 w3 = _mm256_loadu_ps(&filter_FP[filter_subscript3]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s, w2));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w3));


                // d+8 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+8);

                filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+8);
                filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+8);
                filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+8);

                s = _mm256_loadu_ps(&in_FP[in_subscript]);

                w = _mm256_loadu_ps(&filter_FP[filter_subscript]);
                w2 = _mm256_loadu_ps(&filter_FP[filter_subscript2]);
                w3 = _mm256_loadu_ps(&filter_FP[filter_subscript3]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s, w2));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w3));

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
            + (m+1);
          unsigned long long int out_subscript3 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+2);

          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias2;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
          __m128 tempLo3 = _mm256_castps256_ps128(temp3);
          __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
          __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias3;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);

        }
      }
    }
    for (; m < Output_depth_dim; m++) {
      bias = bias_array_FP[m];

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (unsigned int x = 0; x < Output_X_dim; x+=4) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          temp4 = _mm256_setzero_ps();

          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=16) {

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

                __m256 w = _mm256_loadu_ps(&filter_FP[filter_subscript]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s4, w));


                // d+8 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+8);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+8);
                in_subscript3 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+2) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+8);
                in_subscript4 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+3) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+8);

                filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+8);

                s = _mm256_loadu_ps(&in_FP[in_subscript]);
                s2 = _mm256_loadu_ps(&in_FP[in_subscript2]);
                s3 = _mm256_loadu_ps(&in_FP[in_subscript3]);
                s4 = _mm256_loadu_ps(&in_FP[in_subscript4]);

                w = _mm256_loadu_ps(&filter_FP[filter_subscript]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s4, w));
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

          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
          __m128 tempLo3 = _mm256_castps256_ps128(temp3);
          __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
          __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


          temp4 = _mm256_hadd_ps(temp4, temp4);
          temp4 = _mm256_hadd_ps(temp4, temp4);
          __m128 tempLo4 = _mm256_castps256_ps128(temp4);
          __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
          __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

          float sum4 = _mm_cvtss_f32(sseSum4);

          sum4 += bias;
          out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);

        }
      }
    }
  }

  printf("\n from optv3 x3 m3 d16 v1  %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// same as above, for x3m3 non-opt
// ~63 GFLOPS, better perf than above, making instruction cache miss perf more likely? profile with cachegrind/ likwid to find out more
int optimised_layer_v3_x3m3_unroll_d16_v2_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  float bias, bias2, bias3;
  __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9;
  // Calculate loop bounds for unrolled loops
  int m_bound = (Output_depth_dim/ 3) * 3;
  int x_bound = (Output_X_dim/ 3) * 3; 

  unsigned int m, x;


  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (m = 0; m < m_bound; m+=3) { //channels
      bias = bias_array_FP[m];
      bias2 = bias_array_FP[m+1];
      bias3 = bias_array_FP[m+2];

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
          // temp = 0.0f;

          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=16) {

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


                unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;

                __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                __m256 s2 = _mm256_load_ps(&in_FP[in_subscript2]);
                __m256 s3 = _mm256_load_ps(&in_FP[in_subscript3]);

                __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]); 
                __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);
                __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript3]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s, w2));
                temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s2, w2));
                temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s3, w2));
                temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w3));
                temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w3));
                temp9 = _mm256_add_ps(temp9, _mm256_mul_ps(s3, w3));


                // d+8 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+8);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+8);
                in_subscript3 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+2) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+8);


                filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+8);
                filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+8);
                filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+8);

                s = _mm256_load_ps(&in_FP[in_subscript]);
                s2 = _mm256_load_ps(&in_FP[in_subscript2]);
                s3 = _mm256_load_ps(&in_FP[in_subscript3]);

                w = _mm256_load_ps(&filter_FP[filter_subscript]); 
                w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);
                w3 = _mm256_load_ps(&filter_FP[filter_subscript3]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s, w2));
                temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s2, w2));
                temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s3, w2));
                temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w3));
                temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w3));
                temp9 = _mm256_add_ps(temp9, _mm256_mul_ps(s3, w3));

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
            + (m+1);
          unsigned long long int out_subscript5 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+1);
          unsigned long long int out_subscript6 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+2) * Output_depth_dim
            + (m+1);
          unsigned long long int out_subscript7 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+2);
          unsigned long long int out_subscript8 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+2);
          unsigned long long int out_subscript9 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+2) * Output_depth_dim
            + (m+2);



          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
          __m128 tempLo3 = _mm256_castps256_ps128(temp3);
          __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
          __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


          temp4 = _mm256_hadd_ps(temp4, temp4);
          temp4 = _mm256_hadd_ps(temp4, temp4);
          __m128 tempLo4 = _mm256_castps256_ps128(temp4);
          __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
          __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

          float sum4 = _mm_cvtss_f32(sseSum4);

          sum4 += bias2;
          out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);


          temp5 = _mm256_hadd_ps(temp5, temp5);
          temp5 = _mm256_hadd_ps(temp5, temp5);
          __m128 tempLo5 = _mm256_castps256_ps128(temp5);
          __m128 tempHi5 = _mm256_extractf128_ps(temp5, 1);
          __m128 sseSum5 = _mm_add_ps(tempLo5, tempHi5);

          float sum5 = _mm_cvtss_f32(sseSum5);

          sum5 += bias2;
          out_to_compare_with_FP[out_subscript5] = Relu_float(sum5);


          temp6 = _mm256_hadd_ps(temp6, temp6);
          temp6 = _mm256_hadd_ps(temp6, temp6);
          __m128 tempLo6 = _mm256_castps256_ps128(temp6);
          __m128 tempHi6 = _mm256_extractf128_ps(temp6, 1);
          __m128 sseSum6 = _mm_add_ps(tempLo6, tempHi6);

          float sum6 = _mm_cvtss_f32(sseSum6);

          sum6 += bias2;
          out_to_compare_with_FP[out_subscript6] = Relu_float(sum6);


          temp7 = _mm256_hadd_ps(temp7, temp7);
          temp7 = _mm256_hadd_ps(temp7, temp7);
          __m128 tempLo7 = _mm256_castps256_ps128(temp7);
          __m128 tempHi7 = _mm256_extractf128_ps(temp7, 1);
          __m128 sseSum7 = _mm_add_ps(tempLo7, tempHi7);

          float sum7 = _mm_cvtss_f32(sseSum7);

          sum7 += bias3;
          out_to_compare_with_FP[out_subscript7] = Relu_float(sum7);


          temp8 = _mm256_hadd_ps(temp8, temp8);
          temp8 = _mm256_hadd_ps(temp8, temp8);
          __m128 tempLo8 = _mm256_castps256_ps128(temp8);
          __m128 tempHi8 = _mm256_extractf128_ps(temp8, 1);
          __m128 sseSum8 = _mm_add_ps(tempLo8, tempHi8);

          float sum8 = _mm_cvtss_f32(sseSum8);

          sum8 += bias3;
          out_to_compare_with_FP[out_subscript8] = Relu_float(sum8);


          temp9 = _mm256_hadd_ps(temp9, temp9);
          temp9 = _mm256_hadd_ps(temp9, temp9);
          __m128 tempLo9 = _mm256_castps256_ps128(temp9);
          __m128 tempHi9 = _mm256_extractf128_ps(temp9, 1);
          __m128 sseSum9 = _mm_add_ps(tempLo9, tempHi9);

          float sum9 = _mm_cvtss_f32(sseSum9);

          sum9 += bias3;
          out_to_compare_with_FP[out_subscript9] = Relu_float(sum9);
        }
        // overflow/ fallback x loop
        for (; x < Output_X_dim; x++) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=16) {

                unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + d;

                unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;

                __m256 s = _mm256_loadu_ps(&in_FP[in_subscript]);

                __m256 w = _mm256_loadu_ps(&filter_FP[filter_subscript]);
                __m256 w2 = _mm256_loadu_ps(&filter_FP[filter_subscript2]);
                __m256 w3 = _mm256_loadu_ps(&filter_FP[filter_subscript3]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s, w2));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w3));


                // d+8 ------------------------------------------------------------
                // |
                // |
                // v


                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+8);

                filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+8);
                filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+8);
                filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+8);

                s = _mm256_loadu_ps(&in_FP[in_subscript]);

                w = _mm256_loadu_ps(&filter_FP[filter_subscript]);
                w2 = _mm256_loadu_ps(&filter_FP[filter_subscript2]);
                w3 = _mm256_loadu_ps(&filter_FP[filter_subscript3]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s, w2));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w3));

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
            + (m+1);
          unsigned long long int out_subscript3 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+2);

          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias2;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
          __m128 tempLo3 = _mm256_castps256_ps128(temp3);
          __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
          __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias3;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);

        }
      }
    }
    for (; m < Output_depth_dim; m++) {
      bias = bias_array_FP[m];

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (unsigned int x = 0; x < Output_X_dim; x++) {	//Output Width
          temp = _mm256_setzero_ps();

          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=16) {

                unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + d;

                unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;

                __m256 s = _mm256_loadu_ps(&in_FP[in_subscript]);
                __m256 w = _mm256_loadu_ps(&filter_FP[filter_subscript]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));


                // d+8 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+8);

                filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+8);

                s = _mm256_loadu_ps(&in_FP[in_subscript]);
                w = _mm256_loadu_ps(&filter_FP[filter_subscript]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
              }
            }
          }


          unsigned long long int out_subscript = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + m;

          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);

        }
      }
    }
  }

  printf("\n from optv3 x3 m3 hadd d16 v2 %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// unroll d by factor of 2 (d16) (non register pressure)
// ~60 GFLOPS, 
int optimised_layer_v3_x2m4_unroll_d16_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  float bias, bias2, bias3, bias4;
  __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int m = 0; m < Output_depth_dim; m+=4) { //channels
      bias = bias_array_FP[m];
      bias2 = bias_array_FP[m+1];
      bias3 = bias_array_FP[m+2];
      bias4 = bias_array_FP[m+3];

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
          // temp = 0.0f;

          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=16) {

                unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + d;
                unsigned long long int in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + d;


                unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript4 = (m+3) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;

                __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                __m256 s2 = _mm256_load_ps(&in_FP[in_subscript2]);

                __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]);
                __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);
                __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript3]);
                __m256 w4 = _mm256_load_ps(&filter_FP[filter_subscript4]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w2));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s2, w2));
                temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s, w3));
                temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s2, w3));
                temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w4));
                temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w4));


                // d+8 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+8);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+8);


                filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+8);
                filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+8);
                filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+8);
                filter_subscript4 = (m+3) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+8);

                s = _mm256_load_ps(&in_FP[in_subscript]);
                s2 = _mm256_load_ps(&in_FP[in_subscript2]);

                w = _mm256_load_ps(&filter_FP[filter_subscript]);
                w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);
                w3 = _mm256_load_ps(&filter_FP[filter_subscript3]);
                w4 = _mm256_load_ps(&filter_FP[filter_subscript4]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w2));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s2, w2));
                temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s, w3));
                temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s2, w3));
                temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w4));
                temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w4));
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
            + (m+1);
          unsigned long long int out_subscript4 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+1);
          unsigned long long int out_subscript5 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+2);
          unsigned long long int out_subscript6 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+2);
          unsigned long long int out_subscript7 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+3);
          unsigned long long int out_subscript8 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+3);



          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
           tempLo = _mm256_castps256_ps128(temp3);
           tempHi = _mm256_extractf128_ps(temp3, 1);
           sseSum = _mm_add_ps(tempLo, tempHi);

          float sum3 = _mm_cvtss_f32(sseSum);

          sum3 += bias2;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


          temp4 = _mm256_hadd_ps(temp4, temp4);
          temp4 = _mm256_hadd_ps(temp4, temp4);
           tempLo2 = _mm256_castps256_ps128(temp4);
           tempHi2 = _mm256_extractf128_ps(temp4, 1);
           sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum4 = _mm_cvtss_f32(sseSum2);

          sum4 += bias2;
          out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);


          temp5 = _mm256_hadd_ps(temp5, temp5);
          temp5 = _mm256_hadd_ps(temp5, temp5);
           tempLo = _mm256_castps256_ps128(temp5);
           tempHi = _mm256_extractf128_ps(temp5, 1);
           sseSum = _mm_add_ps(tempLo, tempHi);

          float sum5 = _mm_cvtss_f32(sseSum);

          sum5 += bias3;
          out_to_compare_with_FP[out_subscript5] = Relu_float(sum5);


          temp6 = _mm256_hadd_ps(temp6, temp6);
          temp6 = _mm256_hadd_ps(temp6, temp6);
           tempLo2 = _mm256_castps256_ps128(temp6);
           tempHi2 = _mm256_extractf128_ps(temp6, 1);
           sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum6 = _mm_cvtss_f32(sseSum2);

          sum6 += bias3;
          out_to_compare_with_FP[out_subscript6] = Relu_float(sum6);


          temp7 = _mm256_hadd_ps(temp7, temp7);
          temp7 = _mm256_hadd_ps(temp7, temp7);
           tempLo = _mm256_castps256_ps128(temp7);
           tempHi = _mm256_extractf128_ps(temp7, 1);
           sseSum = _mm_add_ps(tempLo, tempHi);

          float sum7 = _mm_cvtss_f32(sseSum);

          sum7 += bias4;
          out_to_compare_with_FP[out_subscript7] = Relu_float(sum7);


          temp8 = _mm256_hadd_ps(temp8, temp8);
          temp8 = _mm256_hadd_ps(temp8, temp8);
           tempLo2 = _mm256_castps256_ps128(temp8);
           tempHi2 = _mm256_extractf128_ps(temp8, 1);
           sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum8 = _mm_cvtss_f32(sseSum2);

          sum8 += bias4;
          out_to_compare_with_FP[out_subscript8] = Relu_float(sum8);
        }
      }
    }
  }

  printf("\n from optv3 x2 m4 d16 %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// unroll d by factor of 2 (d16) (non register pressure)
// ~59 GFLOPS
int optimised_layer_v3_x4m2_unroll_d16_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  float bias, bias2;
  __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int m = 0; m < Output_depth_dim; m+=2) { //channels
      bias = bias_array_FP[m];
      bias2 = bias_array_FP[m+1];

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
          // temp = 0.0f;

          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=16) {

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
                unsigned long long int filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;

                __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                __m256 s2 = _mm256_load_ps(&in_FP[in_subscript2]);
                __m256 s3 = _mm256_load_ps(&in_FP[in_subscript3]);
                __m256 s4 = _mm256_load_ps(&in_FP[in_subscript4]);

                __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]);
                __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s4, w));
                temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s, w2));
                temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s2, w2));
                temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s3, w2));
                temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s4, w2));


                // d+8 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+8);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+8);
                in_subscript3 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+2) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+8);
                in_subscript4 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+3) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+8);


                filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+8);
                filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+8);

                s = _mm256_load_ps(&in_FP[in_subscript]);
                s2 = _mm256_load_ps(&in_FP[in_subscript2]);
                s3 = _mm256_load_ps(&in_FP[in_subscript3]);
                s4 = _mm256_load_ps(&in_FP[in_subscript4]);

                w = _mm256_load_ps(&filter_FP[filter_subscript]);
                w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s4, w));
                temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s, w2));
                temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s2, w2));
                temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s3, w2));
                temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s4, w2));
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
            + (m+1);
          unsigned long long int out_subscript6 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+1);
          unsigned long long int out_subscript7 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+2) * Output_depth_dim
            + (m+1);
          unsigned long long int out_subscript8 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+3) * Output_depth_dim
            + (m+1);



          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
          __m128 tempLo3 = _mm256_castps256_ps128(temp3);
          __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
          __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


          temp4 = _mm256_hadd_ps(temp4, temp4);
          temp4 = _mm256_hadd_ps(temp4, temp4);
          __m128 tempLo4 = _mm256_castps256_ps128(temp4);
          __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
          __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

          float sum4 = _mm_cvtss_f32(sseSum4);

          sum4 += bias;
          out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);


          temp5 = _mm256_hadd_ps(temp5, temp5);
          temp5 = _mm256_hadd_ps(temp5, temp5);
          __m128 tempLo5 = _mm256_castps256_ps128(temp5);
          __m128 tempHi5 = _mm256_extractf128_ps(temp5, 1);
          __m128 sseSum5 = _mm_add_ps(tempLo5, tempHi5);

          float sum5 = _mm_cvtss_f32(sseSum5);

          sum5 += bias2;
          out_to_compare_with_FP[out_subscript5] = Relu_float(sum5);


          temp6 = _mm256_hadd_ps(temp6, temp6);
          temp6 = _mm256_hadd_ps(temp6, temp6);
          __m128 tempLo6 = _mm256_castps256_ps128(temp6);
          __m128 tempHi6 = _mm256_extractf128_ps(temp6, 1);
          __m128 sseSum6 = _mm_add_ps(tempLo6, tempHi6);

          float sum6 = _mm_cvtss_f32(sseSum6);

          sum6 += bias2;
          out_to_compare_with_FP[out_subscript6] = Relu_float(sum6);


          temp7 = _mm256_hadd_ps(temp7, temp7);
          temp7 = _mm256_hadd_ps(temp7, temp7);
          __m128 tempLo7 = _mm256_castps256_ps128(temp7);
          __m128 tempHi7 = _mm256_extractf128_ps(temp7, 1);
          __m128 sseSum7 = _mm_add_ps(tempLo7, tempHi7);

          float sum7 = _mm_cvtss_f32(sseSum7);

          sum7 += bias2;
          out_to_compare_with_FP[out_subscript7] = Relu_float(sum7);


          temp8 = _mm256_hadd_ps(temp8, temp8);
          temp8 = _mm256_hadd_ps(temp8, temp8);
          __m128 tempLo8 = _mm256_castps256_ps128(temp8);
          __m128 tempHi8 = _mm256_extractf128_ps(temp8, 1);
          __m128 sseSum8 = _mm_add_ps(tempLo8, tempHi8);

          float sum8 = _mm_cvtss_f32(sseSum8);

          sum8 += bias2;
          out_to_compare_with_FP[out_subscript8] = Relu_float(sum8);
        }
      }
    }
  }

  printf("\n from optv3 x4 m2 d16 %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// unroll d by factor of 2 (d16) (hadd instructions)
// ~58 GFLOPS, 
int optimised_layer_v3_x2m2_unroll_d16_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  float bias, bias2;
  __m256 temp, temp2, temp3, temp4;

  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int m = 0; m < Output_depth_dim; m+=2) { //channels
      bias = bias_array_FP[m];
      bias2 = bias_array_FP[m+1];

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (unsigned int x = 0; x < Output_X_dim; x+=2) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          temp4 = _mm256_setzero_ps();
          // temp = 0.0f;

          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=16) {

                unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + d;
                unsigned long long int in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + d;


                unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;

                __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                __m256 s2 = _mm256_load_ps(&in_FP[in_subscript2]);

                __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]);
                __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w2));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s2, w2));


                // d+8 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+8);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+8);


                filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+8);
                filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+8);

                s = _mm256_load_ps(&in_FP[in_subscript]);
                s2 = _mm256_load_ps(&in_FP[in_subscript2]);

                w = _mm256_load_ps(&filter_FP[filter_subscript]);
                w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w2));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s2, w2));

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
            + (m+1);
          unsigned long long int out_subscript4 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+1);



          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
          __m128 tempLo3 = _mm256_castps256_ps128(temp3);
          __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
          __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias2;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


          temp4 = _mm256_hadd_ps(temp4, temp4);
          temp4 = _mm256_hadd_ps(temp4, temp4);
          __m128 tempLo4 = _mm256_castps256_ps128(temp4);
          __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
          __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

          float sum4 = _mm_cvtss_f32(sseSum4);

          sum4 += bias2;
          out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);

        }
      }
    }
  }

  printf("\n from optv3 x2 m2 hadd d16 %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// same as above, but increased simd register usage in d loop (was 8, now 16)
// ~41 GFLOPS, 
int optimised_layer_v3_x2m2_unroll_d16_opt_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  float bias, bias2;
  __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int m = 0; m < Output_depth_dim; m+=2) { //channels
      bias = bias_array_FP[m];
      bias2 = bias_array_FP[m+1];

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
          // temp = 0.0f;

          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=16) {

                unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + d;
                unsigned long long int in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + d;


                unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;

                __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                __m256 s2 = _mm256_load_ps(&in_FP[in_subscript2]);

                __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]);
                __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w2));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s2, w2));


                // d+8 ------------------------------------------------------------
                // |
                // |
                // v

                unsigned long long int in_subscript3 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+8);
                unsigned long long int in_subscript4 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+8);


                unsigned long long int filter_subscript3 = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+8);
                unsigned long long int filter_subscript4 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+8);

                __m256 s3 = _mm256_load_ps(&in_FP[in_subscript3]);
                __m256 s4 = _mm256_load_ps(&in_FP[in_subscript4]);

                __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript3]);
                __m256 w4 = _mm256_load_ps(&filter_FP[filter_subscript4]);

                temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s3, w3));
                temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s4, w3));
                temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s3, w4));
                temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s4, w4));
              }
            }
          }

          temp = _mm256_add_ps(temp, temp5);
          temp2 = _mm256_add_ps(temp2, temp6);
          temp3 = _mm256_add_ps(temp3, temp7);
          temp4 = _mm256_add_ps(temp4, temp8);

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
            + (m+1);
          unsigned long long int out_subscript4 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+1);



          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
          __m128 tempLo3 = _mm256_castps256_ps128(temp3);
          __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
          __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias2;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


          temp4 = _mm256_hadd_ps(temp4, temp4);
          temp4 = _mm256_hadd_ps(temp4, temp4);
          __m128 tempLo4 = _mm256_castps256_ps128(temp4);
          __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
          __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

          float sum4 = _mm_cvtss_f32(sseSum4);

          sum4 += bias2;
          out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);

        }
      }
    }
  }

  printf("\n from optv3 x2 m2 hadd d16 opt %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// unroll d by factor of 4 (d32)
// 69 GFLOPS, investigate perf difference
int optimised_layer_v3_x3m3_unroll_d32_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  float bias, bias2, bias3;
  __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9;
  // Calculate loop bounds for unrolled loops
  int m_bound = (Output_depth_dim/ 3) * 3;
  int x_bound = (Output_X_dim/ 3) * 3; 

  unsigned int m, x;


  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (m = 0; m < m_bound; m+=3) { //channels
      bias = bias_array_FP[m];
      bias2 = bias_array_FP[m+1];
      bias3 = bias_array_FP[m+2];

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
          // temp = 0.0f;

          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=32) {

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


                unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;

                __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                __m256 s2 = _mm256_load_ps(&in_FP[in_subscript2]);
                __m256 s3 = _mm256_load_ps(&in_FP[in_subscript3]);

                __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]); 
                __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);
                __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript3]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s, w2));
                temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s2, w2));
                temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s3, w2));
                temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w3));
                temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w3));
                temp9 = _mm256_add_ps(temp9, _mm256_mul_ps(s3, w3));


                // d+8 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+8);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+8);
                in_subscript3 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+2) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+8);


                filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+8);
                filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+8);
                filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+8);

                s = _mm256_load_ps(&in_FP[in_subscript]);
                s2 = _mm256_load_ps(&in_FP[in_subscript2]);
                s3 = _mm256_load_ps(&in_FP[in_subscript3]);

                w = _mm256_load_ps(&filter_FP[filter_subscript]); 
                w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);
                w3 = _mm256_load_ps(&filter_FP[filter_subscript3]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s, w2));
                temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s2, w2));
                temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s3, w2));
                temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w3));
                temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w3));
                temp9 = _mm256_add_ps(temp9, _mm256_mul_ps(s3, w3));


                // d+16 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+16);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+16);
                in_subscript3 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+2) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+16);


                filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+16);
                filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+16);
                filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+16);

                s = _mm256_load_ps(&in_FP[in_subscript]);
                s2 = _mm256_load_ps(&in_FP[in_subscript2]);
                s3 = _mm256_load_ps(&in_FP[in_subscript3]);

                w = _mm256_load_ps(&filter_FP[filter_subscript]); 
                w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);
                w3 = _mm256_load_ps(&filter_FP[filter_subscript3]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s, w2));
                temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s2, w2));
                temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s3, w2));
                temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w3));
                temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w3));
                temp9 = _mm256_add_ps(temp9, _mm256_mul_ps(s3, w3));


                // d+24 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+24);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+24);
                in_subscript3 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+2) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+24);


                filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+24);
                filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+24);
                filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+24);

                s = _mm256_load_ps(&in_FP[in_subscript]);
                s2 = _mm256_load_ps(&in_FP[in_subscript2]);
                s3 = _mm256_load_ps(&in_FP[in_subscript3]);

                w = _mm256_load_ps(&filter_FP[filter_subscript]); 
                w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);
                w3 = _mm256_load_ps(&filter_FP[filter_subscript3]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s, w2));
                temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s2, w2));
                temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s3, w2));
                temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w3));
                temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w3));
                temp9 = _mm256_add_ps(temp9, _mm256_mul_ps(s3, w3));

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
            + (m+1);
          unsigned long long int out_subscript5 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+1);
          unsigned long long int out_subscript6 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+2) * Output_depth_dim
            + (m+1);
          unsigned long long int out_subscript7 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+2);
          unsigned long long int out_subscript8 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+2);
          unsigned long long int out_subscript9 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+2) * Output_depth_dim
            + (m+2);



          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
          __m128 tempLo3 = _mm256_castps256_ps128(temp3);
          __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
          __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


          temp4 = _mm256_hadd_ps(temp4, temp4);
          temp4 = _mm256_hadd_ps(temp4, temp4);
          __m128 tempLo4 = _mm256_castps256_ps128(temp4);
          __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
          __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

          float sum4 = _mm_cvtss_f32(sseSum4);

          sum4 += bias2;
          out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);


          temp5 = _mm256_hadd_ps(temp5, temp5);
          temp5 = _mm256_hadd_ps(temp5, temp5);
          __m128 tempLo5 = _mm256_castps256_ps128(temp5);
          __m128 tempHi5 = _mm256_extractf128_ps(temp5, 1);
          __m128 sseSum5 = _mm_add_ps(tempLo5, tempHi5);

          float sum5 = _mm_cvtss_f32(sseSum5);

          sum5 += bias2;
          out_to_compare_with_FP[out_subscript5] = Relu_float(sum5);


          temp6 = _mm256_hadd_ps(temp6, temp6);
          temp6 = _mm256_hadd_ps(temp6, temp6);
          __m128 tempLo6 = _mm256_castps256_ps128(temp6);
          __m128 tempHi6 = _mm256_extractf128_ps(temp6, 1);
          __m128 sseSum6 = _mm_add_ps(tempLo6, tempHi6);

          float sum6 = _mm_cvtss_f32(sseSum6);

          sum6 += bias2;
          out_to_compare_with_FP[out_subscript6] = Relu_float(sum6);


          temp7 = _mm256_hadd_ps(temp7, temp7);
          temp7 = _mm256_hadd_ps(temp7, temp7);
          __m128 tempLo7 = _mm256_castps256_ps128(temp7);
          __m128 tempHi7 = _mm256_extractf128_ps(temp7, 1);
          __m128 sseSum7 = _mm_add_ps(tempLo7, tempHi7);

          float sum7 = _mm_cvtss_f32(sseSum7);

          sum7 += bias3;
          out_to_compare_with_FP[out_subscript7] = Relu_float(sum7);


          temp8 = _mm256_hadd_ps(temp8, temp8);
          temp8 = _mm256_hadd_ps(temp8, temp8);
          __m128 tempLo8 = _mm256_castps256_ps128(temp8);
          __m128 tempHi8 = _mm256_extractf128_ps(temp8, 1);
          __m128 sseSum8 = _mm_add_ps(tempLo8, tempHi8);

          float sum8 = _mm_cvtss_f32(sseSum8);

          sum8 += bias3;
          out_to_compare_with_FP[out_subscript8] = Relu_float(sum8);


          temp9 = _mm256_hadd_ps(temp9, temp9);
          temp9 = _mm256_hadd_ps(temp9, temp9);
          __m128 tempLo9 = _mm256_castps256_ps128(temp9);
          __m128 tempHi9 = _mm256_extractf128_ps(temp9, 1);
          __m128 sseSum9 = _mm_add_ps(tempLo9, tempHi9);

          float sum9 = _mm_cvtss_f32(sseSum9);

          sum9 += bias3;
          out_to_compare_with_FP[out_subscript9] = Relu_float(sum9);
        }
        // overflow/ fallback x loop
        for (; x < Output_X_dim; x++) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=32) {

                unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + d;

                unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;

                __m256 s = _mm256_loadu_ps(&in_FP[in_subscript]);

                __m256 w = _mm256_loadu_ps(&filter_FP[filter_subscript]);
                __m256 w2 = _mm256_loadu_ps(&filter_FP[filter_subscript2]);
                __m256 w3 = _mm256_loadu_ps(&filter_FP[filter_subscript3]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s, w2));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w3));


                // d+8 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+8);

                filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+8);
                filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+8);
                filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+8);

                s = _mm256_loadu_ps(&in_FP[in_subscript]);

                w = _mm256_loadu_ps(&filter_FP[filter_subscript]);
                w2 = _mm256_loadu_ps(&filter_FP[filter_subscript2]);
                w3 = _mm256_loadu_ps(&filter_FP[filter_subscript3]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s, w2));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w3));

                // d+16 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+16);

                filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+16);
                filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+16);
                filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+16);

                s = _mm256_loadu_ps(&in_FP[in_subscript]);

                w = _mm256_loadu_ps(&filter_FP[filter_subscript]);
                w2 = _mm256_loadu_ps(&filter_FP[filter_subscript2]);
                w3 = _mm256_loadu_ps(&filter_FP[filter_subscript3]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s, w2));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w3));

                // d+24 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+24);

                filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+24);
                filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+24);
                filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+24);

                s = _mm256_loadu_ps(&in_FP[in_subscript]);

                w = _mm256_loadu_ps(&filter_FP[filter_subscript]);
                w2 = _mm256_loadu_ps(&filter_FP[filter_subscript2]);
                w3 = _mm256_loadu_ps(&filter_FP[filter_subscript3]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s, w2));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w3));

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
            + (m+1);
          unsigned long long int out_subscript3 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+2);

          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias2;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
          __m128 tempLo3 = _mm256_castps256_ps128(temp3);
          __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
          __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias3;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);

        }
      }
    }
    for (; m < Output_depth_dim; m++) {
      bias = bias_array_FP[m];

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (unsigned int x = 0; x < Output_X_dim; x+=4) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          temp4 = _mm256_setzero_ps();

          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=32) {

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

                __m256 w = _mm256_loadu_ps(&filter_FP[filter_subscript]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s4, w));


                // d+8 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+8);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+8);
                in_subscript3 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+2) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+8);
                in_subscript4 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+3) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+8);

                filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+8);

                s = _mm256_loadu_ps(&in_FP[in_subscript]);
                s2 = _mm256_loadu_ps(&in_FP[in_subscript2]);
                s3 = _mm256_loadu_ps(&in_FP[in_subscript3]);
                s4 = _mm256_loadu_ps(&in_FP[in_subscript4]);

                w = _mm256_loadu_ps(&filter_FP[filter_subscript]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s4, w));

                // d+16 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+16);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+16);
                in_subscript3 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+2) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+16);
                in_subscript4 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+3) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+16);

                filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+16);

                s = _mm256_loadu_ps(&in_FP[in_subscript]);
                s2 = _mm256_loadu_ps(&in_FP[in_subscript2]);
                s3 = _mm256_loadu_ps(&in_FP[in_subscript3]);
                s4 = _mm256_loadu_ps(&in_FP[in_subscript4]);

                w = _mm256_loadu_ps(&filter_FP[filter_subscript]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s4, w));

                // d+8 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+24);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+24);
                in_subscript3 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+2) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+24);
                in_subscript4 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+3) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+24);

                filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+24);

                s = _mm256_loadu_ps(&in_FP[in_subscript]);
                s2 = _mm256_loadu_ps(&in_FP[in_subscript2]);
                s3 = _mm256_loadu_ps(&in_FP[in_subscript3]);
                s4 = _mm256_loadu_ps(&in_FP[in_subscript4]);

                w = _mm256_loadu_ps(&filter_FP[filter_subscript]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s4, w));
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

          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
          __m128 tempLo3 = _mm256_castps256_ps128(temp3);
          __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
          __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


          temp4 = _mm256_hadd_ps(temp4, temp4);
          temp4 = _mm256_hadd_ps(temp4, temp4);
          __m128 tempLo4 = _mm256_castps256_ps128(temp4);
          __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
          __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

          float sum4 = _mm_cvtss_f32(sseSum4);

          sum4 += bias;
          out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);

        }
      }
    }
  }

  printf("\n from optv3 x3 m3 d32 v1  %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// same as above, for x3m3 non-opt
// ~67 GFLOPS, better perf than above, making instruction cache miss perf more likely? profile with cachegrind/ likwid to find out more
int optimised_layer_v3_x3m3_unroll_d32_v2_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  float bias, bias2, bias3;
  __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9;
  // Calculate loop bounds for unrolled loops
  int m_bound = (Output_depth_dim/ 3) * 3;
  int x_bound = (Output_X_dim/ 3) * 3; 

  unsigned int m, x;


  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (m = 0; m < m_bound; m+=3) { //channels
      bias = bias_array_FP[m];
      bias2 = bias_array_FP[m+1];
      bias3 = bias_array_FP[m+2];

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
          // temp = 0.0f;

          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=32) {

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


                unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;

                __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                __m256 s2 = _mm256_load_ps(&in_FP[in_subscript2]);
                __m256 s3 = _mm256_load_ps(&in_FP[in_subscript3]);

                __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]); 
                __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);
                __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript3]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s, w2));
                temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s2, w2));
                temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s3, w2));
                temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w3));
                temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w3));
                temp9 = _mm256_add_ps(temp9, _mm256_mul_ps(s3, w3));


                // d+8 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+8);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+8);
                in_subscript3 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+2) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+8);


                filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+8);
                filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+8);
                filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+8);

                s = _mm256_load_ps(&in_FP[in_subscript]);
                s2 = _mm256_load_ps(&in_FP[in_subscript2]);
                s3 = _mm256_load_ps(&in_FP[in_subscript3]);

                w = _mm256_load_ps(&filter_FP[filter_subscript]); 
                w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);
                w3 = _mm256_load_ps(&filter_FP[filter_subscript3]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s, w2));
                temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s2, w2));
                temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s3, w2));
                temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w3));
                temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w3));
                temp9 = _mm256_add_ps(temp9, _mm256_mul_ps(s3, w3));


                // d+16 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+16);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+16);
                in_subscript3 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+2) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+16);


                filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+16);
                filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+16);
                filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+16);

                s = _mm256_load_ps(&in_FP[in_subscript]);
                s2 = _mm256_load_ps(&in_FP[in_subscript2]);
                s3 = _mm256_load_ps(&in_FP[in_subscript3]);

                w = _mm256_load_ps(&filter_FP[filter_subscript]); 
                w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);
                w3 = _mm256_load_ps(&filter_FP[filter_subscript3]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s, w2));
                temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s2, w2));
                temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s3, w2));
                temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w3));
                temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w3));
                temp9 = _mm256_add_ps(temp9, _mm256_mul_ps(s3, w3));


                // d+24 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+24);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+24);
                in_subscript3 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+2) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+24);


                filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+24);
                filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+24);
                filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+24);

                s = _mm256_load_ps(&in_FP[in_subscript]);
                s2 = _mm256_load_ps(&in_FP[in_subscript2]);
                s3 = _mm256_load_ps(&in_FP[in_subscript3]);

                w = _mm256_load_ps(&filter_FP[filter_subscript]); 
                w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);
                w3 = _mm256_load_ps(&filter_FP[filter_subscript3]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s, w2));
                temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s2, w2));
                temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s3, w2));
                temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w3));
                temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w3));
                temp9 = _mm256_add_ps(temp9, _mm256_mul_ps(s3, w3));

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
            + (m+1);
          unsigned long long int out_subscript5 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+1);
          unsigned long long int out_subscript6 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+2) * Output_depth_dim
            + (m+1);
          unsigned long long int out_subscript7 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+2);
          unsigned long long int out_subscript8 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+2);
          unsigned long long int out_subscript9 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+2) * Output_depth_dim
            + (m+2);



          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
          __m128 tempLo3 = _mm256_castps256_ps128(temp3);
          __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
          __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


          temp4 = _mm256_hadd_ps(temp4, temp4);
          temp4 = _mm256_hadd_ps(temp4, temp4);
          __m128 tempLo4 = _mm256_castps256_ps128(temp4);
          __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
          __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

          float sum4 = _mm_cvtss_f32(sseSum4);

          sum4 += bias2;
          out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);


          temp5 = _mm256_hadd_ps(temp5, temp5);
          temp5 = _mm256_hadd_ps(temp5, temp5);
          __m128 tempLo5 = _mm256_castps256_ps128(temp5);
          __m128 tempHi5 = _mm256_extractf128_ps(temp5, 1);
          __m128 sseSum5 = _mm_add_ps(tempLo5, tempHi5);

          float sum5 = _mm_cvtss_f32(sseSum5);

          sum5 += bias2;
          out_to_compare_with_FP[out_subscript5] = Relu_float(sum5);


          temp6 = _mm256_hadd_ps(temp6, temp6);
          temp6 = _mm256_hadd_ps(temp6, temp6);
          __m128 tempLo6 = _mm256_castps256_ps128(temp6);
          __m128 tempHi6 = _mm256_extractf128_ps(temp6, 1);
          __m128 sseSum6 = _mm_add_ps(tempLo6, tempHi6);

          float sum6 = _mm_cvtss_f32(sseSum6);

          sum6 += bias2;
          out_to_compare_with_FP[out_subscript6] = Relu_float(sum6);


          temp7 = _mm256_hadd_ps(temp7, temp7);
          temp7 = _mm256_hadd_ps(temp7, temp7);
          __m128 tempLo7 = _mm256_castps256_ps128(temp7);
          __m128 tempHi7 = _mm256_extractf128_ps(temp7, 1);
          __m128 sseSum7 = _mm_add_ps(tempLo7, tempHi7);

          float sum7 = _mm_cvtss_f32(sseSum7);

          sum7 += bias3;
          out_to_compare_with_FP[out_subscript7] = Relu_float(sum7);


          temp8 = _mm256_hadd_ps(temp8, temp8);
          temp8 = _mm256_hadd_ps(temp8, temp8);
          __m128 tempLo8 = _mm256_castps256_ps128(temp8);
          __m128 tempHi8 = _mm256_extractf128_ps(temp8, 1);
          __m128 sseSum8 = _mm_add_ps(tempLo8, tempHi8);

          float sum8 = _mm_cvtss_f32(sseSum8);

          sum8 += bias3;
          out_to_compare_with_FP[out_subscript8] = Relu_float(sum8);


          temp9 = _mm256_hadd_ps(temp9, temp9);
          temp9 = _mm256_hadd_ps(temp9, temp9);
          __m128 tempLo9 = _mm256_castps256_ps128(temp9);
          __m128 tempHi9 = _mm256_extractf128_ps(temp9, 1);
          __m128 sseSum9 = _mm_add_ps(tempLo9, tempHi9);

          float sum9 = _mm_cvtss_f32(sseSum9);

          sum9 += bias3;
          out_to_compare_with_FP[out_subscript9] = Relu_float(sum9);
        }
        // overflow/ fallback x loop
        for (; x < Output_X_dim; x++) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=32) {

                unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + d;

                unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;

                __m256 s = _mm256_loadu_ps(&in_FP[in_subscript]);

                __m256 w = _mm256_loadu_ps(&filter_FP[filter_subscript]);
                __m256 w2 = _mm256_loadu_ps(&filter_FP[filter_subscript2]);
                __m256 w3 = _mm256_loadu_ps(&filter_FP[filter_subscript3]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s, w2));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w3));


                // d+8 ------------------------------------------------------------
                // |
                // |
                // v


                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+8);

                filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+8);
                filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+8);
                filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+8);

                s = _mm256_loadu_ps(&in_FP[in_subscript]);

                w = _mm256_loadu_ps(&filter_FP[filter_subscript]);
                w2 = _mm256_loadu_ps(&filter_FP[filter_subscript2]);
                w3 = _mm256_loadu_ps(&filter_FP[filter_subscript3]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s, w2));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w3));


                // d+16 ------------------------------------------------------------
                // |
                // |
                // v


                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+16);

                filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+16);
                filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+16);
                filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+16);

                s = _mm256_loadu_ps(&in_FP[in_subscript]);

                w = _mm256_loadu_ps(&filter_FP[filter_subscript]);
                w2 = _mm256_loadu_ps(&filter_FP[filter_subscript2]);
                w3 = _mm256_loadu_ps(&filter_FP[filter_subscript3]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s, w2));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w3));


                // d+24 ------------------------------------------------------------
                // |
                // |
                // v


                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+24);

                filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+24);
                filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+24);
                filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+24);

                s = _mm256_loadu_ps(&in_FP[in_subscript]);

                w = _mm256_loadu_ps(&filter_FP[filter_subscript]);
                w2 = _mm256_loadu_ps(&filter_FP[filter_subscript2]);
                w3 = _mm256_loadu_ps(&filter_FP[filter_subscript3]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s, w2));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w3));

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
            + (m+1);
          unsigned long long int out_subscript3 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+2);

          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias2;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
          __m128 tempLo3 = _mm256_castps256_ps128(temp3);
          __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
          __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias3;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);

        }
      }
    }
    for (; m < Output_depth_dim; m++) {
      bias = bias_array_FP[m];

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (unsigned int x = 0; x < Output_X_dim; x++) {	//Output Width
          temp = _mm256_setzero_ps();

          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=32) {

                unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + d;

                unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;

                __m256 s = _mm256_loadu_ps(&in_FP[in_subscript]);
                __m256 w = _mm256_loadu_ps(&filter_FP[filter_subscript]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));


                // d+8 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+8);

                filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+8);

                s = _mm256_loadu_ps(&in_FP[in_subscript]);
                w = _mm256_loadu_ps(&filter_FP[filter_subscript]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));


                // d+16 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+16);

                filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+16);

                s = _mm256_loadu_ps(&in_FP[in_subscript]);
                w = _mm256_loadu_ps(&filter_FP[filter_subscript]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));


                // d+24 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+24);

                filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+24);

                s = _mm256_loadu_ps(&in_FP[in_subscript]);
                w = _mm256_loadu_ps(&filter_FP[filter_subscript]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
              }
            }
          }


          unsigned long long int out_subscript = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + m;

          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);

        }
      }
    }
  }

  printf("\n from optv3 x3 m3 hadd d32 v2 %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// unroll d by factor of 4 (d32)
// ~71 GFLOPS, 
int optimised_layer_v3_x2m4_unroll_d32_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  float bias, bias2, bias3, bias4;
  __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int m = 0; m < Output_depth_dim; m+=4) { //channels
      bias = bias_array_FP[m];
      bias2 = bias_array_FP[m+1];
      bias3 = bias_array_FP[m+2];
      bias4 = bias_array_FP[m+3];

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
          // temp = 0.0f;

          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=32) {

                unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + d;
                unsigned long long int in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + d;


                unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript4 = (m+3) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;

                __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                __m256 s2 = _mm256_load_ps(&in_FP[in_subscript2]);

                __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]);
                __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);
                __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript3]);
                __m256 w4 = _mm256_load_ps(&filter_FP[filter_subscript4]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w2));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s2, w2));
                temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s, w3));
                temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s2, w3));
                temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w4));
                temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w4));


                // d+8 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+8);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+8);


                filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+8);
                filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+8);
                filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+8);
                filter_subscript4 = (m+3) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+8);

                s = _mm256_load_ps(&in_FP[in_subscript]);
                s2 = _mm256_load_ps(&in_FP[in_subscript2]);

                w = _mm256_load_ps(&filter_FP[filter_subscript]);
                w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);
                w3 = _mm256_load_ps(&filter_FP[filter_subscript3]);
                w4 = _mm256_load_ps(&filter_FP[filter_subscript4]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w2));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s2, w2));
                temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s, w3));
                temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s2, w3));
                temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w4));
                temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w4));


                // d+16 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+16);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+16);


                filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+16);
                filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+16);
                filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+16);
                filter_subscript4 = (m+3) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+16);

                s = _mm256_load_ps(&in_FP[in_subscript]);
                s2 = _mm256_load_ps(&in_FP[in_subscript2]);

                w = _mm256_load_ps(&filter_FP[filter_subscript]);
                w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);
                w3 = _mm256_load_ps(&filter_FP[filter_subscript3]);
                w4 = _mm256_load_ps(&filter_FP[filter_subscript4]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w2));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s2, w2));
                temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s, w3));
                temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s2, w3));
                temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w4));
                temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w4));


                // d+24 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+24);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+24);


                filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+24);
                filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+24);
                filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+24);
                filter_subscript4 = (m+3) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+24);

                s = _mm256_load_ps(&in_FP[in_subscript]);
                s2 = _mm256_load_ps(&in_FP[in_subscript2]);

                w = _mm256_load_ps(&filter_FP[filter_subscript]);
                w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);
                w3 = _mm256_load_ps(&filter_FP[filter_subscript3]);
                w4 = _mm256_load_ps(&filter_FP[filter_subscript4]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w2));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s2, w2));
                temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s, w3));
                temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s2, w3));
                temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w4));
                temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w4));
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
            + (m+1);
          unsigned long long int out_subscript4 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+1);
          unsigned long long int out_subscript5 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+2);
          unsigned long long int out_subscript6 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+2);
          unsigned long long int out_subscript7 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+3);
          unsigned long long int out_subscript8 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+3);



          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
           tempLo = _mm256_castps256_ps128(temp3);
           tempHi = _mm256_extractf128_ps(temp3, 1);
           sseSum = _mm_add_ps(tempLo, tempHi);

          float sum3 = _mm_cvtss_f32(sseSum);

          sum3 += bias2;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


          temp4 = _mm256_hadd_ps(temp4, temp4);
          temp4 = _mm256_hadd_ps(temp4, temp4);
           tempLo2 = _mm256_castps256_ps128(temp4);
           tempHi2 = _mm256_extractf128_ps(temp4, 1);
           sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum4 = _mm_cvtss_f32(sseSum2);

          sum4 += bias2;
          out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);


          temp5 = _mm256_hadd_ps(temp5, temp5);
          temp5 = _mm256_hadd_ps(temp5, temp5);
           tempLo = _mm256_castps256_ps128(temp5);
           tempHi = _mm256_extractf128_ps(temp5, 1);
           sseSum = _mm_add_ps(tempLo, tempHi);

          float sum5 = _mm_cvtss_f32(sseSum);

          sum5 += bias3;
          out_to_compare_with_FP[out_subscript5] = Relu_float(sum5);


          temp6 = _mm256_hadd_ps(temp6, temp6);
          temp6 = _mm256_hadd_ps(temp6, temp6);
           tempLo2 = _mm256_castps256_ps128(temp6);
           tempHi2 = _mm256_extractf128_ps(temp6, 1);
           sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum6 = _mm_cvtss_f32(sseSum2);

          sum6 += bias3;
          out_to_compare_with_FP[out_subscript6] = Relu_float(sum6);


          temp7 = _mm256_hadd_ps(temp7, temp7);
          temp7 = _mm256_hadd_ps(temp7, temp7);
           tempLo = _mm256_castps256_ps128(temp7);
           tempHi = _mm256_extractf128_ps(temp7, 1);
           sseSum = _mm_add_ps(tempLo, tempHi);

          float sum7 = _mm_cvtss_f32(sseSum);

          sum7 += bias4;
          out_to_compare_with_FP[out_subscript7] = Relu_float(sum7);


          temp8 = _mm256_hadd_ps(temp8, temp8);
          temp8 = _mm256_hadd_ps(temp8, temp8);
           tempLo2 = _mm256_castps256_ps128(temp8);
           tempHi2 = _mm256_extractf128_ps(temp8, 1);
           sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum8 = _mm_cvtss_f32(sseSum2);

          sum8 += bias4;
          out_to_compare_with_FP[out_subscript8] = Relu_float(sum8);
        }
      }
    }
  }

  printf("\n from optv3 x2 m4 d32 %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// unroll d by factor of 4 (d32)
// ~68 GFLOPS, makes sense that x2m4 performs better (albeit slightly) since it calculates twice the outputs at once
int optimised_layer_v3_x4m2_unroll_d32_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  float bias, bias2;
  __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int m = 0; m < Output_depth_dim; m+=2) { //channels
      bias = bias_array_FP[m];
      bias2 = bias_array_FP[m+1];

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
          // temp = 0.0f;

          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=32) {

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
                unsigned long long int filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;

                __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                __m256 s2 = _mm256_load_ps(&in_FP[in_subscript2]);
                __m256 s3 = _mm256_load_ps(&in_FP[in_subscript3]);
                __m256 s4 = _mm256_load_ps(&in_FP[in_subscript4]);

                __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]);
                __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s4, w));
                temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s, w2));
                temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s2, w2));
                temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s3, w2));
                temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s4, w2));


                // d+8 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+8);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+8);
                in_subscript3 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+2) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+8);
                in_subscript4 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+3) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+8);


                filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+8);
                filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+8);

                s = _mm256_load_ps(&in_FP[in_subscript]);
                s2 = _mm256_load_ps(&in_FP[in_subscript2]);
                s3 = _mm256_load_ps(&in_FP[in_subscript3]);
                s4 = _mm256_load_ps(&in_FP[in_subscript4]);

                w = _mm256_load_ps(&filter_FP[filter_subscript]);
                w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s4, w));
                temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s, w2));
                temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s2, w2));
                temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s3, w2));
                temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s4, w2));


                // d+16 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+16);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+16);
                in_subscript3 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+2) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+16);
                in_subscript4 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+3) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+16);


                filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+16);
                filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+16);

                s = _mm256_load_ps(&in_FP[in_subscript]);
                s2 = _mm256_load_ps(&in_FP[in_subscript2]);
                s3 = _mm256_load_ps(&in_FP[in_subscript3]);
                s4 = _mm256_load_ps(&in_FP[in_subscript4]);

                w = _mm256_load_ps(&filter_FP[filter_subscript]);
                w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s4, w));
                temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s, w2));
                temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s2, w2));
                temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s3, w2));
                temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s4, w2));


                // d+24 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+24);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+24);
                in_subscript3 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+2) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+24);
                in_subscript4 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+3) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+24);


                filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+24);
                filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+24);

                s = _mm256_load_ps(&in_FP[in_subscript]);
                s2 = _mm256_load_ps(&in_FP[in_subscript2]);
                s3 = _mm256_load_ps(&in_FP[in_subscript3]);
                s4 = _mm256_load_ps(&in_FP[in_subscript4]);

                w = _mm256_load_ps(&filter_FP[filter_subscript]);
                w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s4, w));
                temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s, w2));
                temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s2, w2));
                temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s3, w2));
                temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s4, w2));
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
            + (m+1);
          unsigned long long int out_subscript6 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+1);
          unsigned long long int out_subscript7 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+2) * Output_depth_dim
            + (m+1);
          unsigned long long int out_subscript8 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+3) * Output_depth_dim
            + (m+1);



          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
          __m128 tempLo3 = _mm256_castps256_ps128(temp3);
          __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
          __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


          temp4 = _mm256_hadd_ps(temp4, temp4);
          temp4 = _mm256_hadd_ps(temp4, temp4);
          __m128 tempLo4 = _mm256_castps256_ps128(temp4);
          __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
          __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

          float sum4 = _mm_cvtss_f32(sseSum4);

          sum4 += bias;
          out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);


          temp5 = _mm256_hadd_ps(temp5, temp5);
          temp5 = _mm256_hadd_ps(temp5, temp5);
          __m128 tempLo5 = _mm256_castps256_ps128(temp5);
          __m128 tempHi5 = _mm256_extractf128_ps(temp5, 1);
          __m128 sseSum5 = _mm_add_ps(tempLo5, tempHi5);

          float sum5 = _mm_cvtss_f32(sseSum5);

          sum5 += bias2;
          out_to_compare_with_FP[out_subscript5] = Relu_float(sum5);


          temp6 = _mm256_hadd_ps(temp6, temp6);
          temp6 = _mm256_hadd_ps(temp6, temp6);
          __m128 tempLo6 = _mm256_castps256_ps128(temp6);
          __m128 tempHi6 = _mm256_extractf128_ps(temp6, 1);
          __m128 sseSum6 = _mm_add_ps(tempLo6, tempHi6);

          float sum6 = _mm_cvtss_f32(sseSum6);

          sum6 += bias2;
          out_to_compare_with_FP[out_subscript6] = Relu_float(sum6);


          temp7 = _mm256_hadd_ps(temp7, temp7);
          temp7 = _mm256_hadd_ps(temp7, temp7);
          __m128 tempLo7 = _mm256_castps256_ps128(temp7);
          __m128 tempHi7 = _mm256_extractf128_ps(temp7, 1);
          __m128 sseSum7 = _mm_add_ps(tempLo7, tempHi7);

          float sum7 = _mm_cvtss_f32(sseSum7);

          sum7 += bias2;
          out_to_compare_with_FP[out_subscript7] = Relu_float(sum7);


          temp8 = _mm256_hadd_ps(temp8, temp8);
          temp8 = _mm256_hadd_ps(temp8, temp8);
          __m128 tempLo8 = _mm256_castps256_ps128(temp8);
          __m128 tempHi8 = _mm256_extractf128_ps(temp8, 1);
          __m128 sseSum8 = _mm_add_ps(tempLo8, tempHi8);

          float sum8 = _mm_cvtss_f32(sseSum8);

          sum8 += bias2;
          out_to_compare_with_FP[out_subscript8] = Relu_float(sum8);
        }
      }
    }
  }

  printf("\n from optv3 x4 m2 d32 %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// unroll d by factor of 2 (d32) (hadd instructions)
// ~49 GFLOPS, worse performance? possibly increase it by increasing simd register usage
int optimised_layer_v3_x2m2_unroll_d32_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  float bias, bias2;
  __m256 temp, temp2, temp3, temp4;

  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int m = 0; m < Output_depth_dim; m+=2) { //channels
      bias = bias_array_FP[m];
      bias2 = bias_array_FP[m+1];

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (unsigned int x = 0; x < Output_X_dim; x+=2) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          temp4 = _mm256_setzero_ps();
          // temp = 0.0f;

          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=32) {

                unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + d;
                unsigned long long int in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + d;


                unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;

                __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                __m256 s2 = _mm256_load_ps(&in_FP[in_subscript2]);

                __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]);
                __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w2));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s2, w2));


                // d+8 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+8);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+8);


                filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+8);
                filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+8);

                s = _mm256_load_ps(&in_FP[in_subscript]);
                s2 = _mm256_load_ps(&in_FP[in_subscript2]);

                w = _mm256_load_ps(&filter_FP[filter_subscript]);
                w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w2));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s2, w2));


                // d+16 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+16);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+16);


                filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+16);
                filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+16);

                s = _mm256_load_ps(&in_FP[in_subscript]);
                s2 = _mm256_load_ps(&in_FP[in_subscript2]);

                w = _mm256_load_ps(&filter_FP[filter_subscript]);
                w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w2));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s2, w2));


                // d+24 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+24);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+24);


                filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+24);
                filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+24);

                s = _mm256_load_ps(&in_FP[in_subscript]);
                s2 = _mm256_load_ps(&in_FP[in_subscript2]);

                w = _mm256_load_ps(&filter_FP[filter_subscript]);
                w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w2));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s2, w2));

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
            + (m+1);
          unsigned long long int out_subscript4 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+1);



          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
          __m128 tempLo3 = _mm256_castps256_ps128(temp3);
          __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
          __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias2;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


          temp4 = _mm256_hadd_ps(temp4, temp4);
          temp4 = _mm256_hadd_ps(temp4, temp4);
          __m128 tempLo4 = _mm256_castps256_ps128(temp4);
          __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
          __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

          float sum4 = _mm_cvtss_f32(sseSum4);

          sum4 += bias2;
          out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);

        }
      }
    }
  }

  printf("\n from optv3 x2 m2 hadd d32 %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// same as above, but increased simd register usage in d loop (was 8, now 16)
// ~64 GFLOPS, 
int optimised_layer_v3_x2m2_unroll_d32_opt_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  float bias, bias2;
  __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int m = 0; m < Output_depth_dim; m+=2) { //channels
      bias = bias_array_FP[m];
      bias2 = bias_array_FP[m+1];

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
          // temp = 0.0f;

          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=32) {

                unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + d;
                unsigned long long int in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + d;


                unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;

                __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                __m256 s2 = _mm256_load_ps(&in_FP[in_subscript2]);

                __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]);
                __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w2));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s2, w2));


                // d+8 ------------------------------------------------------------
                // |
                // |
                // v

                unsigned long long int in_subscript3 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+8);
                unsigned long long int in_subscript4 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+8);


                unsigned long long int filter_subscript3 = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+8);
                unsigned long long int filter_subscript4 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+8);

                __m256 s3 = _mm256_load_ps(&in_FP[in_subscript3]);
                __m256 s4 = _mm256_load_ps(&in_FP[in_subscript4]);

                __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript3]);
                __m256 w4 = _mm256_load_ps(&filter_FP[filter_subscript4]);

                temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s3, w3));
                temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s4, w3));
                temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s3, w4));
                temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s4, w4));


                // d+16 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+16);
                in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+16);


                filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+16);
                filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+16);

                s = _mm256_load_ps(&in_FP[in_subscript]);
                s2 = _mm256_load_ps(&in_FP[in_subscript2]);

                w = _mm256_load_ps(&filter_FP[filter_subscript]);
                w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w2));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s2, w2));


                // d+24 ------------------------------------------------------------
                // |
                // |
                // v

                in_subscript3 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+24);
                in_subscript4 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                  + (d+24);


                filter_subscript3 = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+24);
                filter_subscript4 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + (d+24);

                s3 = _mm256_load_ps(&in_FP[in_subscript3]);
                s4 = _mm256_load_ps(&in_FP[in_subscript4]);

                w3 = _mm256_load_ps(&filter_FP[filter_subscript3]);
                w4 = _mm256_load_ps(&filter_FP[filter_subscript4]);

                temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s3, w3));
                temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s4, w3));
                temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s3, w4));
                temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s4, w4));
              }
            }
          }

          temp = _mm256_add_ps(temp, temp5);
          temp2 = _mm256_add_ps(temp2, temp6);
          temp3 = _mm256_add_ps(temp3, temp7);
          temp4 = _mm256_add_ps(temp4, temp8);

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
            + (m+1);
          unsigned long long int out_subscript4 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+1);



          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
          __m128 tempLo3 = _mm256_castps256_ps128(temp3);
          __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
          __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias2;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


          temp4 = _mm256_hadd_ps(temp4, temp4);
          temp4 = _mm256_hadd_ps(temp4, temp4);
          __m128 tempLo4 = _mm256_castps256_ps128(temp4);
          __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
          __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

          float sum4 = _mm_cvtss_f32(sseSum4);

          sum4 += bias2;
          out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);

        }
      }
    }
  }

  printf("\n from optv3 x2 m2 hadd d32 opt %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}






// v4 - loop tiling - check (val) cachegrind outputs or just test with increased in/out sizes


// loop tiling on y loop, optimised_layer_v2_unroll_x3m3_hadd_opt_FP()
// tile =2/4 ~76 GFLOPS, slightly more performant than non-tiled?
int optimised_layer_v4_x3m3_tiled_y_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  float bias, bias2, bias3;
  __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9;
  // Calculate loop bounds for unrolled loops
  int m_bound = (Output_depth_dim/ 3) * 3;
  int x_bound = (Output_X_dim/ 3) * 3; 
  #define y_tile 4   //1,2,4,13,26 

  unsigned int m, x;

  for (unsigned int yy = 0; yy < Output_Y_dim; yy+=y_tile) {
    for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
      for (m = 0; m < m_bound; m+=3) { //channels
        bias = bias_array_FP[m];
        bias2 = bias_array_FP[m+1];
        bias3 = bias_array_FP[m+2];

        for (unsigned int y = yy; y < yy + y_tile; y++) {	//Output height
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
            // temp = 0.0f;

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


                  unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                    + off_y * Mask_X_dim * Input_depth_dim
                    + off_x * Input_depth_dim
                    + d;
                  unsigned long long int filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                    + off_y * Mask_X_dim * Input_depth_dim
                    + off_x * Input_depth_dim
                    + d;
                  unsigned long long int filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                    + off_y * Mask_X_dim * Input_depth_dim
                    + off_x * Input_depth_dim
                    + d;

                  __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                  __m256 s2 = _mm256_load_ps(&in_FP[in_subscript2]);
                  __m256 s3 = _mm256_load_ps(&in_FP[in_subscript3]);

                  __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]); 
                  __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);
                  __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript3]);

                  temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                  temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                  temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                  temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s, w2));
                  temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s2, w2));
                  temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s3, w2));
                  temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w3));
                  temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w3));
                  temp9 = _mm256_add_ps(temp9, _mm256_mul_ps(s3, w3));


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
              x * Output_depth_dim
              + (m+1);
            unsigned long long int out_subscript5 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
              y * (Output_depth_dim * Output_X_dim) +
              (x+1) * Output_depth_dim
              + (m+1);
            unsigned long long int out_subscript6 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
              y * (Output_depth_dim * Output_X_dim) +
              (x+2) * Output_depth_dim
              + (m+1);
            unsigned long long int out_subscript7 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
              y * (Output_depth_dim * Output_X_dim) +
              x * Output_depth_dim
              + (m+2);
            unsigned long long int out_subscript8 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
              y * (Output_depth_dim * Output_X_dim) +
              (x+1) * Output_depth_dim
              + (m+2);
            unsigned long long int out_subscript9 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
              y * (Output_depth_dim * Output_X_dim) +
              (x+2) * Output_depth_dim
              + (m+2);



            temp = _mm256_hadd_ps(temp, temp);
            temp = _mm256_hadd_ps(temp, temp);
            __m128 tempLo = _mm256_castps256_ps128(temp);
            __m128 tempHi = _mm256_extractf128_ps(temp, 1);
            __m128 sseSum = _mm_add_ps(tempLo, tempHi);

            float sum = _mm_cvtss_f32(sseSum);

            sum += bias;
            out_to_compare_with_FP[out_subscript] = Relu_float(sum);


            temp2 = _mm256_hadd_ps(temp2, temp2);
            temp2 = _mm256_hadd_ps(temp2, temp2);
            __m128 tempLo2 = _mm256_castps256_ps128(temp2);
            __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
            __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum2 = _mm_cvtss_f32(sseSum2);

            sum2 += bias;
            out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


            temp3 = _mm256_hadd_ps(temp3, temp3);
            temp3 = _mm256_hadd_ps(temp3, temp3);
            __m128 tempLo3 = _mm256_castps256_ps128(temp3);
            __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
            __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

            float sum3 = _mm_cvtss_f32(sseSum3);

            sum3 += bias;
            out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


            temp4 = _mm256_hadd_ps(temp4, temp4);
            temp4 = _mm256_hadd_ps(temp4, temp4);
            __m128 tempLo4 = _mm256_castps256_ps128(temp4);
            __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
            __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

            float sum4 = _mm_cvtss_f32(sseSum4);

            sum4 += bias2;
            out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);


            temp5 = _mm256_hadd_ps(temp5, temp5);
            temp5 = _mm256_hadd_ps(temp5, temp5);
            __m128 tempLo5 = _mm256_castps256_ps128(temp5);
            __m128 tempHi5 = _mm256_extractf128_ps(temp5, 1);
            __m128 sseSum5 = _mm_add_ps(tempLo5, tempHi5);

            float sum5 = _mm_cvtss_f32(sseSum5);

            sum5 += bias2;
            out_to_compare_with_FP[out_subscript5] = Relu_float(sum5);


            temp6 = _mm256_hadd_ps(temp6, temp6);
            temp6 = _mm256_hadd_ps(temp6, temp6);
            __m128 tempLo6 = _mm256_castps256_ps128(temp6);
            __m128 tempHi6 = _mm256_extractf128_ps(temp6, 1);
            __m128 sseSum6 = _mm_add_ps(tempLo6, tempHi6);

            float sum6 = _mm_cvtss_f32(sseSum6);

            sum6 += bias2;
            out_to_compare_with_FP[out_subscript6] = Relu_float(sum6);


            temp7 = _mm256_hadd_ps(temp7, temp7);
            temp7 = _mm256_hadd_ps(temp7, temp7);
            __m128 tempLo7 = _mm256_castps256_ps128(temp7);
            __m128 tempHi7 = _mm256_extractf128_ps(temp7, 1);
            __m128 sseSum7 = _mm_add_ps(tempLo7, tempHi7);

            float sum7 = _mm_cvtss_f32(sseSum7);

            sum7 += bias3;
            out_to_compare_with_FP[out_subscript7] = Relu_float(sum7);


            temp8 = _mm256_hadd_ps(temp8, temp8);
            temp8 = _mm256_hadd_ps(temp8, temp8);
            __m128 tempLo8 = _mm256_castps256_ps128(temp8);
            __m128 tempHi8 = _mm256_extractf128_ps(temp8, 1);
            __m128 sseSum8 = _mm_add_ps(tempLo8, tempHi8);

            float sum8 = _mm_cvtss_f32(sseSum8);

            sum8 += bias3;
            out_to_compare_with_FP[out_subscript8] = Relu_float(sum8);


            temp9 = _mm256_hadd_ps(temp9, temp9);
            temp9 = _mm256_hadd_ps(temp9, temp9);
            __m128 tempLo9 = _mm256_castps256_ps128(temp9);
            __m128 tempHi9 = _mm256_extractf128_ps(temp9, 1);
            __m128 sseSum9 = _mm_add_ps(tempLo9, tempHi9);

            float sum9 = _mm_cvtss_f32(sseSum9);

            sum9 += bias3;
            out_to_compare_with_FP[out_subscript9] = Relu_float(sum9);
          }
          // overflow/ fallback x loop
          for (; x < Output_X_dim; x++) {	//Output Width
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

                  unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                    + off_y * Mask_X_dim * Input_depth_dim
                    + off_x * Input_depth_dim
                    + d;
                  unsigned long long int filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                    + off_y * Mask_X_dim * Input_depth_dim
                    + off_x * Input_depth_dim
                    + d;
                  unsigned long long int filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                    + off_y * Mask_X_dim * Input_depth_dim
                    + off_x * Input_depth_dim
                    + d;

                  __m256 s = _mm256_loadu_ps(&in_FP[in_subscript]);

                  __m256 w = _mm256_loadu_ps(&filter_FP[filter_subscript]);
                  __m256 w2 = _mm256_loadu_ps(&filter_FP[filter_subscript2]);
                  __m256 w3 = _mm256_loadu_ps(&filter_FP[filter_subscript3]);

                  temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                  temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s, w2));
                  temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w3));

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
              x * Output_depth_dim
              + (m+1);
            unsigned long long int out_subscript3 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
              y * (Output_depth_dim * Output_X_dim) +
              x * Output_depth_dim
              + (m+2);

            temp = _mm256_hadd_ps(temp, temp);
            temp = _mm256_hadd_ps(temp, temp);
            __m128 tempLo = _mm256_castps256_ps128(temp);
            __m128 tempHi = _mm256_extractf128_ps(temp, 1);
            __m128 sseSum = _mm_add_ps(tempLo, tempHi);

            float sum = _mm_cvtss_f32(sseSum);

            sum += bias;
            out_to_compare_with_FP[out_subscript] = Relu_float(sum);


            temp2 = _mm256_hadd_ps(temp2, temp2);
            temp2 = _mm256_hadd_ps(temp2, temp2);
            __m128 tempLo2 = _mm256_castps256_ps128(temp2);
            __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
            __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum2 = _mm_cvtss_f32(sseSum2);

            sum2 += bias2;
            out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


            temp3 = _mm256_hadd_ps(temp3, temp3);
            temp3 = _mm256_hadd_ps(temp3, temp3);
            __m128 tempLo3 = _mm256_castps256_ps128(temp3);
            __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
            __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

            float sum3 = _mm_cvtss_f32(sseSum3);

            sum3 += bias3;
            out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);

          }
        }
      }
      for (; m < Output_depth_dim; m++) {
        bias = bias_array_FP[m];

        for (unsigned int y = yy; y < yy + y_tile; y++) {	//Output height
          for (unsigned int x = 0; x < Output_X_dim; x+=4) {	//Output Width
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

                  __m256 w = _mm256_loadu_ps(&filter_FP[filter_subscript]);

                  temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                  temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                  temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                  temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s4, w));
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

            temp = _mm256_hadd_ps(temp, temp);
            temp = _mm256_hadd_ps(temp, temp);
            __m128 tempLo = _mm256_castps256_ps128(temp);
            __m128 tempHi = _mm256_extractf128_ps(temp, 1);
            __m128 sseSum = _mm_add_ps(tempLo, tempHi);

            float sum = _mm_cvtss_f32(sseSum);

            sum += bias;
            out_to_compare_with_FP[out_subscript] = Relu_float(sum);


            temp2 = _mm256_hadd_ps(temp2, temp2);
            temp2 = _mm256_hadd_ps(temp2, temp2);
            __m128 tempLo2 = _mm256_castps256_ps128(temp2);
            __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
            __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum2 = _mm_cvtss_f32(sseSum2);

            sum2 += bias;
            out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


            temp3 = _mm256_hadd_ps(temp3, temp3);
            temp3 = _mm256_hadd_ps(temp3, temp3);
            __m128 tempLo3 = _mm256_castps256_ps128(temp3);
            __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
            __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

            float sum3 = _mm_cvtss_f32(sseSum3);

            sum3 += bias;
            out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


            temp4 = _mm256_hadd_ps(temp4, temp4);
            temp4 = _mm256_hadd_ps(temp4, temp4);
            __m128 tempLo4 = _mm256_castps256_ps128(temp4);
            __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
            __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

            float sum4 = _mm_cvtss_f32(sseSum4);

            sum4 += bias;
            out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);

          }
        }
      }
    }
  }

  #undef y_tile

  printf("\n from optv4 x3 m3, tiled y loop %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// loop tiling on x loop, optimised_layer_v2_unroll_x3m3_hadd_opt_FP()
// tile = 4 ~61 GFLOPS, tile = 13 ~71 GFLOPS (13 not really useful, since i'm making generalised conv layer func and x/y == multiple of 4)
int optimised_layer_v4_x3m3_tiled_x_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  float bias, bias2, bias3;
  __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9;

  #define x_tile 4   //1,2,4,13,26 

  // Calculate loop bounds for unrolled loops
  int m_bound = (Output_depth_dim/ 3) * 3;
  int x_bound = (x_tile/ 3) * 3; 

  unsigned int m, x;

  for (unsigned int xx = 0; xx < Output_X_dim; xx+=x_tile) {
    for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
      for (m = 0; m < m_bound; m+=3) { //channels
        bias = bias_array_FP[m];
        bias2 = bias_array_FP[m+1];
        bias3 = bias_array_FP[m+2];

        for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
          for (x = xx; x < xx + x_bound; x+=3) {	//Output Width
            temp = _mm256_setzero_ps();
            temp2 = _mm256_setzero_ps();
            temp3 = _mm256_setzero_ps();
            temp4 = _mm256_setzero_ps();
            temp5 = _mm256_setzero_ps();
            temp6 = _mm256_setzero_ps();
            temp7 = _mm256_setzero_ps();
            temp8 = _mm256_setzero_ps();
            temp9 = _mm256_setzero_ps();
            // temp = 0.0f;

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


                  unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                    + off_y * Mask_X_dim * Input_depth_dim
                    + off_x * Input_depth_dim
                    + d;
                  unsigned long long int filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                    + off_y * Mask_X_dim * Input_depth_dim
                    + off_x * Input_depth_dim
                    + d;
                  unsigned long long int filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                    + off_y * Mask_X_dim * Input_depth_dim
                    + off_x * Input_depth_dim
                    + d;

                  __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                  __m256 s2 = _mm256_load_ps(&in_FP[in_subscript2]);
                  __m256 s3 = _mm256_load_ps(&in_FP[in_subscript3]);

                  __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]); 
                  __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);
                  __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript3]);

                  temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                  temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                  temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                  temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s, w2));
                  temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s2, w2));
                  temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s3, w2));
                  temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w3));
                  temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w3));
                  temp9 = _mm256_add_ps(temp9, _mm256_mul_ps(s3, w3));


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
              x * Output_depth_dim
              + (m+1);
            unsigned long long int out_subscript5 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
              y * (Output_depth_dim * Output_X_dim) +
              (x+1) * Output_depth_dim
              + (m+1);
            unsigned long long int out_subscript6 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
              y * (Output_depth_dim * Output_X_dim) +
              (x+2) * Output_depth_dim
              + (m+1);
            unsigned long long int out_subscript7 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
              y * (Output_depth_dim * Output_X_dim) +
              x * Output_depth_dim
              + (m+2);
            unsigned long long int out_subscript8 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
              y * (Output_depth_dim * Output_X_dim) +
              (x+1) * Output_depth_dim
              + (m+2);
            unsigned long long int out_subscript9 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
              y * (Output_depth_dim * Output_X_dim) +
              (x+2) * Output_depth_dim
              + (m+2);



            temp = _mm256_hadd_ps(temp, temp);
            temp = _mm256_hadd_ps(temp, temp);
            __m128 tempLo = _mm256_castps256_ps128(temp);
            __m128 tempHi = _mm256_extractf128_ps(temp, 1);
            __m128 sseSum = _mm_add_ps(tempLo, tempHi);

            float sum = _mm_cvtss_f32(sseSum);

            sum += bias;
            out_to_compare_with_FP[out_subscript] = Relu_float(sum);


            temp2 = _mm256_hadd_ps(temp2, temp2);
            temp2 = _mm256_hadd_ps(temp2, temp2);
            __m128 tempLo2 = _mm256_castps256_ps128(temp2);
            __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
            __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum2 = _mm_cvtss_f32(sseSum2);

            sum2 += bias;
            out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


            temp3 = _mm256_hadd_ps(temp3, temp3);
            temp3 = _mm256_hadd_ps(temp3, temp3);
            __m128 tempLo3 = _mm256_castps256_ps128(temp3);
            __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
            __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

            float sum3 = _mm_cvtss_f32(sseSum3);

            sum3 += bias;
            out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


            temp4 = _mm256_hadd_ps(temp4, temp4);
            temp4 = _mm256_hadd_ps(temp4, temp4);
            __m128 tempLo4 = _mm256_castps256_ps128(temp4);
            __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
            __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

            float sum4 = _mm_cvtss_f32(sseSum4);

            sum4 += bias2;
            out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);


            temp5 = _mm256_hadd_ps(temp5, temp5);
            temp5 = _mm256_hadd_ps(temp5, temp5);
            __m128 tempLo5 = _mm256_castps256_ps128(temp5);
            __m128 tempHi5 = _mm256_extractf128_ps(temp5, 1);
            __m128 sseSum5 = _mm_add_ps(tempLo5, tempHi5);

            float sum5 = _mm_cvtss_f32(sseSum5);

            sum5 += bias2;
            out_to_compare_with_FP[out_subscript5] = Relu_float(sum5);


            temp6 = _mm256_hadd_ps(temp6, temp6);
            temp6 = _mm256_hadd_ps(temp6, temp6);
            __m128 tempLo6 = _mm256_castps256_ps128(temp6);
            __m128 tempHi6 = _mm256_extractf128_ps(temp6, 1);
            __m128 sseSum6 = _mm_add_ps(tempLo6, tempHi6);

            float sum6 = _mm_cvtss_f32(sseSum6);

            sum6 += bias2;
            out_to_compare_with_FP[out_subscript6] = Relu_float(sum6);


            temp7 = _mm256_hadd_ps(temp7, temp7);
            temp7 = _mm256_hadd_ps(temp7, temp7);
            __m128 tempLo7 = _mm256_castps256_ps128(temp7);
            __m128 tempHi7 = _mm256_extractf128_ps(temp7, 1);
            __m128 sseSum7 = _mm_add_ps(tempLo7, tempHi7);

            float sum7 = _mm_cvtss_f32(sseSum7);

            sum7 += bias3;
            out_to_compare_with_FP[out_subscript7] = Relu_float(sum7);


            temp8 = _mm256_hadd_ps(temp8, temp8);
            temp8 = _mm256_hadd_ps(temp8, temp8);
            __m128 tempLo8 = _mm256_castps256_ps128(temp8);
            __m128 tempHi8 = _mm256_extractf128_ps(temp8, 1);
            __m128 sseSum8 = _mm_add_ps(tempLo8, tempHi8);

            float sum8 = _mm_cvtss_f32(sseSum8);

            sum8 += bias3;
            out_to_compare_with_FP[out_subscript8] = Relu_float(sum8);


            temp9 = _mm256_hadd_ps(temp9, temp9);
            temp9 = _mm256_hadd_ps(temp9, temp9);
            __m128 tempLo9 = _mm256_castps256_ps128(temp9);
            __m128 tempHi9 = _mm256_extractf128_ps(temp9, 1);
            __m128 sseSum9 = _mm_add_ps(tempLo9, tempHi9);

            float sum9 = _mm_cvtss_f32(sseSum9);

            sum9 += bias3;
            out_to_compare_with_FP[out_subscript9] = Relu_float(sum9);
          }
          // overflow/ fallback x loop
          for (; x < xx + x_tile; x++) {	//Output Width
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

                  unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                    + off_y * Mask_X_dim * Input_depth_dim
                    + off_x * Input_depth_dim
                    + d;
                  unsigned long long int filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                    + off_y * Mask_X_dim * Input_depth_dim
                    + off_x * Input_depth_dim
                    + d;
                  unsigned long long int filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                    + off_y * Mask_X_dim * Input_depth_dim
                    + off_x * Input_depth_dim
                    + d;

                  __m256 s = _mm256_loadu_ps(&in_FP[in_subscript]);

                  __m256 w = _mm256_loadu_ps(&filter_FP[filter_subscript]);
                  __m256 w2 = _mm256_loadu_ps(&filter_FP[filter_subscript2]);
                  __m256 w3 = _mm256_loadu_ps(&filter_FP[filter_subscript3]);

                  temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                  temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s, w2));
                  temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w3));

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
              x * Output_depth_dim
              + (m+1);
            unsigned long long int out_subscript3 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
              y * (Output_depth_dim * Output_X_dim) +
              x * Output_depth_dim
              + (m+2);

            temp = _mm256_hadd_ps(temp, temp);
            temp = _mm256_hadd_ps(temp, temp);
            __m128 tempLo = _mm256_castps256_ps128(temp);
            __m128 tempHi = _mm256_extractf128_ps(temp, 1);
            __m128 sseSum = _mm_add_ps(tempLo, tempHi);

            float sum = _mm_cvtss_f32(sseSum);

            sum += bias;
            out_to_compare_with_FP[out_subscript] = Relu_float(sum);


            temp2 = _mm256_hadd_ps(temp2, temp2);
            temp2 = _mm256_hadd_ps(temp2, temp2);
            __m128 tempLo2 = _mm256_castps256_ps128(temp2);
            __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
            __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum2 = _mm_cvtss_f32(sseSum2);

            sum2 += bias2;
            out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


            temp3 = _mm256_hadd_ps(temp3, temp3);
            temp3 = _mm256_hadd_ps(temp3, temp3);
            __m128 tempLo3 = _mm256_castps256_ps128(temp3);
            __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
            __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

            float sum3 = _mm_cvtss_f32(sseSum3);

            sum3 += bias3;
            out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);

          }
        }
      }
      for (; m < Output_depth_dim; m++) {
        bias = bias_array_FP[m];

        for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
          for (unsigned int x = 0; x < xx + x_tile; x+=4) {	//Output Width
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

                  __m256 w = _mm256_loadu_ps(&filter_FP[filter_subscript]);

                  temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                  temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                  temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                  temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s4, w));
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

            temp = _mm256_hadd_ps(temp, temp);
            temp = _mm256_hadd_ps(temp, temp);
            __m128 tempLo = _mm256_castps256_ps128(temp);
            __m128 tempHi = _mm256_extractf128_ps(temp, 1);
            __m128 sseSum = _mm_add_ps(tempLo, tempHi);

            float sum = _mm_cvtss_f32(sseSum);

            sum += bias;
            out_to_compare_with_FP[out_subscript] = Relu_float(sum);


            temp2 = _mm256_hadd_ps(temp2, temp2);
            temp2 = _mm256_hadd_ps(temp2, temp2);
            __m128 tempLo2 = _mm256_castps256_ps128(temp2);
            __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
            __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum2 = _mm_cvtss_f32(sseSum2);

            sum2 += bias;
            out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


            temp3 = _mm256_hadd_ps(temp3, temp3);
            temp3 = _mm256_hadd_ps(temp3, temp3);
            __m128 tempLo3 = _mm256_castps256_ps128(temp3);
            __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
            __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

            float sum3 = _mm_cvtss_f32(sseSum3);

            sum3 += bias;
            out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


            temp4 = _mm256_hadd_ps(temp4, temp4);
            temp4 = _mm256_hadd_ps(temp4, temp4);
            __m128 tempLo4 = _mm256_castps256_ps128(temp4);
            __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
            __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

            float sum4 = _mm_cvtss_f32(sseSum4);

            sum4 += bias;
            out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);

          }
        }
      }
    }
  }

  #undef x_tile
  printf("\n from optv4 x3 m3, tiled x loop %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// loop tiling on d loop, optimised_layer_v2_unroll_x3m3_hadd_opt_FP()
// tile = 8 ~40 GFLOPS, tile = 16 ~63 GFLOPS, tile = 32 ~68. (check perf for larger input vals)
int optimised_layer_v4_x3m3_tiled_d_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  float bias, bias2, bias3;
  __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9;
  // Calculate loop bounds for unrolled loops
  int m_bound = (Output_depth_dim/ 3) * 3;
  int x_bound = (Output_X_dim/ 3) * 3; 

  #define d_tile 32

  unsigned int m, x;


  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (m = 0; m < m_bound; m+=3) { //channels
      bias = bias_array_FP[m];
      bias2 = bias_array_FP[m+1];
      bias3 = bias_array_FP[m+2];

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
          // temp = 0.0f;

          for (unsigned int dd = 0; dd < Input_depth_dim; dd+=d_tile) {
            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                for (unsigned int d = dd; d < dd + d_tile; d+=8) {

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


                  unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                    + off_y * Mask_X_dim * Input_depth_dim
                    + off_x * Input_depth_dim
                    + d;
                  unsigned long long int filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                    + off_y * Mask_X_dim * Input_depth_dim
                    + off_x * Input_depth_dim
                    + d;
                  unsigned long long int filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                    + off_y * Mask_X_dim * Input_depth_dim
                    + off_x * Input_depth_dim
                    + d;

                  __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                  __m256 s2 = _mm256_load_ps(&in_FP[in_subscript2]);
                  __m256 s3 = _mm256_load_ps(&in_FP[in_subscript3]);

                  __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]); 
                  __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);
                  __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript3]);

                  temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                  temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                  temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                  temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s, w2));
                  temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s2, w2));
                  temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s3, w2));
                  temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w3));
                  temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w3));
                  temp9 = _mm256_add_ps(temp9, _mm256_mul_ps(s3, w3));


                  // float s = in_FP[in_subscript];
                  // float w = filter_FP[filter_subscript];
                  // temp = temp + s * w;
                }
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
            + (m+1);
          unsigned long long int out_subscript5 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+1);
          unsigned long long int out_subscript6 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+2) * Output_depth_dim
            + (m+1);
          unsigned long long int out_subscript7 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+2);
          unsigned long long int out_subscript8 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+2);
          unsigned long long int out_subscript9 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+2) * Output_depth_dim
            + (m+2);



          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
          __m128 tempLo3 = _mm256_castps256_ps128(temp3);
          __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
          __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


          temp4 = _mm256_hadd_ps(temp4, temp4);
          temp4 = _mm256_hadd_ps(temp4, temp4);
          __m128 tempLo4 = _mm256_castps256_ps128(temp4);
          __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
          __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

          float sum4 = _mm_cvtss_f32(sseSum4);

          sum4 += bias2;
          out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);


          temp5 = _mm256_hadd_ps(temp5, temp5);
          temp5 = _mm256_hadd_ps(temp5, temp5);
          __m128 tempLo5 = _mm256_castps256_ps128(temp5);
          __m128 tempHi5 = _mm256_extractf128_ps(temp5, 1);
          __m128 sseSum5 = _mm_add_ps(tempLo5, tempHi5);

          float sum5 = _mm_cvtss_f32(sseSum5);

          sum5 += bias2;
          out_to_compare_with_FP[out_subscript5] = Relu_float(sum5);


          temp6 = _mm256_hadd_ps(temp6, temp6);
          temp6 = _mm256_hadd_ps(temp6, temp6);
          __m128 tempLo6 = _mm256_castps256_ps128(temp6);
          __m128 tempHi6 = _mm256_extractf128_ps(temp6, 1);
          __m128 sseSum6 = _mm_add_ps(tempLo6, tempHi6);

          float sum6 = _mm_cvtss_f32(sseSum6);

          sum6 += bias2;
          out_to_compare_with_FP[out_subscript6] = Relu_float(sum6);


          temp7 = _mm256_hadd_ps(temp7, temp7);
          temp7 = _mm256_hadd_ps(temp7, temp7);
          __m128 tempLo7 = _mm256_castps256_ps128(temp7);
          __m128 tempHi7 = _mm256_extractf128_ps(temp7, 1);
          __m128 sseSum7 = _mm_add_ps(tempLo7, tempHi7);

          float sum7 = _mm_cvtss_f32(sseSum7);

          sum7 += bias3;
          out_to_compare_with_FP[out_subscript7] = Relu_float(sum7);


          temp8 = _mm256_hadd_ps(temp8, temp8);
          temp8 = _mm256_hadd_ps(temp8, temp8);
          __m128 tempLo8 = _mm256_castps256_ps128(temp8);
          __m128 tempHi8 = _mm256_extractf128_ps(temp8, 1);
          __m128 sseSum8 = _mm_add_ps(tempLo8, tempHi8);

          float sum8 = _mm_cvtss_f32(sseSum8);

          sum8 += bias3;
          out_to_compare_with_FP[out_subscript8] = Relu_float(sum8);


          temp9 = _mm256_hadd_ps(temp9, temp9);
          temp9 = _mm256_hadd_ps(temp9, temp9);
          __m128 tempLo9 = _mm256_castps256_ps128(temp9);
          __m128 tempHi9 = _mm256_extractf128_ps(temp9, 1);
          __m128 sseSum9 = _mm_add_ps(tempLo9, tempHi9);

          float sum9 = _mm_cvtss_f32(sseSum9);

          sum9 += bias3;
          out_to_compare_with_FP[out_subscript9] = Relu_float(sum9);
        }
        // overflow/ fallback x loop
        for (; x < Output_X_dim; x++) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          for (unsigned int dd = 0; dd < Input_depth_dim; dd+=d_tile) {
            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                for (unsigned int d = dd; d < dd+d_tile; d+=8) {

                  unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                    + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                    + (x * Stride_X_dim + off_x) * Input_depth_dim
                    + d;

                  unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                    + off_y * Mask_X_dim * Input_depth_dim
                    + off_x * Input_depth_dim
                    + d;
                  unsigned long long int filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                    + off_y * Mask_X_dim * Input_depth_dim
                    + off_x * Input_depth_dim
                    + d;
                  unsigned long long int filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                    + off_y * Mask_X_dim * Input_depth_dim
                    + off_x * Input_depth_dim
                    + d;

                  __m256 s = _mm256_loadu_ps(&in_FP[in_subscript]);

                  __m256 w = _mm256_loadu_ps(&filter_FP[filter_subscript]);
                  __m256 w2 = _mm256_loadu_ps(&filter_FP[filter_subscript2]);
                  __m256 w3 = _mm256_loadu_ps(&filter_FP[filter_subscript3]);

                  temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                  temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s, w2));
                  temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w3));

                  // float s = in_FP[in_subscript];
                  // float w = filter_FP[filter_subscript];
                  // temp = temp + s * w;
                }
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
            + (m+1);
          unsigned long long int out_subscript3 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+2);

          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias2;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
          __m128 tempLo3 = _mm256_castps256_ps128(temp3);
          __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
          __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias3;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);

        }
      }
    }
    for (; m < Output_depth_dim; m++) {
      bias = bias_array_FP[m];

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (unsigned int x = 0; x < Output_X_dim; x+=4) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          temp4 = _mm256_setzero_ps();

          for (unsigned int dd = 0; dd < Input_depth_dim; dd+=d_tile) {
            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                for (unsigned int d = dd; d < dd+d_tile; d+=8) {

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

                  __m256 w = _mm256_loadu_ps(&filter_FP[filter_subscript]);

                  temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                  temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                  temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                  temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s4, w));
                }
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

          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
          __m128 tempLo3 = _mm256_castps256_ps128(temp3);
          __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
          __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


          temp4 = _mm256_hadd_ps(temp4, temp4);
          temp4 = _mm256_hadd_ps(temp4, temp4);
          __m128 tempLo4 = _mm256_castps256_ps128(temp4);
          __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
          __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

          float sum4 = _mm_cvtss_f32(sseSum4);

          sum4 += bias;
          out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);

        }
      }
    }
  }
  #undef d_tile
  printf("\n from optv4 x3 m3, tiling d loop %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// loop tiling on x loop, optimised_layer_v2_unroll_x2m4_hadd_register_pressure_FP()
// tile = 2, ~72 GFLOPS, tile = 4 ~71 GFLOPS, tile = 13 ~67 GFLOPS, 
int optimised_layer_v4_x2m4_tiled_x_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  float bias, bias2, bias3, bias4;
  __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

  #define x_tile 4

  for (unsigned int xx = 0; xx < Output_X_dim; xx+=x_tile) {
    for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
      for (unsigned int m = 0; m < Output_depth_dim; m+=4) { //channels
        bias = bias_array_FP[m];
        bias2 = bias_array_FP[m+1];
        bias3 = bias_array_FP[m+2];
        bias4 = bias_array_FP[m+3];

        for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
          for (unsigned int x = xx; x < xx + x_tile; x+=2) {	//Output Width
            temp = _mm256_setzero_ps();
            temp2 = _mm256_setzero_ps();
            temp3 = _mm256_setzero_ps();
            temp4 = _mm256_setzero_ps();
            temp5 = _mm256_setzero_ps();
            temp6 = _mm256_setzero_ps();
            temp7 = _mm256_setzero_ps();
            temp8 = _mm256_setzero_ps();
            // temp = 0.0f;

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


                  unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                    + off_y * Mask_X_dim * Input_depth_dim
                    + off_x * Input_depth_dim
                    + d;
                  unsigned long long int filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                    + off_y * Mask_X_dim * Input_depth_dim
                    + off_x * Input_depth_dim
                    + d;
                  unsigned long long int filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                    + off_y * Mask_X_dim * Input_depth_dim
                    + off_x * Input_depth_dim
                    + d;
                  unsigned long long int filter_subscript4 = (m+3) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                    + off_y * Mask_X_dim * Input_depth_dim
                    + off_x * Input_depth_dim
                    + d;

                  __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                  __m256 s2 = _mm256_load_ps(&in_FP[in_subscript2]);

                  __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]);
                  __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);
                  __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript3]);
                  __m256 w4 = _mm256_load_ps(&filter_FP[filter_subscript4]);

                  temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                  temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                  temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w2));
                  temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s2, w2));
                  temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s, w3));
                  temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s2, w3));
                  temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w4));
                  temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w4));


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
              x * Output_depth_dim
              + (m+1);
            unsigned long long int out_subscript4 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
              y * (Output_depth_dim * Output_X_dim) +
              (x+1) * Output_depth_dim
              + (m+1);
            unsigned long long int out_subscript5 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
              y * (Output_depth_dim * Output_X_dim) +
              x * Output_depth_dim
              + (m+2);
            unsigned long long int out_subscript6 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
              y * (Output_depth_dim * Output_X_dim) +
              (x+1) * Output_depth_dim
              + (m+2);
            unsigned long long int out_subscript7 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
              y * (Output_depth_dim * Output_X_dim) +
              x * Output_depth_dim
              + (m+3);
            unsigned long long int out_subscript8 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
              y * (Output_depth_dim * Output_X_dim) +
              (x+1) * Output_depth_dim
              + (m+3);



            temp = _mm256_hadd_ps(temp, temp);
            temp = _mm256_hadd_ps(temp, temp);
            __m128 tempLo = _mm256_castps256_ps128(temp);
            __m128 tempHi = _mm256_extractf128_ps(temp, 1);
            __m128 sseSum = _mm_add_ps(tempLo, tempHi);

            float sum = _mm_cvtss_f32(sseSum);

            sum += bias;
            out_to_compare_with_FP[out_subscript] = Relu_float(sum);


            temp2 = _mm256_hadd_ps(temp2, temp2);
            temp2 = _mm256_hadd_ps(temp2, temp2);
            __m128 tempLo2 = _mm256_castps256_ps128(temp2);
            __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
            __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum2 = _mm_cvtss_f32(sseSum2);

            sum2 += bias;
            out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


            temp3 = _mm256_hadd_ps(temp3, temp3);
            temp3 = _mm256_hadd_ps(temp3, temp3);
            tempLo = _mm256_castps256_ps128(temp3);
            tempHi = _mm256_extractf128_ps(temp3, 1);
            sseSum = _mm_add_ps(tempLo, tempHi);

            float sum3 = _mm_cvtss_f32(sseSum);

            sum3 += bias2;
            out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


            temp4 = _mm256_hadd_ps(temp4, temp4);
            temp4 = _mm256_hadd_ps(temp4, temp4);
            tempLo2 = _mm256_castps256_ps128(temp4);
            tempHi2 = _mm256_extractf128_ps(temp4, 1);
            sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum4 = _mm_cvtss_f32(sseSum2);

            sum4 += bias2;
            out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);


            temp5 = _mm256_hadd_ps(temp5, temp5);
            temp5 = _mm256_hadd_ps(temp5, temp5);
            tempLo = _mm256_castps256_ps128(temp5);
            tempHi = _mm256_extractf128_ps(temp5, 1);
            sseSum = _mm_add_ps(tempLo, tempHi);

            float sum5 = _mm_cvtss_f32(sseSum);

            sum5 += bias3;
            out_to_compare_with_FP[out_subscript5] = Relu_float(sum5);


            temp6 = _mm256_hadd_ps(temp6, temp6);
            temp6 = _mm256_hadd_ps(temp6, temp6);
            tempLo2 = _mm256_castps256_ps128(temp6);
            tempHi2 = _mm256_extractf128_ps(temp6, 1);
            sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum6 = _mm_cvtss_f32(sseSum2);

            sum6 += bias3;
            out_to_compare_with_FP[out_subscript6] = Relu_float(sum6);


            temp7 = _mm256_hadd_ps(temp7, temp7);
            temp7 = _mm256_hadd_ps(temp7, temp7);
            tempLo = _mm256_castps256_ps128(temp7);
            tempHi = _mm256_extractf128_ps(temp7, 1);
            sseSum = _mm_add_ps(tempLo, tempHi);

            float sum7 = _mm_cvtss_f32(sseSum);

            sum7 += bias4;
            out_to_compare_with_FP[out_subscript7] = Relu_float(sum7);


            temp8 = _mm256_hadd_ps(temp8, temp8);
            temp8 = _mm256_hadd_ps(temp8, temp8);
            tempLo2 = _mm256_castps256_ps128(temp8);
            tempHi2 = _mm256_extractf128_ps(temp8, 1);
            sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum8 = _mm_cvtss_f32(sseSum2);

            sum8 += bias4;
            out_to_compare_with_FP[out_subscript8] = Relu_float(sum8);
          }
        }
      }
    }
  }
  #undef x_tile
  printf("\n from optv4 x2 m4 hadd - tiled x loop %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// loop tiling on m loop, optimised_layer_v2_unroll_x2m4_hadd_register_pressure_FP()
// tile = 4 ~70 GFLOPS, tile =8 ~72 GFLOPS, tile = 16 ~72 GFLOPS, 
int optimised_layer_v4_x2m4_tiled_m_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  float bias, bias2, bias3, bias4;
  __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;
  #define m_tile 8

  for (unsigned int mm = 0; mm < Output_depth_dim; mm+=m_tile) {
    for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
      for (unsigned int m = mm; m < mm+m_tile; m+=4) { //channels
        bias = bias_array_FP[m];
        bias2 = bias_array_FP[m+1];
        bias3 = bias_array_FP[m+2];
        bias4 = bias_array_FP[m+3];

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
            // temp = 0.0f;

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


                  unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                    + off_y * Mask_X_dim * Input_depth_dim
                    + off_x * Input_depth_dim
                    + d;
                  unsigned long long int filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                    + off_y * Mask_X_dim * Input_depth_dim
                    + off_x * Input_depth_dim
                    + d;
                  unsigned long long int filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                    + off_y * Mask_X_dim * Input_depth_dim
                    + off_x * Input_depth_dim
                    + d;
                  unsigned long long int filter_subscript4 = (m+3) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                    + off_y * Mask_X_dim * Input_depth_dim
                    + off_x * Input_depth_dim
                    + d;

                  __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                  __m256 s2 = _mm256_load_ps(&in_FP[in_subscript2]);

                  __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]);
                  __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);
                  __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript3]);
                  __m256 w4 = _mm256_load_ps(&filter_FP[filter_subscript4]);

                  temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                  temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                  temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w2));
                  temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s2, w2));
                  temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s, w3));
                  temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s2, w3));
                  temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w4));
                  temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w4));


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
              x * Output_depth_dim
              + (m+1);
            unsigned long long int out_subscript4 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
              y * (Output_depth_dim * Output_X_dim) +
              (x+1) * Output_depth_dim
              + (m+1);
            unsigned long long int out_subscript5 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
              y * (Output_depth_dim * Output_X_dim) +
              x * Output_depth_dim
              + (m+2);
            unsigned long long int out_subscript6 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
              y * (Output_depth_dim * Output_X_dim) +
              (x+1) * Output_depth_dim
              + (m+2);
            unsigned long long int out_subscript7 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
              y * (Output_depth_dim * Output_X_dim) +
              x * Output_depth_dim
              + (m+3);
            unsigned long long int out_subscript8 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
              y * (Output_depth_dim * Output_X_dim) +
              (x+1) * Output_depth_dim
              + (m+3);



            temp = _mm256_hadd_ps(temp, temp);
            temp = _mm256_hadd_ps(temp, temp);
            __m128 tempLo = _mm256_castps256_ps128(temp);
            __m128 tempHi = _mm256_extractf128_ps(temp, 1);
            __m128 sseSum = _mm_add_ps(tempLo, tempHi);

            float sum = _mm_cvtss_f32(sseSum);

            sum += bias;
            out_to_compare_with_FP[out_subscript] = Relu_float(sum);


            temp2 = _mm256_hadd_ps(temp2, temp2);
            temp2 = _mm256_hadd_ps(temp2, temp2);
            __m128 tempLo2 = _mm256_castps256_ps128(temp2);
            __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
            __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum2 = _mm_cvtss_f32(sseSum2);

            sum2 += bias;
            out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


            temp3 = _mm256_hadd_ps(temp3, temp3);
            temp3 = _mm256_hadd_ps(temp3, temp3);
            tempLo = _mm256_castps256_ps128(temp3);
            tempHi = _mm256_extractf128_ps(temp3, 1);
            sseSum = _mm_add_ps(tempLo, tempHi);

            float sum3 = _mm_cvtss_f32(sseSum);

            sum3 += bias2;
            out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


            temp4 = _mm256_hadd_ps(temp4, temp4);
            temp4 = _mm256_hadd_ps(temp4, temp4);
            tempLo2 = _mm256_castps256_ps128(temp4);
            tempHi2 = _mm256_extractf128_ps(temp4, 1);
            sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum4 = _mm_cvtss_f32(sseSum2);

            sum4 += bias2;
            out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);


            temp5 = _mm256_hadd_ps(temp5, temp5);
            temp5 = _mm256_hadd_ps(temp5, temp5);
            tempLo = _mm256_castps256_ps128(temp5);
            tempHi = _mm256_extractf128_ps(temp5, 1);
            sseSum = _mm_add_ps(tempLo, tempHi);

            float sum5 = _mm_cvtss_f32(sseSum);

            sum5 += bias3;
            out_to_compare_with_FP[out_subscript5] = Relu_float(sum5);


            temp6 = _mm256_hadd_ps(temp6, temp6);
            temp6 = _mm256_hadd_ps(temp6, temp6);
            tempLo2 = _mm256_castps256_ps128(temp6);
            tempHi2 = _mm256_extractf128_ps(temp6, 1);
            sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum6 = _mm_cvtss_f32(sseSum2);

            sum6 += bias3;
            out_to_compare_with_FP[out_subscript6] = Relu_float(sum6);


            temp7 = _mm256_hadd_ps(temp7, temp7);
            temp7 = _mm256_hadd_ps(temp7, temp7);
            tempLo = _mm256_castps256_ps128(temp7);
            tempHi = _mm256_extractf128_ps(temp7, 1);
            sseSum = _mm_add_ps(tempLo, tempHi);

            float sum7 = _mm_cvtss_f32(sseSum);

            sum7 += bias4;
            out_to_compare_with_FP[out_subscript7] = Relu_float(sum7);


            temp8 = _mm256_hadd_ps(temp8, temp8);
            temp8 = _mm256_hadd_ps(temp8, temp8);
            tempLo2 = _mm256_castps256_ps128(temp8);
            tempHi2 = _mm256_extractf128_ps(temp8, 1);
            sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum8 = _mm_cvtss_f32(sseSum2);

            sum8 += bias4;
            out_to_compare_with_FP[out_subscript8] = Relu_float(sum8);
          }
        }
      }
    }
  }
  #undef m_tile
  printf("\n from optv4 x2 m4 hadd - tiled m loop %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// loop tiling on m loop (swapped m and y loops), optimised_layer_v2_unroll_x2m4_hadd_register_pressure_FP(), also applied loop interchange (perf increase from that?)
// tile = 4 ~70 GFLOPS, tile =8 ~72 GFLOPS, tile = 16 ~73 GFLOPS, 
// *********** according to perf causes less cache misses, most cache misses with tile = 16 *********** actually caused by loop interchange
int optimised_layer_v4_x2m4_tiled_m_loop_interchange_m_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  float bias, bias2, bias3, bias4;
  __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;
  #define m_tile 16

  for (unsigned int mm = 0; mm < Output_depth_dim; mm+=m_tile) {
    for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (unsigned int m = mm; m < mm+m_tile; m+=4) { //channels
          bias = bias_array_FP[m];
          bias2 = bias_array_FP[m+1];
          bias3 = bias_array_FP[m+2];
          bias4 = bias_array_FP[m+3];

          for (unsigned int x = 0; x < Output_X_dim; x+=2) {	//Output Width
            temp = _mm256_setzero_ps();
            temp2 = _mm256_setzero_ps();
            temp3 = _mm256_setzero_ps();
            temp4 = _mm256_setzero_ps();
            temp5 = _mm256_setzero_ps();
            temp6 = _mm256_setzero_ps();
            temp7 = _mm256_setzero_ps();
            temp8 = _mm256_setzero_ps();
            // temp = 0.0f;

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


                  unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                    + off_y * Mask_X_dim * Input_depth_dim
                    + off_x * Input_depth_dim
                    + d;
                  unsigned long long int filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                    + off_y * Mask_X_dim * Input_depth_dim
                    + off_x * Input_depth_dim
                    + d;
                  unsigned long long int filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                    + off_y * Mask_X_dim * Input_depth_dim
                    + off_x * Input_depth_dim
                    + d;
                  unsigned long long int filter_subscript4 = (m+3) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                    + off_y * Mask_X_dim * Input_depth_dim
                    + off_x * Input_depth_dim
                    + d;

                  __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                  __m256 s2 = _mm256_load_ps(&in_FP[in_subscript2]);

                  __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]);
                  __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);
                  __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript3]);
                  __m256 w4 = _mm256_load_ps(&filter_FP[filter_subscript4]);

                  temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                  temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                  temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w2));
                  temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s2, w2));
                  temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s, w3));
                  temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s2, w3));
                  temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w4));
                  temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w4));


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
              x * Output_depth_dim
              + (m+1);
            unsigned long long int out_subscript4 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
              y * (Output_depth_dim * Output_X_dim) +
              (x+1) * Output_depth_dim
              + (m+1);
            unsigned long long int out_subscript5 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
              y * (Output_depth_dim * Output_X_dim) +
              x * Output_depth_dim
              + (m+2);
            unsigned long long int out_subscript6 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
              y * (Output_depth_dim * Output_X_dim) +
              (x+1) * Output_depth_dim
              + (m+2);
            unsigned long long int out_subscript7 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
              y * (Output_depth_dim * Output_X_dim) +
              x * Output_depth_dim
              + (m+3);
            unsigned long long int out_subscript8 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
              y * (Output_depth_dim * Output_X_dim) +
              (x+1) * Output_depth_dim
              + (m+3);



            temp = _mm256_hadd_ps(temp, temp);
            temp = _mm256_hadd_ps(temp, temp);
            __m128 tempLo = _mm256_castps256_ps128(temp);
            __m128 tempHi = _mm256_extractf128_ps(temp, 1);
            __m128 sseSum = _mm_add_ps(tempLo, tempHi);

            float sum = _mm_cvtss_f32(sseSum);

            sum += bias;
            out_to_compare_with_FP[out_subscript] = Relu_float(sum);


            temp2 = _mm256_hadd_ps(temp2, temp2);
            temp2 = _mm256_hadd_ps(temp2, temp2);
            __m128 tempLo2 = _mm256_castps256_ps128(temp2);
            __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
            __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum2 = _mm_cvtss_f32(sseSum2);

            sum2 += bias;
            out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


            temp3 = _mm256_hadd_ps(temp3, temp3);
            temp3 = _mm256_hadd_ps(temp3, temp3);
            tempLo = _mm256_castps256_ps128(temp3);
            tempHi = _mm256_extractf128_ps(temp3, 1);
            sseSum = _mm_add_ps(tempLo, tempHi);

            float sum3 = _mm_cvtss_f32(sseSum);

            sum3 += bias2;
            out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


            temp4 = _mm256_hadd_ps(temp4, temp4);
            temp4 = _mm256_hadd_ps(temp4, temp4);
            tempLo2 = _mm256_castps256_ps128(temp4);
            tempHi2 = _mm256_extractf128_ps(temp4, 1);
            sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum4 = _mm_cvtss_f32(sseSum2);

            sum4 += bias2;
            out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);


            temp5 = _mm256_hadd_ps(temp5, temp5);
            temp5 = _mm256_hadd_ps(temp5, temp5);
            tempLo = _mm256_castps256_ps128(temp5);
            tempHi = _mm256_extractf128_ps(temp5, 1);
            sseSum = _mm_add_ps(tempLo, tempHi);

            float sum5 = _mm_cvtss_f32(sseSum);

            sum5 += bias3;
            out_to_compare_with_FP[out_subscript5] = Relu_float(sum5);


            temp6 = _mm256_hadd_ps(temp6, temp6);
            temp6 = _mm256_hadd_ps(temp6, temp6);
            tempLo2 = _mm256_castps256_ps128(temp6);
            tempHi2 = _mm256_extractf128_ps(temp6, 1);
            sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum6 = _mm_cvtss_f32(sseSum2);

            sum6 += bias3;
            out_to_compare_with_FP[out_subscript6] = Relu_float(sum6);


            temp7 = _mm256_hadd_ps(temp7, temp7);
            temp7 = _mm256_hadd_ps(temp7, temp7);
            tempLo = _mm256_castps256_ps128(temp7);
            tempHi = _mm256_extractf128_ps(temp7, 1);
            sseSum = _mm_add_ps(tempLo, tempHi);

            float sum7 = _mm_cvtss_f32(sseSum);

            sum7 += bias4;
            out_to_compare_with_FP[out_subscript7] = Relu_float(sum7);


            temp8 = _mm256_hadd_ps(temp8, temp8);
            temp8 = _mm256_hadd_ps(temp8, temp8);
            tempLo2 = _mm256_castps256_ps128(temp8);
            tempHi2 = _mm256_extractf128_ps(temp8, 1);
            sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum8 = _mm_cvtss_f32(sseSum2);

            sum8 += bias4;
            out_to_compare_with_FP[out_subscript8] = Relu_float(sum8);
          }
        }
      }
    }
  }
  #undef m_tile
  printf("\n from optv4 x2 m4 hadd - tiled m loop, moved m %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// loop tiling on d loop, optimised_layer_v2_unroll_x2m4_hadd_register_pressure_FP()
// tile = 8 ~30 GFLOPS, tile = 16 ~54 GFLOPS, tile = 32 ~64 GFLOPS, 
int optimised_layer_v4_x2m4_tiled_d_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  float bias, bias2, bias3, bias4;
  __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

  #define d_tile 32

  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int m = 0; m < Output_depth_dim; m+=4) { //channels
      bias = bias_array_FP[m];
      bias2 = bias_array_FP[m+1];
      bias3 = bias_array_FP[m+2];
      bias4 = bias_array_FP[m+3];

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
          // temp = 0.0f;

          for (unsigned int dd = 0; dd < Input_depth_dim; dd+=d_tile) {
            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                for (unsigned int d = dd; d < dd+d_tile; d+=8) {

                  unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                    + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                    + (x * Stride_X_dim + off_x) * Input_depth_dim
                    + d;
                  unsigned long long int in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                    + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                    + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
                    + d;


                  unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                    + off_y * Mask_X_dim * Input_depth_dim
                    + off_x * Input_depth_dim
                    + d;
                  unsigned long long int filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                    + off_y * Mask_X_dim * Input_depth_dim
                    + off_x * Input_depth_dim
                    + d;
                  unsigned long long int filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                    + off_y * Mask_X_dim * Input_depth_dim
                    + off_x * Input_depth_dim
                    + d;
                  unsigned long long int filter_subscript4 = (m+3) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                    + off_y * Mask_X_dim * Input_depth_dim
                    + off_x * Input_depth_dim
                    + d;

                  __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                  __m256 s2 = _mm256_load_ps(&in_FP[in_subscript2]);

                  __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]);
                  __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);
                  __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript3]);
                  __m256 w4 = _mm256_load_ps(&filter_FP[filter_subscript4]);

                  temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                  temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                  temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w2));
                  temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s2, w2));
                  temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s, w3));
                  temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s2, w3));
                  temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w4));
                  temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w4));


                  // float s = in_FP[in_subscript];
                  // float w = filter_FP[filter_subscript];
                  // temp = temp + s * w;
                }
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
            + (m+1);
          unsigned long long int out_subscript4 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+1);
          unsigned long long int out_subscript5 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+2);
          unsigned long long int out_subscript6 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+2);
          unsigned long long int out_subscript7 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+3);
          unsigned long long int out_subscript8 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+3);



          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
           tempLo = _mm256_castps256_ps128(temp3);
           tempHi = _mm256_extractf128_ps(temp3, 1);
           sseSum = _mm_add_ps(tempLo, tempHi);

          float sum3 = _mm_cvtss_f32(sseSum);

          sum3 += bias2;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


          temp4 = _mm256_hadd_ps(temp4, temp4);
          temp4 = _mm256_hadd_ps(temp4, temp4);
           tempLo2 = _mm256_castps256_ps128(temp4);
           tempHi2 = _mm256_extractf128_ps(temp4, 1);
           sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum4 = _mm_cvtss_f32(sseSum2);

          sum4 += bias2;
          out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);


          temp5 = _mm256_hadd_ps(temp5, temp5);
          temp5 = _mm256_hadd_ps(temp5, temp5);
           tempLo = _mm256_castps256_ps128(temp5);
           tempHi = _mm256_extractf128_ps(temp5, 1);
           sseSum = _mm_add_ps(tempLo, tempHi);

          float sum5 = _mm_cvtss_f32(sseSum);

          sum5 += bias3;
          out_to_compare_with_FP[out_subscript5] = Relu_float(sum5);


          temp6 = _mm256_hadd_ps(temp6, temp6);
          temp6 = _mm256_hadd_ps(temp6, temp6);
           tempLo2 = _mm256_castps256_ps128(temp6);
           tempHi2 = _mm256_extractf128_ps(temp6, 1);
           sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum6 = _mm_cvtss_f32(sseSum2);

          sum6 += bias3;
          out_to_compare_with_FP[out_subscript6] = Relu_float(sum6);


          temp7 = _mm256_hadd_ps(temp7, temp7);
          temp7 = _mm256_hadd_ps(temp7, temp7);
           tempLo = _mm256_castps256_ps128(temp7);
           tempHi = _mm256_extractf128_ps(temp7, 1);
           sseSum = _mm_add_ps(tempLo, tempHi);

          float sum7 = _mm_cvtss_f32(sseSum);

          sum7 += bias4;
          out_to_compare_with_FP[out_subscript7] = Relu_float(sum7);


          temp8 = _mm256_hadd_ps(temp8, temp8);
          temp8 = _mm256_hadd_ps(temp8, temp8);
           tempLo2 = _mm256_castps256_ps128(temp8);
           tempHi2 = _mm256_extractf128_ps(temp8, 1);
           sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum8 = _mm_cvtss_f32(sseSum2);

          sum8 += bias4;
          out_to_compare_with_FP[out_subscript8] = Relu_float(sum8);
        }
      }
    }
  }
  #undef d_tile
  printf("\n from optv4 x2 m4 hadd - tiled d loop %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// loop tiling on m loop (swapped m and y loops) also x loop, optimised_layer_v2_unroll_x2m4_hadd_register_pressure_FP()
// tile =  ~ GFLOPS,
int optimised_layer_v4_x2m4_tiled_x_m_loop_interchange_m_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  float bias, bias2, bias3, bias4;
  __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;
  #define m_tile 8
  #define x_tile 4

  for (unsigned int mm = 0; mm < Output_depth_dim; mm+=m_tile) {
    for (unsigned int xx = 0; xx < Output_X_dim; xx+=x_tile) {
      for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
        for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
          for (unsigned int m = mm; m < mm+m_tile; m+=4) { //channels
            bias = bias_array_FP[m];
            bias2 = bias_array_FP[m+1];
            bias3 = bias_array_FP[m+2];
            bias4 = bias_array_FP[m+3];

            for (unsigned int x = xx; x < xx+x_tile; x+=2) {	//Output Width
              temp = _mm256_setzero_ps();
              temp2 = _mm256_setzero_ps();
              temp3 = _mm256_setzero_ps();
              temp4 = _mm256_setzero_ps();
              temp5 = _mm256_setzero_ps();
              temp6 = _mm256_setzero_ps();
              temp7 = _mm256_setzero_ps();
              temp8 = _mm256_setzero_ps();
              // temp = 0.0f;

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


                    unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                      + off_y * Mask_X_dim * Input_depth_dim
                      + off_x * Input_depth_dim
                      + d;
                    unsigned long long int filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                      + off_y * Mask_X_dim * Input_depth_dim
                      + off_x * Input_depth_dim
                      + d;
                    unsigned long long int filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                      + off_y * Mask_X_dim * Input_depth_dim
                      + off_x * Input_depth_dim
                      + d;
                    unsigned long long int filter_subscript4 = (m+3) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                      + off_y * Mask_X_dim * Input_depth_dim
                      + off_x * Input_depth_dim
                      + d;

                    __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                    __m256 s2 = _mm256_load_ps(&in_FP[in_subscript2]);

                    __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]);
                    __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);
                    __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript3]);
                    __m256 w4 = _mm256_load_ps(&filter_FP[filter_subscript4]);

                    temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                    temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                    temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w2));
                    temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s2, w2));
                    temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s, w3));
                    temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s2, w3));
                    temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w4));
                    temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w4));


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
                x * Output_depth_dim
                + (m+1);
              unsigned long long int out_subscript4 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
                y * (Output_depth_dim * Output_X_dim) +
                (x+1) * Output_depth_dim
                + (m+1);
              unsigned long long int out_subscript5 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
                y * (Output_depth_dim * Output_X_dim) +
                x * Output_depth_dim
                + (m+2);
              unsigned long long int out_subscript6 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
                y * (Output_depth_dim * Output_X_dim) +
                (x+1) * Output_depth_dim
                + (m+2);
              unsigned long long int out_subscript7 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
                y * (Output_depth_dim * Output_X_dim) +
                x * Output_depth_dim
                + (m+3);
              unsigned long long int out_subscript8 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
                y * (Output_depth_dim * Output_X_dim) +
                (x+1) * Output_depth_dim
                + (m+3);



              temp = _mm256_hadd_ps(temp, temp);
              temp = _mm256_hadd_ps(temp, temp);
              __m128 tempLo = _mm256_castps256_ps128(temp);
              __m128 tempHi = _mm256_extractf128_ps(temp, 1);
              __m128 sseSum = _mm_add_ps(tempLo, tempHi);

              float sum = _mm_cvtss_f32(sseSum);

              sum += bias;
              out_to_compare_with_FP[out_subscript] = Relu_float(sum);


              temp2 = _mm256_hadd_ps(temp2, temp2);
              temp2 = _mm256_hadd_ps(temp2, temp2);
              __m128 tempLo2 = _mm256_castps256_ps128(temp2);
              __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
              __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

              float sum2 = _mm_cvtss_f32(sseSum2);

              sum2 += bias;
              out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


              temp3 = _mm256_hadd_ps(temp3, temp3);
              temp3 = _mm256_hadd_ps(temp3, temp3);
              tempLo = _mm256_castps256_ps128(temp3);
              tempHi = _mm256_extractf128_ps(temp3, 1);
              sseSum = _mm_add_ps(tempLo, tempHi);

              float sum3 = _mm_cvtss_f32(sseSum);

              sum3 += bias2;
              out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


              temp4 = _mm256_hadd_ps(temp4, temp4);
              temp4 = _mm256_hadd_ps(temp4, temp4);
              tempLo2 = _mm256_castps256_ps128(temp4);
              tempHi2 = _mm256_extractf128_ps(temp4, 1);
              sseSum2 = _mm_add_ps(tempLo2, tempHi2);

              float sum4 = _mm_cvtss_f32(sseSum2);

              sum4 += bias2;
              out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);


              temp5 = _mm256_hadd_ps(temp5, temp5);
              temp5 = _mm256_hadd_ps(temp5, temp5);
              tempLo = _mm256_castps256_ps128(temp5);
              tempHi = _mm256_extractf128_ps(temp5, 1);
              sseSum = _mm_add_ps(tempLo, tempHi);

              float sum5 = _mm_cvtss_f32(sseSum);

              sum5 += bias3;
              out_to_compare_with_FP[out_subscript5] = Relu_float(sum5);


              temp6 = _mm256_hadd_ps(temp6, temp6);
              temp6 = _mm256_hadd_ps(temp6, temp6);
              tempLo2 = _mm256_castps256_ps128(temp6);
              tempHi2 = _mm256_extractf128_ps(temp6, 1);
              sseSum2 = _mm_add_ps(tempLo2, tempHi2);

              float sum6 = _mm_cvtss_f32(sseSum2);

              sum6 += bias3;
              out_to_compare_with_FP[out_subscript6] = Relu_float(sum6);


              temp7 = _mm256_hadd_ps(temp7, temp7);
              temp7 = _mm256_hadd_ps(temp7, temp7);
              tempLo = _mm256_castps256_ps128(temp7);
              tempHi = _mm256_extractf128_ps(temp7, 1);
              sseSum = _mm_add_ps(tempLo, tempHi);

              float sum7 = _mm_cvtss_f32(sseSum);

              sum7 += bias4;
              out_to_compare_with_FP[out_subscript7] = Relu_float(sum7);


              temp8 = _mm256_hadd_ps(temp8, temp8);
              temp8 = _mm256_hadd_ps(temp8, temp8);
              tempLo2 = _mm256_castps256_ps128(temp8);
              tempHi2 = _mm256_extractf128_ps(temp8, 1);
              sseSum2 = _mm_add_ps(tempLo2, tempHi2);

              float sum8 = _mm_cvtss_f32(sseSum2);

              sum8 += bias4;
              out_to_compare_with_FP[out_subscript8] = Relu_float(sum8);
            }
          }
        }
      }
    }
  }
  #undef m_tile
  #undef x_tile
  printf("\n from optv4 x2 m4 hadd - tiled m loop, moved m %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// loop tiling not super worth since more computational bound as opposed to memory? check arithmetic intensity with likwid/ vTune
// test applying again after parallelisation, as each core has dedicated L1/ L2 cache on zen 3






// v5 test loop interchange


// loop interchange on m and y - optimised_layer_v2_unroll_x2m4_hadd_register_pressure_FP()
// ~72/73 GFLOPS, 
int optimised_layer_v5_x2m4_loop_interchange_m_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  float bias, bias2, bias3, bias4;
  __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
      for (unsigned int m = 0; m < Output_depth_dim; m+=4) { //channels
        bias = bias_array_FP[m];
        bias2 = bias_array_FP[m+1];
        bias3 = bias_array_FP[m+2];
        bias4 = bias_array_FP[m+3];

        for (unsigned int x = 0; x < Output_X_dim; x+=2) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          temp4 = _mm256_setzero_ps();
          temp5 = _mm256_setzero_ps();
          temp6 = _mm256_setzero_ps();
          temp7 = _mm256_setzero_ps();
          temp8 = _mm256_setzero_ps();
          // temp = 0.0f;

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


                unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript4 = (m+3) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;

                __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                __m256 s2 = _mm256_load_ps(&in_FP[in_subscript2]);

                __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]);
                __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);
                __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript3]);
                __m256 w4 = _mm256_load_ps(&filter_FP[filter_subscript4]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w2));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s2, w2));
                temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s, w3));
                temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s2, w3));
                temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w4));
                temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w4));


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
            x * Output_depth_dim
            + (m+1);
          unsigned long long int out_subscript4 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+1);
          unsigned long long int out_subscript5 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+2);
          unsigned long long int out_subscript6 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+2);
          unsigned long long int out_subscript7 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+3);
          unsigned long long int out_subscript8 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+3);



          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
           tempLo = _mm256_castps256_ps128(temp3);
           tempHi = _mm256_extractf128_ps(temp3, 1);
           sseSum = _mm_add_ps(tempLo, tempHi);

          float sum3 = _mm_cvtss_f32(sseSum);

          sum3 += bias2;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


          temp4 = _mm256_hadd_ps(temp4, temp4);
          temp4 = _mm256_hadd_ps(temp4, temp4);
           tempLo2 = _mm256_castps256_ps128(temp4);
           tempHi2 = _mm256_extractf128_ps(temp4, 1);
           sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum4 = _mm_cvtss_f32(sseSum2);

          sum4 += bias2;
          out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);


          temp5 = _mm256_hadd_ps(temp5, temp5);
          temp5 = _mm256_hadd_ps(temp5, temp5);
           tempLo = _mm256_castps256_ps128(temp5);
           tempHi = _mm256_extractf128_ps(temp5, 1);
           sseSum = _mm_add_ps(tempLo, tempHi);

          float sum5 = _mm_cvtss_f32(sseSum);

          sum5 += bias3;
          out_to_compare_with_FP[out_subscript5] = Relu_float(sum5);


          temp6 = _mm256_hadd_ps(temp6, temp6);
          temp6 = _mm256_hadd_ps(temp6, temp6);
           tempLo2 = _mm256_castps256_ps128(temp6);
           tempHi2 = _mm256_extractf128_ps(temp6, 1);
           sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum6 = _mm_cvtss_f32(sseSum2);

          sum6 += bias3;
          out_to_compare_with_FP[out_subscript6] = Relu_float(sum6);


          temp7 = _mm256_hadd_ps(temp7, temp7);
          temp7 = _mm256_hadd_ps(temp7, temp7);
           tempLo = _mm256_castps256_ps128(temp7);
           tempHi = _mm256_extractf128_ps(temp7, 1);
           sseSum = _mm_add_ps(tempLo, tempHi);

          float sum7 = _mm_cvtss_f32(sseSum);

          sum7 += bias4;
          out_to_compare_with_FP[out_subscript7] = Relu_float(sum7);


          temp8 = _mm256_hadd_ps(temp8, temp8);
          temp8 = _mm256_hadd_ps(temp8, temp8);
           tempLo2 = _mm256_castps256_ps128(temp8);
           tempHi2 = _mm256_extractf128_ps(temp8, 1);
           sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum8 = _mm_cvtss_f32(sseSum2);

          sum8 += bias4;
          out_to_compare_with_FP[out_subscript8] = Relu_float(sum8);
        }
      }
    }
  }

  printf("\n from optv5 x2 m4 - loop interchange %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// loop interchange on m and y - optimised_layer_v2_unroll_x3m3_hadd_opt_FP()
// ~76 GFLOPS, same perf but less than half the cache misses
int optimised_layer_v5_x3m3_loop_interchange_m_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  float bias, bias2, bias3;
  __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9;
  // Calculate loop bounds for unrolled loops
  int m_bound = (Output_depth_dim/ 3) * 3;
  int x_bound = (Output_X_dim/ 3) * 3; 

  unsigned int m, x;


  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
      for (m = 0; m < m_bound; m+=3) { //channels
        bias = bias_array_FP[m];
        bias2 = bias_array_FP[m+1];
        bias3 = bias_array_FP[m+2];

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
          // temp = 0.0f;

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


                unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;

                __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                __m256 s2 = _mm256_load_ps(&in_FP[in_subscript2]);
                __m256 s3 = _mm256_load_ps(&in_FP[in_subscript3]);

                __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]); 
                __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);
                __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript3]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s, w2));
                temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s2, w2));
                temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s3, w2));
                temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w3));
                temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w3));
                temp9 = _mm256_add_ps(temp9, _mm256_mul_ps(s3, w3));


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
            x * Output_depth_dim
            + (m+1);
          unsigned long long int out_subscript5 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+1);
          unsigned long long int out_subscript6 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+2) * Output_depth_dim
            + (m+1);
          unsigned long long int out_subscript7 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+2);
          unsigned long long int out_subscript8 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+1) * Output_depth_dim
            + (m+2);
          unsigned long long int out_subscript9 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            (x+2) * Output_depth_dim
            + (m+2);



          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
          __m128 tempLo3 = _mm256_castps256_ps128(temp3);
          __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
          __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


          temp4 = _mm256_hadd_ps(temp4, temp4);
          temp4 = _mm256_hadd_ps(temp4, temp4);
          __m128 tempLo4 = _mm256_castps256_ps128(temp4);
          __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
          __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

          float sum4 = _mm_cvtss_f32(sseSum4);

          sum4 += bias2;
          out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);


          temp5 = _mm256_hadd_ps(temp5, temp5);
          temp5 = _mm256_hadd_ps(temp5, temp5);
          __m128 tempLo5 = _mm256_castps256_ps128(temp5);
          __m128 tempHi5 = _mm256_extractf128_ps(temp5, 1);
          __m128 sseSum5 = _mm_add_ps(tempLo5, tempHi5);

          float sum5 = _mm_cvtss_f32(sseSum5);

          sum5 += bias2;
          out_to_compare_with_FP[out_subscript5] = Relu_float(sum5);


          temp6 = _mm256_hadd_ps(temp6, temp6);
          temp6 = _mm256_hadd_ps(temp6, temp6);
          __m128 tempLo6 = _mm256_castps256_ps128(temp6);
          __m128 tempHi6 = _mm256_extractf128_ps(temp6, 1);
          __m128 sseSum6 = _mm_add_ps(tempLo6, tempHi6);

          float sum6 = _mm_cvtss_f32(sseSum6);

          sum6 += bias2;
          out_to_compare_with_FP[out_subscript6] = Relu_float(sum6);


          temp7 = _mm256_hadd_ps(temp7, temp7);
          temp7 = _mm256_hadd_ps(temp7, temp7);
          __m128 tempLo7 = _mm256_castps256_ps128(temp7);
          __m128 tempHi7 = _mm256_extractf128_ps(temp7, 1);
          __m128 sseSum7 = _mm_add_ps(tempLo7, tempHi7);

          float sum7 = _mm_cvtss_f32(sseSum7);

          sum7 += bias3;
          out_to_compare_with_FP[out_subscript7] = Relu_float(sum7);


          temp8 = _mm256_hadd_ps(temp8, temp8);
          temp8 = _mm256_hadd_ps(temp8, temp8);
          __m128 tempLo8 = _mm256_castps256_ps128(temp8);
          __m128 tempHi8 = _mm256_extractf128_ps(temp8, 1);
          __m128 sseSum8 = _mm_add_ps(tempLo8, tempHi8);

          float sum8 = _mm_cvtss_f32(sseSum8);

          sum8 += bias3;
          out_to_compare_with_FP[out_subscript8] = Relu_float(sum8);


          temp9 = _mm256_hadd_ps(temp9, temp9);
          temp9 = _mm256_hadd_ps(temp9, temp9);
          __m128 tempLo9 = _mm256_castps256_ps128(temp9);
          __m128 tempHi9 = _mm256_extractf128_ps(temp9, 1);
          __m128 sseSum9 = _mm_add_ps(tempLo9, tempHi9);

          float sum9 = _mm_cvtss_f32(sseSum9);

          sum9 += bias3;
          out_to_compare_with_FP[out_subscript9] = Relu_float(sum9);
        }
        // overflow/ fallback x loop
        for (; x < Output_X_dim; x++) {	//Output Width
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

                unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;
                unsigned long long int filter_subscript3 = (m+2) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;

                __m256 s = _mm256_loadu_ps(&in_FP[in_subscript]);

                __m256 w = _mm256_loadu_ps(&filter_FP[filter_subscript]);
                __m256 w2 = _mm256_loadu_ps(&filter_FP[filter_subscript2]);
                __m256 w3 = _mm256_loadu_ps(&filter_FP[filter_subscript3]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s, w2));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w3));

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
            x * Output_depth_dim
            + (m+1);
          unsigned long long int out_subscript3 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + (m+2);

          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias2;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
          __m128 tempLo3 = _mm256_castps256_ps128(temp3);
          __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
          __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias3;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);

        }
      }
      for (; m < Output_depth_dim; m++) {
        bias = bias_array_FP[m];

        for (unsigned int x = 0; x < Output_X_dim; x+=4) {	//Output Width
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

                __m256 w = _mm256_loadu_ps(&filter_FP[filter_subscript]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s4, w));
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

          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
          __m128 tempLo3 = _mm256_castps256_ps128(temp3);
          __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
          __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


          temp4 = _mm256_hadd_ps(temp4, temp4);
          temp4 = _mm256_hadd_ps(temp4, temp4);
          __m128 tempLo4 = _mm256_castps256_ps128(temp4);
          __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
          __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

          float sum4 = _mm_cvtss_f32(sseSum4);

          sum4 += bias;
          out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);

        }
      }
    }
  }

  printf("\n from optv5 x3 m3 hadd opt - loop interchange %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}






// v6 apply strength reduction - using v5 funcs as base
// test using macro functions 


// applied left shift where possible - optimised_layer_v5_x2m4_loop_interchange_m_FP()
// ~73 GFLOPS, same perf as before. maybe optimised out by -O3? asm shows shl isn't added by compiler automatically
int optimised_layer_v6_x2m4_left_shift_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  #define left_shift(x) (__builtin_ctz(x))  // inline function effectively using inline assembly to return num of trailing zeros (i.e. 128 returns 7, 2 * 128 == 2 << 7)

  float bias, bias2, bias3, bias4;
  __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
      for (unsigned int m = 0; m < Output_depth_dim; m+=4) { //channels
        bias = bias_array_FP[m];
        bias2 = bias_array_FP[m+1];
        bias3 = bias_array_FP[m+2];
        bias4 = bias_array_FP[m+3];

        for (unsigned int x = 0; x < Output_X_dim; x+=2) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          temp4 = _mm256_setzero_ps();
          temp5 = _mm256_setzero_ps();
          temp6 = _mm256_setzero_ps();
          temp7 = _mm256_setzero_ps();
          temp8 = _mm256_setzero_ps();
          // temp = 0.0f;

          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim << left_shift(Input_depth_dim))
                  + ((y * Stride_Y_dim + off_y) * Input_X_dim  << left_shift(Input_depth_dim))
                  + ((x * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                  + d;
                unsigned long long int in_subscript2 = b * (Input_Y_dim * Input_X_dim << left_shift(Input_depth_dim))
                  + ((y * Stride_Y_dim + off_y) * Input_X_dim << left_shift(Input_depth_dim))
                  + (((x+1) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                  + d;


                unsigned long long int filter_subscript = (m * Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))
                  + (off_y * Mask_X_dim << left_shift(Input_depth_dim))
                  + (off_x << left_shift(Input_depth_dim))
                  + d;
                unsigned long long int filter_subscript2 = ((m+1) * Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))
                  + (off_y * Mask_X_dim << left_shift(Input_depth_dim))
                  + (off_x << left_shift(Input_depth_dim))
                  + d;
                unsigned long long int filter_subscript3 = ((m+2) * Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))
                  + (off_y * Mask_X_dim << left_shift(Input_depth_dim))
                  + (off_x << left_shift(Input_depth_dim))
                  + d;
                unsigned long long int filter_subscript4 = ((m+3) * Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))
                  + (off_y * Mask_X_dim << left_shift(Input_depth_dim))
                  + (off_x << left_shift(Input_depth_dim))
                  + d;

                __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                __m256 s2 = _mm256_load_ps(&in_FP[in_subscript2]);

                __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]);
                __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);
                __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript3]);
                __m256 w4 = _mm256_load_ps(&filter_FP[filter_subscript4]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w2));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s2, w2));
                temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s, w3));
                temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s2, w3));
                temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w4));
                temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w4));


                // float s = in_FP[in_subscript];
                // float w = filter_FP[filter_subscript];
                // temp = temp + s * w;
              }
            }
          }


          unsigned long long int out_subscript = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
            y * (Output_X_dim << left_shift(Output_depth_dim)) +
            (x << left_shift(Output_depth_dim))
            + m;
          unsigned long long int out_subscript2 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
            y * (Output_X_dim << left_shift(Output_depth_dim)) +
            ((x+1) << left_shift(Output_depth_dim))
            + m;
          unsigned long long int out_subscript3 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
            y * (Output_X_dim << left_shift(Output_depth_dim)) +
            (x << left_shift(Output_depth_dim))
            + (m+1);
          unsigned long long int out_subscript4 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
            y * (Output_X_dim << left_shift(Output_depth_dim)) +
            ((x+1) << left_shift(Output_depth_dim))
            + (m+1);
          unsigned long long int out_subscript5 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
            y * (Output_X_dim << left_shift(Output_depth_dim)) +
            (x << left_shift(Output_depth_dim))
            + (m+2);
          unsigned long long int out_subscript6 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
            y * (Output_X_dim << left_shift(Output_depth_dim)) +
            ((x+1) << left_shift(Output_depth_dim))
            + (m+2);
          unsigned long long int out_subscript7 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
            y * (Output_X_dim << left_shift(Output_depth_dim)) +
            (x << left_shift(Output_depth_dim))
            + (m+3);
          unsigned long long int out_subscript8 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
            y * (Output_X_dim << left_shift(Output_depth_dim)) +
            ((x+1) << left_shift(Output_depth_dim))
            + (m+3);



          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
           tempLo = _mm256_castps256_ps128(temp3);
           tempHi = _mm256_extractf128_ps(temp3, 1);
           sseSum = _mm_add_ps(tempLo, tempHi);

          float sum3 = _mm_cvtss_f32(sseSum);

          sum3 += bias2;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


          temp4 = _mm256_hadd_ps(temp4, temp4);
          temp4 = _mm256_hadd_ps(temp4, temp4);
           tempLo2 = _mm256_castps256_ps128(temp4);
           tempHi2 = _mm256_extractf128_ps(temp4, 1);
           sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum4 = _mm_cvtss_f32(sseSum2);

          sum4 += bias2;
          out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);


          temp5 = _mm256_hadd_ps(temp5, temp5);
          temp5 = _mm256_hadd_ps(temp5, temp5);
           tempLo = _mm256_castps256_ps128(temp5);
           tempHi = _mm256_extractf128_ps(temp5, 1);
           sseSum = _mm_add_ps(tempLo, tempHi);

          float sum5 = _mm_cvtss_f32(sseSum);

          sum5 += bias3;
          out_to_compare_with_FP[out_subscript5] = Relu_float(sum5);


          temp6 = _mm256_hadd_ps(temp6, temp6);
          temp6 = _mm256_hadd_ps(temp6, temp6);
           tempLo2 = _mm256_castps256_ps128(temp6);
           tempHi2 = _mm256_extractf128_ps(temp6, 1);
           sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum6 = _mm_cvtss_f32(sseSum2);

          sum6 += bias3;
          out_to_compare_with_FP[out_subscript6] = Relu_float(sum6);


          temp7 = _mm256_hadd_ps(temp7, temp7);
          temp7 = _mm256_hadd_ps(temp7, temp7);
           tempLo = _mm256_castps256_ps128(temp7);
           tempHi = _mm256_extractf128_ps(temp7, 1);
           sseSum = _mm_add_ps(tempLo, tempHi);

          float sum7 = _mm_cvtss_f32(sseSum);

          sum7 += bias4;
          out_to_compare_with_FP[out_subscript7] = Relu_float(sum7);


          temp8 = _mm256_hadd_ps(temp8, temp8);
          temp8 = _mm256_hadd_ps(temp8, temp8);
           tempLo2 = _mm256_castps256_ps128(temp8);
           tempHi2 = _mm256_extractf128_ps(temp8, 1);
           sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum8 = _mm_cvtss_f32(sseSum2);

          sum8 += bias4;
          out_to_compare_with_FP[out_subscript8] = Relu_float(sum8);
        }
      }
    }
  }

  printf("\n from optv6 x2 m4 - strength reduction %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// store in/output_depth_dim in const vars in case compiler doesn't replace left_shift(x) macros with constant values - optimised_layer_v6_x2m4_left_shift_FP
// ~71 GFLOPS, 
int optimised_layer_v6_x2m4_left_shift_opt_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  #define left_shift(x) (__builtin_ctz(x))  // inline function effectively using inline assembly to return num of trailing zeros (i.e. 128 returns 7, 2 * 128 == 2 << 7)

  float bias, bias2, bias3, bias4;
  __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;
  const unsigned long long int in_depth_lshift = left_shift(Input_depth_dim);     // store as const unsigned long long int to avoid expensive conversion costs
  const unsigned long long int out_depth_lshift = left_shift(Output_depth_dim);   // store as const unsigned long long int to avoid expensive conversion costs

  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
      for (unsigned int m = 0; m < Output_depth_dim; m+=4) { //channels
        bias = bias_array_FP[m];
        bias2 = bias_array_FP[m+1];
        bias3 = bias_array_FP[m+2];
        bias4 = bias_array_FP[m+3];

        for (unsigned int x = 0; x < Output_X_dim; x+=2) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          temp4 = _mm256_setzero_ps();
          temp5 = _mm256_setzero_ps();
          temp6 = _mm256_setzero_ps();
          temp7 = _mm256_setzero_ps();
          temp8 = _mm256_setzero_ps();
          // temp = 0.0f;

          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim << in_depth_lshift)
                  + ((y * Stride_Y_dim + off_y) * Input_X_dim  << in_depth_lshift)
                  + ((x * Stride_X_dim + off_x) << in_depth_lshift)
                  + d;
                unsigned long long int in_subscript2 = b * (Input_Y_dim * Input_X_dim << in_depth_lshift)
                  + ((y * Stride_Y_dim + off_y) * Input_X_dim << in_depth_lshift)
                  + (((x+1) * Stride_X_dim + off_x) << in_depth_lshift)
                  + d;


                unsigned long long int filter_subscript = (m * Mask_Y_dim * Mask_X_dim << in_depth_lshift)
                  + (off_y * Mask_X_dim << in_depth_lshift)
                  + (off_x << in_depth_lshift)
                  + d;
                unsigned long long int filter_subscript2 = ((m+1) * Mask_Y_dim * Mask_X_dim << in_depth_lshift)
                  + (off_y * Mask_X_dim << in_depth_lshift)
                  + (off_x << in_depth_lshift)
                  + d;
                unsigned long long int filter_subscript3 = ((m+2) * Mask_Y_dim * Mask_X_dim << in_depth_lshift)
                  + (off_y * Mask_X_dim << in_depth_lshift)
                  + (off_x << in_depth_lshift)
                  + d;
                unsigned long long int filter_subscript4 = ((m+3) * Mask_Y_dim * Mask_X_dim << in_depth_lshift)
                  + (off_y * Mask_X_dim << in_depth_lshift)
                  + (off_x << in_depth_lshift)
                  + d;

                __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                __m256 s2 = _mm256_load_ps(&in_FP[in_subscript2]);

                __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]);
                __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);
                __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript3]);
                __m256 w4 = _mm256_load_ps(&filter_FP[filter_subscript4]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w2));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s2, w2));
                temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s, w3));
                temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s2, w3));
                temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w4));
                temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w4));


                // float s = in_FP[in_subscript];
                // float w = filter_FP[filter_subscript];
                // temp = temp + s * w;
              }
            }
          }


          unsigned long long int out_subscript = b * (Output_X_dim * Output_Y_dim << out_depth_lshift) +
            y * (Output_X_dim << out_depth_lshift) +
            (x << out_depth_lshift)
            + m;
          unsigned long long int out_subscript2 = b * (Output_X_dim * Output_Y_dim << out_depth_lshift) +
            y * (Output_X_dim << out_depth_lshift) +
            ((x+1) << out_depth_lshift)
            + m;
          unsigned long long int out_subscript3 = b * (Output_X_dim * Output_Y_dim << out_depth_lshift) +
            y * (Output_X_dim << out_depth_lshift) +
            (x << out_depth_lshift)
            + (m+1);
          unsigned long long int out_subscript4 = b * (Output_X_dim * Output_Y_dim << out_depth_lshift) +
            y * (Output_X_dim << out_depth_lshift) +
            ((x+1) << out_depth_lshift)
            + (m+1);
          unsigned long long int out_subscript5 = b * (Output_X_dim * Output_Y_dim << out_depth_lshift) +
            y * (Output_X_dim << out_depth_lshift) +
            (x << out_depth_lshift)
            + (m+2);
          unsigned long long int out_subscript6 = b * (Output_X_dim * Output_Y_dim << out_depth_lshift) +
            y * (Output_X_dim << out_depth_lshift) +
            ((x+1) << out_depth_lshift)
            + (m+2);
          unsigned long long int out_subscript7 = b * (Output_X_dim * Output_Y_dim << out_depth_lshift) +
            y * (Output_X_dim << out_depth_lshift) +
            (x << out_depth_lshift)
            + (m+3);
          unsigned long long int out_subscript8 = b * (Output_X_dim * Output_Y_dim << out_depth_lshift) +
            y * (Output_X_dim << out_depth_lshift) +
            ((x+1) << out_depth_lshift)
            + (m+3);



          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
           tempLo = _mm256_castps256_ps128(temp3);
           tempHi = _mm256_extractf128_ps(temp3, 1);
           sseSum = _mm_add_ps(tempLo, tempHi);

          float sum3 = _mm_cvtss_f32(sseSum);

          sum3 += bias2;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


          temp4 = _mm256_hadd_ps(temp4, temp4);
          temp4 = _mm256_hadd_ps(temp4, temp4);
           tempLo2 = _mm256_castps256_ps128(temp4);
           tempHi2 = _mm256_extractf128_ps(temp4, 1);
           sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum4 = _mm_cvtss_f32(sseSum2);

          sum4 += bias2;
          out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);


          temp5 = _mm256_hadd_ps(temp5, temp5);
          temp5 = _mm256_hadd_ps(temp5, temp5);
           tempLo = _mm256_castps256_ps128(temp5);
           tempHi = _mm256_extractf128_ps(temp5, 1);
           sseSum = _mm_add_ps(tempLo, tempHi);

          float sum5 = _mm_cvtss_f32(sseSum);

          sum5 += bias3;
          out_to_compare_with_FP[out_subscript5] = Relu_float(sum5);


          temp6 = _mm256_hadd_ps(temp6, temp6);
          temp6 = _mm256_hadd_ps(temp6, temp6);
           tempLo2 = _mm256_castps256_ps128(temp6);
           tempHi2 = _mm256_extractf128_ps(temp6, 1);
           sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum6 = _mm_cvtss_f32(sseSum2);

          sum6 += bias3;
          out_to_compare_with_FP[out_subscript6] = Relu_float(sum6);


          temp7 = _mm256_hadd_ps(temp7, temp7);
          temp7 = _mm256_hadd_ps(temp7, temp7);
           tempLo = _mm256_castps256_ps128(temp7);
           tempHi = _mm256_extractf128_ps(temp7, 1);
           sseSum = _mm_add_ps(tempLo, tempHi);

          float sum7 = _mm_cvtss_f32(sseSum);

          sum7 += bias4;
          out_to_compare_with_FP[out_subscript7] = Relu_float(sum7);


          temp8 = _mm256_hadd_ps(temp8, temp8);
          temp8 = _mm256_hadd_ps(temp8, temp8);
           tempLo2 = _mm256_castps256_ps128(temp8);
           tempHi2 = _mm256_extractf128_ps(temp8, 1);
           sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum8 = _mm_cvtss_f32(sseSum2);

          sum8 += bias4;
          out_to_compare_with_FP[out_subscript8] = Relu_float(sum8);
        }
      }
    }
  }

  printf("\n from optv6 x2 m4 - strength reduction %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// reduce int register pressure in d and x loop (also reduces num of ops done) - optimised_layer_v6_x2m4_left_shift_FP
// ~ 73 GFLOPS
int optimised_layer_v6_x2m4_left_shift_register_pressure_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  #define left_shift(x) (__builtin_ctz(x))  // inline function effectively using inline assembly to return num of trailing zeros (i.e. 128 returns 7, 2 * 128 == 2 << 7)

  float bias, bias2, bias3, bias4;
  __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
      for (unsigned int m = 0; m < Output_depth_dim; m+=4) { //channels
        bias = bias_array_FP[m];
        bias2 = bias_array_FP[m+1];
        bias3 = bias_array_FP[m+2];
        bias4 = bias_array_FP[m+3];

        for (unsigned int x = 0; x < Output_X_dim; x+=2) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          temp4 = _mm256_setzero_ps();
          temp5 = _mm256_setzero_ps();
          temp6 = _mm256_setzero_ps();
          temp7 = _mm256_setzero_ps();
          temp8 = _mm256_setzero_ps();
          // temp = 0.0f;

          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim << left_shift(Input_depth_dim))
                  + ((y * Stride_Y_dim + off_y) * Input_X_dim  << left_shift(Input_depth_dim))
                  + ((x * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                  + d;
                // unsigned long long int in_subscript2 = b * (Input_Y_dim * Input_X_dim << left_shift(Input_depth_dim))
                //   + ((y * Stride_Y_dim + off_y) * Input_X_dim << left_shift(Input_depth_dim))
                //   + (((x+1) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                //   + d;


                unsigned long long int filter_subscript = (m * Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))
                  + (off_y * Mask_X_dim << left_shift(Input_depth_dim))
                  + (off_x << left_shift(Input_depth_dim))
                  + d;
                // unsigned long long int filter_subscript2 = ((m+1) * Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))
                //   + (off_y * Mask_X_dim << left_shift(Input_depth_dim))
                //   + (off_x << left_shift(Input_depth_dim))
                //   + d;
                // unsigned long long int filter_subscript3 = ((m+2) * Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))
                //   + (off_y * Mask_X_dim << left_shift(Input_depth_dim))
                //   + (off_x << left_shift(Input_depth_dim))
                //   + d;
                // unsigned long long int filter_subscript4 = ((m+3) * Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))
                //   + (off_y * Mask_X_dim << left_shift(Input_depth_dim))
                //   + (off_x << left_shift(Input_depth_dim))
                //   + d;

                __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                __m256 s2 = _mm256_load_ps(&in_FP[in_subscript + Input_depth_dim]);

                __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]);
                __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript + (Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))]);
                __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript + (2 * (Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim)))]);
                __m256 w4 = _mm256_load_ps(&filter_FP[filter_subscript + (3 * (Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim)))]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w2));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s2, w2));
                temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s, w3));
                temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s2, w3));
                temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w4));
                temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w4));


                // float s = in_FP[in_subscript];
                // float w = filter_FP[filter_subscript];
                // temp = temp + s * w;
              }
            }
          }


          unsigned long long int out_subscript = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
            y * (Output_X_dim << left_shift(Output_depth_dim)) +
            (x << left_shift(Output_depth_dim))
            + m;
          unsigned long long int out_subscript2 = out_subscript + Output_depth_dim;
          // unsigned long long int out_subscript2 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   ((x+1) << left_shift(Output_depth_dim))
          //   + m;
          // unsigned long long int out_subscript3 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   (x << left_shift(Output_depth_dim))
          //   + (m+1);
          // unsigned long long int out_subscript4 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   ((x+1) << left_shift(Output_depth_dim))
          //   + (m+1);
          // unsigned long long int out_subscript5 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   (x << left_shift(Output_depth_dim))
          //   + (m+2);
          // unsigned long long int out_subscript6 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   ((x+1) << left_shift(Output_depth_dim))
          //   + (m+2);
          // unsigned long long int out_subscript7 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   (x << left_shift(Output_depth_dim))
          //   + (m+3);
          // unsigned long long int out_subscript8 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   ((x+1) << left_shift(Output_depth_dim))
          //   + (m+3);



          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
           tempLo = _mm256_castps256_ps128(temp3);
           tempHi = _mm256_extractf128_ps(temp3, 1);
           sseSum = _mm_add_ps(tempLo, tempHi);

          float sum3 = _mm_cvtss_f32(sseSum);

          sum3 += bias2;
          out_to_compare_with_FP[out_subscript + 1] = Relu_float(sum3);


          temp4 = _mm256_hadd_ps(temp4, temp4);
          temp4 = _mm256_hadd_ps(temp4, temp4);
           tempLo2 = _mm256_castps256_ps128(temp4);
           tempHi2 = _mm256_extractf128_ps(temp4, 1);
           sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum4 = _mm_cvtss_f32(sseSum2);

          sum4 += bias2;
          out_to_compare_with_FP[out_subscript2 + 1] = Relu_float(sum4);


          temp5 = _mm256_hadd_ps(temp5, temp5);
          temp5 = _mm256_hadd_ps(temp5, temp5);
           tempLo = _mm256_castps256_ps128(temp5);
           tempHi = _mm256_extractf128_ps(temp5, 1);
           sseSum = _mm_add_ps(tempLo, tempHi);

          float sum5 = _mm_cvtss_f32(sseSum);

          sum5 += bias3;
          out_to_compare_with_FP[out_subscript + 2] = Relu_float(sum5);


          temp6 = _mm256_hadd_ps(temp6, temp6);
          temp6 = _mm256_hadd_ps(temp6, temp6);
           tempLo2 = _mm256_castps256_ps128(temp6);
           tempHi2 = _mm256_extractf128_ps(temp6, 1);
           sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum6 = _mm_cvtss_f32(sseSum2);

          sum6 += bias3;
          out_to_compare_with_FP[out_subscript2 + 2] = Relu_float(sum6);


          temp7 = _mm256_hadd_ps(temp7, temp7);
          temp7 = _mm256_hadd_ps(temp7, temp7);
           tempLo = _mm256_castps256_ps128(temp7);
           tempHi = _mm256_extractf128_ps(temp7, 1);
           sseSum = _mm_add_ps(tempLo, tempHi);

          float sum7 = _mm_cvtss_f32(sseSum);

          sum7 += bias4;
          out_to_compare_with_FP[out_subscript + 3] = Relu_float(sum7);


          temp8 = _mm256_hadd_ps(temp8, temp8);
          temp8 = _mm256_hadd_ps(temp8, temp8);
           tempLo2 = _mm256_castps256_ps128(temp8);
           tempHi2 = _mm256_extractf128_ps(temp8, 1);
           sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum8 = _mm_cvtss_f32(sseSum2);

          sum8 += bias4;
          out_to_compare_with_FP[out_subscript2 + 3] = Relu_float(sum8);
        }
      }
    }
  }
  #undef left_shift
  printf("\n from optv6 x2 m4 - strength reduction, register pressure %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// store in/output_depth_dim in const vars in case compiler doesn't replace left_shift(x) macros with constant values - optimised_layer_v6_x2m4_left_shift_register_pressure_FP
// ~ 72 GFLOPS
int optimised_layer_v6_x2m4_left_shift_register_pressure_opt_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  #define left_shift(x) (__builtin_ctz(x))  // inline function effectively using inline assembly to return num of trailing zeros (i.e. 128 returns 7, 2 * 128 == 2 << 7)

  float bias, bias2, bias3, bias4;
  __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;
  const unsigned long long int in_depth_lshift = left_shift(Input_depth_dim);     // store as const unsigned long long int to avoid expensive conversion costs
  const unsigned long long int out_depth_lshift = left_shift(Output_depth_dim);   // store as const unsigned long long int to avoid expensive conversion costs

  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
      for (unsigned int m = 0; m < Output_depth_dim; m+=4) { //channels
        bias = bias_array_FP[m];
        bias2 = bias_array_FP[m+1];
        bias3 = bias_array_FP[m+2];
        bias4 = bias_array_FP[m+3];

        for (unsigned int x = 0; x < Output_X_dim; x+=2) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          temp4 = _mm256_setzero_ps();
          temp5 = _mm256_setzero_ps();
          temp6 = _mm256_setzero_ps();
          temp7 = _mm256_setzero_ps();
          temp8 = _mm256_setzero_ps();
          // temp = 0.0f;

          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim << in_depth_lshift)
                  + ((y * Stride_Y_dim + off_y) * Input_X_dim  << in_depth_lshift)
                  + ((x * Stride_X_dim + off_x) << in_depth_lshift)
                  + d;
                // unsigned long long int in_subscript2 = b * (Input_Y_dim * Input_X_dim << left_shift(Input_depth_dim))
                //   + ((y * Stride_Y_dim + off_y) * Input_X_dim << left_shift(Input_depth_dim))
                //   + (((x+1) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                //   + d;


                unsigned long long int filter_subscript = (m * Mask_Y_dim * Mask_X_dim << in_depth_lshift)
                  + (off_y * Mask_X_dim << in_depth_lshift)
                  + (off_x << in_depth_lshift)
                  + d;
                // unsigned long long int filter_subscript2 = ((m+1) * Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))
                //   + (off_y * Mask_X_dim << left_shift(Input_depth_dim))
                //   + (off_x << left_shift(Input_depth_dim))
                //   + d;
                // unsigned long long int filter_subscript3 = ((m+2) * Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))
                //   + (off_y * Mask_X_dim << left_shift(Input_depth_dim))
                //   + (off_x << left_shift(Input_depth_dim))
                //   + d;
                // unsigned long long int filter_subscript4 = ((m+3) * Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))
                //   + (off_y * Mask_X_dim << left_shift(Input_depth_dim))
                //   + (off_x << left_shift(Input_depth_dim))
                //   + d;

                __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                __m256 s2 = _mm256_load_ps(&in_FP[in_subscript + Input_depth_dim]);

                __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]);
                __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript + (Mask_Y_dim * Mask_X_dim << in_depth_lshift)]);
                __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript + (2 * (Mask_Y_dim * Mask_X_dim << in_depth_lshift))]);
                __m256 w4 = _mm256_load_ps(&filter_FP[filter_subscript + (3 * (Mask_Y_dim * Mask_X_dim << in_depth_lshift))]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w2));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s2, w2));
                temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s, w3));
                temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s2, w3));
                temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w4));
                temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w4));


                // float s = in_FP[in_subscript];
                // float w = filter_FP[filter_subscript];
                // temp = temp + s * w;
              }
            }
          }


          unsigned long long int out_subscript = b * (Output_X_dim * Output_Y_dim << out_depth_lshift) +
            y * (Output_X_dim << out_depth_lshift) +
            (x << out_depth_lshift)
            + m;
          unsigned long long int out_subscript2 = out_subscript + Output_depth_dim;
          // unsigned long long int out_subscript2 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   ((x+1) << left_shift(Output_depth_dim))
          //   + m;
          // unsigned long long int out_subscript3 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   (x << left_shift(Output_depth_dim))
          //   + (m+1);
          // unsigned long long int out_subscript4 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   ((x+1) << left_shift(Output_depth_dim))
          //   + (m+1);
          // unsigned long long int out_subscript5 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   (x << left_shift(Output_depth_dim))
          //   + (m+2);
          // unsigned long long int out_subscript6 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   ((x+1) << left_shift(Output_depth_dim))
          //   + (m+2);
          // unsigned long long int out_subscript7 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   (x << left_shift(Output_depth_dim))
          //   + (m+3);
          // unsigned long long int out_subscript8 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   ((x+1) << left_shift(Output_depth_dim))
          //   + (m+3);



          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
           tempLo = _mm256_castps256_ps128(temp3);
           tempHi = _mm256_extractf128_ps(temp3, 1);
           sseSum = _mm_add_ps(tempLo, tempHi);

          float sum3 = _mm_cvtss_f32(sseSum);

          sum3 += bias2;
          out_to_compare_with_FP[out_subscript + 1] = Relu_float(sum3);


          temp4 = _mm256_hadd_ps(temp4, temp4);
          temp4 = _mm256_hadd_ps(temp4, temp4);
           tempLo2 = _mm256_castps256_ps128(temp4);
           tempHi2 = _mm256_extractf128_ps(temp4, 1);
           sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum4 = _mm_cvtss_f32(sseSum2);

          sum4 += bias2;
          out_to_compare_with_FP[out_subscript2 + 1] = Relu_float(sum4);


          temp5 = _mm256_hadd_ps(temp5, temp5);
          temp5 = _mm256_hadd_ps(temp5, temp5);
           tempLo = _mm256_castps256_ps128(temp5);
           tempHi = _mm256_extractf128_ps(temp5, 1);
           sseSum = _mm_add_ps(tempLo, tempHi);

          float sum5 = _mm_cvtss_f32(sseSum);

          sum5 += bias3;
          out_to_compare_with_FP[out_subscript + 2] = Relu_float(sum5);


          temp6 = _mm256_hadd_ps(temp6, temp6);
          temp6 = _mm256_hadd_ps(temp6, temp6);
           tempLo2 = _mm256_castps256_ps128(temp6);
           tempHi2 = _mm256_extractf128_ps(temp6, 1);
           sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum6 = _mm_cvtss_f32(sseSum2);

          sum6 += bias3;
          out_to_compare_with_FP[out_subscript2 + 2] = Relu_float(sum6);


          temp7 = _mm256_hadd_ps(temp7, temp7);
          temp7 = _mm256_hadd_ps(temp7, temp7);
           tempLo = _mm256_castps256_ps128(temp7);
           tempHi = _mm256_extractf128_ps(temp7, 1);
           sseSum = _mm_add_ps(tempLo, tempHi);

          float sum7 = _mm_cvtss_f32(sseSum);

          sum7 += bias4;
          out_to_compare_with_FP[out_subscript + 3] = Relu_float(sum7);


          temp8 = _mm256_hadd_ps(temp8, temp8);
          temp8 = _mm256_hadd_ps(temp8, temp8);
           tempLo2 = _mm256_castps256_ps128(temp8);
           tempHi2 = _mm256_extractf128_ps(temp8, 1);
           sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum8 = _mm_cvtss_f32(sseSum2);

          sum8 += bias4;
          out_to_compare_with_FP[out_subscript2 + 3] = Relu_float(sum8);
        }
      }
    }
  }
  #undef left_shift
  printf("\n from optv6 x2 m4 - strength reduction, register pressure opt %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// applied left shift where possible - optimised_layer_v5_x3m3_loop_interchange_m_FP()
// ~77 GFLOPS, 
int optimised_layer_v6_x3m3_left_shift_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  #define left_shift(x) (__builtin_ctz(x))  // inline function effectively using inline assembly to return num of trailing zeros (i.e. 128 returns 7, 2 * 128 == 2 << 7)
  
  float bias, bias2, bias3;
  __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9;
  // Calculate loop bounds for unrolled loops
  int m_bound = (Output_depth_dim/ 3) * 3;
  int x_bound = (Output_X_dim/ 3) * 3; 

  unsigned int m, x;

  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
      for (m = 0; m < m_bound; m+=3) { //channels
        bias = bias_array_FP[m];
        bias2 = bias_array_FP[m+1];
        bias3 = bias_array_FP[m+2];

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
          // temp = 0.0f;

          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim << left_shift(Input_depth_dim))
                  + ((y * Stride_Y_dim + off_y) * Input_X_dim << left_shift(Input_depth_dim))
                  + ((x * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                  + d;
                unsigned long long int in_subscript2 = b * (Input_Y_dim * Input_X_dim << left_shift(Input_depth_dim))
                  + ((y * Stride_Y_dim + off_y) * Input_X_dim << left_shift(Input_depth_dim))
                  + (((x+1) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                  + d;
                unsigned long long int in_subscript3 = b * (Input_Y_dim * Input_X_dim << left_shift(Input_depth_dim))
                  + ((y * Stride_Y_dim + off_y) * Input_X_dim << left_shift(Input_depth_dim))
                  + (((x+2) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                  + d;


                unsigned long long int filter_subscript = (m * Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))
                  + (off_y * Mask_X_dim << left_shift(Input_depth_dim))
                  + (off_x << left_shift(Input_depth_dim))
                  + d;
                unsigned long long int filter_subscript2 = ((m+1) * Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))
                  + (off_y * Mask_X_dim << left_shift(Input_depth_dim))
                  + (off_x << left_shift(Input_depth_dim))
                  + d;
                unsigned long long int filter_subscript3 = ((m+2) * Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))
                  + (off_y * Mask_X_dim << left_shift(Input_depth_dim))
                  + (off_x << left_shift(Input_depth_dim))
                  + d;

                __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                __m256 s2 = _mm256_load_ps(&in_FP[in_subscript2]);
                __m256 s3 = _mm256_load_ps(&in_FP[in_subscript3]);

                __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]); 
                __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);
                __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript3]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s, w2));
                temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s2, w2));
                temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s3, w2));
                temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w3));
                temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w3));
                temp9 = _mm256_add_ps(temp9, _mm256_mul_ps(s3, w3));


                // float s = in_FP[in_subscript];
                // float w = filter_FP[filter_subscript];
                // temp = temp + s * w;
              }
            }
          }


          unsigned long long int out_subscript = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
            y * (Output_X_dim << left_shift(Output_depth_dim)) +
            (x << left_shift(Output_depth_dim))
            + m;
          unsigned long long int out_subscript2 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
            y * (Output_X_dim << left_shift(Output_depth_dim)) +
            ((x+1) << left_shift(Output_depth_dim))
            + m;
          unsigned long long int out_subscript3 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
            y * (Output_X_dim << left_shift(Output_depth_dim)) +
            ((x+2) << left_shift(Output_depth_dim))
            + m;
          unsigned long long int out_subscript4 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
            y * (Output_X_dim << left_shift(Output_depth_dim)) +
            (x << left_shift(Output_depth_dim))
            + (m+1);
          unsigned long long int out_subscript5 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
            y * (Output_X_dim << left_shift(Output_depth_dim)) +
            ((x+1) << left_shift(Output_depth_dim))
            + (m+1);
          unsigned long long int out_subscript6 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
            y * (Output_X_dim << left_shift(Output_depth_dim)) +
            ((x+2) << left_shift(Output_depth_dim))
            + (m+1);
          unsigned long long int out_subscript7 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
            y * (Output_X_dim << left_shift(Output_depth_dim)) +
            (x << left_shift(Output_depth_dim))
            + (m+2);
          unsigned long long int out_subscript8 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
            y * (Output_X_dim << left_shift(Output_depth_dim)) +
            ((x+1) << left_shift(Output_depth_dim))
            + (m+2);
          unsigned long long int out_subscript9 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
            y * (Output_X_dim << left_shift(Output_depth_dim)) +
            ((x+2) << left_shift(Output_depth_dim))
            + (m+2);



          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
          __m128 tempLo3 = _mm256_castps256_ps128(temp3);
          __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
          __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


          temp4 = _mm256_hadd_ps(temp4, temp4);
          temp4 = _mm256_hadd_ps(temp4, temp4);
          __m128 tempLo4 = _mm256_castps256_ps128(temp4);
          __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
          __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

          float sum4 = _mm_cvtss_f32(sseSum4);

          sum4 += bias2;
          out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);


          temp5 = _mm256_hadd_ps(temp5, temp5);
          temp5 = _mm256_hadd_ps(temp5, temp5);
          __m128 tempLo5 = _mm256_castps256_ps128(temp5);
          __m128 tempHi5 = _mm256_extractf128_ps(temp5, 1);
          __m128 sseSum5 = _mm_add_ps(tempLo5, tempHi5);

          float sum5 = _mm_cvtss_f32(sseSum5);

          sum5 += bias2;
          out_to_compare_with_FP[out_subscript5] = Relu_float(sum5);


          temp6 = _mm256_hadd_ps(temp6, temp6);
          temp6 = _mm256_hadd_ps(temp6, temp6);
          __m128 tempLo6 = _mm256_castps256_ps128(temp6);
          __m128 tempHi6 = _mm256_extractf128_ps(temp6, 1);
          __m128 sseSum6 = _mm_add_ps(tempLo6, tempHi6);

          float sum6 = _mm_cvtss_f32(sseSum6);

          sum6 += bias2;
          out_to_compare_with_FP[out_subscript6] = Relu_float(sum6);


          temp7 = _mm256_hadd_ps(temp7, temp7);
          temp7 = _mm256_hadd_ps(temp7, temp7);
          __m128 tempLo7 = _mm256_castps256_ps128(temp7);
          __m128 tempHi7 = _mm256_extractf128_ps(temp7, 1);
          __m128 sseSum7 = _mm_add_ps(tempLo7, tempHi7);

          float sum7 = _mm_cvtss_f32(sseSum7);

          sum7 += bias3;
          out_to_compare_with_FP[out_subscript7] = Relu_float(sum7);


          temp8 = _mm256_hadd_ps(temp8, temp8);
          temp8 = _mm256_hadd_ps(temp8, temp8);
          __m128 tempLo8 = _mm256_castps256_ps128(temp8);
          __m128 tempHi8 = _mm256_extractf128_ps(temp8, 1);
          __m128 sseSum8 = _mm_add_ps(tempLo8, tempHi8);

          float sum8 = _mm_cvtss_f32(sseSum8);

          sum8 += bias3;
          out_to_compare_with_FP[out_subscript8] = Relu_float(sum8);


          temp9 = _mm256_hadd_ps(temp9, temp9);
          temp9 = _mm256_hadd_ps(temp9, temp9);
          __m128 tempLo9 = _mm256_castps256_ps128(temp9);
          __m128 tempHi9 = _mm256_extractf128_ps(temp9, 1);
          __m128 sseSum9 = _mm_add_ps(tempLo9, tempHi9);

          float sum9 = _mm_cvtss_f32(sseSum9);

          sum9 += bias3;
          out_to_compare_with_FP[out_subscript9] = Relu_float(sum9);
        }
        // overflow/ fallback x loop
        for (; x < Output_X_dim; x++) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim << left_shift(Input_depth_dim))
                  + ((y * Stride_Y_dim + off_y) * Input_X_dim << left_shift(Input_depth_dim))
                  + ((x * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                  + d;

                unsigned long long int filter_subscript = (m * Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))
                  + (off_y * Mask_X_dim << left_shift(Input_depth_dim))
                  + (off_x << left_shift(Input_depth_dim))
                  + d;
                unsigned long long int filter_subscript2 = ((m+1) * Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))
                  + (off_y * Mask_X_dim << left_shift(Input_depth_dim))
                  + (off_x << left_shift(Input_depth_dim))
                  + d;
                unsigned long long int filter_subscript3 = ((m+2) * Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))
                  + (off_y * Mask_X_dim << left_shift(Input_depth_dim))
                  + (off_x << left_shift(Input_depth_dim))
                  + d;

                __m256 s = _mm256_loadu_ps(&in_FP[in_subscript]);

                __m256 w = _mm256_loadu_ps(&filter_FP[filter_subscript]);
                __m256 w2 = _mm256_loadu_ps(&filter_FP[filter_subscript2]);
                __m256 w3 = _mm256_loadu_ps(&filter_FP[filter_subscript3]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s, w2));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w3));

                // float s = in_FP[in_subscript];
                // float w = filter_FP[filter_subscript];
                // temp = temp + s * w;
              }
            }
          }
          unsigned long long int out_subscript = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
            y * (Output_X_dim << left_shift(Output_depth_dim)) +
            (x << left_shift(Output_depth_dim))
            + m;
          unsigned long long int out_subscript2 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
            y * (Output_X_dim << left_shift(Output_depth_dim)) +
            (x << left_shift(Output_depth_dim))
            + (m+1);
          unsigned long long int out_subscript3 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
            y * (Output_X_dim << left_shift(Output_depth_dim)) +
            (x << left_shift(Output_depth_dim))
            + (m+2);

          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias2;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
          __m128 tempLo3 = _mm256_castps256_ps128(temp3);
          __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
          __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias3;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);

        }
      }
      for (; m < Output_depth_dim; m++) {
        bias = bias_array_FP[m];

        for (unsigned int x = 0; x < Output_X_dim; x+=4) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          temp4 = _mm256_setzero_ps();

          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim << left_shift(Input_depth_dim))
                  + ((y * Stride_Y_dim + off_y) * Input_X_dim << left_shift(Input_depth_dim))
                  + ((x * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                  + d;
                unsigned long long int in_subscript2 = b * (Input_Y_dim * Input_X_dim << left_shift(Input_depth_dim))
                  + ((y * Stride_Y_dim + off_y) * Input_X_dim << left_shift(Input_depth_dim))
                  + (((x+1) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                  + d;
                unsigned long long int in_subscript3 = b * (Input_Y_dim * Input_X_dim << left_shift(Input_depth_dim))
                  + ((y * Stride_Y_dim + off_y) * Input_X_dim << left_shift(Input_depth_dim))
                  + (((x+2) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                  + d;
                unsigned long long int in_subscript4 = b * (Input_Y_dim * Input_X_dim << left_shift(Input_depth_dim))
                  + ((y * Stride_Y_dim + off_y) * Input_X_dim << left_shift(Input_depth_dim))
                  + (((x+3) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                  + d;

                unsigned long long int filter_subscript = (m * Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))
                  + (off_y * Mask_X_dim << left_shift(Input_depth_dim))
                  + (off_x << left_shift(Input_depth_dim))
                  + d;

                __m256 s = _mm256_loadu_ps(&in_FP[in_subscript]);
                __m256 s2 = _mm256_loadu_ps(&in_FP[in_subscript2]);
                __m256 s3 = _mm256_loadu_ps(&in_FP[in_subscript3]);
                __m256 s4 = _mm256_loadu_ps(&in_FP[in_subscript4]);

                __m256 w = _mm256_loadu_ps(&filter_FP[filter_subscript]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s4, w));
              }
            }
          }

          unsigned long long int out_subscript = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
            y * (Output_X_dim << left_shift(Output_depth_dim)) +
            (x << left_shift(Output_depth_dim))
            + m;
          unsigned long long int out_subscript2 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
            y * (Output_X_dim << left_shift(Output_depth_dim)) +
            ((x+1) << left_shift(Output_depth_dim))
            + m;
          unsigned long long int out_subscript3 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
            y * (Output_X_dim << left_shift(Output_depth_dim)) +
            ((x+2) << left_shift(Output_depth_dim))
            + m;
          unsigned long long int out_subscript4 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
            y * (Output_X_dim << left_shift(Output_depth_dim)) +
            ((x+3) << left_shift(Output_depth_dim))
            + m;

          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
          __m128 tempLo3 = _mm256_castps256_ps128(temp3);
          __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
          __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


          temp4 = _mm256_hadd_ps(temp4, temp4);
          temp4 = _mm256_hadd_ps(temp4, temp4);
          __m128 tempLo4 = _mm256_castps256_ps128(temp4);
          __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
          __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

          float sum4 = _mm_cvtss_f32(sseSum4);

          sum4 += bias;
          out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);

        }
      }
    }
  }
  #undef left_shift
  printf("\n from optv6 x3 m3 hadd opt - left shift %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// reduce int register pressure in d and x loop (also reduces num of ops done) - optimised_layer_v5_x3m3_loop_interchange_m_FP()
// ~74 GFLOPS, maybe increased code size is causing it? 
int optimised_layer_v6_x3m3_left_shift_register_pressure_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  #define left_shift(x) (__builtin_ctz(x))  // inline function effectively using inline assembly to return num of trailing zeros (i.e. 128 returns 7, 2 * 128 == 2 << 7)
  
  float bias, bias2, bias3;
  __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9;
  // Calculate loop bounds for unrolled loops
  int m_bound = (Output_depth_dim/ 3) * 3;
  int x_bound = (Output_X_dim/ 3) * 3; 

  unsigned int m, x;

  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
      for (m = 0; m < m_bound; m+=3) { //channels
        bias = bias_array_FP[m];
        bias2 = bias_array_FP[m+1];
        bias3 = bias_array_FP[m+2];

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
          // temp = 0.0f;

          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim << left_shift(Input_depth_dim))
                  + ((y * Stride_Y_dim + off_y) * Input_X_dim << left_shift(Input_depth_dim))
                  + ((x * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                  + d;
                // unsigned long long int in_subscript2 = b * (Input_Y_dim * Input_X_dim << left_shift(Input_depth_dim))
                //   + ((y * Stride_Y_dim + off_y) * Input_X_dim << left_shift(Input_depth_dim))
                //   + (((x+1) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                //   + d;
                // unsigned long long int in_subscript3 = b * (Input_Y_dim * Input_X_dim << left_shift(Input_depth_dim))
                //   + ((y * Stride_Y_dim + off_y) * Input_X_dim << left_shift(Input_depth_dim))
                //   + (((x+2) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                //   + d;


                unsigned long long int filter_subscript = (m * Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))
                  + (off_y * Mask_X_dim << left_shift(Input_depth_dim))
                  + (off_x << left_shift(Input_depth_dim))
                  + d;
                // unsigned long long int filter_subscript2 = ((m+1) * Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))
                //   + (off_y * Mask_X_dim << left_shift(Input_depth_dim))
                //   + (off_x << left_shift(Input_depth_dim))
                //   + d;
                // unsigned long long int filter_subscript3 = ((m+2) * Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))
                //   + (off_y * Mask_X_dim << left_shift(Input_depth_dim))
                //   + (off_x << left_shift(Input_depth_dim))
                //   + d;

                __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                __m256 s2 = _mm256_load_ps(&in_FP[in_subscript + Input_depth_dim]);
                __m256 s3 = _mm256_load_ps(&in_FP[in_subscript + (2 << left_shift(Input_depth_dim))]);

                __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]); 
                __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript + (Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))]);
                __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript + (2 * (Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim)))]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s, w2));
                temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s2, w2));
                temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s3, w2));
                temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w3));
                temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w3));
                temp9 = _mm256_add_ps(temp9, _mm256_mul_ps(s3, w3));


                // float s = in_FP[in_subscript];
                // float w = filter_FP[filter_subscript];
                // temp = temp + s * w;
              }
            }
          }


          unsigned long long int out_subscript = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
            y * (Output_X_dim << left_shift(Output_depth_dim)) +
            (x << left_shift(Output_depth_dim))
            + m;
          unsigned long long int out_subscript2 = out_subscript + Output_depth_dim;
          unsigned long long int out_subscript3 = out_subscript2 + Output_depth_dim;
          
          // unsigned long long int out_subscript2 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   ((x+1) << left_shift(Output_depth_dim))
          //   + m;
          // unsigned long long int out_subscript3 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   ((x+2) << left_shift(Output_depth_dim))
          //   + m;
          // unsigned long long int out_subscript4 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   (x << left_shift(Output_depth_dim))
          //   + (m+1);
          // unsigned long long int out_subscript5 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   ((x+1) << left_shift(Output_depth_dim))
          //   + (m+1);
          // unsigned long long int out_subscript6 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   ((x+2) << left_shift(Output_depth_dim))
          //   + (m+1);
          // unsigned long long int out_subscript7 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   (x << left_shift(Output_depth_dim))
          //   + (m+2);
          // unsigned long long int out_subscript8 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   ((x+1) << left_shift(Output_depth_dim))
          //   + (m+2);
          // unsigned long long int out_subscript9 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   ((x+2) << left_shift(Output_depth_dim))
          //   + (m+2);



          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
          __m128 tempLo3 = _mm256_castps256_ps128(temp3);
          __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
          __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


          temp4 = _mm256_hadd_ps(temp4, temp4);
          temp4 = _mm256_hadd_ps(temp4, temp4);
          __m128 tempLo4 = _mm256_castps256_ps128(temp4);
          __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
          __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

          float sum4 = _mm_cvtss_f32(sseSum4);

          sum4 += bias2;
          out_to_compare_with_FP[out_subscript + 1] = Relu_float(sum4);


          temp5 = _mm256_hadd_ps(temp5, temp5);
          temp5 = _mm256_hadd_ps(temp5, temp5);
          __m128 tempLo5 = _mm256_castps256_ps128(temp5);
          __m128 tempHi5 = _mm256_extractf128_ps(temp5, 1);
          __m128 sseSum5 = _mm_add_ps(tempLo5, tempHi5);

          float sum5 = _mm_cvtss_f32(sseSum5);

          sum5 += bias2;
          out_to_compare_with_FP[out_subscript2 + 1] = Relu_float(sum5);


          temp6 = _mm256_hadd_ps(temp6, temp6);
          temp6 = _mm256_hadd_ps(temp6, temp6);
          __m128 tempLo6 = _mm256_castps256_ps128(temp6);
          __m128 tempHi6 = _mm256_extractf128_ps(temp6, 1);
          __m128 sseSum6 = _mm_add_ps(tempLo6, tempHi6);

          float sum6 = _mm_cvtss_f32(sseSum6);

          sum6 += bias2;
          out_to_compare_with_FP[out_subscript3 + 1] = Relu_float(sum6);


          temp7 = _mm256_hadd_ps(temp7, temp7);
          temp7 = _mm256_hadd_ps(temp7, temp7);
          __m128 tempLo7 = _mm256_castps256_ps128(temp7);
          __m128 tempHi7 = _mm256_extractf128_ps(temp7, 1);
          __m128 sseSum7 = _mm_add_ps(tempLo7, tempHi7);

          float sum7 = _mm_cvtss_f32(sseSum7);

          sum7 += bias3;
          out_to_compare_with_FP[out_subscript + 2] = Relu_float(sum7);


          temp8 = _mm256_hadd_ps(temp8, temp8);
          temp8 = _mm256_hadd_ps(temp8, temp8);
          __m128 tempLo8 = _mm256_castps256_ps128(temp8);
          __m128 tempHi8 = _mm256_extractf128_ps(temp8, 1);
          __m128 sseSum8 = _mm_add_ps(tempLo8, tempHi8);

          float sum8 = _mm_cvtss_f32(sseSum8);

          sum8 += bias3;
          out_to_compare_with_FP[out_subscript2 + 2] = Relu_float(sum8);


          temp9 = _mm256_hadd_ps(temp9, temp9);
          temp9 = _mm256_hadd_ps(temp9, temp9);
          __m128 tempLo9 = _mm256_castps256_ps128(temp9);
          __m128 tempHi9 = _mm256_extractf128_ps(temp9, 1);
          __m128 sseSum9 = _mm_add_ps(tempLo9, tempHi9);

          float sum9 = _mm_cvtss_f32(sseSum9);

          sum9 += bias3;
          out_to_compare_with_FP[out_subscript3 + 2] = Relu_float(sum9);
        }
        // overflow/ fallback x loop
        for (; x < Output_X_dim; x++) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim << left_shift(Input_depth_dim))
                  + ((y * Stride_Y_dim + off_y) * Input_X_dim << left_shift(Input_depth_dim))
                  + ((x * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                  + d;

                unsigned long long int filter_subscript = (m * Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))
                  + (off_y * Mask_X_dim << left_shift(Input_depth_dim))
                  + (off_x << left_shift(Input_depth_dim))
                  + d;
                // unsigned long long int filter_subscript2 = ((m+1) * Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))
                //   + (off_y * Mask_X_dim << left_shift(Input_depth_dim))
                //   + (off_x << left_shift(Input_depth_dim))
                //   + d;
                // unsigned long long int filter_subscript3 = ((m+2) * Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))
                //   + (off_y * Mask_X_dim << left_shift(Input_depth_dim))
                //   + (off_x << left_shift(Input_depth_dim))
                //   + d;

                __m256 s = _mm256_loadu_ps(&in_FP[in_subscript]);

                __m256 w = _mm256_loadu_ps(&filter_FP[filter_subscript]);
                __m256 w2 = _mm256_loadu_ps(&filter_FP[filter_subscript + (Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))]);
                __m256 w3 = _mm256_loadu_ps(&filter_FP[filter_subscript + (2 * (Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim)))]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s, w2));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w3));

                // float s = in_FP[in_subscript];
                // float w = filter_FP[filter_subscript];
                // temp = temp + s * w;
              }
            }
          }
          unsigned long long int out_subscript = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
            y * (Output_X_dim << left_shift(Output_depth_dim)) +
            (x << left_shift(Output_depth_dim))
            + m;
          // unsigned long long int out_subscript2 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   (x << left_shift(Output_depth_dim))
          //   + (m+1);
          // unsigned long long int out_subscript3 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   (x << left_shift(Output_depth_dim))
          //   + (m+2);

          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias2;
          out_to_compare_with_FP[out_subscript + 1] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
          __m128 tempLo3 = _mm256_castps256_ps128(temp3);
          __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
          __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias3;
          out_to_compare_with_FP[out_subscript + 2] = Relu_float(sum3);

        }
      }
      for (; m < Output_depth_dim; m++) {
        bias = bias_array_FP[m];

        for (unsigned int x = 0; x < Output_X_dim; x+=4) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          temp4 = _mm256_setzero_ps();

          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim << left_shift(Input_depth_dim))
                  + ((y * Stride_Y_dim + off_y) * Input_X_dim << left_shift(Input_depth_dim))
                  + ((x * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                  + d;
                // unsigned long long int in_subscript2 = b * (Input_Y_dim * Input_X_dim << left_shift(Input_depth_dim))
                //   + ((y * Stride_Y_dim + off_y) * Input_X_dim << left_shift(Input_depth_dim))
                //   + (((x+1) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                //   + d;
                // unsigned long long int in_subscript3 = b * (Input_Y_dim * Input_X_dim << left_shift(Input_depth_dim))
                //   + ((y * Stride_Y_dim + off_y) * Input_X_dim << left_shift(Input_depth_dim))
                //   + (((x+2) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                //   + d;
                // unsigned long long int in_subscript4 = b * (Input_Y_dim * Input_X_dim << left_shift(Input_depth_dim))
                //   + ((y * Stride_Y_dim + off_y) * Input_X_dim << left_shift(Input_depth_dim))
                //   + (((x+3) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                //   + d;

                unsigned long long int filter_subscript = (m * Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))
                  + (off_y * Mask_X_dim << left_shift(Input_depth_dim))
                  + (off_x << left_shift(Input_depth_dim))
                  + d;

                __m256 s = _mm256_loadu_ps(&in_FP[in_subscript]);
                __m256 s2 = _mm256_loadu_ps(&in_FP[in_subscript + Input_depth_dim]);
                __m256 s3 = _mm256_loadu_ps(&in_FP[in_subscript + (2 << left_shift(Input_depth_dim))]);
                __m256 s4 = _mm256_loadu_ps(&in_FP[in_subscript + (3 << left_shift(Input_depth_dim))]);

                __m256 w = _mm256_loadu_ps(&filter_FP[filter_subscript]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s4, w));
              }
            }
          }

          unsigned long long int out_subscript = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
            y * (Output_X_dim << left_shift(Output_depth_dim)) +
            (x << left_shift(Output_depth_dim))
            + m;
          unsigned long long int out_subscript2 = out_subscript + Output_depth_dim;
          unsigned long long int out_subscript3 = out_subscript2 + Output_depth_dim;
          unsigned long long int out_subscript4 = out_subscript3 + Output_depth_dim;
        
          // unsigned long long int out_subscript2 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   ((x+1) << left_shift(Output_depth_dim))
          //   + m;
          // unsigned long long int out_subscript3 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   ((x+2) << left_shift(Output_depth_dim))
          //   + m;
          // unsigned long long int out_subscript4 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   ((x+3) << left_shift(Output_depth_dim))
          //   + m;

          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
          __m128 tempLo3 = _mm256_castps256_ps128(temp3);
          __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
          __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


          temp4 = _mm256_hadd_ps(temp4, temp4);
          temp4 = _mm256_hadd_ps(temp4, temp4);
          __m128 tempLo4 = _mm256_castps256_ps128(temp4);
          __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
          __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

          float sum4 = _mm_cvtss_f32(sseSum4);

          sum4 += bias;
          out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);

        }
      }
    }
  }
  #undef left_shift
  printf("\n from optv6 x3 m3 hadd opt - left shift, register pressure %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// reduce int register pressure in d and x loop (also reduces num of ops done) - optimised_layer_v5_x3m3_loop_interchange_m_FP()
// ~76 GFLOPS, investigate why slightly lower perf, maybe bc I've created dependencies? (sub2 relies on sub1, sub3 relies on sub2, etc.)
int optimised_layer_v6_x3m3_left_shift_register_pressure_opt_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  #define left_shift(x) (__builtin_ctz(x))  // inline function effectively using inline assembly to return num of trailing zeros (i.e. 128 returns 7, 2 * 128 == 2 << 7)
  
  float bias, bias2, bias3;
  __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9;
  // Calculate loop bounds for unrolled loops
  int m_bound = (Output_depth_dim/ 3) * 3;
  int x_bound = (Output_X_dim/ 3) * 3; 

  unsigned int m, x;

  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
      for (m = 0; m < m_bound; m+=3) { //channels
        bias = bias_array_FP[m];
        bias2 = bias_array_FP[m+1];
        bias3 = bias_array_FP[m+2];

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
          // temp = 0.0f;

          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim << left_shift(Input_depth_dim))
                  + ((y * Stride_Y_dim + off_y) * Input_X_dim << left_shift(Input_depth_dim))
                  + ((x * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                  + d;
                // unsigned long long int in_subscript2 = b * (Input_Y_dim * Input_X_dim << left_shift(Input_depth_dim))
                //   + ((y * Stride_Y_dim + off_y) * Input_X_dim << left_shift(Input_depth_dim))
                //   + (((x+1) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                //   + d;
                // unsigned long long int in_subscript3 = b * (Input_Y_dim * Input_X_dim << left_shift(Input_depth_dim))
                //   + ((y * Stride_Y_dim + off_y) * Input_X_dim << left_shift(Input_depth_dim))
                //   + (((x+2) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                //   + d;


                unsigned long long int filter_subscript = (m * Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))
                  + (off_y * Mask_X_dim << left_shift(Input_depth_dim))
                  + (off_x << left_shift(Input_depth_dim))
                  + d;
                // unsigned long long int filter_subscript2 = ((m+1) * Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))
                //   + (off_y * Mask_X_dim << left_shift(Input_depth_dim))
                //   + (off_x << left_shift(Input_depth_dim))
                //   + d;
                // unsigned long long int filter_subscript3 = ((m+2) * Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))
                //   + (off_y * Mask_X_dim << left_shift(Input_depth_dim))
                //   + (off_x << left_shift(Input_depth_dim))
                //   + d;

                __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                __m256 s2 = _mm256_load_ps(&in_FP[in_subscript + Input_depth_dim]);
                __m256 s3 = _mm256_load_ps(&in_FP[in_subscript + (2 << left_shift(Input_depth_dim))]);

                __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]); 
                __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript + (Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))]);
                __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript + (2 * (Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim)))]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s, w2));
                temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s2, w2));
                temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s3, w2));
                temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w3));
                temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w3));
                temp9 = _mm256_add_ps(temp9, _mm256_mul_ps(s3, w3));


                // float s = in_FP[in_subscript];
                // float w = filter_FP[filter_subscript];
                // temp = temp + s * w;
              }
            }
          }


          unsigned long long int out_subscript = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
            y * (Output_X_dim << left_shift(Output_depth_dim)) +
            (x << left_shift(Output_depth_dim))
            + m;
          unsigned long long int out_subscript2 = out_subscript + Output_depth_dim;
          unsigned long long int out_subscript3 = out_subscript2 + Output_depth_dim;
          unsigned long long int out_subscript4 = out_subscript + 1;
          unsigned long long int out_subscript5 = out_subscript2 + 1;
          unsigned long long int out_subscript6 = out_subscript3 + 1;
          unsigned long long int out_subscript7 = out_subscript4 + 1;
          unsigned long long int out_subscript8 = out_subscript5 + 1;
          unsigned long long int out_subscript9 = out_subscript6 + 1;
          
          // unsigned long long int out_subscript2 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   ((x+1) << left_shift(Output_depth_dim))
          //   + m;
          // unsigned long long int out_subscript3 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   ((x+2) << left_shift(Output_depth_dim))
          //   + m;
          // unsigned long long int out_subscript4 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   (x << left_shift(Output_depth_dim))
          //   + (m+1);
          // unsigned long long int out_subscript5 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   ((x+1) << left_shift(Output_depth_dim))
          //   + (m+1);
          // unsigned long long int out_subscript6 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   ((x+2) << left_shift(Output_depth_dim))
          //   + (m+1);
          // unsigned long long int out_subscript7 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   (x << left_shift(Output_depth_dim))
          //   + (m+2);
          // unsigned long long int out_subscript8 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   ((x+1) << left_shift(Output_depth_dim))
          //   + (m+2);
          // unsigned long long int out_subscript9 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   ((x+2) << left_shift(Output_depth_dim))
          //   + (m+2);



          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
          __m128 tempLo3 = _mm256_castps256_ps128(temp3);
          __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
          __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


          temp4 = _mm256_hadd_ps(temp4, temp4);
          temp4 = _mm256_hadd_ps(temp4, temp4);
          __m128 tempLo4 = _mm256_castps256_ps128(temp4);
          __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
          __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

          float sum4 = _mm_cvtss_f32(sseSum4);

          sum4 += bias2;
          out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);


          temp5 = _mm256_hadd_ps(temp5, temp5);
          temp5 = _mm256_hadd_ps(temp5, temp5);
          __m128 tempLo5 = _mm256_castps256_ps128(temp5);
          __m128 tempHi5 = _mm256_extractf128_ps(temp5, 1);
          __m128 sseSum5 = _mm_add_ps(tempLo5, tempHi5);

          float sum5 = _mm_cvtss_f32(sseSum5);

          sum5 += bias2;
          out_to_compare_with_FP[out_subscript5] = Relu_float(sum5);


          temp6 = _mm256_hadd_ps(temp6, temp6);
          temp6 = _mm256_hadd_ps(temp6, temp6);
          __m128 tempLo6 = _mm256_castps256_ps128(temp6);
          __m128 tempHi6 = _mm256_extractf128_ps(temp6, 1);
          __m128 sseSum6 = _mm_add_ps(tempLo6, tempHi6);

          float sum6 = _mm_cvtss_f32(sseSum6);

          sum6 += bias2;
          out_to_compare_with_FP[out_subscript6] = Relu_float(sum6);


          temp7 = _mm256_hadd_ps(temp7, temp7);
          temp7 = _mm256_hadd_ps(temp7, temp7);
          __m128 tempLo7 = _mm256_castps256_ps128(temp7);
          __m128 tempHi7 = _mm256_extractf128_ps(temp7, 1);
          __m128 sseSum7 = _mm_add_ps(tempLo7, tempHi7);

          float sum7 = _mm_cvtss_f32(sseSum7);

          sum7 += bias3;
          out_to_compare_with_FP[out_subscript7] = Relu_float(sum7);


          temp8 = _mm256_hadd_ps(temp8, temp8);
          temp8 = _mm256_hadd_ps(temp8, temp8);
          __m128 tempLo8 = _mm256_castps256_ps128(temp8);
          __m128 tempHi8 = _mm256_extractf128_ps(temp8, 1);
          __m128 sseSum8 = _mm_add_ps(tempLo8, tempHi8);

          float sum8 = _mm_cvtss_f32(sseSum8);

          sum8 += bias3;
          out_to_compare_with_FP[out_subscript8] = Relu_float(sum8);


          temp9 = _mm256_hadd_ps(temp9, temp9);
          temp9 = _mm256_hadd_ps(temp9, temp9);
          __m128 tempLo9 = _mm256_castps256_ps128(temp9);
          __m128 tempHi9 = _mm256_extractf128_ps(temp9, 1);
          __m128 sseSum9 = _mm_add_ps(tempLo9, tempHi9);

          float sum9 = _mm_cvtss_f32(sseSum9);

          sum9 += bias3;
          out_to_compare_with_FP[out_subscript9] = Relu_float(sum9);
        }
        // overflow/ fallback x loop
        for (; x < Output_X_dim; x++) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim << left_shift(Input_depth_dim))
                  + ((y * Stride_Y_dim + off_y) * Input_X_dim << left_shift(Input_depth_dim))
                  + ((x * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                  + d;

                unsigned long long int filter_subscript = (m * Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))
                  + (off_y * Mask_X_dim << left_shift(Input_depth_dim))
                  + (off_x << left_shift(Input_depth_dim))
                  + d;
                // unsigned long long int filter_subscript2 = ((m+1) * Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))
                //   + (off_y * Mask_X_dim << left_shift(Input_depth_dim))
                //   + (off_x << left_shift(Input_depth_dim))
                //   + d;
                // unsigned long long int filter_subscript3 = ((m+2) * Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))
                //   + (off_y * Mask_X_dim << left_shift(Input_depth_dim))
                //   + (off_x << left_shift(Input_depth_dim))
                //   + d;

                __m256 s = _mm256_loadu_ps(&in_FP[in_subscript]);

                __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]); 
                __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript + (Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))]);
                __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript + (2 * (Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim)))]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s, w2));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w3));

                // float s = in_FP[in_subscript];
                // float w = filter_FP[filter_subscript];
                // temp = temp + s * w;
              }
            }
          }
          unsigned long long int out_subscript = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
            y * (Output_X_dim << left_shift(Output_depth_dim)) +
            (x << left_shift(Output_depth_dim))
            + m;
          unsigned long long int out_subscript2 = out_subscript + 1;
          unsigned long long int out_subscript3 = out_subscript2 + 1;
          // unsigned long long int out_subscript2 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   (x << left_shift(Output_depth_dim))
          //   + (m+1);
          // unsigned long long int out_subscript3 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   (x << left_shift(Output_depth_dim))
          //   + (m+2);

          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias2;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
          __m128 tempLo3 = _mm256_castps256_ps128(temp3);
          __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
          __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias3;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);

        }
      }
      for (; m < Output_depth_dim; m++) {
        bias = bias_array_FP[m];

        for (unsigned int x = 0; x < Output_X_dim; x+=4) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          temp4 = _mm256_setzero_ps();

          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim << left_shift(Input_depth_dim))
                  + ((y * Stride_Y_dim + off_y) * Input_X_dim << left_shift(Input_depth_dim))
                  + ((x * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                  + d;
                // unsigned long long int in_subscript2 = b * (Input_Y_dim * Input_X_dim << left_shift(Input_depth_dim))
                //   + ((y * Stride_Y_dim + off_y) * Input_X_dim << left_shift(Input_depth_dim))
                //   + (((x+1) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                //   + d;
                // unsigned long long int in_subscript3 = b * (Input_Y_dim * Input_X_dim << left_shift(Input_depth_dim))
                //   + ((y * Stride_Y_dim + off_y) * Input_X_dim << left_shift(Input_depth_dim))
                //   + (((x+2) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                //   + d;
                // unsigned long long int in_subscript4 = b * (Input_Y_dim * Input_X_dim << left_shift(Input_depth_dim))
                //   + ((y * Stride_Y_dim + off_y) * Input_X_dim << left_shift(Input_depth_dim))
                //   + (((x+3) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                //   + d;

                unsigned long long int filter_subscript = (m * Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))
                  + (off_y * Mask_X_dim << left_shift(Input_depth_dim))
                  + (off_x << left_shift(Input_depth_dim))
                  + d;

                __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                __m256 s2 = _mm256_load_ps(&in_FP[in_subscript + Input_depth_dim]);
                __m256 s3 = _mm256_load_ps(&in_FP[in_subscript + (2 << left_shift(Input_depth_dim))]);
                __m256 s4 = _mm256_load_ps(&in_FP[in_subscript + (3 << left_shift(Input_depth_dim))]);


                __m256 w = _mm256_loadu_ps(&filter_FP[filter_subscript]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s4, w));
              }
            }
          }

          unsigned long long int out_subscript = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
            y * (Output_X_dim << left_shift(Output_depth_dim)) +
            (x << left_shift(Output_depth_dim))
            + m;
          unsigned long long int out_subscript2 = out_subscript + Output_depth_dim;
          unsigned long long int out_subscript3 = out_subscript2 + Output_depth_dim;
          unsigned long long int out_subscript4 = out_subscript3 + Output_depth_dim;
          // unsigned long long int out_subscript2 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   ((x+1) << left_shift(Output_depth_dim))
          //   + m;
          // unsigned long long int out_subscript3 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   ((x+2) << left_shift(Output_depth_dim))
          //   + m;
          // unsigned long long int out_subscript4 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   ((x+3) << left_shift(Output_depth_dim))
          //   + m;

          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
          __m128 tempLo3 = _mm256_castps256_ps128(temp3);
          __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
          __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


          temp4 = _mm256_hadd_ps(temp4, temp4);
          temp4 = _mm256_hadd_ps(temp4, temp4);
          __m128 tempLo4 = _mm256_castps256_ps128(temp4);
          __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
          __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

          float sum4 = _mm_cvtss_f32(sseSum4);

          sum4 += bias;
          out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);

        }
      }
    }
  }
  #undef left_shift
  printf("\n from optv6 x3 m3 hadd opt - left shift, register pressure opt %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}
 





// v7 moving code outside of loops - using v6 funcs as base


// moving operations outside of loop body - optimised_layer_v6_x2m4_left_shift_register_pressure_FP()
// ~84 GFLOPS
int optimised_layer_v7_x2m4_ops_outside_loop_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  #define left_shift(x) (__builtin_ctz(x))  // inline function effectively using inline assembly to return num of trailing zeros (i.e. 128 returns 7, 2 * 128 == 2 << 7)

  float bias, bias2, bias3, bias4;
  __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

  unsigned long long int in_yx_depth = Input_Y_dim * Input_X_dim << left_shift(Input_depth_dim);
  unsigned long long int in_x_depth = Input_X_dim  << left_shift(Input_depth_dim);

  unsigned long long int mask_yx_depth = Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim);
  unsigned long long int mask_x_depth = Mask_X_dim << left_shift(Input_depth_dim);

  unsigned long long int out_yx_depth = Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim);
  unsigned long long int out_x_depth = Output_X_dim << left_shift(Output_depth_dim);

  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
      for (unsigned int m = 0; m < Output_depth_dim; m+=4) { //channels
        bias = bias_array_FP[m];
        bias2 = bias_array_FP[m+1];
        bias3 = bias_array_FP[m+2];
        bias4 = bias_array_FP[m+3];

        for (unsigned int x = 0; x < Output_X_dim; x+=2) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          temp4 = _mm256_setzero_ps();
          temp5 = _mm256_setzero_ps();
          temp6 = _mm256_setzero_ps();
          temp7 = _mm256_setzero_ps();
          temp8 = _mm256_setzero_ps();
          // temp = 0.0f;

          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                unsigned long long int in_subscript = b * (in_yx_depth)
                  + ((y * Stride_Y_dim + off_y) * in_x_depth)
                  + ((x * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                  + d;
                // unsigned long long int in_subscript2 = b * (Input_Y_dim * Input_X_dim << left_shift(Input_depth_dim))
                //   + ((y * Stride_Y_dim + off_y) * Input_X_dim << left_shift(Input_depth_dim))
                //   + (((x+1) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                //   + d;


                unsigned long long int filter_subscript = (m * mask_yx_depth)
                  + (off_y * mask_x_depth)
                  + (off_x << left_shift(Input_depth_dim))
                  + d;
                // unsigned long long int filter_subscript2 = ((m+1) * Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))
                //   + (off_y * Mask_X_dim << left_shift(Input_depth_dim))
                //   + (off_x << left_shift(Input_depth_dim))
                //   + d;
                // unsigned long long int filter_subscript3 = ((m+2) * Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))
                //   + (off_y * Mask_X_dim << left_shift(Input_depth_dim))
                //   + (off_x << left_shift(Input_depth_dim))
                //   + d;
                // unsigned long long int filter_subscript4 = ((m+3) * Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))
                //   + (off_y * Mask_X_dim << left_shift(Input_depth_dim))
                //   + (off_x << left_shift(Input_depth_dim))
                //   + d;

                __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                __m256 s2 = _mm256_load_ps(&in_FP[in_subscript + Input_depth_dim]);

                __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]);
                __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript + (Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))]);
                __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript + (2 * (Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim)))]);
                __m256 w4 = _mm256_load_ps(&filter_FP[filter_subscript + (3 * (Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim)))]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w2));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s2, w2));
                temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s, w3));
                temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s2, w3));
                temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w4));
                temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w4));


                // float s = in_FP[in_subscript];
                // float w = filter_FP[filter_subscript];
                // temp = temp + s * w;
              }
            }
          }


          unsigned long long int out_subscript = b * (out_yx_depth) 
            + y * (out_x_depth) 
            + (x << left_shift(Output_depth_dim))
            + m;
          unsigned long long int out_subscript2 = out_subscript + Output_depth_dim;
          // unsigned long long int out_subscript2 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   ((x+1) << left_shift(Output_depth_dim))
          //   + m;
          // unsigned long long int out_subscript3 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   (x << left_shift(Output_depth_dim))
          //   + (m+1);
          // unsigned long long int out_subscript4 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   ((x+1) << left_shift(Output_depth_dim))
          //   + (m+1);
          // unsigned long long int out_subscript5 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   (x << left_shift(Output_depth_dim))
          //   + (m+2);
          // unsigned long long int out_subscript6 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   ((x+1) << left_shift(Output_depth_dim))
          //   + (m+2);
          // unsigned long long int out_subscript7 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   (x << left_shift(Output_depth_dim))
          //   + (m+3);
          // unsigned long long int out_subscript8 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   ((x+1) << left_shift(Output_depth_dim))
          //   + (m+3);



          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
           tempLo = _mm256_castps256_ps128(temp3);
           tempHi = _mm256_extractf128_ps(temp3, 1);
           sseSum = _mm_add_ps(tempLo, tempHi);

          float sum3 = _mm_cvtss_f32(sseSum);

          sum3 += bias2;
          out_to_compare_with_FP[out_subscript + 1] = Relu_float(sum3);


          temp4 = _mm256_hadd_ps(temp4, temp4);
          temp4 = _mm256_hadd_ps(temp4, temp4);
           tempLo2 = _mm256_castps256_ps128(temp4);
           tempHi2 = _mm256_extractf128_ps(temp4, 1);
           sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum4 = _mm_cvtss_f32(sseSum2);

          sum4 += bias2;
          out_to_compare_with_FP[out_subscript2 + 1] = Relu_float(sum4);


          temp5 = _mm256_hadd_ps(temp5, temp5);
          temp5 = _mm256_hadd_ps(temp5, temp5);
           tempLo = _mm256_castps256_ps128(temp5);
           tempHi = _mm256_extractf128_ps(temp5, 1);
           sseSum = _mm_add_ps(tempLo, tempHi);

          float sum5 = _mm_cvtss_f32(sseSum);

          sum5 += bias3;
          out_to_compare_with_FP[out_subscript + 2] = Relu_float(sum5);


          temp6 = _mm256_hadd_ps(temp6, temp6);
          temp6 = _mm256_hadd_ps(temp6, temp6);
           tempLo2 = _mm256_castps256_ps128(temp6);
           tempHi2 = _mm256_extractf128_ps(temp6, 1);
           sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum6 = _mm_cvtss_f32(sseSum2);

          sum6 += bias3;
          out_to_compare_with_FP[out_subscript2 + 2] = Relu_float(sum6);


          temp7 = _mm256_hadd_ps(temp7, temp7);
          temp7 = _mm256_hadd_ps(temp7, temp7);
           tempLo = _mm256_castps256_ps128(temp7);
           tempHi = _mm256_extractf128_ps(temp7, 1);
           sseSum = _mm_add_ps(tempLo, tempHi);

          float sum7 = _mm_cvtss_f32(sseSum);

          sum7 += bias4;
          out_to_compare_with_FP[out_subscript + 3] = Relu_float(sum7);


          temp8 = _mm256_hadd_ps(temp8, temp8);
          temp8 = _mm256_hadd_ps(temp8, temp8);
           tempLo2 = _mm256_castps256_ps128(temp8);
           tempHi2 = _mm256_extractf128_ps(temp8, 1);
           sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum8 = _mm_cvtss_f32(sseSum2);

          sum8 += bias4;
          out_to_compare_with_FP[out_subscript2 + 3] = Relu_float(sum8);
        }
      }
    }
  }
  #undef left_shift
  printf("\n from optv7 x2 m4 - moving ops outside loop %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// applied left shift where possible - optimised_layer_v5_x3m3_loop_interchange_m_FP()
// ~86 GFLOPS, 
int optimised_layer_v7_x3m3_ops_outside_loop_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  #define left_shift(x) (__builtin_ctz(x))  // inline function effectively using inline assembly to return num of trailing zeros (i.e. 128 returns 7, 2 * 128 == 2 << 7)
  
  float bias, bias2, bias3;
  __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9;

  unsigned long long int in_yx_depth = Input_Y_dim * Input_X_dim << left_shift(Input_depth_dim);
  unsigned long long int in_x_depth = Input_X_dim  << left_shift(Input_depth_dim);

  unsigned long long int mask_yx_depth = Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim);
  unsigned long long int mask_x_depth = Mask_X_dim << left_shift(Input_depth_dim);

  unsigned long long int out_yx_depth = Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim);
  unsigned long long int out_x_depth = Output_X_dim << left_shift(Output_depth_dim);

  // Calculate loop bounds for unrolled loops
  int m_bound = (Output_depth_dim/ 3) * 3;
  int x_bound = (Output_X_dim/ 3) * 3; 

  unsigned int m, x;

  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
      for (m = 0; m < m_bound; m+=3) { //channels
        bias = bias_array_FP[m];
        bias2 = bias_array_FP[m+1];
        bias3 = bias_array_FP[m+2];

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
          // temp = 0.0f;

          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                unsigned long long int in_subscript = b * (in_yx_depth)
                  + ((y * Stride_Y_dim + off_y) * in_x_depth)
                  + ((x * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                  + d;
                unsigned long long int in_subscript2 = b * (in_yx_depth)
                  + ((y * Stride_Y_dim + off_y) * in_x_depth)
                  + (((x+1) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                  + d;
                unsigned long long int in_subscript3 = b * (in_yx_depth)
                  + ((y * Stride_Y_dim + off_y) * in_x_depth)
                  + (((x+2) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                  + d;


                unsigned long long int filter_subscript = (m * mask_yx_depth)
                  + (off_y * mask_x_depth)
                  + (off_x << left_shift(Input_depth_dim))
                  + d;
                unsigned long long int filter_subscript2 = ((m+1) * mask_yx_depth)
                  + (off_y * mask_x_depth)
                  + (off_x << left_shift(Input_depth_dim))
                  + d;
                unsigned long long int filter_subscript3 = ((m+2) * mask_yx_depth)
                  + (off_y * mask_x_depth)
                  + (off_x << left_shift(Input_depth_dim))
                  + d;

                __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                __m256 s2 = _mm256_load_ps(&in_FP[in_subscript2]);
                __m256 s3 = _mm256_load_ps(&in_FP[in_subscript3]);

                __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]); 
                __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);
                __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript3]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s, w2));
                temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s2, w2));
                temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s3, w2));
                temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w3));
                temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w3));
                temp9 = _mm256_add_ps(temp9, _mm256_mul_ps(s3, w3));


                // float s = in_FP[in_subscript];
                // float w = filter_FP[filter_subscript];
                // temp = temp + s * w;
              }
            }
          }


          unsigned long long int out_subscript = b * (out_yx_depth) +
            y * (out_x_depth) +
            (x << left_shift(Output_depth_dim))
            + m;
          unsigned long long int out_subscript2 = b * (out_yx_depth) +
            y * (out_x_depth) +
            ((x+1) << left_shift(Output_depth_dim))
            + m;
          unsigned long long int out_subscript3 = b * (out_yx_depth) +
            y * (out_x_depth) +
            ((x+2) << left_shift(Output_depth_dim))
            + m;
          unsigned long long int out_subscript4 = b * (out_yx_depth) +
            y * (out_x_depth) +
            (x << left_shift(Output_depth_dim))
            + (m+1);
          unsigned long long int out_subscript5 = b * (out_yx_depth) +
            y * (out_x_depth) +
            ((x+1) << left_shift(Output_depth_dim))
            + (m+1);
          unsigned long long int out_subscript6 = b * (out_yx_depth) +
            y * (out_x_depth) +
            ((x+2) << left_shift(Output_depth_dim))
            + (m+1);
          unsigned long long int out_subscript7 = b * (out_yx_depth) +
            y * (out_x_depth) +
            (x << left_shift(Output_depth_dim))
            + (m+2);
          unsigned long long int out_subscript8 = b * (out_yx_depth) +
            y * (out_x_depth) +
            ((x+1) << left_shift(Output_depth_dim))
            + (m+2);
          unsigned long long int out_subscript9 = b * (out_yx_depth) +
            y * (out_x_depth) +
            ((x+2) << left_shift(Output_depth_dim))
            + (m+2);



          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
          __m128 tempLo3 = _mm256_castps256_ps128(temp3);
          __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
          __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


          temp4 = _mm256_hadd_ps(temp4, temp4);
          temp4 = _mm256_hadd_ps(temp4, temp4);
          __m128 tempLo4 = _mm256_castps256_ps128(temp4);
          __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
          __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

          float sum4 = _mm_cvtss_f32(sseSum4);

          sum4 += bias2;
          out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);


          temp5 = _mm256_hadd_ps(temp5, temp5);
          temp5 = _mm256_hadd_ps(temp5, temp5);
          __m128 tempLo5 = _mm256_castps256_ps128(temp5);
          __m128 tempHi5 = _mm256_extractf128_ps(temp5, 1);
          __m128 sseSum5 = _mm_add_ps(tempLo5, tempHi5);

          float sum5 = _mm_cvtss_f32(sseSum5);

          sum5 += bias2;
          out_to_compare_with_FP[out_subscript5] = Relu_float(sum5);


          temp6 = _mm256_hadd_ps(temp6, temp6);
          temp6 = _mm256_hadd_ps(temp6, temp6);
          __m128 tempLo6 = _mm256_castps256_ps128(temp6);
          __m128 tempHi6 = _mm256_extractf128_ps(temp6, 1);
          __m128 sseSum6 = _mm_add_ps(tempLo6, tempHi6);

          float sum6 = _mm_cvtss_f32(sseSum6);

          sum6 += bias2;
          out_to_compare_with_FP[out_subscript6] = Relu_float(sum6);


          temp7 = _mm256_hadd_ps(temp7, temp7);
          temp7 = _mm256_hadd_ps(temp7, temp7);
          __m128 tempLo7 = _mm256_castps256_ps128(temp7);
          __m128 tempHi7 = _mm256_extractf128_ps(temp7, 1);
          __m128 sseSum7 = _mm_add_ps(tempLo7, tempHi7);

          float sum7 = _mm_cvtss_f32(sseSum7);

          sum7 += bias3;
          out_to_compare_with_FP[out_subscript7] = Relu_float(sum7);


          temp8 = _mm256_hadd_ps(temp8, temp8);
          temp8 = _mm256_hadd_ps(temp8, temp8);
          __m128 tempLo8 = _mm256_castps256_ps128(temp8);
          __m128 tempHi8 = _mm256_extractf128_ps(temp8, 1);
          __m128 sseSum8 = _mm_add_ps(tempLo8, tempHi8);

          float sum8 = _mm_cvtss_f32(sseSum8);

          sum8 += bias3;
          out_to_compare_with_FP[out_subscript8] = Relu_float(sum8);


          temp9 = _mm256_hadd_ps(temp9, temp9);
          temp9 = _mm256_hadd_ps(temp9, temp9);
          __m128 tempLo9 = _mm256_castps256_ps128(temp9);
          __m128 tempHi9 = _mm256_extractf128_ps(temp9, 1);
          __m128 sseSum9 = _mm_add_ps(tempLo9, tempHi9);

          float sum9 = _mm_cvtss_f32(sseSum9);

          sum9 += bias3;
          out_to_compare_with_FP[out_subscript9] = Relu_float(sum9);
        }
        // overflow/ fallback x loop
        for (; x < Output_X_dim; x++) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                unsigned long long int in_subscript = b * (in_yx_depth)
                  + ((y * Stride_Y_dim + off_y) * in_x_depth)
                  + ((x * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                  + d;

                unsigned long long int filter_subscript = (m * mask_yx_depth)
                  + (off_y * mask_x_depth)
                  + (off_x << left_shift(Input_depth_dim))
                  + d;
                unsigned long long int filter_subscript2 = ((m+1) * mask_yx_depth)
                  + (off_y * mask_x_depth)
                  + (off_x << left_shift(Input_depth_dim))
                  + d;
                unsigned long long int filter_subscript3 = ((m+2) * mask_yx_depth)
                  + (off_y * mask_x_depth)
                  + (off_x << left_shift(Input_depth_dim))
                  + d;

                __m256 s = _mm256_loadu_ps(&in_FP[in_subscript]);

                __m256 w = _mm256_loadu_ps(&filter_FP[filter_subscript]);
                __m256 w2 = _mm256_loadu_ps(&filter_FP[filter_subscript2]);
                __m256 w3 = _mm256_loadu_ps(&filter_FP[filter_subscript3]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s, w2));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w3));

                // float s = in_FP[in_subscript];
                // float w = filter_FP[filter_subscript];
                // temp = temp + s * w;
              }
            }
          }
          unsigned long long int out_subscript = b * (out_yx_depth) +
            y * (out_x_depth) +
            (x << left_shift(Output_depth_dim))
            + m;
          unsigned long long int out_subscript2 = b * (out_yx_depth) +
            y * (out_x_depth) +
            (x << left_shift(Output_depth_dim))
            + (m+1);
          unsigned long long int out_subscript3 = b * (out_yx_depth) +
            y * (out_x_depth) +
            (x << left_shift(Output_depth_dim))
            + (m+2);

          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias2;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
          __m128 tempLo3 = _mm256_castps256_ps128(temp3);
          __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
          __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias3;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);

        }
      }
      for (; m < Output_depth_dim; m++) {
        bias = bias_array_FP[m];

        for (unsigned int x = 0; x < Output_X_dim; x+=4) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          temp4 = _mm256_setzero_ps();

          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                unsigned long long int in_subscript = b * (in_yx_depth)
                  + ((y * Stride_Y_dim + off_y) * in_x_depth)
                  + ((x * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                  + d;
                unsigned long long int in_subscript2 = b * (in_yx_depth)
                  + ((y * Stride_Y_dim + off_y) * in_x_depth)
                  + (((x+1) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                  + d;
                unsigned long long int in_subscript3 = b * (in_yx_depth)
                  + ((y * Stride_Y_dim + off_y) * in_x_depth)
                  + (((x+2) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                  + d;
                unsigned long long int in_subscript4 = b * (in_yx_depth)
                  + ((y * Stride_Y_dim + off_y) * in_x_depth)
                  + (((x+3) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                  + d;

                unsigned long long int filter_subscript = (m * mask_yx_depth)
                  + (off_y * mask_x_depth)
                  + (off_x << left_shift(Input_depth_dim))
                  + d;

                __m256 s = _mm256_loadu_ps(&in_FP[in_subscript]);
                __m256 s2 = _mm256_loadu_ps(&in_FP[in_subscript2]);
                __m256 s3 = _mm256_loadu_ps(&in_FP[in_subscript3]);
                __m256 s4 = _mm256_loadu_ps(&in_FP[in_subscript4]);

                __m256 w = _mm256_loadu_ps(&filter_FP[filter_subscript]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s4, w));
              }
            }
          }

          unsigned long long int out_subscript = b * (out_yx_depth) +
            y * (out_x_depth) +
            (x << left_shift(Output_depth_dim))
            + m;
          unsigned long long int out_subscript2 = b * (out_yx_depth) +
            y * (out_x_depth) +
            ((x+1) << left_shift(Output_depth_dim))
            + m;
          unsigned long long int out_subscript3 = b * (out_yx_depth) +
            y * (out_x_depth) +
            ((x+2) << left_shift(Output_depth_dim))
            + m;
          unsigned long long int out_subscript4 = b * (out_yx_depth) +
            y * (out_x_depth) +
            ((x+3) << left_shift(Output_depth_dim))
            + m;

          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = Relu_float(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias;
          out_to_compare_with_FP[out_subscript2] = Relu_float(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
          __m128 tempLo3 = _mm256_castps256_ps128(temp3);
          __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
          __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias;
          out_to_compare_with_FP[out_subscript3] = Relu_float(sum3);


          temp4 = _mm256_hadd_ps(temp4, temp4);
          temp4 = _mm256_hadd_ps(temp4, temp4);
          __m128 tempLo4 = _mm256_castps256_ps128(temp4);
          __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
          __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

          float sum4 = _mm_cvtss_f32(sseSum4);

          sum4 += bias;
          out_to_compare_with_FP[out_subscript4] = Relu_float(sum4);

        }
      }
    }
  }
  #undef left_shift
  printf("\n from optv7 x3 m3 - moving ops outside loop %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}
 





// v8 inlining ReLu function - using v7 funcs as base


// inline ReLu operation - optimised_layer_v7_x2m4_ops_outside_loop_FP()
// ~86 GFLOPS
int optimised_layer_v8_x2m4_inline_relu_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  #define left_shift(x) (__builtin_ctz(x))  // inline function effectively using inline assembly to return num of trailing zeros (i.e. 128 returns 7, 2 * 128 == 2 << 7)
  #define RELU(x) ((x) < 0.0f ? 0.0f : (x)) // return x if greater than 0, else return 0

  float bias, bias2, bias3, bias4;
  __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

  unsigned long long int in_yx_depth = Input_Y_dim * Input_X_dim << left_shift(Input_depth_dim);
  unsigned long long int in_x_depth = Input_X_dim  << left_shift(Input_depth_dim);

  unsigned long long int mask_yx_depth = Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim);
  unsigned long long int mask_x_depth = Mask_X_dim << left_shift(Input_depth_dim);

  unsigned long long int out_yx_depth = Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim);
  unsigned long long int out_x_depth = Output_X_dim << left_shift(Output_depth_dim);

  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
      for (unsigned int m = 0; m < Output_depth_dim; m+=4) { //channels
        bias = bias_array_FP[m];
        bias2 = bias_array_FP[m+1];
        bias3 = bias_array_FP[m+2];
        bias4 = bias_array_FP[m+3];

        for (unsigned int x = 0; x < Output_X_dim; x+=2) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          temp4 = _mm256_setzero_ps();
          temp5 = _mm256_setzero_ps();
          temp6 = _mm256_setzero_ps();
          temp7 = _mm256_setzero_ps();
          temp8 = _mm256_setzero_ps();
          // temp = 0.0f;

          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                unsigned long long int in_subscript = b * (in_yx_depth)
                  + ((y * Stride_Y_dim + off_y) * in_x_depth)
                  + ((x * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                  + d;
                // unsigned long long int in_subscript2 = b * (Input_Y_dim * Input_X_dim << left_shift(Input_depth_dim))
                //   + ((y * Stride_Y_dim + off_y) * Input_X_dim << left_shift(Input_depth_dim))
                //   + (((x+1) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                //   + d;


                unsigned long long int filter_subscript = (m * mask_yx_depth)
                  + (off_y * mask_x_depth)
                  + (off_x << left_shift(Input_depth_dim))
                  + d;
                // unsigned long long int filter_subscript2 = ((m+1) * Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))
                //   + (off_y * Mask_X_dim << left_shift(Input_depth_dim))
                //   + (off_x << left_shift(Input_depth_dim))
                //   + d;
                // unsigned long long int filter_subscript3 = ((m+2) * Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))
                //   + (off_y * Mask_X_dim << left_shift(Input_depth_dim))
                //   + (off_x << left_shift(Input_depth_dim))
                //   + d;
                // unsigned long long int filter_subscript4 = ((m+3) * Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))
                //   + (off_y * Mask_X_dim << left_shift(Input_depth_dim))
                //   + (off_x << left_shift(Input_depth_dim))
                //   + d;

                __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                __m256 s2 = _mm256_load_ps(&in_FP[in_subscript + Input_depth_dim]);

                __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]);
                __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript + (Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))]);
                __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript + (2 * (Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim)))]);
                __m256 w4 = _mm256_load_ps(&filter_FP[filter_subscript + (3 * (Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim)))]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w2));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s2, w2));
                temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s, w3));
                temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s2, w3));
                temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w4));
                temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w4));


                // float s = in_FP[in_subscript];
                // float w = filter_FP[filter_subscript];
                // temp = temp + s * w;
              }
            }
          }


          unsigned long long int out_subscript = b * (out_yx_depth) 
            + y * (out_x_depth) 
            + (x << left_shift(Output_depth_dim))
            + m;
          unsigned long long int out_subscript2 = out_subscript + Output_depth_dim;
          // unsigned long long int out_subscript2 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   ((x+1) << left_shift(Output_depth_dim))
          //   + m;
          // unsigned long long int out_subscript3 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   (x << left_shift(Output_depth_dim))
          //   + (m+1);
          // unsigned long long int out_subscript4 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   ((x+1) << left_shift(Output_depth_dim))
          //   + (m+1);
          // unsigned long long int out_subscript5 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   (x << left_shift(Output_depth_dim))
          //   + (m+2);
          // unsigned long long int out_subscript6 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   ((x+1) << left_shift(Output_depth_dim))
          //   + (m+2);
          // unsigned long long int out_subscript7 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   (x << left_shift(Output_depth_dim))
          //   + (m+3);
          // unsigned long long int out_subscript8 = b * (Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim)) +
          //   y * (Output_X_dim << left_shift(Output_depth_dim)) +
          //   ((x+1) << left_shift(Output_depth_dim))
          //   + (m+3);



          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = RELU(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias;
          out_to_compare_with_FP[out_subscript2] = RELU(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
           tempLo = _mm256_castps256_ps128(temp3);
           tempHi = _mm256_extractf128_ps(temp3, 1);
           sseSum = _mm_add_ps(tempLo, tempHi);

          float sum3 = _mm_cvtss_f32(sseSum);

          sum3 += bias2;
          out_to_compare_with_FP[out_subscript + 1] = RELU(sum3);


          temp4 = _mm256_hadd_ps(temp4, temp4);
          temp4 = _mm256_hadd_ps(temp4, temp4);
           tempLo2 = _mm256_castps256_ps128(temp4);
           tempHi2 = _mm256_extractf128_ps(temp4, 1);
           sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum4 = _mm_cvtss_f32(sseSum2);

          sum4 += bias2;
          out_to_compare_with_FP[out_subscript2 + 1] = RELU(sum4);


          temp5 = _mm256_hadd_ps(temp5, temp5);
          temp5 = _mm256_hadd_ps(temp5, temp5);
           tempLo = _mm256_castps256_ps128(temp5);
           tempHi = _mm256_extractf128_ps(temp5, 1);
           sseSum = _mm_add_ps(tempLo, tempHi);

          float sum5 = _mm_cvtss_f32(sseSum);

          sum5 += bias3;
          out_to_compare_with_FP[out_subscript + 2] = RELU(sum5);


          temp6 = _mm256_hadd_ps(temp6, temp6);
          temp6 = _mm256_hadd_ps(temp6, temp6);
           tempLo2 = _mm256_castps256_ps128(temp6);
           tempHi2 = _mm256_extractf128_ps(temp6, 1);
           sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum6 = _mm_cvtss_f32(sseSum2);

          sum6 += bias3;
          out_to_compare_with_FP[out_subscript2 + 2] = RELU(sum6);


          temp7 = _mm256_hadd_ps(temp7, temp7);
          temp7 = _mm256_hadd_ps(temp7, temp7);
           tempLo = _mm256_castps256_ps128(temp7);
           tempHi = _mm256_extractf128_ps(temp7, 1);
           sseSum = _mm_add_ps(tempLo, tempHi);

          float sum7 = _mm_cvtss_f32(sseSum);

          sum7 += bias4;
          out_to_compare_with_FP[out_subscript + 3] = RELU(sum7);


          temp8 = _mm256_hadd_ps(temp8, temp8);
          temp8 = _mm256_hadd_ps(temp8, temp8);
           tempLo2 = _mm256_castps256_ps128(temp8);
           tempHi2 = _mm256_extractf128_ps(temp8, 1);
           sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum8 = _mm_cvtss_f32(sseSum2);

          sum8 += bias4;
          out_to_compare_with_FP[out_subscript2 + 3] = RELU(sum8);
        }
      }
    }
  }
  #undef left_shift
  #undef RELU
  printf("\n from optv8 x2 m4 - inlined ReLu %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// inline ReLu operation - optimised_layer_v7_x3m3_ops_outside_loop_FP()
// ~87 GFLOPS, 
int optimised_layer_v8_x3m3_inline_relu_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  #define left_shift(x) (__builtin_ctz(x))  // inline function effectively using inline assembly to return num of trailing zeros (i.e. 128 returns 7, 2 * 128 == 2 << 7)
  #define RELU(x) ((x) < 0.0f ? 0.0f : (x)) // return x if greater than 0, else return 0
  
  float bias, bias2, bias3;
  __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9;

  unsigned long long int in_yx_depth = Input_Y_dim * Input_X_dim << left_shift(Input_depth_dim);
  unsigned long long int in_x_depth = Input_X_dim  << left_shift(Input_depth_dim);

  unsigned long long int mask_yx_depth = Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim);
  unsigned long long int mask_x_depth = Mask_X_dim << left_shift(Input_depth_dim);

  unsigned long long int out_yx_depth = Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim);
  unsigned long long int out_x_depth = Output_X_dim << left_shift(Output_depth_dim);

  // Calculate loop bounds for unrolled loops
  int m_bound = (Output_depth_dim/ 3) * 3;
  int x_bound = (Output_X_dim/ 3) * 3; 

  unsigned int m, x;

  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
      for (m = 0; m < m_bound; m+=3) { //channels
        bias = bias_array_FP[m];
        bias2 = bias_array_FP[m+1];
        bias3 = bias_array_FP[m+2];

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
          // temp = 0.0f;

          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                unsigned long long int in_subscript = b * (in_yx_depth)
                  + ((y * Stride_Y_dim + off_y) * in_x_depth)
                  + ((x * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                  + d;
                unsigned long long int in_subscript2 = b * (in_yx_depth)
                  + ((y * Stride_Y_dim + off_y) * in_x_depth)
                  + (((x+1) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                  + d;
                unsigned long long int in_subscript3 = b * (in_yx_depth)
                  + ((y * Stride_Y_dim + off_y) * in_x_depth)
                  + (((x+2) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                  + d;


                unsigned long long int filter_subscript = (m * mask_yx_depth)
                  + (off_y * mask_x_depth)
                  + (off_x << left_shift(Input_depth_dim))
                  + d;
                unsigned long long int filter_subscript2 = ((m+1) * mask_yx_depth)
                  + (off_y * mask_x_depth)
                  + (off_x << left_shift(Input_depth_dim))
                  + d;
                unsigned long long int filter_subscript3 = ((m+2) * mask_yx_depth)
                  + (off_y * mask_x_depth)
                  + (off_x << left_shift(Input_depth_dim))
                  + d;

                __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                __m256 s2 = _mm256_load_ps(&in_FP[in_subscript2]);
                __m256 s3 = _mm256_load_ps(&in_FP[in_subscript3]);

                __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]); 
                __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);
                __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript3]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s, w2));
                temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s2, w2));
                temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s3, w2));
                temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w3));
                temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w3));
                temp9 = _mm256_add_ps(temp9, _mm256_mul_ps(s3, w3));


                // float s = in_FP[in_subscript];
                // float w = filter_FP[filter_subscript];
                // temp = temp + s * w;
              }
            }
          }


          unsigned long long int out_subscript = b * (out_yx_depth) +
            y * (out_x_depth) +
            (x << left_shift(Output_depth_dim))
            + m;
          unsigned long long int out_subscript2 = b * (out_yx_depth) +
            y * (out_x_depth) +
            ((x+1) << left_shift(Output_depth_dim))
            + m;
          unsigned long long int out_subscript3 = b * (out_yx_depth) +
            y * (out_x_depth) +
            ((x+2) << left_shift(Output_depth_dim))
            + m;
          unsigned long long int out_subscript4 = b * (out_yx_depth) +
            y * (out_x_depth) +
            (x << left_shift(Output_depth_dim))
            + (m+1);
          unsigned long long int out_subscript5 = b * (out_yx_depth) +
            y * (out_x_depth) +
            ((x+1) << left_shift(Output_depth_dim))
            + (m+1);
          unsigned long long int out_subscript6 = b * (out_yx_depth) +
            y * (out_x_depth) +
            ((x+2) << left_shift(Output_depth_dim))
            + (m+1);
          unsigned long long int out_subscript7 = b * (out_yx_depth) +
            y * (out_x_depth) +
            (x << left_shift(Output_depth_dim))
            + (m+2);
          unsigned long long int out_subscript8 = b * (out_yx_depth) +
            y * (out_x_depth) +
            ((x+1) << left_shift(Output_depth_dim))
            + (m+2);
          unsigned long long int out_subscript9 = b * (out_yx_depth) +
            y * (out_x_depth) +
            ((x+2) << left_shift(Output_depth_dim))
            + (m+2);



          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = RELU(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias;
          out_to_compare_with_FP[out_subscript2] = RELU(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
          __m128 tempLo3 = _mm256_castps256_ps128(temp3);
          __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
          __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias;
          out_to_compare_with_FP[out_subscript3] = RELU(sum3);


          temp4 = _mm256_hadd_ps(temp4, temp4);
          temp4 = _mm256_hadd_ps(temp4, temp4);
          __m128 tempLo4 = _mm256_castps256_ps128(temp4);
          __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
          __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

          float sum4 = _mm_cvtss_f32(sseSum4);

          sum4 += bias2;
          out_to_compare_with_FP[out_subscript4] = RELU(sum4);


          temp5 = _mm256_hadd_ps(temp5, temp5);
          temp5 = _mm256_hadd_ps(temp5, temp5);
          __m128 tempLo5 = _mm256_castps256_ps128(temp5);
          __m128 tempHi5 = _mm256_extractf128_ps(temp5, 1);
          __m128 sseSum5 = _mm_add_ps(tempLo5, tempHi5);

          float sum5 = _mm_cvtss_f32(sseSum5);

          sum5 += bias2;
          out_to_compare_with_FP[out_subscript5] = RELU(sum5);


          temp6 = _mm256_hadd_ps(temp6, temp6);
          temp6 = _mm256_hadd_ps(temp6, temp6);
          __m128 tempLo6 = _mm256_castps256_ps128(temp6);
          __m128 tempHi6 = _mm256_extractf128_ps(temp6, 1);
          __m128 sseSum6 = _mm_add_ps(tempLo6, tempHi6);

          float sum6 = _mm_cvtss_f32(sseSum6);

          sum6 += bias2;
          out_to_compare_with_FP[out_subscript6] = RELU(sum6);


          temp7 = _mm256_hadd_ps(temp7, temp7);
          temp7 = _mm256_hadd_ps(temp7, temp7);
          __m128 tempLo7 = _mm256_castps256_ps128(temp7);
          __m128 tempHi7 = _mm256_extractf128_ps(temp7, 1);
          __m128 sseSum7 = _mm_add_ps(tempLo7, tempHi7);

          float sum7 = _mm_cvtss_f32(sseSum7);

          sum7 += bias3;
          out_to_compare_with_FP[out_subscript7] = RELU(sum7);


          temp8 = _mm256_hadd_ps(temp8, temp8);
          temp8 = _mm256_hadd_ps(temp8, temp8);
          __m128 tempLo8 = _mm256_castps256_ps128(temp8);
          __m128 tempHi8 = _mm256_extractf128_ps(temp8, 1);
          __m128 sseSum8 = _mm_add_ps(tempLo8, tempHi8);

          float sum8 = _mm_cvtss_f32(sseSum8);

          sum8 += bias3;
          out_to_compare_with_FP[out_subscript8] = RELU(sum8);


          temp9 = _mm256_hadd_ps(temp9, temp9);
          temp9 = _mm256_hadd_ps(temp9, temp9);
          __m128 tempLo9 = _mm256_castps256_ps128(temp9);
          __m128 tempHi9 = _mm256_extractf128_ps(temp9, 1);
          __m128 sseSum9 = _mm_add_ps(tempLo9, tempHi9);

          float sum9 = _mm_cvtss_f32(sseSum9);

          sum9 += bias3;
          out_to_compare_with_FP[out_subscript9] = RELU(sum9);
        }
        // overflow/ fallback x loop
        for (; x < Output_X_dim; x++) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                unsigned long long int in_subscript = b * (in_yx_depth)
                  + ((y * Stride_Y_dim + off_y) * in_x_depth)
                  + ((x * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                  + d;

                unsigned long long int filter_subscript = (m * mask_yx_depth)
                  + (off_y * mask_x_depth)
                  + (off_x << left_shift(Input_depth_dim))
                  + d;
                unsigned long long int filter_subscript2 = ((m+1) * mask_yx_depth)
                  + (off_y * mask_x_depth)
                  + (off_x << left_shift(Input_depth_dim))
                  + d;
                unsigned long long int filter_subscript3 = ((m+2) * mask_yx_depth)
                  + (off_y * mask_x_depth)
                  + (off_x << left_shift(Input_depth_dim))
                  + d;

                __m256 s = _mm256_loadu_ps(&in_FP[in_subscript]);

                __m256 w = _mm256_loadu_ps(&filter_FP[filter_subscript]);
                __m256 w2 = _mm256_loadu_ps(&filter_FP[filter_subscript2]);
                __m256 w3 = _mm256_loadu_ps(&filter_FP[filter_subscript3]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s, w2));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w3));

                // float s = in_FP[in_subscript];
                // float w = filter_FP[filter_subscript];
                // temp = temp + s * w;
              }
            }
          }
          unsigned long long int out_subscript = b * (out_yx_depth) +
            y * (out_x_depth) +
            (x << left_shift(Output_depth_dim))
            + m;
          unsigned long long int out_subscript2 = b * (out_yx_depth) +
            y * (out_x_depth) +
            (x << left_shift(Output_depth_dim))
            + (m+1);
          unsigned long long int out_subscript3 = b * (out_yx_depth) +
            y * (out_x_depth) +
            (x << left_shift(Output_depth_dim))
            + (m+2);

          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = RELU(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias2;
          out_to_compare_with_FP[out_subscript2] = RELU(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
          __m128 tempLo3 = _mm256_castps256_ps128(temp3);
          __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
          __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias3;
          out_to_compare_with_FP[out_subscript3] = RELU(sum3);

        }
      }
      for (; m < Output_depth_dim; m++) {
        bias = bias_array_FP[m];

        for (unsigned int x = 0; x < Output_X_dim; x+=4) {	//Output Width
          temp = _mm256_setzero_ps();
          temp2 = _mm256_setzero_ps();
          temp3 = _mm256_setzero_ps();
          temp4 = _mm256_setzero_ps();

          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                unsigned long long int in_subscript = b * (in_yx_depth)
                  + ((y * Stride_Y_dim + off_y) * in_x_depth)
                  + ((x * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                  + d;
                unsigned long long int in_subscript2 = b * (in_yx_depth)
                  + ((y * Stride_Y_dim + off_y) * in_x_depth)
                  + (((x+1) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                  + d;
                unsigned long long int in_subscript3 = b * (in_yx_depth)
                  + ((y * Stride_Y_dim + off_y) * in_x_depth)
                  + (((x+2) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                  + d;
                unsigned long long int in_subscript4 = b * (in_yx_depth)
                  + ((y * Stride_Y_dim + off_y) * in_x_depth)
                  + (((x+3) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                  + d;

                unsigned long long int filter_subscript = (m * mask_yx_depth)
                  + (off_y * mask_x_depth)
                  + (off_x << left_shift(Input_depth_dim))
                  + d;

                __m256 s = _mm256_loadu_ps(&in_FP[in_subscript]);
                __m256 s2 = _mm256_loadu_ps(&in_FP[in_subscript2]);
                __m256 s3 = _mm256_loadu_ps(&in_FP[in_subscript3]);
                __m256 s4 = _mm256_loadu_ps(&in_FP[in_subscript4]);

                __m256 w = _mm256_loadu_ps(&filter_FP[filter_subscript]);

                temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s4, w));
              }
            }
          }

          unsigned long long int out_subscript = b * (out_yx_depth) +
            y * (out_x_depth) +
            (x << left_shift(Output_depth_dim))
            + m;
          unsigned long long int out_subscript2 = b * (out_yx_depth) +
            y * (out_x_depth) +
            ((x+1) << left_shift(Output_depth_dim))
            + m;
          unsigned long long int out_subscript3 = b * (out_yx_depth) +
            y * (out_x_depth) +
            ((x+2) << left_shift(Output_depth_dim))
            + m;
          unsigned long long int out_subscript4 = b * (out_yx_depth) +
            y * (out_x_depth) +
            ((x+3) << left_shift(Output_depth_dim))
            + m;

          temp = _mm256_hadd_ps(temp, temp);
          temp = _mm256_hadd_ps(temp, temp);
          __m128 tempLo = _mm256_castps256_ps128(temp);
          __m128 tempHi = _mm256_extractf128_ps(temp, 1);
          __m128 sseSum = _mm_add_ps(tempLo, tempHi);

          float sum = _mm_cvtss_f32(sseSum);

          sum += bias;
          out_to_compare_with_FP[out_subscript] = RELU(sum);


          temp2 = _mm256_hadd_ps(temp2, temp2);
          temp2 = _mm256_hadd_ps(temp2, temp2);
          __m128 tempLo2 = _mm256_castps256_ps128(temp2);
          __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
          __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

          float sum2 = _mm_cvtss_f32(sseSum2);

          sum2 += bias;
          out_to_compare_with_FP[out_subscript2] = RELU(sum2);


          temp3 = _mm256_hadd_ps(temp3, temp3);
          temp3 = _mm256_hadd_ps(temp3, temp3);
          __m128 tempLo3 = _mm256_castps256_ps128(temp3);
          __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
          __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

          float sum3 = _mm_cvtss_f32(sseSum3);

          sum3 += bias;
          out_to_compare_with_FP[out_subscript3] = RELU(sum3);


          temp4 = _mm256_hadd_ps(temp4, temp4);
          temp4 = _mm256_hadd_ps(temp4, temp4);
          __m128 tempLo4 = _mm256_castps256_ps128(temp4);
          __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
          __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

          float sum4 = _mm_cvtss_f32(sseSum4);

          sum4 += bias;
          out_to_compare_with_FP[out_subscript4] = RELU(sum4);

        }
      }
    }
  }
  #undef left_shift
  #undef RELU
  printf("\n from optv8 x3 m3 - inlined ReLu %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}
  





// v9 parallelisation - using v8 funcs as base


// parallised with omp - optimised_layer_v8_x2m4_inline_relu_FP()
// 12 threads ~391 GFLOPS, 6 threads ~401 GFLOPS, as program isn't both memory and computationally bound there's no need for 12 threads (results in extra threads stalling?)
int optimised_layer_v9_x2m4_omp_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  #define left_shift(x) (__builtin_ctz(x))  // inline function effectively using inline assembly to return num of trailing zeros (i.e. 128 returns 7, 2 * 128 == 2 << 7)
  #define RELU(x) ((x) < 0.0f ? 0.0f : (x)) // return x if greater than 0, else return 0

  float bias, bias2, bias3, bias4;
  __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

  unsigned long long int in_yx_depth = Input_Y_dim * Input_X_dim << left_shift(Input_depth_dim);
  unsigned long long int in_x_depth = Input_X_dim  << left_shift(Input_depth_dim);

  unsigned long long int mask_yx_depth = Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim);
  unsigned long long int mask_x_depth = Mask_X_dim << left_shift(Input_depth_dim);

  unsigned long long int out_yx_depth = Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim);
  unsigned long long int out_x_depth = Output_X_dim << left_shift(Output_depth_dim);

  #pragma omp parallel for private(bias, bias2, bias3, bias4, temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8) \
    shared(in_yx_depth, in_x_depth, mask_yx_depth, mask_x_depth, out_yx_depth, out_x_depth) default(shared) schedule(static)
  {
      
    for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (unsigned int m = 0; m < Output_depth_dim; m+=4) { //channels
          bias = bias_array_FP[m];
          bias2 = bias_array_FP[m+1];
          bias3 = bias_array_FP[m+2];
          bias4 = bias_array_FP[m+3];

          for (unsigned int x = 0; x < Output_X_dim; x+=2) {	//Output Width
            temp = _mm256_setzero_ps();
            temp2 = _mm256_setzero_ps();
            temp3 = _mm256_setzero_ps();
            temp4 = _mm256_setzero_ps();
            temp5 = _mm256_setzero_ps();
            temp6 = _mm256_setzero_ps();
            temp7 = _mm256_setzero_ps();
            temp8 = _mm256_setzero_ps();
            // temp = 0.0f;

            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                  unsigned long long int in_subscript = b * (in_yx_depth)
                    + ((y * Stride_Y_dim + off_y) * in_x_depth)
                    + ((x * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                    + d;

                  unsigned long long int filter_subscript = (m * mask_yx_depth)
                    + (off_y * mask_x_depth)
                    + (off_x << left_shift(Input_depth_dim))
                    + d;

                  __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                  __m256 s2 = _mm256_load_ps(&in_FP[in_subscript + Input_depth_dim]);

                  __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]);
                  __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript + (Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))]);
                  __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript + (2 * (Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim)))]);
                  __m256 w4 = _mm256_load_ps(&filter_FP[filter_subscript + (3 * (Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim)))]);

                  temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                  temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                  temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w2));
                  temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s2, w2));
                  temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s, w3));
                  temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s2, w3));
                  temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w4));
                  temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w4));

                }
              }
            }


            unsigned long long int out_subscript = b * (out_yx_depth) 
              + y * (out_x_depth) 
              + (x << left_shift(Output_depth_dim))
              + m;
            unsigned long long int out_subscript2 = out_subscript + Output_depth_dim;



            temp = _mm256_hadd_ps(temp, temp);
            temp = _mm256_hadd_ps(temp, temp);
            __m128 tempLo = _mm256_castps256_ps128(temp);
            __m128 tempHi = _mm256_extractf128_ps(temp, 1);
            __m128 sseSum = _mm_add_ps(tempLo, tempHi);

            float sum = _mm_cvtss_f32(sseSum);

            sum += bias;
            out_to_compare_with_FP[out_subscript] = RELU(sum);


            temp2 = _mm256_hadd_ps(temp2, temp2);
            temp2 = _mm256_hadd_ps(temp2, temp2);
            __m128 tempLo2 = _mm256_castps256_ps128(temp2);
            __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
            __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum2 = _mm_cvtss_f32(sseSum2);

            sum2 += bias;
            out_to_compare_with_FP[out_subscript2] = RELU(sum2);


            temp3 = _mm256_hadd_ps(temp3, temp3);
            temp3 = _mm256_hadd_ps(temp3, temp3);
            tempLo = _mm256_castps256_ps128(temp3);
            tempHi = _mm256_extractf128_ps(temp3, 1);
            sseSum = _mm_add_ps(tempLo, tempHi);

            float sum3 = _mm_cvtss_f32(sseSum);

            sum3 += bias2;
            out_to_compare_with_FP[out_subscript + 1] = RELU(sum3);


            temp4 = _mm256_hadd_ps(temp4, temp4);
            temp4 = _mm256_hadd_ps(temp4, temp4);
            tempLo2 = _mm256_castps256_ps128(temp4);
            tempHi2 = _mm256_extractf128_ps(temp4, 1);
            sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum4 = _mm_cvtss_f32(sseSum2);

            sum4 += bias2;
            out_to_compare_with_FP[out_subscript2 + 1] = RELU(sum4);


            temp5 = _mm256_hadd_ps(temp5, temp5);
            temp5 = _mm256_hadd_ps(temp5, temp5);
            tempLo = _mm256_castps256_ps128(temp5);
            tempHi = _mm256_extractf128_ps(temp5, 1);
            sseSum = _mm_add_ps(tempLo, tempHi);

            float sum5 = _mm_cvtss_f32(sseSum);

            sum5 += bias3;
            out_to_compare_with_FP[out_subscript + 2] = RELU(sum5);


            temp6 = _mm256_hadd_ps(temp6, temp6);
            temp6 = _mm256_hadd_ps(temp6, temp6);
            tempLo2 = _mm256_castps256_ps128(temp6);
            tempHi2 = _mm256_extractf128_ps(temp6, 1);
            sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum6 = _mm_cvtss_f32(sseSum2);

            sum6 += bias3;
            out_to_compare_with_FP[out_subscript2 + 2] = RELU(sum6);


            temp7 = _mm256_hadd_ps(temp7, temp7);
            temp7 = _mm256_hadd_ps(temp7, temp7);
            tempLo = _mm256_castps256_ps128(temp7);
            tempHi = _mm256_extractf128_ps(temp7, 1);
            sseSum = _mm_add_ps(tempLo, tempHi);

            float sum7 = _mm_cvtss_f32(sseSum);

            sum7 += bias4;
            out_to_compare_with_FP[out_subscript + 3] = RELU(sum7);


            temp8 = _mm256_hadd_ps(temp8, temp8);
            temp8 = _mm256_hadd_ps(temp8, temp8);
            tempLo2 = _mm256_castps256_ps128(temp8);
            tempHi2 = _mm256_extractf128_ps(temp8, 1);
            sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum8 = _mm_cvtss_f32(sseSum2);

            sum8 += bias4;
            out_to_compare_with_FP[out_subscript2 + 3] = RELU(sum8);
          }
        }
      }
    }
  }
  #undef left_shift
  #undef RELU
  printf("\n from optv9 x2 m4 - parallelized %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// same as above, but with collapse() pragma
// collapse(2) ~474 GFLOPS, collapse(3) ~471 GFLOPS, collapse(4) ~464 GFLOPS - collapse(2) increases perf as it unrolls y loop completely increasing parallelism
int optimised_layer_v9_x2m4_omp_collapsed_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  #define left_shift(x) (__builtin_ctz(x))  // inline function effectively using inline assembly to return num of trailing zeros (i.e. 128 returns 7, 2 * 128 == 2 << 7)
  #define RELU(x) ((x) < 0.0f ? 0.0f : (x)) // return x if greater than 0, else return 0

  float bias, bias2, bias3, bias4;
  __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

  unsigned long long int in_yx_depth = Input_Y_dim * Input_X_dim << left_shift(Input_depth_dim);
  unsigned long long int in_x_depth = Input_X_dim  << left_shift(Input_depth_dim);

  unsigned long long int mask_yx_depth = Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim);
  unsigned long long int mask_x_depth = Mask_X_dim << left_shift(Input_depth_dim);

  unsigned long long int out_yx_depth = Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim);
  unsigned long long int out_x_depth = Output_X_dim << left_shift(Output_depth_dim);

  #pragma omp parallel for private(bias, bias2, bias3, bias4, temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8) \
    shared(in_yx_depth, in_x_depth, mask_yx_depth, mask_x_depth, out_yx_depth, out_x_depth) default(shared) collapse(2) schedule(static)
  {
      
    for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (unsigned int m = 0; m < Output_depth_dim; m+=4) { //channels
          bias = bias_array_FP[m];
          bias2 = bias_array_FP[m+1];
          bias3 = bias_array_FP[m+2];
          bias4 = bias_array_FP[m+3];

          for (unsigned int x = 0; x < Output_X_dim; x+=2) {	//Output Width
            temp = _mm256_setzero_ps();
            temp2 = _mm256_setzero_ps();
            temp3 = _mm256_setzero_ps();
            temp4 = _mm256_setzero_ps();
            temp5 = _mm256_setzero_ps();
            temp6 = _mm256_setzero_ps();
            temp7 = _mm256_setzero_ps();
            temp8 = _mm256_setzero_ps();
            // temp = 0.0f;

            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                  unsigned long long int in_subscript = b * (in_yx_depth)
                    + ((y * Stride_Y_dim + off_y) * in_x_depth)
                    + ((x * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                    + d;

                  unsigned long long int filter_subscript = (m * mask_yx_depth)
                    + (off_y * mask_x_depth)
                    + (off_x << left_shift(Input_depth_dim))
                    + d;

                  __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                  __m256 s2 = _mm256_load_ps(&in_FP[in_subscript + Input_depth_dim]);

                  __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]);
                  __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript + (Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))]);
                  __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript + (2 * (Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim)))]);
                  __m256 w4 = _mm256_load_ps(&filter_FP[filter_subscript + (3 * (Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim)))]);

                  temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                  temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                  temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w2));
                  temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s2, w2));
                  temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s, w3));
                  temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s2, w3));
                  temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w4));
                  temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w4));

                }
              }
            }


            unsigned long long int out_subscript = b * (out_yx_depth) 
              + y * (out_x_depth) 
              + (x << left_shift(Output_depth_dim))
              + m;
            unsigned long long int out_subscript2 = out_subscript + Output_depth_dim;



            temp = _mm256_hadd_ps(temp, temp);
            temp = _mm256_hadd_ps(temp, temp);
            __m128 tempLo = _mm256_castps256_ps128(temp);
            __m128 tempHi = _mm256_extractf128_ps(temp, 1);
            __m128 sseSum = _mm_add_ps(tempLo, tempHi);

            float sum = _mm_cvtss_f32(sseSum);

            sum += bias;
            out_to_compare_with_FP[out_subscript] = RELU(sum);


            temp2 = _mm256_hadd_ps(temp2, temp2);
            temp2 = _mm256_hadd_ps(temp2, temp2);
            __m128 tempLo2 = _mm256_castps256_ps128(temp2);
            __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
            __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum2 = _mm_cvtss_f32(sseSum2);

            sum2 += bias;
            out_to_compare_with_FP[out_subscript2] = RELU(sum2);


            temp3 = _mm256_hadd_ps(temp3, temp3);
            temp3 = _mm256_hadd_ps(temp3, temp3);
            tempLo = _mm256_castps256_ps128(temp3);
            tempHi = _mm256_extractf128_ps(temp3, 1);
            sseSum = _mm_add_ps(tempLo, tempHi);

            float sum3 = _mm_cvtss_f32(sseSum);

            sum3 += bias2;
            out_to_compare_with_FP[out_subscript + 1] = RELU(sum3);


            temp4 = _mm256_hadd_ps(temp4, temp4);
            temp4 = _mm256_hadd_ps(temp4, temp4);
            tempLo2 = _mm256_castps256_ps128(temp4);
            tempHi2 = _mm256_extractf128_ps(temp4, 1);
            sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum4 = _mm_cvtss_f32(sseSum2);

            sum4 += bias2;
            out_to_compare_with_FP[out_subscript2 + 1] = RELU(sum4);


            temp5 = _mm256_hadd_ps(temp5, temp5);
            temp5 = _mm256_hadd_ps(temp5, temp5);
            tempLo = _mm256_castps256_ps128(temp5);
            tempHi = _mm256_extractf128_ps(temp5, 1);
            sseSum = _mm_add_ps(tempLo, tempHi);

            float sum5 = _mm_cvtss_f32(sseSum);

            sum5 += bias3;
            out_to_compare_with_FP[out_subscript + 2] = RELU(sum5);


            temp6 = _mm256_hadd_ps(temp6, temp6);
            temp6 = _mm256_hadd_ps(temp6, temp6);
            tempLo2 = _mm256_castps256_ps128(temp6);
            tempHi2 = _mm256_extractf128_ps(temp6, 1);
            sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum6 = _mm_cvtss_f32(sseSum2);

            sum6 += bias3;
            out_to_compare_with_FP[out_subscript2 + 2] = RELU(sum6);


            temp7 = _mm256_hadd_ps(temp7, temp7);
            temp7 = _mm256_hadd_ps(temp7, temp7);
            tempLo = _mm256_castps256_ps128(temp7);
            tempHi = _mm256_extractf128_ps(temp7, 1);
            sseSum = _mm_add_ps(tempLo, tempHi);

            float sum7 = _mm_cvtss_f32(sseSum);

            sum7 += bias4;
            out_to_compare_with_FP[out_subscript + 3] = RELU(sum7);


            temp8 = _mm256_hadd_ps(temp8, temp8);
            temp8 = _mm256_hadd_ps(temp8, temp8);
            tempLo2 = _mm256_castps256_ps128(temp8);
            tempHi2 = _mm256_extractf128_ps(temp8, 1);
            sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum8 = _mm_cvtss_f32(sseSum2);

            sum8 += bias4;
            out_to_compare_with_FP[out_subscript2 + 3] = RELU(sum8);
          }
        }
      }
    }
  }
  #undef left_shift
  #undef RELU
  printf("\n from optv9 x2 m4 - parallelized, collapsed %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// parallised with omp - optimised_layer_v8_x3m3_inline_relu_FP()
// 12 threads ~402 GFLOPS, 6 threads ~412 GFLOPS, as program isn't both memory and computationally bound there's no need for 12 threads (results in extra threads stalling?)
int optimised_layer_v9_x3m3_omp_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  #define left_shift(x) (__builtin_ctz(x))  // inline function effectively using inline assembly to return num of trailing zeros (i.e. 128 returns 7, 2 * 128 == 2 << 7)
  #define RELU(x) ((x) < 0.0f ? 0.0f : (x)) // return x if greater than 0, else return 0
  
  float bias, bias2, bias3;
  __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9;

  unsigned long long int in_yx_depth = Input_Y_dim * Input_X_dim << left_shift(Input_depth_dim);
  unsigned long long int in_x_depth = Input_X_dim  << left_shift(Input_depth_dim);

  unsigned long long int mask_yx_depth = Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim);
  unsigned long long int mask_x_depth = Mask_X_dim << left_shift(Input_depth_dim);

  unsigned long long int out_yx_depth = Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim);
  unsigned long long int out_x_depth = Output_X_dim << left_shift(Output_depth_dim);

  // Calculate loop bounds for unrolled loops
  int m_bound = (Output_depth_dim/ 3) * 3;
  int x_bound = (Output_X_dim/ 3) * 3; 

  unsigned int m, x;
  #pragma omp parallel for private(bias, bias2, bias3, temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9, m, x) \
    shared(in_yx_depth, in_x_depth, mask_yx_depth, mask_x_depth, out_yx_depth, out_x_depth, m_bound, x_bound) default(shared) schedule(static)
  {
    for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (m = 0; m < m_bound; m+=3) { //channels
          bias = bias_array_FP[m];
          bias2 = bias_array_FP[m+1];
          bias3 = bias_array_FP[m+2];

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
            // temp = 0.0f;

            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                  unsigned long long int in_subscript = b * (in_yx_depth)
                    + ((y * Stride_Y_dim + off_y) * in_x_depth)
                    + ((x * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                    + d;
                  unsigned long long int in_subscript2 = b * (in_yx_depth)
                    + ((y * Stride_Y_dim + off_y) * in_x_depth)
                    + (((x+1) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                    + d;
                  unsigned long long int in_subscript3 = b * (in_yx_depth)
                    + ((y * Stride_Y_dim + off_y) * in_x_depth)
                    + (((x+2) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                    + d;


                  unsigned long long int filter_subscript = (m * mask_yx_depth)
                    + (off_y * mask_x_depth)
                    + (off_x << left_shift(Input_depth_dim))
                    + d;
                  unsigned long long int filter_subscript2 = ((m+1) * mask_yx_depth)
                    + (off_y * mask_x_depth)
                    + (off_x << left_shift(Input_depth_dim))
                    + d;
                  unsigned long long int filter_subscript3 = ((m+2) * mask_yx_depth)
                    + (off_y * mask_x_depth)
                    + (off_x << left_shift(Input_depth_dim))
                    + d;

                  __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                  __m256 s2 = _mm256_load_ps(&in_FP[in_subscript2]);
                  __m256 s3 = _mm256_load_ps(&in_FP[in_subscript3]);

                  __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]); 
                  __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);
                  __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript3]);

                  temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                  temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                  temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                  temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s, w2));
                  temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s2, w2));
                  temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s3, w2));
                  temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w3));
                  temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w3));
                  temp9 = _mm256_add_ps(temp9, _mm256_mul_ps(s3, w3));


                  // float s = in_FP[in_subscript];
                  // float w = filter_FP[filter_subscript];
                  // temp = temp + s * w;
                }
              }
            }


            unsigned long long int out_subscript = b * (out_yx_depth) +
              y * (out_x_depth) +
              (x << left_shift(Output_depth_dim))
              + m;
            unsigned long long int out_subscript2 = b * (out_yx_depth) +
              y * (out_x_depth) +
              ((x+1) << left_shift(Output_depth_dim))
              + m;
            unsigned long long int out_subscript3 = b * (out_yx_depth) +
              y * (out_x_depth) +
              ((x+2) << left_shift(Output_depth_dim))
              + m;
            unsigned long long int out_subscript4 = b * (out_yx_depth) +
              y * (out_x_depth) +
              (x << left_shift(Output_depth_dim))
              + (m+1);
            unsigned long long int out_subscript5 = b * (out_yx_depth) +
              y * (out_x_depth) +
              ((x+1) << left_shift(Output_depth_dim))
              + (m+1);
            unsigned long long int out_subscript6 = b * (out_yx_depth) +
              y * (out_x_depth) +
              ((x+2) << left_shift(Output_depth_dim))
              + (m+1);
            unsigned long long int out_subscript7 = b * (out_yx_depth) +
              y * (out_x_depth) +
              (x << left_shift(Output_depth_dim))
              + (m+2);
            unsigned long long int out_subscript8 = b * (out_yx_depth) +
              y * (out_x_depth) +
              ((x+1) << left_shift(Output_depth_dim))
              + (m+2);
            unsigned long long int out_subscript9 = b * (out_yx_depth) +
              y * (out_x_depth) +
              ((x+2) << left_shift(Output_depth_dim))
              + (m+2);



            temp = _mm256_hadd_ps(temp, temp);
            temp = _mm256_hadd_ps(temp, temp);
            __m128 tempLo = _mm256_castps256_ps128(temp);
            __m128 tempHi = _mm256_extractf128_ps(temp, 1);
            __m128 sseSum = _mm_add_ps(tempLo, tempHi);

            float sum = _mm_cvtss_f32(sseSum);

            sum += bias;
            out_to_compare_with_FP[out_subscript] = RELU(sum);


            temp2 = _mm256_hadd_ps(temp2, temp2);
            temp2 = _mm256_hadd_ps(temp2, temp2);
            __m128 tempLo2 = _mm256_castps256_ps128(temp2);
            __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
            __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum2 = _mm_cvtss_f32(sseSum2);

            sum2 += bias;
            out_to_compare_with_FP[out_subscript2] = RELU(sum2);


            temp3 = _mm256_hadd_ps(temp3, temp3);
            temp3 = _mm256_hadd_ps(temp3, temp3);
            __m128 tempLo3 = _mm256_castps256_ps128(temp3);
            __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
            __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

            float sum3 = _mm_cvtss_f32(sseSum3);

            sum3 += bias;
            out_to_compare_with_FP[out_subscript3] = RELU(sum3);


            temp4 = _mm256_hadd_ps(temp4, temp4);
            temp4 = _mm256_hadd_ps(temp4, temp4);
            __m128 tempLo4 = _mm256_castps256_ps128(temp4);
            __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
            __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

            float sum4 = _mm_cvtss_f32(sseSum4);

            sum4 += bias2;
            out_to_compare_with_FP[out_subscript4] = RELU(sum4);


            temp5 = _mm256_hadd_ps(temp5, temp5);
            temp5 = _mm256_hadd_ps(temp5, temp5);
            __m128 tempLo5 = _mm256_castps256_ps128(temp5);
            __m128 tempHi5 = _mm256_extractf128_ps(temp5, 1);
            __m128 sseSum5 = _mm_add_ps(tempLo5, tempHi5);

            float sum5 = _mm_cvtss_f32(sseSum5);

            sum5 += bias2;
            out_to_compare_with_FP[out_subscript5] = RELU(sum5);


            temp6 = _mm256_hadd_ps(temp6, temp6);
            temp6 = _mm256_hadd_ps(temp6, temp6);
            __m128 tempLo6 = _mm256_castps256_ps128(temp6);
            __m128 tempHi6 = _mm256_extractf128_ps(temp6, 1);
            __m128 sseSum6 = _mm_add_ps(tempLo6, tempHi6);

            float sum6 = _mm_cvtss_f32(sseSum6);

            sum6 += bias2;
            out_to_compare_with_FP[out_subscript6] = RELU(sum6);


            temp7 = _mm256_hadd_ps(temp7, temp7);
            temp7 = _mm256_hadd_ps(temp7, temp7);
            __m128 tempLo7 = _mm256_castps256_ps128(temp7);
            __m128 tempHi7 = _mm256_extractf128_ps(temp7, 1);
            __m128 sseSum7 = _mm_add_ps(tempLo7, tempHi7);

            float sum7 = _mm_cvtss_f32(sseSum7);

            sum7 += bias3;
            out_to_compare_with_FP[out_subscript7] = RELU(sum7);


            temp8 = _mm256_hadd_ps(temp8, temp8);
            temp8 = _mm256_hadd_ps(temp8, temp8);
            __m128 tempLo8 = _mm256_castps256_ps128(temp8);
            __m128 tempHi8 = _mm256_extractf128_ps(temp8, 1);
            __m128 sseSum8 = _mm_add_ps(tempLo8, tempHi8);

            float sum8 = _mm_cvtss_f32(sseSum8);

            sum8 += bias3;
            out_to_compare_with_FP[out_subscript8] = RELU(sum8);


            temp9 = _mm256_hadd_ps(temp9, temp9);
            temp9 = _mm256_hadd_ps(temp9, temp9);
            __m128 tempLo9 = _mm256_castps256_ps128(temp9);
            __m128 tempHi9 = _mm256_extractf128_ps(temp9, 1);
            __m128 sseSum9 = _mm_add_ps(tempLo9, tempHi9);

            float sum9 = _mm_cvtss_f32(sseSum9);

            sum9 += bias3;
            out_to_compare_with_FP[out_subscript9] = RELU(sum9);
          }
          // overflow/ fallback x loop
          for (; x < Output_X_dim; x++) {	//Output Width
            temp = _mm256_setzero_ps();
            temp2 = _mm256_setzero_ps();
            temp3 = _mm256_setzero_ps();
            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                  unsigned long long int in_subscript = b * (in_yx_depth)
                    + ((y * Stride_Y_dim + off_y) * in_x_depth)
                    + ((x * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                    + d;

                  unsigned long long int filter_subscript = (m * mask_yx_depth)
                    + (off_y * mask_x_depth)
                    + (off_x << left_shift(Input_depth_dim))
                    + d;
                  unsigned long long int filter_subscript2 = ((m+1) * mask_yx_depth)
                    + (off_y * mask_x_depth)
                    + (off_x << left_shift(Input_depth_dim))
                    + d;
                  unsigned long long int filter_subscript3 = ((m+2) * mask_yx_depth)
                    + (off_y * mask_x_depth)
                    + (off_x << left_shift(Input_depth_dim))
                    + d;

                  __m256 s = _mm256_loadu_ps(&in_FP[in_subscript]);

                  __m256 w = _mm256_loadu_ps(&filter_FP[filter_subscript]);
                  __m256 w2 = _mm256_loadu_ps(&filter_FP[filter_subscript2]);
                  __m256 w3 = _mm256_loadu_ps(&filter_FP[filter_subscript3]);

                  temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                  temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s, w2));
                  temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w3));

                  // float s = in_FP[in_subscript];
                  // float w = filter_FP[filter_subscript];
                  // temp = temp + s * w;
                }
              }
            }
            unsigned long long int out_subscript = b * (out_yx_depth) +
              y * (out_x_depth) +
              (x << left_shift(Output_depth_dim))
              + m;
            unsigned long long int out_subscript2 = b * (out_yx_depth) +
              y * (out_x_depth) +
              (x << left_shift(Output_depth_dim))
              + (m+1);
            unsigned long long int out_subscript3 = b * (out_yx_depth) +
              y * (out_x_depth) +
              (x << left_shift(Output_depth_dim))
              + (m+2);

            temp = _mm256_hadd_ps(temp, temp);
            temp = _mm256_hadd_ps(temp, temp);
            __m128 tempLo = _mm256_castps256_ps128(temp);
            __m128 tempHi = _mm256_extractf128_ps(temp, 1);
            __m128 sseSum = _mm_add_ps(tempLo, tempHi);

            float sum = _mm_cvtss_f32(sseSum);

            sum += bias;
            out_to_compare_with_FP[out_subscript] = RELU(sum);


            temp2 = _mm256_hadd_ps(temp2, temp2);
            temp2 = _mm256_hadd_ps(temp2, temp2);
            __m128 tempLo2 = _mm256_castps256_ps128(temp2);
            __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
            __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum2 = _mm_cvtss_f32(sseSum2);

            sum2 += bias2;
            out_to_compare_with_FP[out_subscript2] = RELU(sum2);


            temp3 = _mm256_hadd_ps(temp3, temp3);
            temp3 = _mm256_hadd_ps(temp3, temp3);
            __m128 tempLo3 = _mm256_castps256_ps128(temp3);
            __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
            __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

            float sum3 = _mm_cvtss_f32(sseSum3);

            sum3 += bias3;
            out_to_compare_with_FP[out_subscript3] = RELU(sum3);

          }
        }
        for (; m < Output_depth_dim; m++) {
          bias = bias_array_FP[m];

          for (unsigned int x = 0; x < Output_X_dim; x+=4) {	//Output Width
            temp = _mm256_setzero_ps();
            temp2 = _mm256_setzero_ps();
            temp3 = _mm256_setzero_ps();
            temp4 = _mm256_setzero_ps();

            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                  unsigned long long int in_subscript = b * (in_yx_depth)
                    + ((y * Stride_Y_dim + off_y) * in_x_depth)
                    + ((x * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                    + d;
                  unsigned long long int in_subscript2 = b * (in_yx_depth)
                    + ((y * Stride_Y_dim + off_y) * in_x_depth)
                    + (((x+1) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                    + d;
                  unsigned long long int in_subscript3 = b * (in_yx_depth)
                    + ((y * Stride_Y_dim + off_y) * in_x_depth)
                    + (((x+2) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                    + d;
                  unsigned long long int in_subscript4 = b * (in_yx_depth)
                    + ((y * Stride_Y_dim + off_y) * in_x_depth)
                    + (((x+3) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                    + d;

                  unsigned long long int filter_subscript = (m * mask_yx_depth)
                    + (off_y * mask_x_depth)
                    + (off_x << left_shift(Input_depth_dim))
                    + d;

                  __m256 s = _mm256_loadu_ps(&in_FP[in_subscript]);
                  __m256 s2 = _mm256_loadu_ps(&in_FP[in_subscript2]);
                  __m256 s3 = _mm256_loadu_ps(&in_FP[in_subscript3]);
                  __m256 s4 = _mm256_loadu_ps(&in_FP[in_subscript4]);

                  __m256 w = _mm256_loadu_ps(&filter_FP[filter_subscript]);

                  temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                  temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                  temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                  temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s4, w));
                }
              }
            }

            unsigned long long int out_subscript = b * (out_yx_depth) +
              y * (out_x_depth) +
              (x << left_shift(Output_depth_dim))
              + m;
            unsigned long long int out_subscript2 = b * (out_yx_depth) +
              y * (out_x_depth) +
              ((x+1) << left_shift(Output_depth_dim))
              + m;
            unsigned long long int out_subscript3 = b * (out_yx_depth) +
              y * (out_x_depth) +
              ((x+2) << left_shift(Output_depth_dim))
              + m;
            unsigned long long int out_subscript4 = b * (out_yx_depth) +
              y * (out_x_depth) +
              ((x+3) << left_shift(Output_depth_dim))
              + m;

            temp = _mm256_hadd_ps(temp, temp);
            temp = _mm256_hadd_ps(temp, temp);
            __m128 tempLo = _mm256_castps256_ps128(temp);
            __m128 tempHi = _mm256_extractf128_ps(temp, 1);
            __m128 sseSum = _mm_add_ps(tempLo, tempHi);

            float sum = _mm_cvtss_f32(sseSum);

            sum += bias;
            out_to_compare_with_FP[out_subscript] = RELU(sum);


            temp2 = _mm256_hadd_ps(temp2, temp2);
            temp2 = _mm256_hadd_ps(temp2, temp2);
            __m128 tempLo2 = _mm256_castps256_ps128(temp2);
            __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
            __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum2 = _mm_cvtss_f32(sseSum2);

            sum2 += bias;
            out_to_compare_with_FP[out_subscript2] = RELU(sum2);


            temp3 = _mm256_hadd_ps(temp3, temp3);
            temp3 = _mm256_hadd_ps(temp3, temp3);
            __m128 tempLo3 = _mm256_castps256_ps128(temp3);
            __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
            __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

            float sum3 = _mm_cvtss_f32(sseSum3);

            sum3 += bias;
            out_to_compare_with_FP[out_subscript3] = RELU(sum3);


            temp4 = _mm256_hadd_ps(temp4, temp4);
            temp4 = _mm256_hadd_ps(temp4, temp4);
            __m128 tempLo4 = _mm256_castps256_ps128(temp4);
            __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
            __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

            float sum4 = _mm_cvtss_f32(sseSum4);

            sum4 += bias;
            out_to_compare_with_FP[out_subscript4] = RELU(sum4);

          }
        }
      }
    }
  }
  #undef left_shift
  #undef RELU
  printf("\n from optv9 x3 m3 - parallelized %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// same as above, but with collapse() pragma
// collapse(2) ~477 GFLOPS, can only do collapse(2) due to there being multiple m and x loops - collapse(2) increases perf as it unrolls y loop completely increasing parallelism
int optimised_layer_v9_x3m3_omp_collapsed_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  #define left_shift(x) (__builtin_ctz(x))  // inline function effectively using inline assembly to return num of trailing zeros (i.e. 128 returns 7, 2 * 128 == 2 << 7)
  #define RELU(x) ((x) < 0.0f ? 0.0f : (x)) // return x if greater than 0, else return 0
  
  float bias, bias2, bias3;
  __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9;

  unsigned long long int in_yx_depth = Input_Y_dim * Input_X_dim << left_shift(Input_depth_dim);
  unsigned long long int in_x_depth = Input_X_dim  << left_shift(Input_depth_dim);

  unsigned long long int mask_yx_depth = Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim);
  unsigned long long int mask_x_depth = Mask_X_dim << left_shift(Input_depth_dim);

  unsigned long long int out_yx_depth = Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim);
  unsigned long long int out_x_depth = Output_X_dim << left_shift(Output_depth_dim);

  // Calculate loop bounds for unrolled loops
  int m_bound = (Output_depth_dim/ 3) * 3;
  int x_bound = (Output_X_dim/ 3) * 3; 

  unsigned int m, x;
  #pragma omp parallel for private(bias, bias2, bias3, temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9, m, x) \
    shared(in_yx_depth, in_x_depth, mask_yx_depth, mask_x_depth, out_yx_depth, out_x_depth, m_bound, x_bound) default(shared) collapse(2) schedule(static)
  {
    for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (m = 0; m < m_bound; m+=3) { //channels
          bias = bias_array_FP[m];
          bias2 = bias_array_FP[m+1];
          bias3 = bias_array_FP[m+2];

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
            // temp = 0.0f;

            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                  unsigned long long int in_subscript = b * (in_yx_depth)
                    + ((y * Stride_Y_dim + off_y) * in_x_depth)
                    + ((x * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                    + d;
                  unsigned long long int in_subscript2 = b * (in_yx_depth)
                    + ((y * Stride_Y_dim + off_y) * in_x_depth)
                    + (((x+1) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                    + d;
                  unsigned long long int in_subscript3 = b * (in_yx_depth)
                    + ((y * Stride_Y_dim + off_y) * in_x_depth)
                    + (((x+2) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                    + d;


                  unsigned long long int filter_subscript = (m * mask_yx_depth)
                    + (off_y * mask_x_depth)
                    + (off_x << left_shift(Input_depth_dim))
                    + d;
                  unsigned long long int filter_subscript2 = ((m+1) * mask_yx_depth)
                    + (off_y * mask_x_depth)
                    + (off_x << left_shift(Input_depth_dim))
                    + d;
                  unsigned long long int filter_subscript3 = ((m+2) * mask_yx_depth)
                    + (off_y * mask_x_depth)
                    + (off_x << left_shift(Input_depth_dim))
                    + d;

                  __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                  __m256 s2 = _mm256_load_ps(&in_FP[in_subscript2]);
                  __m256 s3 = _mm256_load_ps(&in_FP[in_subscript3]);

                  __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]); 
                  __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);
                  __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript3]);

                  temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                  temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                  temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                  temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s, w2));
                  temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s2, w2));
                  temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s3, w2));
                  temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w3));
                  temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w3));
                  temp9 = _mm256_add_ps(temp9, _mm256_mul_ps(s3, w3));


                  // float s = in_FP[in_subscript];
                  // float w = filter_FP[filter_subscript];
                  // temp = temp + s * w;
                }
              }
            }


            unsigned long long int out_subscript = b * (out_yx_depth) +
              y * (out_x_depth) +
              (x << left_shift(Output_depth_dim))
              + m;
            unsigned long long int out_subscript2 = b * (out_yx_depth) +
              y * (out_x_depth) +
              ((x+1) << left_shift(Output_depth_dim))
              + m;
            unsigned long long int out_subscript3 = b * (out_yx_depth) +
              y * (out_x_depth) +
              ((x+2) << left_shift(Output_depth_dim))
              + m;
            unsigned long long int out_subscript4 = b * (out_yx_depth) +
              y * (out_x_depth) +
              (x << left_shift(Output_depth_dim))
              + (m+1);
            unsigned long long int out_subscript5 = b * (out_yx_depth) +
              y * (out_x_depth) +
              ((x+1) << left_shift(Output_depth_dim))
              + (m+1);
            unsigned long long int out_subscript6 = b * (out_yx_depth) +
              y * (out_x_depth) +
              ((x+2) << left_shift(Output_depth_dim))
              + (m+1);
            unsigned long long int out_subscript7 = b * (out_yx_depth) +
              y * (out_x_depth) +
              (x << left_shift(Output_depth_dim))
              + (m+2);
            unsigned long long int out_subscript8 = b * (out_yx_depth) +
              y * (out_x_depth) +
              ((x+1) << left_shift(Output_depth_dim))
              + (m+2);
            unsigned long long int out_subscript9 = b * (out_yx_depth) +
              y * (out_x_depth) +
              ((x+2) << left_shift(Output_depth_dim))
              + (m+2);



            temp = _mm256_hadd_ps(temp, temp);
            temp = _mm256_hadd_ps(temp, temp);
            __m128 tempLo = _mm256_castps256_ps128(temp);
            __m128 tempHi = _mm256_extractf128_ps(temp, 1);
            __m128 sseSum = _mm_add_ps(tempLo, tempHi);

            float sum = _mm_cvtss_f32(sseSum);

            sum += bias;
            out_to_compare_with_FP[out_subscript] = RELU(sum);


            temp2 = _mm256_hadd_ps(temp2, temp2);
            temp2 = _mm256_hadd_ps(temp2, temp2);
            __m128 tempLo2 = _mm256_castps256_ps128(temp2);
            __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
            __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum2 = _mm_cvtss_f32(sseSum2);

            sum2 += bias;
            out_to_compare_with_FP[out_subscript2] = RELU(sum2);


            temp3 = _mm256_hadd_ps(temp3, temp3);
            temp3 = _mm256_hadd_ps(temp3, temp3);
            __m128 tempLo3 = _mm256_castps256_ps128(temp3);
            __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
            __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

            float sum3 = _mm_cvtss_f32(sseSum3);

            sum3 += bias;
            out_to_compare_with_FP[out_subscript3] = RELU(sum3);


            temp4 = _mm256_hadd_ps(temp4, temp4);
            temp4 = _mm256_hadd_ps(temp4, temp4);
            __m128 tempLo4 = _mm256_castps256_ps128(temp4);
            __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
            __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

            float sum4 = _mm_cvtss_f32(sseSum4);

            sum4 += bias2;
            out_to_compare_with_FP[out_subscript4] = RELU(sum4);


            temp5 = _mm256_hadd_ps(temp5, temp5);
            temp5 = _mm256_hadd_ps(temp5, temp5);
            __m128 tempLo5 = _mm256_castps256_ps128(temp5);
            __m128 tempHi5 = _mm256_extractf128_ps(temp5, 1);
            __m128 sseSum5 = _mm_add_ps(tempLo5, tempHi5);

            float sum5 = _mm_cvtss_f32(sseSum5);

            sum5 += bias2;
            out_to_compare_with_FP[out_subscript5] = RELU(sum5);


            temp6 = _mm256_hadd_ps(temp6, temp6);
            temp6 = _mm256_hadd_ps(temp6, temp6);
            __m128 tempLo6 = _mm256_castps256_ps128(temp6);
            __m128 tempHi6 = _mm256_extractf128_ps(temp6, 1);
            __m128 sseSum6 = _mm_add_ps(tempLo6, tempHi6);

            float sum6 = _mm_cvtss_f32(sseSum6);

            sum6 += bias2;
            out_to_compare_with_FP[out_subscript6] = RELU(sum6);


            temp7 = _mm256_hadd_ps(temp7, temp7);
            temp7 = _mm256_hadd_ps(temp7, temp7);
            __m128 tempLo7 = _mm256_castps256_ps128(temp7);
            __m128 tempHi7 = _mm256_extractf128_ps(temp7, 1);
            __m128 sseSum7 = _mm_add_ps(tempLo7, tempHi7);

            float sum7 = _mm_cvtss_f32(sseSum7);

            sum7 += bias3;
            out_to_compare_with_FP[out_subscript7] = RELU(sum7);


            temp8 = _mm256_hadd_ps(temp8, temp8);
            temp8 = _mm256_hadd_ps(temp8, temp8);
            __m128 tempLo8 = _mm256_castps256_ps128(temp8);
            __m128 tempHi8 = _mm256_extractf128_ps(temp8, 1);
            __m128 sseSum8 = _mm_add_ps(tempLo8, tempHi8);

            float sum8 = _mm_cvtss_f32(sseSum8);

            sum8 += bias3;
            out_to_compare_with_FP[out_subscript8] = RELU(sum8);


            temp9 = _mm256_hadd_ps(temp9, temp9);
            temp9 = _mm256_hadd_ps(temp9, temp9);
            __m128 tempLo9 = _mm256_castps256_ps128(temp9);
            __m128 tempHi9 = _mm256_extractf128_ps(temp9, 1);
            __m128 sseSum9 = _mm_add_ps(tempLo9, tempHi9);

            float sum9 = _mm_cvtss_f32(sseSum9);

            sum9 += bias3;
            out_to_compare_with_FP[out_subscript9] = RELU(sum9);
          }
          // overflow/ fallback x loop
          for (; x < Output_X_dim; x++) {	//Output Width
            temp = _mm256_setzero_ps();
            temp2 = _mm256_setzero_ps();
            temp3 = _mm256_setzero_ps();
            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                  unsigned long long int in_subscript = b * (in_yx_depth)
                    + ((y * Stride_Y_dim + off_y) * in_x_depth)
                    + ((x * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                    + d;

                  unsigned long long int filter_subscript = (m * mask_yx_depth)
                    + (off_y * mask_x_depth)
                    + (off_x << left_shift(Input_depth_dim))
                    + d;
                  unsigned long long int filter_subscript2 = ((m+1) * mask_yx_depth)
                    + (off_y * mask_x_depth)
                    + (off_x << left_shift(Input_depth_dim))
                    + d;
                  unsigned long long int filter_subscript3 = ((m+2) * mask_yx_depth)
                    + (off_y * mask_x_depth)
                    + (off_x << left_shift(Input_depth_dim))
                    + d;

                  __m256 s = _mm256_loadu_ps(&in_FP[in_subscript]);

                  __m256 w = _mm256_loadu_ps(&filter_FP[filter_subscript]);
                  __m256 w2 = _mm256_loadu_ps(&filter_FP[filter_subscript2]);
                  __m256 w3 = _mm256_loadu_ps(&filter_FP[filter_subscript3]);

                  temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                  temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s, w2));
                  temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w3));

                  // float s = in_FP[in_subscript];
                  // float w = filter_FP[filter_subscript];
                  // temp = temp + s * w;
                }
              }
            }
            unsigned long long int out_subscript = b * (out_yx_depth) +
              y * (out_x_depth) +
              (x << left_shift(Output_depth_dim))
              + m;
            unsigned long long int out_subscript2 = b * (out_yx_depth) +
              y * (out_x_depth) +
              (x << left_shift(Output_depth_dim))
              + (m+1);
            unsigned long long int out_subscript3 = b * (out_yx_depth) +
              y * (out_x_depth) +
              (x << left_shift(Output_depth_dim))
              + (m+2);

            temp = _mm256_hadd_ps(temp, temp);
            temp = _mm256_hadd_ps(temp, temp);
            __m128 tempLo = _mm256_castps256_ps128(temp);
            __m128 tempHi = _mm256_extractf128_ps(temp, 1);
            __m128 sseSum = _mm_add_ps(tempLo, tempHi);

            float sum = _mm_cvtss_f32(sseSum);

            sum += bias;
            out_to_compare_with_FP[out_subscript] = RELU(sum);


            temp2 = _mm256_hadd_ps(temp2, temp2);
            temp2 = _mm256_hadd_ps(temp2, temp2);
            __m128 tempLo2 = _mm256_castps256_ps128(temp2);
            __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
            __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum2 = _mm_cvtss_f32(sseSum2);

            sum2 += bias2;
            out_to_compare_with_FP[out_subscript2] = RELU(sum2);


            temp3 = _mm256_hadd_ps(temp3, temp3);
            temp3 = _mm256_hadd_ps(temp3, temp3);
            __m128 tempLo3 = _mm256_castps256_ps128(temp3);
            __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
            __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

            float sum3 = _mm_cvtss_f32(sseSum3);

            sum3 += bias3;
            out_to_compare_with_FP[out_subscript3] = RELU(sum3);

          }
        }
        for (; m < Output_depth_dim; m++) {
          bias = bias_array_FP[m];

          for (unsigned int x = 0; x < Output_X_dim; x+=4) {	//Output Width
            temp = _mm256_setzero_ps();
            temp2 = _mm256_setzero_ps();
            temp3 = _mm256_setzero_ps();
            temp4 = _mm256_setzero_ps();

            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                  unsigned long long int in_subscript = b * (in_yx_depth)
                    + ((y * Stride_Y_dim + off_y) * in_x_depth)
                    + ((x * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                    + d;
                  unsigned long long int in_subscript2 = b * (in_yx_depth)
                    + ((y * Stride_Y_dim + off_y) * in_x_depth)
                    + (((x+1) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                    + d;
                  unsigned long long int in_subscript3 = b * (in_yx_depth)
                    + ((y * Stride_Y_dim + off_y) * in_x_depth)
                    + (((x+2) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                    + d;
                  unsigned long long int in_subscript4 = b * (in_yx_depth)
                    + ((y * Stride_Y_dim + off_y) * in_x_depth)
                    + (((x+3) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                    + d;

                  unsigned long long int filter_subscript = (m * mask_yx_depth)
                    + (off_y * mask_x_depth)
                    + (off_x << left_shift(Input_depth_dim))
                    + d;

                  __m256 s = _mm256_loadu_ps(&in_FP[in_subscript]);
                  __m256 s2 = _mm256_loadu_ps(&in_FP[in_subscript2]);
                  __m256 s3 = _mm256_loadu_ps(&in_FP[in_subscript3]);
                  __m256 s4 = _mm256_loadu_ps(&in_FP[in_subscript4]);

                  __m256 w = _mm256_loadu_ps(&filter_FP[filter_subscript]);

                  temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                  temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                  temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                  temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s4, w));
                }
              }
            }

            unsigned long long int out_subscript = b * (out_yx_depth) +
              y * (out_x_depth) +
              (x << left_shift(Output_depth_dim))
              + m;
            unsigned long long int out_subscript2 = b * (out_yx_depth) +
              y * (out_x_depth) +
              ((x+1) << left_shift(Output_depth_dim))
              + m;
            unsigned long long int out_subscript3 = b * (out_yx_depth) +
              y * (out_x_depth) +
              ((x+2) << left_shift(Output_depth_dim))
              + m;
            unsigned long long int out_subscript4 = b * (out_yx_depth) +
              y * (out_x_depth) +
              ((x+3) << left_shift(Output_depth_dim))
              + m;

            temp = _mm256_hadd_ps(temp, temp);
            temp = _mm256_hadd_ps(temp, temp);
            __m128 tempLo = _mm256_castps256_ps128(temp);
            __m128 tempHi = _mm256_extractf128_ps(temp, 1);
            __m128 sseSum = _mm_add_ps(tempLo, tempHi);

            float sum = _mm_cvtss_f32(sseSum);

            sum += bias;
            out_to_compare_with_FP[out_subscript] = RELU(sum);


            temp2 = _mm256_hadd_ps(temp2, temp2);
            temp2 = _mm256_hadd_ps(temp2, temp2);
            __m128 tempLo2 = _mm256_castps256_ps128(temp2);
            __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
            __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum2 = _mm_cvtss_f32(sseSum2);

            sum2 += bias;
            out_to_compare_with_FP[out_subscript2] = RELU(sum2);


            temp3 = _mm256_hadd_ps(temp3, temp3);
            temp3 = _mm256_hadd_ps(temp3, temp3);
            __m128 tempLo3 = _mm256_castps256_ps128(temp3);
            __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
            __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

            float sum3 = _mm_cvtss_f32(sseSum3);

            sum3 += bias;
            out_to_compare_with_FP[out_subscript3] = RELU(sum3);


            temp4 = _mm256_hadd_ps(temp4, temp4);
            temp4 = _mm256_hadd_ps(temp4, temp4);
            __m128 tempLo4 = _mm256_castps256_ps128(temp4);
            __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
            __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

            float sum4 = _mm_cvtss_f32(sseSum4);

            sum4 += bias;
            out_to_compare_with_FP[out_subscript4] = RELU(sum4);

          }
        }
      }
    }
  }
  #undef left_shift
  #undef RELU
  printf("\n from optv9 x3 m3 - parallelized %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}






// v10 constants - using v9 funcs as base


// adding const keyword to variables defined outside of loop body - optimised_layer_v9_x2m4_omp_collapsed_FP()
// ~456 GFLOPS, investigate perf decrease (**** when same code is in different function, achieves ~474 GFLOPS)
int optimised_layer_v10_x2m4_const_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  #define left_shift(x) (__builtin_ctz(x))  // inline function effectively using inline assembly to return num of trailing zeros (i.e. 128 returns 7, 2 * 128 == 2 << 7)
  #define RELU(x) ((x) < 0.0f ? 0.0f : (x)) // return x if greater than 0, else return 0

  float bias, bias2, bias3, bias4;
  __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

  const unsigned long long int in_yx_depth = Input_Y_dim * Input_X_dim << left_shift(Input_depth_dim);
  const unsigned long long int in_x_depth = Input_X_dim  << left_shift(Input_depth_dim);

  const unsigned long long int mask_yx_depth = Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim);
  const unsigned long long int mask_x_depth = Mask_X_dim << left_shift(Input_depth_dim);

  const unsigned long long int out_yx_depth = Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim);
  const unsigned long long int out_x_depth = Output_X_dim << left_shift(Output_depth_dim);

  #pragma omp parallel for private(bias, bias2, bias3, bias4, temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8) \
    shared(in_yx_depth, in_x_depth, mask_yx_depth, mask_x_depth, out_yx_depth, out_x_depth) default(shared) collapse(2) schedule(static)
  {
    for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (unsigned int m = 0; m < Output_depth_dim; m+=4) { //channels
          bias = bias_array_FP[m];
          bias2 = bias_array_FP[m+1];
          bias3 = bias_array_FP[m+2];
          bias4 = bias_array_FP[m+3];

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

                  unsigned long long int in_subscript = b * (in_yx_depth)
                    + ((y * Stride_Y_dim + off_y) * in_x_depth)
                    + ((x * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                    + d;

                  unsigned long long int filter_subscript = (m * mask_yx_depth)
                    + (off_y * mask_x_depth)
                    + (off_x << left_shift(Input_depth_dim))
                    + d;

                  __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                  __m256 s2 = _mm256_load_ps(&in_FP[in_subscript + Input_depth_dim]);

                  __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]);
                  __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript + (Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))]);
                  __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript + (2 * (Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim)))]);
                  __m256 w4 = _mm256_load_ps(&filter_FP[filter_subscript + (3 * (Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim)))]);

                  temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                  temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                  temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w2));
                  temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s2, w2));
                  temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s, w3));
                  temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s2, w3));
                  temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w4));
                  temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w4));

                }
              }
            }


            unsigned long long int out_subscript = b * (out_yx_depth) 
              + y * (out_x_depth) 
              + (x << left_shift(Output_depth_dim))
              + m;
            unsigned long long int out_subscript2 = out_subscript + Output_depth_dim;



            temp = _mm256_hadd_ps(temp, temp);
            temp = _mm256_hadd_ps(temp, temp);
            __m128 tempLo = _mm256_castps256_ps128(temp);
            __m128 tempHi = _mm256_extractf128_ps(temp, 1);
            __m128 sseSum = _mm_add_ps(tempLo, tempHi);

            float sum = _mm_cvtss_f32(sseSum);

            sum += bias;
            out_to_compare_with_FP[out_subscript] = RELU(sum);


            temp2 = _mm256_hadd_ps(temp2, temp2);
            temp2 = _mm256_hadd_ps(temp2, temp2);
            __m128 tempLo2 = _mm256_castps256_ps128(temp2);
            __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
            __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum2 = _mm_cvtss_f32(sseSum2);

            sum2 += bias;
            out_to_compare_with_FP[out_subscript2] = RELU(sum2);


            temp3 = _mm256_hadd_ps(temp3, temp3);
            temp3 = _mm256_hadd_ps(temp3, temp3);
            tempLo = _mm256_castps256_ps128(temp3);
            tempHi = _mm256_extractf128_ps(temp3, 1);
            sseSum = _mm_add_ps(tempLo, tempHi);

            float sum3 = _mm_cvtss_f32(sseSum);

            sum3 += bias2;
            out_to_compare_with_FP[out_subscript + 1] = RELU(sum3);


            temp4 = _mm256_hadd_ps(temp4, temp4);
            temp4 = _mm256_hadd_ps(temp4, temp4);
            tempLo2 = _mm256_castps256_ps128(temp4);
            tempHi2 = _mm256_extractf128_ps(temp4, 1);
            sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum4 = _mm_cvtss_f32(sseSum2);

            sum4 += bias2;
            out_to_compare_with_FP[out_subscript2 + 1] = RELU(sum4);


            temp5 = _mm256_hadd_ps(temp5, temp5);
            temp5 = _mm256_hadd_ps(temp5, temp5);
            tempLo = _mm256_castps256_ps128(temp5);
            tempHi = _mm256_extractf128_ps(temp5, 1);
            sseSum = _mm_add_ps(tempLo, tempHi);

            float sum5 = _mm_cvtss_f32(sseSum);

            sum5 += bias3;
            out_to_compare_with_FP[out_subscript + 2] = RELU(sum5);


            temp6 = _mm256_hadd_ps(temp6, temp6);
            temp6 = _mm256_hadd_ps(temp6, temp6);
            tempLo2 = _mm256_castps256_ps128(temp6);
            tempHi2 = _mm256_extractf128_ps(temp6, 1);
            sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum6 = _mm_cvtss_f32(sseSum2);

            sum6 += bias3;
            out_to_compare_with_FP[out_subscript2 + 2] = RELU(sum6);


            temp7 = _mm256_hadd_ps(temp7, temp7);
            temp7 = _mm256_hadd_ps(temp7, temp7);
            tempLo = _mm256_castps256_ps128(temp7);
            tempHi = _mm256_extractf128_ps(temp7, 1);
            sseSum = _mm_add_ps(tempLo, tempHi);

            float sum7 = _mm_cvtss_f32(sseSum);

            sum7 += bias4;
            out_to_compare_with_FP[out_subscript + 3] = RELU(sum7);


            temp8 = _mm256_hadd_ps(temp8, temp8);
            temp8 = _mm256_hadd_ps(temp8, temp8);
            tempLo2 = _mm256_castps256_ps128(temp8);
            tempHi2 = _mm256_extractf128_ps(temp8, 1);
            sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum8 = _mm_cvtss_f32(sseSum2);

            sum8 += bias4;
            out_to_compare_with_FP[out_subscript2 + 3] = RELU(sum8);
          }
        }
      }
    }
  }
  #undef left_shift
  #undef RELU
  printf("\n from optv10 x2 m4 - using const keyword %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// adding const keyword to variables defined outside of loop body - optimised_layer_v9_x3m3_omp_collapsed_FP()
// ~486 GFLOPS
int optimised_layer_v10_x3m3_const_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  #define left_shift(x) (__builtin_ctz(x))  // inline function effectively using inline assembly to return num of trailing zeros (i.e. 128 returns 7, 2 * 128 == 2 << 7)
  #define RELU(x) ((x) < 0.0f ? 0.0f : (x)) // return x if greater than 0, else return 0
  
  float bias, bias2, bias3;
  __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9;

  const unsigned long long int in_yx_depth = Input_Y_dim * Input_X_dim << left_shift(Input_depth_dim);
  const unsigned long long int in_x_depth = Input_X_dim  << left_shift(Input_depth_dim);

  const unsigned long long int mask_yx_depth = Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim);
  const unsigned long long int mask_x_depth = Mask_X_dim << left_shift(Input_depth_dim);

  const unsigned long long int out_yx_depth = Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim);
  const unsigned long long int out_x_depth = Output_X_dim << left_shift(Output_depth_dim);

  // Calculate loop bounds for unrolled loops
  const int m_bound = (Output_depth_dim/ 3) * 3;
  const int x_bound = (Output_X_dim/ 3) * 3; 

  unsigned int m, x;
  #pragma omp parallel for private(bias, bias2, bias3, temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9, m, x) \
    shared(in_yx_depth, in_x_depth, mask_yx_depth, mask_x_depth, out_yx_depth, out_x_depth, m_bound, x_bound) default(shared) collapse(2) schedule(static)
  {
    for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (m = 0; m < m_bound; m+=3) { //channels
          bias = bias_array_FP[m];
          bias2 = bias_array_FP[m+1];
          bias3 = bias_array_FP[m+2];

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
            // temp = 0.0f;

            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                  unsigned long long int in_subscript = b * (in_yx_depth)
                    + ((y * Stride_Y_dim + off_y) * in_x_depth)
                    + ((x * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                    + d;
                  unsigned long long int in_subscript2 = b * (in_yx_depth)
                    + ((y * Stride_Y_dim + off_y) * in_x_depth)
                    + (((x+1) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                    + d;
                  unsigned long long int in_subscript3 = b * (in_yx_depth)
                    + ((y * Stride_Y_dim + off_y) * in_x_depth)
                    + (((x+2) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                    + d;


                  unsigned long long int filter_subscript = (m * mask_yx_depth)
                    + (off_y * mask_x_depth)
                    + (off_x << left_shift(Input_depth_dim))
                    + d;
                  unsigned long long int filter_subscript2 = ((m+1) * mask_yx_depth)
                    + (off_y * mask_x_depth)
                    + (off_x << left_shift(Input_depth_dim))
                    + d;
                  unsigned long long int filter_subscript3 = ((m+2) * mask_yx_depth)
                    + (off_y * mask_x_depth)
                    + (off_x << left_shift(Input_depth_dim))
                    + d;

                  __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                  __m256 s2 = _mm256_load_ps(&in_FP[in_subscript2]);
                  __m256 s3 = _mm256_load_ps(&in_FP[in_subscript3]);

                  __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]); 
                  __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);
                  __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript3]);

                  temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                  temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                  temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                  temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s, w2));
                  temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s2, w2));
                  temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s3, w2));
                  temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w3));
                  temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w3));
                  temp9 = _mm256_add_ps(temp9, _mm256_mul_ps(s3, w3));


                  // float s = in_FP[in_subscript];
                  // float w = filter_FP[filter_subscript];
                  // temp = temp + s * w;
                }
              }
            }


            unsigned long long int out_subscript = b * (out_yx_depth) +
              y * (out_x_depth) +
              (x << left_shift(Output_depth_dim))
              + m;
            unsigned long long int out_subscript2 = b * (out_yx_depth) +
              y * (out_x_depth) +
              ((x+1) << left_shift(Output_depth_dim))
              + m;
            unsigned long long int out_subscript3 = b * (out_yx_depth) +
              y * (out_x_depth) +
              ((x+2) << left_shift(Output_depth_dim))
              + m;
            unsigned long long int out_subscript4 = b * (out_yx_depth) +
              y * (out_x_depth) +
              (x << left_shift(Output_depth_dim))
              + (m+1);
            unsigned long long int out_subscript5 = b * (out_yx_depth) +
              y * (out_x_depth) +
              ((x+1) << left_shift(Output_depth_dim))
              + (m+1);
            unsigned long long int out_subscript6 = b * (out_yx_depth) +
              y * (out_x_depth) +
              ((x+2) << left_shift(Output_depth_dim))
              + (m+1);
            unsigned long long int out_subscript7 = b * (out_yx_depth) +
              y * (out_x_depth) +
              (x << left_shift(Output_depth_dim))
              + (m+2);
            unsigned long long int out_subscript8 = b * (out_yx_depth) +
              y * (out_x_depth) +
              ((x+1) << left_shift(Output_depth_dim))
              + (m+2);
            unsigned long long int out_subscript9 = b * (out_yx_depth) +
              y * (out_x_depth) +
              ((x+2) << left_shift(Output_depth_dim))
              + (m+2);



            temp = _mm256_hadd_ps(temp, temp);
            temp = _mm256_hadd_ps(temp, temp);
            __m128 tempLo = _mm256_castps256_ps128(temp);
            __m128 tempHi = _mm256_extractf128_ps(temp, 1);
            __m128 sseSum = _mm_add_ps(tempLo, tempHi);

            float sum = _mm_cvtss_f32(sseSum);

            sum += bias;
            out_to_compare_with_FP[out_subscript] = RELU(sum);


            temp2 = _mm256_hadd_ps(temp2, temp2);
            temp2 = _mm256_hadd_ps(temp2, temp2);
            __m128 tempLo2 = _mm256_castps256_ps128(temp2);
            __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
            __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum2 = _mm_cvtss_f32(sseSum2);

            sum2 += bias;
            out_to_compare_with_FP[out_subscript2] = RELU(sum2);


            temp3 = _mm256_hadd_ps(temp3, temp3);
            temp3 = _mm256_hadd_ps(temp3, temp3);
            __m128 tempLo3 = _mm256_castps256_ps128(temp3);
            __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
            __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

            float sum3 = _mm_cvtss_f32(sseSum3);

            sum3 += bias;
            out_to_compare_with_FP[out_subscript3] = RELU(sum3);


            temp4 = _mm256_hadd_ps(temp4, temp4);
            temp4 = _mm256_hadd_ps(temp4, temp4);
            __m128 tempLo4 = _mm256_castps256_ps128(temp4);
            __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
            __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

            float sum4 = _mm_cvtss_f32(sseSum4);

            sum4 += bias2;
            out_to_compare_with_FP[out_subscript4] = RELU(sum4);


            temp5 = _mm256_hadd_ps(temp5, temp5);
            temp5 = _mm256_hadd_ps(temp5, temp5);
            __m128 tempLo5 = _mm256_castps256_ps128(temp5);
            __m128 tempHi5 = _mm256_extractf128_ps(temp5, 1);
            __m128 sseSum5 = _mm_add_ps(tempLo5, tempHi5);

            float sum5 = _mm_cvtss_f32(sseSum5);

            sum5 += bias2;
            out_to_compare_with_FP[out_subscript5] = RELU(sum5);


            temp6 = _mm256_hadd_ps(temp6, temp6);
            temp6 = _mm256_hadd_ps(temp6, temp6);
            __m128 tempLo6 = _mm256_castps256_ps128(temp6);
            __m128 tempHi6 = _mm256_extractf128_ps(temp6, 1);
            __m128 sseSum6 = _mm_add_ps(tempLo6, tempHi6);

            float sum6 = _mm_cvtss_f32(sseSum6);

            sum6 += bias2;
            out_to_compare_with_FP[out_subscript6] = RELU(sum6);


            temp7 = _mm256_hadd_ps(temp7, temp7);
            temp7 = _mm256_hadd_ps(temp7, temp7);
            __m128 tempLo7 = _mm256_castps256_ps128(temp7);
            __m128 tempHi7 = _mm256_extractf128_ps(temp7, 1);
            __m128 sseSum7 = _mm_add_ps(tempLo7, tempHi7);

            float sum7 = _mm_cvtss_f32(sseSum7);

            sum7 += bias3;
            out_to_compare_with_FP[out_subscript7] = RELU(sum7);


            temp8 = _mm256_hadd_ps(temp8, temp8);
            temp8 = _mm256_hadd_ps(temp8, temp8);
            __m128 tempLo8 = _mm256_castps256_ps128(temp8);
            __m128 tempHi8 = _mm256_extractf128_ps(temp8, 1);
            __m128 sseSum8 = _mm_add_ps(tempLo8, tempHi8);

            float sum8 = _mm_cvtss_f32(sseSum8);

            sum8 += bias3;
            out_to_compare_with_FP[out_subscript8] = RELU(sum8);


            temp9 = _mm256_hadd_ps(temp9, temp9);
            temp9 = _mm256_hadd_ps(temp9, temp9);
            __m128 tempLo9 = _mm256_castps256_ps128(temp9);
            __m128 tempHi9 = _mm256_extractf128_ps(temp9, 1);
            __m128 sseSum9 = _mm_add_ps(tempLo9, tempHi9);

            float sum9 = _mm_cvtss_f32(sseSum9);

            sum9 += bias3;
            out_to_compare_with_FP[out_subscript9] = RELU(sum9);
          }
          // overflow/ fallback x loop
          for (; x < Output_X_dim; x++) {	//Output Width
            temp = _mm256_setzero_ps();
            temp2 = _mm256_setzero_ps();
            temp3 = _mm256_setzero_ps();
            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                  unsigned long long int in_subscript = b * (in_yx_depth)
                    + ((y * Stride_Y_dim + off_y) * in_x_depth)
                    + ((x * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                    + d;

                  unsigned long long int filter_subscript = (m * mask_yx_depth)
                    + (off_y * mask_x_depth)
                    + (off_x << left_shift(Input_depth_dim))
                    + d;
                  unsigned long long int filter_subscript2 = ((m+1) * mask_yx_depth)
                    + (off_y * mask_x_depth)
                    + (off_x << left_shift(Input_depth_dim))
                    + d;
                  unsigned long long int filter_subscript3 = ((m+2) * mask_yx_depth)
                    + (off_y * mask_x_depth)
                    + (off_x << left_shift(Input_depth_dim))
                    + d;

                  __m256 s = _mm256_loadu_ps(&in_FP[in_subscript]);

                  __m256 w = _mm256_loadu_ps(&filter_FP[filter_subscript]);
                  __m256 w2 = _mm256_loadu_ps(&filter_FP[filter_subscript2]);
                  __m256 w3 = _mm256_loadu_ps(&filter_FP[filter_subscript3]);

                  temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                  temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s, w2));
                  temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w3));

                  // float s = in_FP[in_subscript];
                  // float w = filter_FP[filter_subscript];
                  // temp = temp + s * w;
                }
              }
            }
            unsigned long long int out_subscript = b * (out_yx_depth) +
              y * (out_x_depth) +
              (x << left_shift(Output_depth_dim))
              + m;
            unsigned long long int out_subscript2 = b * (out_yx_depth) +
              y * (out_x_depth) +
              (x << left_shift(Output_depth_dim))
              + (m+1);
            unsigned long long int out_subscript3 = b * (out_yx_depth) +
              y * (out_x_depth) +
              (x << left_shift(Output_depth_dim))
              + (m+2);

            temp = _mm256_hadd_ps(temp, temp);
            temp = _mm256_hadd_ps(temp, temp);
            __m128 tempLo = _mm256_castps256_ps128(temp);
            __m128 tempHi = _mm256_extractf128_ps(temp, 1);
            __m128 sseSum = _mm_add_ps(tempLo, tempHi);

            float sum = _mm_cvtss_f32(sseSum);

            sum += bias;
            out_to_compare_with_FP[out_subscript] = RELU(sum);


            temp2 = _mm256_hadd_ps(temp2, temp2);
            temp2 = _mm256_hadd_ps(temp2, temp2);
            __m128 tempLo2 = _mm256_castps256_ps128(temp2);
            __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
            __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum2 = _mm_cvtss_f32(sseSum2);

            sum2 += bias2;
            out_to_compare_with_FP[out_subscript2] = RELU(sum2);


            temp3 = _mm256_hadd_ps(temp3, temp3);
            temp3 = _mm256_hadd_ps(temp3, temp3);
            __m128 tempLo3 = _mm256_castps256_ps128(temp3);
            __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
            __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

            float sum3 = _mm_cvtss_f32(sseSum3);

            sum3 += bias3;
            out_to_compare_with_FP[out_subscript3] = RELU(sum3);

          }
        }
        for (; m < Output_depth_dim; m++) {
          bias = bias_array_FP[m];

          for (unsigned int x = 0; x < Output_X_dim; x+=4) {	//Output Width
            temp = _mm256_setzero_ps();
            temp2 = _mm256_setzero_ps();
            temp3 = _mm256_setzero_ps();
            temp4 = _mm256_setzero_ps();

            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                  unsigned long long int in_subscript = b * (in_yx_depth)
                    + ((y * Stride_Y_dim + off_y) * in_x_depth)
                    + ((x * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                    + d;
                  unsigned long long int in_subscript2 = b * (in_yx_depth)
                    + ((y * Stride_Y_dim + off_y) * in_x_depth)
                    + (((x+1) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                    + d;
                  unsigned long long int in_subscript3 = b * (in_yx_depth)
                    + ((y * Stride_Y_dim + off_y) * in_x_depth)
                    + (((x+2) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                    + d;
                  unsigned long long int in_subscript4 = b * (in_yx_depth)
                    + ((y * Stride_Y_dim + off_y) * in_x_depth)
                    + (((x+3) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                    + d;

                  unsigned long long int filter_subscript = (m * mask_yx_depth)
                    + (off_y * mask_x_depth)
                    + (off_x << left_shift(Input_depth_dim))
                    + d;

                  __m256 s = _mm256_loadu_ps(&in_FP[in_subscript]);
                  __m256 s2 = _mm256_loadu_ps(&in_FP[in_subscript2]);
                  __m256 s3 = _mm256_loadu_ps(&in_FP[in_subscript3]);
                  __m256 s4 = _mm256_loadu_ps(&in_FP[in_subscript4]);

                  __m256 w = _mm256_loadu_ps(&filter_FP[filter_subscript]);

                  temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                  temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                  temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                  temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s4, w));
                }
              }
            }

            unsigned long long int out_subscript = b * (out_yx_depth) +
              y * (out_x_depth) +
              (x << left_shift(Output_depth_dim))
              + m;
            unsigned long long int out_subscript2 = b * (out_yx_depth) +
              y * (out_x_depth) +
              ((x+1) << left_shift(Output_depth_dim))
              + m;
            unsigned long long int out_subscript3 = b * (out_yx_depth) +
              y * (out_x_depth) +
              ((x+2) << left_shift(Output_depth_dim))
              + m;
            unsigned long long int out_subscript4 = b * (out_yx_depth) +
              y * (out_x_depth) +
              ((x+3) << left_shift(Output_depth_dim))
              + m;

            temp = _mm256_hadd_ps(temp, temp);
            temp = _mm256_hadd_ps(temp, temp);
            __m128 tempLo = _mm256_castps256_ps128(temp);
            __m128 tempHi = _mm256_extractf128_ps(temp, 1);
            __m128 sseSum = _mm_add_ps(tempLo, tempHi);

            float sum = _mm_cvtss_f32(sseSum);

            sum += bias;
            out_to_compare_with_FP[out_subscript] = RELU(sum);


            temp2 = _mm256_hadd_ps(temp2, temp2);
            temp2 = _mm256_hadd_ps(temp2, temp2);
            __m128 tempLo2 = _mm256_castps256_ps128(temp2);
            __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
            __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum2 = _mm_cvtss_f32(sseSum2);

            sum2 += bias;
            out_to_compare_with_FP[out_subscript2] = RELU(sum2);


            temp3 = _mm256_hadd_ps(temp3, temp3);
            temp3 = _mm256_hadd_ps(temp3, temp3);
            __m128 tempLo3 = _mm256_castps256_ps128(temp3);
            __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
            __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

            float sum3 = _mm_cvtss_f32(sseSum3);

            sum3 += bias;
            out_to_compare_with_FP[out_subscript3] = RELU(sum3);


            temp4 = _mm256_hadd_ps(temp4, temp4);
            temp4 = _mm256_hadd_ps(temp4, temp4);
            __m128 tempLo4 = _mm256_castps256_ps128(temp4);
            __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
            __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

            float sum4 = _mm_cvtss_f32(sseSum4);

            sum4 += bias;
            out_to_compare_with_FP[out_subscript4] = RELU(sum4);

          }
        }
      }
    }
  }
  #undef left_shift
  #undef RELU
  printf("\n from optv10 x3 m3 - using const keyword %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}






// v11 declare vars inside of loop body - using v9 & v10 funcs as base


// declaring vars inside of loop body if possible - optimised_layer_v9_x2m4_omp_collapsed_FP()
// ~474 GFLOPS
int optimised_layer_v11_x2m4_var_declaration_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  #define left_shift(x) (__builtin_ctz(x))  // inline function effectively using inline assembly to return num of trailing zeros (i.e. 128 returns 7, 2 * 128 == 2 << 7)
  #define RELU(x) ((x) < 0.0f ? 0.0f : (x)) // return x if greater than 0, else return 0

  // float bias, bias2, bias3, bias4;
  // __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

  const unsigned long long int in_yx_depth = Input_Y_dim * Input_X_dim << left_shift(Input_depth_dim);
  const unsigned long long int in_x_depth = Input_X_dim  << left_shift(Input_depth_dim);

  const unsigned long long int mask_yx_depth = Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim);
  const unsigned long long int mask_x_depth = Mask_X_dim << left_shift(Input_depth_dim);

  const unsigned long long int out_yx_depth = Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim);
  const unsigned long long int out_x_depth = Output_X_dim << left_shift(Output_depth_dim);

  #pragma omp parallel for shared(in_yx_depth, in_x_depth, mask_yx_depth, mask_x_depth, out_yx_depth, out_x_depth) \
    default(shared) collapse(2) schedule(static)
  {
    for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (unsigned int m = 0; m < Output_depth_dim; m+=4) { //channels
          const float bias = bias_array_FP[m];
          const float bias2 = bias_array_FP[m+1];
          const float bias3 = bias_array_FP[m+2];
          const float bias4 = bias_array_FP[m+3];

          for (unsigned int x = 0; x < Output_X_dim; x+=2) {	//Output Width
            __m256 temp = _mm256_setzero_ps();
            __m256 temp2 = _mm256_setzero_ps();
            __m256 temp3 = _mm256_setzero_ps();
            __m256 temp4 = _mm256_setzero_ps();
            __m256 temp5 = _mm256_setzero_ps();
            __m256 temp6 = _mm256_setzero_ps();
            __m256 temp7 = _mm256_setzero_ps();
            __m256 temp8 = _mm256_setzero_ps();

            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                  unsigned long long int in_subscript = b * (in_yx_depth)
                    + ((y * Stride_Y_dim + off_y) * in_x_depth)
                    + ((x * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                    + d;

                  unsigned long long int filter_subscript = (m * mask_yx_depth)
                    + (off_y * mask_x_depth)
                    + (off_x << left_shift(Input_depth_dim))
                    + d;

                  __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                  __m256 s2 = _mm256_load_ps(&in_FP[in_subscript + Input_depth_dim]);

                  __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]);
                  __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript + (Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))]);
                  __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript + (2 * (Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim)))]);
                  __m256 w4 = _mm256_load_ps(&filter_FP[filter_subscript + (3 * (Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim)))]);

                  temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                  temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                  temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w2));
                  temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s2, w2));
                  temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s, w3));
                  temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s2, w3));
                  temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w4));
                  temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w4));

                }
              }
            }


            unsigned long long int out_subscript = b * (out_yx_depth) 
              + y * (out_x_depth) 
              + (x << left_shift(Output_depth_dim))
              + m;
            unsigned long long int out_subscript2 = out_subscript + Output_depth_dim;



            temp = _mm256_hadd_ps(temp, temp);
            temp = _mm256_hadd_ps(temp, temp);
            __m128 tempLo = _mm256_castps256_ps128(temp);
            __m128 tempHi = _mm256_extractf128_ps(temp, 1);
            __m128 sseSum = _mm_add_ps(tempLo, tempHi);

            float sum = _mm_cvtss_f32(sseSum);

            sum += bias;
            out_to_compare_with_FP[out_subscript] = RELU(sum);


            temp2 = _mm256_hadd_ps(temp2, temp2);
            temp2 = _mm256_hadd_ps(temp2, temp2);
            __m128 tempLo2 = _mm256_castps256_ps128(temp2);
            __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
            __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum2 = _mm_cvtss_f32(sseSum2);

            sum2 += bias;
            out_to_compare_with_FP[out_subscript2] = RELU(sum2);


            temp3 = _mm256_hadd_ps(temp3, temp3);
            temp3 = _mm256_hadd_ps(temp3, temp3);
            tempLo = _mm256_castps256_ps128(temp3);
            tempHi = _mm256_extractf128_ps(temp3, 1);
            sseSum = _mm_add_ps(tempLo, tempHi);

            float sum3 = _mm_cvtss_f32(sseSum);

            sum3 += bias2;
            out_to_compare_with_FP[out_subscript + 1] = RELU(sum3);


            temp4 = _mm256_hadd_ps(temp4, temp4);
            temp4 = _mm256_hadd_ps(temp4, temp4);
            tempLo2 = _mm256_castps256_ps128(temp4);
            tempHi2 = _mm256_extractf128_ps(temp4, 1);
            sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum4 = _mm_cvtss_f32(sseSum2);

            sum4 += bias2;
            out_to_compare_with_FP[out_subscript2 + 1] = RELU(sum4);


            temp5 = _mm256_hadd_ps(temp5, temp5);
            temp5 = _mm256_hadd_ps(temp5, temp5);
            tempLo = _mm256_castps256_ps128(temp5);
            tempHi = _mm256_extractf128_ps(temp5, 1);
            sseSum = _mm_add_ps(tempLo, tempHi);

            float sum5 = _mm_cvtss_f32(sseSum);

            sum5 += bias3;
            out_to_compare_with_FP[out_subscript + 2] = RELU(sum5);


            temp6 = _mm256_hadd_ps(temp6, temp6);
            temp6 = _mm256_hadd_ps(temp6, temp6);
            tempLo2 = _mm256_castps256_ps128(temp6);
            tempHi2 = _mm256_extractf128_ps(temp6, 1);
            sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum6 = _mm_cvtss_f32(sseSum2);

            sum6 += bias3;
            out_to_compare_with_FP[out_subscript2 + 2] = RELU(sum6);


            temp7 = _mm256_hadd_ps(temp7, temp7);
            temp7 = _mm256_hadd_ps(temp7, temp7);
            tempLo = _mm256_castps256_ps128(temp7);
            tempHi = _mm256_extractf128_ps(temp7, 1);
            sseSum = _mm_add_ps(tempLo, tempHi);

            float sum7 = _mm_cvtss_f32(sseSum);

            sum7 += bias4;
            out_to_compare_with_FP[out_subscript + 3] = RELU(sum7);


            temp8 = _mm256_hadd_ps(temp8, temp8);
            temp8 = _mm256_hadd_ps(temp8, temp8);
            tempLo2 = _mm256_castps256_ps128(temp8);
            tempHi2 = _mm256_extractf128_ps(temp8, 1);
            sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum8 = _mm_cvtss_f32(sseSum2);

            sum8 += bias4;
            out_to_compare_with_FP[out_subscript2 + 3] = RELU(sum8);
          }
        }
      }
    }
  }
  #undef left_shift
  #undef RELU
  printf("\n from optv11 x2 m4 - declaring var in loop %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// declaring vars inside of loop body if possible - optimised_layer_v10_x3m3_const_FP()
// ~489 GFLOPS
int optimised_layer_v11_x3m3_var_declaration_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  #define left_shift(x) (__builtin_ctz(x))  // inline function effectively using inline assembly to return num of trailing zeros (i.e. 128 returns 7, 2 * 128 == 2 << 7)
  #define RELU(x) ((x) < 0.0f ? 0.0f : (x)) // return x if greater than 0, else return 0
  
  // float bias, bias2, bias3;
  // __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9;

  const unsigned long long int in_yx_depth = Input_Y_dim * Input_X_dim << left_shift(Input_depth_dim);
  const unsigned long long int in_x_depth = Input_X_dim  << left_shift(Input_depth_dim);

  const unsigned long long int mask_yx_depth = Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim);
  const unsigned long long int mask_x_depth = Mask_X_dim << left_shift(Input_depth_dim);

  const unsigned long long int out_yx_depth = Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim);
  const unsigned long long int out_x_depth = Output_X_dim << left_shift(Output_depth_dim);

  // Calculate loop bounds for unrolled loops
  const int m_bound = (Output_depth_dim/ 3) * 3;
  const int x_bound = (Output_X_dim/ 3) * 3; 

  unsigned int m, x;
  #pragma omp parallel for private(m, x) shared(in_yx_depth, in_x_depth, mask_yx_depth, mask_x_depth, out_yx_depth, out_x_depth, m_bound, x_bound) \
    default(shared) collapse(2) schedule(static)
  {
    for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (m = 0; m < m_bound; m+=3) { //channels
          const float bias = bias_array_FP[m];
          const float bias2 = bias_array_FP[m+1];
          const float bias3 = bias_array_FP[m+2];

          for (x = 0; x < x_bound; x+=3) {	//Output Width
            __m256 temp = _mm256_setzero_ps();
            __m256 temp2 = _mm256_setzero_ps();
            __m256 temp3 = _mm256_setzero_ps();
            __m256 temp4 = _mm256_setzero_ps();
            __m256 temp5 = _mm256_setzero_ps();
            __m256 temp6 = _mm256_setzero_ps();
            __m256 temp7 = _mm256_setzero_ps();
            __m256 temp8 = _mm256_setzero_ps();
            __m256 temp9 = _mm256_setzero_ps();
            // temp = 0.0f;

            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                  unsigned long long int in_subscript = b * (in_yx_depth)
                    + ((y * Stride_Y_dim + off_y) * in_x_depth)
                    + ((x * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                    + d;
                  unsigned long long int in_subscript2 = b * (in_yx_depth)
                    + ((y * Stride_Y_dim + off_y) * in_x_depth)
                    + (((x+1) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                    + d;
                  unsigned long long int in_subscript3 = b * (in_yx_depth)
                    + ((y * Stride_Y_dim + off_y) * in_x_depth)
                    + (((x+2) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                    + d;


                  unsigned long long int filter_subscript = (m * mask_yx_depth)
                    + (off_y * mask_x_depth)
                    + (off_x << left_shift(Input_depth_dim))
                    + d;
                  unsigned long long int filter_subscript2 = ((m+1) * mask_yx_depth)
                    + (off_y * mask_x_depth)
                    + (off_x << left_shift(Input_depth_dim))
                    + d;
                  unsigned long long int filter_subscript3 = ((m+2) * mask_yx_depth)
                    + (off_y * mask_x_depth)
                    + (off_x << left_shift(Input_depth_dim))
                    + d;

                  __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                  __m256 s2 = _mm256_load_ps(&in_FP[in_subscript2]);
                  __m256 s3 = _mm256_load_ps(&in_FP[in_subscript3]);

                  __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]); 
                  __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);
                  __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript3]);

                  temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                  temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                  temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                  temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s, w2));
                  temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s2, w2));
                  temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s3, w2));
                  temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w3));
                  temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w3));
                  temp9 = _mm256_add_ps(temp9, _mm256_mul_ps(s3, w3));


                  // float s = in_FP[in_subscript];
                  // float w = filter_FP[filter_subscript];
                  // temp = temp + s * w;
                }
              }
            }


            unsigned long long int out_subscript = b * (out_yx_depth) +
              y * (out_x_depth) +
              (x << left_shift(Output_depth_dim))
              + m;
            unsigned long long int out_subscript2 = b * (out_yx_depth) +
              y * (out_x_depth) +
              ((x+1) << left_shift(Output_depth_dim))
              + m;
            unsigned long long int out_subscript3 = b * (out_yx_depth) +
              y * (out_x_depth) +
              ((x+2) << left_shift(Output_depth_dim))
              + m;
            unsigned long long int out_subscript4 = b * (out_yx_depth) +
              y * (out_x_depth) +
              (x << left_shift(Output_depth_dim))
              + (m+1);
            unsigned long long int out_subscript5 = b * (out_yx_depth) +
              y * (out_x_depth) +
              ((x+1) << left_shift(Output_depth_dim))
              + (m+1);
            unsigned long long int out_subscript6 = b * (out_yx_depth) +
              y * (out_x_depth) +
              ((x+2) << left_shift(Output_depth_dim))
              + (m+1);
            unsigned long long int out_subscript7 = b * (out_yx_depth) +
              y * (out_x_depth) +
              (x << left_shift(Output_depth_dim))
              + (m+2);
            unsigned long long int out_subscript8 = b * (out_yx_depth) +
              y * (out_x_depth) +
              ((x+1) << left_shift(Output_depth_dim))
              + (m+2);
            unsigned long long int out_subscript9 = b * (out_yx_depth) +
              y * (out_x_depth) +
              ((x+2) << left_shift(Output_depth_dim))
              + (m+2);



            temp = _mm256_hadd_ps(temp, temp);
            temp = _mm256_hadd_ps(temp, temp);
            __m128 tempLo = _mm256_castps256_ps128(temp);
            __m128 tempHi = _mm256_extractf128_ps(temp, 1);
            __m128 sseSum = _mm_add_ps(tempLo, tempHi);

            float sum = _mm_cvtss_f32(sseSum);

            sum += bias;
            out_to_compare_with_FP[out_subscript] = RELU(sum);


            temp2 = _mm256_hadd_ps(temp2, temp2);
            temp2 = _mm256_hadd_ps(temp2, temp2);
            __m128 tempLo2 = _mm256_castps256_ps128(temp2);
            __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
            __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum2 = _mm_cvtss_f32(sseSum2);

            sum2 += bias;
            out_to_compare_with_FP[out_subscript2] = RELU(sum2);


            temp3 = _mm256_hadd_ps(temp3, temp3);
            temp3 = _mm256_hadd_ps(temp3, temp3);
            __m128 tempLo3 = _mm256_castps256_ps128(temp3);
            __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
            __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

            float sum3 = _mm_cvtss_f32(sseSum3);

            sum3 += bias;
            out_to_compare_with_FP[out_subscript3] = RELU(sum3);


            temp4 = _mm256_hadd_ps(temp4, temp4);
            temp4 = _mm256_hadd_ps(temp4, temp4);
            __m128 tempLo4 = _mm256_castps256_ps128(temp4);
            __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
            __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

            float sum4 = _mm_cvtss_f32(sseSum4);

            sum4 += bias2;
            out_to_compare_with_FP[out_subscript4] = RELU(sum4);


            temp5 = _mm256_hadd_ps(temp5, temp5);
            temp5 = _mm256_hadd_ps(temp5, temp5);
            __m128 tempLo5 = _mm256_castps256_ps128(temp5);
            __m128 tempHi5 = _mm256_extractf128_ps(temp5, 1);
            __m128 sseSum5 = _mm_add_ps(tempLo5, tempHi5);

            float sum5 = _mm_cvtss_f32(sseSum5);

            sum5 += bias2;
            out_to_compare_with_FP[out_subscript5] = RELU(sum5);


            temp6 = _mm256_hadd_ps(temp6, temp6);
            temp6 = _mm256_hadd_ps(temp6, temp6);
            __m128 tempLo6 = _mm256_castps256_ps128(temp6);
            __m128 tempHi6 = _mm256_extractf128_ps(temp6, 1);
            __m128 sseSum6 = _mm_add_ps(tempLo6, tempHi6);

            float sum6 = _mm_cvtss_f32(sseSum6);

            sum6 += bias2;
            out_to_compare_with_FP[out_subscript6] = RELU(sum6);


            temp7 = _mm256_hadd_ps(temp7, temp7);
            temp7 = _mm256_hadd_ps(temp7, temp7);
            __m128 tempLo7 = _mm256_castps256_ps128(temp7);
            __m128 tempHi7 = _mm256_extractf128_ps(temp7, 1);
            __m128 sseSum7 = _mm_add_ps(tempLo7, tempHi7);

            float sum7 = _mm_cvtss_f32(sseSum7);

            sum7 += bias3;
            out_to_compare_with_FP[out_subscript7] = RELU(sum7);


            temp8 = _mm256_hadd_ps(temp8, temp8);
            temp8 = _mm256_hadd_ps(temp8, temp8);
            __m128 tempLo8 = _mm256_castps256_ps128(temp8);
            __m128 tempHi8 = _mm256_extractf128_ps(temp8, 1);
            __m128 sseSum8 = _mm_add_ps(tempLo8, tempHi8);

            float sum8 = _mm_cvtss_f32(sseSum8);

            sum8 += bias3;
            out_to_compare_with_FP[out_subscript8] = RELU(sum8);


            temp9 = _mm256_hadd_ps(temp9, temp9);
            temp9 = _mm256_hadd_ps(temp9, temp9);
            __m128 tempLo9 = _mm256_castps256_ps128(temp9);
            __m128 tempHi9 = _mm256_extractf128_ps(temp9, 1);
            __m128 sseSum9 = _mm_add_ps(tempLo9, tempHi9);

            float sum9 = _mm_cvtss_f32(sseSum9);

            sum9 += bias3;
            out_to_compare_with_FP[out_subscript9] = RELU(sum9);
          }
          // overflow/ fallback x loop
          for (; x < Output_X_dim; x++) {	//Output Width
            __m256 temp = _mm256_setzero_ps();
            __m256 temp2 = _mm256_setzero_ps();
            __m256 temp3 = _mm256_setzero_ps();
            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                  unsigned long long int in_subscript = b * (in_yx_depth)
                    + ((y * Stride_Y_dim + off_y) * in_x_depth)
                    + ((x * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                    + d;

                  unsigned long long int filter_subscript = (m * mask_yx_depth)
                    + (off_y * mask_x_depth)
                    + (off_x << left_shift(Input_depth_dim))
                    + d;
                  unsigned long long int filter_subscript2 = ((m+1) * mask_yx_depth)
                    + (off_y * mask_x_depth)
                    + (off_x << left_shift(Input_depth_dim))
                    + d;
                  unsigned long long int filter_subscript3 = ((m+2) * mask_yx_depth)
                    + (off_y * mask_x_depth)
                    + (off_x << left_shift(Input_depth_dim))
                    + d;

                  __m256 s = _mm256_loadu_ps(&in_FP[in_subscript]);

                  __m256 w = _mm256_loadu_ps(&filter_FP[filter_subscript]);
                  __m256 w2 = _mm256_loadu_ps(&filter_FP[filter_subscript2]);
                  __m256 w3 = _mm256_loadu_ps(&filter_FP[filter_subscript3]);

                  temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                  temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s, w2));
                  temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w3));

                  // float s = in_FP[in_subscript];
                  // float w = filter_FP[filter_subscript];
                  // temp = temp + s * w;
                }
              }
            }
            unsigned long long int out_subscript = b * (out_yx_depth) +
              y * (out_x_depth) +
              (x << left_shift(Output_depth_dim))
              + m;
            unsigned long long int out_subscript2 = b * (out_yx_depth) +
              y * (out_x_depth) +
              (x << left_shift(Output_depth_dim))
              + (m+1);
            unsigned long long int out_subscript3 = b * (out_yx_depth) +
              y * (out_x_depth) +
              (x << left_shift(Output_depth_dim))
              + (m+2);

            temp = _mm256_hadd_ps(temp, temp);
            temp = _mm256_hadd_ps(temp, temp);
            __m128 tempLo = _mm256_castps256_ps128(temp);
            __m128 tempHi = _mm256_extractf128_ps(temp, 1);
            __m128 sseSum = _mm_add_ps(tempLo, tempHi);

            float sum = _mm_cvtss_f32(sseSum);

            sum += bias;
            out_to_compare_with_FP[out_subscript] = RELU(sum);


            temp2 = _mm256_hadd_ps(temp2, temp2);
            temp2 = _mm256_hadd_ps(temp2, temp2);
            __m128 tempLo2 = _mm256_castps256_ps128(temp2);
            __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
            __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum2 = _mm_cvtss_f32(sseSum2);

            sum2 += bias2;
            out_to_compare_with_FP[out_subscript2] = RELU(sum2);


            temp3 = _mm256_hadd_ps(temp3, temp3);
            temp3 = _mm256_hadd_ps(temp3, temp3);
            __m128 tempLo3 = _mm256_castps256_ps128(temp3);
            __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
            __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

            float sum3 = _mm_cvtss_f32(sseSum3);

            sum3 += bias3;
            out_to_compare_with_FP[out_subscript3] = RELU(sum3);

          }
        }
        for (; m < Output_depth_dim; m++) {
          const float bias = bias_array_FP[m];

          for (unsigned int x = 0; x < Output_X_dim; x+=4) {	//Output Width
            __m256 temp = _mm256_setzero_ps();
            __m256 temp2 = _mm256_setzero_ps();
            __m256 temp3 = _mm256_setzero_ps();
            __m256 temp4 = _mm256_setzero_ps();

            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                  unsigned long long int in_subscript = b * (in_yx_depth)
                    + ((y * Stride_Y_dim + off_y) * in_x_depth)
                    + ((x * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                    + d;
                  unsigned long long int in_subscript2 = b * (in_yx_depth)
                    + ((y * Stride_Y_dim + off_y) * in_x_depth)
                    + (((x+1) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                    + d;
                  unsigned long long int in_subscript3 = b * (in_yx_depth)
                    + ((y * Stride_Y_dim + off_y) * in_x_depth)
                    + (((x+2) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                    + d;
                  unsigned long long int in_subscript4 = b * (in_yx_depth)
                    + ((y * Stride_Y_dim + off_y) * in_x_depth)
                    + (((x+3) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                    + d;

                  unsigned long long int filter_subscript = (m * mask_yx_depth)
                    + (off_y * mask_x_depth)
                    + (off_x << left_shift(Input_depth_dim))
                    + d;

                  __m256 s = _mm256_loadu_ps(&in_FP[in_subscript]);
                  __m256 s2 = _mm256_loadu_ps(&in_FP[in_subscript2]);
                  __m256 s3 = _mm256_loadu_ps(&in_FP[in_subscript3]);
                  __m256 s4 = _mm256_loadu_ps(&in_FP[in_subscript4]);

                  __m256 w = _mm256_loadu_ps(&filter_FP[filter_subscript]);

                  temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                  temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                  temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                  temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s4, w));
                }
              }
            }

            unsigned long long int out_subscript = b * (out_yx_depth) +
              y * (out_x_depth) +
              (x << left_shift(Output_depth_dim))
              + m;
            unsigned long long int out_subscript2 = b * (out_yx_depth) +
              y * (out_x_depth) +
              ((x+1) << left_shift(Output_depth_dim))
              + m;
            unsigned long long int out_subscript3 = b * (out_yx_depth) +
              y * (out_x_depth) +
              ((x+2) << left_shift(Output_depth_dim))
              + m;
            unsigned long long int out_subscript4 = b * (out_yx_depth) +
              y * (out_x_depth) +
              ((x+3) << left_shift(Output_depth_dim))
              + m;

            temp = _mm256_hadd_ps(temp, temp);
            temp = _mm256_hadd_ps(temp, temp);
            __m128 tempLo = _mm256_castps256_ps128(temp);
            __m128 tempHi = _mm256_extractf128_ps(temp, 1);
            __m128 sseSum = _mm_add_ps(tempLo, tempHi);

            float sum = _mm_cvtss_f32(sseSum);

            sum += bias;
            out_to_compare_with_FP[out_subscript] = RELU(sum);


            temp2 = _mm256_hadd_ps(temp2, temp2);
            temp2 = _mm256_hadd_ps(temp2, temp2);
            __m128 tempLo2 = _mm256_castps256_ps128(temp2);
            __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
            __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum2 = _mm_cvtss_f32(sseSum2);

            sum2 += bias;
            out_to_compare_with_FP[out_subscript2] = RELU(sum2);


            temp3 = _mm256_hadd_ps(temp3, temp3);
            temp3 = _mm256_hadd_ps(temp3, temp3);
            __m128 tempLo3 = _mm256_castps256_ps128(temp3);
            __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
            __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

            float sum3 = _mm_cvtss_f32(sseSum3);

            sum3 += bias;
            out_to_compare_with_FP[out_subscript3] = RELU(sum3);


            temp4 = _mm256_hadd_ps(temp4, temp4);
            temp4 = _mm256_hadd_ps(temp4, temp4);
            __m128 tempLo4 = _mm256_castps256_ps128(temp4);
            __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
            __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

            float sum4 = _mm_cvtss_f32(sseSum4);

            sum4 += bias;
            out_to_compare_with_FP[out_subscript4] = RELU(sum4);

          }
        }
      }
    }
  }
  #undef left_shift
  #undef RELU
  printf("\n from optv11 x3 m3 - declaring var in loop %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}






// v12 using fmadd intrinsics - using v11 funcs as base


// replacing add/ mul intrinsics with fmadd - optimised_layer_v11_x2m4_var_declaration_FP()
// ~507 GFLOPS
int optimised_layer_v12_x2m4_fmadd_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  #define left_shift(x) (__builtin_ctz(x))  // inline function effectively using inline assembly to return num of trailing zeros (i.e. 128 returns 7, 2 * 128 == 2 << 7)
  #define RELU(x) ((x) < 0.0f ? 0.0f : (x)) // return x if greater than 0, else return 0

  // float bias, bias2, bias3, bias4;
  // __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

  const unsigned long long int in_yx_depth = Input_Y_dim * Input_X_dim << left_shift(Input_depth_dim);
  const unsigned long long int in_x_depth = Input_X_dim  << left_shift(Input_depth_dim);

  const unsigned long long int mask_yx_depth = Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim);
  const unsigned long long int mask_x_depth = Mask_X_dim << left_shift(Input_depth_dim);

  const unsigned long long int out_yx_depth = Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim);
  const unsigned long long int out_x_depth = Output_X_dim << left_shift(Output_depth_dim);

  #pragma omp parallel for shared(in_yx_depth, in_x_depth, mask_yx_depth, mask_x_depth, out_yx_depth, out_x_depth) \
    default(shared) collapse(2) schedule(static)
  {
    for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (unsigned int m = 0; m < Output_depth_dim; m+=4) { //channels
          const float bias = bias_array_FP[m];
          const float bias2 = bias_array_FP[m+1];
          const float bias3 = bias_array_FP[m+2];
          const float bias4 = bias_array_FP[m+3];

          for (unsigned int x = 0; x < Output_X_dim; x+=2) {	//Output Width
            __m256 temp = _mm256_setzero_ps();
            __m256 temp2 = _mm256_setzero_ps();
            __m256 temp3 = _mm256_setzero_ps();
            __m256 temp4 = _mm256_setzero_ps();
            __m256 temp5 = _mm256_setzero_ps();
            __m256 temp6 = _mm256_setzero_ps();
            __m256 temp7 = _mm256_setzero_ps();
            __m256 temp8 = _mm256_setzero_ps();

            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                  unsigned long long int in_subscript = b * (in_yx_depth)
                    + ((y * Stride_Y_dim + off_y) * in_x_depth)
                    + ((x * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                    + d;

                  unsigned long long int filter_subscript = (m * mask_yx_depth)
                    + (off_y * mask_x_depth)
                    + (off_x << left_shift(Input_depth_dim))
                    + d;

                  __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                  __m256 s2 = _mm256_load_ps(&in_FP[in_subscript + Input_depth_dim]);

                  __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]);
                  __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript + (Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))]);
                  __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript + (2 * (Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim)))]);
                  __m256 w4 = _mm256_load_ps(&filter_FP[filter_subscript + (3 * (Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim)))]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s, w2, temp3);
                  temp4 = _mm256_fmadd_ps(s2, w2, temp4);
                  temp5 = _mm256_fmadd_ps(s, w3, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w3, temp6);
                  temp7 = _mm256_fmadd_ps(s, w4, temp7);
                  temp8 = _mm256_fmadd_ps(s2, w4, temp8);

                  // temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                  // temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                  // temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w2));
                  // temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s2, w2));
                  // temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s, w3));
                  // temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s2, w3));
                  // temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w4));
                  // temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w4));





                  // in_subscript += 8;

                  // filter_subscript += 8;

                  //  s = _mm256_load_ps(&in_FP[in_subscript]);
                  //  s2 = _mm256_load_ps(&in_FP[in_subscript + Input_depth_dim]);

                  //  w = _mm256_load_ps(&filter_FP[filter_subscript]);
                  //  w2 = _mm256_load_ps(&filter_FP[filter_subscript + (Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))]);
                  //  w3 = _mm256_load_ps(&filter_FP[filter_subscript + (2 * (Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim)))]);
                  //  w4 = _mm256_load_ps(&filter_FP[filter_subscript + (3 * (Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim)))]);


                  // temp = _mm256_fmadd_ps(s, w, temp);
                  // temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  // temp3 = _mm256_fmadd_ps(s, w2, temp3);
                  // temp4 = _mm256_fmadd_ps(s2, w2, temp4);
                  // temp5 = _mm256_fmadd_ps(s, w3, temp5);
                  // temp6 = _mm256_fmadd_ps(s2, w3, temp6);
                  // temp7 = _mm256_fmadd_ps(s, w4, temp7);
                  // temp8 = _mm256_fmadd_ps(s2, w4, temp8);


                  // in_subscript += 8;

                  // filter_subscript += 8;

                  //  s = _mm256_load_ps(&in_FP[in_subscript]);
                  //  s2 = _mm256_load_ps(&in_FP[in_subscript + Input_depth_dim]);

                  //  w = _mm256_load_ps(&filter_FP[filter_subscript]);
                  //  w2 = _mm256_load_ps(&filter_FP[filter_subscript + (Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))]);
                  //  w3 = _mm256_load_ps(&filter_FP[filter_subscript + (2 * (Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim)))]);
                  //  w4 = _mm256_load_ps(&filter_FP[filter_subscript + (3 * (Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim)))]);


                  // temp = _mm256_fmadd_ps(s, w, temp);
                  // temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  // temp3 = _mm256_fmadd_ps(s, w2, temp3);
                  // temp4 = _mm256_fmadd_ps(s2, w2, temp4);
                  // temp5 = _mm256_fmadd_ps(s, w3, temp5);
                  // temp6 = _mm256_fmadd_ps(s2, w3, temp6);
                  // temp7 = _mm256_fmadd_ps(s, w4, temp7);
                  // temp8 = _mm256_fmadd_ps(s2, w4, temp8);


                  // in_subscript += 8;

                  // filter_subscript += 8;

                  //  s = _mm256_load_ps(&in_FP[in_subscript]);
                  //  s2 = _mm256_load_ps(&in_FP[in_subscript + Input_depth_dim]);

                  //  w = _mm256_load_ps(&filter_FP[filter_subscript]);
                  //  w2 = _mm256_load_ps(&filter_FP[filter_subscript + (Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))]);
                  //  w3 = _mm256_load_ps(&filter_FP[filter_subscript + (2 * (Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim)))]);
                  //  w4 = _mm256_load_ps(&filter_FP[filter_subscript + (3 * (Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim)))]);


                  // temp = _mm256_fmadd_ps(s, w, temp);
                  // temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  // temp3 = _mm256_fmadd_ps(s, w2, temp3);
                  // temp4 = _mm256_fmadd_ps(s2, w2, temp4);
                  // temp5 = _mm256_fmadd_ps(s, w3, temp5);
                  // temp6 = _mm256_fmadd_ps(s2, w3, temp6);
                  // temp7 = _mm256_fmadd_ps(s, w4, temp7);
                  // temp8 = _mm256_fmadd_ps(s2, w4, temp8);

                }
              }
            }


            unsigned long long int out_subscript = b * (out_yx_depth) 
              + y * (out_x_depth) 
              + (x << left_shift(Output_depth_dim))
              + m;
            unsigned long long int out_subscript2 = out_subscript + Output_depth_dim;



            temp = _mm256_hadd_ps(temp, temp);
            temp = _mm256_hadd_ps(temp, temp);
            __m128 tempLo = _mm256_castps256_ps128(temp);
            __m128 tempHi = _mm256_extractf128_ps(temp, 1);
            __m128 sseSum = _mm_add_ps(tempLo, tempHi);

            float sum = _mm_cvtss_f32(sseSum);

            sum += bias;
            out_to_compare_with_FP[out_subscript] = RELU(sum);


            temp2 = _mm256_hadd_ps(temp2, temp2);
            temp2 = _mm256_hadd_ps(temp2, temp2);
            __m128 tempLo2 = _mm256_castps256_ps128(temp2);
            __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
            __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum2 = _mm_cvtss_f32(sseSum2);

            sum2 += bias;
            out_to_compare_with_FP[out_subscript2] = RELU(sum2);


            temp3 = _mm256_hadd_ps(temp3, temp3);
            temp3 = _mm256_hadd_ps(temp3, temp3);
            tempLo = _mm256_castps256_ps128(temp3);
            tempHi = _mm256_extractf128_ps(temp3, 1);
            sseSum = _mm_add_ps(tempLo, tempHi);

            float sum3 = _mm_cvtss_f32(sseSum);

            sum3 += bias2;
            out_to_compare_with_FP[out_subscript + 1] = RELU(sum3);


            temp4 = _mm256_hadd_ps(temp4, temp4);
            temp4 = _mm256_hadd_ps(temp4, temp4);
            tempLo2 = _mm256_castps256_ps128(temp4);
            tempHi2 = _mm256_extractf128_ps(temp4, 1);
            sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum4 = _mm_cvtss_f32(sseSum2);

            sum4 += bias2;
            out_to_compare_with_FP[out_subscript2 + 1] = RELU(sum4);


            temp5 = _mm256_hadd_ps(temp5, temp5);
            temp5 = _mm256_hadd_ps(temp5, temp5);
            tempLo = _mm256_castps256_ps128(temp5);
            tempHi = _mm256_extractf128_ps(temp5, 1);
            sseSum = _mm_add_ps(tempLo, tempHi);

            float sum5 = _mm_cvtss_f32(sseSum);

            sum5 += bias3;
            out_to_compare_with_FP[out_subscript + 2] = RELU(sum5);


            temp6 = _mm256_hadd_ps(temp6, temp6);
            temp6 = _mm256_hadd_ps(temp6, temp6);
            tempLo2 = _mm256_castps256_ps128(temp6);
            tempHi2 = _mm256_extractf128_ps(temp6, 1);
            sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum6 = _mm_cvtss_f32(sseSum2);

            sum6 += bias3;
            out_to_compare_with_FP[out_subscript2 + 2] = RELU(sum6);


            temp7 = _mm256_hadd_ps(temp7, temp7);
            temp7 = _mm256_hadd_ps(temp7, temp7);
            tempLo = _mm256_castps256_ps128(temp7);
            tempHi = _mm256_extractf128_ps(temp7, 1);
            sseSum = _mm_add_ps(tempLo, tempHi);

            float sum7 = _mm_cvtss_f32(sseSum);

            sum7 += bias4;
            out_to_compare_with_FP[out_subscript + 3] = RELU(sum7);


            temp8 = _mm256_hadd_ps(temp8, temp8);
            temp8 = _mm256_hadd_ps(temp8, temp8);
            tempLo2 = _mm256_castps256_ps128(temp8);
            tempHi2 = _mm256_extractf128_ps(temp8, 1);
            sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum8 = _mm_cvtss_f32(sseSum2);

            sum8 += bias4;
            out_to_compare_with_FP[out_subscript2 + 3] = RELU(sum8);
          }
        }
      }
    }
  }
  #undef left_shift
  #undef RELU
  printf("\n from optv12 x2 m4 - fmadd instructions %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// replacing add/ mul intrinsics with fmadd - optimised_layer_v11_x3m3_var_declaration_FP()
// ~548 GFLOPS
int optimised_layer_v12_x3m3_fmadd_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  #define left_shift(x) (__builtin_ctz(x))  // inline function effectively using inline assembly to return num of trailing zeros (i.e. 128 returns 7, 2 * 128 == 2 << 7)
  #define RELU(x) ((x) < 0.0f ? 0.0f : (x)) // return x if greater than 0, else return 0
  
  // float bias, bias2, bias3;
  // __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9;

  const unsigned long long int in_yx_depth = Input_Y_dim * Input_X_dim << left_shift(Input_depth_dim);
  const unsigned long long int in_x_depth = Input_X_dim  << left_shift(Input_depth_dim);

  const unsigned long long int mask_yx_depth = Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim);
  const unsigned long long int mask_x_depth = Mask_X_dim << left_shift(Input_depth_dim);

  const unsigned long long int out_yx_depth = Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim);
  const unsigned long long int out_x_depth = Output_X_dim << left_shift(Output_depth_dim);

  // Calculate loop bounds for unrolled loops
  const int m_bound = (Output_depth_dim/ 3) * 3;
  const int x_bound = (Output_X_dim/ 3) * 3; 

  unsigned int m, x;
  #pragma omp parallel for private(m, x) shared(in_yx_depth, in_x_depth, mask_yx_depth, mask_x_depth, out_yx_depth, out_x_depth, m_bound, x_bound) \
    default(shared) collapse(2) schedule(static)
  {
    for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (m = 0; m < m_bound; m+=3) { //channels
          const float bias = bias_array_FP[m];
          const float bias2 = bias_array_FP[m+1];
          const float bias3 = bias_array_FP[m+2];

          for (x = 0; x < x_bound; x+=3) {	//Output Width
            __m256 temp = _mm256_setzero_ps();
            __m256 temp2 = _mm256_setzero_ps();
            __m256 temp3 = _mm256_setzero_ps();
            __m256 temp4 = _mm256_setzero_ps();
            __m256 temp5 = _mm256_setzero_ps();
            __m256 temp6 = _mm256_setzero_ps();
            __m256 temp7 = _mm256_setzero_ps();
            __m256 temp8 = _mm256_setzero_ps();
            __m256 temp9 = _mm256_setzero_ps();
            // temp = 0.0f;

            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                  unsigned long long int in_subscript = b * (in_yx_depth)
                    + ((y * Stride_Y_dim + off_y) * in_x_depth)
                    + ((x * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                    + d;
                  unsigned long long int in_subscript2 = b * (in_yx_depth)
                    + ((y * Stride_Y_dim + off_y) * in_x_depth)
                    + (((x+1) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                    + d;
                  unsigned long long int in_subscript3 = b * (in_yx_depth)
                    + ((y * Stride_Y_dim + off_y) * in_x_depth)
                    + (((x+2) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                    + d;


                  unsigned long long int filter_subscript = (m * mask_yx_depth)
                    + (off_y * mask_x_depth)
                    + (off_x << left_shift(Input_depth_dim))
                    + d;
                  unsigned long long int filter_subscript2 = ((m+1) * mask_yx_depth)
                    + (off_y * mask_x_depth)
                    + (off_x << left_shift(Input_depth_dim))
                    + d;
                  unsigned long long int filter_subscript3 = ((m+2) * mask_yx_depth)
                    + (off_y * mask_x_depth)
                    + (off_x << left_shift(Input_depth_dim))
                    + d;

                  __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                  __m256 s2 = _mm256_load_ps(&in_FP[in_subscript2]);
                  __m256 s3 = _mm256_load_ps(&in_FP[in_subscript3]);

                  __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]); 
                  __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);
                  __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript3]);

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s, w2, temp4);
                  temp5 = _mm256_fmadd_ps(s2, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s3, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s, w3, temp7);
                  temp8 = _mm256_fmadd_ps(s2, w3, temp8);
                  temp9 = _mm256_fmadd_ps(s3, w3, temp9);

                  // temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                  // temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                  // temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                  // temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s, w2));
                  // temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s2, w2));
                  // temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s3, w2));
                  // temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w3));
                  // temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w3));
                  // temp9 = _mm256_add_ps(temp9, _mm256_mul_ps(s3, w3));
                }
              }
            }


            unsigned long long int out_subscript = b * (out_yx_depth) +
              y * (out_x_depth) +
              (x << left_shift(Output_depth_dim))
              + m;
            unsigned long long int out_subscript2 = b * (out_yx_depth) +
              y * (out_x_depth) +
              ((x+1) << left_shift(Output_depth_dim))
              + m;
            unsigned long long int out_subscript3 = b * (out_yx_depth) +
              y * (out_x_depth) +
              ((x+2) << left_shift(Output_depth_dim))
              + m;
            unsigned long long int out_subscript4 = b * (out_yx_depth) +
              y * (out_x_depth) +
              (x << left_shift(Output_depth_dim))
              + (m+1);
            unsigned long long int out_subscript5 = b * (out_yx_depth) +
              y * (out_x_depth) +
              ((x+1) << left_shift(Output_depth_dim))
              + (m+1);
            unsigned long long int out_subscript6 = b * (out_yx_depth) +
              y * (out_x_depth) +
              ((x+2) << left_shift(Output_depth_dim))
              + (m+1);
            unsigned long long int out_subscript7 = b * (out_yx_depth) +
              y * (out_x_depth) +
              (x << left_shift(Output_depth_dim))
              + (m+2);
            unsigned long long int out_subscript8 = b * (out_yx_depth) +
              y * (out_x_depth) +
              ((x+1) << left_shift(Output_depth_dim))
              + (m+2);
            unsigned long long int out_subscript9 = b * (out_yx_depth) +
              y * (out_x_depth) +
              ((x+2) << left_shift(Output_depth_dim))
              + (m+2);



            temp = _mm256_hadd_ps(temp, temp);
            temp = _mm256_hadd_ps(temp, temp);
            __m128 tempLo = _mm256_castps256_ps128(temp);
            __m128 tempHi = _mm256_extractf128_ps(temp, 1);
            __m128 sseSum = _mm_add_ps(tempLo, tempHi);

            float sum = _mm_cvtss_f32(sseSum);

            sum += bias;
            out_to_compare_with_FP[out_subscript] = RELU(sum);


            temp2 = _mm256_hadd_ps(temp2, temp2);
            temp2 = _mm256_hadd_ps(temp2, temp2);
            __m128 tempLo2 = _mm256_castps256_ps128(temp2);
            __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
            __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum2 = _mm_cvtss_f32(sseSum2);

            sum2 += bias;
            out_to_compare_with_FP[out_subscript2] = RELU(sum2);


            temp3 = _mm256_hadd_ps(temp3, temp3);
            temp3 = _mm256_hadd_ps(temp3, temp3);
            __m128 tempLo3 = _mm256_castps256_ps128(temp3);
            __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
            __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

            float sum3 = _mm_cvtss_f32(sseSum3);

            sum3 += bias;
            out_to_compare_with_FP[out_subscript3] = RELU(sum3);


            temp4 = _mm256_hadd_ps(temp4, temp4);
            temp4 = _mm256_hadd_ps(temp4, temp4);
            __m128 tempLo4 = _mm256_castps256_ps128(temp4);
            __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
            __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

            float sum4 = _mm_cvtss_f32(sseSum4);

            sum4 += bias2;
            out_to_compare_with_FP[out_subscript4] = RELU(sum4);


            temp5 = _mm256_hadd_ps(temp5, temp5);
            temp5 = _mm256_hadd_ps(temp5, temp5);
            __m128 tempLo5 = _mm256_castps256_ps128(temp5);
            __m128 tempHi5 = _mm256_extractf128_ps(temp5, 1);
            __m128 sseSum5 = _mm_add_ps(tempLo5, tempHi5);

            float sum5 = _mm_cvtss_f32(sseSum5);

            sum5 += bias2;
            out_to_compare_with_FP[out_subscript5] = RELU(sum5);


            temp6 = _mm256_hadd_ps(temp6, temp6);
            temp6 = _mm256_hadd_ps(temp6, temp6);
            __m128 tempLo6 = _mm256_castps256_ps128(temp6);
            __m128 tempHi6 = _mm256_extractf128_ps(temp6, 1);
            __m128 sseSum6 = _mm_add_ps(tempLo6, tempHi6);

            float sum6 = _mm_cvtss_f32(sseSum6);

            sum6 += bias2;
            out_to_compare_with_FP[out_subscript6] = RELU(sum6);


            temp7 = _mm256_hadd_ps(temp7, temp7);
            temp7 = _mm256_hadd_ps(temp7, temp7);
            __m128 tempLo7 = _mm256_castps256_ps128(temp7);
            __m128 tempHi7 = _mm256_extractf128_ps(temp7, 1);
            __m128 sseSum7 = _mm_add_ps(tempLo7, tempHi7);

            float sum7 = _mm_cvtss_f32(sseSum7);

            sum7 += bias3;
            out_to_compare_with_FP[out_subscript7] = RELU(sum7);


            temp8 = _mm256_hadd_ps(temp8, temp8);
            temp8 = _mm256_hadd_ps(temp8, temp8);
            __m128 tempLo8 = _mm256_castps256_ps128(temp8);
            __m128 tempHi8 = _mm256_extractf128_ps(temp8, 1);
            __m128 sseSum8 = _mm_add_ps(tempLo8, tempHi8);

            float sum8 = _mm_cvtss_f32(sseSum8);

            sum8 += bias3;
            out_to_compare_with_FP[out_subscript8] = RELU(sum8);


            temp9 = _mm256_hadd_ps(temp9, temp9);
            temp9 = _mm256_hadd_ps(temp9, temp9);
            __m128 tempLo9 = _mm256_castps256_ps128(temp9);
            __m128 tempHi9 = _mm256_extractf128_ps(temp9, 1);
            __m128 sseSum9 = _mm_add_ps(tempLo9, tempHi9);

            float sum9 = _mm_cvtss_f32(sseSum9);

            sum9 += bias3;
            out_to_compare_with_FP[out_subscript9] = RELU(sum9);
          }
          // overflow/ fallback x loop
          for (; x < Output_X_dim; x++) {	//Output Width
            __m256 temp = _mm256_setzero_ps();
            __m256 temp2 = _mm256_setzero_ps();
            __m256 temp3 = _mm256_setzero_ps();
            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                  unsigned long long int in_subscript = b * (in_yx_depth)
                    + ((y * Stride_Y_dim + off_y) * in_x_depth)
                    + ((x * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                    + d;

                  unsigned long long int filter_subscript = (m * mask_yx_depth)
                    + (off_y * mask_x_depth)
                    + (off_x << left_shift(Input_depth_dim))
                    + d;
                  unsigned long long int filter_subscript2 = ((m+1) * mask_yx_depth)
                    + (off_y * mask_x_depth)
                    + (off_x << left_shift(Input_depth_dim))
                    + d;
                  unsigned long long int filter_subscript3 = ((m+2) * mask_yx_depth)
                    + (off_y * mask_x_depth)
                    + (off_x << left_shift(Input_depth_dim))
                    + d;

                  __m256 s = _mm256_loadu_ps(&in_FP[in_subscript]);

                  __m256 w = _mm256_loadu_ps(&filter_FP[filter_subscript]);
                  __m256 w2 = _mm256_loadu_ps(&filter_FP[filter_subscript2]);
                  __m256 w3 = _mm256_loadu_ps(&filter_FP[filter_subscript3]);

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s, w2, temp2);
                  temp3 = _mm256_fmadd_ps(s, w3, temp3);


                  // temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                  // temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s, w2));
                  // temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w3));
                }
              }
            }
            unsigned long long int out_subscript = b * (out_yx_depth) +
              y * (out_x_depth) +
              (x << left_shift(Output_depth_dim))
              + m;
            unsigned long long int out_subscript2 = b * (out_yx_depth) +
              y * (out_x_depth) +
              (x << left_shift(Output_depth_dim))
              + (m+1);
            unsigned long long int out_subscript3 = b * (out_yx_depth) +
              y * (out_x_depth) +
              (x << left_shift(Output_depth_dim))
              + (m+2);

            temp = _mm256_hadd_ps(temp, temp);
            temp = _mm256_hadd_ps(temp, temp);
            __m128 tempLo = _mm256_castps256_ps128(temp);
            __m128 tempHi = _mm256_extractf128_ps(temp, 1);
            __m128 sseSum = _mm_add_ps(tempLo, tempHi);

            float sum = _mm_cvtss_f32(sseSum);

            sum += bias;
            out_to_compare_with_FP[out_subscript] = RELU(sum);


            temp2 = _mm256_hadd_ps(temp2, temp2);
            temp2 = _mm256_hadd_ps(temp2, temp2);
            __m128 tempLo2 = _mm256_castps256_ps128(temp2);
            __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
            __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum2 = _mm_cvtss_f32(sseSum2);

            sum2 += bias2;
            out_to_compare_with_FP[out_subscript2] = RELU(sum2);


            temp3 = _mm256_hadd_ps(temp3, temp3);
            temp3 = _mm256_hadd_ps(temp3, temp3);
            __m128 tempLo3 = _mm256_castps256_ps128(temp3);
            __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
            __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

            float sum3 = _mm_cvtss_f32(sseSum3);

            sum3 += bias3;
            out_to_compare_with_FP[out_subscript3] = RELU(sum3);

          }
        }
        for (; m < Output_depth_dim; m++) {
          const float bias = bias_array_FP[m];

          for (unsigned int x = 0; x < Output_X_dim; x+=4) {	//Output Width
            __m256 temp = _mm256_setzero_ps();
            __m256 temp2 = _mm256_setzero_ps();
            __m256 temp3 = _mm256_setzero_ps();
            __m256 temp4 = _mm256_setzero_ps();

            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                  unsigned long long int in_subscript = b * (in_yx_depth)
                    + ((y * Stride_Y_dim + off_y) * in_x_depth)
                    + ((x * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                    + d;
                  unsigned long long int in_subscript2 = b * (in_yx_depth)
                    + ((y * Stride_Y_dim + off_y) * in_x_depth)
                    + (((x+1) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                    + d;
                  unsigned long long int in_subscript3 = b * (in_yx_depth)
                    + ((y * Stride_Y_dim + off_y) * in_x_depth)
                    + (((x+2) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                    + d;
                  unsigned long long int in_subscript4 = b * (in_yx_depth)
                    + ((y * Stride_Y_dim + off_y) * in_x_depth)
                    + (((x+3) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                    + d;

                  unsigned long long int filter_subscript = (m * mask_yx_depth)
                    + (off_y * mask_x_depth)
                    + (off_x << left_shift(Input_depth_dim))
                    + d;

                  __m256 s = _mm256_loadu_ps(&in_FP[in_subscript]);
                  __m256 s2 = _mm256_loadu_ps(&in_FP[in_subscript2]);
                  __m256 s3 = _mm256_loadu_ps(&in_FP[in_subscript3]);
                  __m256 s4 = _mm256_loadu_ps(&in_FP[in_subscript4]);

                  __m256 w = _mm256_loadu_ps(&filter_FP[filter_subscript]);

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);


                  // temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                  // temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                  // temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                  // temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s4, w));
                }
              }
            }

            unsigned long long int out_subscript = b * (out_yx_depth) +
              y * (out_x_depth) +
              (x << left_shift(Output_depth_dim))
              + m;
            unsigned long long int out_subscript2 = b * (out_yx_depth) +
              y * (out_x_depth) +
              ((x+1) << left_shift(Output_depth_dim))
              + m;
            unsigned long long int out_subscript3 = b * (out_yx_depth) +
              y * (out_x_depth) +
              ((x+2) << left_shift(Output_depth_dim))
              + m;
            unsigned long long int out_subscript4 = b * (out_yx_depth) +
              y * (out_x_depth) +
              ((x+3) << left_shift(Output_depth_dim))
              + m;

            temp = _mm256_hadd_ps(temp, temp);
            temp = _mm256_hadd_ps(temp, temp);
            __m128 tempLo = _mm256_castps256_ps128(temp);
            __m128 tempHi = _mm256_extractf128_ps(temp, 1);
            __m128 sseSum = _mm_add_ps(tempLo, tempHi);

            float sum = _mm_cvtss_f32(sseSum);

            sum += bias;
            out_to_compare_with_FP[out_subscript] = RELU(sum);


            temp2 = _mm256_hadd_ps(temp2, temp2);
            temp2 = _mm256_hadd_ps(temp2, temp2);
            __m128 tempLo2 = _mm256_castps256_ps128(temp2);
            __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
            __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum2 = _mm_cvtss_f32(sseSum2);

            sum2 += bias;
            out_to_compare_with_FP[out_subscript2] = RELU(sum2);


            temp3 = _mm256_hadd_ps(temp3, temp3);
            temp3 = _mm256_hadd_ps(temp3, temp3);
            __m128 tempLo3 = _mm256_castps256_ps128(temp3);
            __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
            __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

            float sum3 = _mm_cvtss_f32(sseSum3);

            sum3 += bias;
            out_to_compare_with_FP[out_subscript3] = RELU(sum3);


            temp4 = _mm256_hadd_ps(temp4, temp4);
            temp4 = _mm256_hadd_ps(temp4, temp4);
            __m128 tempLo4 = _mm256_castps256_ps128(temp4);
            __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
            __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

            float sum4 = _mm_cvtss_f32(sseSum4);

            sum4 += bias;
            out_to_compare_with_FP[out_subscript4] = RELU(sum4);

          }
        }
      }
    }
  }
  #undef left_shift
  #undef RELU
  printf("\n from optv12 x3 m3 - fmadd instructions %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}






// v13 loop tiling - using v12 funcs as base


// loop tiling on m loop - optimised_layer_v12_x2m4_fmadd_FP()
// output= 2^14 ~493 GFLOPS (~473 GFLOPS for v12), impact of cache misses begins to outweigh computation cost
int optimised_layer_v13_x2m4_loop_tiling_m_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  #define left_shift(x) (__builtin_ctz(x))  // inline function effectively using inline assembly to return num of trailing zeros (i.e. 128 returns 7, 2 * 128 == 2 << 7)
  #define RELU(x) ((x) < 0.0f ? 0.0f : (x)) // return x if greater than 0, else return 0

  // float bias, bias2, bias3, bias4;
  // __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

  const unsigned long long int in_yx_depth = Input_Y_dim * Input_X_dim << left_shift(Input_depth_dim);
  const unsigned long long int in_x_depth = Input_X_dim  << left_shift(Input_depth_dim);

  const unsigned long long int mask_yx_depth = Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim);
  const unsigned long long int mask_x_depth = Mask_X_dim << left_shift(Input_depth_dim);

  const unsigned long long int out_yx_depth = Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim);
  const unsigned long long int out_x_depth = Output_X_dim << left_shift(Output_depth_dim);


  #define m_tile 16
  

  #pragma omp parallel for shared(in_yx_depth, in_x_depth, mask_yx_depth, mask_x_depth, out_yx_depth, out_x_depth) \
    default(shared) collapse(3) schedule(static)
  {
    for (unsigned int mm = 0; mm < Output_depth_dim; mm+=m_tile) {
      for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
        for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
          for (unsigned int m = mm; m < mm+m_tile; m+=4) { //channels
            const float bias = bias_array_FP[m];
            const float bias2 = bias_array_FP[m+1];
            const float bias3 = bias_array_FP[m+2];
            const float bias4 = bias_array_FP[m+3];

            for (unsigned int x = 0; x < Output_X_dim; x+=2) {	//Output Width
              __m256 temp = _mm256_setzero_ps();
              __m256 temp2 = _mm256_setzero_ps();
              __m256 temp3 = _mm256_setzero_ps();
              __m256 temp4 = _mm256_setzero_ps();
              __m256 temp5 = _mm256_setzero_ps();
              __m256 temp6 = _mm256_setzero_ps();
              __m256 temp7 = _mm256_setzero_ps();
              __m256 temp8 = _mm256_setzero_ps();

              for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
                for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                  for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                    unsigned long long int in_subscript = b * (in_yx_depth)
                      + ((y * Stride_Y_dim + off_y) * in_x_depth)
                      + ((x * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                      + d;

                    unsigned long long int filter_subscript = (m * mask_yx_depth)
                      + (off_y * mask_x_depth)
                      + (off_x << left_shift(Input_depth_dim))
                      + d;

                    __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                    __m256 s2 = _mm256_load_ps(&in_FP[in_subscript + Input_depth_dim]);

                    __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]);
                    __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript + (Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim))]);
                    __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript + (2 * (Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim)))]);
                    __m256 w4 = _mm256_load_ps(&filter_FP[filter_subscript + (3 * (Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim)))]);


                    temp = _mm256_fmadd_ps(s, w, temp);
                    temp2 = _mm256_fmadd_ps(s2, w, temp2);
                    temp3 = _mm256_fmadd_ps(s, w2, temp3);
                    temp4 = _mm256_fmadd_ps(s2, w2, temp4);
                    temp5 = _mm256_fmadd_ps(s, w3, temp5);
                    temp6 = _mm256_fmadd_ps(s2, w3, temp6);
                    temp7 = _mm256_fmadd_ps(s, w4, temp7);
                    temp8 = _mm256_fmadd_ps(s2, w4, temp8);

                    // temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                    // temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                    // temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w2));
                    // temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s2, w2));
                    // temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s, w3));
                    // temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s2, w3));
                    // temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w4));
                    // temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w4));
                  }
                }
              }


              unsigned long long int out_subscript = b * (out_yx_depth) 
                + y * (out_x_depth) 
                + (x << left_shift(Output_depth_dim))
                + m;
              unsigned long long int out_subscript2 = out_subscript + Output_depth_dim;



              temp = _mm256_hadd_ps(temp, temp);
              temp = _mm256_hadd_ps(temp, temp);
              __m128 tempLo = _mm256_castps256_ps128(temp);
              __m128 tempHi = _mm256_extractf128_ps(temp, 1);
              __m128 sseSum = _mm_add_ps(tempLo, tempHi);

              float sum = _mm_cvtss_f32(sseSum);

              sum += bias;
              out_to_compare_with_FP[out_subscript] = RELU(sum);


              temp2 = _mm256_hadd_ps(temp2, temp2);
              temp2 = _mm256_hadd_ps(temp2, temp2);
              __m128 tempLo2 = _mm256_castps256_ps128(temp2);
              __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
              __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

              float sum2 = _mm_cvtss_f32(sseSum2);

              sum2 += bias;
              out_to_compare_with_FP[out_subscript2] = RELU(sum2);


              temp3 = _mm256_hadd_ps(temp3, temp3);
              temp3 = _mm256_hadd_ps(temp3, temp3);
              tempLo = _mm256_castps256_ps128(temp3);
              tempHi = _mm256_extractf128_ps(temp3, 1);
              sseSum = _mm_add_ps(tempLo, tempHi);

              float sum3 = _mm_cvtss_f32(sseSum);

              sum3 += bias2;
              out_to_compare_with_FP[out_subscript + 1] = RELU(sum3);


              temp4 = _mm256_hadd_ps(temp4, temp4);
              temp4 = _mm256_hadd_ps(temp4, temp4);
              tempLo2 = _mm256_castps256_ps128(temp4);
              tempHi2 = _mm256_extractf128_ps(temp4, 1);
              sseSum2 = _mm_add_ps(tempLo2, tempHi2);

              float sum4 = _mm_cvtss_f32(sseSum2);

              sum4 += bias2;
              out_to_compare_with_FP[out_subscript2 + 1] = RELU(sum4);


              temp5 = _mm256_hadd_ps(temp5, temp5);
              temp5 = _mm256_hadd_ps(temp5, temp5);
              tempLo = _mm256_castps256_ps128(temp5);
              tempHi = _mm256_extractf128_ps(temp5, 1);
              sseSum = _mm_add_ps(tempLo, tempHi);

              float sum5 = _mm_cvtss_f32(sseSum);

              sum5 += bias3;
              out_to_compare_with_FP[out_subscript + 2] = RELU(sum5);


              temp6 = _mm256_hadd_ps(temp6, temp6);
              temp6 = _mm256_hadd_ps(temp6, temp6);
              tempLo2 = _mm256_castps256_ps128(temp6);
              tempHi2 = _mm256_extractf128_ps(temp6, 1);
              sseSum2 = _mm_add_ps(tempLo2, tempHi2);

              float sum6 = _mm_cvtss_f32(sseSum2);

              sum6 += bias3;
              out_to_compare_with_FP[out_subscript2 + 2] = RELU(sum6);


              temp7 = _mm256_hadd_ps(temp7, temp7);
              temp7 = _mm256_hadd_ps(temp7, temp7);
              tempLo = _mm256_castps256_ps128(temp7);
              tempHi = _mm256_extractf128_ps(temp7, 1);
              sseSum = _mm_add_ps(tempLo, tempHi);

              float sum7 = _mm_cvtss_f32(sseSum);

              sum7 += bias4;
              out_to_compare_with_FP[out_subscript + 3] = RELU(sum7);


              temp8 = _mm256_hadd_ps(temp8, temp8);
              temp8 = _mm256_hadd_ps(temp8, temp8);
              tempLo2 = _mm256_castps256_ps128(temp8);
              tempHi2 = _mm256_extractf128_ps(temp8, 1);
              sseSum2 = _mm_add_ps(tempLo2, tempHi2);

              float sum8 = _mm_cvtss_f32(sseSum2);

              sum8 += bias4;
              out_to_compare_with_FP[out_subscript2 + 3] = RELU(sum8);
            }
          }
        }
      }
    }
  }
  #undef left_shift
  #undef RELU
  #undef m_tile
  printf("\n from optv13 x2 m4 - loop tiling on m %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// loop tiling on m loop - optimised_layer_v12_x3m3_fmadd_FP()
// output= 2^14 ~507 GFLOPS (~518 GFLOPS vor v12)
int optimised_layer_v13_x3m3_loop_tiling_m_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  #define left_shift(x) (__builtin_ctz(x))  // inline function effectively using inline assembly to return num of trailing zeros (i.e. 128 returns 7, 2 * 128 == 2 << 7)
  #define RELU(x) ((x) < 0.0f ? 0.0f : (x)) // return x if greater than 0, else return 0
  #define m_tile 16
  
  // float bias, bias2, bias3;
  // __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9;

  const unsigned long long int in_yx_depth = Input_Y_dim * Input_X_dim << left_shift(Input_depth_dim);
  const unsigned long long int in_x_depth = Input_X_dim  << left_shift(Input_depth_dim);

  const unsigned long long int mask_yx_depth = Mask_Y_dim * Mask_X_dim << left_shift(Input_depth_dim);
  const unsigned long long int mask_x_depth = Mask_X_dim << left_shift(Input_depth_dim);

  const unsigned long long int out_yx_depth = Output_X_dim * Output_Y_dim << left_shift(Output_depth_dim);
  const unsigned long long int out_x_depth = Output_X_dim << left_shift(Output_depth_dim);

  // Calculate loop bounds for unrolled loops
  const int m_bound = (m_tile/ 3) * 3;
  const int x_bound = (Output_X_dim/ 3) * 3;

  unsigned int m, x;
  #pragma omp parallel for private(m, x) shared(in_yx_depth, in_x_depth, mask_yx_depth, mask_x_depth, out_yx_depth, out_x_depth, m_bound, x_bound) \
    default(shared) collapse(2) schedule(static)
  {
    for (unsigned int mm = 0; mm < Output_depth_dim; mm+=m_tile) {
      for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
        for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
          for (m = mm; m < mm+m_bound; m+=3) { //channels
            const float bias = bias_array_FP[m];
            const float bias2 = bias_array_FP[m+1];
            const float bias3 = bias_array_FP[m+2];

            for (x = 0; x < x_bound; x+=3) {	//Output Width
              __m256 temp = _mm256_setzero_ps();
              __m256 temp2 = _mm256_setzero_ps();
              __m256 temp3 = _mm256_setzero_ps();
              __m256 temp4 = _mm256_setzero_ps();
              __m256 temp5 = _mm256_setzero_ps();
              __m256 temp6 = _mm256_setzero_ps();
              __m256 temp7 = _mm256_setzero_ps();
              __m256 temp8 = _mm256_setzero_ps();
              __m256 temp9 = _mm256_setzero_ps();
              // temp = 0.0f;

              for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
                for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                  for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                    unsigned long long int in_subscript = b * (in_yx_depth)
                      + ((y * Stride_Y_dim + off_y) * in_x_depth)
                      + ((x * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                      + d;
                    unsigned long long int in_subscript2 = b * (in_yx_depth)
                      + ((y * Stride_Y_dim + off_y) * in_x_depth)
                      + (((x+1) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                      + d;
                    unsigned long long int in_subscript3 = b * (in_yx_depth)
                      + ((y * Stride_Y_dim + off_y) * in_x_depth)
                      + (((x+2) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                      + d;


                    unsigned long long int filter_subscript = (m * mask_yx_depth)
                      + (off_y * mask_x_depth)
                      + (off_x << left_shift(Input_depth_dim))
                      + d;
                    unsigned long long int filter_subscript2 = ((m+1) * mask_yx_depth)
                      + (off_y * mask_x_depth)
                      + (off_x << left_shift(Input_depth_dim))
                      + d;
                    unsigned long long int filter_subscript3 = ((m+2) * mask_yx_depth)
                      + (off_y * mask_x_depth)
                      + (off_x << left_shift(Input_depth_dim))
                      + d;

                    __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                    __m256 s2 = _mm256_load_ps(&in_FP[in_subscript2]);
                    __m256 s3 = _mm256_load_ps(&in_FP[in_subscript3]);

                    __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]); 
                    __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript2]);
                    __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript3]);

                    temp = _mm256_fmadd_ps(s, w, temp);
                    temp2 = _mm256_fmadd_ps(s2, w, temp2);
                    temp3 = _mm256_fmadd_ps(s3, w, temp3);
                    temp4 = _mm256_fmadd_ps(s, w2, temp4);
                    temp5 = _mm256_fmadd_ps(s2, w2, temp5);
                    temp6 = _mm256_fmadd_ps(s3, w2, temp6);
                    temp7 = _mm256_fmadd_ps(s, w3, temp7);
                    temp8 = _mm256_fmadd_ps(s2, w3, temp8);
                    temp9 = _mm256_fmadd_ps(s3, w3, temp9);

                    // temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                    // temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                    // temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                    // temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s, w2));
                    // temp5 = _mm256_add_ps(temp5, _mm256_mul_ps(s2, w2));
                    // temp6 = _mm256_add_ps(temp6, _mm256_mul_ps(s3, w2));
                    // temp7 = _mm256_add_ps(temp7, _mm256_mul_ps(s, w3));
                    // temp8 = _mm256_add_ps(temp8, _mm256_mul_ps(s2, w3));
                    // temp9 = _mm256_add_ps(temp9, _mm256_mul_ps(s3, w3));
                  }
                }
              }


              unsigned long long int out_subscript = b * (out_yx_depth) +
                y * (out_x_depth) +
                (x << left_shift(Output_depth_dim))
                + m;
              unsigned long long int out_subscript2 = b * (out_yx_depth) +
                y * (out_x_depth) +
                ((x+1) << left_shift(Output_depth_dim))
                + m;
              unsigned long long int out_subscript3 = b * (out_yx_depth) +
                y * (out_x_depth) +
                ((x+2) << left_shift(Output_depth_dim))
                + m;
              unsigned long long int out_subscript4 = b * (out_yx_depth) +
                y * (out_x_depth) +
                (x << left_shift(Output_depth_dim))
                + (m+1);
              unsigned long long int out_subscript5 = b * (out_yx_depth) +
                y * (out_x_depth) +
                ((x+1) << left_shift(Output_depth_dim))
                + (m+1);
              unsigned long long int out_subscript6 = b * (out_yx_depth) +
                y * (out_x_depth) +
                ((x+2) << left_shift(Output_depth_dim))
                + (m+1);
              unsigned long long int out_subscript7 = b * (out_yx_depth) +
                y * (out_x_depth) +
                (x << left_shift(Output_depth_dim))
                + (m+2);
              unsigned long long int out_subscript8 = b * (out_yx_depth) +
                y * (out_x_depth) +
                ((x+1) << left_shift(Output_depth_dim))
                + (m+2);
              unsigned long long int out_subscript9 = b * (out_yx_depth) +
                y * (out_x_depth) +
                ((x+2) << left_shift(Output_depth_dim))
                + (m+2);



              temp = _mm256_hadd_ps(temp, temp);
              temp = _mm256_hadd_ps(temp, temp);
              __m128 tempLo = _mm256_castps256_ps128(temp);
              __m128 tempHi = _mm256_extractf128_ps(temp, 1);
              __m128 sseSum = _mm_add_ps(tempLo, tempHi);

              float sum = _mm_cvtss_f32(sseSum);

              sum += bias;
              out_to_compare_with_FP[out_subscript] = RELU(sum);


              temp2 = _mm256_hadd_ps(temp2, temp2);
              temp2 = _mm256_hadd_ps(temp2, temp2);
              __m128 tempLo2 = _mm256_castps256_ps128(temp2);
              __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
              __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

              float sum2 = _mm_cvtss_f32(sseSum2);

              sum2 += bias;
              out_to_compare_with_FP[out_subscript2] = RELU(sum2);


              temp3 = _mm256_hadd_ps(temp3, temp3);
              temp3 = _mm256_hadd_ps(temp3, temp3);
              __m128 tempLo3 = _mm256_castps256_ps128(temp3);
              __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
              __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

              float sum3 = _mm_cvtss_f32(sseSum3);

              sum3 += bias;
              out_to_compare_with_FP[out_subscript3] = RELU(sum3);


              temp4 = _mm256_hadd_ps(temp4, temp4);
              temp4 = _mm256_hadd_ps(temp4, temp4);
              __m128 tempLo4 = _mm256_castps256_ps128(temp4);
              __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
              __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

              float sum4 = _mm_cvtss_f32(sseSum4);

              sum4 += bias2;
              out_to_compare_with_FP[out_subscript4] = RELU(sum4);


              temp5 = _mm256_hadd_ps(temp5, temp5);
              temp5 = _mm256_hadd_ps(temp5, temp5);
              __m128 tempLo5 = _mm256_castps256_ps128(temp5);
              __m128 tempHi5 = _mm256_extractf128_ps(temp5, 1);
              __m128 sseSum5 = _mm_add_ps(tempLo5, tempHi5);

              float sum5 = _mm_cvtss_f32(sseSum5);

              sum5 += bias2;
              out_to_compare_with_FP[out_subscript5] = RELU(sum5);


              temp6 = _mm256_hadd_ps(temp6, temp6);
              temp6 = _mm256_hadd_ps(temp6, temp6);
              __m128 tempLo6 = _mm256_castps256_ps128(temp6);
              __m128 tempHi6 = _mm256_extractf128_ps(temp6, 1);
              __m128 sseSum6 = _mm_add_ps(tempLo6, tempHi6);

              float sum6 = _mm_cvtss_f32(sseSum6);

              sum6 += bias2;
              out_to_compare_with_FP[out_subscript6] = RELU(sum6);


              temp7 = _mm256_hadd_ps(temp7, temp7);
              temp7 = _mm256_hadd_ps(temp7, temp7);
              __m128 tempLo7 = _mm256_castps256_ps128(temp7);
              __m128 tempHi7 = _mm256_extractf128_ps(temp7, 1);
              __m128 sseSum7 = _mm_add_ps(tempLo7, tempHi7);

              float sum7 = _mm_cvtss_f32(sseSum7);

              sum7 += bias3;
              out_to_compare_with_FP[out_subscript7] = RELU(sum7);


              temp8 = _mm256_hadd_ps(temp8, temp8);
              temp8 = _mm256_hadd_ps(temp8, temp8);
              __m128 tempLo8 = _mm256_castps256_ps128(temp8);
              __m128 tempHi8 = _mm256_extractf128_ps(temp8, 1);
              __m128 sseSum8 = _mm_add_ps(tempLo8, tempHi8);

              float sum8 = _mm_cvtss_f32(sseSum8);

              sum8 += bias3;
              out_to_compare_with_FP[out_subscript8] = RELU(sum8);


              temp9 = _mm256_hadd_ps(temp9, temp9);
              temp9 = _mm256_hadd_ps(temp9, temp9);
              __m128 tempLo9 = _mm256_castps256_ps128(temp9);
              __m128 tempHi9 = _mm256_extractf128_ps(temp9, 1);
              __m128 sseSum9 = _mm_add_ps(tempLo9, tempHi9);

              float sum9 = _mm_cvtss_f32(sseSum9);

              sum9 += bias3;
              out_to_compare_with_FP[out_subscript9] = RELU(sum9);
            }
            // overflow/ fallback x loop
            for (; x < Output_X_dim; x++) {	//Output Width
              __m256 temp = _mm256_setzero_ps();
              __m256 temp2 = _mm256_setzero_ps();
              __m256 temp3 = _mm256_setzero_ps();
              for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
                for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                  for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                    unsigned long long int in_subscript = b * (in_yx_depth)
                      + ((y * Stride_Y_dim + off_y) * in_x_depth)
                      + ((x * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                      + d;

                    unsigned long long int filter_subscript = (m * mask_yx_depth)
                      + (off_y * mask_x_depth)
                      + (off_x << left_shift(Input_depth_dim))
                      + d;
                    unsigned long long int filter_subscript2 = ((m+1) * mask_yx_depth)
                      + (off_y * mask_x_depth)
                      + (off_x << left_shift(Input_depth_dim))
                      + d;
                    unsigned long long int filter_subscript3 = ((m+2) * mask_yx_depth)
                      + (off_y * mask_x_depth)
                      + (off_x << left_shift(Input_depth_dim))
                      + d;

                    __m256 s = _mm256_loadu_ps(&in_FP[in_subscript]);

                    __m256 w = _mm256_loadu_ps(&filter_FP[filter_subscript]);
                    __m256 w2 = _mm256_loadu_ps(&filter_FP[filter_subscript2]);
                    __m256 w3 = _mm256_loadu_ps(&filter_FP[filter_subscript3]);

                    temp = _mm256_fmadd_ps(s, w, temp);
                    temp2 = _mm256_fmadd_ps(s, w2, temp2);
                    temp3 = _mm256_fmadd_ps(s, w3, temp3);


                    // temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                    // temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s, w2));
                    // temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s, w3));
                  }
                }
              }
              unsigned long long int out_subscript = b * (out_yx_depth) +
                y * (out_x_depth) +
                (x << left_shift(Output_depth_dim))
                + m;
              unsigned long long int out_subscript2 = b * (out_yx_depth) +
                y * (out_x_depth) +
                (x << left_shift(Output_depth_dim))
                + (m+1);
              unsigned long long int out_subscript3 = b * (out_yx_depth) +
                y * (out_x_depth) +
                (x << left_shift(Output_depth_dim))
                + (m+2);

              temp = _mm256_hadd_ps(temp, temp);
              temp = _mm256_hadd_ps(temp, temp);
              __m128 tempLo = _mm256_castps256_ps128(temp);
              __m128 tempHi = _mm256_extractf128_ps(temp, 1);
              __m128 sseSum = _mm_add_ps(tempLo, tempHi);

              float sum = _mm_cvtss_f32(sseSum);

              sum += bias;
              out_to_compare_with_FP[out_subscript] = RELU(sum);


              temp2 = _mm256_hadd_ps(temp2, temp2);
              temp2 = _mm256_hadd_ps(temp2, temp2);
              __m128 tempLo2 = _mm256_castps256_ps128(temp2);
              __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
              __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

              float sum2 = _mm_cvtss_f32(sseSum2);

              sum2 += bias2;
              out_to_compare_with_FP[out_subscript2] = RELU(sum2);


              temp3 = _mm256_hadd_ps(temp3, temp3);
              temp3 = _mm256_hadd_ps(temp3, temp3);
              __m128 tempLo3 = _mm256_castps256_ps128(temp3);
              __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
              __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

              float sum3 = _mm_cvtss_f32(sseSum3);

              sum3 += bias3;
              out_to_compare_with_FP[out_subscript3] = RELU(sum3);

            }
          }
          for (; m < mm+m_tile; m++) {
            const float bias = bias_array_FP[m];

            for (unsigned int x = 0; x < Output_X_dim; x+=4) {	//Output Width
              __m256 temp = _mm256_setzero_ps();
              __m256 temp2 = _mm256_setzero_ps();
              __m256 temp3 = _mm256_setzero_ps();
              __m256 temp4 = _mm256_setzero_ps();

              for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
                for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                  for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                    unsigned long long int in_subscript = b * (in_yx_depth)
                      + ((y * Stride_Y_dim + off_y) * in_x_depth)
                      + ((x * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                      + d;
                    unsigned long long int in_subscript2 = b * (in_yx_depth)
                      + ((y * Stride_Y_dim + off_y) * in_x_depth)
                      + (((x+1) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                      + d;
                    unsigned long long int in_subscript3 = b * (in_yx_depth)
                      + ((y * Stride_Y_dim + off_y) * in_x_depth)
                      + (((x+2) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                      + d;
                    unsigned long long int in_subscript4 = b * (in_yx_depth)
                      + ((y * Stride_Y_dim + off_y) * in_x_depth)
                      + (((x+3) * Stride_X_dim + off_x) << left_shift(Input_depth_dim))
                      + d;

                    unsigned long long int filter_subscript = (m * mask_yx_depth)
                      + (off_y * mask_x_depth)
                      + (off_x << left_shift(Input_depth_dim))
                      + d;

                    __m256 s = _mm256_loadu_ps(&in_FP[in_subscript]);
                    __m256 s2 = _mm256_loadu_ps(&in_FP[in_subscript2]);
                    __m256 s3 = _mm256_loadu_ps(&in_FP[in_subscript3]);
                    __m256 s4 = _mm256_loadu_ps(&in_FP[in_subscript4]);

                    __m256 w = _mm256_loadu_ps(&filter_FP[filter_subscript]);

                    temp = _mm256_fmadd_ps(s, w, temp);
                    temp2 = _mm256_fmadd_ps(s2, w, temp2);
                    temp3 = _mm256_fmadd_ps(s3, w, temp3);
                    temp4 = _mm256_fmadd_ps(s4, w, temp4);


                    // temp = _mm256_add_ps(temp, _mm256_mul_ps(s, w));
                    // temp2 = _mm256_add_ps(temp2, _mm256_mul_ps(s2, w));
                    // temp3 = _mm256_add_ps(temp3, _mm256_mul_ps(s3, w));
                    // temp4 = _mm256_add_ps(temp4, _mm256_mul_ps(s4, w));
                  }
                }
              }

              unsigned long long int out_subscript = b * (out_yx_depth) +
                y * (out_x_depth) +
                (x << left_shift(Output_depth_dim))
                + m;
              unsigned long long int out_subscript2 = b * (out_yx_depth) +
                y * (out_x_depth) +
                ((x+1) << left_shift(Output_depth_dim))
                + m;
              unsigned long long int out_subscript3 = b * (out_yx_depth) +
                y * (out_x_depth) +
                ((x+2) << left_shift(Output_depth_dim))
                + m;
              unsigned long long int out_subscript4 = b * (out_yx_depth) +
                y * (out_x_depth) +
                ((x+3) << left_shift(Output_depth_dim))
                + m;

              temp = _mm256_hadd_ps(temp, temp);
              temp = _mm256_hadd_ps(temp, temp);
              __m128 tempLo = _mm256_castps256_ps128(temp);
              __m128 tempHi = _mm256_extractf128_ps(temp, 1);
              __m128 sseSum = _mm_add_ps(tempLo, tempHi);

              float sum = _mm_cvtss_f32(sseSum);

              sum += bias;
              out_to_compare_with_FP[out_subscript] = RELU(sum);


              temp2 = _mm256_hadd_ps(temp2, temp2);
              temp2 = _mm256_hadd_ps(temp2, temp2);
              __m128 tempLo2 = _mm256_castps256_ps128(temp2);
              __m128 tempHi2 = _mm256_extractf128_ps(temp2, 1);
              __m128 sseSum2 = _mm_add_ps(tempLo2, tempHi2);

              float sum2 = _mm_cvtss_f32(sseSum2);

              sum2 += bias;
              out_to_compare_with_FP[out_subscript2] = RELU(sum2);


              temp3 = _mm256_hadd_ps(temp3, temp3);
              temp3 = _mm256_hadd_ps(temp3, temp3);
              __m128 tempLo3 = _mm256_castps256_ps128(temp3);
              __m128 tempHi3 = _mm256_extractf128_ps(temp3, 1);
              __m128 sseSum3 = _mm_add_ps(tempLo3, tempHi3);

              float sum3 = _mm_cvtss_f32(sseSum3);

              sum3 += bias;
              out_to_compare_with_FP[out_subscript3] = RELU(sum3);


              temp4 = _mm256_hadd_ps(temp4, temp4);
              temp4 = _mm256_hadd_ps(temp4, temp4);
              __m128 tempLo4 = _mm256_castps256_ps128(temp4);
              __m128 tempHi4 = _mm256_extractf128_ps(temp4, 1);
              __m128 sseSum4 = _mm_add_ps(tempLo4, tempHi4);

              float sum4 = _mm_cvtss_f32(sseSum4);

              sum4 += bias;
              out_to_compare_with_FP[out_subscript4] = RELU(sum4);

            }
          }
        }
      }
    }
  }
  #undef left_shift
  #undef RELU
  #undef m_tile
  printf("\n from optv13 x3 m3 - loop tiling %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}






// unroll d again
// v14 unroll d - using v12 & 13 funcs as base