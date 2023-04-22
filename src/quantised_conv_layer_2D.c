#include "convolution_layer_2D.h"
#include <emmintrin.h>
#include <xmmintrin.h>



// vectorised d loop, still room for improvement re. intrinsics used
// 75 GFLOPS, 1.875x speedup from unopt
int optimised_layerv1_vectorised_Char(const unsigned char* in_Char, const signed char* filter_Char, const int* bias_array_Int, unsigned char* out_to_compare_with_Char) {

  int temp, bias;

  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int m = 0; m < Output_depth_dim; m++) { //channels
      bias = bias_array_Int[m];

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (unsigned int x = 0; x < Output_X_dim; x++) {	//Output Width
          __m256i temp_vec = _mm256_setzero_si256();

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


                __m256i s_l = _mm256_cvtepu8_epi16(_mm_load_si128((const __m128i*)&in_Char[in_subscript]));
                __m256i s_h = _mm256_cvtepu8_epi16(_mm_load_si128((const __m128i*)&in_Char[in_subscript+16]));
                __m256i w_l = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*)&filter_Char[filter_subscript]));
                __m256i w_h = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i*)&filter_Char[filter_subscript+16]));
                __m256i inter_vec = _mm256_madd_epi16(s_l, w_l);
                inter_vec = _mm256_add_epi32(inter_vec, _mm256_madd_epi16(s_h, w_h));
                temp_vec = _mm256_add_epi32(temp_vec, inter_vec);


                // unsigned char s = in_Char[in_subscript];
                // signed char w = filter_Char[filter_subscript];
                // temp = temp + s * w;
              }
            }
          }
        
          __m256i temp1 = _mm256_hadd_epi32(temp_vec, temp_vec);
          __m256i temp2 = _mm256_hadd_epi32(temp1, temp1);
          __m128i tempLo = _mm256_castsi256_si128(temp2);
          __m128i tempHi = _mm256_extracti128_si256(temp2, 1);
          __m128 sum = _mm_add_epi32(tempLo, tempHi);
          temp = _mm_cvtsi128_si32(sum);

          temp += bias;

          unsigned long long int out_subscript = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
              y * (Output_depth_dim * Output_X_dim) +
              x * Output_depth_dim
              + m;

          // temp += bias;
          out_to_compare_with_Char[out_subscript] = Relu_int(temp);

        }
      }
    }
  }

  printf("\n from quantised vect_d %d %d ", out_to_compare_with_Char[0], out_to_compare_with_Char[1]);
  return 0;
}





// 40 GFLOPS - test again after reboot just in-case
int unoptimized_layer_Char(const unsigned char* in_Char, const signed char* filter_Char, const int* bias_array_Int, unsigned char* out_to_compare_with_Char) {

  int temp, bias;

  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int m = 0; m < Output_depth_dim; m++) { //channels
      for (unsigned int od = 0; od < 1; od++) {
        for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
          for (unsigned int x = 0; x < Output_X_dim; x++) {	//Output Width
            bias = bias_array_Int[m];
            temp = 0;
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

                  unsigned char s = in_Char[in_subscript];
                  signed char w = filter_Char[filter_subscript];
                  temp = temp + s * w;


                }
              }
            }


            unsigned long long int out_subscript = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
              y * (Output_depth_dim * Output_X_dim) +
              x * Output_depth_dim
              + m;

            temp += bias;
            out_to_compare_with_Char[out_subscript] = Relu_int(temp);

          }
        }
      }
    }
  }

  printf("\n from quantised unopt %d %d ", out_to_compare_with_Char[0], out_to_compare_with_Char[1]);
  return 0;
}


int Relu_int(const int temp) {

  // return temp > 0.0f ? temp : 0.0f;
  if (temp < 0)
    return 0;
  else
    return temp;

}