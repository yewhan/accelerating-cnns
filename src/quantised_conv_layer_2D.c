#include "convolution_layer_2D.h"
#include <emmintrin.h>
#include <xmmintrin.h>
#include <immintrin.h>






// int optimised_layerv1_arraycopying_vectorised_Char(const unsigned char* in_Char, const signed char* filter_Char, const int* bias_array_Int, unsigned char* out_to_compare_with_Char) {


//   unsigned int filter_Char_length = Output_depth_dim * Mask_Y_dim * Mask_X_dim * Input_depth_dim;
//   signed char* filter_Char_copy = (signed char*)_mm_malloc(filter_Char_length * sizeof(signed char), 64);
//   if (filter_Char_copy == NULL) {
//     printf("\nerror with malloc allocating filter array copy");
//     exit(EXIT_FAILURE);
//   }

//   // array copying - filter_Char into form usable for vectorising m loop
//   #pragma omp for collapse(4) schedule(static)
//   for (unsigned int m = 0; m < Output_depth_dim; m += 8) {
//     for (unsigned int y = 0; y < Mask_Y_dim; y++) {
//       for (unsigned int x = 0; x < Mask_X_dim; x++) {
//         for (unsigned int d = 0; d < Input_depth_dim; d++) {
//           for (unsigned int mm = m; mm < m + 8; mm++) {
//             unsigned long long int old_subscript = mm * Mask_Y_dim * Mask_X_dim * Input_depth_dim
//               + y * Mask_X_dim * Input_depth_dim
//               + x * Input_depth_dim
//               + d;
              
//             unsigned long long int new_subscript = y * Mask_X_dim * Input_depth_dim * Output_depth_dim
//               + x * Input_depth_dim * Output_depth_dim
//               + d * Output_depth_dim
//               + mm;

//             filter_Char_copy[new_subscript] = filter_Char[old_subscript];
//           }
//         }
//       }
//     }
//   }

//   for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
//     for (unsigned int m = 0; m < Output_depth_dim; m+=32) { //channels
//       __m256i bias = _mm256_load_si256((__m256i*)&bias_array_Int[m]);
//       __m256i bias2 = _mm256_load_si256((__m256i*)&bias_array_Int[m+8]);
//       __m256i bias3 = _mm256_load_si256((__m256i*)&bias_array_Int[m+16]);
//       __m256i bias4 = _mm256_load_si256((__m256i*)&bias_array_Int[m+24]);
//       // bias = bias_array_Int[m];

//       for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
//         for (unsigned int x = 0; x < Output_X_dim; x++) {	//Output Width
//           __m256i temp = _mm256_setzero_si256(); 
//           __m256i temp2 = _mm256_setzero_si256();
//           __m256i temp3 = _mm256_setzero_si256();
//           __m256i temp4 = _mm256_setzero_si256();
//           // temp = 0;

//           for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
//             for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
//               for (unsigned int d = 0; d < Input_depth_dim; d++) {

//                 unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
//                   + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
//                   + (x * Stride_X_dim + off_x) * Input_depth_dim
//                   + d;

//                 unsigned long long int filter_subscript = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
//                   + off_x * Input_depth_dim * Output_depth_dim
//                   + d * Output_depth_dim
//                   + m;




//                 // unsigned char s = in_Char[in_subscript];
//                 // signed char w = filter_Char[filter_subscript];
//                 // temp = temp + s * w;
//               }
//             }
//           }


//           unsigned long long int out_subscript = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
//             y * (Output_depth_dim * Output_X_dim) +
//             x * Output_depth_dim
//             + m;


//           // out_to_compare_with_Char[out_subscript] = Relu_int(temp);

//         }
//       }
//     }
//   }

//   printf("\n from quantised unopt %d %d ", out_to_compare_with_Char[0], out_to_compare_with_Char[1]);
//   return 0;
  
// }






int optimised_layerv1_arraycopying_vectorised_Char(const unsigned char* in_Char, const signed char* filter_Char, const int* bias_array_Int, unsigned char* out_to_compare_with_Char) {


  unsigned int filter_Char_length = Output_depth_dim * Mask_Y_dim * Mask_X_dim * Input_depth_dim;
  signed char* filter_Char_copy = (signed char*)_mm_malloc(filter_Char_length * sizeof(signed char), 64);
  if (filter_Char_copy == NULL) {
    printf("\nerror with malloc allocating filter array copy");
    exit(EXIT_FAILURE);
  }

  // array copying - filter_Char into form usable for vectorising m loop
  // vectorise in here? filter_Char_copy[new_subscript] = _mm256_set1_epi32(*(unsigned int *) &filter_Char[old_subscript]);
  unsigned long long int new_subscript = 0;
  for (unsigned int m = 0; m < Output_depth_dim; m++) {
    for (unsigned int y = 0; y < Mask_Y_dim; y++) {
      for (unsigned int x = 0; x < Mask_X_dim; x++) {
        for (unsigned int d = 0; d < Input_depth_dim; d+=4) {
          unsigned long long int old_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
            + y * Mask_X_dim * Input_depth_dim
            + x * Input_depth_dim
            + d;
          
          filter_Char_copy[new_subscript++] = filter_Char[old_subscript];
          filter_Char_copy[new_subscript++] = filter_Char[old_subscript+1];
          filter_Char_copy[new_subscript++] = filter_Char[old_subscript+2];
          filter_Char_copy[new_subscript++] = filter_Char[old_subscript+3];
          // new_subscript++;
        }
      }
    }
  }

  // for (unsigned int y = 0; y < Mask_Y_dim; y++) {
  //   for (unsigned int x = 0; x < Mask_X_dim; x++) {
  //     for (unsigned int d = 0; d < Input_depth_dim; d++) {
  //       for (unsigned int m = 0; m < Output_depth_dim; m++) {
  //         unsigned long long int old_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
  //           + y * Mask_X_dim * Input_depth_dim
  //           + x * Input_depth_dim
  //           + d;
              
  //         unsigned long long int new_subscript = y * Mask_X_dim * Input_depth_dim * Output_depth_dim
  //           + x * Input_depth_dim * Output_depth_dim
  //           + d * Output_depth_dim
  //           + m;

  //         filter_Char_copy[new_subscript] = filter_Char[old_subscript];
  //       }
  //     }
  //   }
  // }


  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int m = 0; m < Output_depth_dim; m+=8) { //channels
      __m256i bias = _mm256_load_si256((__m256i*)&bias_array_Int[m]);
      // bias = bias_array_Int[m];

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (unsigned int x = 0; x < Output_X_dim; x++) {	//Output Width
          __m256i temp = _mm256_setzero_si256(); 
          // temp = 0;

          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=4) {

                unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + d;

                unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;

                // unsigned long long int filter_subscript = off_y * Mask_X_dim * Input_depth_dim * Output_depth_dim
                //   + off_x * Input_depth_dim * Output_depth_dim
                //   + d * Output_depth_dim
                //   + m;

                // unsigned char d0 = in_Char[in_subscript];
                // unsigned char d1 = in_Char[in_subscript + 1];
                // unsigned char d2 = in_Char[in_subscript + 2];
                // unsigned char d3 = in_Char[in_subscript + 3];

                // __m256i s = _mm256_setr_epi8(d0,d1,d2,d3,d0,d1,d2,d3,d0,d1,d2,d3,d0,d1,
                //   d2,d3,d0,d1,d2,d3,d0,d1,d2,d3,d0,d1,d2,d3,d0,d1,d2,d3);

                __m256i s = _mm256_set1_epi32(*(unsigned int *)&in_Char[in_subscript]); // loading in 4, 8bit chars (32 bit worth of data) and broadcast across s

                __m256i w = _mm256_loadu_si256((const __m256i*)&filter_Char_copy[filter_subscript]);

                __m256i inter_vec = _mm256_maddubs_epi16(s, w);
                inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                temp = _mm256_add_epi32(temp, inter_vec);
              }
            }
          }


          unsigned long long int out_subscript = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
            y * (Output_depth_dim * Output_X_dim) +
            x * Output_depth_dim
            + m;

          temp = _mm256_add_epi32(temp, bias);
          temp = _mm256_max_epi32(temp, _mm256_set1_epi32(0));

          for (int i = 0; i < 8; i++) {
            out_to_compare_with_Char[out_subscript+i] = temp[i];
          }
          // _mm256_store_si256((__m256i*)&out_to_compare_with_Char[out_subscript], temp);

        }
      }
    }
  }
  
  printf("\n from quantised array_copying vectorised %d %d ", out_to_compare_with_Char[0], out_to_compare_with_Char[1]);
  return 0;
}




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


                __m256i s = _mm256_load_si256((const __m256i*)&in_Char[in_subscript]);
                __m256i w = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript]);

                __m256i inter_vec = _mm256_maddubs_epi16(s,w);
                inter_vec = _mm256_madd_epi16(inter_vec,_mm256_set1_epi16(1));  // widen inter_vec to 8, 32byte values to prevent overflow
                temp_vec = _mm256_add_epi32(temp_vec,inter_vec);


                // unsigned char s = in_Char[in_subscript];
                // signed char w = filter_Char[filter_subscript];
                // temp = temp + s * w;
              }
            }
          }

          temp_vec = _mm256_hadd_epi32(temp_vec, temp_vec);
          temp_vec = _mm256_hadd_epi32(temp_vec, temp_vec);
          __m128i tempLo = _mm256_castsi256_si128(temp_vec);
          __m128i tempHi = _mm256_extracti128_si256(temp_vec, 1);
          __m128 sum = _mm_add_epi32(tempLo, tempHi);
          temp = _mm_cvtsi128_si32(sum);

          // __m256i temp1 = _mm256_hadd_epi32(temp_vec, temp_vec);
          // __m256i temp2 = _mm256_hadd_epi32(temp1, temp1);
          // __m128i tempLo = _mm256_castsi256_si128(temp2);
          // __m128i tempHi = _mm256_extracti128_si256(temp2, 1);
          // __m128 sum = _mm_add_epi32(tempLo, tempHi);
          // temp = _mm_cvtsi128_si32(sum);

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



// register block/ unroll x by factor of 2 (x+=2)
// 102 GFLOPS, 1.36x speedup from unopt
int optimised_layerv2_unroll_x2_Char(const unsigned char* in_Char, const signed char* filter_Char, const int* bias_array_Int, unsigned char* out_to_compare_with_Char) {

  int temp, temp2, bias;

  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int m = 0; m < Output_depth_dim; m++) { //channels
      bias = bias_array_Int[m];

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (unsigned int x = 0; x < Output_X_dim; x+=2) {	//Output Width
          __m256i temp_vec = _mm256_setzero_si256();
          __m256i temp_vec2 = _mm256_setzero_si256();

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


                __m256i s = _mm256_load_si256((const __m256i*)&in_Char[in_subscript]);
                __m256i s2 = _mm256_load_si256((const __m256i*)&in_Char[in_subscript2]);
                __m256i w = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript]);

                __m256i inter_vec = _mm256_maddubs_epi16(s,w);
                inter_vec = _mm256_madd_epi16(inter_vec,_mm256_set1_epi16(1));
                temp_vec = _mm256_add_epi32(temp_vec,inter_vec);

                __m256i inter_vec2 = _mm256_maddubs_epi16(s2,w);
                inter_vec2 = _mm256_madd_epi16(inter_vec2,_mm256_set1_epi16(1));
                temp_vec2 = _mm256_add_epi32(temp_vec2,inter_vec2);



                // unsigned char s = in_Char[in_subscript];
                // signed char w = filter_Char[filter_subscript];
                // temp = temp + s * w;
              }
            }
          }
        
          temp_vec = _mm256_hadd_epi32(temp_vec, temp_vec);
          temp_vec = _mm256_hadd_epi32(temp_vec, temp_vec);
          __m128i tempLo = _mm256_castsi256_si128(temp_vec);
          __m128i tempHi = _mm256_extracti128_si256(temp_vec, 1);
          __m128 sum = _mm_add_epi32(tempLo, tempHi);
          temp = _mm_cvtsi128_si32(sum);

          temp_vec2 = _mm256_hadd_epi32(temp_vec2, temp_vec2);
          temp_vec2 = _mm256_hadd_epi32(temp_vec2, temp_vec2);
          __m128i tempLo2 = _mm256_castsi256_si128(temp_vec2);
          __m128i tempHi2 = _mm256_extracti128_si256(temp_vec2, 1);
          __m128 sum2 = _mm_add_epi32(tempLo2, tempHi2);
          temp2 = _mm_cvtsi128_si32(sum2);



          temp += bias;
          temp2 += bias;

          unsigned long long int out_subscript = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
              y * (Output_depth_dim * Output_X_dim) +
              x * Output_depth_dim
              + m;
            unsigned long long int out_subscript2 = b * (Output_depth_dim * Output_X_dim * Output_Y_dim) +
              y * (Output_depth_dim * Output_X_dim) +
              (x+1) * Output_depth_dim
              + m;

          // temp += bias;
          out_to_compare_with_Char[out_subscript] = Relu_int(temp);
          out_to_compare_with_Char[out_subscript2] = Relu_int(temp2);

        }
      }
    }
  }

  printf("\n from quantised v2 %d %d ", out_to_compare_with_Char[0], out_to_compare_with_Char[1]);
  return 0;
}



// register block/ unroll m by factor of 2 (m+=2)
// 120 GFLOPS, 1.18x speedup, still room for improvement re. register pressure
int optimised_layerv3_unroll_m2_Char(const unsigned char* in_Char, const signed char* filter_Char, const int* bias_array_Int, unsigned char* out_to_compare_with_Char) {

  int temp, temp2, temp3, temp4, bias, bias2;

  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int m = 0; m < Output_depth_dim; m+=2) { //channels
      bias = bias_array_Int[m];
      bias2 = bias_array_Int[m+1];

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (unsigned int x = 0; x < Output_X_dim; x+=2) {	//Output Width
          __m256i temp_vec = _mm256_setzero_si256();
          __m256i temp_vec2 = _mm256_setzero_si256();
          __m256i temp_vec3 = _mm256_setzero_si256();
          __m256i temp_vec4 = _mm256_setzero_si256();

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


                __m256i s = _mm256_load_si256((const __m256i*)&in_Char[in_subscript]);
                __m256i s2 = _mm256_load_si256((const __m256i*)&in_Char[in_subscript2]);
                __m256i w = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript]);
                __m256i w2 = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript2]);

                __m256i inter_vec = _mm256_maddubs_epi16(s,w);
                inter_vec = _mm256_madd_epi16(inter_vec,_mm256_set1_epi16(1));
                temp_vec = _mm256_add_epi32(temp_vec,inter_vec);

                __m256i inter_vec2 = _mm256_maddubs_epi16(s2,w);
                inter_vec2 = _mm256_madd_epi16(inter_vec2,_mm256_set1_epi16(1));
                temp_vec2 = _mm256_add_epi32(temp_vec2,inter_vec2);

                __m256i inter_vec3 = _mm256_maddubs_epi16(s,w2);
                inter_vec3 = _mm256_madd_epi16(inter_vec3,_mm256_set1_epi16(1));
                temp_vec3 = _mm256_add_epi32(temp_vec3,inter_vec3);

                __m256i inter_vec4 = _mm256_maddubs_epi16(s2,w2);
                inter_vec4 = _mm256_madd_epi16(inter_vec4,_mm256_set1_epi16(1));
                temp_vec4 = _mm256_add_epi32(temp_vec4,inter_vec4);



                // unsigned char s = in_Char[in_subscript];
                // signed char w = filter_Char[filter_subscript];
                // temp = temp + s * w;
              }
            }
          }
        
          temp_vec = _mm256_hadd_epi32(temp_vec, temp_vec);
          temp_vec = _mm256_hadd_epi32(temp_vec, temp_vec);
          __m128i tempLo = _mm256_castsi256_si128(temp_vec);
          __m128i tempHi = _mm256_extracti128_si256(temp_vec, 1);
          __m128 sum = _mm_add_epi32(tempLo, tempHi);
          temp = _mm_cvtsi128_si32(sum);

          temp_vec2 = _mm256_hadd_epi32(temp_vec2, temp_vec2);
          temp_vec2 = _mm256_hadd_epi32(temp_vec2, temp_vec2);
          __m128i tempLo2 = _mm256_castsi256_si128(temp_vec2);
          __m128i tempHi2 = _mm256_extracti128_si256(temp_vec2, 1);
          __m128 sum2 = _mm_add_epi32(tempLo2, tempHi2);
          temp2 = _mm_cvtsi128_si32(sum2);
        
          temp_vec3 = _mm256_hadd_epi32(temp_vec3, temp_vec3);
          temp_vec3 = _mm256_hadd_epi32(temp_vec3, temp_vec3);
          __m128i tempLo3 = _mm256_castsi256_si128(temp_vec3);
          __m128i tempHi3 = _mm256_extracti128_si256(temp_vec3, 1);
          __m128 sum3 = _mm_add_epi32(tempLo3, tempHi3);
          temp3 = _mm_cvtsi128_si32(sum3);

          temp_vec4 = _mm256_hadd_epi32(temp_vec4, temp_vec4);
          temp_vec4 = _mm256_hadd_epi32(temp_vec4, temp_vec4);
          __m128i tempLo4 = _mm256_castsi256_si128(temp_vec4);
          __m128i tempHi4 = _mm256_extracti128_si256(temp_vec4, 1);
          __m128 sum4 = _mm_add_epi32(tempLo4, tempHi4);
          temp4 = _mm_cvtsi128_si32(sum4);


          temp += bias;
          temp2 += bias;
          temp3 += bias2;
          temp4 += bias2;

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


          out_to_compare_with_Char[out_subscript] = Relu_int(temp);
          out_to_compare_with_Char[out_subscript2] = Relu_int(temp2);

          out_to_compare_with_Char[out_subscript3] = Relu_int(temp3);
          out_to_compare_with_Char[out_subscript4] = Relu_int(temp4);

        }
      }
    }
  }

  printf("\n from quantised v3 %d %d ", out_to_compare_with_Char[0], out_to_compare_with_Char[1]);
  return 0;
}



// reduce general register pressure in d loop
// 123 GFLOPS, 1.03x speedup
int optimised_layerv4_general_register_pressure_d_Char(const unsigned char* in_Char, const signed char* filter_Char, const int* bias_array_Int, unsigned char* out_to_compare_with_Char) {

  int temp, temp2, temp3, temp4, bias, bias2;

  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
    for (unsigned int m = 0; m < Output_depth_dim; m+=2) { //channels
      bias = bias_array_Int[m];
      bias2 = bias_array_Int[m+1];

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        for (unsigned int x = 0; x < Output_X_dim; x+=2) {	//Output Width
          __m256i temp_vec = _mm256_setzero_si256();
          __m256i temp_vec2 = _mm256_setzero_si256();
          __m256i temp_vec3 = _mm256_setzero_si256();
          __m256i temp_vec4 = _mm256_setzero_si256();

          for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
            for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
              for (unsigned int d = 0; d < Input_depth_dim; d+=32) {

                unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                + (x * Stride_X_dim + off_x) * Input_depth_dim
                + d;

              // unsigned long long int in_subscript2 = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
              //   + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
              //   + ((x+1) * Stride_X_dim + off_x) * Input_depth_dim
              //   + d;

                

                unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                + off_y * Mask_X_dim * Input_depth_dim
                + off_x * Input_depth_dim
                + d;

                // unsigned long long int filter_subscript2 = (m+1) * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                // + off_y * Mask_X_dim * Input_depth_dim
                // + off_x * Input_depth_dim
                // + d;


                // __m256i s = _mm256_load_si256((const __m256i*)&in_Char[in_subscript]);
                // __m256i w = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript]);
                // temp_vec = _mm256_dpbsud_epi32(s, w, temp_vec);


                __m256i s = _mm256_load_si256((const __m256i*)&in_Char[in_subscript]);
                __m256i s2 = _mm256_load_si256((const __m256i*)&in_Char[in_subscript + Input_depth_dim]);
                __m256i w = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript]);
                __m256i w2 = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript + (Mask_Y_dim * Mask_X_dim * Input_depth_dim)]);

                __m256i inter_vec = _mm256_maddubs_epi16(s,w);
                inter_vec = _mm256_madd_epi16(inter_vec,_mm256_set1_epi16(1));
                temp_vec = _mm256_add_epi32(temp_vec,inter_vec);

                __m256i inter_vec2 = _mm256_maddubs_epi16(s2,w);
                inter_vec2 = _mm256_madd_epi16(inter_vec2,_mm256_set1_epi16(1));
                temp_vec2 = _mm256_add_epi32(temp_vec2,inter_vec2);

                __m256i inter_vec3 = _mm256_maddubs_epi16(s,w2);
                inter_vec3 = _mm256_madd_epi16(inter_vec3,_mm256_set1_epi16(1));
                temp_vec3 = _mm256_add_epi32(temp_vec3,inter_vec3);

                __m256i inter_vec4 = _mm256_maddubs_epi16(s2,w2);
                inter_vec4 = _mm256_madd_epi16(inter_vec4,_mm256_set1_epi16(1));
                temp_vec4 = _mm256_add_epi32(temp_vec4,inter_vec4);



                // unsigned char s = in_Char[in_subscript];
                // signed char w = filter_Char[filter_subscript];
                // temp = temp + s * w;
              }
            }
          }
        
          temp_vec = _mm256_hadd_epi32(temp_vec, temp_vec);
          temp_vec = _mm256_hadd_epi32(temp_vec, temp_vec);
          __m128i tempLo = _mm256_castsi256_si128(temp_vec);
          __m128i tempHi = _mm256_extracti128_si256(temp_vec, 1);
          __m128 sum = _mm_add_epi32(tempLo, tempHi);
          temp = _mm_cvtsi128_si32(sum);

          temp_vec2 = _mm256_hadd_epi32(temp_vec2, temp_vec2);
          temp_vec2 = _mm256_hadd_epi32(temp_vec2, temp_vec2);
          __m128i tempLo2 = _mm256_castsi256_si128(temp_vec2);
          __m128i tempHi2 = _mm256_extracti128_si256(temp_vec2, 1);
          __m128 sum2 = _mm_add_epi32(tempLo2, tempHi2);
          temp2 = _mm_cvtsi128_si32(sum2);
        
          temp_vec3 = _mm256_hadd_epi32(temp_vec3, temp_vec3);
          temp_vec3 = _mm256_hadd_epi32(temp_vec3, temp_vec3);
          __m128i tempLo3 = _mm256_castsi256_si128(temp_vec3);
          __m128i tempHi3 = _mm256_extracti128_si256(temp_vec3, 1);
          __m128 sum3 = _mm_add_epi32(tempLo3, tempHi3);
          temp3 = _mm_cvtsi128_si32(sum3);

          temp_vec4 = _mm256_hadd_epi32(temp_vec4, temp_vec4);
          temp_vec4 = _mm256_hadd_epi32(temp_vec4, temp_vec4);
          __m128i tempLo4 = _mm256_castsi256_si128(temp_vec4);
          __m128i tempHi4 = _mm256_extracti128_si256(temp_vec4, 1);
          __m128 sum4 = _mm_add_epi32(tempLo4, tempHi4);
          temp4 = _mm_cvtsi128_si32(sum4);


          temp += bias;
          temp2 += bias;
          temp3 += bias2;
          temp4 += bias2;

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


          out_to_compare_with_Char[out_subscript] = Relu_int(temp);
          out_to_compare_with_Char[out_subscript2] = Relu_int(temp2);

          out_to_compare_with_Char[out_subscript3] = Relu_int(temp3);
          out_to_compare_with_Char[out_subscript4] = Relu_int(temp4);

        }
      }
    }
  }

  printf("\n from quantised v4 %d %d ", out_to_compare_with_Char[0], out_to_compare_with_Char[1]);
  return 0;
}



// reduce general register pressure in d loop
// 123 GFLOPS, 1.03x speedup
int optimised_layerv5_loop_tiling_d_Char(const unsigned char* in_Char, const signed char* filter_Char, const int* bias_array_Int, unsigned char* out_to_compare_with_Char) {

  #define tile 32

  int temp, temp2, temp3, temp4, bias, bias2;
  for (unsigned int dd = 0; dd < Input_depth_dim; dd+=tile) {
    for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
      for (unsigned int m = 0; m < Output_depth_dim; m+=2) { //channels
        bias = bias_array_Int[m];
        bias2 = bias_array_Int[m+1];

        for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
          for (unsigned int x = 0; x < Output_X_dim; x+=2) {	//Output Width
            __m256i temp_vec = _mm256_setzero_si256();
            __m256i temp_vec2 = _mm256_setzero_si256();
            __m256i temp_vec3 = _mm256_setzero_si256();
            __m256i temp_vec4 = _mm256_setzero_si256();

            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                for (unsigned int d = dd; d < dd + tile; d+=32) {

                  unsigned long long int in_subscript = b * (Input_Y_dim * Input_X_dim * Input_depth_dim)
                  + (y * Stride_Y_dim + off_y) * Input_X_dim * Input_depth_dim
                  + (x * Stride_X_dim + off_x) * Input_depth_dim
                  + d;


                  unsigned long long int filter_subscript = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim
                  + off_y * Mask_X_dim * Input_depth_dim
                  + off_x * Input_depth_dim
                  + d;


                __m256i s = _mm256_load_si256((const __m256i*)&in_Char[in_subscript]);
                __m256i s2 = _mm256_load_si256((const __m256i*)&in_Char[in_subscript + Input_depth_dim]);
                __m256i w = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript]);
                __m256i w2 = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript + (Mask_Y_dim * Mask_X_dim * Input_depth_dim)]);

                __m256i inter_vec = _mm256_maddubs_epi16(s,w);
                inter_vec = _mm256_madd_epi16(inter_vec,_mm256_set1_epi16(1));
                temp_vec = _mm256_add_epi32(temp_vec,inter_vec);

                __m256i inter_vec2 = _mm256_maddubs_epi16(s2,w);
                inter_vec2 = _mm256_madd_epi16(inter_vec2,_mm256_set1_epi16(1));
                temp_vec2 = _mm256_add_epi32(temp_vec2,inter_vec2);

                __m256i inter_vec3 = _mm256_maddubs_epi16(s,w2);
                inter_vec3 = _mm256_madd_epi16(inter_vec3,_mm256_set1_epi16(1));
                temp_vec3 = _mm256_add_epi32(temp_vec3,inter_vec3);

                __m256i inter_vec4 = _mm256_maddubs_epi16(s2,w2);
                inter_vec4 = _mm256_madd_epi16(inter_vec4,_mm256_set1_epi16(1));
                temp_vec4 = _mm256_add_epi32(temp_vec4,inter_vec4);




                  // unsigned char s = in_Char[in_subscript];
                  // signed char w = filter_Char[filter_subscript];
                  // temp = temp + s * w;
                }
              }
            }
          
            temp_vec = _mm256_hadd_epi32(temp_vec, temp_vec);
            temp_vec = _mm256_hadd_epi32(temp_vec, temp_vec);
            __m128i tempLo = _mm256_castsi256_si128(temp_vec);
            __m128i tempHi = _mm256_extracti128_si256(temp_vec, 1);
            __m128 sum = _mm_add_epi32(tempLo, tempHi);
            temp = _mm_cvtsi128_si32(sum);

            temp_vec2 = _mm256_hadd_epi32(temp_vec2, temp_vec2);
            temp_vec2 = _mm256_hadd_epi32(temp_vec2, temp_vec2);
            __m128i tempLo2 = _mm256_castsi256_si128(temp_vec2);
            __m128i tempHi2 = _mm256_extracti128_si256(temp_vec2, 1);
            __m128 sum2 = _mm_add_epi32(tempLo2, tempHi2);
            temp2 = _mm_cvtsi128_si32(sum2);
          
            temp_vec3 = _mm256_hadd_epi32(temp_vec3, temp_vec3);
            temp_vec3 = _mm256_hadd_epi32(temp_vec3, temp_vec3);
            __m128i tempLo3 = _mm256_castsi256_si128(temp_vec3);
            __m128i tempHi3 = _mm256_extracti128_si256(temp_vec3, 1);
            __m128 sum3 = _mm_add_epi32(tempLo3, tempHi3);
            temp3 = _mm_cvtsi128_si32(sum3);

            temp_vec4 = _mm256_hadd_epi32(temp_vec4, temp_vec4);
            temp_vec4 = _mm256_hadd_epi32(temp_vec4, temp_vec4);
            __m128i tempLo4 = _mm256_castsi256_si128(temp_vec4);
            __m128i tempHi4 = _mm256_extracti128_si256(temp_vec4, 1);
            __m128 sum4 = _mm_add_epi32(tempLo4, tempHi4);
            temp4 = _mm_cvtsi128_si32(sum4);


            temp += bias;
            temp2 += bias;
            temp3 += bias2;
            temp4 += bias2;

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


            out_to_compare_with_Char[out_subscript] = Relu_int(temp);
            out_to_compare_with_Char[out_subscript2] = Relu_int(temp2);

            out_to_compare_with_Char[out_subscript3] = Relu_int(temp3);
            out_to_compare_with_Char[out_subscript4] = Relu_int(temp4);

          }
        }
      }
    }
  }

  printf("\n from quantised v5 %d %d ", out_to_compare_with_Char[0], out_to_compare_with_Char[1]);
  return 0;
}





// 40 GFLOPS, 20x speedup from non-quantised - test again after reboot just in-case
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