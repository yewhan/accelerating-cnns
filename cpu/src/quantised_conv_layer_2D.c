#include "quantised_conv_layer_2D.h"
#include <emmintrin.h>
#include <tmmintrin.h>
#include <xmmintrin.h>
#include <immintrin.h>



// profiling function
int profile_Char(const unsigned char* in_Char, const signed char* filter_Char, const int* bias_array_Int, unsigned char* out_to_compare_with_Char) {
  #define left_shift(x) (__builtin_ctz(x))  // inline function effectively using inline assembly to return num of trailing zeros (i.e. 128 returns 7, 2 * 128 == 2 << 7)
  #define RELU(x) ((x) < 0 ? 0 : (x)) // return x if greater than 0, else return 0

  // float bias, bias2, bias3, bias4;
  // __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

  const unsigned long long int in_l_shift = left_shift(Input_depth_dim);
  const unsigned long long int out_l_shift = left_shift(Output_depth_dim);

  const unsigned long long int in_yx_depth = Input_Y_dim * Input_X_dim << in_l_shift;
  const unsigned long long int in_x_depth = Input_X_dim  << in_l_shift;

  const unsigned long long int mask_yx_depth = Mask_Y_dim * Mask_X_dim << in_l_shift;
  const unsigned long long int mask_x_depth = Mask_X_dim << in_l_shift;

  const unsigned long long int out_yx_depth = Output_X_dim * Output_Y_dim << out_l_shift;
  const unsigned long long int out_x_depth = Output_X_dim << out_l_shift;

  // const unsigned int m_bound = (Output_depth_dim/ 2) * 2;
  const unsigned int x_bound = (Input_X_dim/ 4) * 4;
  const unsigned int d_bound = (Input_depth_dim / 64) * 64;
  
  unsigned int x, d;

  #pragma omp parallel for private(x) shared(in_yx_depth, in_x_depth, mask_yx_depth, mask_x_depth, out_yx_depth, out_x_depth) \
    default(shared) collapse(2) schedule(static)
  {
    for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
      const unsigned long long int b_depth = b * in_yx_depth;
      const unsigned long long int b_out = b * out_yx_depth;

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        const unsigned long long int y_stride = y * Stride_Y_dim;
        const unsigned long long int y_out = y * out_x_depth;

        for (unsigned int m = 0; m < Output_depth_dim; m+=2) { //channels
          const unsigned long long int m_mask = m * mask_yx_depth;

          const int bias = bias_array_Int[m];
          const int bias2 = bias_array_Int[m+1];

          for (x = 0; x < x_bound; x+=4) {	//Output Width
            const unsigned long long int x_stride = x * Stride_X_dim;
            int temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

            __m256i temp_vec = _mm256_setzero_si256();
            __m256i temp_vec2 = _mm256_setzero_si256();
            __m256i temp_vec3 = _mm256_setzero_si256();
            __m256i temp_vec4 = _mm256_setzero_si256();
            __m256i temp_vec5 = _mm256_setzero_si256();
            __m256i temp_vec6 = _mm256_setzero_si256();
            __m256i temp_vec7 = _mm256_setzero_si256();
            __m256i temp_vec8 = _mm256_setzero_si256();

            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              const unsigned long long int off_y_stride = (y_stride + off_y) * in_x_depth;
              const unsigned long long int off_y_mask = off_y * mask_x_depth;

              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                const unsigned long long int off_x_stride = (x_stride + off_x) << in_l_shift;
                const unsigned long long int off_x_depth = off_x << in_l_shift;

                for (d = 0; d < d_bound; d+=64) {

                  unsigned long long int in_subscript = b_depth
                    + off_y_stride
                    + off_x_stride
                    + d;

                  unsigned long long int filter_subscript = m_mask
                    + off_y_mask
                    + off_x_depth
                    + d;

                   __m256i s = _mm256_load_si256(( __m256i*)&in_Char[in_subscript]);
                   __m256i s2 = _mm256_load_si256(( __m256i*)&in_Char[in_subscript + Input_depth_dim]);
                   __m256i s3 = _mm256_load_si256(( __m256i*)&in_Char[in_subscript + (2 << in_l_shift)]);
                   __m256i s4 = _mm256_load_si256(( __m256i*)&in_Char[in_subscript + (3 << in_l_shift)]);

                   __m256i w = _mm256_load_si256(( __m256i*)&filter_Char[filter_subscript]);
                   __m256i w2 = _mm256_load_si256(( __m256i*)&filter_Char[filter_subscript + mask_yx_depth]);


                  __m256i inter_vec = _mm256_maddubs_epi16(s,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec = _mm256_add_epi32(temp_vec, inter_vec);

                  __m256i inter_vec2 = _mm256_maddubs_epi16(s2,w);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec2 = _mm256_add_epi32(temp_vec2, inter_vec2);

                  inter_vec = _mm256_maddubs_epi16(s3,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec3 = _mm256_add_epi32(temp_vec3, inter_vec);

                  inter_vec2 = _mm256_maddubs_epi16(s4, w);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec4 = _mm256_add_epi32(temp_vec4, inter_vec2);

                  inter_vec = _mm256_maddubs_epi16(s,w2);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec5 = _mm256_add_epi32(temp_vec5, inter_vec);

                  inter_vec2 = _mm256_maddubs_epi16(s2, w2);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec6 = _mm256_add_epi32(temp_vec6, inter_vec2);

                  inter_vec = _mm256_maddubs_epi16(s3,w2);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec7 = _mm256_add_epi32(temp_vec7, inter_vec);

                  inter_vec2 = _mm256_maddubs_epi16(s4, w2);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec8 = _mm256_add_epi32(temp_vec8, inter_vec2);

                  // d+32 ------------------------------------------------------------
                  // |
                  // |
                  // v

                  // in_subscript += 32;

                  // filter_subscript += 32;

                  s = _mm256_load_si256((const __m256i*)&in_Char[in_subscript + 32]);
                  s2 = _mm256_load_si256((const __m256i*)&in_Char[in_subscript + 32 + Input_depth_dim]);
                  s3 = _mm256_load_si256((const __m256i*)&in_Char[in_subscript + 32 + (2 << in_l_shift)]);
                  s4 = _mm256_load_si256((const __m256i*)&in_Char[in_subscript + 32 + (3 << in_l_shift)]);

                  w = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript + 32]);
                  w2 = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript + 32 + mask_yx_depth]);


                  inter_vec = _mm256_maddubs_epi16(s,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec = _mm256_add_epi32(temp_vec, inter_vec);

                  inter_vec2 = _mm256_maddubs_epi16(s2,w);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec2 = _mm256_add_epi32(temp_vec2, inter_vec2);

                  inter_vec = _mm256_maddubs_epi16(s3,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec3 = _mm256_add_epi32(temp_vec3, inter_vec);

                  inter_vec2 = _mm256_maddubs_epi16(s4, w);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec4 = _mm256_add_epi32(temp_vec4, inter_vec2);

                  inter_vec = _mm256_maddubs_epi16(s,w2);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec5 = _mm256_add_epi32(temp_vec5, inter_vec);

                  inter_vec2 = _mm256_maddubs_epi16(s2, w2);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec6 = _mm256_add_epi32(temp_vec6, inter_vec2);

                  inter_vec = _mm256_maddubs_epi16(s3,w2);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec7 = _mm256_add_epi32(temp_vec7, inter_vec);

                  inter_vec2 = _mm256_maddubs_epi16(s4, w2);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec8 = _mm256_add_epi32(temp_vec8, inter_vec2);
                }
                // fallback loop in case Input_depth < 64
                for (; d < Input_depth_dim; d+=32) {
                  unsigned long long int in_subscript = b_depth
                    + off_y_stride
                    + off_x_stride
                    + d;

                  unsigned long long int filter_subscript = m_mask
                    + off_y_mask
                    + off_x_depth
                    + d;

                   __m256i s = _mm256_load_si256(( __m256i*)&in_Char[in_subscript]);
                   __m256i s2 = _mm256_load_si256(( __m256i*)&in_Char[in_subscript + Input_depth_dim]);
                   __m256i s3 = _mm256_load_si256(( __m256i*)&in_Char[in_subscript + (2 << in_l_shift)]);
                   __m256i s4 = _mm256_load_si256(( __m256i*)&in_Char[in_subscript + (3 << in_l_shift)]);

                   __m256i w = _mm256_load_si256(( __m256i*)&filter_Char[filter_subscript]);
                   __m256i w2 = _mm256_load_si256(( __m256i*)&filter_Char[filter_subscript + mask_yx_depth]);


                  __m256i inter_vec = _mm256_maddubs_epi16(s,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec = _mm256_add_epi32(temp_vec, inter_vec);

                  __m256i inter_vec2 = _mm256_maddubs_epi16(s2,w);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec2 = _mm256_add_epi32(temp_vec2, inter_vec2);

                  inter_vec = _mm256_maddubs_epi16(s3,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec3 = _mm256_add_epi32(temp_vec3, inter_vec);

                  inter_vec2 = _mm256_maddubs_epi16(s4, w);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec4 = _mm256_add_epi32(temp_vec4, inter_vec2);

                  inter_vec = _mm256_maddubs_epi16(s,w2);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec5 = _mm256_add_epi32(temp_vec5, inter_vec);

                  inter_vec2 = _mm256_maddubs_epi16(s2, w2);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec6 = _mm256_add_epi32(temp_vec6, inter_vec2);

                  inter_vec = _mm256_maddubs_epi16(s3,w2);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec7 = _mm256_add_epi32(temp_vec7, inter_vec);

                  inter_vec2 = _mm256_maddubs_epi16(s4, w2);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec8 = _mm256_add_epi32(temp_vec8, inter_vec2);
                }
              }
            }

            unsigned long long int out_subscript = b_out
              + y_out
              + (x << out_l_shift)
              + m;
            // unsigned long long int out_subscript2 = out_subscript + Output_depth_dim;
            // unsigned long long int out_subscript3 = out_subscript + (2 << out_l_shift);
            // unsigned long long int out_subscript4 = out_subscript + (3 << out_l_shift);




            // **************** optimised hadd ****************
            __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec), _mm256_extracti128_si256(temp_vec, 1));
            __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
            __m128i sum64 = _mm_add_epi32(hi64, sum128);
            __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            __m128i sum32 = _mm_add_epi32(sum64, hi32);
            temp = _mm_cvtsi128_si32(sum32);
            temp += bias;
            out_to_compare_with_Char[out_subscript] = RELU(temp);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec2), _mm256_extracti128_si256(temp_vec2, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp2 = _mm_cvtsi128_si32(sum32);
            temp2 += bias;
            out_to_compare_with_Char[out_subscript + (1 << out_l_shift)] = RELU(temp2);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec3), _mm256_extracti128_si256(temp_vec3, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp3 = _mm_cvtsi128_si32(sum32);
            temp3 += bias;
            out_to_compare_with_Char[out_subscript + (2 << out_l_shift)] = RELU(temp3);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec4), _mm256_extracti128_si256(temp_vec4, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp4 = _mm_cvtsi128_si32(sum32);
            temp4 += bias;
            out_to_compare_with_Char[out_subscript + (3 << out_l_shift)] = RELU(temp4);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec5), _mm256_extracti128_si256(temp_vec5, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp5 = _mm_cvtsi128_si32(sum32);
            temp5 += bias2;
            out_to_compare_with_Char[out_subscript+1] = RELU(temp5);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec6), _mm256_extracti128_si256(temp_vec6, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp6 = _mm_cvtsi128_si32(sum32);
            temp6 += bias2;
            out_to_compare_with_Char[out_subscript+1+ (1 << out_l_shift)] = RELU(temp6);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec7), _mm256_extracti128_si256(temp_vec7, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp7 = _mm_cvtsi128_si32(sum32);
            temp7 += bias2;
            out_to_compare_with_Char[out_subscript+1+ (2 << out_l_shift)] = RELU(temp7);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec8), _mm256_extracti128_si256(temp_vec8, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp8 = _mm_cvtsi128_si32(sum32);
            temp8 += bias2;
            out_to_compare_with_Char[out_subscript+1+ (3 << out_l_shift)] = RELU(temp8);
          }
          // fallback loop
          for (;x < Output_X_dim; x++) {	//Output Width
            const unsigned long long int x_stride = x * Stride_X_dim;
            int temp, temp2;

            __m256i temp_vec = _mm256_setzero_si256();
            __m256i temp_vec2 = _mm256_setzero_si256();

            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              const unsigned long long int off_y_stride = (y_stride + off_y) * in_x_depth;
              const unsigned long long int off_y_mask = off_y * mask_x_depth;

              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                const unsigned long long int off_x_stride = (x_stride + off_x) << in_l_shift;
                const unsigned long long int off_x_depth = off_x << in_l_shift;

                for (unsigned int d = 0; d < Input_depth_dim; d+=64) {

                  unsigned long long int in_subscript = b_depth
                    + off_y_stride
                    + off_x_stride
                    + d;

                  unsigned long long int filter_subscript = m_mask
                    + off_y_mask
                    + off_x_depth
                    + d;

                   __m256i s = _mm256_load_si256(( __m256i*)&in_Char[in_subscript]);

                   __m256i w = _mm256_load_si256(( __m256i*)&filter_Char[filter_subscript]);
                   __m256i w2 = _mm256_load_si256(( __m256i*)&filter_Char[filter_subscript + mask_yx_depth]);


                  __m256i inter_vec = _mm256_maddubs_epi16(s,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec = _mm256_add_epi32(temp_vec, inter_vec);

                  __m256i inter_vec2 = _mm256_maddubs_epi16(s,w2);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec2 = _mm256_add_epi32(temp_vec2, inter_vec2);

                  // d+32 ------------------------------------------------------------
                  // |
                  // |
                  // v

                  // in_subscript += 32;

                  // filter_subscript += 32;

                  s = _mm256_load_si256((const __m256i*)&in_Char[in_subscript + 32]);

                  w = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript + 32]);
                  w2 = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript + 32 + mask_yx_depth]);


                  inter_vec = _mm256_maddubs_epi16(s,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec = _mm256_add_epi32(temp_vec, inter_vec);

                  inter_vec2 = _mm256_maddubs_epi16(s,w2);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec2 = _mm256_add_epi32(temp_vec2, inter_vec2);
                }
              }
            }

            unsigned long long int out_subscript = b_out
              + y_out
              + (x << out_l_shift)
              + m;




            // **************** optimised hadd ****************
            __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec), _mm256_extracti128_si256(temp_vec, 1));
            __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
            __m128i sum64 = _mm_add_epi32(hi64, sum128);
            __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            __m128i sum32 = _mm_add_epi32(sum64, hi32);
            temp = _mm_cvtsi128_si32(sum32);
            temp += bias;
            out_to_compare_with_Char[out_subscript] = RELU(temp);


            __m128i sum128_2 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec2), _mm256_extracti128_si256(temp_vec2, 1));
            __m128i hi64_2 = _mm_unpackhi_epi64(sum128_2, sum128_2);
            __m128i sum64_2 = _mm_add_epi32(hi64_2, sum128_2);
            __m128i hi32_2 = _mm_shuffle_epi32(sum64_2, _MM_SHUFFLE(2,3,0,1));
            __m128i sum32_2 = _mm_add_epi32(sum64_2, hi32_2);
            temp2 = _mm_cvtsi128_si32(sum32_2);
            temp2 += bias2;
            out_to_compare_with_Char[out_subscript + 1] = RELU(temp2);

          }
        }
      }
    }
  }
  // #undef left_shift
  #undef RELU
  printf("\n from quantised opt m2x4 %d %d ", out_to_compare_with_Char[0], out_to_compare_with_Char[1]);
  return 0;
}









// generalised (final) functions

// this is assumed to be used as a non-initial convolution layer. thus the input and output depths are powers of 2.
// d has been unrolled by a factor of 2
// ~749 GFLOPS without hitting fallback loops, 642 when using squeezenet dims & x y = 57
int opt_m2x4_deep_AVX_d64_Char(const unsigned char* in_Char, const signed char* filter_Char, const int* bias_array_Int, unsigned char* out_to_compare_with_Char) {
  #define left_shift(x) (__builtin_ctz(x))  // inline function effectively using inline assembly to return num of trailing zeros (i.e. 128 returns 7, 2 * 128 == 2 << 7)
  #define RELU(x) ((x) < 0 ? 0 : (x)) // return x if greater than 0, else return 0

  // float bias, bias2, bias3, bias4;
  // __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

  const unsigned long long int in_l_shift = left_shift(Input_depth_dim);
  const unsigned long long int out_l_shift = left_shift(Output_depth_dim);

  const unsigned long long int in_yx_depth = Input_Y_dim * Input_X_dim << in_l_shift;
  const unsigned long long int in_x_depth = Input_X_dim  << in_l_shift;

  const unsigned long long int mask_yx_depth = Mask_Y_dim * Mask_X_dim << in_l_shift;
  const unsigned long long int mask_x_depth = Mask_X_dim << in_l_shift;

  const unsigned long long int out_yx_depth = Output_X_dim * Output_Y_dim << out_l_shift;
  const unsigned long long int out_x_depth = Output_X_dim << out_l_shift;

  // const unsigned int m_bound = (Output_depth_dim/ 2) * 2;
  const unsigned int x_bound = (Input_X_dim/ 4) * 4;
  const unsigned int d_bound = (Input_depth_dim / 64) * 64;
  
  unsigned int x, d;

  #pragma omp parallel for private(x) shared(in_yx_depth, in_x_depth, mask_yx_depth, mask_x_depth, out_yx_depth, out_x_depth) \
    default(shared) collapse(2) schedule(static)
  {
    for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
      const unsigned long long int b_depth = b * in_yx_depth;
      const unsigned long long int b_out = b * out_yx_depth;

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        const unsigned long long int y_stride = y * Stride_Y_dim;
        const unsigned long long int y_out = y * out_x_depth;

        for (unsigned int m = 0; m < Output_depth_dim; m+=2) { //channels
          const unsigned long long int m_mask = m * mask_yx_depth;

          const int bias = bias_array_Int[m];
          const int bias2 = bias_array_Int[m+1];

          for (x = 0; x < x_bound; x+=4) {	//Output Width
            const unsigned long long int x_stride = x * Stride_X_dim;
            int temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

            __m256i temp_vec = _mm256_setzero_si256();
            __m256i temp_vec2 = _mm256_setzero_si256();
            __m256i temp_vec3 = _mm256_setzero_si256();
            __m256i temp_vec4 = _mm256_setzero_si256();
            __m256i temp_vec5 = _mm256_setzero_si256();
            __m256i temp_vec6 = _mm256_setzero_si256();
            __m256i temp_vec7 = _mm256_setzero_si256();
            __m256i temp_vec8 = _mm256_setzero_si256();

            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              const unsigned long long int off_y_stride = (y_stride + off_y) * in_x_depth;
              const unsigned long long int off_y_mask = off_y * mask_x_depth;

              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                const unsigned long long int off_x_stride = (x_stride + off_x) << in_l_shift;
                const unsigned long long int off_x_depth = off_x << in_l_shift;

                for (d = 0; d < d_bound; d+=64) {

                  unsigned long long int in_subscript = b_depth
                    + off_y_stride
                    + off_x_stride
                    + d;

                  unsigned long long int filter_subscript = m_mask
                    + off_y_mask
                    + off_x_depth
                    + d;

                   __m256i s = _mm256_load_si256(( __m256i*)&in_Char[in_subscript]);
                   __m256i s2 = _mm256_load_si256(( __m256i*)&in_Char[in_subscript + Input_depth_dim]);
                   __m256i s3 = _mm256_load_si256(( __m256i*)&in_Char[in_subscript + (2 << in_l_shift)]);
                   __m256i s4 = _mm256_load_si256(( __m256i*)&in_Char[in_subscript + (3 << in_l_shift)]);

                   __m256i w = _mm256_load_si256(( __m256i*)&filter_Char[filter_subscript]);
                   __m256i w2 = _mm256_load_si256(( __m256i*)&filter_Char[filter_subscript + mask_yx_depth]);


                  __m256i inter_vec = _mm256_maddubs_epi16(s,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec = _mm256_add_epi32(temp_vec, inter_vec);

                  __m256i inter_vec2 = _mm256_maddubs_epi16(s2,w);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec2 = _mm256_add_epi32(temp_vec2, inter_vec2);

                  inter_vec = _mm256_maddubs_epi16(s3,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec3 = _mm256_add_epi32(temp_vec3, inter_vec);

                  inter_vec2 = _mm256_maddubs_epi16(s4, w);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec4 = _mm256_add_epi32(temp_vec4, inter_vec2);

                  inter_vec = _mm256_maddubs_epi16(s,w2);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec5 = _mm256_add_epi32(temp_vec5, inter_vec);

                  inter_vec2 = _mm256_maddubs_epi16(s2, w2);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec6 = _mm256_add_epi32(temp_vec6, inter_vec2);

                  inter_vec = _mm256_maddubs_epi16(s3,w2);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec7 = _mm256_add_epi32(temp_vec7, inter_vec);

                  inter_vec2 = _mm256_maddubs_epi16(s4, w2);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec8 = _mm256_add_epi32(temp_vec8, inter_vec2);

                  // d+32 ------------------------------------------------------------
                  // |
                  // |
                  // v

                  // in_subscript += 32;

                  // filter_subscript += 32;

                  s = _mm256_load_si256((const __m256i*)&in_Char[in_subscript + 32]);
                  s2 = _mm256_load_si256((const __m256i*)&in_Char[in_subscript + 32 + Input_depth_dim]);
                  s3 = _mm256_load_si256((const __m256i*)&in_Char[in_subscript + 32 + (2 << in_l_shift)]);
                  s4 = _mm256_load_si256((const __m256i*)&in_Char[in_subscript + 32 + (3 << in_l_shift)]);

                  w = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript + 32]);
                  w2 = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript + 32 + mask_yx_depth]);


                  inter_vec = _mm256_maddubs_epi16(s,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec = _mm256_add_epi32(temp_vec, inter_vec);

                  inter_vec2 = _mm256_maddubs_epi16(s2,w);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec2 = _mm256_add_epi32(temp_vec2, inter_vec2);

                  inter_vec = _mm256_maddubs_epi16(s3,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec3 = _mm256_add_epi32(temp_vec3, inter_vec);

                  inter_vec2 = _mm256_maddubs_epi16(s4, w);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec4 = _mm256_add_epi32(temp_vec4, inter_vec2);

                  inter_vec = _mm256_maddubs_epi16(s,w2);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec5 = _mm256_add_epi32(temp_vec5, inter_vec);

                  inter_vec2 = _mm256_maddubs_epi16(s2, w2);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec6 = _mm256_add_epi32(temp_vec6, inter_vec2);

                  inter_vec = _mm256_maddubs_epi16(s3,w2);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec7 = _mm256_add_epi32(temp_vec7, inter_vec);

                  inter_vec2 = _mm256_maddubs_epi16(s4, w2);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec8 = _mm256_add_epi32(temp_vec8, inter_vec2);
                }
                // fallback loop in case Input_depth < 64
                for (; d < Input_depth_dim; d+=32) {
                  unsigned long long int in_subscript = b_depth
                    + off_y_stride
                    + off_x_stride
                    + d;

                  unsigned long long int filter_subscript = m_mask
                    + off_y_mask
                    + off_x_depth
                    + d;

                   __m256i s = _mm256_load_si256(( __m256i*)&in_Char[in_subscript]);
                   __m256i s2 = _mm256_load_si256(( __m256i*)&in_Char[in_subscript + Input_depth_dim]);
                   __m256i s3 = _mm256_load_si256(( __m256i*)&in_Char[in_subscript + (2 << in_l_shift)]);
                   __m256i s4 = _mm256_load_si256(( __m256i*)&in_Char[in_subscript + (3 << in_l_shift)]);

                   __m256i w = _mm256_load_si256(( __m256i*)&filter_Char[filter_subscript]);
                   __m256i w2 = _mm256_load_si256(( __m256i*)&filter_Char[filter_subscript + mask_yx_depth]);


                  __m256i inter_vec = _mm256_maddubs_epi16(s,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec = _mm256_add_epi32(temp_vec, inter_vec);

                  __m256i inter_vec2 = _mm256_maddubs_epi16(s2,w);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec2 = _mm256_add_epi32(temp_vec2, inter_vec2);

                  inter_vec = _mm256_maddubs_epi16(s3,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec3 = _mm256_add_epi32(temp_vec3, inter_vec);

                  inter_vec2 = _mm256_maddubs_epi16(s4, w);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec4 = _mm256_add_epi32(temp_vec4, inter_vec2);

                  inter_vec = _mm256_maddubs_epi16(s,w2);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec5 = _mm256_add_epi32(temp_vec5, inter_vec);

                  inter_vec2 = _mm256_maddubs_epi16(s2, w2);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec6 = _mm256_add_epi32(temp_vec6, inter_vec2);

                  inter_vec = _mm256_maddubs_epi16(s3,w2);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec7 = _mm256_add_epi32(temp_vec7, inter_vec);

                  inter_vec2 = _mm256_maddubs_epi16(s4, w2);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec8 = _mm256_add_epi32(temp_vec8, inter_vec2);
                }
              }
            }

            unsigned long long int out_subscript = b_out
              + y_out
              + (x << out_l_shift)
              + m;
            // unsigned long long int out_subscript2 = out_subscript + Output_depth_dim;
            // unsigned long long int out_subscript3 = out_subscript + (2 << out_l_shift);
            // unsigned long long int out_subscript4 = out_subscript + (3 << out_l_shift);




            // **************** optimised hadd ****************
            __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec), _mm256_extracti128_si256(temp_vec, 1));
            __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
            __m128i sum64 = _mm_add_epi32(hi64, sum128);
            __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            __m128i sum32 = _mm_add_epi32(sum64, hi32);
            temp = _mm_cvtsi128_si32(sum32);
            temp += bias;
            out_to_compare_with_Char[out_subscript] = RELU(temp);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec2), _mm256_extracti128_si256(temp_vec2, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp2 = _mm_cvtsi128_si32(sum32);
            temp2 += bias;
            out_to_compare_with_Char[out_subscript + (1 << out_l_shift)] = RELU(temp2);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec3), _mm256_extracti128_si256(temp_vec3, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp3 = _mm_cvtsi128_si32(sum32);
            temp3 += bias;
            out_to_compare_with_Char[out_subscript + (2 << out_l_shift)] = RELU(temp3);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec4), _mm256_extracti128_si256(temp_vec4, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp4 = _mm_cvtsi128_si32(sum32);
            temp4 += bias;
            out_to_compare_with_Char[out_subscript + (3 << out_l_shift)] = RELU(temp4);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec5), _mm256_extracti128_si256(temp_vec5, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp5 = _mm_cvtsi128_si32(sum32);
            temp5 += bias2;
            out_to_compare_with_Char[out_subscript+1] = RELU(temp5);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec6), _mm256_extracti128_si256(temp_vec6, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp6 = _mm_cvtsi128_si32(sum32);
            temp6 += bias2;
            out_to_compare_with_Char[out_subscript+1+ (1 << out_l_shift)] = RELU(temp6);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec7), _mm256_extracti128_si256(temp_vec7, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp7 = _mm_cvtsi128_si32(sum32);
            temp7 += bias2;
            out_to_compare_with_Char[out_subscript+1+ (2 << out_l_shift)] = RELU(temp7);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec8), _mm256_extracti128_si256(temp_vec8, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp8 = _mm_cvtsi128_si32(sum32);
            temp8 += bias2;
            out_to_compare_with_Char[out_subscript+1+ (3 << out_l_shift)] = RELU(temp8);
          }
          // fallback loop
          for (;x < Output_X_dim; x++) {	//Output Width
            const unsigned long long int x_stride = x * Stride_X_dim;
            int temp, temp2;

            __m256i temp_vec = _mm256_setzero_si256();
            __m256i temp_vec2 = _mm256_setzero_si256();

            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              const unsigned long long int off_y_stride = (y_stride + off_y) * in_x_depth;
              const unsigned long long int off_y_mask = off_y * mask_x_depth;

              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                const unsigned long long int off_x_stride = (x_stride + off_x) << in_l_shift;
                const unsigned long long int off_x_depth = off_x << in_l_shift;

                for (unsigned int d = 0; d < Input_depth_dim; d+=64) {

                  unsigned long long int in_subscript = b_depth
                    + off_y_stride
                    + off_x_stride
                    + d;

                  unsigned long long int filter_subscript = m_mask
                    + off_y_mask
                    + off_x_depth
                    + d;

                   __m256i s = _mm256_load_si256(( __m256i*)&in_Char[in_subscript]);

                   __m256i w = _mm256_load_si256(( __m256i*)&filter_Char[filter_subscript]);
                   __m256i w2 = _mm256_load_si256(( __m256i*)&filter_Char[filter_subscript + mask_yx_depth]);


                  __m256i inter_vec = _mm256_maddubs_epi16(s,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec = _mm256_add_epi32(temp_vec, inter_vec);

                  __m256i inter_vec2 = _mm256_maddubs_epi16(s,w2);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec2 = _mm256_add_epi32(temp_vec2, inter_vec2);

                  // d+32 ------------------------------------------------------------
                  // |
                  // |
                  // v

                  // in_subscript += 32;

                  // filter_subscript += 32;

                  s = _mm256_load_si256((const __m256i*)&in_Char[in_subscript + 32]);

                  w = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript + 32]);
                  w2 = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript + 32 + mask_yx_depth]);


                  inter_vec = _mm256_maddubs_epi16(s,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec = _mm256_add_epi32(temp_vec, inter_vec);

                  inter_vec2 = _mm256_maddubs_epi16(s,w2);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec2 = _mm256_add_epi32(temp_vec2, inter_vec2);
                }
              }
            }

            unsigned long long int out_subscript = b_out
              + y_out
              + (x << out_l_shift)
              + m;




            // **************** optimised hadd ****************
            __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec), _mm256_extracti128_si256(temp_vec, 1));
            __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
            __m128i sum64 = _mm_add_epi32(hi64, sum128);
            __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            __m128i sum32 = _mm_add_epi32(sum64, hi32);
            temp = _mm_cvtsi128_si32(sum32);
            temp += bias;
            out_to_compare_with_Char[out_subscript] = RELU(temp);


            __m128i sum128_2 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec2), _mm256_extracti128_si256(temp_vec2, 1));
            __m128i hi64_2 = _mm_unpackhi_epi64(sum128_2, sum128_2);
            __m128i sum64_2 = _mm_add_epi32(hi64_2, sum128_2);
            __m128i hi32_2 = _mm_shuffle_epi32(sum64_2, _MM_SHUFFLE(2,3,0,1));
            __m128i sum32_2 = _mm_add_epi32(sum64_2, hi32_2);
            temp2 = _mm_cvtsi128_si32(sum32_2);
            temp2 += bias2;
            out_to_compare_with_Char[out_subscript + 1] = RELU(temp2);

          }
        }
      }
    }
  }
  // #undef left_shift
  #undef RELU
  printf("\n from quantised opt m2x4 %d %d ", out_to_compare_with_Char[0], out_to_compare_with_Char[1]);
  return 0;
}


// this is assumed to be used as a non-initial convolution layer. thus the input and output depths are powers of 2
// ~749 GFLOPS without hitting fallback loops, 642 when using squeezenet dims & x y = 57
int opt_m2x4_deep_AVX_d32_Char(const unsigned char* in_Char, const signed char* filter_Char, const int* bias_array_Int, unsigned char* out_to_compare_with_Char) {
  #define left_shift(x) (__builtin_ctz(x))  // inline function effectively using inline assembly to return num of trailing zeros (i.e. 128 returns 7, 2 * 128 == 2 << 7)
  #define RELU(x) ((x) < 0 ? 0 : (x)) // return x if greater than 0, else return 0

  // float bias, bias2, bias3, bias4;
  // __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

  const unsigned long long int in_l_shift = left_shift(Input_depth_dim);
  const unsigned long long int out_l_shift = left_shift(Output_depth_dim);

  const unsigned long long int in_yx_depth = Input_Y_dim * Input_X_dim << in_l_shift;
  const unsigned long long int in_x_depth = Input_X_dim  << in_l_shift;

  const unsigned long long int mask_yx_depth = Mask_Y_dim * Mask_X_dim << in_l_shift;
  const unsigned long long int mask_x_depth = Mask_X_dim << in_l_shift;

  const unsigned long long int out_yx_depth = Output_X_dim * Output_Y_dim << out_l_shift;
  const unsigned long long int out_x_depth = Output_X_dim << out_l_shift;

  // const unsigned int m_bound = (Output_depth_dim/ 2) * 2;
  const unsigned int x_bound = (Input_X_dim/ 4) * 4;
  const unsigned int d_bound = (Input_depth_dim/ 32) * 32;
  
  unsigned int x, d;

  #pragma omp parallel for private(x) shared(in_yx_depth, in_x_depth, mask_yx_depth, mask_x_depth, out_yx_depth, out_x_depth) \
    default(shared) collapse(2) schedule(static)
  {
    for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
      const unsigned long long int b_depth = b * in_yx_depth;
      const unsigned long long int b_out = b * out_yx_depth;

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        const unsigned long long int y_stride = y * Stride_Y_dim;
        const unsigned long long int y_out = y * out_x_depth;

        for (unsigned int m = 0; m < Output_depth_dim; m+=2) { //channels
          const unsigned long long int m_mask = m * mask_yx_depth;

          const int bias = bias_array_Int[m];
          const int bias2 = bias_array_Int[m+1];

          for (x = 0; x < x_bound; x+=4) {	//Output Width
            const unsigned long long int x_stride = x * Stride_X_dim;
            int temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

            __m256i temp_vec = _mm256_setzero_si256();
            __m256i temp_vec2 = _mm256_setzero_si256();
            __m256i temp_vec3 = _mm256_setzero_si256();
            __m256i temp_vec4 = _mm256_setzero_si256();
            __m256i temp_vec5 = _mm256_setzero_si256();
            __m256i temp_vec6 = _mm256_setzero_si256();
            __m256i temp_vec7 = _mm256_setzero_si256();
            __m256i temp_vec8 = _mm256_setzero_si256();

            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              const unsigned long long int off_y_stride = (y_stride + off_y) * in_x_depth;
              const unsigned long long int off_y_mask = off_y * mask_x_depth;

              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                const unsigned long long int off_x_stride = (x_stride + off_x) << in_l_shift;
                const unsigned long long int off_x_depth = off_x << in_l_shift;

                for (unsigned int d = 0; d < d_bound; d+=32) {

                  unsigned long long int in_subscript = b_depth
                    + off_y_stride
                    + off_x_stride
                    + d;

                  unsigned long long int filter_subscript = m_mask
                    + off_y_mask
                    + off_x_depth
                    + d;

                   __m256i s = _mm256_load_si256(( __m256i*)&in_Char[in_subscript]);
                   __m256i s2 = _mm256_load_si256(( __m256i*)&in_Char[in_subscript + Input_depth_dim]);
                   __m256i s3 = _mm256_load_si256(( __m256i*)&in_Char[in_subscript + (2 << in_l_shift)]);
                   __m256i s4 = _mm256_load_si256(( __m256i*)&in_Char[in_subscript + (3 << in_l_shift)]);

                   __m256i w = _mm256_load_si256(( __m256i*)&filter_Char[filter_subscript]);
                   __m256i w2 = _mm256_load_si256(( __m256i*)&filter_Char[filter_subscript + mask_yx_depth]);


                  __m256i inter_vec = _mm256_maddubs_epi16(s,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec = _mm256_add_epi32(temp_vec, inter_vec);

                  __m256i inter_vec2 = _mm256_maddubs_epi16(s2,w);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec2 = _mm256_add_epi32(temp_vec2, inter_vec2);

                  inter_vec = _mm256_maddubs_epi16(s3,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec3 = _mm256_add_epi32(temp_vec3, inter_vec);

                  inter_vec2 = _mm256_maddubs_epi16(s4, w);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec4 = _mm256_add_epi32(temp_vec4, inter_vec2);

                  inter_vec = _mm256_maddubs_epi16(s,w2);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec5 = _mm256_add_epi32(temp_vec5, inter_vec);

                  inter_vec2 = _mm256_maddubs_epi16(s2, w2);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec6 = _mm256_add_epi32(temp_vec6, inter_vec2);

                  inter_vec = _mm256_maddubs_epi16(s3,w2);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec7 = _mm256_add_epi32(temp_vec7, inter_vec);

                  inter_vec2 = _mm256_maddubs_epi16(s4, w2);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec8 = _mm256_add_epi32(temp_vec8, inter_vec2);
                }
              }
            }

            unsigned long long int out_subscript = b_out
              + y_out
              + (x << out_l_shift)
              + m;
            // unsigned long long int out_subscript2 = out_subscript + Output_depth_dim;
            // unsigned long long int out_subscript3 = out_subscript + (2 << out_l_shift);
            // unsigned long long int out_subscript4 = out_subscript + (3 << out_l_shift);




            // **************** optimised hadd ****************
            __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec), _mm256_extracti128_si256(temp_vec, 1));
            __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
            __m128i sum64 = _mm_add_epi32(hi64, sum128);
            __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            __m128i sum32 = _mm_add_epi32(sum64, hi32);
            temp = _mm_cvtsi128_si32(sum32);
            temp += bias;
            out_to_compare_with_Char[out_subscript] = RELU(temp);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec2), _mm256_extracti128_si256(temp_vec2, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp2 = _mm_cvtsi128_si32(sum32);
            temp2 += bias;
            out_to_compare_with_Char[out_subscript + (1 << out_l_shift)] = RELU(temp2);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec3), _mm256_extracti128_si256(temp_vec3, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp3 = _mm_cvtsi128_si32(sum32);
            temp3 += bias;
            out_to_compare_with_Char[out_subscript + (2 << out_l_shift)] = RELU(temp3);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec4), _mm256_extracti128_si256(temp_vec4, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp4 = _mm_cvtsi128_si32(sum32);
            temp4 += bias;
            out_to_compare_with_Char[out_subscript + (3 << out_l_shift)] = RELU(temp4);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec5), _mm256_extracti128_si256(temp_vec5, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp5 = _mm_cvtsi128_si32(sum32);
            temp5 += bias2;
            out_to_compare_with_Char[out_subscript+1] = RELU(temp5);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec6), _mm256_extracti128_si256(temp_vec6, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp6 = _mm_cvtsi128_si32(sum32);
            temp6 += bias2;
            out_to_compare_with_Char[out_subscript+1+ (1 << out_l_shift)] = RELU(temp6);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec7), _mm256_extracti128_si256(temp_vec7, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp7 = _mm_cvtsi128_si32(sum32);
            temp7 += bias2;
            out_to_compare_with_Char[out_subscript+1+ (2 << out_l_shift)] = RELU(temp7);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec8), _mm256_extracti128_si256(temp_vec8, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp8 = _mm_cvtsi128_si32(sum32);
            temp8 += bias2;
            out_to_compare_with_Char[out_subscript+1+ (3 << out_l_shift)] = RELU(temp8);
          }
          // fallback loop
          for (;x < Output_X_dim; x++) {	//Output Width
            const unsigned long long int x_stride = x * Stride_X_dim;
            int temp, temp2;

            __m256i temp_vec = _mm256_setzero_si256();
            __m256i temp_vec2 = _mm256_setzero_si256();

            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              const unsigned long long int off_y_stride = (y_stride + off_y) * in_x_depth;
              const unsigned long long int off_y_mask = off_y * mask_x_depth;

              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                const unsigned long long int off_x_stride = (x_stride + off_x) << in_l_shift;
                const unsigned long long int off_x_depth = off_x << in_l_shift;

                for (unsigned int d = 0; d < Input_depth_dim; d+=32) {

                  unsigned long long int in_subscript = b_depth
                    + off_y_stride
                    + off_x_stride
                    + d;

                  unsigned long long int filter_subscript = m_mask
                    + off_y_mask
                    + off_x_depth
                    + d;

                   __m256i s = _mm256_load_si256(( __m256i*)&in_Char[in_subscript]);

                   __m256i w = _mm256_load_si256(( __m256i*)&filter_Char[filter_subscript]);
                   __m256i w2 = _mm256_load_si256(( __m256i*)&filter_Char[filter_subscript + mask_yx_depth]);


                  __m256i inter_vec = _mm256_maddubs_epi16(s,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec = _mm256_add_epi32(temp_vec, inter_vec);

                  __m256i inter_vec2 = _mm256_maddubs_epi16(s,w2);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec2 = _mm256_add_epi32(temp_vec2, inter_vec2);
                }
              }
            }

            unsigned long long int out_subscript = b_out
              + y_out
              + (x << out_l_shift)
              + m;




            // **************** optimised hadd ****************
            __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec), _mm256_extracti128_si256(temp_vec, 1));
            __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
            __m128i sum64 = _mm_add_epi32(hi64, sum128);
            __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            __m128i sum32 = _mm_add_epi32(sum64, hi32);
            temp = _mm_cvtsi128_si32(sum32);
            temp += bias;
            out_to_compare_with_Char[out_subscript] = RELU(temp);


            __m128i sum128_2 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec2), _mm256_extracti128_si256(temp_vec2, 1));
            __m128i hi64_2 = _mm_unpackhi_epi64(sum128_2, sum128_2);
            __m128i sum64_2 = _mm_add_epi32(hi64_2, sum128_2);
            __m128i hi32_2 = _mm_shuffle_epi32(sum64_2, _MM_SHUFFLE(2,3,0,1));
            __m128i sum32_2 = _mm_add_epi32(sum64_2, hi32_2);
            temp2 = _mm_cvtsi128_si32(sum32_2);
            temp2 += bias2;
            out_to_compare_with_Char[out_subscript + 1] = RELU(temp2);

          }
        }
      }
    }
  }
  // #undef left_shift
  #undef RELU
  printf("\n from quantised opt m2x4 %d %d ", out_to_compare_with_Char[0], out_to_compare_with_Char[1]);
  return 0;
}


// sse function for when input_depth_dim = 16
// 374 GFLOPS without hitting fallback loops
int opt_m2x4_deep_SSE_d16_Char(const unsigned char* in_Char, const signed char* filter_Char, const int* bias_array_Int, unsigned char* out_to_compare_with_Char) {
  #define left_shift(x) (__builtin_ctz(x))  // inline function effectively using inline assembly to return num of trailing zeros (i.e. 128 returns 7, 2 * 128 == 2 << 7)
  #define RELU(x) ((x) < 0 ? 0 : (x)) // return x if greater than 0, else return 0

  // float bias, bias2, bias3, bias4;
  // __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

  const unsigned long long int in_l_shift = left_shift(Input_depth_dim);
  const unsigned long long int out_l_shift = left_shift(Output_depth_dim);

  const unsigned long long int in_yx_depth = Input_Y_dim * Input_X_dim << in_l_shift;
  const unsigned long long int in_x_depth = Input_X_dim  << in_l_shift;

  const unsigned long long int mask_yx_depth = Mask_Y_dim * Mask_X_dim << in_l_shift;
  const unsigned long long int mask_x_depth = Mask_X_dim << in_l_shift;

  const unsigned long long int out_yx_depth = Output_X_dim * Output_Y_dim << out_l_shift;
  const unsigned long long int out_x_depth = Output_X_dim << out_l_shift;

  // const unsigned int m_bound = (Output_depth_dim/ 2) * 2;
  const unsigned int x_bound = (Input_X_dim/ 4) * 4;
  
  unsigned int x;

  #pragma omp parallel for private(x) shared(in_yx_depth, in_x_depth, mask_yx_depth, mask_x_depth, out_yx_depth, out_x_depth) \
    default(shared) collapse(2) schedule(static)
  {
    for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
      const unsigned long long int b_depth = b * in_yx_depth;
      const unsigned long long int b_out = b * out_yx_depth;

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        const unsigned long long int y_stride = y * Stride_Y_dim;
        const unsigned long long int y_out = y * out_x_depth;

        for (unsigned int m = 0; m < Output_depth_dim; m+=2) { //channels
          const unsigned long long int m_mask = m * mask_yx_depth;

          const int bias = bias_array_Int[m];
          const int bias2 = bias_array_Int[m+1];

          for (x = 0; x < x_bound; x+=4) {	//Output Width
            const unsigned long long int x_stride = x * Stride_X_dim;
            int temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

            __m128i temp_vec = _mm_setzero_si128();
            __m128i temp_vec2 = _mm_setzero_si128();
            __m128i temp_vec3 = _mm_setzero_si128();
            __m128i temp_vec4 = _mm_setzero_si128();
            __m128i temp_vec5 = _mm_setzero_si128();
            __m128i temp_vec6 = _mm_setzero_si128();
            __m128i temp_vec7 = _mm_setzero_si128();
            __m128i temp_vec8 = _mm_setzero_si128();

            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              const unsigned long long int off_y_stride = (y_stride + off_y) * in_x_depth;
              const unsigned long long int off_y_mask = off_y * mask_x_depth;

              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                const unsigned long long int off_x_stride = (x_stride + off_x) << in_l_shift;
                const unsigned long long int off_x_depth = off_x << in_l_shift;

                for (unsigned int d = 0; d < Input_depth_dim; d+=16) {

                  unsigned long long int in_subscript = b_depth
                    + off_y_stride
                    + off_x_stride
                    + d;

                  unsigned long long int filter_subscript = m_mask
                    + off_y_mask
                    + off_x_depth
                    + d;

                  __m128i s = _mm_load_si128(( __m128i*)&in_Char[in_subscript]);
                  __m128i s2 = _mm_load_si128(( __m128i*)&in_Char[in_subscript + Input_depth_dim]);
                  __m128i s3 = _mm_load_si128(( __m128i*)&in_Char[in_subscript + (2 << in_l_shift)]);
                  __m128i s4 = _mm_load_si128(( __m128i*)&in_Char[in_subscript + (3 << in_l_shift)]);

                  __m128i w = _mm_load_si128(( __m128i*)&filter_Char[filter_subscript]);
                  __m128i w2 = _mm_load_si128(( __m128i*)&filter_Char[filter_subscript + mask_yx_depth]);

                  __m128i inter_vec = _mm_maddubs_epi16(s,w);
                  inter_vec = _mm_madd_epi16(inter_vec, _mm_set1_epi16(1));
                  temp_vec = _mm_add_epi32(temp_vec, inter_vec);

                  __m128i inter_vec2 = _mm_maddubs_epi16(s2,w);
                  inter_vec2 = _mm_madd_epi16(inter_vec2, _mm_set1_epi16(1));
                  temp_vec2 = _mm_add_epi32(temp_vec2, inter_vec2);

                  inter_vec = _mm_maddubs_epi16(s3,w);
                  inter_vec = _mm_madd_epi16(inter_vec, _mm_set1_epi16(1));
                  temp_vec3 = _mm_add_epi32(temp_vec3, inter_vec);

                  inter_vec2 = _mm_maddubs_epi16(s4, w);
                  inter_vec2 = _mm_madd_epi16(inter_vec2, _mm_set1_epi16(1));
                  temp_vec4 = _mm_add_epi32(temp_vec4, inter_vec2);

                  inter_vec = _mm_maddubs_epi16(s,w2);
                  inter_vec = _mm_madd_epi16(inter_vec, _mm_set1_epi16(1));
                  temp_vec5 = _mm_add_epi32(temp_vec5, inter_vec);

                  inter_vec2 = _mm_maddubs_epi16(s2, w2);
                  inter_vec2 = _mm_madd_epi16(inter_vec2, _mm_set1_epi16(1));
                  temp_vec6 = _mm_add_epi32(temp_vec6, inter_vec2);

                  inter_vec = _mm_maddubs_epi16(s3,w2);
                  inter_vec = _mm_madd_epi16(inter_vec, _mm_set1_epi16(1));
                  temp_vec7 = _mm_add_epi32(temp_vec7, inter_vec);

                  inter_vec2 = _mm_maddubs_epi16(s4, w2);
                  inter_vec2 = _mm_madd_epi16(inter_vec2, _mm_set1_epi16(1));
                  temp_vec8 = _mm_add_epi32(temp_vec8, inter_vec2);                
                }
              }
            }

            unsigned long long int out_subscript = b_out
              + y_out
              + (x << out_l_shift)
              + m;
            // unsigned long long int out_subscript2 = out_subscript + Output_depth_dim;
            // unsigned long long int out_subscript3 = out_subscript + (2 << out_l_shift);
            // unsigned long long int out_subscript4 = out_subscript + (3 << out_l_shift);




            // **************** optimised hadd ***************

            __m128i vsum = _mm_add_epi32(temp_vec, _mm_srli_si128(temp_vec, 8));
            vsum = _mm_add_epi32(vsum, _mm_srli_si128(vsum, 4));
            temp = _mm_cvtsi128_si32(vsum);
            temp += bias;
            out_to_compare_with_Char[out_subscript] = RELU(temp);

            __m128i vsum2 = _mm_add_epi32(temp_vec2, _mm_srli_si128(temp_vec2, 8));
            vsum2 = _mm_add_epi32(vsum2, _mm_srli_si128(vsum2, 4));
            temp2 = _mm_cvtsi128_si32(vsum2);
            temp2 += bias;
            out_to_compare_with_Char[out_subscript + (1 << out_l_shift)] = RELU(temp2);


            __m128i vsum3 = _mm_add_epi32(temp_vec3, _mm_srli_si128(temp_vec3, 8));
            vsum3 = _mm_add_epi32(vsum3, _mm_srli_si128(vsum3, 4));
            temp3 = _mm_cvtsi128_si32(vsum3);
            temp3 += bias;
            out_to_compare_with_Char[out_subscript + (2 << out_l_shift)] = RELU(temp3);


            __m128i vsum4 = _mm_add_epi32(temp_vec4, _mm_srli_si128(temp_vec4, 8));
            vsum4 = _mm_add_epi32(vsum4, _mm_srli_si128(vsum4, 4));
            temp4 = _mm_cvtsi128_si32(vsum4);
            temp4 += bias;
            out_to_compare_with_Char[out_subscript + (3 << out_l_shift)] = RELU(temp4);


            __m128i vsum5 = _mm_add_epi32(temp_vec5, _mm_srli_si128(temp_vec5, 8));
            vsum5 = _mm_add_epi32(vsum5, _mm_srli_si128(vsum5, 4));
            temp5 = _mm_cvtsi128_si32(vsum5);
            temp5 += bias2;
            out_to_compare_with_Char[out_subscript+1] = RELU(temp5);


            __m128i vsum6 = _mm_add_epi32(temp_vec6, _mm_srli_si128(temp_vec6, 8));
            vsum6 = _mm_add_epi32(vsum6, _mm_srli_si128(vsum6, 4));
            temp6 = _mm_cvtsi128_si32(vsum6);
            temp6 += bias2;
            out_to_compare_with_Char[out_subscript+1+ (1 << out_l_shift)] = RELU(temp6);


            __m128i vsum7 = _mm_add_epi32(temp_vec7, _mm_srli_si128(temp_vec7, 8));
            vsum7 = _mm_add_epi32(vsum7, _mm_srli_si128(vsum7, 4));
            temp7 = _mm_cvtsi128_si32(vsum7);
            temp7 += bias2;
            out_to_compare_with_Char[out_subscript+1+ (2 << out_l_shift)] = RELU(temp7);


            __m128i vsum8 = _mm_add_epi32(temp_vec8, _mm_srli_si128(temp_vec8, 8));
            vsum8 = _mm_add_epi32(vsum8, _mm_srli_si128(vsum8, 4));
            temp8 = _mm_cvtsi128_si32(vsum8);
            temp8 += bias2;
            out_to_compare_with_Char[out_subscript+1+ (3 << out_l_shift)] = RELU(temp8);

          }
          // fallback loop
          for (;x < Output_X_dim; x++) {	//Output Width
            const unsigned long long int x_stride = x * Stride_X_dim;
            int temp, temp2;

            __m128i temp_vec = _mm_setzero_si128();
            __m128i temp_vec2 = _mm_setzero_si128();

            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              const unsigned long long int off_y_stride = (y_stride + off_y) * in_x_depth;
              const unsigned long long int off_y_mask = off_y * mask_x_depth;

              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                const unsigned long long int off_x_stride = (x_stride + off_x) << in_l_shift;
                const unsigned long long int off_x_depth = off_x << in_l_shift;

                for (unsigned int d = 0; d < Input_depth_dim; d+=32) {

                  unsigned long long int in_subscript = b_depth
                    + off_y_stride
                    + off_x_stride
                    + d;

                  unsigned long long int filter_subscript = m_mask
                    + off_y_mask
                    + off_x_depth
                    + d;
                  

                  __m128i s = _mm_load_si128(( __m128i*)&in_Char[in_subscript]);


                  __m128i w = _mm_load_si128(( __m128i*)&filter_Char[filter_subscript]);
                  __m128i w2 = _mm_load_si128(( __m128i*)&filter_Char[filter_subscript + mask_yx_depth]);

                  __m128i inter_vec = _mm_maddubs_epi16(s,w);
                  inter_vec = _mm_madd_epi16(inter_vec, _mm_set1_epi16(1));
                  temp_vec = _mm_add_epi32(temp_vec, inter_vec);

                  __m128i inter_vec2 = _mm_maddubs_epi16(s,w2);
                  inter_vec2 = _mm_madd_epi16(inter_vec2, _mm_set1_epi16(1));
                  temp_vec2 = _mm_add_epi32(temp_vec2, inter_vec2);
                }
              }
            }

            unsigned long long int out_subscript = b_out
              + y_out
              + (x << out_l_shift)
              + m;


            // **************** optimised hadd ****************
            __m128i vsum = _mm_add_epi32(temp_vec, _mm_srli_si128(temp_vec, 8));
            vsum = _mm_add_epi32(vsum, _mm_srli_si128(vsum, 4));
            temp = _mm_cvtsi128_si32(vsum);
            temp += bias;
            out_to_compare_with_Char[out_subscript] = RELU(temp);

            __m128i vsum2 = _mm_add_epi32(temp_vec2, _mm_srli_si128(temp_vec2, 8));
            vsum2 = _mm_add_epi32(vsum2, _mm_srli_si128(vsum2, 4));
            temp2 = _mm_cvtsi128_si32(vsum2);
            temp2 += bias2;
            out_to_compare_with_Char[out_subscript + 1] = RELU(temp2);
          }
        }
      }
    }
  }
  // #undef left_shift
  #undef RELU
  printf("\n from quantised opt m2x4 %d %d ", out_to_compare_with_Char[0], out_to_compare_with_Char[1]);
  return 0;
}










// iterative functions 
// __________________________________________________________________________________________________________________________________________________________________________________




// fallback loop for m

// this is assumed to be used as a non-initial convolution layer. thus the input and output depths are powers of 2
// ~749 GFLOPS without hitting fallback loops, 642 when using squeezenet dims & x y = 57
int opt_m2x4_deep_d64_m_fallback_Char(const unsigned char* in_Char, const signed char* filter_Char, const int* bias_array_Int, unsigned char* out_to_compare_with_Char) {
  #define left_shift(x) (__builtin_ctz(x))  // inline function effectively using inline assembly to return num of trailing zeros (i.e. 128 returns 7, 2 * 128 == 2 << 7)
  #define RELU(x) ((x) < 0 ? 0 : (x)) // return x if greater than 0, else return 0
  #define x_tile 4

  // float bias, bias2, bias3, bias4;
  // __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

  const unsigned long long int in_l_shift = left_shift(Input_depth_dim);
  const unsigned long long int out_l_shift = left_shift(Output_depth_dim);

  const unsigned long long int in_yx_depth = Input_Y_dim * Input_X_dim << in_l_shift;
  const unsigned long long int in_x_depth = Input_X_dim  << in_l_shift;

  const unsigned long long int mask_yx_depth = Mask_Y_dim * Mask_X_dim << in_l_shift;
  const unsigned long long int mask_x_depth = Mask_X_dim << in_l_shift;

  const unsigned long long int out_yx_depth = Output_X_dim * Output_Y_dim << out_l_shift;
  const unsigned long long int out_x_depth = Output_X_dim << out_l_shift;

  // const unsigned int m_bound = (Output_depth_dim/ 2) * 2;
  const unsigned int x_bound = (Input_X_dim/ 4) * 4;
  
  unsigned int x, m;

  #pragma omp parallel for private(x, m) shared(in_yx_depth, in_x_depth, mask_yx_depth, mask_x_depth, out_yx_depth, out_x_depth) \
    default(shared) collapse(2) schedule(static)
  {
    for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
      const unsigned long long int b_depth = b * in_yx_depth;
      const unsigned long long int b_out = b * out_yx_depth;

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        const unsigned long long int y_stride = y * Stride_Y_dim;
        const unsigned long long int y_out = y * out_x_depth;

        for (m = 0; m < Output_depth_dim; m+=2) { //channels
          const unsigned long long int m_mask = m * mask_yx_depth;

          const int bias = bias_array_Int[m];
          const int bias2 = bias_array_Int[m+1];

          for (x = 0; x < x_bound; x+=4) {	//Output Width
            const unsigned long long int x_stride = x * Stride_X_dim;
            int temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

            __m256i temp_vec = _mm256_setzero_si256();
            __m256i temp_vec2 = _mm256_setzero_si256();
            __m256i temp_vec3 = _mm256_setzero_si256();
            __m256i temp_vec4 = _mm256_setzero_si256();
            __m256i temp_vec5 = _mm256_setzero_si256();
            __m256i temp_vec6 = _mm256_setzero_si256();
            __m256i temp_vec7 = _mm256_setzero_si256();
            __m256i temp_vec8 = _mm256_setzero_si256();

            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              const unsigned long long int off_y_stride = (y_stride + off_y) * in_x_depth;
              const unsigned long long int off_y_mask = off_y * mask_x_depth;

              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                const unsigned long long int off_x_stride = (x_stride + off_x) << in_l_shift;
                const unsigned long long int off_x_depth = off_x << in_l_shift;

                for (unsigned int d = 0; d < Input_depth_dim; d+=64) {

                  unsigned long long int in_subscript = b_depth
                    + off_y_stride
                    + off_x_stride
                    + d;

                  unsigned long long int filter_subscript = m_mask
                    + off_y_mask
                    + off_x_depth
                    + d;

                   __m256i s = _mm256_load_si256(( __m256i*)&in_Char[in_subscript]);
                   __m256i s2 = _mm256_load_si256(( __m256i*)&in_Char[in_subscript + Input_depth_dim]);
                   __m256i s3 = _mm256_load_si256(( __m256i*)&in_Char[in_subscript + (2 << in_l_shift)]);
                   __m256i s4 = _mm256_load_si256(( __m256i*)&in_Char[in_subscript + (3 << in_l_shift)]);

                   __m256i w = _mm256_load_si256(( __m256i*)&filter_Char[filter_subscript]);
                   __m256i w2 = _mm256_load_si256(( __m256i*)&filter_Char[filter_subscript + mask_yx_depth]);


                  __m256i inter_vec = _mm256_maddubs_epi16(s,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec = _mm256_add_epi32(temp_vec, inter_vec);

                  __m256i inter_vec2 = _mm256_maddubs_epi16(s2,w);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec2 = _mm256_add_epi32(temp_vec2, inter_vec2);

                  inter_vec = _mm256_maddubs_epi16(s3,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec3 = _mm256_add_epi32(temp_vec3, inter_vec);

                  inter_vec2 = _mm256_maddubs_epi16(s4, w);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec4 = _mm256_add_epi32(temp_vec4, inter_vec2);

                  inter_vec = _mm256_maddubs_epi16(s,w2);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec5 = _mm256_add_epi32(temp_vec5, inter_vec);

                  inter_vec2 = _mm256_maddubs_epi16(s2, w2);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec6 = _mm256_add_epi32(temp_vec6, inter_vec2);

                  inter_vec = _mm256_maddubs_epi16(s3,w2);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec7 = _mm256_add_epi32(temp_vec7, inter_vec);

                  inter_vec2 = _mm256_maddubs_epi16(s4, w2);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec8 = _mm256_add_epi32(temp_vec8, inter_vec2);

                  // d+32 ------------------------------------------------------------
                  // |
                  // |
                  // v

                  // in_subscript += 32;

                  // filter_subscript += 32;

                  s = _mm256_load_si256((const __m256i*)&in_Char[in_subscript + 32]);
                  s2 = _mm256_load_si256((const __m256i*)&in_Char[in_subscript + 32 + Input_depth_dim]);
                  s3 = _mm256_load_si256((const __m256i*)&in_Char[in_subscript + 32 + (2 << in_l_shift)]);
                  s4 = _mm256_load_si256((const __m256i*)&in_Char[in_subscript + 32 + (3 << in_l_shift)]);

                  w = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript + 32]);
                  w2 = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript + 32 + mask_yx_depth]);


                  inter_vec = _mm256_maddubs_epi16(s,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec = _mm256_add_epi32(temp_vec, inter_vec);

                  inter_vec2 = _mm256_maddubs_epi16(s2,w);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec2 = _mm256_add_epi32(temp_vec2, inter_vec2);

                  inter_vec = _mm256_maddubs_epi16(s3,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec3 = _mm256_add_epi32(temp_vec3, inter_vec);

                  inter_vec2 = _mm256_maddubs_epi16(s4, w);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec4 = _mm256_add_epi32(temp_vec4, inter_vec2);

                  inter_vec = _mm256_maddubs_epi16(s,w2);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec5 = _mm256_add_epi32(temp_vec5, inter_vec);

                  inter_vec2 = _mm256_maddubs_epi16(s2, w2);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec6 = _mm256_add_epi32(temp_vec6, inter_vec2);

                  inter_vec = _mm256_maddubs_epi16(s3,w2);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec7 = _mm256_add_epi32(temp_vec7, inter_vec);

                  inter_vec2 = _mm256_maddubs_epi16(s4, w2);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec8 = _mm256_add_epi32(temp_vec8, inter_vec2);
                }
              }
            }

            unsigned long long int out_subscript = b_out
              + y_out
              + (x << out_l_shift)
              + m;
            // unsigned long long int out_subscript2 = out_subscript + Output_depth_dim;
            // unsigned long long int out_subscript3 = out_subscript + (2 << out_l_shift);
            // unsigned long long int out_subscript4 = out_subscript + (3 << out_l_shift);




            // **************** optimised hadd ****************
            __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec), _mm256_extracti128_si256(temp_vec, 1));
            __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
            __m128i sum64 = _mm_add_epi32(hi64, sum128);
            __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            __m128i sum32 = _mm_add_epi32(sum64, hi32);
            temp = _mm_cvtsi128_si32(sum32);
            temp += bias;
            out_to_compare_with_Char[out_subscript] = RELU(temp);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec2), _mm256_extracti128_si256(temp_vec2, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp2 = _mm_cvtsi128_si32(sum32);
            temp2 += bias;
            out_to_compare_with_Char[out_subscript + (1 << out_l_shift)] = RELU(temp2);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec3), _mm256_extracti128_si256(temp_vec3, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp3 = _mm_cvtsi128_si32(sum32);
            temp3 += bias;
            out_to_compare_with_Char[out_subscript + (2 << out_l_shift)] = RELU(temp3);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec4), _mm256_extracti128_si256(temp_vec4, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp4 = _mm_cvtsi128_si32(sum32);
            temp4 += bias;
            out_to_compare_with_Char[out_subscript + (3 << out_l_shift)] = RELU(temp4);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec5), _mm256_extracti128_si256(temp_vec5, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp5 = _mm_cvtsi128_si32(sum32);
            temp5 += bias2;
            out_to_compare_with_Char[out_subscript+1] = RELU(temp5);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec6), _mm256_extracti128_si256(temp_vec6, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp6 = _mm_cvtsi128_si32(sum32);
            temp6 += bias2;
            out_to_compare_with_Char[out_subscript+1+ (1 << out_l_shift)] = RELU(temp6);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec7), _mm256_extracti128_si256(temp_vec7, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp7 = _mm_cvtsi128_si32(sum32);
            temp7 += bias2;
            out_to_compare_with_Char[out_subscript+1+ (2 << out_l_shift)] = RELU(temp7);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec8), _mm256_extracti128_si256(temp_vec8, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp8 = _mm_cvtsi128_si32(sum32);
            temp8 += bias2;
            out_to_compare_with_Char[out_subscript+1+ (3 << out_l_shift)] = RELU(temp8);
          }
          // fallback loop
          for (;x < Output_X_dim; x++) {	//Output Width
            const unsigned long long int x_stride = x * Stride_X_dim;
            int temp, temp2;

            __m256i temp_vec = _mm256_setzero_si256();
            __m256i temp_vec2 = _mm256_setzero_si256();

            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              const unsigned long long int off_y_stride = (y_stride + off_y) * in_x_depth;
              const unsigned long long int off_y_mask = off_y * mask_x_depth;

              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                const unsigned long long int off_x_stride = (x_stride + off_x) << in_l_shift;
                const unsigned long long int off_x_depth = off_x << in_l_shift;

                for (unsigned int d = 0; d < Input_depth_dim; d+=64) {

                  unsigned long long int in_subscript = b_depth
                    + off_y_stride
                    + off_x_stride
                    + d;

                  unsigned long long int filter_subscript = m_mask
                    + off_y_mask
                    + off_x_depth
                    + d;

                   __m256i s = _mm256_load_si256(( __m256i*)&in_Char[in_subscript]);

                   __m256i w = _mm256_load_si256(( __m256i*)&filter_Char[filter_subscript]);
                   __m256i w2 = _mm256_load_si256(( __m256i*)&filter_Char[filter_subscript + mask_yx_depth]);


                  __m256i inter_vec = _mm256_maddubs_epi16(s,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec = _mm256_add_epi32(temp_vec, inter_vec);

                  __m256i inter_vec2 = _mm256_maddubs_epi16(s,w2);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec2 = _mm256_add_epi32(temp_vec2, inter_vec2);

                  // d+32 ------------------------------------------------------------
                  // |
                  // |
                  // v

                  // in_subscript += 32;

                  // filter_subscript += 32;

                  s = _mm256_load_si256((const __m256i*)&in_Char[in_subscript + 32]);

                  w = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript + 32]);
                  w2 = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript + 32 + mask_yx_depth]);


                  inter_vec = _mm256_maddubs_epi16(s,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec = _mm256_add_epi32(temp_vec, inter_vec);

                  inter_vec2 = _mm256_maddubs_epi16(s,w2);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec2 = _mm256_add_epi32(temp_vec2, inter_vec2);
                }
              }
            }

            unsigned long long int out_subscript = b_out
              + y_out
              + (x << out_l_shift)
              + m;




            // **************** optimised hadd ****************
            __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec), _mm256_extracti128_si256(temp_vec, 1));
            __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
            __m128i sum64 = _mm_add_epi32(hi64, sum128);
            __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            __m128i sum32 = _mm_add_epi32(sum64, hi32);
            temp = _mm_cvtsi128_si32(sum32);
            temp += bias;
            out_to_compare_with_Char[out_subscript] = RELU(temp);


            __m128i sum128_2 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec2), _mm256_extracti128_si256(temp_vec2, 1));
            __m128i hi64_2 = _mm_unpackhi_epi64(sum128_2, sum128_2);
            __m128i sum64_2 = _mm_add_epi32(hi64_2, sum128_2);
            __m128i hi32_2 = _mm_shuffle_epi32(sum64_2, _MM_SHUFFLE(2,3,0,1));
            __m128i sum32_2 = _mm_add_epi32(sum64_2, hi32_2);
            temp2 = _mm_cvtsi128_si32(sum32_2);
            temp2 += bias2;
            out_to_compare_with_Char[out_subscript + 1] = RELU(temp2);

          }
        }
        // fallback loop
        for (; m < Output_depth_dim; m++) { //channels
          const unsigned long long int m_mask = m * mask_yx_depth;

          const int bias = bias_array_Int[m];

          for (x = 0; x < x_bound; x+=4) {	//Output Width
            const unsigned long long int x_stride = x * Stride_X_dim;
            int temp, temp2, temp3, temp4;

            __m256i temp_vec = _mm256_setzero_si256();
            __m256i temp_vec2 = _mm256_setzero_si256();
            __m256i temp_vec3 = _mm256_setzero_si256();
            __m256i temp_vec4 = _mm256_setzero_si256();

            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              const unsigned long long int off_y_stride = (y_stride + off_y) * in_x_depth;
              const unsigned long long int off_y_mask = off_y * mask_x_depth;

              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                const unsigned long long int off_x_stride = (x_stride + off_x) << in_l_shift;
                const unsigned long long int off_x_depth = off_x << in_l_shift;

                for (unsigned int d = 0; d < Input_depth_dim; d+=64) {

                  unsigned long long int in_subscript = b_depth
                    + off_y_stride
                    + off_x_stride
                    + d;

                  unsigned long long int filter_subscript = m_mask
                    + off_y_mask
                    + off_x_depth
                    + d;

                   __m256i s = _mm256_load_si256(( __m256i*)&in_Char[in_subscript]);
                   __m256i s2 = _mm256_load_si256(( __m256i*)&in_Char[in_subscript + Input_depth_dim]);
                   __m256i s3 = _mm256_load_si256(( __m256i*)&in_Char[in_subscript + (2 << in_l_shift)]);
                   __m256i s4 = _mm256_load_si256(( __m256i*)&in_Char[in_subscript + (3 << in_l_shift)]);

                   __m256i w = _mm256_load_si256(( __m256i*)&filter_Char[filter_subscript]);


                  __m256i inter_vec = _mm256_maddubs_epi16(s,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec = _mm256_add_epi32(temp_vec, inter_vec);

                  __m256i inter_vec2 = _mm256_maddubs_epi16(s2,w);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec2 = _mm256_add_epi32(temp_vec2, inter_vec2);

                  inter_vec = _mm256_maddubs_epi16(s3,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec3 = _mm256_add_epi32(temp_vec3, inter_vec);

                  inter_vec2 = _mm256_maddubs_epi16(s4, w);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec4 = _mm256_add_epi32(temp_vec4, inter_vec2);

                  // d+32 ------------------------------------------------------------
                  // |
                  // |
                  // v

                  // in_subscript += 32;

                  // filter_subscript += 32;

                  s = _mm256_load_si256((const __m256i*)&in_Char[in_subscript + 32]);
                  s2 = _mm256_load_si256((const __m256i*)&in_Char[in_subscript + 32 + Input_depth_dim]);
                  s3 = _mm256_load_si256((const __m256i*)&in_Char[in_subscript + 32 + (2 << in_l_shift)]);
                  s4 = _mm256_load_si256((const __m256i*)&in_Char[in_subscript + 32 + (3 << in_l_shift)]);

                  w = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript + 32]);


                  inter_vec = _mm256_maddubs_epi16(s,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec = _mm256_add_epi32(temp_vec, inter_vec);

                  inter_vec2 = _mm256_maddubs_epi16(s2,w);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec2 = _mm256_add_epi32(temp_vec2, inter_vec2);

                  inter_vec = _mm256_maddubs_epi16(s3,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec3 = _mm256_add_epi32(temp_vec3, inter_vec);

                  inter_vec2 = _mm256_maddubs_epi16(s4, w);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec4 = _mm256_add_epi32(temp_vec4, inter_vec2);
                }
              }
            }

            unsigned long long int out_subscript = b_out
              + y_out
              + (x << out_l_shift)
              + m;
            // unsigned long long int out_subscript2 = out_subscript + Output_depth_dim;
            // unsigned long long int out_subscript3 = out_subscript + (2 << out_l_shift);
            // unsigned long long int out_subscript4 = out_subscript + (3 << out_l_shift);




            // **************** optimised hadd ****************
            __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec), _mm256_extracti128_si256(temp_vec, 1));
            __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
            __m128i sum64 = _mm_add_epi32(hi64, sum128);
            __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            __m128i sum32 = _mm_add_epi32(sum64, hi32);
            temp = _mm_cvtsi128_si32(sum32);
            temp += bias;
            out_to_compare_with_Char[out_subscript] = RELU(temp);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec2), _mm256_extracti128_si256(temp_vec2, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp2 = _mm_cvtsi128_si32(sum32);
            temp2 += bias;
            out_to_compare_with_Char[out_subscript + (1 << out_l_shift)] = RELU(temp2);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec3), _mm256_extracti128_si256(temp_vec3, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp3 = _mm_cvtsi128_si32(sum32);
            temp3 += bias;
            out_to_compare_with_Char[out_subscript + (2 << out_l_shift)] = RELU(temp3);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec4), _mm256_extracti128_si256(temp_vec4, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp4 = _mm_cvtsi128_si32(sum32);
            temp4 += bias;
            out_to_compare_with_Char[out_subscript + (3 << out_l_shift)] = RELU(temp4);

          }
          // fallback loop
          for (;x < Output_X_dim; x++) {	//Output Width
            const unsigned long long int x_stride = x * Stride_X_dim;
            int temp, temp2;

            __m256i temp_vec = _mm256_setzero_si256();

            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              const unsigned long long int off_y_stride = (y_stride + off_y) * in_x_depth;
              const unsigned long long int off_y_mask = off_y * mask_x_depth;

              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                const unsigned long long int off_x_stride = (x_stride + off_x) << in_l_shift;
                const unsigned long long int off_x_depth = off_x << in_l_shift;

                for (unsigned int d = 0; d < Input_depth_dim; d+=64) {

                  unsigned long long int in_subscript = b_depth
                    + off_y_stride
                    + off_x_stride
                    + d;

                  unsigned long long int filter_subscript = m_mask
                    + off_y_mask
                    + off_x_depth
                    + d;

                   __m256i s = _mm256_load_si256(( __m256i*)&in_Char[in_subscript]);

                   __m256i w = _mm256_load_si256(( __m256i*)&filter_Char[filter_subscript]);


                  __m256i inter_vec = _mm256_maddubs_epi16(s,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec = _mm256_add_epi32(temp_vec, inter_vec);

                  // d+32 ------------------------------------------------------------
                  // |
                  // |
                  // v

                  // in_subscript += 32;

                  // filter_subscript += 32;

                  s = _mm256_load_si256((const __m256i*)&in_Char[in_subscript + 32]);

                  w = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript + 32]);


                  inter_vec = _mm256_maddubs_epi16(s,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec = _mm256_add_epi32(temp_vec, inter_vec);

                }
              }
            }

            unsigned long long int out_subscript = b_out
              + y_out
              + (x << out_l_shift)
              + m;


            // **************** optimised hadd ****************
            __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec), _mm256_extracti128_si256(temp_vec, 1));
            __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
            __m128i sum64 = _mm_add_epi32(hi64, sum128);
            __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            __m128i sum32 = _mm_add_epi32(sum64, hi32);
            temp = _mm_cvtsi128_si32(sum32);
            temp += bias;
            out_to_compare_with_Char[out_subscript] = RELU(temp);

          }
        }
      }
    }
  }
  // #undef left_shift
  #undef RELU
  printf("\n from quantised opt m2x4 %d %d ", out_to_compare_with_Char[0], out_to_compare_with_Char[1]);
  return 0;
}







// taking best performing vectorised (d loop) func and applying to use Char
// ~748 GFLOPS
int opt_m2x4_Char(const unsigned char* in_Char, const signed char* filter_Char, const int* bias_array_Int, unsigned char* out_to_compare_with_Char) {
  #define left_shift(x) (__builtin_ctz(x))  // inline function effectively using inline assembly to return num of trailing zeros (i.e. 128 returns 7, 2 * 128 == 2 << 7)
  #define RELU(x) ((x) < 0 ? 0 : (x)) // return x if greater than 0, else return 0

  // float bias, bias2, bias3, bias4;
  // __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

  const unsigned long long int in_l_shift = left_shift(Input_depth_dim);
  const unsigned long long int out_l_shift = left_shift(Output_depth_dim);

  const unsigned long long int in_yx_depth = Input_Y_dim * Input_X_dim << in_l_shift;
  const unsigned long long int in_x_depth = Input_X_dim  << in_l_shift;

  const unsigned long long int mask_yx_depth = Mask_Y_dim * Mask_X_dim << in_l_shift;
  const unsigned long long int mask_x_depth = Mask_X_dim << in_l_shift;

  const unsigned long long int out_yx_depth = Output_X_dim * Output_Y_dim << out_l_shift;
  const unsigned long long int out_x_depth = Output_X_dim << out_l_shift;

  #pragma omp parallel for shared(in_yx_depth, in_x_depth, mask_yx_depth, mask_x_depth, out_yx_depth, out_x_depth) \
    default(shared) collapse(2) schedule(static)
  {
    for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
      const unsigned long long int b_depth = b * in_yx_depth;
      const unsigned long long int b_out = b * out_yx_depth;

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        const unsigned long long int y_stride = y * Stride_Y_dim;
        const unsigned long long int y_out = y * out_x_depth;

        for (unsigned int m = 0; m < Output_depth_dim; m+=2) { //channels
          const unsigned long long int m_mask = m * mask_yx_depth;

          const int bias = bias_array_Int[m];
          const int bias2 = bias_array_Int[m+1];

          for (unsigned int x = 0; x < Input_X_dim; x+=4) {	//Output Width
            const unsigned long long int x_stride = x * Stride_X_dim;
            int temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

            __m256i temp_vec = _mm256_setzero_si256();
            __m256i temp_vec2 = _mm256_setzero_si256();
            __m256i temp_vec3 = _mm256_setzero_si256();
            __m256i temp_vec4 = _mm256_setzero_si256();
            __m256i temp_vec5 = _mm256_setzero_si256();
            __m256i temp_vec6 = _mm256_setzero_si256();
            __m256i temp_vec7 = _mm256_setzero_si256();
            __m256i temp_vec8 = _mm256_setzero_si256();

            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              const unsigned long long int off_y_stride = (y_stride + off_y) * in_x_depth;
              const unsigned long long int off_y_mask = off_y * mask_x_depth;

              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                const unsigned long long int off_x_stride = (x_stride + off_x) << in_l_shift;
                const unsigned long long int off_x_depth = off_x << in_l_shift;

                for (unsigned int d = 0; d < Input_depth_dim; d+=64) {

                  unsigned long long int in_subscript = b_depth
                    + off_y_stride
                    + off_x_stride
                    + d;

                  unsigned long long int filter_subscript = m_mask
                    + off_y_mask
                    + off_x_depth
                    + d;

                   __m256i s = _mm256_load_si256(( __m256i*)&in_Char[in_subscript]);
                   __m256i s2 = _mm256_load_si256(( __m256i*)&in_Char[in_subscript + Input_depth_dim]);
                   __m256i s3 = _mm256_load_si256(( __m256i*)&in_Char[in_subscript + (2 << in_l_shift)]);
                   __m256i s4 = _mm256_load_si256(( __m256i*)&in_Char[in_subscript + (3 << in_l_shift)]);

                   __m256i w = _mm256_load_si256(( __m256i*)&filter_Char[filter_subscript]);
                   __m256i w2 = _mm256_load_si256(( __m256i*)&filter_Char[filter_subscript + mask_yx_depth]);


                  __m256i inter_vec = _mm256_maddubs_epi16(s,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec = _mm256_add_epi32(temp_vec, inter_vec);

                  __m256i inter_vec2 = _mm256_maddubs_epi16(s2,w);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec2 = _mm256_add_epi32(temp_vec2, inter_vec2);

                  inter_vec = _mm256_maddubs_epi16(s3,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec3 = _mm256_add_epi32(temp_vec3, inter_vec);

                  inter_vec2 = _mm256_maddubs_epi16(s4, w);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec4 = _mm256_add_epi32(temp_vec4, inter_vec2);

                  inter_vec = _mm256_maddubs_epi16(s,w2);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec5 = _mm256_add_epi32(temp_vec5, inter_vec);

                  inter_vec2 = _mm256_maddubs_epi16(s2, w2);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec6 = _mm256_add_epi32(temp_vec6, inter_vec2);

                  inter_vec = _mm256_maddubs_epi16(s3,w2);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec7 = _mm256_add_epi32(temp_vec7, inter_vec);

                  inter_vec2 = _mm256_maddubs_epi16(s4, w2);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec8 = _mm256_add_epi32(temp_vec8, inter_vec2);

                  // d+32 ------------------------------------------------------------
                  // |
                  // |
                  // v

                  // in_subscript += 32;

                  // filter_subscript += 32;

                  s = _mm256_load_si256((const __m256i*)&in_Char[in_subscript + 32]);
                  s2 = _mm256_load_si256((const __m256i*)&in_Char[in_subscript + 32 + Input_depth_dim]);
                  s3 = _mm256_load_si256((const __m256i*)&in_Char[in_subscript + 32 + (2 << in_l_shift)]);
                  s4 = _mm256_load_si256((const __m256i*)&in_Char[in_subscript + 32 + (3 << in_l_shift)]);

                  w = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript + 32]);
                  w2 = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript + 32 + mask_yx_depth]);


                  inter_vec = _mm256_maddubs_epi16(s,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec = _mm256_add_epi32(temp_vec, inter_vec);

                  inter_vec2 = _mm256_maddubs_epi16(s2,w);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec2 = _mm256_add_epi32(temp_vec2, inter_vec2);

                  inter_vec = _mm256_maddubs_epi16(s3,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec3 = _mm256_add_epi32(temp_vec3, inter_vec);

                  inter_vec2 = _mm256_maddubs_epi16(s4, w);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec4 = _mm256_add_epi32(temp_vec4, inter_vec2);

                  inter_vec = _mm256_maddubs_epi16(s,w2);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec5 = _mm256_add_epi32(temp_vec5, inter_vec);

                  inter_vec2 = _mm256_maddubs_epi16(s2, w2);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec6 = _mm256_add_epi32(temp_vec6, inter_vec2);

                  inter_vec = _mm256_maddubs_epi16(s3,w2);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec7 = _mm256_add_epi32(temp_vec7, inter_vec);

                  inter_vec2 = _mm256_maddubs_epi16(s4, w2);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec8 = _mm256_add_epi32(temp_vec8, inter_vec2);
                }
              }
            }

            unsigned long long int out_subscript = b_out
              + y_out
              + (x << out_l_shift)
              + m;
            // unsigned long long int out_subscript2 = out_subscript + Output_depth_dim;
            // unsigned long long int out_subscript3 = out_subscript + (2 << out_l_shift);
            // unsigned long long int out_subscript4 = out_subscript + (3 << out_l_shift);




            // **************** optimised hadd ****************
            __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec), _mm256_extracti128_si256(temp_vec, 1));
            __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
            __m128i sum64 = _mm_add_epi32(hi64, sum128);
            __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            __m128i sum32 = _mm_add_epi32(sum64, hi32);
            temp = _mm_cvtsi128_si32(sum32);
            temp += bias;
            out_to_compare_with_Char[out_subscript] = RELU(temp);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec2), _mm256_extracti128_si256(temp_vec2, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp2 = _mm_cvtsi128_si32(sum32);
            temp2 += bias;
            out_to_compare_with_Char[out_subscript + (1 << out_l_shift)] = RELU(temp2);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec3), _mm256_extracti128_si256(temp_vec3, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp3 = _mm_cvtsi128_si32(sum32);
            temp3 += bias;
            out_to_compare_with_Char[out_subscript + (2 << out_l_shift)] = RELU(temp3);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec4), _mm256_extracti128_si256(temp_vec4, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp4 = _mm_cvtsi128_si32(sum32);
            temp4 += bias;
            out_to_compare_with_Char[out_subscript + (3 << out_l_shift)] = RELU(temp4);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec5), _mm256_extracti128_si256(temp_vec5, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp5 = _mm_cvtsi128_si32(sum32);
            temp5 += bias2;
            out_to_compare_with_Char[out_subscript+1] = RELU(temp5);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec6), _mm256_extracti128_si256(temp_vec6, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp6 = _mm_cvtsi128_si32(sum32);
            temp6 += bias2;
            out_to_compare_with_Char[out_subscript+1+ (1 << out_l_shift)] = RELU(temp6);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec7), _mm256_extracti128_si256(temp_vec7, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp7 = _mm_cvtsi128_si32(sum32);
            temp7 += bias2;
            out_to_compare_with_Char[out_subscript+1+ (2 << out_l_shift)] = RELU(temp7);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec8), _mm256_extracti128_si256(temp_vec8, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp8 = _mm_cvtsi128_si32(sum32);
            temp8 += bias2;
            out_to_compare_with_Char[out_subscript+1+ (3 << out_l_shift)] = RELU(temp8);




            // temp_vec = _mm256_hadd_epi32(temp_vec, temp_vec);
            // temp_vec = _mm256_hadd_epi32(temp_vec, temp_vec);
            // __m128i tempLo = _mm256_castsi256_si128(temp_vec);
            // __m128i tempHi = _mm256_extracti128_si256(temp_vec, 1);
            // __m128 sum = _mm_add_epi32(tempLo, tempHi);
            // temp = _mm_cvtsi128_si32(sum);
            // temp += bias;
            // out_to_compare_with_Char[out_subscript] = RELU(temp);


            // temp_vec2 = _mm256_hadd_epi32(temp_vec2, temp_vec2);
            // temp_vec2 = _mm256_hadd_epi32(temp_vec2, temp_vec2);
            //  tempLo = _mm256_castsi256_si128(temp_vec2);
            //  tempHi = _mm256_extracti128_si256(temp_vec2, 1);
            //  sum = _mm_add_epi32(tempLo, tempHi);
            // temp2 = _mm_cvtsi128_si32(sum);
            // temp2 += bias;
            // out_to_compare_with_Char[out_subscript + (1 << out_l_shift)] = RELU(temp2);
            

            // temp_vec3 = _mm256_hadd_epi32(temp_vec3, temp_vec3);
            // temp_vec3 = _mm256_hadd_epi32(temp_vec3, temp_vec3);
            // tempLo = _mm256_castsi256_si128(temp_vec3);
            // tempHi = _mm256_extractf128_si256(temp_vec3, 1);
            // sum = _mm_add_epi32(tempLo, tempHi);
            // temp3 = _mm_cvtsi128_si32(sum);
            // temp3 += bias;
            // out_to_compare_with_Char[out_subscript + (2 << out_l_shift)] = RELU(temp3);


            // temp_vec4 = _mm256_hadd_epi32(temp_vec4, temp_vec4);
            // temp_vec4 = _mm256_hadd_epi32(temp_vec4, temp_vec4);
            // tempLo = _mm256_castsi256_si128(temp_vec4);
            // tempHi = _mm256_extracti128_si256(temp_vec4, 1);
            // sum = _mm_add_epi32(tempLo, tempHi);
            // temp4 = _mm_cvtsi128_si32(sum);
            // temp4 += bias;
            // out_to_compare_with_Char[out_subscript + (3 << out_l_shift)] = RELU(temp4);


            // temp_vec5 = _mm256_hadd_epi32(temp_vec5, temp_vec5);
            // temp_vec5 = _mm256_hadd_epi32(temp_vec5, temp_vec5);
            // tempLo = _mm256_castsi256_si128(temp_vec5);
            // tempHi = _mm256_extractf128_si256(temp_vec5, 1);
            // sum = _mm_add_epi32(tempLo, tempHi);
            // temp5 = _mm_cvtsi128_si32(sum);
            // temp5 += bias2;
            // out_to_compare_with_Char[out_subscript+1] = RELU(temp5);


            // temp_vec6 = _mm256_hadd_epi32(temp_vec6, temp_vec6);
            // temp_vec6 = _mm256_hadd_epi32(temp_vec6, temp_vec6);
            // tempLo = _mm256_castsi256_si128(temp_vec6);
            // tempHi = _mm256_extracti128_si256(temp_vec6, 1);
            // sum = _mm_add_epi32(tempLo, tempHi);
            // temp6 = _mm_cvtsi128_si32(sum);
            // temp6 += bias2;
            // out_to_compare_with_Char[out_subscript+1 + (1 << out_l_shift)] = RELU(temp6);


            // temp_vec7 = _mm256_hadd_epi32(temp_vec7, temp_vec7);
            // temp_vec7 = _mm256_hadd_epi32(temp_vec7, temp_vec7);
            // tempLo = _mm256_castsi256_si128(temp_vec7);
            // tempHi = _mm256_extractf128_si256(temp_vec7, 1);
            // sum = _mm_add_epi32(tempLo, tempHi);
            // temp7 = _mm_cvtsi128_si32(sum);
            // temp7 += bias2;
            // out_to_compare_with_Char[out_subscript+1 + (2 << out_l_shift)] = RELU(temp7);


            // temp_vec8 = _mm256_hadd_epi32(temp_vec8, temp_vec8);
            // temp_vec8 = _mm256_hadd_epi32(temp_vec8, temp_vec8);
            // tempLo = _mm256_castsi256_si128(temp_vec8);
            // tempHi = _mm256_extracti128_si256(temp_vec8, 1);
            // sum = _mm_add_epi32(tempLo, tempHi);
            // temp8 = _mm_cvtsi128_si32(sum);
            // temp8 += bias2;
            // out_to_compare_with_Char[out_subscript+1 + (3 << out_l_shift)] = RELU(temp8);

          }
        }
      }
    }
  }
  #undef left_shift
  #undef RELU
  printf("\n from quantised opt m2x4 %d %d ", out_to_compare_with_Char[0], out_to_compare_with_Char[1]);
  return 0;
}


// same as above, but unrolled m by 4, and x by 2
// ~746 GFLOPS
int opt_m4x2_Char(const unsigned char* in_Char, const signed char* filter_Char, const int* bias_array_Int, unsigned char* out_to_compare_with_Char) {
  #define left_shift(x) (__builtin_ctz(x))  // inline function effectively using inline assembly to return num of trailing zeros (i.e. 128 returns 7, 2 * 128 == 2 << 7)
  #define RELU(x) ((x) < 0 ? 0 : (x)) // return x if greater than 0, else return 0

  // float bias, bias2, bias3, bias4;
  // __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

  const unsigned long long int in_l_shift = left_shift(Input_depth_dim);
  const unsigned long long int out_l_shift = left_shift(Output_depth_dim);

  const unsigned long long int in_yx_depth = Input_Y_dim * Input_X_dim << in_l_shift;
  const unsigned long long int in_x_depth = Input_X_dim  << in_l_shift;

  const unsigned long long int mask_yx_depth = Mask_Y_dim * Mask_X_dim << in_l_shift;
  const unsigned long long int mask_x_depth = Mask_X_dim << in_l_shift;

  const unsigned long long int out_yx_depth = Output_X_dim * Output_Y_dim << out_l_shift;
  const unsigned long long int out_x_depth = Output_X_dim << out_l_shift;

  #pragma omp parallel for shared(in_l_shift, out_l_shift, in_yx_depth, in_x_depth, mask_yx_depth, mask_x_depth, out_yx_depth, out_x_depth) \
    default(shared) collapse(2) schedule(static)
  {
    for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
      const unsigned long long int b_depth = b * in_yx_depth;

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        const unsigned long long int y_stride = y * Stride_Y_dim;

        for (unsigned int m = 0; m < Output_depth_dim; m+=4) { //channels
          const unsigned long long int m_mask = m * mask_yx_depth;

          const int bias = bias_array_Int[m];
          const int bias2 = bias_array_Int[m+1];
          const int bias3 = bias_array_Int[m+2];
          const int bias4 = bias_array_Int[m+3];

          for (unsigned int x = 0; x < Output_X_dim; x+=2) {	//Output Width
            const unsigned long long int x_stride = x * Stride_X_dim;

            int temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

            __m256i temp_vec = _mm256_setzero_si256();
            __m256i temp_vec2 = _mm256_setzero_si256();
            __m256i temp_vec3 = _mm256_setzero_si256();
            __m256i temp_vec4 = _mm256_setzero_si256();
            __m256i temp_vec5 = _mm256_setzero_si256();
            __m256i temp_vec6 = _mm256_setzero_si256();
            __m256i temp_vec7 = _mm256_setzero_si256();
            __m256i temp_vec8 = _mm256_setzero_si256();

            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              const unsigned long long int off_y_stride = (y_stride + off_y) * in_x_depth;
              const unsigned long long int off_y_mask = off_y * mask_x_depth;

              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                const unsigned long long int off_x_stride = (x_stride + off_x) << in_l_shift;
                const unsigned long long int off_x_mask = off_x << in_l_shift;

                for (unsigned int d = 0; d < Input_depth_dim; d+=32) {

                  const unsigned long long int in_subscript = b_depth + off_y_stride + off_x_stride + d;  // b * (in_yx_depth)
                  //   + ((y * Stride_Y_dim + off_y) * in_x_depth)
                  //   + ((x * Stride_X_dim + off_x) << in_l_shift)
                  //   + d;

                  const unsigned long long int filter_subscript = m_mask + off_y_mask + off_x_mask + d;   // (m * mask_yx_depth)
                  //   + (off_y * mask_x_depth)
                  //   + (off_x << in_l_shift)
                  //   + d;

                  const __m256i s = _mm256_load_si256((const __m256i*)&in_Char[in_subscript]);
                  const __m256i s2 = _mm256_load_si256((const __m256i*)&in_Char[in_subscript + Input_depth_dim]);

                  const __m256i w = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript]);
                  const __m256i w2 = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript + mask_yx_depth]);
                  const __m256i w3 = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript + (2 * mask_yx_depth)]);
                  const __m256i w4 = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript + (3 * mask_yx_depth)]);


                  __m256i inter_vec = _mm256_maddubs_epi16(s,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec = _mm256_add_epi32(temp_vec, inter_vec);

                  __m256i inter_vec2 = _mm256_maddubs_epi16(s2,w);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec2 = _mm256_add_epi32(temp_vec2, inter_vec2);

                  inter_vec = _mm256_maddubs_epi16(s,w2);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec3 = _mm256_add_epi32(temp_vec3, inter_vec);

                  inter_vec2 = _mm256_maddubs_epi16(s2, w2);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec4 = _mm256_add_epi32(temp_vec4, inter_vec2);

                  inter_vec = _mm256_maddubs_epi16(s,w3);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec5 = _mm256_add_epi32(temp_vec5, inter_vec);

                  inter_vec2 = _mm256_maddubs_epi16(s2, w3);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec6 = _mm256_add_epi32(temp_vec6, inter_vec2);

                  inter_vec = _mm256_maddubs_epi16(s,w4);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec7 = _mm256_add_epi32(temp_vec7, inter_vec);

                  inter_vec2 = _mm256_maddubs_epi16(s2, w4);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec8 = _mm256_add_epi32(temp_vec8, inter_vec2);
                }
              }
            }


            unsigned long long int out_subscript = b * (out_yx_depth) 
              + y * (out_x_depth) 
              + (x << out_l_shift)
              + m;
            unsigned long long int out_subscript2 = out_subscript + Output_depth_dim;



            // **************** optimised hadd, results in lower perf due to register pressure? ****************
            __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec), _mm256_extracti128_si256(temp_vec, 1));
            __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
            __m128i sum64 = _mm_add_epi32(hi64, sum128);
            __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            __m128i sum32 = _mm_add_epi32(sum64, hi32);
            temp = _mm_cvtsi128_si32(sum32);
            temp += bias;
            out_to_compare_with_Char[out_subscript] = RELU(temp);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec2), _mm256_extracti128_si256(temp_vec2, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp2 = _mm_cvtsi128_si32(sum32);
            temp2 += bias;
            out_to_compare_with_Char[out_subscript2] = RELU(temp2);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec3), _mm256_extracti128_si256(temp_vec3, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp3 = _mm_cvtsi128_si32(sum32);
            temp3 += bias2;
            out_to_compare_with_Char[out_subscript+1] = RELU(temp3);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec4), _mm256_extracti128_si256(temp_vec4, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp4 = _mm_cvtsi128_si32(sum32);
            temp4 += bias2;
            out_to_compare_with_Char[out_subscript2+1] = RELU(temp4);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec5), _mm256_extracti128_si256(temp_vec5, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp5 = _mm_cvtsi128_si32(sum32);
            temp5 += bias3;
            out_to_compare_with_Char[out_subscript+2] = RELU(temp5);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec6), _mm256_extracti128_si256(temp_vec6, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp6 = _mm_cvtsi128_si32(sum32);
            temp6 += bias3;
            out_to_compare_with_Char[out_subscript2+2] = RELU(temp6);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec7), _mm256_extracti128_si256(temp_vec7, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp7 = _mm_cvtsi128_si32(sum32);
            temp7 += bias4;
            out_to_compare_with_Char[out_subscript+3] = RELU(temp7);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec8), _mm256_extracti128_si256(temp_vec8, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp8 = _mm_cvtsi128_si32(sum32);
            temp8 += bias4;
            out_to_compare_with_Char[out_subscript2+3] = RELU(temp8);



            // temp_vec = _mm256_hadd_epi32(temp_vec, temp_vec);
            // temp_vec = _mm256_hadd_epi32(temp_vec, temp_vec);
            // __m128i tempLo = _mm256_castsi256_si128(temp_vec);
            // __m128i tempHi = _mm256_extracti128_si256(temp_vec, 1);
            // __m128 sum = _mm_add_epi32(tempLo, tempHi);
            // temp = _mm_cvtsi128_si32(sum);
            // temp += bias;
            // out_to_compare_with_Char[out_subscript] = RELU(temp);


            // temp_vec2 = _mm256_hadd_epi32(temp_vec2, temp_vec2);
            // temp_vec2 = _mm256_hadd_epi32(temp_vec2, temp_vec2);
            // __m128i tempLo2 = _mm256_castsi256_si128(temp_vec2);
            // __m128i tempHi2 = _mm256_extracti128_si256(temp_vec2, 1);
            // __m128 sum2 = _mm_add_epi32(tempLo2, tempHi2);
            // temp2 = _mm_cvtsi128_si32(sum2);
            // temp2 += bias;
            // out_to_compare_with_Char[out_subscript2] = RELU(temp2);
            

            // temp_vec3 = _mm256_hadd_epi32(temp_vec3, temp_vec3);
            // temp_vec3 = _mm256_hadd_epi32(temp_vec3, temp_vec3);
            // tempLo = _mm256_castsi256_si128(temp_vec3);
            // tempHi = _mm256_extractf128_si256(temp_vec3, 1);
            // sum = _mm_add_epi32(tempLo, tempHi);
            // temp3 = _mm_cvtsi128_si32(sum);
            // temp3 += bias2;
            // out_to_compare_with_Char[out_subscript+1] = RELU(temp3);


            // temp_vec4 = _mm256_hadd_epi32(temp_vec4, temp_vec4);
            // temp_vec4 = _mm256_hadd_epi32(temp_vec4, temp_vec4);
            // tempLo2 = _mm256_castsi256_si128(temp_vec4);
            // tempHi2 = _mm256_extracti128_si256(temp_vec4, 1);
            // sum2 = _mm_add_epi32(tempLo, tempHi);
            // temp4 = _mm_cvtsi128_si32(sum2);
            // temp4 += bias2;
            // out_to_compare_with_Char[out_subscript2+1] = RELU(temp4);


            // temp_vec5 = _mm256_hadd_epi32(temp_vec5, temp_vec5);
            // temp_vec5 = _mm256_hadd_epi32(temp_vec5, temp_vec5);
            // tempLo = _mm256_castsi256_si128(temp_vec5);
            // tempHi = _mm256_extractf128_si256(temp_vec5, 1);
            // sum = _mm_add_epi32(tempLo, tempHi);
            // temp5 = _mm_cvtsi128_si32(sum);
            // temp5 += bias3;
            // out_to_compare_with_Char[out_subscript+2] = RELU(temp5);


            // temp_vec6 = _mm256_hadd_epi32(temp_vec6, temp_vec6);
            // temp_vec6 = _mm256_hadd_epi32(temp_vec6, temp_vec6);
            // tempLo2 = _mm256_castsi256_si128(temp_vec6);
            // tempHi2 = _mm256_extracti128_si256(temp_vec6, 1);
            // sum2 = _mm_add_epi32(tempLo, tempHi);
            // temp6 = _mm_cvtsi128_si32(sum2);
            // temp6 += bias3;
            // out_to_compare_with_Char[out_subscript2+2] = RELU(temp6);


            // temp_vec7 = _mm256_hadd_epi32(temp_vec7, temp_vec7);
            // temp_vec7 = _mm256_hadd_epi32(temp_vec7, temp_vec7);
            // tempLo = _mm256_castsi256_si128(temp_vec7);
            // tempHi = _mm256_extractf128_si256(temp_vec7, 1);
            // sum = _mm_add_epi32(tempLo, tempHi);
            // temp7 = _mm_cvtsi128_si32(sum);
            // temp7 += bias4;
            // out_to_compare_with_Char[out_subscript+3] = RELU(temp7);


            // temp_vec8 = _mm256_hadd_epi32(temp_vec8, temp_vec8);
            // temp_vec8 = _mm256_hadd_epi32(temp_vec8, temp_vec8);
            // tempLo2 = _mm256_castsi256_si128(temp_vec8);
            // tempHi2 = _mm256_extracti128_si256(temp_vec8, 1);
            // sum2 = _mm_add_epi32(tempLo, tempHi);
            // temp8 = _mm_cvtsi128_si32(sum2);
            // temp8 += bias4;
            // out_to_compare_with_Char[out_subscript2+3] = RELU(temp8);

          }
        }
      }
    }
  }
  #undef left_shift
  #undef RELU
  printf("\n from quantised opt m4x2 %d %d ", out_to_compare_with_Char[0], out_to_compare_with_Char[1]);
  return 0;
}


// unrolled m and x by 2
// ~741 GFLOPS
int opt_m2x2_Char(const unsigned char* in_Char, const signed char* filter_Char, const int* bias_array_Int, unsigned char* out_to_compare_with_Char) {
  #define left_shift(x) (__builtin_ctz(x))  // inline function effectively using inline assembly to return num of trailing zeros (i.e. 128 returns 7, 2 * 128 == 2 << 7)
  #define RELU(x) ((x) < 0 ? 0 : (x)) // return x if greater than 0, else return 0

  // float bias, bias2, bias3, bias4;
  // __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

  const unsigned long long int in_l_shift = left_shift(Input_depth_dim);
  const unsigned long long int out_l_shift = left_shift(Output_depth_dim);

  const unsigned long long int in_yx_depth = Input_Y_dim * Input_X_dim << in_l_shift;
  const unsigned long long int in_x_depth = Input_X_dim  << in_l_shift;

  const unsigned long long int mask_yx_depth = Mask_Y_dim * Mask_X_dim << in_l_shift;
  const unsigned long long int mask_x_depth = Mask_X_dim << in_l_shift;

  const unsigned long long int out_yx_depth = Output_X_dim * Output_Y_dim << out_l_shift;
  const unsigned long long int out_x_depth = Output_X_dim << out_l_shift;

  #pragma omp parallel for shared(in_yx_depth, in_x_depth, mask_yx_depth, mask_x_depth, out_yx_depth, out_x_depth) \
    default(shared) collapse(2) schedule(static)
  {
    for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
      const unsigned long long int b_depth = b * in_yx_depth;
      const unsigned long long int b_out = b * out_yx_depth;

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        const unsigned long long int y_stride = y * Stride_Y_dim;
        const unsigned long long int y_out = y * out_x_depth;

        for (unsigned int m = 0; m < Output_depth_dim; m+=2) { //channels
          const unsigned long long int m_mask = m * mask_yx_depth;

          const int bias = bias_array_Int[m];
          const int bias2 = bias_array_Int[m+1];

          for (unsigned int x = 0; x < Output_X_dim; x+=2) {	//Output Width
            const unsigned long long int x_stride = x * Stride_X_dim;
            int temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

            __m256i temp_vec = _mm256_setzero_si256();
            __m256i temp_vec2 = _mm256_setzero_si256();
            __m256i temp_vec3 = _mm256_setzero_si256();
            __m256i temp_vec4 = _mm256_setzero_si256();

            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              const unsigned long long int off_y_stride = (y_stride + off_y) * in_x_depth;
              const unsigned long long int off_y_mask = off_y * mask_x_depth;

              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                const unsigned long long int off_x_stride = (x_stride + off_x) << in_l_shift;
                const unsigned long long int off_x_depth = off_x << in_l_shift;

                for (unsigned int d = 0; d < Input_depth_dim; d+=64) {

                  unsigned long long int in_subscript = b_depth
                    + off_y_stride
                    + off_x_stride
                    + d;

                  unsigned long long int filter_subscript = m_mask
                    + off_y_mask
                    + off_x_depth
                    + d;

                   __m256i s = _mm256_load_si256((const __m256i*)&in_Char[in_subscript]);
                   __m256i s2 = _mm256_load_si256((const __m256i*)&in_Char[in_subscript + Input_depth_dim]);

                   __m256i w = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript]);
                   __m256i w2 = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript + mask_yx_depth]);


                  __m256i inter_vec = _mm256_maddubs_epi16(s,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec = _mm256_add_epi32(temp_vec, inter_vec);

                  __m256i inter_vec2 = _mm256_maddubs_epi16(s2,w);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec2 = _mm256_add_epi32(temp_vec2, inter_vec2);

                  __m256i inter_vec3 = _mm256_maddubs_epi16(s,w2);
                  inter_vec3 = _mm256_madd_epi16(inter_vec3, _mm256_set1_epi16(1));
                  temp_vec3 = _mm256_add_epi32(temp_vec3, inter_vec3);

                  __m256i inter_vec4 = _mm256_maddubs_epi16(s2, w2);
                  inter_vec4 = _mm256_madd_epi16(inter_vec4, _mm256_set1_epi16(1));
                  temp_vec4 = _mm256_add_epi32(temp_vec4, inter_vec4);


                  // d+32 ------------------------------------------------------------
                  // |
                  // |
                  // v

                  in_subscript += 32;

                  filter_subscript += 32;

                    s = _mm256_load_si256((const __m256i*)&in_Char[in_subscript]);
                    s2 = _mm256_load_si256((const __m256i*)&in_Char[in_subscript + Input_depth_dim]);

                    w = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript]);
                    w2 = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript + mask_yx_depth]);


                   inter_vec = _mm256_maddubs_epi16(s,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec = _mm256_add_epi32(temp_vec, inter_vec);

                   inter_vec2 = _mm256_maddubs_epi16(s2,w);
                  inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  temp_vec2 = _mm256_add_epi32(temp_vec2, inter_vec2);

                   inter_vec3 = _mm256_maddubs_epi16(s,w2);
                  inter_vec3 = _mm256_madd_epi16(inter_vec3, _mm256_set1_epi16(1));
                  temp_vec3 = _mm256_add_epi32(temp_vec3, inter_vec3);

                   inter_vec4 = _mm256_maddubs_epi16(s2, w2);
                  inter_vec4 = _mm256_madd_epi16(inter_vec4, _mm256_set1_epi16(1));
                  temp_vec4 = _mm256_add_epi32(temp_vec4, inter_vec4);
                }
              }
            }

            unsigned long long int out_subscript = b_out
              + y_out
              + (x << out_l_shift)
              + m;
            unsigned long long int out_subscript2 = out_subscript + Output_depth_dim;
            unsigned long long int out_subscript3 = out_subscript + 1;
            unsigned long long int out_subscript4 = out_subscript + 1 + Output_depth_dim;



            // **************** optimised hadd, results in lower perf due to register pressure? ****************
            __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec), _mm256_extracti128_si256(temp_vec, 1));
            __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
            __m128i sum64 = _mm_add_epi32(hi64, sum128);
            __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            __m128i sum32 = _mm_add_epi32(sum64, hi32);
            temp = _mm_cvtsi128_si32(sum32);
            temp += bias;
            out_to_compare_with_Char[out_subscript] = RELU(temp);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec2), _mm256_extracti128_si256(temp_vec2, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp2 = _mm_cvtsi128_si32(sum32);
            temp2 += bias;
            out_to_compare_with_Char[out_subscript2] = RELU(temp2);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec3), _mm256_extracti128_si256(temp_vec3, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp3 = _mm_cvtsi128_si32(sum32);
            temp3 += bias2;
            out_to_compare_with_Char[out_subscript3] = RELU(temp3);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec4), _mm256_extracti128_si256(temp_vec4, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp4 = _mm_cvtsi128_si32(sum32);
            temp4 += bias2;
            out_to_compare_with_Char[out_subscript4] = RELU(temp4);




            // temp_vec = _mm256_hadd_epi32(temp_vec, temp_vec);
            // temp_vec = _mm256_hadd_epi32(temp_vec, temp_vec);
            // __m128i tempLo = _mm256_castsi256_si128(temp_vec);
            // __m128i tempHi = _mm256_extracti128_si256(temp_vec, 1);
            // __m128 sum = _mm_add_epi32(tempLo, tempHi);
            // temp = _mm_cvtsi128_si32(sum);
            // temp += bias;
            // out_to_compare_with_Char[out_subscript] = RELU(temp);


            // temp_vec2 = _mm256_hadd_epi32(temp_vec2, temp_vec2);
            // temp_vec2 = _mm256_hadd_epi32(temp_vec2, temp_vec2);
            // __m128i tempLo2 = _mm256_castsi256_si128(temp_vec2);
            // __m128i tempHi2 = _mm256_extracti128_si256(temp_vec2, 1);
            // __m128 sum2 = _mm_add_epi32(tempLo2, tempHi2);
            // temp2 = _mm_cvtsi128_si32(sum2);
            // temp2 += bias;
            // out_to_compare_with_Char[out_subscript2] = RELU(temp2);
            

            // temp_vec3 = _mm256_hadd_epi32(temp_vec3, temp_vec3);
            // temp_vec3 = _mm256_hadd_epi32(temp_vec3, temp_vec3);
            // tempLo = _mm256_castsi256_si128(temp_vec3);
            // tempHi = _mm256_extractf128_si256(temp_vec3, 1);
            // sum = _mm_add_epi32(tempLo, tempHi);
            // temp3 = _mm_cvtsi128_si32(sum);
            // temp3 += bias2;
            // out_to_compare_with_Char[out_subscript3] = RELU(temp3);


            // temp_vec4 = _mm256_hadd_epi32(temp_vec4, temp_vec4);
            // temp_vec4 = _mm256_hadd_epi32(temp_vec4, temp_vec4);
            // tempLo2 = _mm256_castsi256_si128(temp_vec4);
            // tempHi2 = _mm256_extracti128_si256(temp_vec4, 1);
            // sum2 = _mm_add_epi32(tempLo2, tempHi2);
            // temp4 = _mm_cvtsi128_si32(sum2);
            // temp4 += bias2;
            // out_to_compare_with_Char[out_subscript4] = RELU(temp4);

          }
        }
      }
    }
  }
  #undef left_shift
  #undef RELU
  printf("\n from quantised opt m2x2 %d %d ", out_to_compare_with_Char[0], out_to_compare_with_Char[1]);
  return 0;
}


// *** not working ***
// unrolled m and x by 3
// ~ GFLOPS
int opt_m3x3_Char(const unsigned char* in_Char, const signed char* filter_Char, const int* bias_array_Int, unsigned char* out_to_compare_with_Char) {
  #define left_shift(x) (__builtin_ctz(x))  // inline function effectively using inline assembly to return num of trailing zeros (i.e. 128 returns 7, 2 * 128 == 2 << 7)
  #define RELU(x) ((x) < 0 ? 0 : (x)) // return x if greater than 0, else return 0

  // float bias, bias2, bias3, bias4;
  // __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

  const unsigned long long int in_l_shift = left_shift(Input_depth_dim);
  const unsigned long long int out_l_shift = left_shift(Output_depth_dim);

  const unsigned long long int in_yx_depth = Input_Y_dim * Input_X_dim << in_l_shift;
  const unsigned long long int in_x_depth = Input_X_dim  << in_l_shift;

  const unsigned long long int mask_yx_depth = Mask_Y_dim * Mask_X_dim << in_l_shift;
  const unsigned long long int mask_x_depth = Mask_X_dim << in_l_shift;

  const unsigned long long int out_yx_depth = Output_X_dim * Output_Y_dim << out_l_shift;
  const unsigned long long int out_x_depth = Output_X_dim << out_l_shift;



  // Calculate loop bounds for unrolled loops
  const unsigned int m_bound = (Output_depth_dim/ 3) * 3;
  const unsigned int x_bound = (Output_X_dim/ 3) * 3; 
  const unsigned int x_bound2 = (Output_X_dim/ 4) * 4; 

  unsigned int m, x;

  #pragma omp parallel for private(m, x) shared(m_bound, x_bound, in_l_shift, out_l_shift, in_yx_depth, in_x_depth, mask_yx_depth, mask_x_depth, out_yx_depth, out_x_depth) \
    default(shared) collapse(2) schedule(static)
  {
    for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
      const unsigned long long int b_depth = b * in_yx_depth;
      const unsigned long long int b_out = b * out_yx_depth;

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        const unsigned long long int y_stride = y * Stride_Y_dim;
        const unsigned long long int y_out = y * out_x_depth;

        for (m = 0; m < m_bound; m+=3) { //channels
          const unsigned long long int m_mask = m * mask_yx_depth;

          const int bias = bias_array_Int[m];
          const int bias2 = bias_array_Int[m+1];
          const int bias3 = bias_array_Int[m+2];

          for (x = 0; x < x_bound; x+=3) {	//Output Width
            const unsigned long long int x_stride = x * Stride_X_dim;
            int temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9;

            __m256i temp_vec = _mm256_setzero_si256();
            __m256i temp_vec2 = _mm256_setzero_si256();
            __m256i temp_vec3 = _mm256_setzero_si256();
            __m256i temp_vec4 = _mm256_setzero_si256();
            __m256i temp_vec5 = _mm256_setzero_si256();
            __m256i temp_vec6 = _mm256_setzero_si256();
            __m256i temp_vec7 = _mm256_setzero_si256();
            __m256i temp_vec8 = _mm256_setzero_si256();
            __m256i temp_vec9 = _mm256_setzero_si256();

            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              const unsigned long long int off_y_stride = (y_stride + off_y) * in_x_depth;
              const unsigned long long int off_y_mask = off_y * mask_x_depth;

              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                const unsigned long long int off_x_stride = (x_stride + off_x) << in_l_shift;
                const unsigned long long int off_x_depth = off_x << in_l_shift;

                for (unsigned int d = 0; d < Input_depth_dim; d+=64) {

                  unsigned long long int in_subscript = b_depth
                    + off_y_stride
                    + off_x_stride
                    + d;

                  unsigned long long int filter_subscript = m_mask
                    + off_y_mask
                    + off_x_depth
                    + d;

                   __m256i s = _mm256_load_si256((const __m256i*)&in_Char[in_subscript]);
                   __m256i s2 = _mm256_load_si256((const __m256i*)&in_Char[in_subscript + Input_depth_dim]);
                   __m256i s3 = _mm256_load_si256((const __m256i*)&in_Char[in_subscript + (2 << in_l_shift)]);

                   __m256i w = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript]);
                   __m256i w2 = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript + mask_yx_depth]);
                   __m256i w3 = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript + mask_yx_depth + mask_yx_depth]);


                  __m256i inter_vec = _mm256_maddubs_epi16(s,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec = _mm256_add_epi32(temp_vec, inter_vec);

                   inter_vec = _mm256_maddubs_epi16(s2,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec2 = _mm256_add_epi32(temp_vec2, inter_vec);

                  inter_vec = _mm256_maddubs_epi16(s3,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec3 = _mm256_add_epi32(temp_vec3, inter_vec);

                  inter_vec = _mm256_maddubs_epi16(s, w2);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec4 = _mm256_add_epi32(temp_vec4, inter_vec);

                  inter_vec = _mm256_maddubs_epi16(s2,w2);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec5 = _mm256_add_epi32(temp_vec5, inter_vec);

                  inter_vec = _mm256_maddubs_epi16(s3, w2);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec6 = _mm256_add_epi32(temp_vec6, inter_vec);

                  inter_vec = _mm256_maddubs_epi16(s,w3);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec7 = _mm256_add_epi32(temp_vec7, inter_vec);

                  inter_vec = _mm256_maddubs_epi16(s2, w3);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec8 = _mm256_add_epi32(temp_vec8, inter_vec);

                  inter_vec = _mm256_maddubs_epi16(s3, w3);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec9 = _mm256_add_epi32(temp_vec9, inter_vec);

                  // d+32 ------------------------------------------------------------
                  // |
                  // |
                  // v

                  // in_subscript += 32;

                  // filter_subscript += 32;

                  s = _mm256_load_si256((const __m256i*)&in_Char[in_subscript + 32]);
                  s2 = _mm256_load_si256((const __m256i*)&in_Char[in_subscript + 32 + Input_depth_dim]);
                  s3 = _mm256_load_si256((const __m256i*)&in_Char[in_subscript + 32 + (2 << in_l_shift)]);

                  w = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript + 32]);
                  w2 = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript + 32 + mask_yx_depth]);
                  w2 = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript + 32 + mask_yx_depth + mask_yx_depth]);


                  inter_vec = _mm256_maddubs_epi16(s,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec = _mm256_add_epi32(temp_vec, inter_vec);

                   inter_vec = _mm256_maddubs_epi16(s2,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec2 = _mm256_add_epi32(temp_vec2, inter_vec);

                  inter_vec = _mm256_maddubs_epi16(s3,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec3 = _mm256_add_epi32(temp_vec3, inter_vec);

                  inter_vec = _mm256_maddubs_epi16(s, w2);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec4 = _mm256_add_epi32(temp_vec4, inter_vec);

                  inter_vec = _mm256_maddubs_epi16(s2,w2);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec5 = _mm256_add_epi32(temp_vec5, inter_vec);

                  inter_vec = _mm256_maddubs_epi16(s3, w2);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec6 = _mm256_add_epi32(temp_vec6, inter_vec);

                  inter_vec = _mm256_maddubs_epi16(s,w3);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec7 = _mm256_add_epi32(temp_vec7, inter_vec);

                  inter_vec = _mm256_maddubs_epi16(s2, w3);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec8 = _mm256_add_epi32(temp_vec8, inter_vec);

                  inter_vec = _mm256_maddubs_epi16(s3, w3);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec9 = _mm256_add_epi32(temp_vec9, inter_vec);
                }
              }
            }

            unsigned long long int out_subscript = b_out
              + y_out
              + (x << out_l_shift)
              + m;
            unsigned long long int out_subscript2 = out_subscript + Output_depth_dim;
            unsigned long long int out_subscript3 = out_subscript + (2 << out_l_shift);



            // **************** optimised hadd ****************
            __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec), _mm256_extracti128_si256(temp_vec, 1));
            __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
            __m128i sum64 = _mm_add_epi32(hi64, sum128);
            __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            __m128i sum32 = _mm_add_epi32(sum64, hi32);
            temp = _mm_cvtsi128_si32(sum32);
            temp += bias;
            out_to_compare_with_Char[out_subscript] = RELU(temp);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec2), _mm256_extracti128_si256(temp_vec2, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp2 = _mm_cvtsi128_si32(sum32);
            temp2 += bias;
            out_to_compare_with_Char[out_subscript2] = RELU(temp2);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec3), _mm256_extracti128_si256(temp_vec3, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp3 = _mm_cvtsi128_si32(sum32);
            temp3 += bias;
            out_to_compare_with_Char[out_subscript3] = RELU(temp3);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec4), _mm256_extracti128_si256(temp_vec4, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp4 = _mm_cvtsi128_si32(sum32);
            temp4 += bias2;
            out_to_compare_with_Char[out_subscript+1] = RELU(temp4);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec5), _mm256_extracti128_si256(temp_vec5, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp5 = _mm_cvtsi128_si32(sum32);
            temp5 += bias2;
            out_to_compare_with_Char[out_subscript2+1] = RELU(temp5);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec6), _mm256_extracti128_si256(temp_vec6, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp6 = _mm_cvtsi128_si32(sum32);
            temp6 += bias2;
            out_to_compare_with_Char[out_subscript3+1] = RELU(temp6);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec7), _mm256_extracti128_si256(temp_vec7, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp7 = _mm_cvtsi128_si32(sum32);
            temp7 += bias3;
            out_to_compare_with_Char[out_subscript+2] = RELU(temp7);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec8), _mm256_extracti128_si256(temp_vec8, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp8 = _mm_cvtsi128_si32(sum32);
            temp8 += bias3;
            out_to_compare_with_Char[out_subscript2+2] = RELU(temp8);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec9), _mm256_extracti128_si256(temp_vec9, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp9 = _mm_cvtsi128_si32(sum32);
            temp9 += bias3;
            out_to_compare_with_Char[out_subscript3+2] = RELU(temp9);

          }
          for (; x < Output_X_dim; x++) {
            const unsigned long long int x_stride = x * Stride_X_dim;
            int temp, temp2, temp3;

            __m256i temp_vec = _mm256_setzero_si256();
            __m256i temp_vec2 = _mm256_setzero_si256();
            __m256i temp_vec3 = _mm256_setzero_si256();

            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              const unsigned long long int off_y_stride = (y_stride + off_y) * in_x_depth;
              const unsigned long long int off_y_mask = off_y * mask_x_depth;

              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                const unsigned long long int off_x_stride = (x_stride + off_x) << in_l_shift;
                const unsigned long long int off_x_depth = off_x << in_l_shift;

                for (unsigned int d = 0; d < Input_depth_dim; d+=64) {

                  unsigned long long int in_subscript = b_depth
                    + off_y_stride
                    + off_x_stride
                    + d;

                  unsigned long long int filter_subscript = m_mask
                    + off_y_mask
                    + off_x_depth
                    + d;

                   __m256i s = _mm256_load_si256((const __m256i*)&in_Char[in_subscript]);

                   __m256i w = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript]);
                   __m256i w2 = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript + mask_yx_depth]);
                   __m256i w3 = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript + mask_yx_depth + mask_yx_depth]);


                  __m256i inter_vec = _mm256_maddubs_epi16(s,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec = _mm256_add_epi32(temp_vec, inter_vec);

                   inter_vec = _mm256_maddubs_epi16(s,w2);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec2 = _mm256_add_epi32(temp_vec2, inter_vec);

                  inter_vec = _mm256_maddubs_epi16(s,w3);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec3 = _mm256_add_epi32(temp_vec3, inter_vec);


                  // d+32 ------------------------------------------------------------
                  // |
                  // |
                  // v

                  // in_subscript += 32;

                  // filter_subscript += 32;

                  s = _mm256_load_si256((const __m256i*)&in_Char[in_subscript + 32]);

                  w = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript + 32]);
                  w2 = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript + 32 + mask_yx_depth]);
                  w2 = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript + 32 + mask_yx_depth + mask_yx_depth]);


                  inter_vec = _mm256_maddubs_epi16(s,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec = _mm256_add_epi32(temp_vec, inter_vec);

                   inter_vec = _mm256_maddubs_epi16(s,w2);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec2 = _mm256_add_epi32(temp_vec2, inter_vec);

                  inter_vec = _mm256_maddubs_epi16(s,w3);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec3 = _mm256_add_epi32(temp_vec3, inter_vec);
                }
              }
            }

            unsigned long long int out_subscript = b_out
              + y_out
              + (x << out_l_shift)
              + m;
            unsigned long long int out_subscript2 = out_subscript + 1;
            unsigned long long int out_subscript3 = out_subscript + 2;



            // **************** optimised hadd ****************
            __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec), _mm256_extracti128_si256(temp_vec, 1));
            __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
            __m128i sum64 = _mm_add_epi32(hi64, sum128);
            __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            __m128i sum32 = _mm_add_epi32(sum64, hi32);
            temp = _mm_cvtsi128_si32(sum32);
            temp += bias;
            out_to_compare_with_Char[out_subscript] = RELU(temp);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec2), _mm256_extracti128_si256(temp_vec2, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp2 = _mm_cvtsi128_si32(sum32);
            temp2 += bias2;
            out_to_compare_with_Char[out_subscript2] = RELU(temp2);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec3), _mm256_extracti128_si256(temp_vec3, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp3 = _mm_cvtsi128_si32(sum32);
            temp3 += bias3;
            out_to_compare_with_Char[out_subscript3] = RELU(temp3);
          }
        }
        for (; m < Output_depth_dim; m++) {
          const unsigned long long int m_mask = m * mask_yx_depth;

          const int bias = bias_array_Int[m];

          for (x = 0; x < x_bound2; x+=4) {	//Output Width
            const unsigned long long int x_stride = x * Stride_X_dim;
            int temp, temp2, temp3, temp4;

            __m256i temp_vec = _mm256_setzero_si256();
            __m256i temp_vec2 = _mm256_setzero_si256();
            __m256i temp_vec3 = _mm256_setzero_si256();
            __m256i temp_vec4 = _mm256_setzero_si256();

            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              const unsigned long long int off_y_stride = (y_stride + off_y) * in_x_depth;
              const unsigned long long int off_y_mask = off_y * mask_x_depth;

              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                const unsigned long long int off_x_stride = (x_stride + off_x) << in_l_shift;
                const unsigned long long int off_x_depth = off_x << in_l_shift;

                for (unsigned int d = 0; d < Input_depth_dim; d+=64) {

                  unsigned long long int in_subscript = b_depth
                    + off_y_stride
                    + off_x_stride
                    + d;

                  unsigned long long int filter_subscript = m_mask
                    + off_y_mask
                    + off_x_depth
                    + d;

                   __m256i s = _mm256_load_si256((const __m256i*)&in_Char[in_subscript]);
                   __m256i s2 = _mm256_load_si256((const __m256i*)&in_Char[in_subscript + Input_depth_dim]);
                   __m256i s3 = _mm256_load_si256((const __m256i*)&in_Char[in_subscript + (2 << in_l_shift)]);
                   __m256i s4 = _mm256_load_si256((const __m256i*)&in_Char[in_subscript + (3 << in_l_shift)]);

                   __m256i w = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript]);


                  __m256i inter_vec = _mm256_maddubs_epi16(s,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec = _mm256_add_epi32(temp_vec, inter_vec);

                   inter_vec = _mm256_maddubs_epi16(s2,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec2 = _mm256_add_epi32(temp_vec2, inter_vec);

                  inter_vec = _mm256_maddubs_epi16(s3,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec3 = _mm256_add_epi32(temp_vec3, inter_vec);

                  inter_vec = _mm256_maddubs_epi16(s4, w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec4 = _mm256_add_epi32(temp_vec4, inter_vec);

                  // d+32 ------------------------------------------------------------
                  // |
                  // |
                  // v

                  // in_subscript += 32;

                  // filter_subscript += 32;

                    s = _mm256_load_si256((const __m256i*)&in_Char[in_subscript + 32]);
                    s2 = _mm256_load_si256((const __m256i*)&in_Char[in_subscript + 32 + Input_depth_dim]);
                    s3 = _mm256_load_si256((const __m256i*)&in_Char[in_subscript + 32 + (2 << in_l_shift)]);
                    s4 = _mm256_load_si256((const __m256i*)&in_Char[in_subscript + 32 + (3 << in_l_shift)]);

                    w = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript + 32]);


                   inter_vec = _mm256_maddubs_epi16(s,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec = _mm256_add_epi32(temp_vec, inter_vec);

                   inter_vec = _mm256_maddubs_epi16(s2,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec2 = _mm256_add_epi32(temp_vec2, inter_vec);

                  inter_vec = _mm256_maddubs_epi16(s3,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec3 = _mm256_add_epi32(temp_vec3, inter_vec);

                  inter_vec = _mm256_maddubs_epi16(s4, w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec4 = _mm256_add_epi32(temp_vec4, inter_vec);

                }
              }
            }

            unsigned long long int out_subscript = b_out
              + y_out
              + (x << out_l_shift)
              + m;
            unsigned long long int out_subscript2 = out_subscript + Output_depth_dim;
            unsigned long long int out_subscript3 = out_subscript + (2 << out_l_shift);
            unsigned long long int out_subscript4 = out_subscript + (3 << out_l_shift);



            // **************** optimised hadd, results in lower perf due to register pressure? ****************
            __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec), _mm256_extracti128_si256(temp_vec, 1));
            __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
            __m128i sum64 = _mm_add_epi32(hi64, sum128);
            __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            __m128i sum32 = _mm_add_epi32(sum64, hi32);
            temp = _mm_cvtsi128_si32(sum32);
            temp += bias;
            out_to_compare_with_Char[out_subscript] = RELU(temp);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec2), _mm256_extracti128_si256(temp_vec2, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp2 = _mm_cvtsi128_si32(sum32);
            temp2 += bias;
            out_to_compare_with_Char[out_subscript2] = RELU(temp2);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec3), _mm256_extracti128_si256(temp_vec3, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp3 = _mm_cvtsi128_si32(sum32);
            temp3 += bias;
            out_to_compare_with_Char[out_subscript3] = RELU(temp3);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec4), _mm256_extracti128_si256(temp_vec4, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp4 = _mm_cvtsi128_si32(sum32);
            temp4 += bias;
            out_to_compare_with_Char[out_subscript4] = RELU(temp4);
          }
          for (; x < Output_X_dim; x++) {
            const unsigned long long int x_stride = x * Stride_X_dim;
            int temp;

            __m256i temp_vec = _mm256_setzero_si256();

            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              const unsigned long long int off_y_stride = (y_stride + off_y) * in_x_depth;
              const unsigned long long int off_y_mask = off_y * mask_x_depth;

              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                const unsigned long long int off_x_stride = (x_stride + off_x) << in_l_shift;
                const unsigned long long int off_x_depth = off_x << in_l_shift;

                for (unsigned int d = 0; d < Input_depth_dim; d+=64) {

                  unsigned long long int in_subscript = b_depth
                    + off_y_stride
                    + off_x_stride
                    + d;

                  unsigned long long int filter_subscript = m_mask
                    + off_y_mask
                    + off_x_depth
                    + d;

                   __m256i s = _mm256_load_si256((const __m256i*)&in_Char[in_subscript]);

                   __m256i w = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript]);


                  __m256i inter_vec = _mm256_maddubs_epi16(s,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec = _mm256_add_epi32(temp_vec, inter_vec);

                  // d+32 ------------------------------------------------------------
                  // |
                  // |
                  // v

                  // in_subscript += 32;

                  // filter_subscript += 32;

                    s = _mm256_load_si256((const __m256i*)&in_Char[in_subscript + 32]);

                    w = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript + 32]);


                   inter_vec = _mm256_maddubs_epi16(s,w);
                  inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  temp_vec = _mm256_add_epi32(temp_vec, inter_vec);

                }
              }
            }

            unsigned long long int out_subscript = b_out
              + y_out
              + (x << out_l_shift)
              + m;

            // **************** optimised hadd ****************
            __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec), _mm256_extracti128_si256(temp_vec, 1));
            __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
            __m128i sum64 = _mm_add_epi32(hi64, sum128);
            __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            __m128i sum32 = _mm_add_epi32(sum64, hi32);
            temp = _mm_cvtsi128_si32(sum32);
            temp += bias;
            out_to_compare_with_Char[out_subscript] = RELU(temp);
          }
        }
      }
    }
  }
  #undef left_shift
  #undef RELU
  printf("\n from quantised opt m3x3 %d %d ", out_to_compare_with_Char[0], out_to_compare_with_Char[1]);
  return 0;
}


// *** not working ***
// working on 8 values at a time, instead of 32
// ~ GFLOPS
int opt_m2x4_d8_Char(const unsigned char* in_Char, const signed char* filter_Char, const int* bias_array_Int, unsigned char* out_to_compare_with_Char) {
  #define left_shift(x) (__builtin_ctz(x))  // inline function effectively using inline assembly to return num of trailing zeros (i.e. 128 returns 7, 2 * 128 == 2 << 7)
  #define RELU(x) ((x) < 0 ? 0 : (x)) // return x if greater than 0, else return 0

  // float bias, bias2, bias3, bias4;
  // __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

  const unsigned long long int in_l_shift = left_shift(Input_depth_dim);
  const unsigned long long int out_l_shift = left_shift(Output_depth_dim);

  const unsigned long long int in_yx_depth = Input_Y_dim * Input_X_dim << in_l_shift;
  const unsigned long long int in_x_depth = Input_X_dim  << in_l_shift;

  const unsigned long long int mask_yx_depth = Mask_Y_dim * Mask_X_dim << in_l_shift;
  const unsigned long long int mask_x_depth = Mask_X_dim << in_l_shift;

  const unsigned long long int out_yx_depth = Output_X_dim * Output_Y_dim << out_l_shift;
  const unsigned long long int out_x_depth = Output_X_dim << out_l_shift;

  #pragma omp parallel for shared(in_yx_depth, in_x_depth, mask_yx_depth, mask_x_depth, out_yx_depth, out_x_depth) \
    default(shared) collapse(2) schedule(static)
  {
    for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
      const unsigned long long int b_depth = b * in_yx_depth;
      const unsigned long long int b_out = b * out_yx_depth;

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        const unsigned long long int y_stride = y * Stride_Y_dim;
        const unsigned long long int y_out = y * out_x_depth;

        for (unsigned int m = 0; m < Output_depth_dim; m+=2) { //channels
          const unsigned long long int m_mask = m * mask_yx_depth;

          const int bias = bias_array_Int[m];
          const int bias2 = bias_array_Int[m+1];

          for (unsigned int x = 0; x < Output_X_dim; x+=4) {	//Output Width
            const unsigned long long int x_stride = x * Stride_X_dim;
            int temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8;

            __m256i temp_vec = _mm256_setzero_si256();
            __m256i temp_vec2 = _mm256_setzero_si256();
            __m256i temp_vec3 = _mm256_setzero_si256();
            __m256i temp_vec4 = _mm256_setzero_si256();
            __m256i temp_vec5 = _mm256_setzero_si256();
            __m256i temp_vec6 = _mm256_setzero_si256();
            __m256i temp_vec7 = _mm256_setzero_si256();
            __m256i temp_vec8 = _mm256_setzero_si256();

            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              const unsigned long long int off_y_stride = (y_stride + off_y) * in_x_depth;
              const unsigned long long int off_y_mask = off_y * mask_x_depth;

              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                const unsigned long long int off_x_stride = (x_stride + off_x) << in_l_shift;
                const unsigned long long int off_x_depth = off_x << in_l_shift;

                for (unsigned int d = 0; d < Input_depth_dim; d+=16) {

                  unsigned long long int in_subscript = b_depth
                    + off_y_stride
                    + off_x_stride
                    + d;

                  unsigned long long int filter_subscript = m_mask
                    + off_y_mask
                    + off_x_depth
                    + d;


                  __m256i s = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*)&in_Char[in_subscript]));
                  __m256i s2 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*)&in_Char[in_subscript + Input_depth_dim]));
                  __m256i s3 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*)&in_Char[in_subscript + (2 << in_l_shift)]));
                  __m256i s4 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*)&in_Char[in_subscript + (3 << in_l_shift)]));

                  __m256i w = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*)&filter_Char[filter_subscript]));
                  __m256i w2 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*)&filter_Char[filter_subscript + mask_yx_depth]));
                  
                  temp_vec = _mm256_add_epi32(temp_vec, _mm256_madd_epi16(s, w));
                  temp_vec2 = _mm256_add_epi32(temp_vec2, _mm256_madd_epi16(s2, w));
                  temp_vec3 = _mm256_add_epi32(temp_vec3, _mm256_madd_epi16(s3, w));
                  temp_vec4 = _mm256_add_epi32(temp_vec4, _mm256_madd_epi16(s4, w));
                  temp_vec5 = _mm256_add_epi32(temp_vec5, _mm256_madd_epi16(s, w2));
                  temp_vec6 = _mm256_add_epi32(temp_vec6, _mm256_madd_epi16(s2, w2));
                  temp_vec7 = _mm256_add_epi32(temp_vec7, _mm256_madd_epi16(s3, w2));
                  temp_vec8 = _mm256_add_epi32(temp_vec8, _mm256_madd_epi16(s4, w2));


                  // d+8 ------------------------------------------------------------
                  // |
                  // |
                  // v

                  // in_subscript += 8;

                  // filter_subscript += 8;

                   s = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*)&in_Char[in_subscript + 8]));
                   s2 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*)&in_Char[in_subscript + 8 + Input_depth_dim]));
                   s3 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*)&in_Char[in_subscript + 8 + (2 << in_l_shift)]));
                   s4 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*)&in_Char[in_subscript + 8 + (3 << in_l_shift)]));

                   w = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*)&filter_Char[filter_subscript + 8]));
                   w2 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*)&filter_Char[filter_subscript + 8 + mask_yx_depth]));
                  
                  temp_vec = _mm256_add_epi32(temp_vec, _mm256_madd_epi16(s, w));
                  temp_vec2 = _mm256_add_epi32(temp_vec2, _mm256_madd_epi16(s2, w));
                  temp_vec3 = _mm256_add_epi32(temp_vec3, _mm256_madd_epi16(s3, w));
                  temp_vec4 = _mm256_add_epi32(temp_vec4, _mm256_madd_epi16(s4, w));
                  temp_vec5 = _mm256_add_epi32(temp_vec5, _mm256_madd_epi16(s, w2));
                  temp_vec6 = _mm256_add_epi32(temp_vec6, _mm256_madd_epi16(s2, w2));
                  temp_vec7 = _mm256_add_epi32(temp_vec7, _mm256_madd_epi16(s3, w2));
                  temp_vec8 = _mm256_add_epi32(temp_vec8, _mm256_madd_epi16(s4, w2));


                  // __m256i inter_vec = _mm256_maddubs_epi16(s,w);
                  // inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  // temp_vec = _mm256_add_epi32(temp_vec, inter_vec);

                  // __m256i inter_vec2 = _mm256_maddubs_epi16(s2,w);
                  // inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  // temp_vec2 = _mm256_add_epi32(temp_vec2, inter_vec2);

                  // inter_vec = _mm256_maddubs_epi16(s3,w);
                  // inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  // temp_vec3 = _mm256_add_epi32(temp_vec3, inter_vec);

                  // inter_vec2 = _mm256_maddubs_epi16(s4, w);
                  // inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  // temp_vec4 = _mm256_add_epi32(temp_vec4, inter_vec2);

                  // inter_vec = _mm256_maddubs_epi16(s,w2);
                  // inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  // temp_vec5 = _mm256_add_epi32(temp_vec5, inter_vec);

                  // inter_vec2 = _mm256_maddubs_epi16(s2, w2);
                  // inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  // temp_vec6 = _mm256_add_epi32(temp_vec6, inter_vec2);

                  // inter_vec = _mm256_maddubs_epi16(s3,w2);
                  // inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  // temp_vec7 = _mm256_add_epi32(temp_vec7, inter_vec);

                  // inter_vec2 = _mm256_maddubs_epi16(s4, w2);
                  // inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  // temp_vec8 = _mm256_add_epi32(temp_vec8, inter_vec2);

                  // // d+32 ------------------------------------------------------------
                  // // |
                  // // |
                  // // v

                  // // in_subscript += 32;

                  // // filter_subscript += 32;

                  // s = _mm256_load_si256((const __m256i*)&in_Char[in_subscript + 32]);
                  // s2 = _mm256_load_si256((const __m256i*)&in_Char[in_subscript + 32 + Input_depth_dim]);
                  // s3 = _mm256_load_si256((const __m256i*)&in_Char[in_subscript + 32 + (2 << in_l_shift)]);
                  // s4 = _mm256_load_si256((const __m256i*)&in_Char[in_subscript + 32 + (3 << in_l_shift)]);

                  // w = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript + 32]);
                  // w2 = _mm256_load_si256((const __m256i*)&filter_Char[filter_subscript + 32 + mask_yx_depth]);


                  // inter_vec = _mm256_maddubs_epi16(s,w);
                  // inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  // temp_vec = _mm256_add_epi32(temp_vec, inter_vec);

                  // inter_vec2 = _mm256_maddubs_epi16(s2,w);
                  // inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  // temp_vec2 = _mm256_add_epi32(temp_vec2, inter_vec2);

                  // inter_vec = _mm256_maddubs_epi16(s3,w);
                  // inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  // temp_vec3 = _mm256_add_epi32(temp_vec3, inter_vec);

                  // inter_vec2 = _mm256_maddubs_epi16(s4, w);
                  // inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  // temp_vec4 = _mm256_add_epi32(temp_vec4, inter_vec2);

                  // inter_vec = _mm256_maddubs_epi16(s,w2);
                  // inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  // temp_vec5 = _mm256_add_epi32(temp_vec5, inter_vec);

                  // inter_vec2 = _mm256_maddubs_epi16(s2, w2);
                  // inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  // temp_vec6 = _mm256_add_epi32(temp_vec6, inter_vec2);

                  // inter_vec = _mm256_maddubs_epi16(s3,w2);
                  // inter_vec = _mm256_madd_epi16(inter_vec, _mm256_set1_epi16(1));
                  // temp_vec7 = _mm256_add_epi32(temp_vec7, inter_vec);

                  // inter_vec2 = _mm256_maddubs_epi16(s4, w2);
                  // inter_vec2 = _mm256_madd_epi16(inter_vec2, _mm256_set1_epi16(1));
                  // temp_vec8 = _mm256_add_epi32(temp_vec8, inter_vec2);
                }
              }
            }

            unsigned long long int out_subscript = b_out
              + y_out
              + (x << out_l_shift)
              + m;
            // unsigned long long int out_subscript2 = out_subscript + Output_depth_dim;
            // unsigned long long int out_subscript3 = out_subscript + (2 << out_l_shift);
            // unsigned long long int out_subscript4 = out_subscript + (3 << out_l_shift);




            // **************** optimised hadd ****************
            __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec), _mm256_extracti128_si256(temp_vec, 1));
            __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
            __m128i sum64 = _mm_add_epi32(hi64, sum128);
            __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            __m128i sum32 = _mm_add_epi32(sum64, hi32);
            temp = _mm_cvtsi128_si32(sum32);
            temp += bias;
            out_to_compare_with_Char[out_subscript] = RELU(temp);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec2), _mm256_extracti128_si256(temp_vec2, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp2 = _mm_cvtsi128_si32(sum32);
            temp2 += bias;
            out_to_compare_with_Char[out_subscript + (1 << out_l_shift)] = RELU(temp2);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec3), _mm256_extracti128_si256(temp_vec3, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp3 = _mm_cvtsi128_si32(sum32);
            temp3 += bias;
            out_to_compare_with_Char[out_subscript + (2 << out_l_shift)] = RELU(temp3);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec4), _mm256_extracti128_si256(temp_vec4, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp4 = _mm_cvtsi128_si32(sum32);
            temp4 += bias;
            out_to_compare_with_Char[out_subscript + (3 << out_l_shift)] = RELU(temp4);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec5), _mm256_extracti128_si256(temp_vec5, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp5 = _mm_cvtsi128_si32(sum32);
            temp5 += bias2;
            out_to_compare_with_Char[out_subscript+1] = RELU(temp5);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec6), _mm256_extracti128_si256(temp_vec6, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp6 = _mm_cvtsi128_si32(sum32);
            temp6 += bias2;
            out_to_compare_with_Char[out_subscript+1+ (1 << out_l_shift)] = RELU(temp6);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec7), _mm256_extracti128_si256(temp_vec7, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp7 = _mm_cvtsi128_si32(sum32);
            temp7 += bias2;
            out_to_compare_with_Char[out_subscript+1+ (2 << out_l_shift)] = RELU(temp7);


            sum128 = _mm_add_epi32(_mm256_castsi256_si128(temp_vec8), _mm256_extracti128_si256(temp_vec8, 1));
            hi64 = _mm_unpackhi_epi64(sum128, sum128);
            sum64 = _mm_add_epi32(hi64, sum128);
            hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2,3,0,1));
            sum32 = _mm_add_epi32(sum64, hi32);
            temp8 = _mm_cvtsi128_si32(sum32);
            temp8 += bias2;
            out_to_compare_with_Char[out_subscript+1+ (3 << out_l_shift)] = RELU(temp8);




            // temp_vec = _mm256_hadd_epi32(temp_vec, temp_vec);
            // temp_vec = _mm256_hadd_epi32(temp_vec, temp_vec);
            // __m128i tempLo = _mm256_castsi256_si128(temp_vec);
            // __m128i tempHi = _mm256_extracti128_si256(temp_vec, 1);
            // __m128 sum = _mm_add_epi32(tempLo, tempHi);
            // temp = _mm_cvtsi128_si32(sum);
            // temp += bias;
            // out_to_compare_with_Char[out_subscript] = RELU(temp);


            // temp_vec2 = _mm256_hadd_epi32(temp_vec2, temp_vec2);
            // temp_vec2 = _mm256_hadd_epi32(temp_vec2, temp_vec2);
            //  tempLo = _mm256_castsi256_si128(temp_vec2);
            //  tempHi = _mm256_extracti128_si256(temp_vec2, 1);
            //  sum = _mm_add_epi32(tempLo, tempHi);
            // temp2 = _mm_cvtsi128_si32(sum);
            // temp2 += bias;
            // out_to_compare_with_Char[out_subscript + (1 << out_l_shift)] = RELU(temp2);
            

            // temp_vec3 = _mm256_hadd_epi32(temp_vec3, temp_vec3);
            // temp_vec3 = _mm256_hadd_epi32(temp_vec3, temp_vec3);
            // tempLo = _mm256_castsi256_si128(temp_vec3);
            // tempHi = _mm256_extractf128_si256(temp_vec3, 1);
            // sum = _mm_add_epi32(tempLo, tempHi);
            // temp3 = _mm_cvtsi128_si32(sum);
            // temp3 += bias;
            // out_to_compare_with_Char[out_subscript + (2 << out_l_shift)] = RELU(temp3);


            // temp_vec4 = _mm256_hadd_epi32(temp_vec4, temp_vec4);
            // temp_vec4 = _mm256_hadd_epi32(temp_vec4, temp_vec4);
            // tempLo = _mm256_castsi256_si128(temp_vec4);
            // tempHi = _mm256_extracti128_si256(temp_vec4, 1);
            // sum = _mm_add_epi32(tempLo, tempHi);
            // temp4 = _mm_cvtsi128_si32(sum);
            // temp4 += bias;
            // out_to_compare_with_Char[out_subscript + (3 << out_l_shift)] = RELU(temp4);


            // temp_vec5 = _mm256_hadd_epi32(temp_vec5, temp_vec5);
            // temp_vec5 = _mm256_hadd_epi32(temp_vec5, temp_vec5);
            // tempLo = _mm256_castsi256_si128(temp_vec5);
            // tempHi = _mm256_extractf128_si256(temp_vec5, 1);
            // sum = _mm_add_epi32(tempLo, tempHi);
            // temp5 = _mm_cvtsi128_si32(sum);
            // temp5 += bias2;
            // out_to_compare_with_Char[out_subscript+1] = RELU(temp5);


            // temp_vec6 = _mm256_hadd_epi32(temp_vec6, temp_vec6);
            // temp_vec6 = _mm256_hadd_epi32(temp_vec6, temp_vec6);
            // tempLo = _mm256_castsi256_si128(temp_vec6);
            // tempHi = _mm256_extracti128_si256(temp_vec6, 1);
            // sum = _mm_add_epi32(tempLo, tempHi);
            // temp6 = _mm_cvtsi128_si32(sum);
            // temp6 += bias2;
            // out_to_compare_with_Char[out_subscript+1 + (1 << out_l_shift)] = RELU(temp6);


            // temp_vec7 = _mm256_hadd_epi32(temp_vec7, temp_vec7);
            // temp_vec7 = _mm256_hadd_epi32(temp_vec7, temp_vec7);
            // tempLo = _mm256_castsi256_si128(temp_vec7);
            // tempHi = _mm256_extractf128_si256(temp_vec7, 1);
            // sum = _mm_add_epi32(tempLo, tempHi);
            // temp7 = _mm_cvtsi128_si32(sum);
            // temp7 += bias2;
            // out_to_compare_with_Char[out_subscript+1 + (2 << out_l_shift)] = RELU(temp7);


            // temp_vec8 = _mm256_hadd_epi32(temp_vec8, temp_vec8);
            // temp_vec8 = _mm256_hadd_epi32(temp_vec8, temp_vec8);
            // tempLo = _mm256_castsi256_si128(temp_vec8);
            // tempHi = _mm256_extracti128_si256(temp_vec8, 1);
            // sum = _mm_add_epi32(tempLo, tempHi);
            // temp8 = _mm_cvtsi128_si32(sum);
            // temp8 += bias2;
            // out_to_compare_with_Char[out_subscript+1 + (3 << out_l_shift)] = RELU(temp8);

          }
        }
      }
    }
  }
  #undef left_shift
  #undef RELU
  printf("\n from quantised opt m2x4 %d %d ", out_to_compare_with_Char[0], out_to_compare_with_Char[1]);
  return 0;
}





// *** not working ***
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





// 


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