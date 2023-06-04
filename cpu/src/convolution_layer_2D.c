#include "convolution_layer_2D.h"
#include <xmmintrin.h>

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))


// function used for profilling, to get around potentially TLB-related issues
int profile_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  #define left_shift(x) (__builtin_ctz(x))  // inline function effectively using inline assembly to return num of trailing zeros (i.e. 128 returns 7, 2 * 128 == 2 << 7)

  const unsigned long long int out_l_shift = left_shift(Output_depth_dim);
  const unsigned long long int in_l_shift = left_shift(Input_depth_dim);

  const unsigned long long int mask_yx_depth = Mask_Y_dim * Mask_X_dim << in_l_shift;
  const unsigned long long int mask_x_depth = Mask_X_dim << in_l_shift;

  const unsigned long long int in_yx_depth = Input_Y_dim * Input_X_dim << in_l_shift;
  const unsigned long long int in_x_depth = Input_X_dim << in_l_shift;

  const unsigned long long int in_out_mask = Mask_X_dim << (out_l_shift + in_l_shift);
  const unsigned long long int in_out_depth = out_l_shift + in_l_shift;

  const unsigned long long int out_yx_depth = Output_Y_dim * Output_X_dim << out_l_shift;
  const unsigned long long int out_x_depth = Output_X_dim << out_l_shift;


  const unsigned int filter_FP_length = Output_depth_dim * Mask_Y_dim * Mask_X_dim * Input_depth_dim;
  float* filter_FP_copy = (float*)_mm_malloc(filter_FP_length * sizeof(float), 64);
  if (filter_FP_copy == NULL) {
    printf("\nerror with malloc allocating filter array copy");
    exit(EXIT_FAILURE);
  }

  // array copying - filter_FP into form usable for vectorising m loop
  unsigned long long int new_subscript = 0;
  for (unsigned int y = 0; y < Mask_Y_dim; y++) {
    const unsigned long long int y_old = y * mask_x_depth;
    // const unsigned long long int y_old2 = (y+1) * mask_x_depth;

    for (unsigned int x = 0; x < Mask_X_dim; x++) {
      const unsigned long long int x_old = x << in_l_shift;
      // const unsigned long long int x_old2 = (x+1) << left_shift(Input_depth_dim);

      for (unsigned int d = 0; d < Input_depth_dim; d+=2) {
        for (unsigned int m = 0; m < Output_depth_dim; m+=4) {

          unsigned long long int old_subscript = m * mask_yx_depth
            + y_old
            + x_old
            + d;

          filter_FP_copy[new_subscript] = filter_FP[old_subscript];
          filter_FP_copy[new_subscript+1] = filter_FP[old_subscript + mask_yx_depth];
          filter_FP_copy[new_subscript+2] = filter_FP[old_subscript + (mask_yx_depth << 1)];
          filter_FP_copy[new_subscript+3] = filter_FP[old_subscript + (mask_yx_depth << 1) + mask_yx_depth];
          filter_FP_copy[new_subscript+Output_depth_dim] = filter_FP[old_subscript + 1];
          filter_FP_copy[new_subscript+1+Output_depth_dim] = filter_FP[old_subscript + mask_yx_depth + 1];
          filter_FP_copy[new_subscript+2+Output_depth_dim] = filter_FP[old_subscript + (mask_yx_depth << 1) + 1];
          filter_FP_copy[new_subscript+3+Output_depth_dim] = filter_FP[old_subscript + (mask_yx_depth << 1) + mask_yx_depth + 1];

          new_subscript+=4;

          // filter_FP_copy[new_subscript] = filter_FP[old_subscript];
          // filter_FP_copy[new_subscript+Output_depth_dim] = filter_FP[old_subscript + 1];
          // filter_FP_copy[new_subscript+(2 << out_l_shift)] = filter_FP[old_subscript + 2];
          // filter_FP_copy[new_subscript+(3 << out_l_shift)] = filter_FP[old_subscript + 3];
          // filter_FP_copy[new_subscript+(4 << out_l_shift)] = filter_FP[old_subscript + 4];
          // filter_FP_copy[new_subscript+(5 << out_l_shift)] = filter_FP[old_subscript + 5];
          // filter_FP_copy[new_subscript+(6 << out_l_shift)] = filter_FP[old_subscript + 6];
          // filter_FP_copy[new_subscript+(7 << out_l_shift)] = filter_FP[old_subscript + 7];
          // filter_FP_copy[new_subscript+in_out_mask] = filter_FP[old_subscript3];
          // filter_FP_copy[new_subscript+in_out_mask + Input_depth_dim + Output_depth_dim] = filter_FP[old_subscript4];

          // new_subscript++;
        }
        new_subscript+=Output_depth_dim;
      }
    }
  }

  unsigned int x;
  const unsigned int x_bound = (Output_X_dim/ 4) * 4; 
  // printf("\nOutput_x_dim = %i\nx_bound = %i\n", Output_X_dim, x_bound);

    // main loop body
  #pragma omp parallel for private(x) shared(in_FP, filter_FP_copy, out_l_shift, in_l_shift, in_yx_depth, in_x_depth, in_out_mask, in_out_depth, \
    out_yx_depth, out_x_depth, x_bound) default(shared) collapse(2) schedule(static)
  {
    // main loop body
    for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
      const unsigned long long int b_depth = b * in_yx_depth;
      const unsigned long long int b_out = b * out_yx_depth;

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        const unsigned long long int y_stride = y * Stride_Y_dim;
        const unsigned long long int y_out = y * out_x_depth;

        for (unsigned int m = 0; m < Output_depth_dim; m+=16) { //channels
          const __m256 bias = _mm256_load_ps(&bias_array_FP[m]);
          const __m256 bias2 = _mm256_load_ps(&bias_array_FP[m+8]);

          for (x = 0; x < x_bound; x+=4) {	//Output Width
            const unsigned long long int x_stride = x * Stride_X_dim;

            __m256 temp = _mm256_setzero_ps();
            __m256 temp2 = _mm256_setzero_ps();
            __m256 temp3 = _mm256_setzero_ps();
            __m256 temp4 = _mm256_setzero_ps();
            __m256 temp5 = _mm256_setzero_ps();
            __m256 temp6 = _mm256_setzero_ps();
            __m256 temp7 = _mm256_setzero_ps();
            __m256 temp8 = _mm256_setzero_ps();

            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              const unsigned long long int off_y_stride = (y_stride + off_y) * in_x_depth;
              const unsigned long long int off_y_mask = off_y * in_out_mask;

              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                const unsigned long long int off_x_stride = (x_stride+ off_x) << in_l_shift;
                const unsigned long long int off_x_mask = off_x << in_out_depth;
                
                for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                  unsigned long long int in_subscript = b_depth
                    + off_y_stride
                    + off_x_stride
                    + d;

                  unsigned long long int filter_subscript = off_y_mask
                    + off_x_mask
                    + (d << out_l_shift)
                    + m;
                  // unsigned long long int filter_subscript2 = filter_subscript + 8;
                  // unsigned long long int filter_subscript3 = filter_subscript + 16;
                  // unsigned long long int filter_subscript4 = filter_subscript + 24;

                  
                  __m256 s = _mm256_broadcast_ss(&in_FP[in_subscript]);
                  __m256 s2 = _mm256_broadcast_ss(&in_FP[in_subscript + Input_depth_dim]);
                  __m256 s3 = _mm256_broadcast_ss(&in_FP[in_subscript + (2 << in_l_shift)]);
                  __m256 s4 = _mm256_broadcast_ss(&in_FP[in_subscript + (3 << in_l_shift)]);

                  __m256 w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                  __m256 w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                  // d+1 ------------------------------------------------------------
                  // |
                  // |
                  // v


                  in_subscript += 1;

                  filter_subscript += Output_depth_dim;
                  

                  s = _mm256_broadcast_ss(&in_FP[in_subscript]);
                  s2 = _mm256_broadcast_ss(&in_FP[in_subscript + Input_depth_dim]);
                  s3 = _mm256_broadcast_ss(&in_FP[in_subscript + (2 << in_l_shift)]);
                  s4 = _mm256_broadcast_ss(&in_FP[in_subscript + (3 << in_l_shift)]);

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                  // d+2 ------------------------------------------------------------
                  // |
                  // |
                  // v


                  in_subscript += 1;

                  filter_subscript += Output_depth_dim;
                  

                  s = _mm256_broadcast_ss(&in_FP[in_subscript]);
                  s2 = _mm256_broadcast_ss(&in_FP[in_subscript + Input_depth_dim]);
                  s3 = _mm256_broadcast_ss(&in_FP[in_subscript + (2 << in_l_shift)]);
                  s4 = _mm256_broadcast_ss(&in_FP[in_subscript + (3 << in_l_shift)]);

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);

                  // d+3 ------------------------------------------------------------
                  // |
                  // |
                  // v


                  in_subscript += 1;

                  filter_subscript += Output_depth_dim;
                  

                  s = _mm256_broadcast_ss(&in_FP[in_subscript]);
                  s2 = _mm256_broadcast_ss(&in_FP[in_subscript + Input_depth_dim]);
                  s3 = _mm256_broadcast_ss(&in_FP[in_subscript + (2 << in_l_shift)]);
                  s4 = _mm256_broadcast_ss(&in_FP[in_subscript + (3 << in_l_shift)]);

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                  // d+4 ------------------------------------------------------------
                  // |
                  // |
                  // v


                  in_subscript += 1;

                  filter_subscript += Output_depth_dim;
                  

                  s = _mm256_broadcast_ss(&in_FP[in_subscript]);
                  s2 = _mm256_broadcast_ss(&in_FP[in_subscript + Input_depth_dim]);
                  s3 = _mm256_broadcast_ss(&in_FP[in_subscript + (2 << in_l_shift)]);
                  s4 = _mm256_broadcast_ss(&in_FP[in_subscript + (3 << in_l_shift)]);

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                  // d+5 ------------------------------------------------------------
                  // |
                  // |
                  // v


                  in_subscript += 1;

                  filter_subscript += Output_depth_dim;
                  

                  s = _mm256_broadcast_ss(&in_FP[in_subscript]);
                  s2 = _mm256_broadcast_ss(&in_FP[in_subscript + Input_depth_dim]);
                  s3 = _mm256_broadcast_ss(&in_FP[in_subscript + (2 << in_l_shift)]);
                  s4 = _mm256_broadcast_ss(&in_FP[in_subscript + (3 << in_l_shift)]);

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                  // d+6 ------------------------------------------------------------
                  // |
                  // |
                  // v


                  in_subscript += 1;

                  filter_subscript += Output_depth_dim;
                  

                  s = _mm256_broadcast_ss(&in_FP[in_subscript]);
                  s2 = _mm256_broadcast_ss(&in_FP[in_subscript + Input_depth_dim]);
                  s3 = _mm256_broadcast_ss(&in_FP[in_subscript + (2 << in_l_shift)]);
                  s4 = _mm256_broadcast_ss(&in_FP[in_subscript + (3 << in_l_shift)]);

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                  // d+7 ------------------------------------------------------------
                  // |
                  // |
                  // v


                  in_subscript += 1;

                  filter_subscript += Output_depth_dim;
                  

                  s = _mm256_broadcast_ss(&in_FP[in_subscript]);
                  s2 = _mm256_broadcast_ss(&in_FP[in_subscript + Input_depth_dim]);
                  s3 = _mm256_broadcast_ss(&in_FP[in_subscript + (2 << in_l_shift)]);
                  s4 = _mm256_broadcast_ss(&in_FP[in_subscript + (3 << in_l_shift)]);

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


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

            unsigned long long int out_subscript = b_out +
              y_out +
              (x << out_l_shift)
              + m;
            const unsigned long long int out_subscript2 = out_subscript + Output_depth_dim;
            const unsigned long long int out_subscript3 = out_subscript + (2 << out_l_shift);
            const unsigned long long int out_subscript4 = out_subscript + (3 << out_l_shift);
            const unsigned long long int out_subscript5 = out_subscript + 8;
            const unsigned long long int out_subscript6 = out_subscript + Output_depth_dim + 8;
            const unsigned long long int out_subscript7 = out_subscript + (2 << out_l_shift) + 8;
            const unsigned long long int out_subscript8 = out_subscript + (3 << out_l_shift) + 8;

            
            temp = _mm256_add_ps(temp, bias);
            temp = _mm256_max_ps(temp, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript], temp);

            temp2 = _mm256_add_ps(temp2, bias);
            temp2 = _mm256_max_ps(temp2, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript2], temp2); // x+1

            temp3 = _mm256_add_ps(temp3, bias);
            temp3 = _mm256_max_ps(temp3, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript3], temp3); // x+2

            temp4 = _mm256_add_ps(temp4, bias);
            temp4 = _mm256_max_ps(temp4, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript4], temp4); // x+3

            temp5 = _mm256_add_ps(temp5, bias2);
            temp5 = _mm256_max_ps(temp5, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + 8], temp5); // m+8

            temp6 = _mm256_add_ps(temp6, bias2);
            temp6 = _mm256_max_ps(temp6, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript2 + 8], temp6); // m+8, x+1

            temp7 = _mm256_add_ps(temp7, bias2);
            temp7 = _mm256_max_ps(temp7, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript3 + 8], temp7); // m+8, x+2

            temp8 = _mm256_add_ps(temp8, bias2);
            temp8 = _mm256_max_ps(temp8, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript4 + 8], temp8); // m+8, x+3
          }
          for (; x < Output_X_dim; x++) {
            const unsigned long long int x_stride = x * Stride_X_dim;

            __m256 temp = _mm256_setzero_ps();
            __m256 temp2 = _mm256_setzero_ps();

            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              const unsigned long long int off_y_stride = (y_stride + off_y) * in_x_depth;
              const unsigned long long int off_y_mask = off_y * in_out_mask;

              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                const unsigned long long int off_x_stride = (x_stride+ off_x) << in_l_shift;
                const unsigned long long int off_x_mask = off_x << in_out_depth;
                
                for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                  unsigned long long int in_subscript = b_depth
                    + off_y_stride
                    + off_x_stride
                    + d;

                  unsigned long long int filter_subscript = off_y_mask
                    + off_x_mask
                    + (d << out_l_shift)
                    + m;
                  // unsigned long long int filter_subscript2 = filter_subscript + 8;
                  // unsigned long long int filter_subscript3 = filter_subscript + 16;
                  // unsigned long long int filter_subscript4 = filter_subscript + 24;

                  
                  __m256 s = _mm256_broadcast_ss(&in_FP[in_subscript]);

                  __m256 w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                  __m256 w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s, w2, temp2);



                  // d+1 ------------------------------------------------------------
                  // |
                  // |
                  // v


                  // in_subscript += 1;

                  filter_subscript += Output_depth_dim;
                  

                  s = _mm256_broadcast_ss(&in_FP[in_subscript+1]);

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s, w2, temp2);


                  // d+2 ------------------------------------------------------------
                  // |
                  // |
                  // v


                  // in_subscript += 1;

                  filter_subscript += Output_depth_dim;
                  

                  s = _mm256_broadcast_ss(&in_FP[in_subscript+2]);

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s, w2, temp2);

                  // d+3 ------------------------------------------------------------
                  // |
                  // |
                  // v


                  // in_subscript += 1;

                  filter_subscript += Output_depth_dim;
                  

                  s = _mm256_broadcast_ss(&in_FP[in_subscript+3]);

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s, w2, temp2);


                  // d+4 ------------------------------------------------------------
                  // |
                  // |
                  // v


                  // in_subscript += 1;

                  filter_subscript += Output_depth_dim;
                  

                  s = _mm256_broadcast_ss(&in_FP[in_subscript+4]);

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s, w2, temp2);


                  // d+5 ------------------------------------------------------------
                  // |
                  // |
                  // v


                  // in_subscript += 1;

                  filter_subscript += Output_depth_dim;
                  

                  s = _mm256_broadcast_ss(&in_FP[in_subscript+5]);

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s, w2, temp2);


                  // d+6 ------------------------------------------------------------
                  // |
                  // |
                  // v


                  // in_subscript += 1;

                  filter_subscript += Output_depth_dim;
                  

                  s = _mm256_broadcast_ss(&in_FP[in_subscript+6]);

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s, w2, temp2);


                  // d+7 ------------------------------------------------------------
                  // |
                  // |
                  // v


                  // in_subscript += 1;

                  filter_subscript += Output_depth_dim;
                  

                  s = _mm256_broadcast_ss(&in_FP[in_subscript+7]);

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s, w2, temp2);
                }
              }
            }

            const unsigned long long int out_subscript = b_out +
              y_out +
              (x << out_l_shift)
              + m;
            const unsigned long long int out_subscript2 = out_subscript + 8;

            
            temp = _mm256_add_ps(temp, bias);
            temp = _mm256_max_ps(temp, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript], temp);

            temp2 = _mm256_add_ps(temp2, bias2);
            temp2 = _mm256_max_ps(temp2, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript2], temp2); 
          }
        }
      }
    }
  }
  #undef left_shift
  printf("\n from AC optv15 m16 x4 d8 - edge cases interchanged m and x %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  _mm_free(filter_FP_copy);
  return 0;
}




// finalised array copying loops

// 
// ~563 GFLOPS
int optimised_layer_v16_AC_x4m16d8_edge_cases_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {

  #define left_shift(x) (__builtin_ctz(x))  // inline function effectively using inline assembly to return num of trailing zeros (i.e. 128 returns 7, 2 * 128 == 2 << 7)

  const unsigned long long int out_l_shift = left_shift(Output_depth_dim);
  const unsigned long long int in_l_shift = left_shift(Input_depth_dim);

  const unsigned long long int mask_yx_depth = Mask_Y_dim * Mask_X_dim << in_l_shift;
  const unsigned long long int mask_x_depth = Mask_X_dim << in_l_shift;

  const unsigned long long int in_yx_depth = Input_Y_dim * Input_X_dim << in_l_shift;
  const unsigned long long int in_x_depth = Input_X_dim << in_l_shift;

  const unsigned long long int in_out_mask = Mask_X_dim << (out_l_shift + in_l_shift);
  const unsigned long long int in_out_depth = out_l_shift + in_l_shift;

  const unsigned long long int out_yx_depth = Output_Y_dim * Output_X_dim << out_l_shift;
  const unsigned long long int out_x_depth = Output_X_dim << out_l_shift;


  const unsigned int filter_FP_length = Output_depth_dim * Mask_Y_dim * Mask_X_dim * Input_depth_dim;
  float* filter_FP_copy = (float*)_mm_malloc(filter_FP_length * sizeof(float), 64);
  if (filter_FP_copy == NULL) {
    printf("\nerror with malloc allocating filter array copy");
    exit(EXIT_FAILURE);
  }

  // array copying - filter_FP into form usable for vectorising m loop
  unsigned long long int new_subscript = 0;
  for (unsigned int y = 0; y < Mask_Y_dim; y++) {
    const unsigned long long int y_old = y * mask_x_depth;
    // const unsigned long long int y_old2 = (y+1) * mask_x_depth;

    for (unsigned int x = 0; x < Mask_X_dim; x++) {
      const unsigned long long int x_old = x << in_l_shift;
      // const unsigned long long int x_old2 = (x+1) << left_shift(Input_depth_dim);

      for (unsigned int d = 0; d < Input_depth_dim; d+=2) {
        for (unsigned int m = 0; m < Output_depth_dim; m+=4) {

          unsigned long long int old_subscript = m * mask_yx_depth
            + y_old
            + x_old
            + d;

          filter_FP_copy[new_subscript] = filter_FP[old_subscript];
          filter_FP_copy[new_subscript+1] = filter_FP[old_subscript + mask_yx_depth];
          filter_FP_copy[new_subscript+2] = filter_FP[old_subscript + (mask_yx_depth << 1)];
          filter_FP_copy[new_subscript+3] = filter_FP[old_subscript + (mask_yx_depth << 1) + mask_yx_depth];
          filter_FP_copy[new_subscript+Output_depth_dim] = filter_FP[old_subscript + 1];
          filter_FP_copy[new_subscript+1+Output_depth_dim] = filter_FP[old_subscript + mask_yx_depth + 1];
          filter_FP_copy[new_subscript+2+Output_depth_dim] = filter_FP[old_subscript + (mask_yx_depth << 1) + 1];
          filter_FP_copy[new_subscript+3+Output_depth_dim] = filter_FP[old_subscript + (mask_yx_depth << 1) + mask_yx_depth + 1];

          new_subscript+=4;

          // filter_FP_copy[new_subscript] = filter_FP[old_subscript];
          // filter_FP_copy[new_subscript+Output_depth_dim] = filter_FP[old_subscript + 1];
          // filter_FP_copy[new_subscript+(2 << out_l_shift)] = filter_FP[old_subscript + 2];
          // filter_FP_copy[new_subscript+(3 << out_l_shift)] = filter_FP[old_subscript + 3];
          // filter_FP_copy[new_subscript+(4 << out_l_shift)] = filter_FP[old_subscript + 4];
          // filter_FP_copy[new_subscript+(5 << out_l_shift)] = filter_FP[old_subscript + 5];
          // filter_FP_copy[new_subscript+(6 << out_l_shift)] = filter_FP[old_subscript + 6];
          // filter_FP_copy[new_subscript+(7 << out_l_shift)] = filter_FP[old_subscript + 7];
          // filter_FP_copy[new_subscript+in_out_mask] = filter_FP[old_subscript3];
          // filter_FP_copy[new_subscript+in_out_mask + Input_depth_dim + Output_depth_dim] = filter_FP[old_subscript4];

          // new_subscript++;
        }
        new_subscript+=Output_depth_dim;
      }
    }
  }

  unsigned int x;
  const unsigned int x_bound = (Output_X_dim/ 4) * 4; 

    // main loop body
  #pragma omp parallel for private(x) shared(in_FP, filter_FP_copy, out_l_shift, in_l_shift, in_yx_depth, in_x_depth, in_out_mask, in_out_depth, \
    out_yx_depth, out_x_depth, x_bound) default(shared) collapse(2) schedule(static)
  {
    // main loop body
    for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
      const unsigned long long int b_depth = b * in_yx_depth;
      const unsigned long long int b_out = b * out_yx_depth;

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        const unsigned long long int y_stride = y * Stride_Y_dim;
        const unsigned long long int y_out = y * out_x_depth;

        for (x = 0; x < x_bound; x+=4) {	//Output Width
          const unsigned long long int x_stride = x * Stride_X_dim;
          const unsigned long long int x_out = x << out_l_shift;

          for (unsigned int m = 0; m < Output_depth_dim; m+=16) { //channels
            const __m256 bias = _mm256_load_ps(&bias_array_FP[m]);
            const __m256 bias2 = _mm256_load_ps(&bias_array_FP[m+8]);

            __m256 temp = _mm256_setzero_ps();
            __m256 temp2 = _mm256_setzero_ps();
            __m256 temp3 = _mm256_setzero_ps();
            __m256 temp4 = _mm256_setzero_ps();
            __m256 temp5 = _mm256_setzero_ps();
            __m256 temp6 = _mm256_setzero_ps();
            __m256 temp7 = _mm256_setzero_ps();
            __m256 temp8 = _mm256_setzero_ps();

            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              const unsigned long long int off_y_stride = (y_stride + off_y) * in_x_depth;
              const unsigned long long int off_y_mask = off_y * in_out_mask;

              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                const unsigned long long int off_x_stride = (x_stride+ off_x) << in_l_shift;
                const unsigned long long int off_x_mask = off_x << in_out_depth;
                
                for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                  unsigned long long int in_subscript = b_depth
                    + off_y_stride
                    + off_x_stride
                    + d;

                  unsigned long long int filter_subscript = off_y_mask
                    + off_x_mask
                    + (d << out_l_shift)
                    + m;
                  // unsigned long long int filter_subscript2 = filter_subscript + 8;
                  // unsigned long long int filter_subscript3 = filter_subscript + 16;
                  // unsigned long long int filter_subscript4 = filter_subscript + 24;

                  
                  __m256 s = _mm256_broadcast_ss(&in_FP[in_subscript]);
                  __m256 s2 = _mm256_broadcast_ss(&in_FP[in_subscript + Input_depth_dim]);
                  __m256 s3 = _mm256_broadcast_ss(&in_FP[in_subscript + (2 << in_l_shift)]);
                  __m256 s4 = _mm256_broadcast_ss(&in_FP[in_subscript + (3 << in_l_shift)]);

                  __m256 w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                  __m256 w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                  // d+1 ------------------------------------------------------------
                  // |
                  // |
                  // v


                  in_subscript += 1;

                  filter_subscript += Output_depth_dim;
                  

                  s = _mm256_broadcast_ss(&in_FP[in_subscript]);
                  s2 = _mm256_broadcast_ss(&in_FP[in_subscript + Input_depth_dim]);
                  s3 = _mm256_broadcast_ss(&in_FP[in_subscript + (2 << in_l_shift)]);
                  s4 = _mm256_broadcast_ss(&in_FP[in_subscript + (3 << in_l_shift)]);

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                  // d+2 ------------------------------------------------------------
                  // |
                  // |
                  // v


                  in_subscript += 1;

                  filter_subscript += Output_depth_dim;
                  

                  s = _mm256_broadcast_ss(&in_FP[in_subscript]);
                  s2 = _mm256_broadcast_ss(&in_FP[in_subscript + Input_depth_dim]);
                  s3 = _mm256_broadcast_ss(&in_FP[in_subscript + (2 << in_l_shift)]);
                  s4 = _mm256_broadcast_ss(&in_FP[in_subscript + (3 << in_l_shift)]);

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);

                  // d+3 ------------------------------------------------------------
                  // |
                  // |
                  // v


                  in_subscript += 1;

                  filter_subscript += Output_depth_dim;
                  

                  s = _mm256_broadcast_ss(&in_FP[in_subscript]);
                  s2 = _mm256_broadcast_ss(&in_FP[in_subscript + Input_depth_dim]);
                  s3 = _mm256_broadcast_ss(&in_FP[in_subscript + (2 << in_l_shift)]);
                  s4 = _mm256_broadcast_ss(&in_FP[in_subscript + (3 << in_l_shift)]);

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                  // d+4 ------------------------------------------------------------
                  // |
                  // |
                  // v


                  in_subscript += 1;

                  filter_subscript += Output_depth_dim;
                  

                  s = _mm256_broadcast_ss(&in_FP[in_subscript]);
                  s2 = _mm256_broadcast_ss(&in_FP[in_subscript + Input_depth_dim]);
                  s3 = _mm256_broadcast_ss(&in_FP[in_subscript + (2 << in_l_shift)]);
                  s4 = _mm256_broadcast_ss(&in_FP[in_subscript + (3 << in_l_shift)]);

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                  // d+5 ------------------------------------------------------------
                  // |
                  // |
                  // v


                  in_subscript += 1;

                  filter_subscript += Output_depth_dim;
                  

                  s = _mm256_broadcast_ss(&in_FP[in_subscript]);
                  s2 = _mm256_broadcast_ss(&in_FP[in_subscript + Input_depth_dim]);
                  s3 = _mm256_broadcast_ss(&in_FP[in_subscript + (2 << in_l_shift)]);
                  s4 = _mm256_broadcast_ss(&in_FP[in_subscript + (3 << in_l_shift)]);

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                  // d+6 ------------------------------------------------------------
                  // |
                  // |
                  // v


                  in_subscript += 1;

                  filter_subscript += Output_depth_dim;
                  

                  s = _mm256_broadcast_ss(&in_FP[in_subscript]);
                  s2 = _mm256_broadcast_ss(&in_FP[in_subscript + Input_depth_dim]);
                  s3 = _mm256_broadcast_ss(&in_FP[in_subscript + (2 << in_l_shift)]);
                  s4 = _mm256_broadcast_ss(&in_FP[in_subscript + (3 << in_l_shift)]);

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                  // d+7 ------------------------------------------------------------
                  // |
                  // |
                  // v


                  in_subscript += 1;

                  filter_subscript += Output_depth_dim;
                  

                  s = _mm256_broadcast_ss(&in_FP[in_subscript]);
                  s2 = _mm256_broadcast_ss(&in_FP[in_subscript + Input_depth_dim]);
                  s3 = _mm256_broadcast_ss(&in_FP[in_subscript + (2 << in_l_shift)]);
                  s4 = _mm256_broadcast_ss(&in_FP[in_subscript + (3 << in_l_shift)]);

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


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

            unsigned long long int out_subscript = b_out +
              y_out +
              x_out
              + m;
            const unsigned long long int out_subscript2 = out_subscript + Output_depth_dim;
            const unsigned long long int out_subscript3 = out_subscript + (2 << out_l_shift);
            const unsigned long long int out_subscript4 = out_subscript + (3 << out_l_shift);
            const unsigned long long int out_subscript5 = out_subscript + 8;
            const unsigned long long int out_subscript6 = out_subscript + Output_depth_dim + 8;
            const unsigned long long int out_subscript7 = out_subscript + (2 << out_l_shift) + 8;
            const unsigned long long int out_subscript8 = out_subscript + (3 << out_l_shift) + 8;

            
            temp = _mm256_add_ps(temp, bias);
            temp = _mm256_max_ps(temp, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript], temp);

            temp2 = _mm256_add_ps(temp2, bias);
            temp2 = _mm256_max_ps(temp2, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript2], temp2); // x+1

            temp3 = _mm256_add_ps(temp3, bias);
            temp3 = _mm256_max_ps(temp3, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript3], temp3); // x+2

            temp4 = _mm256_add_ps(temp4, bias);
            temp4 = _mm256_max_ps(temp4, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript4], temp4); // x+3

            temp5 = _mm256_add_ps(temp5, bias2);
            temp5 = _mm256_max_ps(temp5, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + 8], temp5); // m+8

            temp6 = _mm256_add_ps(temp6, bias2);
            temp6 = _mm256_max_ps(temp6, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript2 + 8], temp6); // m+8, x+1

            temp7 = _mm256_add_ps(temp7, bias2);
            temp7 = _mm256_max_ps(temp7, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript3 + 8], temp7); // m+8, x+2

            temp8 = _mm256_add_ps(temp8, bias2);
            temp8 = _mm256_max_ps(temp8, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript4 + 8], temp8); // m+8, x+3
          }
        }
        for (; x < Output_X_dim; x++) {
          const unsigned long long int x_stride = x * Stride_X_dim;
          const unsigned long long int x_out = x << out_l_shift;

          for (unsigned int m = 0; m < Output_depth_dim; m+=16) { //channels
            const __m256 bias = _mm256_load_ps(&bias_array_FP[m]);
            const __m256 bias2 = _mm256_load_ps(&bias_array_FP[m+8]);

            __m256 temp = _mm256_setzero_ps();
            __m256 temp2 = _mm256_setzero_ps();

            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              const unsigned long long int off_y_stride = (y_stride + off_y) * in_x_depth;
              const unsigned long long int off_y_mask = off_y * in_out_mask;

              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                const unsigned long long int off_x_stride = (x_stride+ off_x) << in_l_shift;
                const unsigned long long int off_x_mask = off_x << in_out_depth;
                
                for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                  unsigned long long int in_subscript = b_depth
                    + off_y_stride
                    + off_x_stride
                    + d;

                  unsigned long long int filter_subscript = off_y_mask
                    + off_x_mask
                    + (d << out_l_shift)
                    + m;
                  // unsigned long long int filter_subscript2 = filter_subscript + 8;
                  // unsigned long long int filter_subscript3 = filter_subscript + 16;
                  // unsigned long long int filter_subscript4 = filter_subscript + 24;

                  
                  __m256 s = _mm256_broadcast_ss(&in_FP[in_subscript]);

                  __m256 w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                  __m256 w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);



                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s, w2, temp2);



                  // d+1 ------------------------------------------------------------
                  // |
                  // |
                  // v


                  // in_subscript += 1;

                  filter_subscript += Output_depth_dim;
                  

                  s = _mm256_broadcast_ss(&in_FP[in_subscript+1]);

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s, w2, temp2);


                  // d+2 ------------------------------------------------------------
                  // |
                  // |
                  // v


                  // in_subscript += 1;

                  filter_subscript += Output_depth_dim;
                  

                  s = _mm256_broadcast_ss(&in_FP[in_subscript+2]);

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);



                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s, w2, temp2);

                  // d+3 ------------------------------------------------------------
                  // |
                  // |
                  // v


                  // in_subscript += 1;

                  filter_subscript += Output_depth_dim;
                  

                  s = _mm256_broadcast_ss(&in_FP[in_subscript+3]);

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s, w2, temp2);


                  // d+4 ------------------------------------------------------------
                  // |
                  // |
                  // v


                  // in_subscript += 1;

                  filter_subscript += Output_depth_dim;
                  

                  s = _mm256_broadcast_ss(&in_FP[in_subscript+4]);

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s, w2, temp2);


                  // d+5 ------------------------------------------------------------
                  // |
                  // |
                  // v


                  // in_subscript += 1;

                  filter_subscript += Output_depth_dim;
                  

                  s = _mm256_broadcast_ss(&in_FP[in_subscript+5]);

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s, w2, temp2);


                  // d+6 ------------------------------------------------------------
                  // |
                  // |
                  // v


                  // in_subscript += 1;

                  filter_subscript += Output_depth_dim;
                  

                  s = _mm256_broadcast_ss(&in_FP[in_subscript+6]);

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s, w2, temp2);


                  // d+7 ------------------------------------------------------------
                  // |
                  // |
                  // v


                  // in_subscript += 1;

                  filter_subscript += Output_depth_dim;
                  

                  s = _mm256_broadcast_ss(&in_FP[in_subscript+7]);

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s, w2, temp2);
                }
              }
            }

            unsigned long long int out_subscript = b_out +
              y_out +
              x_out
              + m;
            const unsigned long long int out_subscript2 = out_subscript + 8;
            const unsigned long long int out_subscript3 = out_subscript + 16;
            const unsigned long long int out_subscript4 = out_subscript + 24;

            
            temp = _mm256_add_ps(temp, bias);
            temp = _mm256_max_ps(temp, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript], temp);

            temp2 = _mm256_add_ps(temp2, bias2);
            temp2 = _mm256_max_ps(temp2, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript2], temp2); // x+1
          }
        }
      }
    }
  }
  #undef left_shift
  printf("\n from AC optv16 m16 x4 d8 - edge cases %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  _mm_free(filter_FP_copy);
  return 0;
}


// tiled version of above
// ~495 GFLOPS @ Output_depth_dim = 128, same perf?
int optimised_layer_v16_AC_x4m16d8_tiled_edge_cases_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {

  #define m_tile 16
  #define left_shift(x) (__builtin_ctz(x))  // inline function effectively using inline assembly to return num of trailing zeros (i.e. 128 returns 7, 2 * 128 == 2 << 7)

  const unsigned long long int out_l_shift = left_shift(Output_depth_dim);
  const unsigned long long int in_l_shift = left_shift(Input_depth_dim);

  const unsigned long long int mask_yx_depth = Mask_Y_dim * Mask_X_dim << in_l_shift;
  const unsigned long long int mask_x_depth = Mask_X_dim << in_l_shift;

  const unsigned long long int in_yx_depth = Input_Y_dim * Input_X_dim << in_l_shift;
  const unsigned long long int in_x_depth = Input_X_dim << in_l_shift;

  const unsigned long long int in_out_mask = Mask_X_dim << (out_l_shift + in_l_shift);
  const unsigned long long int in_out_depth = out_l_shift + in_l_shift;

  const unsigned long long int out_yx_depth = Output_Y_dim * Output_X_dim << out_l_shift;
  const unsigned long long int out_x_depth = Output_X_dim << out_l_shift;


  const unsigned int filter_FP_length = Output_depth_dim * Mask_Y_dim * Mask_X_dim * Input_depth_dim;
  float* filter_FP_copy = (float*)_mm_malloc(filter_FP_length * sizeof(float), 64);
  if (filter_FP_copy == NULL) {
    printf("\nerror with malloc allocating filter array copy");
    exit(EXIT_FAILURE);
  }

  // array copying - filter_FP into form usable for vectorising m loop
  unsigned long long int new_subscript = 0;
  for (unsigned int y = 0; y < Mask_Y_dim; y++) {
    const unsigned long long int y_old = y * mask_x_depth;
    // const unsigned long long int y_old2 = (y+1) * mask_x_depth;

    for (unsigned int x = 0; x < Mask_X_dim; x++) {
      const unsigned long long int x_old = x << in_l_shift;
      // const unsigned long long int x_old2 = (x+1) << left_shift(Input_depth_dim);

      for (unsigned int d = 0; d < Input_depth_dim; d+=2) {
        for (unsigned int m = 0; m < Output_depth_dim; m+=4) {

          unsigned long long int old_subscript = m * mask_yx_depth
            + y_old
            + x_old
            + d;

          filter_FP_copy[new_subscript] = filter_FP[old_subscript];
          filter_FP_copy[new_subscript+1] = filter_FP[old_subscript + mask_yx_depth];
          filter_FP_copy[new_subscript+2] = filter_FP[old_subscript + (mask_yx_depth << 1)];
          filter_FP_copy[new_subscript+3] = filter_FP[old_subscript + (mask_yx_depth << 1) + mask_yx_depth];
          filter_FP_copy[new_subscript+Output_depth_dim] = filter_FP[old_subscript + 1];
          filter_FP_copy[new_subscript+1+Output_depth_dim] = filter_FP[old_subscript + mask_yx_depth + 1];
          filter_FP_copy[new_subscript+2+Output_depth_dim] = filter_FP[old_subscript + (mask_yx_depth << 1) + 1];
          filter_FP_copy[new_subscript+3+Output_depth_dim] = filter_FP[old_subscript + (mask_yx_depth << 1) + mask_yx_depth + 1];

          new_subscript+=4;

          // filter_FP_copy[new_subscript] = filter_FP[old_subscript];
          // filter_FP_copy[new_subscript+Output_depth_dim] = filter_FP[old_subscript + 1];
          // filter_FP_copy[new_subscript+(2 << out_l_shift)] = filter_FP[old_subscript + 2];
          // filter_FP_copy[new_subscript+(3 << out_l_shift)] = filter_FP[old_subscript + 3];
          // filter_FP_copy[new_subscript+(4 << out_l_shift)] = filter_FP[old_subscript + 4];
          // filter_FP_copy[new_subscript+(5 << out_l_shift)] = filter_FP[old_subscript + 5];
          // filter_FP_copy[new_subscript+(6 << out_l_shift)] = filter_FP[old_subscript + 6];
          // filter_FP_copy[new_subscript+(7 << out_l_shift)] = filter_FP[old_subscript + 7];
          // filter_FP_copy[new_subscript+in_out_mask] = filter_FP[old_subscript3];
          // filter_FP_copy[new_subscript+in_out_mask + Input_depth_dim + Output_depth_dim] = filter_FP[old_subscript4];

          // new_subscript++;
        }
        new_subscript+=Output_depth_dim;
      }
    }
  }

  unsigned int x;
  const unsigned int x_bound = (Output_X_dim/ 4) * 4; 

    // main loop body
  #pragma omp parallel for private(x) shared(in_FP, filter_FP_copy, out_l_shift, in_l_shift, in_yx_depth, in_x_depth, in_out_mask, in_out_depth, \
    out_yx_depth, out_x_depth, x_bound) default(shared) collapse(3) schedule(static)
  {
    // main loop body
    for (unsigned int mm = 0; mm < Output_depth_dim; mm+=m_tile) {
      for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
        const unsigned long long int b_depth = b * in_yx_depth;
        const unsigned long long int b_out = b * out_yx_depth;

        for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
          const unsigned long long int y_stride = y * Stride_Y_dim;
          const unsigned long long int y_out = y * out_x_depth;

          for (x = 0; x < x_bound; x+=4) {	//Output Width
            const unsigned long long int x_stride = x * Stride_X_dim;
            const unsigned long long int x_out = x << out_l_shift;

            for (unsigned int m = mm; m < mm+m_tile; m+=16) { //channels
              const __m256 bias = _mm256_load_ps(&bias_array_FP[m]);
              const __m256 bias2 = _mm256_load_ps(&bias_array_FP[m+8]);

              __m256 temp = _mm256_setzero_ps();
              __m256 temp2 = _mm256_setzero_ps();
              __m256 temp3 = _mm256_setzero_ps();
              __m256 temp4 = _mm256_setzero_ps();
              __m256 temp5 = _mm256_setzero_ps();
              __m256 temp6 = _mm256_setzero_ps();
              __m256 temp7 = _mm256_setzero_ps();
              __m256 temp8 = _mm256_setzero_ps();

              for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
                const unsigned long long int off_y_stride = (y_stride + off_y) * in_x_depth;
                const unsigned long long int off_y_mask = off_y * in_out_mask;

                for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                  const unsigned long long int off_x_stride = (x_stride+ off_x) << in_l_shift;
                  const unsigned long long int off_x_mask = off_x << in_out_depth;
                  
                  for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                    unsigned long long int in_subscript = b_depth
                      + off_y_stride
                      + off_x_stride
                      + d;

                    unsigned long long int filter_subscript = off_y_mask
                      + off_x_mask
                      + (d << out_l_shift)
                      + m;
                    // unsigned long long int filter_subscript2 = filter_subscript + 8;
                    // unsigned long long int filter_subscript3 = filter_subscript + 16;
                    // unsigned long long int filter_subscript4 = filter_subscript + 24;

                    
                    __m256 s = _mm256_broadcast_ss(&in_FP[in_subscript]);
                    __m256 s2 = _mm256_broadcast_ss(&in_FP[in_subscript + Input_depth_dim]);
                    __m256 s3 = _mm256_broadcast_ss(&in_FP[in_subscript + (2 << in_l_shift)]);
                    __m256 s4 = _mm256_broadcast_ss(&in_FP[in_subscript + (3 << in_l_shift)]);

                    __m256 w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                    __m256 w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                    temp = _mm256_fmadd_ps(s, w, temp);
                    temp2 = _mm256_fmadd_ps(s2, w, temp2);
                    temp3 = _mm256_fmadd_ps(s3, w, temp3);
                    temp4 = _mm256_fmadd_ps(s4, w, temp4);
                    temp5 = _mm256_fmadd_ps(s, w2, temp5);
                    temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                    temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                    temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                    // d+1 ------------------------------------------------------------
                    // |
                    // |
                    // v


                    in_subscript += 1;

                    filter_subscript += Output_depth_dim;
                    

                    s = _mm256_broadcast_ss(&in_FP[in_subscript]);
                    s2 = _mm256_broadcast_ss(&in_FP[in_subscript + Input_depth_dim]);
                    s3 = _mm256_broadcast_ss(&in_FP[in_subscript + (2 << in_l_shift)]);
                    s4 = _mm256_broadcast_ss(&in_FP[in_subscript + (3 << in_l_shift)]);

                    w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                    w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                    temp = _mm256_fmadd_ps(s, w, temp);
                    temp2 = _mm256_fmadd_ps(s2, w, temp2);
                    temp3 = _mm256_fmadd_ps(s3, w, temp3);
                    temp4 = _mm256_fmadd_ps(s4, w, temp4);
                    temp5 = _mm256_fmadd_ps(s, w2, temp5);
                    temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                    temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                    temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                    // d+2 ------------------------------------------------------------
                    // |
                    // |
                    // v


                    in_subscript += 1;

                    filter_subscript += Output_depth_dim;
                    

                    s = _mm256_broadcast_ss(&in_FP[in_subscript]);
                    s2 = _mm256_broadcast_ss(&in_FP[in_subscript + Input_depth_dim]);
                    s3 = _mm256_broadcast_ss(&in_FP[in_subscript + (2 << in_l_shift)]);
                    s4 = _mm256_broadcast_ss(&in_FP[in_subscript + (3 << in_l_shift)]);

                    w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                    w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                    temp = _mm256_fmadd_ps(s, w, temp);
                    temp2 = _mm256_fmadd_ps(s2, w, temp2);
                    temp3 = _mm256_fmadd_ps(s3, w, temp3);
                    temp4 = _mm256_fmadd_ps(s4, w, temp4);
                    temp5 = _mm256_fmadd_ps(s, w2, temp5);
                    temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                    temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                    temp8 = _mm256_fmadd_ps(s4, w2, temp8);

                    // d+3 ------------------------------------------------------------
                    // |
                    // |
                    // v


                    in_subscript += 1;

                    filter_subscript += Output_depth_dim;
                    

                    s = _mm256_broadcast_ss(&in_FP[in_subscript]);
                    s2 = _mm256_broadcast_ss(&in_FP[in_subscript + Input_depth_dim]);
                    s3 = _mm256_broadcast_ss(&in_FP[in_subscript + (2 << in_l_shift)]);
                    s4 = _mm256_broadcast_ss(&in_FP[in_subscript + (3 << in_l_shift)]);

                    w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                    w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                    temp = _mm256_fmadd_ps(s, w, temp);
                    temp2 = _mm256_fmadd_ps(s2, w, temp2);
                    temp3 = _mm256_fmadd_ps(s3, w, temp3);
                    temp4 = _mm256_fmadd_ps(s4, w, temp4);
                    temp5 = _mm256_fmadd_ps(s, w2, temp5);
                    temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                    temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                    temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                    // d+4 ------------------------------------------------------------
                    // |
                    // |
                    // v


                    in_subscript += 1;

                    filter_subscript += Output_depth_dim;
                    

                    s = _mm256_broadcast_ss(&in_FP[in_subscript]);
                    s2 = _mm256_broadcast_ss(&in_FP[in_subscript + Input_depth_dim]);
                    s3 = _mm256_broadcast_ss(&in_FP[in_subscript + (2 << in_l_shift)]);
                    s4 = _mm256_broadcast_ss(&in_FP[in_subscript + (3 << in_l_shift)]);

                    w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                    w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                    temp = _mm256_fmadd_ps(s, w, temp);
                    temp2 = _mm256_fmadd_ps(s2, w, temp2);
                    temp3 = _mm256_fmadd_ps(s3, w, temp3);
                    temp4 = _mm256_fmadd_ps(s4, w, temp4);
                    temp5 = _mm256_fmadd_ps(s, w2, temp5);
                    temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                    temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                    temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                    // d+5 ------------------------------------------------------------
                    // |
                    // |
                    // v


                    in_subscript += 1;

                    filter_subscript += Output_depth_dim;
                    

                    s = _mm256_broadcast_ss(&in_FP[in_subscript]);
                    s2 = _mm256_broadcast_ss(&in_FP[in_subscript + Input_depth_dim]);
                    s3 = _mm256_broadcast_ss(&in_FP[in_subscript + (2 << in_l_shift)]);
                    s4 = _mm256_broadcast_ss(&in_FP[in_subscript + (3 << in_l_shift)]);

                    w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                    w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                    temp = _mm256_fmadd_ps(s, w, temp);
                    temp2 = _mm256_fmadd_ps(s2, w, temp2);
                    temp3 = _mm256_fmadd_ps(s3, w, temp3);
                    temp4 = _mm256_fmadd_ps(s4, w, temp4);
                    temp5 = _mm256_fmadd_ps(s, w2, temp5);
                    temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                    temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                    temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                    // d+6 ------------------------------------------------------------
                    // |
                    // |
                    // v


                    in_subscript += 1;

                    filter_subscript += Output_depth_dim;
                    

                    s = _mm256_broadcast_ss(&in_FP[in_subscript]);
                    s2 = _mm256_broadcast_ss(&in_FP[in_subscript + Input_depth_dim]);
                    s3 = _mm256_broadcast_ss(&in_FP[in_subscript + (2 << in_l_shift)]);
                    s4 = _mm256_broadcast_ss(&in_FP[in_subscript + (3 << in_l_shift)]);

                    w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                    w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                    temp = _mm256_fmadd_ps(s, w, temp);
                    temp2 = _mm256_fmadd_ps(s2, w, temp2);
                    temp3 = _mm256_fmadd_ps(s3, w, temp3);
                    temp4 = _mm256_fmadd_ps(s4, w, temp4);
                    temp5 = _mm256_fmadd_ps(s, w2, temp5);
                    temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                    temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                    temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                    // d+7 ------------------------------------------------------------
                    // |
                    // |
                    // v


                    in_subscript += 1;

                    filter_subscript += Output_depth_dim;
                    

                    s = _mm256_broadcast_ss(&in_FP[in_subscript]);
                    s2 = _mm256_broadcast_ss(&in_FP[in_subscript + Input_depth_dim]);
                    s3 = _mm256_broadcast_ss(&in_FP[in_subscript + (2 << in_l_shift)]);
                    s4 = _mm256_broadcast_ss(&in_FP[in_subscript + (3 << in_l_shift)]);

                    w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                    w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


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

              unsigned long long int out_subscript = b_out +
                y_out +
                x_out
                + m;
              const unsigned long long int out_subscript2 = out_subscript + Output_depth_dim;
              const unsigned long long int out_subscript3 = out_subscript + (2 << out_l_shift);
              const unsigned long long int out_subscript4 = out_subscript + (3 << out_l_shift);
              const unsigned long long int out_subscript5 = out_subscript + 8;
              const unsigned long long int out_subscript6 = out_subscript + Output_depth_dim + 8;
              const unsigned long long int out_subscript7 = out_subscript + (2 << out_l_shift) + 8;
              const unsigned long long int out_subscript8 = out_subscript + (3 << out_l_shift) + 8;

              
              temp = _mm256_add_ps(temp, bias);
              temp = _mm256_max_ps(temp, _mm256_setzero_ps());
              _mm256_store_ps(&out_to_compare_with_FP[out_subscript], temp);

              temp2 = _mm256_add_ps(temp2, bias);
              temp2 = _mm256_max_ps(temp2, _mm256_setzero_ps());
              _mm256_store_ps(&out_to_compare_with_FP[out_subscript2], temp2); // x+1

              temp3 = _mm256_add_ps(temp3, bias);
              temp3 = _mm256_max_ps(temp3, _mm256_setzero_ps());
              _mm256_store_ps(&out_to_compare_with_FP[out_subscript3], temp3); // x+2

              temp4 = _mm256_add_ps(temp4, bias);
              temp4 = _mm256_max_ps(temp4, _mm256_setzero_ps());
              _mm256_store_ps(&out_to_compare_with_FP[out_subscript4], temp4); // x+3

              temp5 = _mm256_add_ps(temp5, bias2);
              temp5 = _mm256_max_ps(temp5, _mm256_setzero_ps());
              _mm256_store_ps(&out_to_compare_with_FP[out_subscript + 8], temp5); // m+8

              temp6 = _mm256_add_ps(temp6, bias2);
              temp6 = _mm256_max_ps(temp6, _mm256_setzero_ps());
              _mm256_store_ps(&out_to_compare_with_FP[out_subscript2 + 8], temp6); // m+8, x+1

              temp7 = _mm256_add_ps(temp7, bias2);
              temp7 = _mm256_max_ps(temp7, _mm256_setzero_ps());
              _mm256_store_ps(&out_to_compare_with_FP[out_subscript3 + 8], temp7); // m+8, x+2

              temp8 = _mm256_add_ps(temp8, bias2);
              temp8 = _mm256_max_ps(temp8, _mm256_setzero_ps());
              _mm256_store_ps(&out_to_compare_with_FP[out_subscript4 + 8], temp8); // m+8, x+3
            }
          }
          for (; x < Output_X_dim; x++) {
            const unsigned long long int x_stride = x * Stride_X_dim;
            const unsigned long long int x_out = x << out_l_shift;

            for (unsigned int m = 0; m < Output_depth_dim; m+=16) { //channels
              const __m256 bias = _mm256_load_ps(&bias_array_FP[m]);
              const __m256 bias2 = _mm256_load_ps(&bias_array_FP[m+8]);

              __m256 temp = _mm256_setzero_ps();
              __m256 temp2 = _mm256_setzero_ps();

              for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
                const unsigned long long int off_y_stride = (y_stride + off_y) * in_x_depth;
                const unsigned long long int off_y_mask = off_y * in_out_mask;

                for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                  const unsigned long long int off_x_stride = (x_stride+ off_x) << in_l_shift;
                  const unsigned long long int off_x_mask = off_x << in_out_depth;
                  
                  for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                    unsigned long long int in_subscript = b_depth
                      + off_y_stride
                      + off_x_stride
                      + d;

                    unsigned long long int filter_subscript = off_y_mask
                      + off_x_mask
                      + (d << out_l_shift)
                      + m;
                    // unsigned long long int filter_subscript2 = filter_subscript + 8;
                    // unsigned long long int filter_subscript3 = filter_subscript + 16;
                    // unsigned long long int filter_subscript4 = filter_subscript + 24;

                    
                    __m256 s = _mm256_broadcast_ss(&in_FP[in_subscript]);

                    __m256 w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                    __m256 w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                    temp = _mm256_fmadd_ps(s, w, temp);
                    temp2 = _mm256_fmadd_ps(s, w2, temp2);




                    // d+1 ------------------------------------------------------------
                    // |
                    // |
                    // v


                    // in_subscript += 1;

                    filter_subscript += Output_depth_dim;
                    

                    s = _mm256_broadcast_ss(&in_FP[in_subscript+1]);

                    w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                    w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                    temp = _mm256_fmadd_ps(s, w, temp);
                    temp2 = _mm256_fmadd_ps(s, w2, temp2);


                    // d+2 ------------------------------------------------------------
                    // |
                    // |
                    // v


                    // in_subscript += 1;

                    filter_subscript += Output_depth_dim;
                    

                    s = _mm256_broadcast_ss(&in_FP[in_subscript+2]);

                    w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                    w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                    temp = _mm256_fmadd_ps(s, w, temp);
                    temp2 = _mm256_fmadd_ps(s, w2, temp2);


                    // d+3 ------------------------------------------------------------
                    // |
                    // |
                    // v


                    // in_subscript += 1;

                    filter_subscript += Output_depth_dim;
                    

                    s = _mm256_broadcast_ss(&in_FP[in_subscript+3]);

                    w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                    w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                    temp = _mm256_fmadd_ps(s, w, temp);
                    temp2 = _mm256_fmadd_ps(s, w2, temp2);



                    // d+4 ------------------------------------------------------------
                    // |
                    // |
                    // v


                    // in_subscript += 1;

                    filter_subscript += Output_depth_dim;
                    

                    s = _mm256_broadcast_ss(&in_FP[in_subscript+4]);

                    w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                    w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                    temp = _mm256_fmadd_ps(s, w, temp);
                    temp2 = _mm256_fmadd_ps(s, w2, temp2);



                    // d+5 ------------------------------------------------------------
                    // |
                    // |
                    // v


                    // in_subscript += 1;

                    filter_subscript += Output_depth_dim;
                    

                    s = _mm256_broadcast_ss(&in_FP[in_subscript+5]);

                    w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                    w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                    temp = _mm256_fmadd_ps(s, w, temp);
                    temp2 = _mm256_fmadd_ps(s, w2, temp2);



                    // d+6 ------------------------------------------------------------
                    // |
                    // |
                    // v


                    // in_subscript += 1;

                    filter_subscript += Output_depth_dim;
                    

                    s = _mm256_broadcast_ss(&in_FP[in_subscript+6]);

                    w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                    w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                    temp = _mm256_fmadd_ps(s, w, temp);
                    temp2 = _mm256_fmadd_ps(s, w2, temp2);


                    // d+7 ------------------------------------------------------------
                    // |
                    // |
                    // v


                    // in_subscript += 1;

                    filter_subscript += Output_depth_dim;
                    

                    s = _mm256_broadcast_ss(&in_FP[in_subscript+7]);

                    w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                    w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                    temp = _mm256_fmadd_ps(s, w, temp);
                    temp2 = _mm256_fmadd_ps(s, w2, temp2);
                  }
                }
              }

              unsigned long long int out_subscript = b_out +
                y_out +
                x_out
                + m;
              const unsigned long long int out_subscript2 = out_subscript + 8;

              
              temp = _mm256_add_ps(temp, bias);
              temp = _mm256_max_ps(temp, _mm256_setzero_ps());
              _mm256_store_ps(&out_to_compare_with_FP[out_subscript], temp);

              temp2 = _mm256_add_ps(temp2, bias2);
              temp2 = _mm256_max_ps(temp2, _mm256_setzero_ps());
              _mm256_store_ps(&out_to_compare_with_FP[out_subscript2], temp2); // x+1
            }
          }
        }
      }
    }
  }
  #undef m_tile
  #undef left_shift
  printf("\n from AC optv16 m16 x4 d8 - edge cases tiled %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  _mm_free(filter_FP_copy);
  return 0;
}



// ~603 GFLOPS, perf increase over optimised_layer_v16_AC_x4m16d8_edge_cases_FP due to having to check for edge cases less often
// only slightly better performing than 2 above when fallback x loop is used (531 GFLOPS vs 536 GFLOPS)
int optimised_layer_v16_AC_x4m16d8_edge_cases_interchange_x_m_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {

  #define left_shift(x) (__builtin_ctz(x))  // inline function effectively using inline assembly to return num of trailing zeros (i.e. 128 returns 7, 2 * 128 == 2 << 7)

  const unsigned long long int out_l_shift = left_shift(Output_depth_dim);
  const unsigned long long int in_l_shift = left_shift(Input_depth_dim);

  const unsigned long long int mask_yx_depth = Mask_Y_dim * Mask_X_dim << in_l_shift;
  const unsigned long long int mask_x_depth = Mask_X_dim << in_l_shift;

  const unsigned long long int in_yx_depth = Input_Y_dim * Input_X_dim << in_l_shift;
  const unsigned long long int in_x_depth = Input_X_dim << in_l_shift;

  const unsigned long long int in_out_mask = Mask_X_dim << (out_l_shift + in_l_shift);
  const unsigned long long int in_out_depth = out_l_shift + in_l_shift;

  const unsigned long long int out_yx_depth = Output_Y_dim * Output_X_dim << out_l_shift;
  const unsigned long long int out_x_depth = Output_X_dim << out_l_shift;


  const unsigned int filter_FP_length = Output_depth_dim * Mask_Y_dim * Mask_X_dim * Input_depth_dim;
  float* filter_FP_copy = (float*)_mm_malloc(filter_FP_length * sizeof(float), 64);
  if (filter_FP_copy == NULL) {
    printf("\nerror with malloc allocating filter array copy");
    exit(EXIT_FAILURE);
  }

  // array copying - filter_FP into form usable for vectorising m loop
  unsigned long long int new_subscript = 0;
  for (unsigned int y = 0; y < Mask_Y_dim; y++) {
    const unsigned long long int y_old = y * mask_x_depth;
    // const unsigned long long int y_old2 = (y+1) * mask_x_depth;

    for (unsigned int x = 0; x < Mask_X_dim; x++) {
      const unsigned long long int x_old = x << in_l_shift;
      // const unsigned long long int x_old2 = (x+1) << left_shift(Input_depth_dim);

      for (unsigned int d = 0; d < Input_depth_dim; d+=2) {
        for (unsigned int m = 0; m < Output_depth_dim; m+=4) {

          unsigned long long int old_subscript = m * mask_yx_depth
            + y_old
            + x_old
            + d;

          filter_FP_copy[new_subscript] = filter_FP[old_subscript];
          filter_FP_copy[new_subscript+1] = filter_FP[old_subscript + mask_yx_depth];
          filter_FP_copy[new_subscript+2] = filter_FP[old_subscript + (mask_yx_depth << 1)];
          filter_FP_copy[new_subscript+3] = filter_FP[old_subscript + (mask_yx_depth << 1) + mask_yx_depth];
          filter_FP_copy[new_subscript+Output_depth_dim] = filter_FP[old_subscript + 1];
          filter_FP_copy[new_subscript+1+Output_depth_dim] = filter_FP[old_subscript + mask_yx_depth + 1];
          filter_FP_copy[new_subscript+2+Output_depth_dim] = filter_FP[old_subscript + (mask_yx_depth << 1) + 1];
          filter_FP_copy[new_subscript+3+Output_depth_dim] = filter_FP[old_subscript + (mask_yx_depth << 1) + mask_yx_depth + 1];

          new_subscript+=4;

          // filter_FP_copy[new_subscript] = filter_FP[old_subscript];
          // filter_FP_copy[new_subscript+Output_depth_dim] = filter_FP[old_subscript + 1];
          // filter_FP_copy[new_subscript+(2 << out_l_shift)] = filter_FP[old_subscript + 2];
          // filter_FP_copy[new_subscript+(3 << out_l_shift)] = filter_FP[old_subscript + 3];
          // filter_FP_copy[new_subscript+(4 << out_l_shift)] = filter_FP[old_subscript + 4];
          // filter_FP_copy[new_subscript+(5 << out_l_shift)] = filter_FP[old_subscript + 5];
          // filter_FP_copy[new_subscript+(6 << out_l_shift)] = filter_FP[old_subscript + 6];
          // filter_FP_copy[new_subscript+(7 << out_l_shift)] = filter_FP[old_subscript + 7];
          // filter_FP_copy[new_subscript+in_out_mask] = filter_FP[old_subscript3];
          // filter_FP_copy[new_subscript+in_out_mask + Input_depth_dim + Output_depth_dim] = filter_FP[old_subscript4];

          // new_subscript++;
        }
        new_subscript+=Output_depth_dim;
      }
    }
  }

  unsigned int x;
  const unsigned int x_bound = (Output_X_dim/ 4) * 4; 
  // printf("\nOutput_x_dim = %i\nx_bound = %i\n", Output_X_dim, x_bound);

    // main loop body
  #pragma omp parallel for private(x) shared(in_FP, filter_FP_copy, out_l_shift, in_l_shift, in_yx_depth, in_x_depth, in_out_mask, in_out_depth, \
    out_yx_depth, out_x_depth, x_bound) default(shared) collapse(2) schedule(static)
  {
    // main loop body
    for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
      const unsigned long long int b_depth = b * in_yx_depth;
      const unsigned long long int b_out = b * out_yx_depth;

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        const unsigned long long int y_stride = y * Stride_Y_dim;
        const unsigned long long int y_out = y * out_x_depth;

        for (unsigned int m = 0; m < Output_depth_dim; m+=16) { //channels
          const __m256 bias = _mm256_load_ps(&bias_array_FP[m]);
          const __m256 bias2 = _mm256_load_ps(&bias_array_FP[m+8]);

          for (x = 0; x < x_bound; x+=4) {	//Output Width
            const unsigned long long int x_stride = x * Stride_X_dim;

            __m256 temp = _mm256_setzero_ps();
            __m256 temp2 = _mm256_setzero_ps();
            __m256 temp3 = _mm256_setzero_ps();
            __m256 temp4 = _mm256_setzero_ps();
            __m256 temp5 = _mm256_setzero_ps();
            __m256 temp6 = _mm256_setzero_ps();
            __m256 temp7 = _mm256_setzero_ps();
            __m256 temp8 = _mm256_setzero_ps();

            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              const unsigned long long int off_y_stride = (y_stride + off_y) * in_x_depth;
              const unsigned long long int off_y_mask = off_y * in_out_mask;

              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                const unsigned long long int off_x_stride = (x_stride+ off_x) << in_l_shift;
                const unsigned long long int off_x_mask = off_x << in_out_depth;
                
                for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                  unsigned long long int in_subscript = b_depth
                    + off_y_stride
                    + off_x_stride
                    + d;

                  unsigned long long int filter_subscript = off_y_mask
                    + off_x_mask
                    + (d << out_l_shift)
                    + m;
                  // unsigned long long int filter_subscript2 = filter_subscript + 8;
                  // unsigned long long int filter_subscript3 = filter_subscript + 16;
                  // unsigned long long int filter_subscript4 = filter_subscript + 24;

                  
                  __m256 s = _mm256_broadcast_ss(&in_FP[in_subscript]);
                  __m256 s2 = _mm256_broadcast_ss(&in_FP[in_subscript + Input_depth_dim]);
                  __m256 s3 = _mm256_broadcast_ss(&in_FP[in_subscript + (2 << in_l_shift)]);
                  __m256 s4 = _mm256_broadcast_ss(&in_FP[in_subscript + (3 << in_l_shift)]);

                  __m256 w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                  __m256 w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                  // d+1 ------------------------------------------------------------
                  // |
                  // |
                  // v


                  in_subscript += 1;

                  filter_subscript += Output_depth_dim;
                  

                  s = _mm256_broadcast_ss(&in_FP[in_subscript]);
                  s2 = _mm256_broadcast_ss(&in_FP[in_subscript + Input_depth_dim]);
                  s3 = _mm256_broadcast_ss(&in_FP[in_subscript + (2 << in_l_shift)]);
                  s4 = _mm256_broadcast_ss(&in_FP[in_subscript + (3 << in_l_shift)]);

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                  // d+2 ------------------------------------------------------------
                  // |
                  // |
                  // v


                  in_subscript += 1;

                  filter_subscript += Output_depth_dim;
                  

                  s = _mm256_broadcast_ss(&in_FP[in_subscript]);
                  s2 = _mm256_broadcast_ss(&in_FP[in_subscript + Input_depth_dim]);
                  s3 = _mm256_broadcast_ss(&in_FP[in_subscript + (2 << in_l_shift)]);
                  s4 = _mm256_broadcast_ss(&in_FP[in_subscript + (3 << in_l_shift)]);

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);

                  // d+3 ------------------------------------------------------------
                  // |
                  // |
                  // v


                  in_subscript += 1;

                  filter_subscript += Output_depth_dim;
                  

                  s = _mm256_broadcast_ss(&in_FP[in_subscript]);
                  s2 = _mm256_broadcast_ss(&in_FP[in_subscript + Input_depth_dim]);
                  s3 = _mm256_broadcast_ss(&in_FP[in_subscript + (2 << in_l_shift)]);
                  s4 = _mm256_broadcast_ss(&in_FP[in_subscript + (3 << in_l_shift)]);

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                  // d+4 ------------------------------------------------------------
                  // |
                  // |
                  // v


                  in_subscript += 1;

                  filter_subscript += Output_depth_dim;
                  

                  s = _mm256_broadcast_ss(&in_FP[in_subscript]);
                  s2 = _mm256_broadcast_ss(&in_FP[in_subscript + Input_depth_dim]);
                  s3 = _mm256_broadcast_ss(&in_FP[in_subscript + (2 << in_l_shift)]);
                  s4 = _mm256_broadcast_ss(&in_FP[in_subscript + (3 << in_l_shift)]);

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                  // d+5 ------------------------------------------------------------
                  // |
                  // |
                  // v


                  in_subscript += 1;

                  filter_subscript += Output_depth_dim;
                  

                  s = _mm256_broadcast_ss(&in_FP[in_subscript]);
                  s2 = _mm256_broadcast_ss(&in_FP[in_subscript + Input_depth_dim]);
                  s3 = _mm256_broadcast_ss(&in_FP[in_subscript + (2 << in_l_shift)]);
                  s4 = _mm256_broadcast_ss(&in_FP[in_subscript + (3 << in_l_shift)]);

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                  // d+6 ------------------------------------------------------------
                  // |
                  // |
                  // v


                  in_subscript += 1;

                  filter_subscript += Output_depth_dim;
                  

                  s = _mm256_broadcast_ss(&in_FP[in_subscript]);
                  s2 = _mm256_broadcast_ss(&in_FP[in_subscript + Input_depth_dim]);
                  s3 = _mm256_broadcast_ss(&in_FP[in_subscript + (2 << in_l_shift)]);
                  s4 = _mm256_broadcast_ss(&in_FP[in_subscript + (3 << in_l_shift)]);

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                  // d+7 ------------------------------------------------------------
                  // |
                  // |
                  // v


                  in_subscript += 1;

                  filter_subscript += Output_depth_dim;
                  

                  s = _mm256_broadcast_ss(&in_FP[in_subscript]);
                  s2 = _mm256_broadcast_ss(&in_FP[in_subscript + Input_depth_dim]);
                  s3 = _mm256_broadcast_ss(&in_FP[in_subscript + (2 << in_l_shift)]);
                  s4 = _mm256_broadcast_ss(&in_FP[in_subscript + (3 << in_l_shift)]);

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


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

            unsigned long long int out_subscript = b_out +
              y_out +
              (x << out_l_shift)
              + m;
            const unsigned long long int out_subscript2 = out_subscript + Output_depth_dim;
            const unsigned long long int out_subscript3 = out_subscript + (2 << out_l_shift);
            const unsigned long long int out_subscript4 = out_subscript + (3 << out_l_shift);
            const unsigned long long int out_subscript5 = out_subscript + 8;
            const unsigned long long int out_subscript6 = out_subscript + Output_depth_dim + 8;
            const unsigned long long int out_subscript7 = out_subscript + (2 << out_l_shift) + 8;
            const unsigned long long int out_subscript8 = out_subscript + (3 << out_l_shift) + 8;

            
            temp = _mm256_add_ps(temp, bias);
            temp = _mm256_max_ps(temp, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript], temp);

            temp2 = _mm256_add_ps(temp2, bias);
            temp2 = _mm256_max_ps(temp2, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript2], temp2); // x+1

            temp3 = _mm256_add_ps(temp3, bias);
            temp3 = _mm256_max_ps(temp3, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript3], temp3); // x+2

            temp4 = _mm256_add_ps(temp4, bias);
            temp4 = _mm256_max_ps(temp4, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript4], temp4); // x+3

            temp5 = _mm256_add_ps(temp5, bias2);
            temp5 = _mm256_max_ps(temp5, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript + 8], temp5); // m+8

            temp6 = _mm256_add_ps(temp6, bias2);
            temp6 = _mm256_max_ps(temp6, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript2 + 8], temp6); // m+8, x+1

            temp7 = _mm256_add_ps(temp7, bias2);
            temp7 = _mm256_max_ps(temp7, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript3 + 8], temp7); // m+8, x+2

            temp8 = _mm256_add_ps(temp8, bias2);
            temp8 = _mm256_max_ps(temp8, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript4 + 8], temp8); // m+8, x+3
          }
          for (; x < Output_X_dim; x++) {
            const unsigned long long int x_stride = x * Stride_X_dim;

            __m256 temp = _mm256_setzero_ps();
            __m256 temp2 = _mm256_setzero_ps();

            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              const unsigned long long int off_y_stride = (y_stride + off_y) * in_x_depth;
              const unsigned long long int off_y_mask = off_y * in_out_mask;

              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                const unsigned long long int off_x_stride = (x_stride+ off_x) << in_l_shift;
                const unsigned long long int off_x_mask = off_x << in_out_depth;
                
                for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                  unsigned long long int in_subscript = b_depth
                    + off_y_stride
                    + off_x_stride
                    + d;

                  unsigned long long int filter_subscript = off_y_mask
                    + off_x_mask
                    + (d << out_l_shift)
                    + m;
                  // unsigned long long int filter_subscript2 = filter_subscript + 8;
                  // unsigned long long int filter_subscript3 = filter_subscript + 16;
                  // unsigned long long int filter_subscript4 = filter_subscript + 24;

                  
                  __m256 s = _mm256_broadcast_ss(&in_FP[in_subscript]);

                  __m256 w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                  __m256 w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s, w2, temp2);



                  // d+1 ------------------------------------------------------------
                  // |
                  // |
                  // v


                  // in_subscript += 1;

                  filter_subscript += Output_depth_dim;
                  

                  s = _mm256_broadcast_ss(&in_FP[in_subscript+1]);

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s, w2, temp2);


                  // d+2 ------------------------------------------------------------
                  // |
                  // |
                  // v


                  // in_subscript += 1;

                  filter_subscript += Output_depth_dim;
                  

                  s = _mm256_broadcast_ss(&in_FP[in_subscript+2]);

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s, w2, temp2);

                  // d+3 ------------------------------------------------------------
                  // |
                  // |
                  // v


                  // in_subscript += 1;

                  filter_subscript += Output_depth_dim;
                  

                  s = _mm256_broadcast_ss(&in_FP[in_subscript+3]);

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s, w2, temp2);


                  // d+4 ------------------------------------------------------------
                  // |
                  // |
                  // v


                  // in_subscript += 1;

                  filter_subscript += Output_depth_dim;
                  

                  s = _mm256_broadcast_ss(&in_FP[in_subscript+4]);

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s, w2, temp2);


                  // d+5 ------------------------------------------------------------
                  // |
                  // |
                  // v


                  // in_subscript += 1;

                  filter_subscript += Output_depth_dim;
                  

                  s = _mm256_broadcast_ss(&in_FP[in_subscript+5]);

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s, w2, temp2);


                  // d+6 ------------------------------------------------------------
                  // |
                  // |
                  // v


                  // in_subscript += 1;

                  filter_subscript += Output_depth_dim;
                  

                  s = _mm256_broadcast_ss(&in_FP[in_subscript+6]);

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s, w2, temp2);


                  // d+7 ------------------------------------------------------------
                  // |
                  // |
                  // v


                  // in_subscript += 1;

                  filter_subscript += Output_depth_dim;
                  

                  s = _mm256_broadcast_ss(&in_FP[in_subscript+7]);

                  w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                  w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s, w2, temp2);
                }
              }
            }

            const unsigned long long int out_subscript = b_out +
              y_out +
              (x << out_l_shift)
              + m;
            const unsigned long long int out_subscript2 = out_subscript + 8;

            
            temp = _mm256_add_ps(temp, bias);
            temp = _mm256_max_ps(temp, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript], temp);

            temp2 = _mm256_add_ps(temp2, bias2);
            temp2 = _mm256_max_ps(temp2, _mm256_setzero_ps());
            _mm256_store_ps(&out_to_compare_with_FP[out_subscript2], temp2); 
          }
        }
      }
    }
  }
  #undef left_shift
  printf("\n from AC optv15 m16 x4 d8 - edge cases interchanged m and x %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  _mm_free(filter_FP_copy);
  return 0;
}


// tiled version of above
// ~500 GFLOPS,
int optimised_layer_v16_AC_x4m16d8_tiled_edge_cases_interchange_x_m_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {

  #define m_tile 16
  #define left_shift(x) (__builtin_ctz(x))  // inline function effectively using inline assembly to return num of trailing zeros (i.e. 128 returns 7, 2 * 128 == 2 << 7)

  const unsigned long long int out_l_shift = left_shift(Output_depth_dim);
  const unsigned long long int in_l_shift = left_shift(Input_depth_dim);

  const unsigned long long int mask_yx_depth = Mask_Y_dim * Mask_X_dim << in_l_shift;
  const unsigned long long int mask_x_depth = Mask_X_dim << in_l_shift;

  const unsigned long long int in_yx_depth = Input_Y_dim * Input_X_dim << in_l_shift;
  const unsigned long long int in_x_depth = Input_X_dim << in_l_shift;

  const unsigned long long int in_out_mask = Mask_X_dim << (out_l_shift + in_l_shift);
  const unsigned long long int in_out_depth = out_l_shift + in_l_shift;

  const unsigned long long int out_yx_depth = Output_Y_dim * Output_X_dim << out_l_shift;
  const unsigned long long int out_x_depth = Output_X_dim << out_l_shift;


  const unsigned int filter_FP_length = Output_depth_dim * Mask_Y_dim * Mask_X_dim * Input_depth_dim;
  float* filter_FP_copy = (float*)_mm_malloc(filter_FP_length * sizeof(float), 64);
  if (filter_FP_copy == NULL) {
    printf("\nerror with malloc allocating filter array copy");
    exit(EXIT_FAILURE);
  }

  // array copying - filter_FP into form usable for vectorising m loop
  unsigned long long int new_subscript = 0;
  for (unsigned int y = 0; y < Mask_Y_dim; y++) {
    const unsigned long long int y_old = y * mask_x_depth;
    // const unsigned long long int y_old2 = (y+1) * mask_x_depth;

    for (unsigned int x = 0; x < Mask_X_dim; x++) {
      const unsigned long long int x_old = x << in_l_shift;
      // const unsigned long long int x_old2 = (x+1) << left_shift(Input_depth_dim);

      for (unsigned int d = 0; d < Input_depth_dim; d+=2) {
        for (unsigned int m = 0; m < Output_depth_dim; m+=4) {

          unsigned long long int old_subscript = m * mask_yx_depth
            + y_old
            + x_old
            + d;

          filter_FP_copy[new_subscript] = filter_FP[old_subscript];
          filter_FP_copy[new_subscript+1] = filter_FP[old_subscript + mask_yx_depth];
          filter_FP_copy[new_subscript+2] = filter_FP[old_subscript + (mask_yx_depth << 1)];
          filter_FP_copy[new_subscript+3] = filter_FP[old_subscript + (mask_yx_depth << 1) + mask_yx_depth];
          filter_FP_copy[new_subscript+Output_depth_dim] = filter_FP[old_subscript + 1];
          filter_FP_copy[new_subscript+1+Output_depth_dim] = filter_FP[old_subscript + mask_yx_depth + 1];
          filter_FP_copy[new_subscript+2+Output_depth_dim] = filter_FP[old_subscript + (mask_yx_depth << 1) + 1];
          filter_FP_copy[new_subscript+3+Output_depth_dim] = filter_FP[old_subscript + (mask_yx_depth << 1) + mask_yx_depth + 1];

          new_subscript+=4;

          // filter_FP_copy[new_subscript] = filter_FP[old_subscript];
          // filter_FP_copy[new_subscript+Output_depth_dim] = filter_FP[old_subscript + 1];
          // filter_FP_copy[new_subscript+(2 << out_l_shift)] = filter_FP[old_subscript + 2];
          // filter_FP_copy[new_subscript+(3 << out_l_shift)] = filter_FP[old_subscript + 3];
          // filter_FP_copy[new_subscript+(4 << out_l_shift)] = filter_FP[old_subscript + 4];
          // filter_FP_copy[new_subscript+(5 << out_l_shift)] = filter_FP[old_subscript + 5];
          // filter_FP_copy[new_subscript+(6 << out_l_shift)] = filter_FP[old_subscript + 6];
          // filter_FP_copy[new_subscript+(7 << out_l_shift)] = filter_FP[old_subscript + 7];
          // filter_FP_copy[new_subscript+in_out_mask] = filter_FP[old_subscript3];
          // filter_FP_copy[new_subscript+in_out_mask + Input_depth_dim + Output_depth_dim] = filter_FP[old_subscript4];

          // new_subscript++;
        }
        new_subscript+=Output_depth_dim;
      }
    }
  }

  unsigned int x;
  const unsigned int x_bound = (Output_X_dim/ 4) * 4; 

    // main loop body
  #pragma omp parallel for private(x) shared(in_FP, filter_FP_copy, out_l_shift, in_l_shift, in_yx_depth, in_x_depth, in_out_mask, in_out_depth, \
    out_yx_depth, out_x_depth, x_bound) default(shared) collapse(2) schedule(static)
  {
    // main loop body
    for (unsigned int mm = 0; mm < Output_depth_dim; mm+=m_tile) {
      for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
        const unsigned long long int b_depth = b * in_yx_depth;
        const unsigned long long int b_out = b * out_yx_depth;

        for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
          const unsigned long long int y_stride = y * Stride_Y_dim;
          const unsigned long long int y_out = y * out_x_depth;

          for (unsigned int m = mm; m < mm+m_tile; m+=16) { //channels
            const __m256 bias = _mm256_load_ps(&bias_array_FP[m]);
            const __m256 bias2 = _mm256_load_ps(&bias_array_FP[m+8]);

            for (x = 0; x < x_bound; x+=4) {	//Output Width
              const unsigned long long int x_stride = x * Stride_X_dim;

              __m256 temp = _mm256_setzero_ps();
              __m256 temp2 = _mm256_setzero_ps();
              __m256 temp3 = _mm256_setzero_ps();
              __m256 temp4 = _mm256_setzero_ps();
              __m256 temp5 = _mm256_setzero_ps();
              __m256 temp6 = _mm256_setzero_ps();
              __m256 temp7 = _mm256_setzero_ps();
              __m256 temp8 = _mm256_setzero_ps();

              for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
                const unsigned long long int off_y_stride = (y_stride + off_y) * in_x_depth;
                const unsigned long long int off_y_mask = off_y * in_out_mask;

                for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                  const unsigned long long int off_x_stride = (x_stride+ off_x) << in_l_shift;
                  const unsigned long long int off_x_mask = off_x << in_out_depth;
                  
                  for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                    unsigned long long int in_subscript = b_depth
                      + off_y_stride
                      + off_x_stride
                      + d;

                    unsigned long long int filter_subscript = off_y_mask
                      + off_x_mask
                      + (d << out_l_shift)
                      + m;
                    // unsigned long long int filter_subscript2 = filter_subscript + 8;
                    // unsigned long long int filter_subscript3 = filter_subscript + 16;
                    // unsigned long long int filter_subscript4 = filter_subscript + 24;

                    
                    __m256 s = _mm256_broadcast_ss(&in_FP[in_subscript]);
                    __m256 s2 = _mm256_broadcast_ss(&in_FP[in_subscript + Input_depth_dim]);
                    __m256 s3 = _mm256_broadcast_ss(&in_FP[in_subscript + (2 << in_l_shift)]);
                    __m256 s4 = _mm256_broadcast_ss(&in_FP[in_subscript + (3 << in_l_shift)]);

                    __m256 w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                    __m256 w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                    temp = _mm256_fmadd_ps(s, w, temp);
                    temp2 = _mm256_fmadd_ps(s2, w, temp2);
                    temp3 = _mm256_fmadd_ps(s3, w, temp3);
                    temp4 = _mm256_fmadd_ps(s4, w, temp4);
                    temp5 = _mm256_fmadd_ps(s, w2, temp5);
                    temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                    temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                    temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                    // d+1 ------------------------------------------------------------
                    // |
                    // |
                    // v


                    in_subscript += 1;

                    filter_subscript += Output_depth_dim;
                    

                    s = _mm256_broadcast_ss(&in_FP[in_subscript]);
                    s2 = _mm256_broadcast_ss(&in_FP[in_subscript + Input_depth_dim]);
                    s3 = _mm256_broadcast_ss(&in_FP[in_subscript + (2 << in_l_shift)]);
                    s4 = _mm256_broadcast_ss(&in_FP[in_subscript + (3 << in_l_shift)]);

                    w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                    w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                    temp = _mm256_fmadd_ps(s, w, temp);
                    temp2 = _mm256_fmadd_ps(s2, w, temp2);
                    temp3 = _mm256_fmadd_ps(s3, w, temp3);
                    temp4 = _mm256_fmadd_ps(s4, w, temp4);
                    temp5 = _mm256_fmadd_ps(s, w2, temp5);
                    temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                    temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                    temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                    // d+2 ------------------------------------------------------------
                    // |
                    // |
                    // v


                    in_subscript += 1;

                    filter_subscript += Output_depth_dim;
                    

                    s = _mm256_broadcast_ss(&in_FP[in_subscript]);
                    s2 = _mm256_broadcast_ss(&in_FP[in_subscript + Input_depth_dim]);
                    s3 = _mm256_broadcast_ss(&in_FP[in_subscript + (2 << in_l_shift)]);
                    s4 = _mm256_broadcast_ss(&in_FP[in_subscript + (3 << in_l_shift)]);

                    w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                    w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                    temp = _mm256_fmadd_ps(s, w, temp);
                    temp2 = _mm256_fmadd_ps(s2, w, temp2);
                    temp3 = _mm256_fmadd_ps(s3, w, temp3);
                    temp4 = _mm256_fmadd_ps(s4, w, temp4);
                    temp5 = _mm256_fmadd_ps(s, w2, temp5);
                    temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                    temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                    temp8 = _mm256_fmadd_ps(s4, w2, temp8);

                    // d+3 ------------------------------------------------------------
                    // |
                    // |
                    // v


                    in_subscript += 1;

                    filter_subscript += Output_depth_dim;
                    

                    s = _mm256_broadcast_ss(&in_FP[in_subscript]);
                    s2 = _mm256_broadcast_ss(&in_FP[in_subscript + Input_depth_dim]);
                    s3 = _mm256_broadcast_ss(&in_FP[in_subscript + (2 << in_l_shift)]);
                    s4 = _mm256_broadcast_ss(&in_FP[in_subscript + (3 << in_l_shift)]);

                    w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                    w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                    temp = _mm256_fmadd_ps(s, w, temp);
                    temp2 = _mm256_fmadd_ps(s2, w, temp2);
                    temp3 = _mm256_fmadd_ps(s3, w, temp3);
                    temp4 = _mm256_fmadd_ps(s4, w, temp4);
                    temp5 = _mm256_fmadd_ps(s, w2, temp5);
                    temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                    temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                    temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                    // d+4 ------------------------------------------------------------
                    // |
                    // |
                    // v


                    in_subscript += 1;

                    filter_subscript += Output_depth_dim;
                    

                    s = _mm256_broadcast_ss(&in_FP[in_subscript]);
                    s2 = _mm256_broadcast_ss(&in_FP[in_subscript + Input_depth_dim]);
                    s3 = _mm256_broadcast_ss(&in_FP[in_subscript + (2 << in_l_shift)]);
                    s4 = _mm256_broadcast_ss(&in_FP[in_subscript + (3 << in_l_shift)]);

                    w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                    w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                    temp = _mm256_fmadd_ps(s, w, temp);
                    temp2 = _mm256_fmadd_ps(s2, w, temp2);
                    temp3 = _mm256_fmadd_ps(s3, w, temp3);
                    temp4 = _mm256_fmadd_ps(s4, w, temp4);
                    temp5 = _mm256_fmadd_ps(s, w2, temp5);
                    temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                    temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                    temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                    // d+5 ------------------------------------------------------------
                    // |
                    // |
                    // v


                    in_subscript += 1;

                    filter_subscript += Output_depth_dim;
                    

                    s = _mm256_broadcast_ss(&in_FP[in_subscript]);
                    s2 = _mm256_broadcast_ss(&in_FP[in_subscript + Input_depth_dim]);
                    s3 = _mm256_broadcast_ss(&in_FP[in_subscript + (2 << in_l_shift)]);
                    s4 = _mm256_broadcast_ss(&in_FP[in_subscript + (3 << in_l_shift)]);

                    w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                    w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                    temp = _mm256_fmadd_ps(s, w, temp);
                    temp2 = _mm256_fmadd_ps(s2, w, temp2);
                    temp3 = _mm256_fmadd_ps(s3, w, temp3);
                    temp4 = _mm256_fmadd_ps(s4, w, temp4);
                    temp5 = _mm256_fmadd_ps(s, w2, temp5);
                    temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                    temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                    temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                    // d+6 ------------------------------------------------------------
                    // |
                    // |
                    // v


                    in_subscript += 1;

                    filter_subscript += Output_depth_dim;
                    

                    s = _mm256_broadcast_ss(&in_FP[in_subscript]);
                    s2 = _mm256_broadcast_ss(&in_FP[in_subscript + Input_depth_dim]);
                    s3 = _mm256_broadcast_ss(&in_FP[in_subscript + (2 << in_l_shift)]);
                    s4 = _mm256_broadcast_ss(&in_FP[in_subscript + (3 << in_l_shift)]);

                    w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                    w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                    temp = _mm256_fmadd_ps(s, w, temp);
                    temp2 = _mm256_fmadd_ps(s2, w, temp2);
                    temp3 = _mm256_fmadd_ps(s3, w, temp3);
                    temp4 = _mm256_fmadd_ps(s4, w, temp4);
                    temp5 = _mm256_fmadd_ps(s, w2, temp5);
                    temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                    temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                    temp8 = _mm256_fmadd_ps(s4, w2, temp8);


                    // d+7 ------------------------------------------------------------
                    // |
                    // |
                    // v


                    in_subscript += 1;

                    filter_subscript += Output_depth_dim;
                    

                    s = _mm256_broadcast_ss(&in_FP[in_subscript]);
                    s2 = _mm256_broadcast_ss(&in_FP[in_subscript + Input_depth_dim]);
                    s3 = _mm256_broadcast_ss(&in_FP[in_subscript + (2 << in_l_shift)]);
                    s4 = _mm256_broadcast_ss(&in_FP[in_subscript + (3 << in_l_shift)]);

                    w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                    w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


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

              unsigned long long int out_subscript = b_out +
                y_out +
                (x << out_l_shift)
                + m;
              const unsigned long long int out_subscript2 = out_subscript + Output_depth_dim;
              const unsigned long long int out_subscript3 = out_subscript + (2 << out_l_shift);
              const unsigned long long int out_subscript4 = out_subscript + (3 << out_l_shift);
              const unsigned long long int out_subscript5 = out_subscript + 8;
              const unsigned long long int out_subscript6 = out_subscript + Output_depth_dim + 8;
              const unsigned long long int out_subscript7 = out_subscript + (2 << out_l_shift) + 8;
              const unsigned long long int out_subscript8 = out_subscript + (3 << out_l_shift) + 8;

              
              temp = _mm256_add_ps(temp, bias);
              temp = _mm256_max_ps(temp, _mm256_setzero_ps());
              _mm256_store_ps(&out_to_compare_with_FP[out_subscript], temp);

              temp2 = _mm256_add_ps(temp2, bias);
              temp2 = _mm256_max_ps(temp2, _mm256_setzero_ps());
              _mm256_store_ps(&out_to_compare_with_FP[out_subscript2], temp2); // x+1

              temp3 = _mm256_add_ps(temp3, bias);
              temp3 = _mm256_max_ps(temp3, _mm256_setzero_ps());
              _mm256_store_ps(&out_to_compare_with_FP[out_subscript3], temp3); // x+2

              temp4 = _mm256_add_ps(temp4, bias);
              temp4 = _mm256_max_ps(temp4, _mm256_setzero_ps());
              _mm256_store_ps(&out_to_compare_with_FP[out_subscript4], temp4); // x+3

              temp5 = _mm256_add_ps(temp5, bias2);
              temp5 = _mm256_max_ps(temp5, _mm256_setzero_ps());
              _mm256_store_ps(&out_to_compare_with_FP[out_subscript + 8], temp5); // m+8

              temp6 = _mm256_add_ps(temp6, bias2);
              temp6 = _mm256_max_ps(temp6, _mm256_setzero_ps());
              _mm256_store_ps(&out_to_compare_with_FP[out_subscript2 + 8], temp6); // m+8, x+1

              temp7 = _mm256_add_ps(temp7, bias2);
              temp7 = _mm256_max_ps(temp7, _mm256_setzero_ps());
              _mm256_store_ps(&out_to_compare_with_FP[out_subscript3 + 8], temp7); // m+8, x+2

              temp8 = _mm256_add_ps(temp8, bias2);
              temp8 = _mm256_max_ps(temp8, _mm256_setzero_ps());
              _mm256_store_ps(&out_to_compare_with_FP[out_subscript4 + 8], temp8); // m+8, x+3
            }
            for (; x < Output_X_dim; x++) {
              const unsigned long long int x_stride = x * Stride_X_dim;

              __m256 temp = _mm256_setzero_ps();
              __m256 temp2 = _mm256_setzero_ps();

              for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
                const unsigned long long int off_y_stride = (y_stride + off_y) * in_x_depth;
                const unsigned long long int off_y_mask = off_y * in_out_mask;

                for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                  const unsigned long long int off_x_stride = (x_stride+ off_x) << in_l_shift;
                  const unsigned long long int off_x_mask = off_x << in_out_depth;
                  
                  for (unsigned int d = 0; d < Input_depth_dim; d+=8) {

                    unsigned long long int in_subscript = b_depth
                      + off_y_stride
                      + off_x_stride
                      + d;

                    unsigned long long int filter_subscript = off_y_mask
                      + off_x_mask
                      + (d << out_l_shift)
                      + m;
                    // unsigned long long int filter_subscript2 = filter_subscript + 8;
                    // unsigned long long int filter_subscript3 = filter_subscript + 16;
                    // unsigned long long int filter_subscript4 = filter_subscript + 24;

                    
                    __m256 s = _mm256_broadcast_ss(&in_FP[in_subscript]);

                    __m256 w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                    __m256 w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                    temp = _mm256_fmadd_ps(s, w, temp);
                    temp2 = _mm256_fmadd_ps(s, w2, temp2);



                    // d+1 ------------------------------------------------------------
                    // |
                    // |
                    // v


                    // in_subscript += 1;

                    filter_subscript += Output_depth_dim;
                    

                    s = _mm256_broadcast_ss(&in_FP[in_subscript+1]);

                    w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                    w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                    temp = _mm256_fmadd_ps(s, w, temp);
                    temp2 = _mm256_fmadd_ps(s, w2, temp2);


                    // d+2 ------------------------------------------------------------
                    // |
                    // |
                    // v


                    // in_subscript += 1;

                    filter_subscript += Output_depth_dim;
                    

                    s = _mm256_broadcast_ss(&in_FP[in_subscript+2]);

                    w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                    w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                    temp = _mm256_fmadd_ps(s, w, temp);
                    temp2 = _mm256_fmadd_ps(s, w2, temp2);

                    // d+3 ------------------------------------------------------------
                    // |
                    // |
                    // v


                    // in_subscript += 1;

                    filter_subscript += Output_depth_dim;
                    

                    s = _mm256_broadcast_ss(&in_FP[in_subscript+3]);

                    w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                    w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                    temp = _mm256_fmadd_ps(s, w, temp);
                    temp2 = _mm256_fmadd_ps(s, w2, temp2);


                    // d+4 ------------------------------------------------------------
                    // |
                    // |
                    // v


                    // in_subscript += 1;

                    filter_subscript += Output_depth_dim;
                    

                    s = _mm256_broadcast_ss(&in_FP[in_subscript+4]);

                    w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                    w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                    temp = _mm256_fmadd_ps(s, w, temp);
                    temp2 = _mm256_fmadd_ps(s, w2, temp2);


                    // d+5 ------------------------------------------------------------
                    // |
                    // |
                    // v


                    // in_subscript += 1;

                    filter_subscript += Output_depth_dim;
                    

                    s = _mm256_broadcast_ss(&in_FP[in_subscript+5]);

                    w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                    w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                    temp = _mm256_fmadd_ps(s, w, temp);
                    temp2 = _mm256_fmadd_ps(s, w2, temp2);


                    // d+6 ------------------------------------------------------------
                    // |
                    // |
                    // v


                    // in_subscript += 1;

                    filter_subscript += Output_depth_dim;
                    

                    s = _mm256_broadcast_ss(&in_FP[in_subscript+6]);

                    w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                    w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                    temp = _mm256_fmadd_ps(s, w, temp);
                    temp2 = _mm256_fmadd_ps(s, w2, temp2);


                    // d+7 ------------------------------------------------------------
                    // |
                    // |
                    // v


                    // in_subscript += 1;

                    filter_subscript += Output_depth_dim;
                    

                    s = _mm256_broadcast_ss(&in_FP[in_subscript+7]);

                    w = _mm256_load_ps(&filter_FP_copy[filter_subscript]);
                    w2 = _mm256_load_ps(&filter_FP_copy[filter_subscript + 8]);


                    temp = _mm256_fmadd_ps(s, w, temp);
                    temp2 = _mm256_fmadd_ps(s, w2, temp2);
                  }
                }
              }

              unsigned long long int out_subscript = b_out +
                y_out +
                (x << out_l_shift)
                + m;
              const unsigned long long int out_subscript2 = out_subscript + 8;

              
              temp = _mm256_add_ps(temp, bias);
              temp = _mm256_max_ps(temp, _mm256_setzero_ps());
              _mm256_store_ps(&out_to_compare_with_FP[out_subscript], temp);

              temp2 = _mm256_add_ps(temp2, bias2);
              temp2 = _mm256_max_ps(temp2, _mm256_setzero_ps());
              _mm256_store_ps(&out_to_compare_with_FP[out_subscript2], temp2); // x+1
            }
          }
        }
      }
    }
  }
  #undef m_tile
  #undef left_shift
  printf("\n from AC optv16 m16 x4 d8 - tiled edge cases interchanged m and x %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  _mm_free(filter_FP_copy);
  return 0;
}


 


// finalised NON-array copying loops


// handling edge cases
// ~592 GFLOPS
int optimised_layer_v15_x4m2_edge_cases_AC(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  #define left_shift(x) (__builtin_ctz(x))  // inline function effectively using inline assembly to return num of trailing zeros (i.e. 128 returns 7, 2 * 128 == 2 << 7)
  #define RELU(x) ((x) < 0.0f ? 0.0f : (x)) // return x if greater than 0, else return 0

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

  unsigned int x;
  const unsigned int x_bound = (Output_X_dim/ 4) * 4; 

  #pragma omp parallel for private(x) shared(x_bound, in_yx_depth, in_x_depth, mask_yx_depth, mask_x_depth, out_yx_depth, out_x_depth) \
    default(shared) collapse(2) schedule(static)
  {
    for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
      const unsigned long long int b_in = b * in_yx_depth;
      const unsigned long long int b_out = b * out_yx_depth;

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        const unsigned long long int y_stride = y * Stride_Y_dim;
        const unsigned long long int y_out = y * out_x_depth;

        for (unsigned int m = 0; m < Output_depth_dim; m+=2) { //channels
          const unsigned long long int m_mask = m * mask_yx_depth;

          const float bias = bias_array_FP[m];
          const float bias2 = bias_array_FP[m+1];

          for (x = 0; x < x_bound; x+=4) {	//Output Width
            const unsigned long long int x_stride = x * Stride_X_dim;

            __m256 temp = _mm256_setzero_ps();
            __m256 temp2 = _mm256_setzero_ps();
            __m256 temp3 = _mm256_setzero_ps();
            __m256 temp4 = _mm256_setzero_ps();
            __m256 temp5 = _mm256_setzero_ps();
            __m256 temp6 = _mm256_setzero_ps();
            __m256 temp7 = _mm256_setzero_ps();
            __m256 temp8 = _mm256_setzero_ps();

            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              const unsigned long long int off_y_stride = (y_stride + off_y) * in_x_depth;
              const unsigned long long int off_y_mask = off_y * mask_x_depth;

              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                const unsigned long long int off_x_stride = (x_stride + off_x) << in_l_shift;
                const unsigned long long int off_x_depth = off_x << in_l_shift;

                for (unsigned int d = 0; d < Input_depth_dim; d+=32) {

                  // performing common subscript elim on subscripts result in perf increase in x4 for some reason
                  const unsigned long long int in_subscript = b_in
                    + off_y_stride
                    + off_x_stride
                    + d;

                  const unsigned long long int filter_subscript = m_mask
                    + off_y_mask
                    + off_x_depth
                    + d;

                  __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                  __m256 s2 = _mm256_load_ps(&in_FP[in_subscript + Input_depth_dim]);
                  __m256 s3 = _mm256_load_ps(&in_FP[in_subscript + (2 << in_l_shift)]);
                  __m256 s4 = _mm256_load_ps(&in_FP[in_subscript + (3 << in_l_shift)]);

                  __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]);
                  __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript + mask_yx_depth]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);

                  // d+8 ------------------------------------------------------------
                  // |
                  // |
                  // v

                  // in_subscript += 8;           // using this, or direct addition to subscript makes no/ little difference as it gets optimised out regardless

                  // filter_subscript += 8;

                  s = _mm256_load_ps(&in_FP[in_subscript + 8]);
                  s2 = _mm256_load_ps(&in_FP[in_subscript + 8 + Input_depth_dim]);
                  s3 = _mm256_load_ps(&in_FP[in_subscript + 8 + (2 << in_l_shift)]);
                  s4 = _mm256_load_ps(&in_FP[in_subscript + 8 + (3 << in_l_shift)]);

                  w = _mm256_load_ps(&filter_FP[filter_subscript + 8]);
                  w2 = _mm256_load_ps(&filter_FP[filter_subscript + 8 + mask_yx_depth]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);

                  // d+16 ------------------------------------------------------------
                  // |
                  // |
                  // v

                  // in_subscript += 8;

                  // filter_subscript += 8;

                  s = _mm256_load_ps(&in_FP[in_subscript + 16]);
                  s2 = _mm256_load_ps(&in_FP[in_subscript + 16 + Input_depth_dim]);
                  s3 = _mm256_load_ps(&in_FP[in_subscript + 16 + (2 << in_l_shift)]);
                  s4 = _mm256_load_ps(&in_FP[in_subscript + 16 + (3 << in_l_shift)]);

                  w = _mm256_load_ps(&filter_FP[filter_subscript + 16]);
                  w2 = _mm256_load_ps(&filter_FP[filter_subscript + 16 + mask_yx_depth]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);
                  temp5 = _mm256_fmadd_ps(s, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s2, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s3, w2, temp7);
                  temp8 = _mm256_fmadd_ps(s4, w2, temp8);

                  // d+24 ------------------------------------------------------------
                  // |
                  // |
                  // v

                  // in_subscript += 8;

                  // filter_subscript += 8;

                  s = _mm256_load_ps(&in_FP[in_subscript + 24]);
                  s2 = _mm256_load_ps(&in_FP[in_subscript + 24 + Input_depth_dim]);
                  s3 = _mm256_load_ps(&in_FP[in_subscript + 24 + (2 << in_l_shift)]);
                  s4 = _mm256_load_ps(&in_FP[in_subscript + 24 + (3 << in_l_shift)]);

                  w = _mm256_load_ps(&filter_FP[filter_subscript + 24]);
                  w2 = _mm256_load_ps(&filter_FP[filter_subscript + 24 + mask_yx_depth]);


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


            const unsigned long long int out_subscript = b_out 
              + y_out
              + (x << out_l_shift)
              + m;
            const unsigned long long int out_subscript2 = out_subscript + Output_depth_dim;
            const unsigned long long int out_subscript3 = out_subscript + (2 << out_l_shift);
            const unsigned long long int out_subscript4 = out_subscript + (3 << out_l_shift);


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

            sum3 += bias;
            out_to_compare_with_FP[out_subscript3] = RELU(sum3);


            temp4 = _mm256_hadd_ps(temp4, temp4);
            temp4 = _mm256_hadd_ps(temp4, temp4);
            tempLo2 = _mm256_castps256_ps128(temp4);
            tempHi2 = _mm256_extractf128_ps(temp4, 1);
            sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum4 = _mm_cvtss_f32(sseSum2);

            sum4 += bias;
            out_to_compare_with_FP[out_subscript4] = RELU(sum4);


            temp5 = _mm256_hadd_ps(temp5, temp5);
            temp5 = _mm256_hadd_ps(temp5, temp5);
            tempLo = _mm256_castps256_ps128(temp5);
            tempHi = _mm256_extractf128_ps(temp5, 1);
            sseSum = _mm_add_ps(tempLo, tempHi);

            float sum5 = _mm_cvtss_f32(sseSum);

            sum5 += bias2;
            out_to_compare_with_FP[out_subscript + 1] = RELU(sum5);


            temp6 = _mm256_hadd_ps(temp6, temp6);
            temp6 = _mm256_hadd_ps(temp6, temp6);
            tempLo2 = _mm256_castps256_ps128(temp6);
            tempHi2 = _mm256_extractf128_ps(temp6, 1);
            sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum6 = _mm_cvtss_f32(sseSum2);

            sum6 += bias2;
            out_to_compare_with_FP[out_subscript2 + 1] = RELU(sum6);


            temp7 = _mm256_hadd_ps(temp7, temp7);
            temp7 = _mm256_hadd_ps(temp7, temp7);
            tempLo = _mm256_castps256_ps128(temp7);
            tempHi = _mm256_extractf128_ps(temp7, 1);
            sseSum = _mm_add_ps(tempLo, tempHi);

            float sum7 = _mm_cvtss_f32(sseSum);

            sum7 += bias2;
            out_to_compare_with_FP[out_subscript3 + 1] = RELU(sum7);


            temp8 = _mm256_hadd_ps(temp8, temp8);
            temp8 = _mm256_hadd_ps(temp8, temp8);
            tempLo2 = _mm256_castps256_ps128(temp8);
            tempHi2 = _mm256_extractf128_ps(temp8, 1);
            sseSum2 = _mm_add_ps(tempLo2, tempHi2);

            float sum8 = _mm_cvtss_f32(sseSum2);

            sum8 += bias2;
            out_to_compare_with_FP[out_subscript4 + 1] = RELU(sum8);
          }
          for (; x < Output_depth_dim; x++) {
            const unsigned long long int x_stride = x * Stride_X_dim;

            __m256 temp = _mm256_setzero_ps();
            __m256 temp2 = _mm256_setzero_ps();

            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              const unsigned long long int off_y_stride = (y_stride + off_y) * in_x_depth;
              const unsigned long long int off_y_mask = off_y * mask_x_depth;

              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                const unsigned long long int off_x_stride = (x_stride + off_x) << in_l_shift;
                const unsigned long long int off_x_depth = off_x << in_l_shift;

                for (unsigned int d = 0; d < Input_depth_dim; d+=32) {

                  // performing common subscript elim on subscripts result in perf increase in x4 for some reason
                  unsigned long long int in_subscript = b_in
                    + off_y_stride
                    + off_x_stride
                    + d;

                  unsigned long long int filter_subscript = m_mask
                    + off_y_mask
                    + off_x_depth
                    + d;

                  __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                  __m256 s2 = _mm256_load_ps(&in_FP[in_subscript + Input_depth_dim]);

                  __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]);
                  __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript + mask_yx_depth]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s, w2, temp2);

                  // d+8 ------------------------------------------------------------
                  // |
                  // |
                  // v

                  // in_subscript += 8;           // using this, or direct addition to subscript makes no/ little difference as it gets optimised out regardless

                  // filter_subscript += 8;

                  s = _mm256_load_ps(&in_FP[in_subscript + 8]);
                  s2 = _mm256_load_ps(&in_FP[in_subscript + Input_depth_dim + 8]);

                  w = _mm256_load_ps(&filter_FP[filter_subscript + 8]);
                  w2 = _mm256_load_ps(&filter_FP[filter_subscript + mask_yx_depth + 8]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s, w2, temp2);

                  // d+16 ------------------------------------------------------------
                  // |
                  // |
                  // v

                  // in_subscript += 8;

                  // filter_subscript += 8;

                  s = _mm256_load_ps(&in_FP[in_subscript + 16]);
                  s2 = _mm256_load_ps(&in_FP[in_subscript + Input_depth_dim + 16]);

                  w = _mm256_load_ps(&filter_FP[filter_subscript + 16]);
                  w2 = _mm256_load_ps(&filter_FP[filter_subscript + mask_yx_depth + 16]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s, w2, temp2);

                  // d+24 ------------------------------------------------------------
                  // |
                  // |
                  // v

                  // in_subscript += 8;

                  // filter_subscript += 8;

                  s = _mm256_load_ps(&in_FP[in_subscript + 24]);

                  w = _mm256_load_ps(&filter_FP[filter_subscript + 24]);
                  w2 = _mm256_load_ps(&filter_FP[filter_subscript + mask_yx_depth + 24]);


                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s, w2, temp2);

                }
              }
            }


            unsigned long long int out_subscript = b_out
              + y_out
              + (x << out_l_shift)
              + m;
            unsigned long long int out_subscript2 = out_subscript + 1;



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
          }
        }
      }
    }
  }
  #undef left_shift
  #undef RELU
  printf("\n from optv15 x4 m2 - edge cases %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
  return 0;
}


// handled edge cases for x on m fallback loop
// ~623 GFLOPS
int optimised_layer_v15_x3m3_edge_cases_AC(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
  #define left_shift(x) (__builtin_ctz(x))  // inline function effectively using inline assembly to return num of trailing zeros (i.e. 128 returns 7, 2 * 128 == 2 << 7)
  #define RELU(x) ((x) < 0.0f ? 0.0f : (x)) // return x if greater than 0, else return 0
  
  // float bias, bias2, bias3;
  // __m256 temp, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9;

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
  #pragma omp parallel for private(m, x) shared(in_yx_depth, in_x_depth, mask_yx_depth, mask_x_depth, out_yx_depth, out_x_depth, m_bound, x_bound, x_bound2) \
    default(shared) collapse(2) schedule(static)
  {
    for (unsigned int b = 0; b < Input_Output_batch_dim; b++) { //batch
        const unsigned long long int b_in = b * in_yx_depth;
        const unsigned long long int b_out = b * out_yx_depth;

      for (unsigned int y = 0; y < Output_Y_dim; y++) {	//Output height
        const unsigned long long int y_stride = y * Stride_Y_dim;
        const unsigned long long int y_out = y * out_x_depth;

        for (m = 0; m < m_bound; m+=3) { //channels
          const unsigned long long int m_mask = m * mask_yx_depth;

          const float bias = bias_array_FP[m];
          const float bias2 = bias_array_FP[m+1];
          const float bias3 = bias_array_FP[m+2];

          for (x = 0; x < x_bound; x+=3) {	//Output Width
            const unsigned long long int x_stride = x * Stride_X_dim;

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
              const unsigned long long int off_y_stride = (y_stride + off_y) * in_x_depth;
              const unsigned long long int off_y_mask = off_y * mask_x_depth;

              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                const unsigned long long int off_x_stride = (x_stride + off_x) << in_l_shift;
                const unsigned long long int off_x_depth = off_x << in_l_shift;

                for (unsigned int d = 0; d < Input_depth_dim; d+=32) {

                  unsigned long long int in_subscript = b_in
                    + off_y_stride
                    + off_x_stride
                    + d;
                  // unsigned long long int in_subscript2 = b * (in_yx_depth)
                  //   + ((y * Stride_Y_dim + off_y) * in_x_depth)
                  //   + (((x+1) * Stride_X_dim + off_x) << in_l_shift)
                  //   + d;
                  // unsigned long long int in_subscript3 = b * (in_yx_depth)
                  //   + ((y * Stride_Y_dim + off_y) * in_x_depth)
                  //   + (((x+2) * Stride_X_dim + off_x) << in_l_shift)
                  //   + d;


                  unsigned long long int filter_subscript = m_mask
                    + off_y_mask
                    + off_x_depth
                    + d;
                  // unsigned long long int filter_subscript2 = ((m+1) * mask_yx_depth)
                  //   + (off_y * mask_x_depth)
                  //   + (off_x << in_l_shift)
                  //   + d;
                  // unsigned long long int filter_subscript3 = ((m+2) * mask_yx_depth)
                  //   + (off_y * mask_x_depth)
                  //   + (off_x << in_l_shift)
                  //   + d;

                  __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                  __m256 s2 = _mm256_load_ps(&in_FP[in_subscript + Input_depth_dim]);
                  __m256 s3 = _mm256_load_ps(&in_FP[in_subscript + (2 << in_l_shift)]);

                  __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]); 
                  __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript + mask_yx_depth]);
                  __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript + mask_yx_depth + mask_yx_depth]);

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s, w2, temp4);
                  temp5 = _mm256_fmadd_ps(s2, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s3, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s, w3, temp7);
                  temp8 = _mm256_fmadd_ps(s2, w3, temp8);
                  temp9 = _mm256_fmadd_ps(s3, w3, temp9);

                  // d+8 ------------------------------------------------------------
                  // |
                  // |
                  // v

                  // in_subscript += 8;

                  filter_subscript += 8;

                   s = _mm256_load_ps(&in_FP[in_subscript + 8]);
                   s2 = _mm256_load_ps(&in_FP[in_subscript + 8 + Input_depth_dim]);
                   s3 = _mm256_load_ps(&in_FP[in_subscript + 8 + (2 << in_l_shift)]);

                   w = _mm256_load_ps(&filter_FP[filter_subscript]); 
                   w2 = _mm256_load_ps(&filter_FP[filter_subscript + mask_yx_depth]);
                   w3 = _mm256_load_ps(&filter_FP[filter_subscript + mask_yx_depth + mask_yx_depth]);

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s, w2, temp4);
                  temp5 = _mm256_fmadd_ps(s2, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s3, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s, w3, temp7);
                  temp8 = _mm256_fmadd_ps(s2, w3, temp8);
                  temp9 = _mm256_fmadd_ps(s3, w3, temp9);

                  // d+16 ------------------------------------------------------------
                  // |
                  // |
                  // v

                  // in_subscript += 8;

                  filter_subscript += 8;

                   s = _mm256_load_ps(&in_FP[in_subscript + 16]);
                   s2 = _mm256_load_ps(&in_FP[in_subscript + 16 + Input_depth_dim]);
                   s3 = _mm256_load_ps(&in_FP[in_subscript + 16 + (2 << in_l_shift)]);

                   w = _mm256_load_ps(&filter_FP[filter_subscript]); 
                   w2 = _mm256_load_ps(&filter_FP[filter_subscript + mask_yx_depth]);
                   w3 = _mm256_load_ps(&filter_FP[filter_subscript + mask_yx_depth + mask_yx_depth]);

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s, w2, temp4);
                  temp5 = _mm256_fmadd_ps(s2, w2, temp5);
                  temp6 = _mm256_fmadd_ps(s3, w2, temp6);
                  temp7 = _mm256_fmadd_ps(s, w3, temp7);
                  temp8 = _mm256_fmadd_ps(s2, w3, temp8);
                  temp9 = _mm256_fmadd_ps(s3, w3, temp9);

                  // d+24 ------------------------------------------------------------
                  // |
                  // |
                  // v

                  // in_subscript += 8;

                  filter_subscript += 8;

                   s = _mm256_load_ps(&in_FP[in_subscript + 24]);
                   s2 = _mm256_load_ps(&in_FP[in_subscript + 24 + Input_depth_dim]);
                   s3 = _mm256_load_ps(&in_FP[in_subscript + 24 + (2 << in_l_shift)]);

                   w = _mm256_load_ps(&filter_FP[filter_subscript]); 
                   w2 = _mm256_load_ps(&filter_FP[filter_subscript + mask_yx_depth]);
                   w3 = _mm256_load_ps(&filter_FP[filter_subscript + mask_yx_depth + mask_yx_depth]);

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


            const unsigned long long int out_subscript = b_out +
              y_out +
              (x << out_l_shift)
              + m;
            const unsigned long long int out_subscript2 = out_subscript + Output_depth_dim;
            const unsigned long long int out_subscript3 = out_subscript + (2 << out_l_shift);
            const unsigned long long int out_subscript4 = out_subscript + 1;
            const unsigned long long int out_subscript5 = out_subscript + 1 + Output_depth_dim;
            const unsigned long long int out_subscript6 = out_subscript + 1 + (2 << out_l_shift);
            const unsigned long long int out_subscript7 = out_subscript + 2;
            const unsigned long long int out_subscript8 = out_subscript + 2 + Output_depth_dim;
            const unsigned long long int out_subscript9 = out_subscript + 2 + (2 << out_l_shift);

            // unsigned long long int out_subscript2 = b * (out_yx_depth) +
            //   y * (out_x_depth) +
            //   ((x+1) << out_l_shift)
            //   + m;
            // unsigned long long int out_subscript3 = b * (out_yx_depth) +
            //   y * (out_x_depth) +
            //   ((x+2) << out_l_shift)
            //   + m;
            // unsigned long long int out_subscript4 = b * (out_yx_depth) +
            //   y * (out_x_depth) +
            //   (x << out_l_shift)
            //   + (m+1);
            // unsigned long long int out_subscript5 = b * (out_yx_depth) +
            //   y * (out_x_depth) +
            //   ((x+1) << out_l_shift)
            //   + (m+1);
            // unsigned long long int out_subscript6 = b * (out_yx_depth) +
            //   y * (out_x_depth) +
            //   ((x+2) << out_l_shift)
            //   + (m+1);
            // unsigned long long int out_subscript7 = b * (out_yx_depth) +
            //   y * (out_x_depth) +
            //   (x << out_l_shift)
            //   + (m+2);
            // unsigned long long int out_subscript8 = b * (out_yx_depth) +
            //   y * (out_x_depth) +
            //   ((x+1) << out_l_shift)
            //   + (m+2);
            // unsigned long long int out_subscript9 = b * (out_yx_depth) +
            //   y * (out_x_depth) +
            //   ((x+2) << out_l_shift)
            //   + (m+2);


            __m128 sseLo = _mm256_castps256_ps128(temp);
						__m128 sseHi = _mm256_extractf128_ps(temp, 1);
						sseLo = _mm_add_ps(sseLo, sseHi);

						__m128 sseShuf = _mm_movehdup_ps(sseLo);
						__m128 sseSum = _mm_add_ps(sseLo, sseShuf);
						sseShuf = _mm_movehl_ps(sseShuf, sseSum);
						sseSum = _mm_add_ss(sseSum, sseShuf);
						float sum = _mm_cvtss_f32(sseSum);
						sum += bias;
						out_to_compare_with_FP[out_subscript] = RELU(sum);


             sseLo = _mm256_castps256_ps128(temp2);
						 sseHi = _mm256_extractf128_ps(temp2, 1);
						sseLo = _mm_add_ps(sseLo, sseHi);

						 sseShuf = _mm_movehdup_ps(sseLo);
						 sseSum = _mm_add_ps(sseLo, sseShuf);
						sseShuf = _mm_movehl_ps(sseShuf, sseSum);
						sseSum = _mm_add_ss(sseSum, sseShuf);
						 sum = _mm_cvtss_f32(sseSum);
						sum += bias;
						out_to_compare_with_FP[out_subscript2] = RELU(sum);


             sseLo = _mm256_castps256_ps128(temp3);
						 sseHi = _mm256_extractf128_ps(temp3, 1);
						sseLo = _mm_add_ps(sseLo, sseHi);

						 sseShuf = _mm_movehdup_ps(sseLo);
						 sseSum = _mm_add_ps(sseLo, sseShuf);
						sseShuf = _mm_movehl_ps(sseShuf, sseSum);
						sseSum = _mm_add_ss(sseSum, sseShuf);
						 sum = _mm_cvtss_f32(sseSum);
						sum += bias;
						out_to_compare_with_FP[out_subscript3] = RELU(sum);


             sseLo = _mm256_castps256_ps128(temp4);
						 sseHi = _mm256_extractf128_ps(temp4, 1);
						sseLo = _mm_add_ps(sseLo, sseHi);

						 sseShuf = _mm_movehdup_ps(sseLo);
						 sseSum = _mm_add_ps(sseLo, sseShuf);
						sseShuf = _mm_movehl_ps(sseShuf, sseSum);
						sseSum = _mm_add_ss(sseSum, sseShuf);
						 sum = _mm_cvtss_f32(sseSum);
						sum += bias2;
						out_to_compare_with_FP[out_subscript4] = RELU(sum);


             sseLo = _mm256_castps256_ps128(temp5);
						 sseHi = _mm256_extractf128_ps(temp5, 1);
						sseLo = _mm_add_ps(sseLo, sseHi);

						 sseShuf = _mm_movehdup_ps(sseLo);
						 sseSum = _mm_add_ps(sseLo, sseShuf);
						sseShuf = _mm_movehl_ps(sseShuf, sseSum);
						sseSum = _mm_add_ss(sseSum, sseShuf);
						 sum = _mm_cvtss_f32(sseSum);
						sum += bias2;
						out_to_compare_with_FP[out_subscript5] = RELU(sum);


             sseLo = _mm256_castps256_ps128(temp6);
						 sseHi = _mm256_extractf128_ps(temp6, 1);
						sseLo = _mm_add_ps(sseLo, sseHi);

						 sseShuf = _mm_movehdup_ps(sseLo);
						 sseSum = _mm_add_ps(sseLo, sseShuf);
						sseShuf = _mm_movehl_ps(sseShuf, sseSum);
						sseSum = _mm_add_ss(sseSum, sseShuf);
						 sum = _mm_cvtss_f32(sseSum);
						sum += bias2;
						out_to_compare_with_FP[out_subscript6] = RELU(sum);


             sseLo = _mm256_castps256_ps128(temp7);
						 sseHi = _mm256_extractf128_ps(temp7, 1);
						sseLo = _mm_add_ps(sseLo, sseHi);

						 sseShuf = _mm_movehdup_ps(sseLo);
						 sseSum = _mm_add_ps(sseLo, sseShuf);
						sseShuf = _mm_movehl_ps(sseShuf, sseSum);
						sseSum = _mm_add_ss(sseSum, sseShuf);
						 sum = _mm_cvtss_f32(sseSum);
						sum += bias3;
						out_to_compare_with_FP[out_subscript7] = RELU(sum);


             sseLo = _mm256_castps256_ps128(temp8);
						 sseHi = _mm256_extractf128_ps(temp8, 1);
						sseLo = _mm_add_ps(sseLo, sseHi);

						 sseShuf = _mm_movehdup_ps(sseLo);
						 sseSum = _mm_add_ps(sseLo, sseShuf);
						sseShuf = _mm_movehl_ps(sseShuf, sseSum);
						sseSum = _mm_add_ss(sseSum, sseShuf);
						 sum = _mm_cvtss_f32(sseSum);
						sum += bias3;
						out_to_compare_with_FP[out_subscript8] = RELU(sum);


             sseLo = _mm256_castps256_ps128(temp9);
						 sseHi = _mm256_extractf128_ps(temp9, 1);
						sseLo = _mm_add_ps(sseLo, sseHi);

						 sseShuf = _mm_movehdup_ps(sseLo);
						 sseSum = _mm_add_ps(sseLo, sseShuf);
						sseShuf = _mm_movehl_ps(sseShuf, sseSum);
						sseSum = _mm_add_ss(sseSum, sseShuf);
						 sum = _mm_cvtss_f32(sseSum);
						sum += bias3;
						out_to_compare_with_FP[out_subscript9] = RELU(sum);
          }
          // overflow/ fallback x loop
          for (; x < Output_X_dim; x++) {	//Output Width
            const unsigned long long int x_stride = x * Stride_X_dim;
            __m256 temp = _mm256_setzero_ps();
            __m256 temp2 = _mm256_setzero_ps();
            __m256 temp3 = _mm256_setzero_ps();
            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              const unsigned long long int off_y_stride = (y_stride + off_y) * in_x_depth;
              const unsigned long long int off_y_mask = off_y * mask_x_depth;

              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                const unsigned long long int off_x_stride = (x_stride + off_x) << in_l_shift;
                const unsigned long long int off_x_depth = off_x << in_l_shift;

                for (unsigned int d = 0; d < Input_depth_dim; d+=32) {

                  unsigned long long int in_subscript = b_in
                    + off_y_stride
                    + off_x_stride
                    + d;

                  unsigned long long int filter_subscript = m_mask
                    + off_y_mask
                    + off_x_depth
                    + d;
                  // unsigned long long int filter_subscript2 = ((m+1) * mask_yx_depth)
                  //   + (off_y * mask_x_depth)
                  //   + (off_x << in_l_shift)
                  //   + d;
                  // unsigned long long int filter_subscript3 = ((m+2) * mask_yx_depth)
                  //   + (off_y * mask_x_depth)
                  //   + (off_x << in_l_shift)
                  //   + d;

                  __m256 s = _mm256_load_ps(&in_FP[in_subscript]);

                  __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]);
                  __m256 w2 = _mm256_load_ps(&filter_FP[filter_subscript + mask_yx_depth]);
                  __m256 w3 = _mm256_load_ps(&filter_FP[filter_subscript + mask_yx_depth + mask_yx_depth]);

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s, w2, temp2);
                  temp3 = _mm256_fmadd_ps(s, w3, temp3);

                  // d+8 ------------------------------------------------------------
                  // |
                  // |
                  // v

                  // in_subscript += 8;

                  // filter_subscript += 8;

                   s = _mm256_load_ps(&in_FP[in_subscript + 8]);

                   w = _mm256_load_ps(&filter_FP[filter_subscript + 8]);
                   w2 = _mm256_load_ps(&filter_FP[filter_subscript + 8 + mask_yx_depth]);
                   w3 = _mm256_load_ps(&filter_FP[filter_subscript + 8 + mask_yx_depth + mask_yx_depth]);

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s, w2, temp2);
                  temp3 = _mm256_fmadd_ps(s, w3, temp3);

                  // d+16 ------------------------------------------------------------
                  // |
                  // |
                  // v

                  // in_subscript += 8;

                  // filter_subscript += 8;

                   s = _mm256_load_ps(&in_FP[in_subscript + 16]);

                   w = _mm256_load_ps(&filter_FP[filter_subscript + 16]);
                   w2 = _mm256_load_ps(&filter_FP[filter_subscript + 16 + mask_yx_depth]);
                   w3 = _mm256_load_ps(&filter_FP[filter_subscript + 16 + mask_yx_depth + mask_yx_depth]);

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s, w2, temp2);
                  temp3 = _mm256_fmadd_ps(s, w3, temp3);

                  // d+24 ------------------------------------------------------------
                  // |
                  // |
                  // v

                  // in_subscript += 8;

                  // filter_subscript += 8;

                   s = _mm256_load_ps(&in_FP[in_subscript + 24]);

                   w = _mm256_load_ps(&filter_FP[filter_subscript + 24]);
                   w2 = _mm256_load_ps(&filter_FP[filter_subscript + 24 + mask_yx_depth]);
                   w3 = _mm256_load_ps(&filter_FP[filter_subscript + 24 + mask_yx_depth + mask_yx_depth]);

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s, w2, temp2);
                  temp3 = _mm256_fmadd_ps(s, w3, temp3);
                }
              }
            }
            unsigned long long int out_subscript = b_out +
              y_out +
              (x << out_l_shift)
              + m;
            const unsigned long long int out_subscript2 = out_subscript + 1;
            const unsigned long long int out_subscript3 = out_subscript + 2;
            // unsigned long long int out_subscript2 = b * (out_yx_depth) +
            //   y * (out_x_depth) +
            //   (x << out_l_shift)
            //   + (m+1);
            // unsigned long long int out_subscript3 = b * (out_yx_depth) +
            //   y * (out_x_depth) +
            //   (x << out_l_shift)
            //   + (m+2);



            __m128 sseLo = _mm256_castps256_ps128(temp);
						__m128 sseHi = _mm256_extractf128_ps(temp, 1);
						sseLo = _mm_add_ps(sseLo, sseHi);

						__m128 sseShuf = _mm_movehdup_ps(sseLo);
						__m128 sseSum = _mm_add_ps(sseLo, sseShuf);
						sseShuf = _mm_movehl_ps(sseShuf, sseSum);
						sseSum = _mm_add_ss(sseSum, sseShuf);
						float sum = _mm_cvtss_f32(sseSum);
						sum += bias;
						out_to_compare_with_FP[out_subscript] = RELU(sum);


             sseLo = _mm256_castps256_ps128(temp2);
						 sseHi = _mm256_extractf128_ps(temp2, 1);
						sseLo = _mm_add_ps(sseLo, sseHi);

						 sseShuf = _mm_movehdup_ps(sseLo);
						 sseSum = _mm_add_ps(sseLo, sseShuf);
						sseShuf = _mm_movehl_ps(sseShuf, sseSum);
						sseSum = _mm_add_ss(sseSum, sseShuf);
						 sum = _mm_cvtss_f32(sseSum);
						sum += bias2;
						out_to_compare_with_FP[out_subscript2] = RELU(sum);


             sseLo = _mm256_castps256_ps128(temp3);
						 sseHi = _mm256_extractf128_ps(temp3, 1);
						sseLo = _mm_add_ps(sseLo, sseHi);

						 sseShuf = _mm_movehdup_ps(sseLo);
						 sseSum = _mm_add_ps(sseLo, sseShuf);
						sseShuf = _mm_movehl_ps(sseShuf, sseSum);
						sseSum = _mm_add_ss(sseSum, sseShuf);
						 sum = _mm_cvtss_f32(sseSum);
						sum += bias3;
						out_to_compare_with_FP[out_subscript3] = RELU(sum);
          }
        }
        for (; m < Output_depth_dim; m++) {
          const unsigned long long int m_mask = m * mask_yx_depth;

          const float bias = bias_array_FP[m];

          for (x = 0; x < x_bound2; x+=4) {	//Output Width
            const unsigned long long int x_stride = x * Stride_X_dim;

            __m256 temp = _mm256_setzero_ps();
            __m256 temp2 = _mm256_setzero_ps();
            __m256 temp3 = _mm256_setzero_ps();
            __m256 temp4 = _mm256_setzero_ps();

            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              const unsigned long long int off_y_stride = (y_stride + off_y) * in_x_depth;
              const unsigned long long int off_y_mask = off_y * mask_x_depth;

              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                const unsigned long long int off_x_stride = (x_stride + off_x) << in_l_shift;
                const unsigned long long int off_x_depth = off_x << in_l_shift;

                for (unsigned int d = 0; d < Input_depth_dim; d+=32) {

                  unsigned long long int in_subscript = b_in
                    + off_y_stride
                    + off_x_stride
                    + d;
                  // unsigned long long int in_subscript2 = b * (in_yx_depth)
                  //   + ((y * Stride_Y_dim + off_y) * in_x_depth)
                  //   + (((x+1) * Stride_X_dim + off_x) << in_l_shift)
                  //   + d;
                  // unsigned long long int in_subscript3 = b * (in_yx_depth)
                  //   + ((y * Stride_Y_dim + off_y) * in_x_depth)
                  //   + (((x+2) * Stride_X_dim + off_x) << in_l_shift)
                  //   + d;
                  // unsigned long long int in_subscript4 = b * (in_yx_depth)
                  //   + ((y * Stride_Y_dim + off_y) * in_x_depth)
                  //   + (((x+3) * Stride_X_dim + off_x) << in_l_shift)
                  //   + d;

                  unsigned long long int filter_subscript = m_mask
                    + off_y_mask
                    + off_x_depth
                    + d;

                  __m256 s = _mm256_load_ps(&in_FP[in_subscript]);
                  __m256 s2 = _mm256_load_ps(&in_FP[in_subscript + Input_depth_dim]);
                  __m256 s3 = _mm256_load_ps(&in_FP[in_subscript + (2 << in_l_shift)]);
                  __m256 s4 = _mm256_load_ps(&in_FP[in_subscript + (3 << in_l_shift)]);

                  __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]);

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);

                  // d+8 ------------------------------------------------------------
                  // |
                  // |
                  // v

                  // in_subscript += 8;

                  // filter_subscript += 8;

                   s = _mm256_load_ps(&in_FP[in_subscript + 8]);
                   s2 = _mm256_load_ps(&in_FP[in_subscript + 8 + Input_depth_dim]);
                   s3 = _mm256_load_ps(&in_FP[in_subscript + 8 + (2 << in_l_shift)]);
                   s4 = _mm256_load_ps(&in_FP[in_subscript + 8 + (3 << in_l_shift)]);

                   w = _mm256_load_ps(&filter_FP[filter_subscript + 8]);

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);

                  // d+8 ------------------------------------------------------------
                  // |
                  // |
                  // v

                  // in_subscript += 8;

                  // filter_subscript += 8;

                   s = _mm256_load_ps(&in_FP[in_subscript + 168]);
                   s2 = _mm256_load_ps(&in_FP[in_subscript + 16 + Input_depth_dim]);
                   s3 = _mm256_load_ps(&in_FP[in_subscript + 16 + (2 << in_l_shift)]);
                   s4 = _mm256_load_ps(&in_FP[in_subscript + 16 + (3 << in_l_shift)]);

                   w = _mm256_load_ps(&filter_FP[filter_subscript + 16]);

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);

                  // d+8 ------------------------------------------------------------
                  // |
                  // |
                  // v

                  // in_subscript += 8;

                  // filter_subscript += 8;

                   s = _mm256_load_ps(&in_FP[in_subscript + 24]);
                   s2 = _mm256_load_ps(&in_FP[in_subscript + 24 + Input_depth_dim]);
                   s3 = _mm256_load_ps(&in_FP[in_subscript + 24 + (2 << in_l_shift)]);
                   s4 = _mm256_load_ps(&in_FP[in_subscript + 24 + (3 << in_l_shift)]);

                   w = _mm256_load_ps(&filter_FP[filter_subscript + 24]);

                  temp = _mm256_fmadd_ps(s, w, temp);
                  temp2 = _mm256_fmadd_ps(s2, w, temp2);
                  temp3 = _mm256_fmadd_ps(s3, w, temp3);
                  temp4 = _mm256_fmadd_ps(s4, w, temp4);

                }
              }
            }

            const unsigned long long int out_subscript = b_out +
              y_out +
              (x << out_l_shift)
              + m;
            const unsigned long long int out_subscript2 = out_subscript + Output_depth_dim;
            const unsigned long long int out_subscript3 = out_subscript + (2 << out_l_shift);
            const unsigned long long int out_subscript4 = out_subscript + (3 << out_l_shift);
            // unsigned long long int out_subscript2 = b * (out_yx_depth) +
            //   y * (out_x_depth) +
            //   ((x+1) << out_l_shift)
            //   + m;
            // unsigned long long int out_subscript3 = b * (out_yx_depth) +
            //   y * (out_x_depth) +
            //   ((x+2) << out_l_shift)
            //   + m;
            // unsigned long long int out_subscript4 = b * (out_yx_depth) +
            //   y * (out_x_depth) +
            //   ((x+3) << out_l_shift)
            //   + m;



            __m128 sseLo = _mm256_castps256_ps128(temp);
						__m128 sseHi = _mm256_extractf128_ps(temp, 1);
						sseLo = _mm_add_ps(sseLo, sseHi);

						__m128 sseShuf = _mm_movehdup_ps(sseLo);
						__m128 sseSum = _mm_add_ps(sseLo, sseShuf);
						sseShuf = _mm_movehl_ps(sseShuf, sseSum);
						sseSum = _mm_add_ss(sseSum, sseShuf);
						float sum = _mm_cvtss_f32(sseSum);
						sum += bias;
						out_to_compare_with_FP[out_subscript] = RELU(sum);


             sseLo = _mm256_castps256_ps128(temp2);
						 sseHi = _mm256_extractf128_ps(temp2, 1);
						sseLo = _mm_add_ps(sseLo, sseHi);

						 sseShuf = _mm_movehdup_ps(sseLo);
						 sseSum = _mm_add_ps(sseLo, sseShuf);
						sseShuf = _mm_movehl_ps(sseShuf, sseSum);
						sseSum = _mm_add_ss(sseSum, sseShuf);
						 sum = _mm_cvtss_f32(sseSum);
						sum += bias;
						out_to_compare_with_FP[out_subscript2] = RELU(sum);


             sseLo = _mm256_castps256_ps128(temp3);
						 sseHi = _mm256_extractf128_ps(temp3, 1);
						sseLo = _mm_add_ps(sseLo, sseHi);

						 sseShuf = _mm_movehdup_ps(sseLo);
						 sseSum = _mm_add_ps(sseLo, sseShuf);
						sseShuf = _mm_movehl_ps(sseShuf, sseSum);
						sseSum = _mm_add_ss(sseSum, sseShuf);
						 sum = _mm_cvtss_f32(sseSum);
						sum += bias;
						out_to_compare_with_FP[out_subscript3] = RELU(sum);


             sseLo = _mm256_castps256_ps128(temp4);
						 sseHi = _mm256_extractf128_ps(temp4, 1);
						sseLo = _mm_add_ps(sseLo, sseHi);

						 sseShuf = _mm_movehdup_ps(sseLo);
						 sseSum = _mm_add_ps(sseLo, sseShuf);
						sseShuf = _mm_movehl_ps(sseShuf, sseSum);
						sseSum = _mm_add_ss(sseSum, sseShuf);
						 sum = _mm_cvtss_f32(sseSum);
						sum += bias;
						out_to_compare_with_FP[out_subscript4] = RELU(sum);
          }
          for (; x < Output_X_dim; x++) {	//Output Width
            const unsigned long long int x_stride = x * Stride_X_dim;
            __m256 temp = _mm256_setzero_ps();

            for (unsigned int off_y = 0; off_y < Mask_Y_dim; off_y++) {
              const unsigned long long int off_y_stride = (y_stride + off_y) * in_x_depth;
              const unsigned long long int off_y_mask = off_y * mask_x_depth;

              for (unsigned int off_x = 0; off_x < Mask_X_dim; off_x++) {
                const unsigned long long int off_x_stride = (x_stride + off_x) << in_l_shift;
                const unsigned long long int off_x_depth = off_x << in_l_shift;

                for (unsigned int d = 0; d < Input_depth_dim; d+=32) {

                  unsigned long long int in_subscript = b_in
                    + off_y_stride
                    + off_x_stride
                    + d;

                  unsigned long long int filter_subscript = m_mask
                    + off_y_mask
                    + off_x_depth
                    + d;

                  __m256 s = _mm256_load_ps(&in_FP[in_subscript]);

                  __m256 w = _mm256_load_ps(&filter_FP[filter_subscript]);

                  temp = _mm256_fmadd_ps(s, w, temp);

                  // d+8 ------------------------------------------------------------
                  // |
                  // |
                  // v

                  // in_subscript += 8;

                  // filter_subscript += 8;

                   s = _mm256_load_ps(&in_FP[in_subscript + 8]);

                   w = _mm256_load_ps(&filter_FP[filter_subscript + 8]);

                  temp = _mm256_fmadd_ps(s, w, temp);

                  // d+16 ------------------------------------------------------------
                  // |
                  // |
                  // v

                  // in_subscript += 8;

                  // filter_subscript += 8;

                   s = _mm256_load_ps(&in_FP[in_subscript + 16]);

                   w = _mm256_load_ps(&filter_FP[filter_subscript + 16]);

                  temp = _mm256_fmadd_ps(s, w, temp);


                  // d+24 ------------------------------------------------------------
                  // |
                  // |
                  // v

                  // in_subscript += 8;

                  // filter_subscript += 8;

                   s = _mm256_load_ps(&in_FP[in_subscript + 24]);

                   w = _mm256_load_ps(&filter_FP[filter_subscript + 24]);

                  temp = _mm256_fmadd_ps(s, w, temp);

                }
              }
            }
            unsigned long long int out_subscript = b_out +
              y_out +
              (x << out_l_shift)
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
						out_to_compare_with_FP[out_subscript] = RELU(sum);

          }
        }
      }
    }
  }
  #undef left_shift
  #undef RELU
  printf("\n from optv14 x3 m3 - cleaned code %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
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