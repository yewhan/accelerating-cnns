#include "convolution_layer_2D.h"

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))


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