/*
------------------DR VASILIOS KELEFOURAS-----------------------------------------------------
------------------UNIVERSITY OF PLYMOUTH, SCHOOL OF ENGINEERING, COMPUTING AND MATHEMATICS---

Altered by Euan Hughes

*/

/*
This is a software application that executes the convolution and ReLU layer in a Deep Neural Network.
The layer input parameters are specified in  void read_layer_dimensions().
*/

#include <mm_malloc.h>

#include "convolution_layer_2D.h"
#include <limits.h>
#include <stdint.h>



// #define QUANTISATION        // **************** COMMENT OUT TO DISABLE QUANTISATION ****************



void read_layer_dimensions();

int load_create_input_output_array_FP();
int load_filter_array_FP();
void load_bias_FP();
void deallocate_FP();
void compare_output_result_FP();
unsigned short int equal_FP(float const a, float const b);

void compare_output_result_Char();
unsigned short int equal_Char(unsigned char const a, unsigned char const b);
void deallocate_Char();



//input dimensions
unsigned int Input_Output_batch_dim;
unsigned int Input_X_dim;
unsigned int Input_Y_dim;
unsigned int Input_depth_dim;

unsigned int Stride_X_dim;
unsigned int Stride_Y_dim;
//unsigned int Stride_Z_dim;

//output dimensions
unsigned int Output_X_dim;
unsigned int Output_Y_dim;
unsigned int Output_depth_dim;
//output batch == input batch

//mask dimensions
unsigned int Mask_X_dim;
unsigned int Mask_Y_dim;
//unsigned int Mask_Z_dim;




float Scale;
unsigned int M0_by_n;
unsigned char Zero_point;
__m256i M0_by_n_vector;
__m256 Scale_vector;


float* in_FP; //pointer to input array
float* in_layout_FP; //pointer to input array
float* filter_FP; //pointer to filter array
float* out_FP; //pointer to output array
float* out_to_compare_with_FP; //pointer to output array to compare with
float* bias_array_FP;



// quantised tensors
unsigned char* in_Char;                   // pointer to input array - char
unsigned char* out_Char;                  // pointer to output array - char
unsigned char* out_to_compare_with_Char;  // pointer to output array to compare with - char
signed char* filter_Char;                        // pointer to filter array - char
int* bias_array_Int;                      // pointer to bias array - int
// int_fast32_t* bias_array_Int;             // pointer to bias array - fast int of 32 bits



#define EPSILON 0.001



int main() {

  double start_time, run_time;
  int i = 0;

  read_layer_dimensions();



  load_bias_FP();
  load_create_input_output_array_FP();
  load_filter_array_FP();


  #ifndef QUANTISATION
    // Dr. Kelefouras' unoptimized layer taken as a base:
    unoptimized_layer_FP(in_FP, filter_FP, bias_array_FP, out_to_compare_with_FP); //to compare
    // optimised_layer_v6_AC_register_pressure_x_FP(in_FP, filter_FP, bias_array_FP, out_to_compare_with_FP);
    // optimised_layer_v8_AC_loop_tiling_m_FP(in_FP, filter_FP, bias_array_FP, out_to_compare_with_FP);
    // optimised_layer_v14_AC_omp_2blocks_FP(in_FP, filter_FP, bias_array_FP, out_to_compare_with_FP);
    // optimised_layer_v15_AC_omp_1block_FP(in_FP, filter_FP, bias_array_FP, out_to_compare_with_FP);


  #else
    // quantised functions
    // unoptimized_layer_Char(in_Char, filter_Char, bias_array_Int, out_to_compare_with_Char);
    optimised_layerv1_vectorised_Char(in_Char, filter_Char, bias_array_Int, out_to_compare_with_Char);

  #endif



  start_time = omp_get_wtime();

  for (int i = 0; i < 10; i++)
  {

  #ifndef QUANTISATION
    // unoptimized_layer_FP(in_FP, filter_FP, bias_array_FP, out_FP);

    // ***** vectorised d loop *****
    // optimised_layer_v1_vectorised_FP(in_FP, filter_FP, bias_array_FP, out_FP);
    // optimised_layer_v1_vectorised_opt_FP(in_FP, filter_FP, bias_array_FP, out_FP);

    // optimised_layer_v2_unroll_x2_FP(in_FP, filter_FP, bias_array_FP, out_FP);
    // optimised_layer_v2_unroll_x4_FP(in_FP, filter_FP, bias_array_FP, out_FP);
    // optimised_layer_v2_unroll_x4_hadd_FP(in_FP, filter_FP, bias_array_FP, out_FP);
    // optimised_layer_v2_unroll_x8_hadd_FP(in_FP, filter_FP, bias_array_FP, out_FP);
    // optimised_layer_v2_unroll_m2_FP(in_FP, filter_FP, bias_array_FP, out_FP);
    // optimised_layer_v2_unroll_m4_FP(in_FP, filter_FP, bias_array_FP, out_FP);
    // optimised_layer_v2_unroll_m4_hadd_FP(in_FP, filter_FP, bias_array_FP, out_FP);
    // optimised_layer_v2_unroll_m8_hadd_FP(in_FP, filter_FP, bias_array_FP, out_FP);
    // optimised_layer_v2_unroll_x2m2_FP(in_FP, filter_FP, bias_array_FP, out_FP);
    // optimised_layer_v2_unroll_x2m2_hadd_FP(in_FP, filter_FP, bias_array_FP, out_FP);
    // optimised_layer_v2_unroll_x4m2_FP(in_FP, filter_FP, bias_array_FP, out_FP);
    // optimised_layer_v2_unroll_x4m2_hadd_FP(in_FP, filter_FP, bias_array_FP, out_FP);
    // optimised_layer_v2_unroll_x2m4_hadd_FP(in_FP, filter_FP, bias_array_FP, out_FP);
    // optimised_layer_v2_unroll_x4m2_hadd_register_pressure_FP(in_FP, filter_FP, bias_array_FP, out_FP);
    // optimised_layer_v2_unroll_x2m4_hadd_register_pressure_FP(in_FP, filter_FP, bias_array_FP, out_FP);
    // optimised_layer_v2_unroll_x3m3_hadd_FP(in_FP, filter_FP, bias_array_FP, out_FP);
    // optimised_layer_v2_unroll_x3m3_hadd_opt_FP(in_FP, filter_FP, bias_array_FP, out_FP);

    // optimised_layer_v3_x3m3_unroll_d16_FP(in_FP, filter_FP, bias_array_FP, out_FP);
    // optimised_layer_v3_x3m3_unroll_d16_v2_FP(in_FP, filter_FP, bias_array_FP, out_FP);
    // optimised_layer_v3_x2m4_unroll_d16_FP(in_FP, filter_FP, bias_array_FP, out_FP);
    // optimised_layer_v3_x4m2_unroll_d16_FP(in_FP, filter_FP, bias_array_FP, out_FP);
    // optimised_layer_v3_x2m2_unroll_d16_FP(in_FP, filter_FP, bias_array_FP, out_FP);
    // optimised_layer_v3_x2m2_unroll_d16_opt_FP(in_FP, filter_FP, bias_array_FP, out_FP);
    // optimised_layer_v3_x3m3_unroll_d32_FP(in_FP, filter_FP, bias_array_FP, out_FP);
    // optimised_layer_v3_x3m3_unroll_d32_v2_FP(in_FP, filter_FP, bias_array_FP, out_FP);
    // optimised_layer_v3_x2m4_unroll_d32_FP(in_FP, filter_FP, bias_array_FP, out_FP);
    // optimised_layer_v3_x4m2_unroll_d32_FP(in_FP, filter_FP, bias_array_FP, out_FP);
    // optimised_layer_v3_x2m2_unroll_d32_FP(in_FP, filter_FP, bias_array_FP, out_FP);
    // optimised_layer_v3_x2m2_unroll_d32_opt_FP(in_FP, filter_FP, bias_array_FP, out_FP);

    // optimised_layer_v4_x3m3_tiled_y_FP(in_FP, filter_FP, bias_array_FP, out_FP);
    // optimised_layer_v4_x3m3_tiled_x_FP(in_FP, filter_FP, bias_array_FP, out_FP);
    // optimised_layer_v4_x3m3_tiled_d_FP(in_FP, filter_FP, bias_array_FP, out_FP);
    // optimised_layer_v4_x2m4_tiled_x_FP(in_FP, filter_FP, bias_array_FP, out_FP);
    optimised_layer_v4_x2m4_tiled_m_FP(in_FP, filter_FP, bias_array_FP, out_FP);
    // optimised_layer_v4_x2m4_tiled_m_moved_m_FP(in_FP, filter_FP, bias_array_FP, out_FP);



    // ***** vectorised m loop, AKA array copying functions *****
    // optimised_layer_v1_AC_vectorised_FP(in_FP, filter_FP, bias_array_FP, out_FP);
    // optimised_layer_v2_AC_unroll_x2_FP(in_FP, filter_FP, bias_array_FP, out_FP);
    // optimised_layer_v3_AC_unroll_x4_FP(in_FP, filter_FP, bias_array_FP, out_FP);
    // optimised_layer_v4_AC_unroll_m16_FP(in_FP, filter_FP, bias_array_FP, out_FP);
    // optimised_layer_v5_AC_register_pressure_d_FP(in_FP, filter_FP, bias_array_FP, out_FP);
    // optimised_layer_v6_AC_register_pressure_x_FP(in_FP, filter_FP, bias_array_FP, out_FP);
    // optimised_layer_v7_AC_strength_reduction_d_FP(in_FP, filter_FP, bias_array_FP, out_FP);
    // optimised_layer_v8_AC_loop_tiling_m_FP(in_FP, filter_FP, bias_array_FP, out_FP);
    // optimised_layer_v9_AC_unroll_d2_FP(in_FP, filter_FP, bias_array_FP, out_FP);
    // optimised_layer_v10_AC_unroll_d4_FP(in_FP, filter_FP, bias_array_FP, out_FP);
    // optimised_layer_v11_AC_unroll_d8_FP(in_FP, filter_FP, bias_array_FP, out_FP);
    // optimised_layer_v12_AC_ops_outside_loop_FP(in_FP, filter_FP, bias_array_FP, out_FP);
    // optimised_layer_v13_AC_sign_unsigned_FP(in_FP, filter_FP, bias_array_FP, out_FP);
    // optimised_layer_v14_AC_omp_2blocks_FP(in_FP, filter_FP, bias_array_FP, out_FP);
    // optimised_layer_v15_AC_omp_1block_FP(in_FP, filter_FP, bias_array_FP, out_FP);
  
  
  #else

    // ***** quantised functions *****
    // unoptimized_layer_Char(in_Char, filter_Char, bias_array_Int, out_Char);
    // optimised_layerv1_vectorised_Char(in_Char, filter_Char, bias_array_Int, out_Char);
    optimised_layerv1_arraycopying_vectorised_Char(in_Char, filter_Char, bias_array_Int, out_Char);
    // optimised_layerv2_unroll_x2_Char(in_Char, filter_Char, bias_array_Int, out_Char);
    // optimised_layerv3_unroll_m2_Char(in_Char, filter_Char, bias_array_Int, out_Char);
    // optimised_layerv4_general_register_pressure_d_Char(in_Char, filter_Char, bias_array_Int, out_Char);

  #endif

  }

  run_time = (omp_get_wtime() - start_time);

  double FLOPS = (double)Input_Output_batch_dim * Output_Y_dim * Output_X_dim * Output_depth_dim;
  FLOPS = (FLOPS * ((double)2 * Mask_Y_dim * Mask_X_dim * Input_depth_dim + 1)) / (run_time/10);

  printf("\n\nTime = %.3e seconds", run_time);
  printf(" or %.0f mseconds", run_time * 1000);//printf time in msecs
  printf("\nGiga FLOPS achieved: %.0f\n", (double)FLOPS / 1000000000);//print Giga FLOPS


  #ifndef QUANTISATION
  compare_output_result_FP();

  deallocate_FP();

  #else
  compare_output_result_Char();
  deallocate_Char();
  #endif

  return 0;
}


void compare_output_result_FP() {

  for (unsigned long long int i = 0; i < (unsigned long long int) Input_Output_batch_dim * Output_Y_dim * Output_X_dim * Output_depth_dim; i++) {
    if (equal_FP(out_FP[i], out_to_compare_with_FP[i]) == 1) {
      printf("\n wrong values (%llu): %f %f", i, out_FP[i], out_to_compare_with_FP[i]);
      int c = getchar();
    }
  }
}

unsigned short int equal_FP(float const a, float const b) {
  float temp = a - b;

  if (b == 0.0f) {//cannot divide with zero
    if (a == 0.0f) {
      return 0;//success
    }
    else {
      return 1;
    }
  }
  else {

    if ((fabs(temp) / fabs(b)) < EPSILON) {
      return 0; //success
    }
    else {
      return 1;
    }
  }
}


void compare_output_result_Char() {
  for (unsigned long long int i = 0; i < (unsigned long long int) Input_Output_batch_dim * Output_Y_dim * Output_X_dim * Output_depth_dim; i++) {
    if (equal_Char(out_Char[i], out_to_compare_with_Char[i]) == 1) {
      printf("\n wrong values (%llu): %d %d", i, out_Char[i], out_to_compare_with_Char[i]);
      // int c = getchar();
    }
  }
}

unsigned short int equal_Char(unsigned char const a, unsigned char const b) {
  // char temp = a - b;

  if (a == b) {
    return 0; //success
  }
  else {
    return 1;
  }

  // if (b == 0) {//cannot divide with zero
  //   if (a == 0) {
  //     return 0;//success
  //   }
  //   else {
  //     return 1;
  //   }
  // }
  // else {

  //   if (!(temp / b)) {
  //     return 0; //success
  //   }
  //   else {
  //     return 1;
  //   }
  // }
}



void read_layer_dimensions() {


    // Input_Output_batch_dim=2000;
    Input_Output_batch_dim=20;
    Input_Y_dim=54;
    Input_X_dim=54;
    Input_depth_dim=256;

    Stride_Y_dim=1;
    Stride_X_dim=1;

    Mask_Y_dim=3;
    Mask_X_dim=3;

    Output_depth_dim=128;
    Output_X_dim=(Input_X_dim-(Mask_X_dim-Stride_X_dim)) / Stride_X_dim;
    Output_Y_dim=(Input_Y_dim-(Mask_Y_dim-Stride_Y_dim)) / Stride_Y_dim;

  unsigned long long int In_size = (unsigned long long int) Input_Output_batch_dim * Input_X_dim * Input_Y_dim * Input_depth_dim;
  unsigned long long int Filter_size = (unsigned long long int) Input_depth_dim * Mask_X_dim * Mask_Y_dim * Output_depth_dim;
  unsigned long long int Out_size = (unsigned long long int) Input_Output_batch_dim * Output_Y_dim * Output_X_dim * Output_depth_dim;






  printf("\n Layer dimensions are read");
  printf("\n Input dims (batch,y,x,depth) = (%d, %d, %d, %d)       - Size in Elements = %llu", Input_Output_batch_dim, Input_Y_dim, Input_X_dim, Input_depth_dim, In_size);
  printf("\n Filter dims (m,y,x,depth) = (%d, %d, %d, %d)           - Size in Elements = %llu", Output_depth_dim, Mask_Y_dim, Mask_X_dim, Input_depth_dim, Filter_size);
  printf("\n Output dims (batch,y,x,out_depth) = (%d, %d, %d, %d) - Size in Elements = %llu", Input_Output_batch_dim, Output_Y_dim, Output_X_dim, Output_depth_dim, Out_size);

}





void load_bias_FP() {

  bias_array_FP = (float*)_mm_malloc(Output_depth_dim * sizeof(float), 64);
  if (bias_array_FP == NULL) {
    printf("\nerror with malloc allocating bias array");
    exit(EXIT_FAILURE);
  }


  #ifdef QUANTISATION
    bias_array_Int = (int*)_mm_malloc(Output_depth_dim * sizeof(int), 64);
    if (bias_array_Int == NULL) {
      printf("\nerror with malloc allocating bias int array");
      exit(EXIT_FAILURE);
    }

    float min = 1.00f;
    float max = 5.00f;
    int levels = 65536;   // maybe swap to max int val?
    float scale = (max - min) / (levels-1);
    float zero_point = -min / scale;
  #endif


  int cnt = 0;
  for (unsigned int i = 0; i < Output_depth_dim; i++) {
    *(bias_array_FP + i) = ((float)(rand() % 5)) + 1;
    //  *(bias_array_FP+i)=0.0f;
    // printf("  %d",*(in+i));

    #ifdef QUANTISATION
      *(bias_array_Int + i) = round((*(bias_array_FP + i) / scale) + zero_point);
    #endif

    cnt++; // for debugging
  }

  #ifdef QUANTISATION
    _mm_free(bias_array_FP);
  #endif
}




//in[] is stored into memory like that : in[Input_Output_batch_dim] [Input_Y_dim] [Input_X_dim] [Input_depth_dim] ;
//out[] is stored into memory like that : out[Input_Output_batch_dim] [Output_Y_dim] [Output_X_dim] [Output_depth_dim] ;
int load_create_input_output_array_FP() {

  unsigned long long int input_size = (unsigned long long int) Input_Output_batch_dim * Input_depth_dim * Input_Y_dim * Input_X_dim;
  unsigned long long int output_size = (unsigned long long int) Input_Output_batch_dim * Output_depth_dim * Output_Y_dim * Output_X_dim;
  unsigned long long int in_subscript, out_subscript;

  in_FP = (float*)_mm_malloc(input_size * sizeof(float), 64);
  if (in_FP == NULL) {
    printf("\nerror with malloc allocating input array");
    exit(EXIT_FAILURE);
  }


  // experiment with using lookup tables?
  // quantization relevant ops
  #ifdef QUANTISATION
    in_Char = (unsigned char*)_mm_malloc(input_size * sizeof(unsigned char), 64);
    if (in_Char == NULL) {
      printf("\nerror with malloc allocating input char array");
      exit(EXIT_FAILURE);
    }

    float min = 0.73f;  // lowest in_FP value == 0.73
    float max = 49.73f; // highest in_FP value == 49.73
    int levels = 256;    // 256 due to char being 2 Byte, maybe swap to 50 levels, 0 to 49?
    float scale = (max - min) / (levels - 1);
    float zero_point = -min / scale;
  #endif


  int cnt = 0;
  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) {
    for (unsigned int y = 0; y < Input_Y_dim; y++) {
      for (unsigned int x = 0; x < Input_X_dim; x++) {
        for (unsigned int d = 0; d < Input_depth_dim; d++) {
          in_subscript = (unsigned long long int) b * Input_Y_dim * Input_X_dim * Input_depth_dim + (unsigned long long int) y * Input_X_dim * Input_depth_dim + (unsigned long long int) x * Input_depth_dim + d;

          in_FP[in_subscript] = ((float)(d % 50)) + 0.73f;

          #ifdef QUANTISATION
            in_Char[in_subscript] = (unsigned char)((in_FP[in_subscript] / scale) + zero_point);
          #endif
          // cnt++; // for debugging
        }
      }
    }
  }

  #ifndef QUANTISATION

  out_FP = (float*)_mm_malloc(output_size * sizeof(float), 64);
  if (out_FP == NULL) {
    printf("\nerror with malloc allocating output array");
    exit(EXIT_FAILURE);
  }

  out_to_compare_with_FP = (float*)_mm_malloc(output_size * sizeof(float), 64);
  if (out_to_compare_with_FP == NULL) {
    printf("\nerror with malloc allocating output array to compare with");
    exit(EXIT_FAILURE);
  }


  #else

    _mm_free(in_FP);

    // allocate memory for quantised tensors
    out_Char = (unsigned char*)_mm_malloc(output_size * sizeof(unsigned char), 64);
    if (out_Char == NULL) {
      printf("\nerror with malloc allocating output char array");
      exit(EXIT_FAILURE);
    }

    out_to_compare_with_Char = (unsigned char*)_mm_malloc(output_size * sizeof(unsigned char), 64);
    if (out_to_compare_with_Char == NULL) {
      printf("\nerror with malloc allocating output char array");
      exit(EXIT_FAILURE);
    }

  #endif


  // cnt = 0;
  for (unsigned int b = 0; b < Input_Output_batch_dim; b++) {
    for (unsigned int y = 0; y < Output_Y_dim; y++) {
      for (unsigned int x = 0; x < Output_X_dim; x++) {
        for (unsigned int m = 0; m < Output_depth_dim; m++) {
          out_subscript = (unsigned long long int) b * Output_depth_dim * Output_X_dim * Output_Y_dim +
            (unsigned long long int) y * Output_depth_dim * Output_X_dim +
            (unsigned long long int) x * Output_depth_dim
            + m;

          #ifndef QUANTISATION

          out_to_compare_with_FP[out_subscript] = 0.0f;
          out_FP[out_subscript] = 0.0f;

          #else
            out_to_compare_with_Char[out_subscript] = 0;
            out_Char[out_subscript] = 0;
          #endif
          // cnt++; // for debugging
        }
      }
    }
  }

  return 0;
}




void deallocate_FP() {
  _mm_free(in_FP);
  _mm_free(out_FP);

  _mm_free(out_to_compare_with_FP);

  _mm_free(bias_array_FP);

  _mm_free(filter_FP);
}

void deallocate_Char() {
  _mm_free(in_Char);
  _mm_free(out_Char);

  _mm_free(out_to_compare_with_Char);

  _mm_free(bias_array_Int);

  _mm_free(filter_Char);
}




// filter array is stored into memory tile-wise
int load_filter_array_FP() {

  unsigned int filter_size = Mask_X_dim * Mask_Y_dim * Input_depth_dim * Output_depth_dim;
  unsigned int y, x, m, d, offset, cnt = 0;

  filter_FP = (float*)_mm_malloc(filter_size * sizeof(float), 64);
  if (filter_FP == NULL) {
    printf("\nerror with malloc allocating filter array");
    exit(EXIT_FAILURE);
  }



  #ifdef QUANTISATION
    filter_Char = (signed char*)_mm_malloc(filter_size * sizeof(char), 64);
    if (filter_Char == NULL) {
      printf("\nerror with malloc allocating filter char array");
      exit(EXIT_FAILURE);
    }

    float min = -7.973f;
    float max = 7.973f;
    int levels = 256;
    float scale = (max - min) / (levels - 1);
    float val, val2;
  #endif

  //read the filter array
  for (m = 0; m < Output_depth_dim; m++) {
    for (y = 0; y < Mask_Y_dim; y++) {
      for (x = 0; x < Mask_X_dim; x++) {
        //printf("\n");
        for (d = 0; d < Input_depth_dim; d += 2) {
          offset = m * Mask_Y_dim * Mask_X_dim * Input_depth_dim +
            y * Mask_X_dim * Input_depth_dim +
            x * Input_depth_dim + d;

          filter_FP[offset] = ((rand() % 8) + 0.973f);
          filter_FP[offset + 1] = -((rand() % 8) + 0.973f);

          // filter_FP[offset] = ((d % 8) + 0.973f);              // FOR DEBUGING *******************
          // filter_FP[offset + 1] = -((d % 8) + 0.973f);
          // printf("\n %d, %d",filter_FP[offset],filter_FP[offset+1]);

          #ifdef QUANTISATION
            filter_Char[offset] = (signed char)fmin(filter_FP[offset]/ scale, SCHAR_MAX);
            filter_Char[offset + 1] = (signed char)fmax(filter_FP[offset+1]/ scale, SCHAR_MIN);

            // dequantistaion for debugging:
            val = filter_Char[offset] * scale;
            val2 = filter_Char[offset + 1] * scale;

            // 1st asymmetric quantisation attempt:
            // filter_Char[offset] = (signed char)((filter_FP[offset] / -scale) + zero_point);
            // filter_Char[offset + 1] = (signed char)((filter_FP[offset + 1] / scale) + zero_point);

            // asymmetric de-quantisation for debugging:
            // val = scale * (filter_Char[offset] - zero_point);
            // val2 = scale * (filter_Char[offset + 1] - zero_point);
          #endif

          // cnt++; // for debugging
        }
      }
    }
  }


  #ifdef QUANTISATION
    _mm_free(filter_FP);
  #endif

  //printf("\n Filter array is created and loaded. \n");
  return 0;
}



