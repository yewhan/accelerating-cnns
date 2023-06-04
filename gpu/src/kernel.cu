//#include <cstdio>

#include "convolution_layer_2D.h"

// declare constant device variables
__constant__ unsigned int d_Mask_Y_dim;
__constant__ unsigned int d_Mask_X_dim;

__constant__ unsigned int d_Input_Y_dim;
__constant__ unsigned int d_Input_X_dim;
__constant__ unsigned int d_Input_depth_dim;
__constant__ unsigned int d_Input_Output_batch_dim;

__constant__ unsigned int d_Stride_Y_dim;
__constant__ unsigned int d_Stride_X_dim;

__constant__ unsigned int d_Output_depth_dim;
__constant__ unsigned int d_Output_X_dim;
__constant__ unsigned int d_Output_Y_dim;



/* ***** TRY USING THE FOLLOWING METHODS TO BATCH TRANSFER DATA TO/ FROM THE DEVICE ***** */
/* 1. cudaMemcpy2D()/ cudaMemcpy3D()
 * 2. cudaMemcpyBatched() *** check to see if using CUDA 11.5+ ***
 * 3. cudaMemcpyAsync()
 * 4. packing data into a single struct and using cudaMemcpy()
 */


int optimised_layer_FP(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
	float* d_in_FP, * d_filter_FP, * d_bias_array_FP, * d_out_to_compare_with_FP;
	cudaMalloc(&d_in_FP, Input_Output_batch_dim * Input_Y_dim * Input_X_dim * Input_depth_dim * sizeof(float));
	cudaMalloc(&d_filter_FP, Output_depth_dim * Mask_Y_dim * Mask_X_dim * Input_depth_dim * sizeof(float));
	cudaMalloc(&d_bias_array_FP, Output_depth_dim * sizeof(float));
	cudaMalloc(&d_out_to_compare_with_FP, Input_Output_batch_dim * Output_Y_dim * Output_X_dim * Output_depth_dim * sizeof(float));

	//cudaMemcpy(d_in_FP, in_FP, Input_Output_batch_dim * Input_Y_dim * Input_X_dim * Input_depth_dim * sizeof(float), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_filter_FP, filter_FP, Output_depth_dim * Mask_Y_dim * Mask_X_dim * Input_depth_dim * sizeof(float), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_bias_array_FP, bias_array_FP, Output_depth_dim * sizeof(float), cudaMemcpyHostToDevice);

	// copy vals to __constant__ vars
	//cudaMemcpyToSymbol(d_Mask_Y_dim, &Mask_Y_dim, sizeof(int), 0, cudaMemcpyHostToDevice);
	//cudaMemcpyToSymbol(d_Mask_X_dim, &Mask_X_dim, sizeof(int), 0, cudaMemcpyHostToDevice);

	//cudaMemcpyToSymbol(d_Input_Y_dim, &Input_Y_dim, sizeof(int), 0, cudaMemcpyHostToDevice);
	//cudaMemcpyToSymbol(d_Input_X_dim, &Input_X_dim, sizeof(int), 0, cudaMemcpyHostToDevice);
	//cudaMemcpyToSymbol(d_Input_depth_dim, &Input_depth_dim, sizeof(int), 0, cudaMemcpyHostToDevice);
	//cudaMemcpyToSymbol(d_Input_Output_batch_dim, &Input_Output_batch_dim, sizeof(int), 0, cudaMemcpyHostToDevice);

	//cudaMemcpyToSymbol(d_Stride_Y_dim, &Stride_Y_dim, sizeof(int), 0, cudaMemcpyHostToDevice);
	//cudaMemcpyToSymbol(d_Stride_X_dim, &Stride_X_dim, sizeof(int), 0, cudaMemcpyHostToDevice);

	//cudaMemcpyToSymbol(d_Output_depth_dim, &Output_depth_dim, sizeof(int), 0, cudaMemcpyHostToDevice);
	//cudaMemcpyToSymbol(d_Output_X_dim, &Output_X_dim, sizeof(int), 0, cudaMemcpyHostToDevice);
	//cudaMemcpyToSymbol(d_Output_Y_dim, &Output_Y_dim, sizeof(int), 0, cudaMemcpyHostToDevice);

	// Create a CUDA stream
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	// Asynchronously copy memory using cudaMemcpyAsync()
	cudaMemcpyAsync(d_in_FP, in_FP, Input_Output_batch_dim * Input_Y_dim * Input_X_dim * Input_depth_dim * sizeof(float), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_filter_FP, filter_FP, Output_depth_dim * Mask_Y_dim * Mask_X_dim * Input_depth_dim * sizeof(float), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_bias_array_FP, bias_array_FP, Output_depth_dim * sizeof(float), cudaMemcpyHostToDevice, stream);

	// Asynchronously copy __constants__ using cudaMemcpyToSymbolAsync()
	cudaMemcpyToSymbolAsync(d_Mask_Y_dim, &Mask_Y_dim, sizeof(int), 0, cudaMemcpyHostToDevice, stream);
	cudaMemcpyToSymbolAsync(d_Mask_X_dim, &Mask_X_dim, sizeof(int), 0, cudaMemcpyHostToDevice, stream);

	cudaMemcpyToSymbolAsync(d_Input_Y_dim, &Input_Y_dim, sizeof(int), 0, cudaMemcpyHostToDevice, stream);
	cudaMemcpyToSymbolAsync(d_Input_X_dim, &Input_X_dim, sizeof(int), 0, cudaMemcpyHostToDevice, stream);
	cudaMemcpyToSymbolAsync(d_Input_depth_dim, &Input_depth_dim, sizeof(int), 0, cudaMemcpyHostToDevice, stream);
	cudaMemcpyToSymbolAsync(d_Input_Output_batch_dim, &Input_Output_batch_dim, sizeof(int), 0, cudaMemcpyHostToDevice, stream);

	cudaMemcpyToSymbolAsync(d_Stride_Y_dim, &Stride_Y_dim, sizeof(int), 0, cudaMemcpyHostToDevice, stream);
	cudaMemcpyToSymbolAsync(d_Stride_X_dim, &Stride_X_dim, sizeof(int), 0, cudaMemcpyHostToDevice, stream);

	cudaMemcpyToSymbolAsync(d_Output_depth_dim, &Output_depth_dim, sizeof(int), 0, cudaMemcpyHostToDevice, stream);
	cudaMemcpyToSymbolAsync(d_Output_X_dim, &Output_X_dim, sizeof(int), 0, cudaMemcpyHostToDevice, stream);
	cudaMemcpyToSymbolAsync(d_Output_Y_dim, &Output_Y_dim, sizeof(int), 0, cudaMemcpyHostToDevice, stream);


	// Synchronize the stream to ensure all memory transfers are completed
	cudaStreamSynchronize(stream);

	// Destroy the CUDA stream
	cudaStreamDestroy(stream);




	// for use with optimised_layer_bmyx_FP_kernel()
	dim3 dimBlock(16, 16, 4); // Optimal for GTX 1080 Ti (1024 threads per block)
	dim3 dimGrid(Input_Output_batch_dim, (Output_depth_dim + 15) / 16, (Output_Y_dim + 15) / 16); // Cover all iterations

	optimised_layer_bmyx_FP_kernel << <dimGrid, dimBlock >> > (d_in_FP, d_filter_FP, d_bias_array_FP, d_out_to_compare_with_FP);



	// for use with optimised_layer_bmy_FP_kernel()
	//dim3 dimBlock(16, 16, 4); // Optimal for GTX 1080 Ti (1024 threads per block)
	//dim3 dimGrid((Input_Output_batch_dim + 15 ) / 16, (Output_depth_dim + 15) / 16, (Output_Y_dim + 3) / 4); // Cover all iterations

	//optimised_layer_bmy_FP_kernel << <dimGrid, dimBlock >> > (d_in_FP, d_filter_FP, d_bias_array_FP, d_out_to_compare_with_FP);





	cudaDeviceSynchronize();

	cudaMemcpy(out_to_compare_with_FP, d_out_to_compare_with_FP, Input_Output_batch_dim * Output_Y_dim * Output_X_dim * Output_depth_dim * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_in_FP);
	cudaFree(d_filter_FP);
	cudaFree(d_bias_array_FP);
	cudaFree(d_out_to_compare_with_FP);

	printf("\n from gpu-opt %f %f ", out_to_compare_with_FP[0], out_to_compare_with_FP[1]);
	return 0;
}





__global__ void optimised_layer_bmyx_FP_kernel(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
	#define RELU(x) ((x) < 0.0f ? 0.0f : (x))

	unsigned int b = blockIdx.x;
	unsigned int m = threadIdx.x + blockIdx.y * blockDim.x;
	unsigned int y = threadIdx.y + blockIdx.z * blockDim.y;
	unsigned int x = threadIdx.z;

	if (b < d_Input_Output_batch_dim && m < d_Output_depth_dim && y < d_Output_Y_dim && x < d_Output_X_dim) {
		const float bias = bias_array_FP[m];

		for (unsigned int x = threadIdx.z; x < d_Output_X_dim; x += blockDim.z) {

			float temp = 0.0f;

			for (unsigned int off_y = 0; off_y < d_Mask_Y_dim; off_y++) {
				for (unsigned int off_x = 0; off_x < d_Mask_X_dim; off_x++) {
					for (unsigned int d = 0; d < d_Input_depth_dim; d++) {
						const unsigned long long int in_subscript = b * (d_Input_Y_dim * d_Input_X_dim * d_Input_depth_dim)
							+ (y * d_Stride_Y_dim + off_y) * d_Input_X_dim * d_Input_depth_dim
							+ (x * d_Stride_X_dim + off_x) * d_Input_depth_dim
							+ d;
						const unsigned long long int filter_subscript = m * d_Mask_Y_dim * d_Mask_X_dim * d_Input_depth_dim
							+ off_y * d_Mask_X_dim * d_Input_depth_dim
							+ off_x * d_Input_depth_dim
							+ d;

						const float s = in_FP[in_subscript];
						const float w = filter_FP[filter_subscript];
						temp = temp + s * w;
					}
				}
			}

			const unsigned long long int out_subscript = b * (d_Output_depth_dim * d_Output_X_dim * d_Output_Y_dim) +
				y * (d_Output_depth_dim * d_Output_X_dim) +
				x * d_Output_depth_dim
				+ m;

			temp += bias;
			out_to_compare_with_FP[out_subscript] = RELU(temp);
		}
	}
}

__global__ void optimised_layer_bmy_FP_kernel(const float* in_FP, const float* filter_FP, const float* bias_array_FP, float* out_to_compare_with_FP) {
#define RELU(x) ((x) < 0.0f ? 0.0f : (x))

	unsigned int b = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int m = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int y = threadIdx.z + blockIdx.z * blockDim.z;
	//unsigned int x = threadIdx.z;

	if (b < d_Input_Output_batch_dim && m < d_Output_depth_dim && y < d_Output_Y_dim) {
		const float bias = bias_array_FP[m];

		for (unsigned int x = 0; x < d_Output_X_dim; x++) {

			float temp = 0.0f;

			for (unsigned int off_y = 0; off_y < d_Mask_Y_dim; off_y++) {
				for (unsigned int off_x = 0; off_x < d_Mask_X_dim; off_x++) {
					for (unsigned int d = 0; d < d_Input_depth_dim; d++) {
						const unsigned long long int in_subscript = b * (d_Input_Y_dim * d_Input_X_dim * d_Input_depth_dim)
							+ (y * d_Stride_Y_dim + off_y) * d_Input_X_dim * d_Input_depth_dim
							+ (x * d_Stride_X_dim + off_x) * d_Input_depth_dim
							+ d;
						const unsigned long long int filter_subscript = m * d_Mask_Y_dim * d_Mask_X_dim * d_Input_depth_dim
							+ off_y * d_Mask_X_dim * d_Input_depth_dim
							+ off_x * d_Input_depth_dim
							+ d;

						const float s = in_FP[in_subscript];
						const float w = filter_FP[filter_subscript];
						temp = temp + s * w;
					}
				}
			}

			const unsigned long long int out_subscript = b * (d_Output_depth_dim * d_Output_X_dim * d_Output_Y_dim) +
				y * (d_Output_depth_dim * d_Output_X_dim) +
				x * d_Output_depth_dim
				+ m;

			temp += bias;
			out_to_compare_with_FP[out_subscript] = RELU(temp);
		}
	}
}


__device__ float Relu_float_kernel(const float temp) {


	if (temp < 0.0f)
		return 0.0f;
	else
		return temp;

}



// 2 GFLOPS (with printf UNcommented)
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


	if (temp < 0.0f)
		return 0.0f;
	else
		return temp;

}