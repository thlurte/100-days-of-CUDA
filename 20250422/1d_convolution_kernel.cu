#include <stdlib.h>
#include <stdio.h>
#define K 3
__global__ void conv1d_kernel(float* input, float* output, int n){
	idx = blockIdx.x * blockDim.x + threadIdx.x;

	float kernel[] =  {-1,0,1};

	if (idx < n - K + 1) {
		float result = 0.0f;
		for (unsigned int i = 0; i < K; i++) {
			result += input[idx + i] * kernel[i];
		}

		output[idx] = result;
	}
}


void conv1d_gpu(float* data_h, float* output_h,int n)
{
	float *data_d, *output_d;

	cudaMalloc((void **)&data_d, n * sizeof(float));
	cudaMalloc((void **)&output_d, n * sizeof(float));
	cudaDeviceSynchronize();

	cudaMemcpy(data_d,data_h,n*sizeof(float),cudamemcpyhosttodevice);
	cudaDeviceSynchronize();

	const int numblocks = 32;
	const int numthreadsperblock = (n + numblocks - 1)/numblocks;

	conv1d_kernel<<<numblocks, numthreadsperblock>>>(data_d,output_d,n);
	cudaDeviceSynchronize();
	
	cudaMemcpy(output_h, output_d, n*sizeof(float), cudamemcpydevicetohost);
	cudaDeviceSynchronize();

	cudaFree(data_d);
	cudaFree(output_d);
}


int main() 
{
	const unsigned int n = 1000000;

	float* data_h = (float *)malloc(n*sizeof(float));
	float* output_h = (float *)malloc(n*sizeof(float));
	
	srand(42);
	for (unsigned int i = 0; i < n ; i++) {
		if (i%2==0) {
			data_h[i] = i*2.0f;
		} else {
			data_h[i] = (float)i;
		}
	}

	conv1d_gpu(data_h,output_h,n);


	free(data_h);
	free(output_h);

	return 0;

}
