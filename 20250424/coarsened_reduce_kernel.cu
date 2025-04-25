#include <stdio.h>
#include <stdlib.h>

#define BLOCK_DIM 1024
#define COARSE_FACTOR 4

__global__ void reduce_kernel(float* input, float* output, unsigned int n)
{
	unsigned int segment = blockIdx.x * blockDim.x*2*COARSE_FACTOR;
	unsigned int i = threadIdx.x + segment;

	__shared__ float input_s[BLOCK_DIM];
	float sum = 0.0f;

	for (unsigned int tile = 0; tile < COARSE_FACTOR*2; +tile) {
		sum += input[i+tile*BLOCK_DIM];
	}
	input_s[threadIdx.x] = sum;
	__syncthreads();

	for (unsigned int stride = BLOCK_DIM; stride > 0; stride /=2)
	{
		if (threadIdx.x < stride)
		{
			input_s[threadIdx.x] += input_s[threadIdx.x + stride];
		}
		__syncthreads();
		if (threadIdx.x == 0)
		{
			output[blockIdx.x] = input[threadIdx.x];
		}
	}
}

void reduce_gpu(float* input, float* output, unsigned int n)
{
	float *input_d;
	float *output_d;
	
    	const unsigned int numThreadsPerBlock = BLOCK_DIM;
    	const unsigned int numElementsPerBlock = numThreadsPerBlock * 2 * COARSE_FACTOR;
    	const unsigned int numBlocks = (n + numElementsPerBlock - 1) / numElementsPerBlock;


	cudaMalloc((void**)&input_d, n*sizeof(float));
	cudaMalloc((void**)&output_d, numBlocks * sizeof(float));
	cudaDeviceSynchronize();

	cudaMemcpy(input_d,input,n*sizeof(float),cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();


	reduce_kernel<<<numBlocks,numThreadsPerBlock>>>(input_d,output_d,n);
	cudaDeviceSynchronize();

	cudaMemcpy(output,output_d,numBlocks*sizeof(float),cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	cudaFree(output_d);
	cudaFree(input_d);
}

int main()
{
	unsigned int n = 1000000;

	float *input = (float*)malloc(n * sizeof(float));
	unsigned int numBlocks = (n + BLOCK_DIM*2-1)/(BLOCK_DIM*2);
	float *output = (float*)malloc(numBlocks * sizeof(float));

	for (unsigned int i = 0; i < n; i++) {
		input[i] = 1.0f;
	}

	reduce_gpu(input,output,n);

	float final_sum = 0.0f;
	for (unsigned int i = 0; i < numBlocks; i++){
		final_sum +=output[i];
	}

	free(input);
	free(output);


	
	return 0;
}
