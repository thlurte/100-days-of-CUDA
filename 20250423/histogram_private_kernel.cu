#include <stdio.h>
#include <stdlib.h>

#define NUM_BINS 7

__global__ void histogram_private_kernel(char* data, int* hist, int length) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < length) {
		unsigned int pos = data[idx] - 'a';
		if (pos < 26 && pos >= 0) {
		atomicAdd(&(hist[blockIdx.x*NUM_BINS+pos/4]),1);
		}
	}
	if (blockIdx.x > 0) {
		__syncthreads();
		for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
			unsigned int bin_val = hist[blockIdx.x*NUM_BINS+bin];
			if (bin_val > 0) {
				atomicAdd(&(hist[bin]),bin_val);
			}
		}
	}
}

void histogram_private_gpu(char *data_h, int *hist_h, int length) {
	char* data_d;
	int* hist_d;

	const unsigned int numBlocks = 256;
	const unsigned int numThreadsPerBlock = (length + numBlocks - 1)/numBlocks;
	
	cudaMalloc((void**)&data_d,length*sizeof(char));
	cudaMalloc((void**)&hist_d,numBlocks*NUM_BINS*sizeof(int));
	cudaMemset(hist_d, 0, numBlocks*NUM_BINS * sizeof(int));
	cudaDeviceSynchronize();
	
	cudaMemcpy(data_d,data_h,length*sizeof(char),cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	histogram_private_kernel<<<numBlocks,numThreadsPerBlock>>>(data_d,hist_d,length);
	cudaDeviceSynchronize();

	cudaMemcpy(hist_h,hist_d,NUM_BINS*sizeof(int),cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	cudaFree(data_d);
	cudaFree(hist_d);
}




int main() 
{
	const unsigned int n = 1000000;
	char* data = (char *)malloc(n*sizeof(char));
	int* hist = (int *)calloc(26,sizeof(int));
	int* hist_gpu = (int *)calloc(26/4+1,sizeof(int));

	char alphabet[] = "abcdefghijklmnopqrstuvwxyz";

	srand(42);
	for (unsigned int i = 0; i < n; i++) {
		int idx = rand() % 26;
		data[i] = alphabet[idx];
		hist[idx]++;
	}

	for (unsigned int i = 0; i < 26 ; i++) {
		printf("%c, %u \n", alphabet[i], hist[i]);
	}

	histogram_private_gpu(data,hist_gpu,n);

	free(data);
	free(hist);
	return 0;

}



