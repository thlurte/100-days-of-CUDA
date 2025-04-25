#include <stdio.h>
#include <stdlib.h>

__global__ void histogram_kernel(char* data, int* hist, int length) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < length) {
		unsigned int pos = data[idx] - 'a';
		if (pos < 26 && pos >= 0) {
		atomicAdd(&(hist[pos]),1);
		}
	}
}

void histogram_gpu(char *data_h, int *hist_h, int length) {
	char* data_d;
	int* hist_d;

	cudaMalloc((void**)&data_d,length*sizeof(char));
	cudaMalloc((void**)&hist_d,26*sizeof(int));
	cudaMemset(hist_d, 0, 26 * sizeof(int));
	cudaDeviceSynchronize();
	
	cudaMemcpy(data_d,data_h,length*sizeof(char),cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	const unsigned int numBlocks = 256;
	const unsigned int numThreadsPerBlock = (length + numBlocks - 1)/numBlocks;
	histogram_kernel<<<numBlocks,numThreadsPerBlock>>>(data_d,hist_d,length);
	cudaDeviceSynchronize();

	cudaMemcpy(hist_h,hist_d,26*sizeof(int),cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	cudaFree(data_d);
	cudaFree(hist_d);
}




int main() 
{
	const unsigned int n = 1000000;
	char* data = (char *)malloc(n*sizeof(char));
	int* hist = (int *)calloc(26,sizeof(int));
	int* hist_gpu = (int *)calloc(26,sizeof(int));

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

	histogram_gpu(data,hist_gpu,n);

	free(data);
	free(hist);
	return 0;

}



