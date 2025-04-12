#include <iostream>
#include <chrono>
#include <fstream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>

// Kernel function
// CUDA runtime system launches a grid of threads 
__global__ void vecAddKernel(float* A, float* B, float* C, int n) {
    // the number of threads in a block is available in a built-in variable named blockDim. 
    // blockDim variable is a struct with three unsigned integer fields (x, y, and z) that help the programmer to organize the threads into a one-, two-, or threedimensional array.
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

// CPU vector addition
void vecAddCPU(float* A, float* B, float* C, int n) {
    for (int i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    std::ofstream log("timing_results.csv");
    log << "n,CPU Time (ms),GPU Time (ms)\n";

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    std::cout << "GPU: " << deviceProp.name << "\n";
    std::cout << "Compute Capability: " << deviceProp.major << "." << deviceProp.minor << "\n";

    for (int n = 1000; n <= 10000000; n *= 10) {
        // Seed random number generator
        srand(time(0));

        float* A_h = new float[n];
        float* B_h = new float[n];
        float* C_h = new float[n];
        float* C_cpu = new float[n];

        for (int i = 0; i < n; i++) {
            A_h[i] = static_cast<float>(rand()) / RAND_MAX; // Random float [0,1)
            B_h[i] = static_cast<float>(rand()) / RAND_MAX;
        }

        // GPU timing using cudaEvent
        cudaEvent_t start_gpu, stop_gpu;
        cudaEventCreate(&start_gpu);
        cudaEventCreate(&stop_gpu);

        float *A_d, *B_d, *C_d;
        int size = n * sizeof(float);
        cudaMalloc(&A_d, size);
        cudaMalloc(&B_d, size);
        cudaMalloc(&C_d, size);

        cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
        cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

        cudaEventRecord(start_gpu);

	
	// total number of threads in each thread block is specified by the host code when a kernel is called
        vecAddKernel<<<(n + 256 - 1) / 256, 256>>>(A_d, B_d, C_d, n);
        cudaEventRecord(stop_gpu);
        
        cudaEventSynchronize(stop_gpu);

        cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

        float gpu_time_ms = 0.0f;
        cudaEventElapsedTime(&gpu_time_ms, start_gpu, stop_gpu);

        // Cleanup GPU memory and events
        cudaFree(A_d);
        cudaFree(B_d);
        cudaFree(C_d);
        cudaEventDestroy(start_gpu);
        cudaEventDestroy(stop_gpu);

        // CPU timing
        auto start_cpu = std::chrono::high_resolution_clock::now();
        vecAddCPU(A_h, B_h, C_cpu, n);
        auto end_cpu = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;

        // Log results
        log << n << "," << cpu_time.count() << "," << gpu_time_ms << "\n";

        // Free memory
        delete[] A_h;
        delete[] B_h;
        delete[] C_cpu;
        delete[] C_h;
    }

    log.close();
    return 0;
}

