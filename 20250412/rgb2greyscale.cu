#include <iostream>
#include <chrono>
#include <fstream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>
#include <opencv2/opencv.hpp>

__global__ void rgb2GreyKernel(unsigned char* red, unsigned char* green, unsigned char* blue, unsigned int width, unsigned int height, unsigned char* grey){
	unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int column = blockIdx.x * blockDim.x + threadIdx.x;


	if (row < height && column < width) {
		unsigned int i = row*width+column;
		grey[i] = red[i]*3/10 + green[i]*6/10 + blue[i]*1/10;
	}
}

void rgb2Grey(unsigned char* red_h, unsigned char* green_h, unsigned char* blue_h, unsigned char* grey_h, unsigned int width, unsigned int height) {

	unsigned char *red_D, *green_D, *blue_D, *grey_D;
	int size = width * height * sizeof(unsigned char);

	// Allocate Memory in the device
	cudaMalloc((void **)&red_D, size);
	cudaMalloc((void **)&green_D, size);
	cudaMalloc((void **)&blue_D, size);

	cudaMalloc((void **)&grey_D, size);
	cudaDeviceSynchronize();

	// Copy data from host to device
	cudaMemcpy(red_D,red_h,size,cudaMemcpyHostToDevice);
	cudaMemcpy(green_D,green_h,size,cudaMemcpyHostToDevice);
	cudaMemcpy(blue_D,blue_h,size,cudaMemcpyHostToDevice);

	cudaMemcpy(grey_D,grey_h,size,cudaMemcpyHostToDevice);

	// Execute the code in kernel
	dim3 numThreadsPerBlock(32,32);
	dim3 numBlocks((width + numThreadsPerBlock.x - 1)/numThreadsPerBlock.x,(height + numThreadsPerBlock.y - 1)/numThreadsPerBlock.y);
	rgb2GreyKernel<<<numBlocks,numThreadsPerBlock>>>(red_D,green_D,blue_D,width,height,grey_D);
	cudaDeviceSynchronize();

  cudaMemcpy(grey_h, grey_D, size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

  cudaFree(red_D);
  cudaFree(green_D);
  cudaFree(blue_D);
  cudaFree(grey_D);

}

int main() {


    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    std::cout << "GPU: " << deviceProp.name << "\n";
    std::cout << "Compute Capability: " << deviceProp.major << "." << deviceProp.minor << "\n";

    // Load the image using OpenCV
    cv::Mat image = cv::imread("file.jpg");
    if (image.empty()) {
        std::cerr << "Error: Image not found.\n";
        return -1;
    }

    // Get image properties
    unsigned int width = image.cols;
    unsigned int height = image.rows;

    // Split channels
    std::vector<cv::Mat> channels(3);
    cv::split(image, channels);

    // Convert OpenCV Mats to unsigned char arrays
    unsigned char *red_h   = channels[2].data;
    unsigned char *green_h = channels[1].data;
    unsigned char *blue_h  = channels[0].data;

    // Allocate memory for grayscale output
    std::vector<unsigned char> grey_h(width * height);

    // Call your CUDA kernel
    rgb2Grey(red_h, green_h, blue_h, grey_h.data(), width, height);

    // Convert grayscale data back to cv::Mat
    cv::Mat grey_image(height, width, CV_8UC1, grey_h.data());

    // Save or show the result
    cv::imwrite("gray_output.jpg", grey_image);






    return 0;
}