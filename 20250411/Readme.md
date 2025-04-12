# Heterogeneous Data Parallel Computing: Summary Notes

These notes summarize Chapter 2 of *Programming Massively Parallel Processors* with contributions from David Luebke, focusing on heterogeneous data parallel computing using CUDA C. The chapter introduces the concept of data parallelism, the structure of CUDA C programs, and the implementation of a vector addition kernel to illustrate key concepts.

---

## 2.1 Data Parallelism

**Definition**: Data parallelism occurs when computations on different parts of a dataset can be performed independently, allowing them to execute simultaneously. This is common in applications processing large datasets, such as image processing, scientific simulations, and molecular dynamics.

**Key Characteristics**:
- **Independence**: Computations on different data elements (e.g., pixels, grid points) do not depend on each other.
- **Scalability**: Data parallelism enables scalable parallel execution on massively parallel processors like GPUs.
- **Examples**:
  - **Image Processing**: Converting a color image to grayscale involves computing a luminance value for each pixel using the formula:
    $$
    L = 0.21r + 0.72g + 0.07b
    $$
    Each pixel’s computation is independent, making it ideal for parallel execution.
  - **Scientific Simulations**: Fluid dynamics or molecular interactions often involve independent calculations for billions of elements.
- **Implementation**: Data parallel code organizes computations around the data, breaking them into independent tasks that can be executed concurrently.

**Data vs. Task Parallelism**:
- **Data Parallelism**: Focuses on dividing a large dataset into smaller chunks processed in parallel (main source of scalability).
- **Task Parallelism**: Involves independent tasks (e.g., vector addition and matrix multiplication) executed concurrently. While useful, it’s less scalable than data parallelism for large datasets.

---

## 2.2 CUDA C Program Structure

**Overview**: CUDA C extends ANSI C to support heterogeneous computing systems with CPUs (host) and GPUs (device). It allows programmers to write both host and device code in a single source file.

**Key Components**:
- **Host Code**: Traditional C code executed on the CPU.
- **Device Code**: Marked with CUDA keywords, executed on the GPU as kernels.
- **Kernels**: Functions that define the parallel computation performed by multiple threads on the GPU.
- **Execution Flow**:
  1. The program starts with host code execution on the CPU.
  2. When a kernel is called, a grid of threads is launched on the GPU to execute the kernel in parallel.
  3. After the kernel completes, control returns to the host until another kernel is launched.
- **Thread Hierarchy**: Threads are organized into a grid of thread blocks, enabling efficient parallel execution.

**Scalability**: CUDA programs scale with hardware. A kernel can run on small GPUs (fewer threads in parallel) or large GPUs (many threads in parallel) without code changes.

---

## 2.3 A Vector Addition Kernel

**Purpose**: Vector addition serves as a simple example to demonstrate CUDA C programming, akin to a "Hello World" for parallel computing.

**Traditional C Implementation**:
- A function `vecAdd(A_h, B_h, C_h, n)` computes `C_h[i] = A_h[i] + B_h[i]` for `i` from 0 to `n-1` using a for-loop.
- Arrays `A_h`, `B_h`, and `C_h` are allocated and initialized on the host (CPU).

**CUDA Implementation**:
- The computation is moved to a GPU kernel to exploit data parallelism.
- Each thread computes one element of the output array `C`, replacing the sequential loop with parallel thread execution.

**Kernel Code**:
```c
__global__ void vecAddKernel(float* A, float* B, float* C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}
```
- `__global__`: Indicates a kernel function executable on the GPU and callable from the host.
- `threadIdx.x`, `blockIdx.x`, `blockDim.x`: Built-in variables to compute a unique global index `i` for each thread.
- `if (i < n)`: Ensures threads do not access out-of-bounds memory for vectors not perfectly divisible by the block size.

---

## 2.4 Device Global Memory and Data Transfer

**Device Global Memory**: GPUs have their own memory (e.g., NVIDIA Volta V100 has 16GB or 32GB), separate from host memory. CUDA programs must manage data transfers between host and device.

**CUDA API Functions**:
- **Memory Allocation**:
  - `cudaMalloc(void** ptr, size_t size)`: Allocates `size` bytes in device global memory, setting `ptr` to point to the allocated memory.
  - Example: `cudaMalloc((void**)&A_d, n * sizeof(float))` allocates space for array `A_d`.
- **Memory Deallocation**:
  - `cudaFree(void* ptr)`: Frees device memory pointed to by `ptr`.
  - Example: `cudaFree(A_d)`.
- **Data Transfer**:
  - `cudaMemcpy(void* dst, void* src, size_t size, cudaMemcpyKind kind)`: Copies `size` bytes from `src` to `dst`.
  - `kind` specifies transfer direction: `cudaMemcpyHostToDevice`, `cudaMemcpyDeviceToHost`, etc.
  - Example: `cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice)` copies array `A_h` from host to `A_d` on device.

**Vector Addition Example**:
- Allocate device memory for `A_d`, `B_d`, `C_d`.
- Copy `A_h`, `B_h` from host to `A_d`, `B_d`.
- Execute kernel to compute `C_d = A_d + B_d`.
- Copy `C_d` back to `C_h` on host.
- Free `A_d`, `B_d`, `C_d`.

**Note**: Frequent data transfers can be inefficient. Real applications often keep data on the device across multiple kernel calls to amortize overhead.

---

## 2.5 Kernel Functions and Threading

**Kernel Functions**:
- Defined with `__global__` keyword, executed by all threads in a grid.
- Follow the Single-Program Multiple-Data (SPMD) model, where all threads run the same code but process different data.
- Example: In `vecAddKernel`, each thread computes one addition.

**Thread Hierarchy**:
- **Grid**: A collection of thread blocks launched by a kernel call.
- **Thread Block**: A group of threads (up to 1024) that execute together.
- **Threads**: Individual execution units within a block.

**Built-in Variables**:
- `threadIdx.x`: Unique thread index within a block (0 to blockDim.x-1).
- `blockIdx.x`: Unique block index within a grid.
- `blockDim.x`: Number of threads per block.
- **Global Index Calculation**: `i = threadIdx.x + blockDim.x * blockIdx.x` gives each thread a unique index to access data.

**Thread Organization**:
- Threads can be organized in 1D, 2D, or 3D arrays, matching the data’s dimensionality.
- Block size should be a multiple of 32 for hardware efficiency (e.g., 256 threads per block in the example).

**Conditional Execution**:
- The `if (i < n)` statement in the kernel ensures correctness for vector lengths not divisible by the block size, preventing out-of-bounds memory access.

---

## 2.6 Calling Kernel Functions

**Syntax**:
```c
kernelName<<<numBlocks, threadsPerBlock>>>(args);
```
- `numBlocks`: Number of thread blocks in the grid.
- `threadsPerBlock`: Number of threads per block.
- Example: `vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n)` launches a grid with enough blocks to cover `n` elements, each block having 256 threads.

**Block Calculation**:
- Use `ceil(n/256.0)` to ensure enough threads are launched to process all `n` elements.
- Example: For `n = 1000`, `ceil(1000/256.0) = 4` blocks, yielding `4 * 256 = 1024` threads (24 threads are idle due to the `if` condition).

**Complete Host Code**:
```c
void vecAdd(float* A_h, float* B_h, float* C_h, int n) {
    float *A_d, *B_d, *C_d;
    int size = n * sizeof(float);
    
    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);
    
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);
    
    vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);
    
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
    
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}
```

**Scalability**: The kernel’s execution order is arbitrary, allowing it to run efficiently on GPUs of varying sizes.

---

## 2.7 Compilation

**NVCC Compiler**:
- NVIDIA’s CUDA C compiler (NVCC) processes CUDA C code, separating host and device code.
- **Host Code**: Compiled with standard C/C++ compilers (e.g., gcc) for CPU execution.
- **Device Code**: Compiled into PTX (virtual binary) files, then translated to GPU-executable object code at runtime.

**Process**:
1. NVCC identifies CUDA keywords (e.g., `__global__`) to distinguish device code.
2. Host code is passed to a standard compiler.
3. Device code is compiled into PTX, then into GPU-specific binaries.

## 2.8 Findings
![CPU vs GPU performance](/20250411/qwe_download.png)
This graph shows how CPU and GPU perform when adding two vectors of increasing size. For small vectors, the CPU is actually faster—mostly because launching GPU kernels and transferring memory adds some overhead. But once the vector size gets big enough (around 10,000 elements), the GPU starts to shine. It handles the heavy lifting way better and scales nicely as the data grows. By the time we hit 10 million elements, the GPU is clearly the better option. TL;DR: CPUs are great for small tasks, but GPUs crush it with large-scale parallel work.
