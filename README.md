## Day 01
 **File** `20250411/vectorAdd.cu`
![CPU vs GPU performance](/20250411/qwe_download.png)
Implemented a basic CUDA program for vector addition, where each thread handles the addition of a single pair of elements from two input arrays and compared it's performance with a sequteial computation.

Chapter 2 introduces the SIMT (Single Instruction, Multiple Threads) execution model. The GPU executes threads in groups of 32 called warps, and although each thread follows the same instruction, divergent control flow (like if-else) within a warp leads to warp divergence, hurting performance. 

Threads are organized hierarchically into blocks and grids. Each thread and block gets unique IDs through built-in variables like threadIdx, blockIdx, etc.

**Reading:**
- Chapter 2 of _Programming Massively Parallel Processors_ (PMPP)
- @onaecO. **"Pointers in C for Absolute Beginners – Full Course."** _YouTube_, uploaded .by freeCodeCamp.org, 15 June 2023, [https://www.youtube.com/watch?v=MIL2BK02X8A](https://www.youtube.com/watch?v=MIL2BK02X8A).
- El Hajj, Izzat. **"Lecture 02 - Data Parallel Programming."** _YouTube_, uploaded by Programming Massively Parallel Processors, 22 July 2022, [https://www.youtube.com/watch?v=iE-xGWBQtH0](https://www.youtube.com/watch?v=iE-xGWBQtH0).

## Day 02
**File** `20250412/rgb2greyscale.cu`
![CPU vs GPU performance](/20250412/seq.png)

Implemented a CUDA program for converting RGB image into greyscale.

CUDA threads are grouped into blocks, and blocks into grids. Blocks are limited in size (typically 1024 threads max), but grids can scale far beyond that. The programmer defines how threads and blocks are laid out — usually in 1D, 2D, or 3D — using dim3 constructs. Threads access their coordinates through threadIdx, blockIdx, and related fields, allowing for intuitive indexing into linear or multidimensional data arrays.

**Reading:**
- Chapter 3 of _Programming Massively Parallel Processors_ (PMPP)
- El Hajj, Izzat. **"Lecture 03 - Multidimensional Grids and Data."** _YouTube_, uploaded by Programming Massively Parallel Processors, 22 July 2022, [https://www.youtube.com/watch?v=iE-xGWBQtH0](https://www.youtube.com/watch?v=iE-xGWBQtH0).

## Day 03
**File** `20250413/Day_3.ipynb` (blur image)
**File** `20250413/Day_3_1.ipynb` (matrix multiplication)

![CPU vs GPU performance](/20250413/qwe_download.png)
Implemented a CUDA program for bluring image.
Implemented a CUDA program for calculating the output of matrix multiplication for two `N` $\times$ `N` matrix.


**Reading:**
- Chapter 3 of _Programming Massively Parallel Processors_ (PMPP)
- El Hajj, Izzat. **"Lecture 03 - Multidimensional Grids and Data."** _YouTube_, uploaded by Programming Massively Parallel Processors, 22 July 2022, [https://www.youtube.com/watch?v=iE-xGWBQtH0](https://www.youtube.com/watch?v=iE-xGWBQtH0).

## Day 04
**File** `20250414/Day_4.ipynb` (relu activation function)
**File** `20250414/Day_4.1.ipynb` (tanh activation function)
**File** `20250414/Day_4.2.ipynb` (sigmoid activation function)

Implemented a CUDA program for activation  functions.

**Reading:**
- Chapter 4 of _Programming Massively Parallel Processors_ (PMPP)
- El Hajj, Izzat. **"Lecture 04 - GPU Architecture."** _YouTube_, uploaded by Programming Massively Parallel Processors, 22 July 2022, [https://www.youtube.com/watch?v=iE-xGWBQtH0](https://www.youtube.com/watch?v=iE-xGWBQtH0).

## Day 05
**File** `20250415/Day_5.ipynb` (transpose a matrix)**

Implemented a CUDA program for transposing a N by N matrix.

**Reading:**
- Chapter 5 of _Programming Massively Parallel Processors_ (PMPP)
- El Hajj, Izzat. **"Lecture 05 - Memory and Tiling."** _YouTube_, uploaded by Programming Massively Parallel Processors, 22 July 2022, [https://www.youtube.com/watch?v=iE-xGWBQtH0](https://www.youtube.com/watch?v=iE-xGWBQtH0).

## Day 06
**File** `20250416/Day_6.ipynb` tiled matrix multiplication

Implemented tiled matrix multiplication algorithm.

**Reading:**
- Chapter 5 of _Programming Massively Parallel Processors_ (PMPP)
- El Hajj, Izzat. **"Lecture 05 - Memory and Tiling."** _YouTube_, uploaded by Programming Massively Parallel Processors, 22 July 2022, [https://www.youtube.com/watch?v=iE-xGWBQtH0](https://www.youtube.com/watch?v=iE-xGWBQtH0).

## Day 07
**File** `20250417/Day_7.ipynb` coarsed matric multiplication

Implemented coarsend matrix multiplication.

**Reading:**
- Chapter 5 of _Programming Massively Parallel Processors_ (PMPP)
- El Hajj, Izzat. **"Lecture 05 - Memory and Tiling."** _YouTube_, uploaded by Programming Massively Parallel Processors, 22 July 2022, [https://www.youtube.com/watch?v=iE-xGWBQtH0](https://www.youtube.com/watch?v=iE-xGWBQtH0).

## Day 08

**File** `20250418/Day_8.ipynb` 
**File** `20250418/Day_8_1.ipynb` 

Implemented atomic reduction algorithm.
Implemented pairwise reduction algorithm.


**Reading:**
- ENCCS. (n.d.). Optimizing the GPU kernel — CUDA training materials. Retrieved April 17, 2025, from https://enccs.github.io/cuda/3.01_ParallelReduction/

## Day 09

**File** `20250419/Day_9.ipynb`

Implemted 2D convolution with shared memory.

- El Hajj, Izzat. **"Lecture 07 - Convolution."** _YouTube_, uploaded by Programming Massively Parallel Processors, 22 July 2022, [https://www.youtube.com/watch?v=iE-xGWBQtH0](https://www.youtube.com/watch?v=iE-xGWBQtH0).

## Day 10

**File** `20250420/Day_10_1.ipynb`
**File** `20250420/Day_10_2.ipynb`

Implemented naive stencil kernel.
Implemented tiled stencil kernel with shared memory.

- El Hajj, Izzat. **"Lecture 08 - Stencil."** _YouTube_, uploaded by Programming Massively Parallel Processors, 22 July 2022, [https://www.youtube.com/watch?v=iE-xGWBQtH0](https://www.youtube.com/watch?v=iE-xGWBQtH0).

## Day 11

**File** `20250421/reduce_kernel.cu`
**File** `20250421/sm_reduce_kernel.cu`

Implemented coarsned reduction kernel
Implemented shared reduction kernel

- El Hajj, Izzat. **"Lecture 10 - Reduction."** _YouTube_, uploaded by Programming Massively Parallel Processors, 22 July 2022, [https://www.youtube.com/watch?v=iE-xGWBQtH0](https://www.youtube.com/watch?v=iE-xGWBQtH0).


## Day 12 

**File** `20250422/1d_convolution_kernel.cu`
**File** `20250422/histogram_kernel.cu`

Implemented naive histogram kernel.
Implemented 1d convolutional kernel.

- El Hajj, Izzat. **"Lecture 09 - Histogram."** _YouTube_, uploaded by Programming Massively Parallel Processors, 22 July 2022, [https://www.youtube.com/watch?v=iE-xGWBQtH0](https://www.youtube.com/watch?v=iE-xGWBQtH0).

## Day 13

**File** `20250423/histogram_private_kernel.cu`

Implemented hisogram kernel with private memory and atomic operations.


- El Hajj, Izzat. **"Lecture 09 - Histogram."** _YouTube_, uploaded by Programming Massively Parallel Processors, 22 July 2022, [https://www.youtube.com/watch?v=iE-xGWBQtH0](https://www.youtube.com/watch?v=iE-xGWBQtH0).

## Day 14


**File** `20250424/coarsened_reduce_kernel.cu`

Implmented coarsened reduction kernel.


- El Hajj, Izzat. **"Lecture 10 - Reduction."** _YouTube_, uploaded by Programming Massively Parallel Processors, 22 July 2022, [https://www.youtube.com/watch?v=iE-xGWBQtH0](https://www.youtube.com/watch?v=iE-xGWBQtH0).

## Day 15

`Revision Chap 2 - 5`

