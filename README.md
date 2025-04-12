## Day 01
 **File** `20250411/vectorAdd.cu`

Implemented a basic CUDA program for vector addition, where each thread handles the addition of a single pair of elements from two input arrays. This involved launching a CUDA kernel to perform the computation in parallel.

**Key Takeaways:**

- Gained hands-on experience with writing a simple CUDA kernel.
- Developed an understanding of CUDA’s execution hierarchy, including grids, blocks, and threads.
- Learned how to allocate and manage GPU memory using functions like `cudaMalloc`, `cudaMemcpy`, and `cudaFree`.
- Compared the performance of the GPU implementation with a CPU-based version to observe speedup and efficiency benefits.
    

**Reading:**

- Chapter 1 of _Programming Massively Parallel Processors_ (PMPP)
- @onaecO. **"Pointers in C for Absolute Beginners – Full Course."** _YouTube_, uploaded .by freeCodeCamp.org, 15 June 2023, [https://www.youtube.com/watch?v=MIL2BK02X8A](https://www.youtube.com/watch?v=MIL2BK02X8A).
- El Hajj, Izzat. **"Lecture 02 - Data Parallel Programming."** _YouTube_, uploaded by Programming Massively Parallel Processors, 22 July 2022, [https://www.youtube.com/watch?v=iE-xGWBQtH0](https://www.youtube.com/watch?v=iE-xGWBQtH0).
