/**
 * ROCm Memory Access Pattern Optimization Examples
 * Simple HIP code demonstrating memory optimization techniques
 */

#include <hip/hip_runtime.h>

// Example 1: Shared Memory Usage for Matrix Multiplication
__global__ void matmul_shared_memory(float* A, float* B, float* C, int N) {
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    
    int row = by * 16 + ty;
    int col = bx * 16 + tx;
    
    float sum = 0.0f;
    
    for (int k = 0; k < (N + 15) / 16; ++k) {
        // Load into shared memory
        if (row < N && k * 16 + tx < N)
            As[ty][tx] = A[row * N + k * 16 + tx];
        else
            As[ty][tx] = 0.0f;
            
        if (col < N && k * 16 + ty < N)
            Bs[ty][tx] = B[(k * 16 + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;
            
        __syncthreads();
        
        // Compute partial result
        for (int i = 0; i < 16; ++i) {
            sum += As[ty][i] * Bs[i][tx];
        }
        
        __syncthreads();
    }
    
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Example 2: Coalesced Memory Access Pattern
__global__ void coalesced_access(float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Good: Coalesced access - consecutive threads access consecutive memory
    if (idx < N) {
        output[idx] = input[idx] * 2.0f;
    }
}

// Example 3: Non-coalesced (inefficient) vs Coalesced (efficient) 
__global__ void bad_memory_pattern(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // BAD: Strided access pattern
    if (idx < N) {
        data[idx * 32] += 1.0f;  // Non-coalesced
    }
}

__global__ void good_memory_pattern(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // GOOD: Sequential access pattern  
    if (idx < N) {
        data[idx] += 1.0f;  // Coalesced
    }
}

// Example 4: Memory Alignment Optimization
__global__ void aligned_copy(float4* src, float4* dst, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Use float4 for 128-bit aligned access
    if (idx < N / 4) {
        dst[idx] = src[idx];  // 4x faster than individual float copies
    }
}