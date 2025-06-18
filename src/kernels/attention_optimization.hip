/**
 * ROCm Attention Mechanism Optimization
 * Implementation of Agent 113's optimization design
 * Target: Stable Diffusion attention performance on RDNA3/CDNA3
 */

#include <hip/hip_runtime.h>
#include <rocblas.h>
#include <math.h>

// Optimized attention kernel with shared memory and coalesced access
__global__ void optimized_attention_kernel(
    const float* Q,     // Query matrix [batch_size, seq_len, d_model]
    const float* K,     // Key matrix [batch_size, seq_len, d_model] 
    const float* V,     // Value matrix [batch_size, seq_len, d_model]
    float* output,      // Output matrix [batch_size, seq_len, d_model]
    const int batch_size,
    const int seq_len,
    const int d_model,
    const int num_heads,
    const float scale
) {
    // Thread and block configuration for RDNA3/CDNA3
    const int head_dim = d_model / num_heads;
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || head_idx >= num_heads || seq_idx >= seq_len) return;
    
    // Shared memory for efficient data reuse
    __shared__ float shared_K[64][64];  // Tile size optimized for RDNA3
    __shared__ float shared_V[64][64];
    __shared__ float attention_scores[64];
    
    const int tid = threadIdx.x;
    const int head_offset = batch_idx * num_heads * seq_len * head_dim + 
                           head_idx * seq_len * head_dim;
    
    // Load query vector (coalesced access)
    float q_vec[64];  // Assuming head_dim <= 64
    for (int i = 0; i < head_dim; i++) {
        q_vec[i] = Q[head_offset + seq_idx * head_dim + i];
    }
    
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    
    // Compute attention scores with tiling
    for (int k_start = 0; k_start < seq_len; k_start += blockDim.x) {
        int k_idx = k_start + tid;
        
        // Load K tile into shared memory (coalesced)
        if (k_idx < seq_len) {
            for (int d = 0; d < head_dim; d++) {
                shared_K[tid][d] = K[head_offset + k_idx * head_dim + d];
            }
        }
        __syncthreads();
        
        // Compute attention scores for this tile
        for (int k = 0; k < min(blockDim.x, seq_len - k_start); k++) {
            float score = 0.0f;
            
            // Dot product Q·K^T (vectorized)
            #pragma unroll
            for (int d = 0; d < head_dim; d++) {
                score += q_vec[d] * shared_K[k][d];
            }
            
            score *= scale;
            max_score = fmaxf(max_score, score);
            attention_scores[k] = score;
        }
        __syncthreads();
        
        // Softmax computation (numerically stable)
        for (int k = 0; k < min(blockDim.x, seq_len - k_start); k++) {
            float exp_score = expf(attention_scores[k] - max_score);
            sum_exp += exp_score;
            attention_scores[k] = exp_score;
        }
        __syncthreads();
    }
    
    // Normalize attention weights
    float inv_sum = 1.0f / sum_exp;
    
    // Compute output with value matrix
    float output_vec[64] = {0.0f};
    
    for (int v_start = 0; v_start < seq_len; v_start += blockDim.x) {
        int v_idx = v_start + tid;
        
        // Load V tile into shared memory
        if (v_idx < seq_len) {
            for (int d = 0; d < head_dim; d++) {
                shared_V[tid][d] = V[head_offset + v_idx * head_dim + d];
            }
        }
        __syncthreads();
        
        // Weighted sum with attention weights
        for (int v = 0; v < min(blockDim.x, seq_len - v_start); v++) {
            float weight = attention_scores[v] * inv_sum;
            
            #pragma unroll
            for (int d = 0; d < head_dim; d++) {
                output_vec[d] += weight * shared_V[v][d];
            }
        }
        __syncthreads();
    }
    
    // Write output (coalesced)
    for (int d = 0; d < head_dim; d++) {
        output[head_offset + seq_idx * head_dim + d] = output_vec[d];
    }
}

// Optimized softmax kernel with reduction and memory coalescing
__global__ void optimized_softmax_kernel(
    const float* input,
    float* output,
    const int batch_size,
    const int seq_len,
    const int d_model
) {
    const int batch_idx = blockIdx.y;
    const int seq_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len) return;
    
    __shared__ float shared_data[256];
    __shared__ float max_val;
    __shared__ float sum_exp;
    
    const int offset = batch_idx * seq_len * d_model + seq_idx * d_model;
    
    // Find maximum for numerical stability (parallel reduction)
    float local_max = -INFINITY;
    for (int i = tid; i < d_model; i += blockDim.x) {
        local_max = fmaxf(local_max, input[offset + i]);
    }
    
    shared_data[tid] = local_max;
    __syncthreads();
    
    // Reduction to find global max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + stride]);
        }
        __syncthreads();
    }
    
    if (tid == 0) max_val = shared_data[0];
    __syncthreads();
    
    // Compute sum of exponentials
    float local_sum = 0.0f;
    for (int i = tid; i < d_model; i += blockDim.x) {
        local_sum += expf(input[offset + i] - max_val);
    }
    
    shared_data[tid] = local_sum;
    __syncthreads();
    
    // Reduction to find global sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) sum_exp = shared_data[0];
    __syncthreads();
    
    // Normalize (softmax output)
    float inv_sum = 1.0f / sum_exp;
    for (int i = tid; i < d_model; i += blockDim.x) {
        output[offset + i] = expf(input[offset + i] - max_val) * inv_sum;
    }
}

// Host function for optimized attention computation
extern "C" {
    
void launch_optimized_attention(
    const float* Q, const float* K, const float* V,
    float* output,
    int batch_size, int seq_len, int d_model, int num_heads,
    hipStream_t stream = 0
) {
    const float scale = 1.0f / sqrtf(d_model / num_heads);
    
    // Optimal thread configuration for RDNA3/CDNA3
    dim3 block_size(64, 1, 1);  // 64 threads per block
    dim3 grid_size(
        (seq_len + block_size.x - 1) / block_size.x,
        num_heads,
        batch_size
    );
    
    // Launch optimized attention kernel
    hipLaunchKernelGGL(
        optimized_attention_kernel,
        grid_size, block_size, 
        0, stream,
        Q, K, V, output,
        batch_size, seq_len, d_model, num_heads, scale
    );
}

void launch_optimized_softmax(
    const float* input, float* output,
    int batch_size, int seq_len, int d_model,
    hipStream_t stream = 0
) {
    dim3 block_size(256, 1, 1);
    dim3 grid_size(seq_len, batch_size, 1);
    
    hipLaunchKernelGGL(
        optimized_softmax_kernel,
        grid_size, block_size,
        0, stream,
        input, output,
        batch_size, seq_len, d_model
    );
}

} // extern "C"