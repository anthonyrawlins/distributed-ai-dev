/**
 * Test suite for ROCm optimization kernels
 * Validates attention and memory optimization implementations
 */

#include <hip/hip_runtime.h>
#include <rocblas.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>

// External kernel functions
extern "C" {
    void launch_optimized_attention(
        const float* Q, const float* K, const float* V, float* output,
        int batch_size, int seq_len, int d_model, int num_heads,
        hipStream_t stream
    );
    
    void launch_optimized_softmax(
        const float* input, float* output,
        int batch_size, int seq_len, int d_model,
        hipStream_t stream
    );
}

class KernelTester {
private:
    std::mt19937 rng{42};
    std::uniform_real_distribution<float> dist{-1.0f, 1.0f};

public:
    void test_attention_kernel() {
        std::cout << "🧠 Testing Optimized Attention Kernel\n";
        std::cout << "====================================\n";
        
        // Test configuration (realistic SD dimensions)
        const int batch_size = 2;
        const int seq_len = 64;    // 8x8 patches for 64x64 image
        const int d_model = 768;   // CLIP dimension
        const int num_heads = 12;
        
        const size_t qkv_size = batch_size * seq_len * d_model;
        const size_t bytes = qkv_size * sizeof(float);
        
        // Allocate host memory
        std::vector<float> h_Q(qkv_size), h_K(qkv_size), h_V(qkv_size);
        std::vector<float> h_output(qkv_size, 0.0f);
        
        // Initialize with random data
        for (size_t i = 0; i < qkv_size; i++) {
            h_Q[i] = dist(rng);
            h_K[i] = dist(rng);
            h_V[i] = dist(rng);
        }
        
        // Allocate device memory
        float *d_Q, *d_K, *d_V, *d_output;
        hipMalloc(&d_Q, bytes);
        hipMalloc(&d_K, bytes);
        hipMalloc(&d_V, bytes);
        hipMalloc(&d_output, bytes);
        
        // Copy to device
        hipMemcpy(d_Q, h_Q.data(), bytes, hipMemcpyHostToDevice);
        hipMemcpy(d_K, h_K.data(), bytes, hipMemcpyHostToDevice);
        hipMemcpy(d_V, h_V.data(), bytes, hipMemcpyHostToDevice);
        
        // Warmup
        for (int i = 0; i < 3; i++) {
            launch_optimized_attention(
                d_Q, d_K, d_V, d_output,
                batch_size, seq_len, d_model, num_heads, 0
            );
        }
        hipDeviceSynchronize();
        
        // Benchmark
        const int num_runs = 10;
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_runs; i++) {
            launch_optimized_attention(
                d_Q, d_K, d_V, d_output,
                batch_size, seq_len, d_model, num_heads, 0
            );
        }
        hipDeviceSynchronize();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // Copy result back
        hipMemcpy(h_output.data(), d_output, bytes, hipMemcpyDeviceToHost);
        
        // Validate (basic sanity check)
        bool valid = true;
        for (size_t i = 0; i < qkv_size; i++) {
            if (std::isnan(h_output[i]) || std::isinf(h_output[i])) {
                valid = false;
                break;
            }
        }
        
        // Results
        double avg_time_ms = duration.count() / (1000.0 * num_runs);
        double flops = 2.0 * batch_size * num_heads * seq_len * seq_len * d_model / num_heads;
        double gflops = (flops / (avg_time_ms * 1e6)) * 1e9;
        
        std::cout << "Batch size: " << batch_size << ", Seq len: " << seq_len 
                  << ", D model: " << d_model << ", Heads: " << num_heads << "\n";
        std::cout << "Average time: " << avg_time_ms << " ms\n";
        std::cout << "Performance: " << gflops << " GFLOPS\n";
        std::cout << "Validation: " << (valid ? "✅ PASSED" : "❌ FAILED") << "\n\n";
        
        // Cleanup
        hipFree(d_Q); hipFree(d_K); hipFree(d_V); hipFree(d_output);
    }
    
    void test_softmax_kernel() {
        std::cout << "🔥 Testing Optimized Softmax Kernel\n";
        std::cout << "=================================\n";
        
        const int batch_size = 4;
        const int seq_len = 64;
        const int d_model = 768;
        
        const size_t data_size = batch_size * seq_len * d_model;
        const size_t bytes = data_size * sizeof(float);
        
        // Allocate and initialize
        std::vector<float> h_input(data_size), h_output(data_size);
        for (size_t i = 0; i < data_size; i++) {
            h_input[i] = dist(rng) * 10.0f;  // Larger range for softmax
        }
        
        float *d_input, *d_output;
        hipMalloc(&d_input, bytes);
        hipMalloc(&d_output, bytes);
        
        hipMemcpy(d_input, h_input.data(), bytes, hipMemcpyHostToDevice);
        
        // Warmup and benchmark
        for (int i = 0; i < 3; i++) {
            launch_optimized_softmax(d_input, d_output, batch_size, seq_len, d_model, 0);
        }
        hipDeviceSynchronize();
        
        const int num_runs = 20;
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_runs; i++) {
            launch_optimized_softmax(d_input, d_output, batch_size, seq_len, d_model, 0);
        }
        hipDeviceSynchronize();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        hipMemcpy(h_output.data(), d_output, bytes, hipMemcpyDeviceToHost);
        
        // Validate softmax properties
        bool valid = true;
        for (int b = 0; b < batch_size; b++) {
            for (int s = 0; s < seq_len; s++) {
                float sum = 0.0f;
                for (int d = 0; d < d_model; d++) {
                    int idx = b * seq_len * d_model + s * d_model + d;
                    float val = h_output[idx];
                    if (val < 0.0f || val > 1.0f || std::isnan(val)) {
                        valid = false;
                        break;
                    }
                    sum += val;
                }
                if (std::abs(sum - 1.0f) > 1e-4f) {
                    valid = false;
                    break;
                }
                if (!valid) break;
            }
            if (!valid) break;
        }
        
        double avg_time_ms = duration.count() / (1000.0 * num_runs);
        
        std::cout << "Batch size: " << batch_size << ", Seq len: " << seq_len 
                  << ", D model: " << d_model << "\n";
        std::cout << "Average time: " << avg_time_ms << " ms\n";
        std::cout << "Validation: " << (valid ? "✅ PASSED" : "❌ FAILED") << "\n\n";
        
        hipFree(d_input); hipFree(d_output);
    }
    
    void run_all_tests() {
        std::cout << "🚀 ROCm Kernel Test Suite\n";
        std::cout << "========================\n\n";
        
        // Check ROCm device
        int device_count;
        hipGetDeviceCount(&device_count);
        if (device_count == 0) {
            std::cerr << "❌ No ROCm devices found\n";
            return;
        }
        
        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, 0);
        std::cout << "🔧 Device: " << prop.name << "\n";
        std::cout << "💾 Memory: " << prop.totalGlobalMem / (1024*1024*1024) << " GB\n";
        std::cout << "⚡ Compute Units: " << prop.multiProcessorCount << "\n\n";
        
        test_attention_kernel();
        test_softmax_kernel();
        
        std::cout << "🎯 All tests completed!\n";
    }
};

int main() {
    KernelTester tester;
    tester.run_all_tests();
    return 0;
}