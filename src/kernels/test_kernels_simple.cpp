/**
 * Simplified test suite for ROCm optimization kernels
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>

// External kernel functions
extern "C" {
    void launch_attention_simplified(
        const float* Q, const float* K, const float* V, float* output,
        int batch_size, int seq_len, int d_model, int num_heads,
        hipStream_t stream
    );
    
    void launch_matmul_optimized(
        const float* A, const float* B, float* C,
        int M, int N, int K,
        hipStream_t stream
    );
}

class SimpleTester {
private:
    std::mt19937 rng{42};
    std::uniform_real_distribution<float> dist{-1.0f, 1.0f};

public:
    void test_attention_performance() {
        std::cout << "🧠 Testing Simplified Attention Kernel\n";
        std::cout << "=====================================\n";
        
        // SD-like dimensions
        const int batch_size = 1;
        const int seq_len = 64;    // 8x8 patches
        const int d_model = 512;   // Smaller for testing
        const int num_heads = 8;
        
        const size_t qkv_size = batch_size * seq_len * d_model;
        const size_t bytes = qkv_size * sizeof(float);
        
        // Host data
        std::vector<float> h_Q(qkv_size), h_K(qkv_size), h_V(qkv_size);
        std::vector<float> h_output(qkv_size, 0.0f);
        
        // Random initialization
        for (size_t i = 0; i < qkv_size; i++) {
            h_Q[i] = dist(rng);
            h_K[i] = dist(rng);
            h_V[i] = dist(rng);
        }
        
        // Device memory
        float *d_Q, *d_K, *d_V, *d_output;
        hipMalloc(&d_Q, bytes);
        hipMalloc(&d_K, bytes);
        hipMalloc(&d_V, bytes);
        hipMalloc(&d_output, bytes);
        
        hipMemcpy(d_Q, h_Q.data(), bytes, hipMemcpyHostToDevice);
        hipMemcpy(d_K, h_K.data(), bytes, hipMemcpyHostToDevice);
        hipMemcpy(d_V, h_V.data(), bytes, hipMemcpyHostToDevice);
        
        // Warmup
        for (int i = 0; i < 3; i++) {
            launch_attention_simplified(
                d_Q, d_K, d_V, d_output,
                batch_size, seq_len, d_model, num_heads, 0
            );
        }
        hipDeviceSynchronize();
        
        // Benchmark
        const int num_runs = 5;
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_runs; i++) {
            launch_attention_simplified(
                d_Q, d_K, d_V, d_output,
                batch_size, seq_len, d_model, num_heads, 0
            );
        }
        hipDeviceSynchronize();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        hipMemcpy(h_output.data(), d_output, bytes, hipMemcpyDeviceToHost);
        
        // Validation
        bool valid = true;
        for (size_t i = 0; i < qkv_size && i < 100; i++) {
            if (std::isnan(h_output[i]) || std::isinf(h_output[i])) {
                valid = false;
                break;
            }
        }
        
        double avg_time_ms = duration.count() / (1000.0 * num_runs);
        
        std::cout << "Configuration: " << batch_size << "x" << seq_len 
                  << "x" << d_model << ", " << num_heads << " heads\n";
        std::cout << "Average time: " << avg_time_ms << " ms\n";
        std::cout << "Validation: " << (valid ? "✅ PASSED" : "❌ FAILED") << "\n\n";
        
        hipFree(d_Q); hipFree(d_K); hipFree(d_V); hipFree(d_output);
    }
    
    void test_matmul_performance() {
        std::cout << "⚡ Testing Optimized Matrix Multiplication\n";
        std::cout << "========================================\n";
        
        const int M = 512, N = 512, K = 512;
        const size_t size_A = M * K;
        const size_t size_B = K * N;
        const size_t size_C = M * N;
        
        std::vector<float> h_A(size_A), h_B(size_B), h_C(size_C, 0.0f);
        
        for (size_t i = 0; i < size_A; i++) h_A[i] = dist(rng);
        for (size_t i = 0; i < size_B; i++) h_B[i] = dist(rng);
        
        float *d_A, *d_B, *d_C;
        hipMalloc(&d_A, size_A * sizeof(float));
        hipMalloc(&d_B, size_B * sizeof(float));
        hipMalloc(&d_C, size_C * sizeof(float));
        
        hipMemcpy(d_A, h_A.data(), size_A * sizeof(float), hipMemcpyHostToDevice);
        hipMemcpy(d_B, h_B.data(), size_B * sizeof(float), hipMemcpyHostToDevice);
        
        // Warmup
        for (int i = 0; i < 3; i++) {
            launch_matmul_optimized(d_A, d_B, d_C, M, N, K, 0);
        }
        hipDeviceSynchronize();
        
        // Benchmark
        const int num_runs = 10;
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_runs; i++) {
            launch_matmul_optimized(d_A, d_B, d_C, M, N, K, 0);
        }
        hipDeviceSynchronize();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        hipMemcpy(h_C.data(), d_C, size_C * sizeof(float), hipMemcpyDeviceToHost);
        
        // Simple validation
        bool valid = !std::isnan(h_C[0]) && !std::isinf(h_C[0]);
        
        double avg_time_ms = duration.count() / (1000.0 * num_runs);
        double flops = 2.0 * M * N * K;
        double gflops = (flops / (avg_time_ms * 1e6)) * 1e9;
        
        std::cout << "Matrix size: " << M << "x" << N << "x" << K << "\n";
        std::cout << "Average time: " << avg_time_ms << " ms\n";
        std::cout << "Performance: " << gflops << " GFLOPS\n";
        std::cout << "Validation: " << (valid ? "✅ PASSED" : "❌ FAILED") << "\n\n";
        
        hipFree(d_A); hipFree(d_B); hipFree(d_C);
    }
    
    void run_all_tests() {
        std::cout << "🚀 ROCm Kernel Performance Test\n";
        std::cout << "==============================\n\n";
        
        int device_count;
        hipGetDeviceCount(&device_count);
        if (device_count == 0) {
            std::cerr << "❌ No ROCm devices found\n";
            return;
        }
        
        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, 0);
        std::cout << "🔧 Device: " << prop.name << "\n";
        std::cout << "💾 Memory: " << prop.totalGlobalMem / (1024*1024*1024) << " GB\n\n";
        
        test_attention_performance();
        test_matmul_performance();
        
        std::cout << "🎯 Performance tests completed!\n";
    }
};

int main() {
    SimpleTester tester;
    tester.run_all_tests();
    return 0;
}