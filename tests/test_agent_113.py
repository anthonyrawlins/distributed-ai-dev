#!/usr/bin/env python3
"""
Test script for Agent 113 (DevStral)
Validates the new agent with ROCm-specific tasks
"""

import asyncio
import aiohttp
import json
import time

class Agent113Tester:
    def __init__(self):
        self.endpoint = "http://192.168.1.113:11434"
        self.model = "devstral:latest"
        
    async def test_basic_connectivity(self):
        """Test basic connectivity and model response"""
        print("Testing basic connectivity...")
        
        payload = {
            "model": self.model,
            "prompt": "Hello! Please confirm you're ready to help with ROCm development.",
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 50}
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.endpoint}/api/generate", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✓ Basic connectivity: {result['response'][:100]}...")
                    return True
                else:
                    print(f"✗ Connectivity failed: {response.status}")
                    return False
    
    async def test_rocm_kernel_optimization(self):
        """Test DevStral's ability to optimize ROCm kernels"""
        print("\nTesting ROCm kernel optimization capabilities...")
        
        prompt = """You are an expert GPU kernel developer specializing in AMD ROCm/HIP. 
Your task is to optimize this attention kernel for RDNA3 architecture:

```cpp
__global__ void naive_attention(float* Q, float* K, float* V, float* output, 
                               int batch_size, int seq_len, int head_dim) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= seq_len) return;
    
    for (int i = 0; i < seq_len; i++) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += Q[tid * head_dim + d] * K[i * head_dim + d];
        }
        // Apply attention weights...
        for (int d = 0; d < head_dim; d++) {
            output[tid * head_dim + d] += score * V[i * head_dim + d];
        }
    }
}
```

Please provide an optimized version focusing on:
1. Memory coalescing for RDNA3
2. Shared memory utilization  
3. Loop unrolling where beneficial
4. Wavefront-aware optimizations

Format your response as JSON with 'optimized_code', 'explanation', and 'performance_notes' fields."""

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 2000
            }
        }
        
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.endpoint}/api/generate", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    elapsed = time.time() - start_time
                    
                    try:
                        # Try to parse as JSON response
                        if result['response'].strip().startswith('{'):
                            parsed = json.loads(result['response'])
                            print("✓ DevStral provided structured optimization:")
                            print(f"  - Response time: {elapsed:.1f}s")
                            print(f"  - Has optimized_code: {'optimized_code' in parsed}")
                            print(f"  - Has explanation: {'explanation' in parsed}")
                            print(f"  - Has performance_notes: {'performance_notes' in parsed}")
                            return True
                        else:
                            print("✓ DevStral provided optimization (non-JSON format):")
                            print(f"  - Response time: {elapsed:.1f}s")
                            print(f"  - Response length: {len(result['response'])} chars")
                            print(f"  - Contains 'shared': {'shared' in result['response'].lower()}")
                            print(f"  - Contains 'coalesce': {'coalesce' in result['response'].lower()}")
                            return True
                    except json.JSONDecodeError:
                        print("✓ DevStral provided optimization (text format):")
                        print(f"  - Response time: {elapsed:.1f}s") 
                        print(f"  - Contains optimization keywords: {any(word in result['response'].lower() for word in ['optimize', 'performance', 'memory'])}")
                        return True
                else:
                    print(f"✗ Kernel optimization test failed: {response.status}")
                    return False
    
    async def test_pytorch_integration_task(self):
        """Test DevStral's PyTorch integration capabilities"""
        print("\nTesting PyTorch integration capabilities...")
        
        prompt = """You are a PyTorch expert specializing in ROCm backend integration.
Create a PyTorch wrapper for an optimized ROCm attention function that:

1. Integrates with torch.nn.functional
2. Supports autograd (backward pass)
3. Has proper device management for ROCm
4. Includes error handling and validation

Provide Python code that follows PyTorch conventions and maintains API compatibility.
Format as JSON with 'code', 'tests', and 'integration_notes' fields."""

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 1500
            }
        }
        
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.endpoint}/api/generate", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    elapsed = time.time() - start_time
                    
                    print("✓ DevStral provided PyTorch integration:")
                    print(f"  - Response time: {elapsed:.1f}s")
                    print(f"  - Contains 'torch': {'torch' in result['response']}")
                    print(f"  - Contains 'autograd': {'autograd' in result['response'].lower()}")
                    print(f"  - Contains 'rocm': {'rocm' in result['response'].lower()}")
                    return True
                else:
                    print(f"✗ PyTorch integration test failed: {response.status}")
                    return False
    
    async def test_model_switching(self):
        """Test switching between different models on the same agent"""
        print("\nTesting model switching capabilities...")
        
        models_to_test = ['devstral:latest', 'deepseek-r1:7b', 'phi4:latest']
        results = {}
        
        for model in models_to_test:
            prompt = f"You are using {model}. Briefly describe your strengths for ROCm development."
            
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 100}
            }
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(f"{self.endpoint}/api/generate", json=payload) as response:
                        if response.status == 200:
                            result = await response.json()
                            results[model] = True
                            print(f"  ✓ {model}: Available")
                        else:
                            results[model] = False
                            print(f"  ✗ {model}: Failed ({response.status})")
            except Exception as e:
                results[model] = False
                print(f"  ✗ {model}: Error ({str(e)})")
        
        available_models = sum(results.values())
        print(f"✓ Model switching test: {available_models}/{len(models_to_test)} models available")
        return available_models > 0

async def main():
    """Run comprehensive Agent 113 testing"""
    print("Agent 113 (DevStral) - Comprehensive Testing")
    print("=" * 50)
    
    tester = Agent113Tester()
    test_results = []
    
    # Run all tests
    tests = [
        ("Basic Connectivity", tester.test_basic_connectivity),
        ("ROCm Kernel Optimization", tester.test_rocm_kernel_optimization),
        ("PyTorch Integration", tester.test_pytorch_integration_task),
        ("Model Switching", tester.test_model_switching)
    ]
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name}: Exception - {str(e)}")
            test_results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 50)
    print("AGENT 113 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, success in test_results if success)
    total = len(test_results)
    
    print(f"Tests Passed: {passed}/{total} ({passed/total:.1%})")
    print("\nDetailed Results:")
    
    for test_name, success in test_results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status} {test_name}")
    
    if passed == total:
        print(f"\n🎉 Agent 113 is READY FOR PRODUCTION!")
        print("Recommended as: Senior DevStral Architect")
        print("Specialization: Complex kernel development and architecture")
    else:
        print(f"\n⚠️  Agent 113 needs attention on failed tests")
    
    print("=" * 50)
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)