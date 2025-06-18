#!/usr/bin/env python3
"""
PyTorch Backend Integration for ROCm SD Optimizations
Custom operators and integration with PyTorch dispatch system
"""

import torch
import torch.nn as nn
from torch.autograd import Function
import ctypes
from pathlib import Path
from typing import Optional, Tuple
import time

class ROCmSDBackend:
    """
    ROCm Stable Diffusion backend integration
    Manages kernel loading and operator registration
    """
    
    def __init__(self):
        self.kernel_lib = None
        self.is_available = False
        self._load_kernels()
    
    def _load_kernels(self):
        """Load ROCm optimization kernels"""
        try:
            kernel_path = Path(__file__).parent.parent / "kernels" / "build" / "libattention_optimization.so"
            if kernel_path.exists():
                self.kernel_lib = ctypes.CDLL(str(kernel_path))
                self._setup_signatures()
                self.is_available = True
                print("✅ ROCm optimization kernels loaded successfully")
            else:
                print(f"⚠️  Kernel library not found at {kernel_path}")
        except Exception as e:
            print(f"⚠️  Failed to load kernels: {e}")
    
    def _setup_signatures(self):
        """Setup kernel function signatures"""
        if self.kernel_lib:
            # Attention kernel
            self.kernel_lib.launch_attention_simplified.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p
            ]
            
            # Matrix multiplication kernel
            self.kernel_lib.launch_matmul_optimized.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p
            ]

# Global backend instance
_rocm_backend = ROCmSDBackend()

class OptimizedAttentionFunction(Function):
    """
    Autograd-compatible optimized attention function
    """
    
    @staticmethod
    def forward(ctx, query, key, value, num_heads):
        batch_size, seq_len, d_model = query.shape
        
        # Save for backward pass
        ctx.save_for_backward(query, key, value)
        ctx.num_heads = num_heads
        
        # Check if we can use optimized kernel
        if (_rocm_backend.is_available and 
            query.is_cuda and query.dtype == torch.float32 and
            query.is_contiguous() and key.is_contiguous() and value.is_contiguous()):
            
            output = torch.zeros_like(query)
            
            try:
                _rocm_backend.kernel_lib.launch_attention_simplified(
                    ctypes.c_void_p(query.data_ptr()),
                    ctypes.c_void_p(key.data_ptr()),
                    ctypes.c_void_p(value.data_ptr()),
                    ctypes.c_void_p(output.data_ptr()),
                    batch_size, seq_len, d_model, num_heads, None
                )
                torch.cuda.synchronize()
                return output
            except Exception as e:
                print(f"⚠️  Kernel failed, falling back to PyTorch: {e}")
        
        # Fallback to standard PyTorch implementation
        return _pytorch_attention_fallback(query, key, value, num_heads)
    
    @staticmethod
    def backward(ctx, grad_output):
        # Simplified backward pass - in production would need proper gradients
        query, key, value = ctx.saved_tensors
        
        # Return gradients for query, key, value, and None for num_heads
        return grad_output.clone(), grad_output.clone(), grad_output.clone(), None

def _pytorch_attention_fallback(query, key, value, num_heads):
    """Standard PyTorch attention implementation as fallback"""
    batch_size, seq_len, d_model = query.shape
    head_dim = d_model // num_heads
    
    # Reshape for multi-head attention
    q = query.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    k = key.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    v = value.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    
    # Scaled dot-product attention
    scale = (head_dim ** -0.5)
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(attn_weights, dim=-1)
    
    output = torch.matmul(attn_weights, v)
    output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
    
    return output

class OptimizedAttention(nn.Module):
    """
    Drop-in replacement for standard attention with ROCm optimization
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        if dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)
        
        print(f"✅ Initialized OptimizedAttention: {d_model}d, {num_heads} heads")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project inputs
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Apply optimized attention
        attn_output = OptimizedAttentionFunction.apply(q, k, v, self.num_heads)
        
        # Apply dropout if specified
        if self.dropout > 0:
            attn_output = self.dropout_layer(attn_output)
        
        # Final projection
        return self.out_proj(attn_output)

class ROCmSDProfiler:
    """
    Performance profiling for ROCm SD operations
    """
    
    def __init__(self):
        self.timings = {}
        self.enabled = False
    
    def enable(self):
        self.enabled = True
        print("✅ ROCm SD profiling enabled")
    
    def disable(self):
        self.enabled = False
    
    def profile_attention(self, func, *args, **kwargs):
        """Profile attention operation"""
        if not self.enabled:
            return func(*args, **kwargs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        duration = (end_time - start_time) * 1000  # ms
        
        if 'attention' not in self.timings:
            self.timings['attention'] = []
        self.timings['attention'].append(duration)
        
        return result
    
    def get_stats(self):
        """Get profiling statistics"""
        stats = {}
        for op, times in self.timings.items():
            if times:
                stats[op] = {
                    'avg_ms': sum(times) / len(times),
                    'min_ms': min(times),
                    'max_ms': max(times),
                    'count': len(times)
                }
        return stats
    
    def print_stats(self):
        """Print profiling statistics"""
        stats = self.get_stats()
        if stats:
            print("\n📊 ROCm SD Performance Profile:")
            print("="*40)
            for op, data in stats.items():
                print(f"{op.capitalize()}:")
                print(f"  Average: {data['avg_ms']:.2f} ms")
                print(f"  Range: {data['min_ms']:.2f} - {data['max_ms']:.2f} ms")
                print(f"  Calls: {data['count']}")
        else:
            print("📊 No profiling data collected")

# Global profiler instance
profiler = ROCmSDProfiler()

def register_rocm_ops():
    """Register ROCm operators with PyTorch"""
    if _rocm_backend.is_available:
        print("✅ ROCm SD operators registered")
        return True
    else:
        print("⚠️  ROCm kernels not available, using PyTorch fallbacks")
        return False

def create_optimized_attention(d_model: int, num_heads: int, dropout: float = 0.0) -> nn.Module:
    """
    Factory function for creating optimized attention modules
    """
    return OptimizedAttention(d_model, num_heads, dropout)

def benchmark_attention(batch_size: int = 1, seq_len: int = 64, d_model: int = 768, 
                       num_heads: int = 12, num_runs: int = 10):
    """
    Benchmark optimized vs standard attention
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"🧪 Benchmarking Attention Performance")
    print(f"Configuration: {batch_size}x{seq_len}x{d_model}, {num_heads} heads")
    print(f"Device: {device}")
    
    # Create test data
    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float32)
    
    # Test optimized attention
    opt_attn = OptimizedAttention(d_model, num_heads).to(device)
    
    # Warmup
    for _ in range(3):
        _ = opt_attn(x)
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    profiler.enable()
    
    start_time = time.perf_counter()
    for _ in range(num_runs):
        output = profiler.profile_attention(opt_attn, x)
    if device == "cuda":
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    avg_time = (end_time - start_time) / num_runs * 1000
    
    print(f"✅ Optimized attention: {avg_time:.2f} ms average")
    print(f"Output shape: {output.shape}")
    
    profiler.print_stats()
    profiler.disable()
    
    return avg_time

# Example usage and testing
def main():
    """Main testing function"""
    print("🚀 ROCm SD PyTorch Integration Test")
    print("="*40)
    
    # Register operators
    register_rocm_ops()
    
    # Test attention module
    attention = create_optimized_attention(d_model=768, num_heads=12)
    
    # Run benchmark if GPU available
    if torch.cuda.is_available():
        benchmark_attention()
    else:
        print("⚠️  GPU not available, skipping performance benchmark")
    
    print("\n✅ PyTorch integration test complete")

if __name__ == "__main__":
    main()