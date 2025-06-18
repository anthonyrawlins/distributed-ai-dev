#!/bin/bash
# ROCm Development Environment Setup Script
# Based on Agent 113's analysis and daily contribution plan

set -e

echo "🚀 Setting up ROCm Development Environment"
echo "Focus: Stable Diffusion optimization on AMD GPUs"
echo "================================================"

# Check if running on supported system
if [[ ! -f /etc/os-release ]]; then
    echo "❌ Cannot detect OS. This script supports Ubuntu/Debian systems."
    exit 1
fi

source /etc/os-release
echo "📋 Detected OS: $NAME $VERSION_ID"

# Verify ROCm installation
echo "🔍 Checking existing ROCm installation..."
if command -v rocm-smi &> /dev/null; then
    echo "✅ ROCm detected: $(rocm-smi --version | head -1)"
    ROCM_VERSION=$(rocm-smi --version | head -1 | grep -oP '\d+\.\d+\.\d+')
else
    echo "⚠️  ROCm not found. Install ROCm first:"
    echo "   https://docs.amd.com/bundle/ROCm-Installation-Guide-v5.7/page/How_to_Install_ROCm.html"
fi

# Create development directory structure
echo "📁 Creating development directories..."
mkdir -p ~/AI/ROCm/repositories
mkdir -p ~/AI/ROCm/benchmarks  
mkdir -p ~/AI/ROCm/optimizations
mkdir -p ~/AI/ROCm/tests

# Install development dependencies
echo "📦 Installing development dependencies..."
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    git \
    python3-pip \
    python3-venv \
    rocm-dev-tools \
    rocprofiler-dev \
    roctracer-dev \
    hip-dev

# Setup Python environment
echo "🐍 Setting up Python environment..."
cd ~/AI/ROCm
if [[ ! -d "venv" ]]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Install PyTorch with ROCm support
echo "🔥 Installing PyTorch with ROCm support..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7

# Install additional ML dependencies
pip install \
    transformers \
    diffusers \
    accelerate \
    huggingface-hub \
    numpy \
    scipy \
    matplotlib \
    jupyter

# Verification tests
echo "🧪 Running verification tests..."

# Test 1: ROCm basic functionality
echo "Test 1: ROCm device detection"
if rocm-smi -d 0 &> /dev/null; then
    echo "✅ ROCm device detected"
    rocm-smi --showproductname
else
    echo "❌ No ROCm devices found"
fi

# Test 2: PyTorch ROCm support
echo "Test 2: PyTorch ROCm support"
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
if torch.cuda.is_available():
    print(f'✅ CUDA backend available: {torch.cuda.device_count()} devices')
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('❌ CUDA backend not available')
"

# Test 3: Simple tensor operation
echo "Test 3: GPU tensor operations"
python3 -c "
import torch
try:
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print('✅ GPU matrix multiplication successful')
    print(f'Result shape: {z.shape}')
except Exception as e:
    print(f'❌ GPU operation failed: {e}')
"

# Setup benchmark script
echo "📊 Creating benchmark script..."
cat > ~/AI/ROCm/benchmarks/sd_benchmark.py << 'EOF'
#!/usr/bin/env python3
"""
Simple Stable Diffusion benchmark for ROCm optimization
"""
import torch
import time
from diffusers import StableDiffusionPipeline

def benchmark_sd_inference():
    print("🎯 Stable Diffusion ROCm Benchmark")
    print("="*40)
    
    # Check device
    if not torch.cuda.is_available():
        print("❌ CUDA/ROCm not available")
        return
        
    device = "cuda"
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load model
    print("📥 Loading Stable Diffusion model...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to(device)
    
    # Warmup
    print("🔥 Warming up...")
    _ = pipe("test", num_inference_steps=1)
    
    # Benchmark
    prompt = "A beautiful landscape with mountains and rivers"
    num_steps = 20
    
    print(f"🚀 Running benchmark: {num_steps} steps")
    torch.cuda.synchronize()
    start_time = time.time()
    
    image = pipe(prompt, num_inference_steps=num_steps).images[0]
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    duration = end_time - start_time
    steps_per_sec = num_steps / duration
    
    print(f"✅ Benchmark Results:")
    print(f"   Duration: {duration:.2f}s")
    print(f"   Steps/sec: {steps_per_sec:.2f}")
    print(f"   Image size: {image.size}")
    
    return duration, steps_per_sec

if __name__ == "__main__":
    benchmark_sd_inference()
EOF

chmod +x ~/AI/ROCm/benchmarks/sd_benchmark.py

# Create optimization workspace
echo "🔧 Setting up optimization workspace..."
cat > ~/AI/ROCm/optimizations/README.md << 'EOF'
# ROCm Optimization Workspace

## Agent 113's Analysis Results

### Top 3 Optimization Priorities:
1. **Attention Mechanism**: Matrix multiplication efficiency, softmax parallelization
2. **Memory Access Patterns**: Coalesced access, shared memory utilization  
3. **VAE Decoder**: FFT-based convolutions, custom upsampling kernels

### Implementation Status:
- [x] Performance analysis complete
- [x] Optimization targets identified
- [x] Development environment setup
- [ ] Attention mechanism optimization
- [ ] Memory access optimization
- [ ] VAE decoder optimization

## Next Steps:
1. Implement Agent 113's attention optimization design
2. Test memory optimization examples
3. Benchmark optimizations against baseline
4. Begin VAE decoder optimization phase
EOF

echo ""
echo "🎉 ROCm Development Environment Setup Complete!"
echo "================================================"
echo "📂 Workspace: ~/AI/ROCm/"
echo "🐍 Python environment: source ~/AI/ROCm/venv/bin/activate"
echo "📊 Run benchmark: python3 ~/AI/ROCm/benchmarks/sd_benchmark.py"
echo "🔧 Optimization workspace: ~/AI/ROCm/optimizations/"
echo ""
echo "🚀 Ready for ROCm Stable Diffusion optimization work!"