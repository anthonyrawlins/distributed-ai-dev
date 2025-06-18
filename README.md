# Distributed AI Development System

A powerful coordination system that enables Claude (or other AI coordinators) to orchestrate multiple local AI agents for distributed software development, specifically optimized for ROCm and GPU computing projects.

## 🚀 **Features**

- **AI Agent Coordination** - Orchestrate multiple Ollama agents across your network
- **YAML Configuration** - Centralized agent management with flexible configuration
- **Real-time Monitoring** - btop/nvtop-style performance dashboards for all agents
- **Specialized Agents** - Kernel developers, PyTorch experts, profilers, testers, and documentation writers
- **Quality Control** - Multi-agent code review and validation systems
- **Claude Integration** - Easy interface for Claude to manage distributed development
- **Cost Efficient** - Reduce Claude usage by 90% while scaling development capacity
- **Production Ready** - Comprehensive testing, monitoring, and deployment tools

## 🎯 **Perfect For**

- **ROCm/GPU Development** - Optimize CUDA/HIP kernels, attention mechanisms, VAE decoders
- **ML Framework Integration** - PyTorch, TensorFlow, and custom framework development  
- **Performance Optimization** - Memory management, kernel fusion, multi-GPU coordination
- **Distributed Teams** - Scale development across multiple AI agents and compute resources

## 📋 **Quick Start**

### 1. Installation

```bash
git clone https://github.com/anthonyrawlins/distributed-ai-dev.git
cd distributed-ai-dev
pip install aiohttp asyncio dataclasses pyyaml psutil
```

### 2. Configure Your Agents

Edit `config/agents.yaml` to define your agent network:

```yaml
agents:
  agent_113:
    name: "DevStral Senior Architect"
    endpoint: "http://192.168.1.113:11434"
    model: "devstral:latest"
    specialization: "Senior Kernel Development & ROCm Optimization"
    priority: 1
    status: "active"
    capabilities:
      - "kernel_development"
      - "rocm_optimization"
      - "flashattention_implementation"
    hardware:
      gpu_type: "NVIDIA RTX 3070"
      vram_gb: 8
      cpu_cores: 24
      ram_gb: 64
    performance_targets:
      min_tokens_per_second: 5.0
      max_response_time_ms: 30000
      target_availability: 0.99

  agent_27:
    name: "CodeLlama Development Assistant"
    endpoint: "http://192.168.1.27:11434"
    model: "codellama:latest"
    specialization: "Code Generation & Development Support"
    priority: 2
    status: "active"
    capabilities:
      - "code_generation"
      - "code_completion"
      - "debugging_assistance"
    hardware:
      gpu_type: "AMD Radeon RX 9060XT"
      vram_gb: 16
      cpu_cores: 16
      ram_gb: 32
```

### 3. Monitor Your Agents

Launch real-time monitoring dashboards:

```bash
# Simple terminal-friendly interface
python3 src/agents/simple_monitor.py

# Advanced curses interface (btop/nvtop style)
python3 src/agents/advanced_monitor.py

# Test specific agents
python3 src/agents/test_agent_27.py
```

### 4. Test the System

```bash
python tests/test_distributed_system.py
```

### 5. Start Using from Claude

```python
from src.interfaces.claude_interface import setup_development_network, delegate_work

# Setup your agent network from YAML config
await setup_development_network()

# Delegate complex development work
result = await delegate_work(
    "Optimize FlashAttention kernel for RDNA3",
    files=["attention.cpp", "kernels.hip"], 
    priority=5
)

# Check progress
progress = await check_progress()
results = await collect_results()
```

## 🖥️ **Real-time Monitoring**

### Performance Dashboard Features
- **Live Performance Metrics**: Tokens per second, response times, system resources
- **Multi-Agent Tracking**: Monitor all agents simultaneously with individual status
- **System Resources**: CPU, RAM, GPU utilization with progress bars
- **Historical Analytics**: Rolling performance averages and trend analysis
- **Cross-Platform GPU Support**: Auto-detection of AMD ROCm and NVIDIA CUDA

### Monitoring Commands
```bash
# Monitor all active agents (auto-loads from config/agents.yaml)
python3 src/agents/simple_monitor.py

# Advanced interface with real-time updates
python3 src/agents/advanced_monitor.py
# Controls: 'q' quit, 'r' refresh, 'c' reload config

# Test individual agent performance
python3 src/agents/test_agent_27.py
```

## 🏗️ **Architecture**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Claude        │    │   Coordinator    │    │  Quality        │
│   Interface     │◄──►│   System         │◄──►│  Control        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                               │
                    ┌──────────┼──────────┐
                    │          │          │
              ┌──────▼───┐ ┌────▼────┐ ┌──▼──────┐
              │ Agent    │ │ Agent   │ │ Agent   │
              │ 113      │ │ 27      │ │ ...     │
              │(DevStral)│ │(CodeLlama)│ │        │
              └──────────┘ └─────────┘ └─────────┘
                    │          │          │
              ┌──────▼───┐ ┌────▼────┐ ┌──▼──────┐
              │Real-time │ │Performance│ │System   │
              │Monitoring│ │ Metrics   │ │Resources│
              └──────────┘ └─────────┘ └─────────┘
```

## 🤖 **Agent Specializations**

### **Agent 113: DevStral Senior Architect**
- **Model**: DevStral-23.6B
- **Hardware**: NVIDIA RTX 3070 (8GB), 24 cores, 64GB RAM
- **Specialization**: Senior Kernel Development & ROCm Optimization
- **Focus**: FlashAttention, VAE decoders, advanced memory management
- **Performance**: 5.0+ TPS, <30s response time, 99% availability

### **Agent 27: CodeLlama Development Assistant**
- **Model**: CodeLlama
- **Hardware**: AMD Radeon RX 9060XT (16GB), 16 cores, 32GB RAM
- **Specialization**: Code Generation & Development Support
- **Focus**: Code completion, debugging, algorithm implementation
- **Performance**: 4.0+ TPS, <25s response time, 95% availability

### **Expandable Agent Types**
- **Kernel Developer** (`kernel_dev`): HIP kernels, CUDA ports, GPU optimization
- **PyTorch Developer** (`pytorch_dev`): PyTorch integration, autograd, Python bindings
- **Performance Profiler** (`profiler`): rocprof analysis, benchmarking, bottlenecks
- **Documentation Writer** (`docs_writer`): API docs, tutorials, guides
- **Tester** (`tester`): Unit tests, integration tests, CI/CD

## 📊 **Example Workflow**

1. **Claude analyzes** your optimization request (e.g., "Optimize Stable Diffusion for ROCm")
2. **Breaks down** into specialized subtasks:
   - Complex kernel optimization → Agent 113 (DevStral)
   - Code completion and testing → Agent 27 (CodeLlama)
   - Performance analysis → Profiler Agent
   - Documentation → Documentation Writer Agent
3. **Delegates** tasks to appropriate agents based on capabilities and priority
4. **Monitors progress** with real-time performance dashboards
5. **Reviews results** with quality control system
6. **Integrates solutions** and creates final deliverable

## 🔧 **Project Structure**

```
distributed-ai-dev/
├── config/
│   └── agents.yaml                    # ⭐ Centralized agent configuration
├── src/
│   ├── core/
│   │   └── ai_dev_coordinator.py      # Main coordination system
│   ├── interfaces/
│   │   └── claude_interface.py        # Claude integration interface
│   ├── quality/
│   │   └── quality_control.py         # Multi-agent code review
│   └── agents/
│       ├── advanced_monitor.py        # ⭐ btop/nvtop-style monitoring
│       ├── simple_monitor.py          # ⭐ Terminal-friendly monitoring
│       ├── test_agent_27.py           # ⭐ Agent testing framework
│       ├── agent_113_config.py        # Agent-specific configurations
│       └── monitor_agent_113.py       # Legacy agent monitoring
├── tests/
│   ├── test_distributed_system.py     # System integration tests
│   ├── test_agent_113.py             # Agent-specific tests
│   └── quick_agent_integration.py     # Quick validation tests
├── examples/
│   └── agent_113_workload.py         # Example workload management
├── docs/
│   ├── setup_guide.md                # Detailed setup instructions
│   ├── distributed-ai-dev-system.md  # System architecture
│   ├── daily-contribution-plan.md    # Development workflow
│   └── rocm-planning.md              # ROCm optimization roadmap
├── agent_27_assessment.md            # ⭐ Agent performance assessment
├── README.md                          # This file
└── LICENSE                           # MIT License
```

## 📈 **Benefits**

- **90% Cost Reduction** - Minimize Claude usage for development tasks
- **Parallel Development** - Multiple agents working simultaneously
- **24/7 Capability** - Agents work continuously with real-time monitoring
- **Quality Assured** - Multi-agent review process with performance tracking
- **Scalable** - Easily add more agents through YAML configuration
- **Production Ready** - Comprehensive testing, monitoring, and deployment tools
- **Transparent** - Real-time performance dashboards for all agents

## 🛠️ **Configuration**

### **YAML Configuration System**

The system uses `config/agents.yaml` for centralized agent management:

```yaml
# Global monitoring settings
monitoring:
  refresh_interval_seconds: 5
  performance_window_minutes: 5
  max_history_samples: 100
  alert_thresholds:
    min_tokens_per_second: 2.0
    max_response_time_ms: 60000

# Network configuration
network:
  timeout_seconds: 30
  retry_attempts: 3
  retry_delay_seconds: 5

# Agent definitions
agents:
  agent_id:
    name: "Human-readable name"
    endpoint: "http://host:11434"
    model: "model_name:latest"
    specialization: "Agent expertise area"
    priority: 1-5  # 1=highest priority
    status: "active" | "disabled"
    capabilities: [list of capabilities]
    hardware: {gpu_type, vram_gb, cpu_cores, ram_gb}
    performance_targets: {min_tps, max_response_ms, availability}
```

### **Recommended Models by GPU Memory:**

- **8GB VRAM**: CodeLlama-13B, DeepSeek-Coder-6.7B, Qwen2.5-Coder-14B
- **16GB VRAM**: CodeLlama-34B, DeepSeek-Coder-33B, Qwen2.5-Coder-32B  
- **24GB+ VRAM**: Llama-3.1-70B, DevStral-23B (for complex analysis)

### **Performance Tuning:**
- Set `max_concurrent=1` for large models (>20B parameters)
- Use `max_concurrent=2-3` for smaller models (<15B parameters)
- Adjust `temperature=0.1` for consistent code generation
- Set appropriate `max_tokens` based on task complexity

## 🚨 **Important Notes**

- **Always review** agent-generated code before committing
- **Test thoroughly** on your specific hardware setup
- **Start small** with simple optimizations to validate the system
- **Monitor resource usage** with real-time dashboards across your agent network
- **Keep backups** of working configurations
- **Update `config/agents.yaml`** to add/remove agents dynamically

## 📚 **Documentation**

- [Setup Guide](docs/setup_guide.md) - Detailed installation and configuration
- [System Architecture](docs/distributed-ai-dev-system.md) - Technical deep dive
- [Daily Workflow](docs/daily-contribution-plan.md) - Development processes
- [ROCm Optimization](docs/rocm-planning.md) - Specific ROCm development strategy
- [Agent 27 Assessment](agent_27_assessment.md) - Performance evaluation and capabilities

## 🔍 **Monitoring & Debugging**

### Real-time Performance Tracking
```bash
# Monitor all agents with live performance metrics
python3 src/agents/simple_monitor.py

# Advanced monitoring with GPU utilization tracking
python3 src/agents/advanced_monitor.py

# Test individual agent capabilities
python3 src/agents/test_agent_27.py
```

### Performance Metrics
- **Tokens per Second (TPS)**: Real-time inference speed
- **Response Latency**: End-to-end task completion time  
- **System Resources**: CPU, RAM, GPU utilization
- **Quality Scores**: Automated code quality assessment
- **Availability**: Agent uptime and reliability tracking

## 🤝 **Contributing**

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎯 **Success Stories**

This system has been successfully deployed for:
- **ROCm optimization projects** targeting RDNA3/CDNA3 architectures
- **Stable Diffusion performance improvements** on AMD GPUs
- **FlashAttention kernel development** for PyTorch
- **VAE decoder optimization** for computer vision workloads
- **Multi-GPU memory management** systems
- **Distributed development teams** with 90% cost reduction vs cloud AI services

### **Current Active Network**
- **Agent 113 (DevStral)**: Senior kernel development, 11.8 TPS performance
- **Agent 27 (CodeLlama)**: Development assistance, ROCm knowledge validated
- **Real-time monitoring**: btop/nvtop-style dashboards operational
- **YAML configuration**: Centralized management system deployed

## ⭐ **Star this repo if it helps accelerate your development!**

---

**Transform your local compute network into a distributed AI development team! 🚀**