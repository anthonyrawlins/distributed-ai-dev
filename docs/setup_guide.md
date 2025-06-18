# Distributed AI Development System - Setup Guide

## 🎯 System Overview

You now have a complete distributed AI development coordination system that allows Claude to orchestrate multiple local Ollama agents working on ROCm optimizations. This multiplies your development capacity while minimizing Claude usage costs.

The system uses environment variables and YAML configuration for flexible deployment on NAS/NFS shared storage, with organized folder structure for better maintainability.

## 📁 Project Structure

```
/home/tony/AI/ROCm/distributed-ai-dev/
├── config/
│   ├── agents.yaml              # Single source of truth for agent configuration
│   └── auto-agent-analysis/     # Agent capability analysis
├── orchestration/               # Agent management and coordination hub
│   ├── claude_interface.py      # Main coordination interface
│   ├── ai_dev_coordinator.py    # Development coordinator
│   ├── monitor_agent_113.py     # Agent monitoring tools
│   └── results/                 # JSON result files from operations
├── reports/                     # Performance reports and analysis
├── tests/                       # All test files and integration tests
├── src/
│   ├── agents/                  # Agent monitoring applications
│   ├── core/                    # Configuration and core utilities
│   ├── interfaces/              # API interfaces
│   └── kernels/                 # ROCm kernel development
├── docs/                        # Documentation and agent summaries
├── bin/                         # Executable launcher scripts
├── shared/                      # Shared workspace for agent coordination
└── .env                         # Environment configuration
```

## 🚀 Quick Start

### 1. Environment Setup

First, configure your environment variables:

```bash
cd /home/tony/AI/ROCm/distributed-ai-dev/

# Copy and customize environment template
cp .env.template .env
nano .env  # Edit for your network setup

# Verify configuration
python3 src/core/config.py
```

### 2. Configure Your Agent Network

Edit your agents in the YAML configuration file:

```bash
nano config/agents.yaml
```

Example configuration:
```yaml
agents:
  agent_113:
    name: "Agent 113 - Kernel Development Architect"
    endpoint: "http://192.168.1.113:11434"  # Your actual IP
    model: "devstral:23.6b"
    specialization: "kernel_dev"
    status: "active"
    capabilities:
      - "HIP kernel optimization"
      - "ROCm architecture design"
      - "Memory management"
    hardware:
      gpu: "AMD Radeon RX 7900 XTX"
      vram: "24GB"
      cores: 16
      ram: "32GB"
  
  agent_27:
    name: "Agent 27 - Code Generation Support"
    endpoint: "http://192.168.1.27:11434"   # Your actual IP
    model: "codellama:latest"
    specialization: "code_generation"
    status: "active"
    capabilities:
      - "GPU programming"
      - "Code implementation"
      - "Technical documentation"
```

### 3. Install Dependencies

```bash
cd /home/tony/AI/ROCm/distributed-ai-dev/

# Install Python dependencies
pip install aiohttp asyncio dataclasses python-dotenv pyyaml

# Or install for system Python
python3 -m pip install --user aiohttp asyncio dataclasses python-dotenv pyyaml
```

### 4. Setup Monitoring Tools

```bash
# Setup executable monitoring tools (one-time)
source setup-monitoring.sh

# This adds monitoring commands to your PATH:
# - simple-monitor
# - advanced-monitor
# - agent-monitor
```

### 5. Test the System

```bash
# Test distributed system
python3 tests/test_distributed_system.py

# Test agent integration
python3 tests/quick_agent_integration.py

# Test unified pipeline
python3 tests/test_unified_pipeline.py
```

### 6. Start Using from Claude

When you need to delegate ROCm development work, use these commands:

```python
# Import from the organized structure
from orchestration.claude_interface import setup_development_network, delegate_work
from orchestration.ai_dev_coordinator import AIDevCoordinator
from src.core.config import DistributedAIConfig

# Setup the network (automatically loads from agents.yaml)
await setup_development_network()

# Load agents from YAML configuration
agents = DistributedAIConfig.load_agents_from_yaml()
print(f"Loaded {len(agents)} active agents")

# Delegate work with shared file access
result = await delegate_work(
    "Optimize FlashAttention kernel for RDNA3",
    files=["/path/to/attention.cpp"], 
    priority=5
)

# Use the coordination interface
coordinator = AIDevCoordinator()

# Check progress and collect results
progress = await check_progress()
results = await collect_results()
```

### 7. Monitor Agent Activity

```bash
# Quick monitoring (after setup-monitoring.sh)
simple-monitor                    # Simple terminal monitor
advanced-monitor -s               # Advanced curses monitor (single agent)
agent-monitor                     # Agent 113 dedicated monitor

# Monitor specific agents
simple-monitor --endpoint http://192.168.1.27:11434 --model codellama:latest
simple-monitor --endpoint http://192.168.1.113:11434 --model devstral:23.6b

# Direct Python monitoring
python3 orchestration/monitor_agent_113.py
python3 src/agents/simple_monitor.py
python3 src/agents/advanced_monitor.py
```

## 🎛️ How Claude Will Use This

### Example Claude Session:

**You:** "Claude, I need to optimize the VAE decoder in Stable Diffusion for ROCm. Can you delegate this to the agent network?"

**Claude:** 
```python
# I'll break this down and delegate to specialized agents
task_result = await delegate_work(
    "Optimize VAE decoder convolutions for Stable Diffusion on RDNA3",
    files=["vae_decoder.py", "conv_kernels.cpp"],
    priority=4
)

# While agents work, I'll research related optimizations
# and plan integration steps...

# Check results in a few minutes
results = await collect_results()
# Present integrated solution to you
```

### Benefits:
- **90% reduction** in Claude usage for coding tasks
- **Parallel development** across your entire network
- **Specialized expertise** - each agent focused on specific domains
- **Quality control** - multi-agent review before integration
- **24/7 development** - agents work while you sleep

## 📊 Current Agent Network

### Agent 113 - Kernel Development Architect
- **Endpoint**: http://192.168.1.113:11434
- **Primary Model**: DevStral 23.6B
- **Hardware**: AMD Radeon RX 7900 XTX (24GB VRAM), 16 cores, 32GB RAM
- **Specialization**: Senior kernel development, architecture design
- **Capabilities**: 
  - HIP kernel optimization
  - ROCm architecture design
  - Memory management strategies
  - Complex performance analysis
- **Status**: ACTIVE - Working on FlashAttention, VAE decoder optimizations

### Agent 27 - Code Generation Support
- **Endpoint**: http://192.168.1.27:11434
- **Primary Models**: CodeLlama (7B), StarCoder2 (15B), DeepSeek Coder V2 (15.7B)
- **Hardware**: AMD Radeon RX 9060XT (16GB VRAM), 16 cores, 32GB RAM
- **Specialization**: Code implementation, GPU programming
- **Capabilities**:
  - GPU programming and CUDA/HIP porting
  - Code generation and implementation
  - Technical documentation
  - Development support and debugging
- **Total Models**: 28 specialized models available
- **Status**: ACTIVE - Ready for intensive development collaboration

### Agent Specialization Mapping
- **kernel_dev**: Architecture design, optimization strategy, complex reasoning
- **code_generation**: Implementation, GPU programming, documentation
- **pytorch_dev**: PyTorch integration, autograd, TunableOp, Python bindings
- **profiler**: rocprof analysis, benchmarking, bottleneck identification
- **docs_writer**: API docs, tutorials, installation guides
- **tester**: Unit tests, integration tests, CI/CD automation

### Collaboration Pattern
- **Agent 113** provides architectural design and optimization strategy
- **Agent 27** implements code and handles GPU programming tasks
- **Both agents** collaborate on optimization and quality assurance
- **Claude** coordinates the overall development workflow

## 🔄 Typical Workflow

1. **Claude analyzes** your ROCm optimization request
2. **Breaks down** into specialized subtasks 
3. **Delegates** to appropriate agents in parallel
4. **Monitors progress** and coordinates between agents
5. **Reviews results** with quality control
6. **Integrates solutions** and creates final PR
7. **Submits to GitHub** with your approval

## ⚙️ Configuration Management

### Environment Variables
The system uses `.env` file for configuration:

```bash
# Key environment variables
DISTRIBUTED_AI_BASE=/home/tony/AI/ROCm/distributed-ai-dev
AGENT_113_URL=http://192.168.1.113:11434
AGENT_113_MODEL=devstral:23.6b
SHARED_WORKSPACE=${DISTRIBUTED_AI_BASE}/shared
CONFIG_DIR=${DISTRIBUTED_AI_BASE}/config
```

### YAML Agent Configuration
All agents are defined in `config/agents.yaml` as the single source of truth:

```yaml
agents:
  agent_113:
    status: "active"          # active/inactive
    priority: 5               # 1-5 priority level
    performance_targets:
      response_time: "< 30s"
      concurrent_tasks: 2
```

### Model Recommendations by Hardware:
- **8GB VRAM**: CodeLlama-13B, DeepSeek-Coder-6.7B, Qwen2.5-Coder-14B
- **16GB VRAM**: CodeLlama-34B, DeepSeek-Coder-33B, Qwen2.5-Coder-32B  
- **24GB VRAM**: DevStral-23.6B, Llama-3.1-70B (for complex analysis)

### Performance Tuning:
- Configure `max_concurrent` in agents.yaml based on model size
- Use shared workspace for file coordination between agents
- Leverage NAS/NFS mounting for distributed file access
- Monitor agents using built-in monitoring tools

## 🚨 Important Notes

- **Always review** agent-generated code before committing
- **Test thoroughly** on your specific hardware
- **Start small** with simple optimizations to validate the system
- **Monitor resource usage** across your network using built-in tools
- **Keep backups** of working configurations
- **Use YAML config** as single source of truth for agent management
- **Environment variables** enable flexible NAS/NFS deployment
- **Shared workspace** facilitates multi-agent file coordination
- **Check agent summaries** in `docs/` for detailed capability analysis

## 📈 Success Metrics

Track your distributed development efficiency:
- Tasks completed per day
- Code quality scores from reviews  
- Performance improvements achieved
- Claude usage reduction percentage (target: 90%)
- Time from idea to working optimization
- Agent utilization and coordination effectiveness

## 📚 Additional Resources

- **Agent Analysis**: See `docs/agent_113_summary.md` and `docs/agent_27_summary.md`
- **Configuration Reference**: `src/core/config.py` for all environment variables
- **Test Suite**: `tests/` directory for system validation
- **Results Tracking**: `orchestration/results/` for agent output
- **Performance Reports**: `reports/` directory for analysis

## 🔧 Troubleshooting

```bash
# Verify configuration
python3 src/core/config.py

# Test agent connectivity
python3 tests/quick_agent_integration.py

# Check shared workspace
ls -la shared/

# Monitor agent status
simple-monitor --refresh 10

# View recent results
ls -la orchestration/results/
```

Your system is now ready to accelerate ROCm development with organized structure, YAML-based configuration, and comprehensive monitoring! 🚀