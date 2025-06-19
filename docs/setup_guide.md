# Distributed AI Development System - Setup Guide

## 🎯 System Overview

You now have a complete distributed AI development coordination system that allows Claude to orchestrate multiple local Ollama agents working on any software development project. This multiplies your development capacity while minimizing Claude usage costs.

The system uses environment variables and YAML configuration for flexible deployment, with organized folder structure for better maintainability.

## 📁 Project Structure

```
distributed-ai-dev/
├── config/
│   └── agents.yaml              # Single source of truth for agent configuration
├── src/
│   ├── agents/                  # Agent monitoring applications
│   ├── core/                    # Configuration and core utilities
│   ├── interfaces/              # API interfaces
│   └── quality/                 # Quality control systems
├── tests/                       # All test files and integration tests
├── examples/                    # Example workload management
├── docs/                        # Documentation and guides
├── reports/                     # Performance reports and analysis
├── shared/                      # Shared workspace for agent coordination
└── requirements.txt             # Python dependencies
```

## 🚀 Quick Start

### 1. Environment Setup

First, clone and set up the repository:

```bash
git clone https://github.com/anthonyrawlins/distributed-ai-dev.git
cd distributed-ai-dev

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Configure Your Agent Network

Edit the `config/agents.yaml` file to match your infrastructure:

```yaml
agents:
  acacia_agent:
    name: "ACACIA Infrastructure Specialist"
    endpoint: "http://192.168.1.72:11434"
    model: "deepseek-r1:7b"
    specialization: "Infrastructure, DevOps & System Architecture"
    status: "active"
    # ... hardware and capability configuration

  walnut_agent:
    name: "WALNUT Senior Full-Stack Developer"
    endpoint: "http://192.168.1.27:11434"
    model: "starcoder2:15b"
    specialization: "Senior Full-Stack Development & Architecture" 
    status: "active"
    # ... hardware and capability configuration

  ironwood_agent:
    name: "IRONWOOD Backend Development Specialist"
    endpoint: "http://192.168.1.113:11434"
    model: "deepseek-coder-v2"
    specialization: "Backend Development & Code Analysis"
    status: "active"
    # ... hardware and capability configuration
```

### 3. Verify Agent Connectivity

Test that all your agents are accessible:

```bash
# Test basic connectivity to all agents
python3 tests/test_distributed_system.py

# Monitor agent performance
python3 src/agents/simple_monitor.py

# Advanced monitoring with detailed metrics
python3 src/agents/advanced_monitor.py
```

### 4. Test Individual Agents

```bash
# Test specific agent capabilities
python3 tests/test_agents.py

# Quick integration test
python3 tests/quick_agent_integration.py
```

## 🔧 Detailed Configuration

### Agent Configuration Parameters

Each agent in `config/agents.yaml` supports these parameters:

```yaml
agent_id:
  name: "Human-readable agent name"
  endpoint: "http://ip_address:11434"           # Ollama API endpoint
  model: "model_name:tag"                       # Default model to use
  specialization: "Agent expertise area"         # What this agent is best at
  priority: 1-5                                 # Task priority (1=highest)
  status: "active" | "disabled"                 # Current operational status
  
  capabilities:                                 # List of agent capabilities
    - "capability_1"
    - "capability_2"
    
  hardware:                                     # Hardware specifications
    gpu_type: "GPU model"
    vram_gb: 8
    cpu_cores: 16
    ram_gb: 64
    
  performance_targets:                          # Expected performance
    min_tokens_per_second: 5.0
    max_response_time_ms: 30000
    target_availability: 0.99
    
  available_models:                             # All models on this agent
    - "model1:latest"
    - "model2:latest"
```

### Global Configuration

Configure system-wide settings in `config/agents.yaml`:

```yaml
monitoring:
  refresh_interval_seconds: 5              # How often to refresh metrics
  performance_window_minutes: 5            # Performance averaging window
  max_history_samples: 100                 # Historical data retention

network:
  timeout_seconds: 30                      # Request timeout
  retry_attempts: 3                        # Number of retries
  retry_delay_seconds: 5                   # Delay between retries

logging:
  level: "INFO"                            # Log level
  file: "logs/agent_monitoring.log"        # Log file location
  max_size_mb: 100                         # Max log file size
  backup_count: 5                          # Number of backup log files
```

## 🖥️ Monitoring and Management

### Real-time Monitoring

The system provides two monitoring interfaces:

#### Simple Monitor (Terminal-friendly)
```bash
python3 src/agents/simple_monitor.py
```
- Clean terminal output
- Real-time performance metrics
- Easy to read status indicators
- Perfect for logging and automation

#### Advanced Monitor (Interactive)
```bash
python3 src/agents/advanced_monitor.py
```
- btop/nvtop-style interface
- Interactive controls (q=quit, r=refresh, c=reload config)
- Color-coded status indicators
- Real-time resource utilization

### Performance Metrics

The monitoring system tracks:
- **Tokens per Second (TPS)**: Inference speed
- **Response Latency**: End-to-end response time
- **Availability**: Agent uptime percentage
- **Resource Usage**: CPU, RAM, GPU utilization
- **Queue Status**: Pending tasks and queue depth

### Alerting Thresholds

Configure alerting in `config/agents.yaml`:

```yaml
monitoring:
  alert_thresholds:
    min_tokens_per_second: 2.0              # Alert if TPS drops below this
    max_response_time_ms: 60000             # Alert if response time exceeds this
    min_availability: 0.85                  # Alert if availability drops below this
```

## 🔌 Integration with Claude

### Basic Usage

```python
from src.interfaces.claude_interface import setup_development_network, delegate_work

# Initialize the agent network
await setup_development_network()

# Delegate development tasks
result = await delegate_work(
    task="Create a React component for user authentication",
    files=["auth.tsx", "types.ts"],
    priority=5,
    preferred_agent="walnut_agent"
)

# Monitor progress
progress = await check_progress()
results = await collect_results()
```

### Advanced Coordination

```python
from src.core.ai_dev_coordinator import AIDevCoordinator

coordinator = AIDevCoordinator()

# Multi-agent task coordination
await coordinator.coordinate_project({
    "frontend": {"agent": "walnut_agent", "tasks": ["components", "routing"]},
    "backend": {"agent": "ironwood_agent", "tasks": ["api", "database"]},
    "infrastructure": {"agent": "acacia_agent", "tasks": ["docker", "deployment"]}
})
```

## 🧪 Testing and Validation

### System Integration Tests

```bash
# Full system test
python3 tests/test_distributed_system.py

# Test specific agents
python3 tests/test_agents.py

# Quick validation
python3 tests/quick_agent_integration.py
```

### Performance Validation

```bash
# Benchmark agent performance
python3 examples/agent_workload.py

# Load testing
python3 tests/load_test_agents.py  # If available
```

## 🔍 Troubleshooting

### Common Issues

#### Agent Not Responding
```bash
# Check agent connectivity
curl http://192.168.1.xx:11434/api/version

# Check Ollama service status (on agent machine)
sudo systemctl status ollama

# Restart Ollama if needed
sudo systemctl restart ollama
```

#### Performance Issues
```bash
# Monitor system resources
python3 src/agents/advanced_monitor.py

# Check model loading
curl http://192.168.1.xx:11434/api/ps

# View Ollama logs
journalctl -u ollama -f
```

#### Configuration Problems
```bash
# Validate YAML syntax
python3 -c "import yaml; yaml.safe_load(open('config/agents.yaml'))"

# Test configuration loading
python3 src/core/config.py
```

### Log Analysis

```bash
# View recent logs
tail -f logs/agent_monitoring.log

# Search for errors
grep -i error logs/agent_monitoring.log

# Performance analysis
grep -i "performance" logs/agent_monitoring.log
```

## 📊 Performance Optimization

### Model Selection

Choose models based on task requirements:
- **Large complex tasks**: starcoder2:15b, deepseek-coder-v2
- **General development**: devstral:latest, qwen3:latest
- **Quick responses**: phi4, llama3.1:8b
- **Specialized tasks**: deepseek-r1:7b (reasoning), codellama (code)

### Resource Management

```yaml
# Optimize concurrent requests based on hardware
performance_targets:
  max_concurrent_requests: 1    # For large models (>20B params)
  max_concurrent_requests: 3    # For smaller models (<15B params)
```

### Network Optimization

```yaml
network:
  timeout_seconds: 30           # Adjust based on model size
  retry_attempts: 3             # Increase for unreliable networks
  keep_alive: true              # Maintain connections
```

## 🔄 Maintenance and Updates

### Regular Maintenance

```bash
# Update models on agents
# (Run on each agent machine)
ollama pull starcoder2:15b
ollama pull deepseek-coder-v2
ollama pull qwen3:latest

# Clean up old logs
find logs/ -name "*.log" -mtime +30 -delete

# Update system dependencies
pip install -r requirements.txt --upgrade
```

### Configuration Updates

```bash
# Reload configuration without restart
# Advanced monitor supports 'c' key to reload config

# Validate new configuration
python3 src/core/config.py

# Restart monitoring systems
pkill -f simple_monitor.py
python3 src/agents/simple_monitor.py &
```

## 🚀 Production Deployment

### Systemd Service Setup

Create a systemd service for continuous monitoring:

```ini
[Unit]
Description=Distributed AI Development Monitor
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/distributed-ai-dev
ExecStart=/usr/bin/python3 src/agents/simple_monitor.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python3", "src/agents/simple_monitor.py"]
```

### Security Considerations

- **Network Security**: Ensure Ollama APIs are not exposed to public internet
- **Access Control**: Use firewall rules to restrict access to agent endpoints
- **Monitoring**: Set up log monitoring and alerting for security events
- **Updates**: Keep Python dependencies and system packages updated

This setup guide provides everything needed to deploy and manage your distributed AI development system effectively across any infrastructure.