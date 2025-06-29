# General-Purpose Distributed AI Agent Coordination System

A powerful, flexible, and portable coordination system that enables a primary AI assistant (like Gemini or Claude) to orchestrate multiple, specialized local AI agents for any software development task.

## 🚀 Features

- **General-Purpose Agent Coordination**: Orchestrate multiple Ollama agents with diverse specializations across your local network.
- **YAML Configuration**: Centrally manage your entire agent network through a single, easy-to-use YAML file. No hardcoded values.
- **Real-time Monitoring**: A unified monitoring dashboard with both simple and advanced (curses-based) UI modes to track agent performance and system resources.
- **Flexible Specializations**: Define your own agent roles (e.g., `react_developer`, `python_expert`, `database_admin`, `technical_writer`). The system is not tied to any specific domain.
- **Content-Based File Handling**: Pass file content directly to agents, removing the need for complex shared network drives.
- **Extensible & Portable**: Designed to be easily configured and run on any local network with machines capable of running Ollama.

## 🎯 Perfect For

- **Web Application Development**: Full-stack JavaScript, React, Node.js, etc.
- **API Development**: RESTful services, GraphQL, microservices.
- **DevOps & Infrastructure**: Docker, Kubernetes, CI/CD pipelines.
- **Data Science & ML**: Python scripting, data analysis, model exploration.
- **Documentation**: Generating technical guides, API docs, and tutorials.
- **Any task you can delegate to a specialized AI agent!**

## 📋 Quick Start

### 1. Installation

An `install.sh` script will be provided to automate this process.

```bash
# 1. Clone the repository
git clone https://github.com/anthonyrawlins/distributed-ai-dev.git
cd distributed-ai-dev

# 2. Create a virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure your agents
cp config/agents.yaml.template config/agents.yaml
nano config/agents.yaml # Edit to define your agents
```

### 2. Configure Your Agents

Edit `config/agents.yaml` to define your agent network. This is the central control panel for your system.

```yaml
# Global monitoring settings
monitoring:
  refresh_interval_seconds: 5

# Agent Definitions
agents:
  frontend_dev:
    name: "Frontend Developer"
    endpoint: "http://192.168.1.10:11434" # Replace with your agent's IP
    model: "starcoder2:15b"
    specialization: "react_developer"
    status: "active"

  backend_dev:
    name: "Backend Developer"
    endpoint: "http://192.168.1.11:11434" # Replace with your agent's IP
    model: "deepseek-coder-v2"
    specialization: "python_backend_api"
    status: "active"

  docs_writer:
    name: "Documentation Writer"
    endpoint: "http://192.168.1.12:11434" # Replace with your agent's IP
    model: "llama3.1:8b"
    specialization: "technical_writer"
    status: "disabled" # This agent will not be loaded
```

### 3. Monitor Your Agents

Launch the unified real-time monitoring dashboard.

```bash
# For a simple, universally compatible UI
python3 src/agents/monitor.py --ui simple

# For an advanced, btop-style UI (requires curses)
python3 src/agents/monitor.py --ui advanced
```

### 4. Delegate Work

Use the `claude_interface.py` (or adapt it for your primary AI) to delegate tasks.

```python
from src.interfaces.claude_interface import setup_development_network, delegate_work

# 1. Load agents from your config file
await setup_development_network("config/agents.yaml")

# 2. Delegate a task to a specific specialization
task_id = await delegate_work(
    specialization="react_developer",
    task_description="Create a React component for a simple button.",
    files={
        "Button.tsx": "import React from 'react';",
        "Button.css": ".button { color: red; }"
    }
)

print(f"Work delegated. Task ID: {task_id}")

# 3. Check progress and get results
progress = await check_progress()
results = await collect_results([task_id])
```

## 🏗️ Architecture

The system follows a simple coordinator-agent architecture.

```
┌─────────────────┐      ┌──────────────────┐
│                 │      │                  │
│  Primary AI     ├─────►│   Coordinator    │
│ (Gemini/Claude) │      │ (This System)    │
│                 │      │                  │
└─────────────────┘      └────────┬─────────┘
                                  │
                                  │ Delegates Tasks
                                  │
                  ┌───────────────┼───────────────┐
                  │               │               │
            ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐
            │  Agent 1  │   │  Agent 2  │   │  Agent N  │
            │ (React)   │   │ (Python)  │   │  (Docs)   │
            └───────────┘   └───────────┘   └───────────┘
```

## 🔧 Project Structure

```
distributed-ai-dev/
├── config/
│   └── agents.yaml.template     # ⭐ Agent configuration template
├── src/
│   ├── core/
│   │   └── ai_dev_coordinator.py  # Main coordination logic
│   ├── interfaces/
│   │   └── claude_interface.py    # Interface for the primary AI
│   └── agents/
│       ├── monitor.py             # ⭐ Unified monitoring dashboard
│       └── monitoring_utils.py    # Shared monitoring logic
├── tests/
├── docs/
├── README.md                      # This file
└── requirements.txt
```

## 🤝 Contributing

Contributions are welcome! Please see the `TODOS.md` file for the planned refactoring work.

## 📄 License

This project is licensed under the MIT License.
