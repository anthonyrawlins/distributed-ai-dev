# Distributed AI Development System: Refactoring Plan

This document outlines the step-by-step plan to refactor the project from a niche, hardcoded tool into the powerful, general-purpose, and portable multi-agent coordination system envisioned in the `README.md`.

### 1. Codebase Changes

The primary goal is to decouple the system from its hardcoded, niche focus on ROCm/GPU development and transform it into a flexible, configuration-driven tool.

*   **Generalize the Core Coordinator (`src/core/ai_dev_coordinator.py`)**
    *   [ ] **Remove `AgentType` Enum:** Delete the `AgentType` enum to allow for arbitrary specializations.
    *   [ ] **Abstract the Prompting System:** Remove the hardcoded `agent_prompts` dictionary. The prompt sent to the agent should be constructed dynamically based on the task description and the agent's `specialization` string from the config.
    *   [ ] **Update Task Creation:** Modify `create_task` to accept a `specialization` string (e.g., "react_developer") instead of an `AgentType`. The coordinator will use this string to find a matching agent.

*   **Refactor the Claude Interface (`src/interfaces/claude_interface.py`)**
    *   [ ] **Rename `delegate_rocm_optimization`:** Change the function name to something more generic, like `delegate_task`, to reflect its new general-purpose nature.
    *   [ ] **Remove Specialized Logic:** Delete the `_analyze_optimization_task` method. The responsibility for breaking down complex tasks should lie with the primary user (e.g., Claude), not be hardcoded with heuristics for a specific domain.
    *   [ ] **Implement Content-Based File Handling:**
        *   Modify `delegate_task` to read the *content* of any files listed in the `files` argument.
        *   Pass the file content (e.g., as a dictionary of `{"filename": "file_content"}`) within the task's `context`.
        *   The agent's prompt should be updated to include the file content, and the agent should be instructed to return the modified content in its response. This removes the fragile dependency on a shared network drive.

*   **Unify and Improve Monitoring Tools (`src/agents/`)**
    *   [ ] **Consolidate Monitoring Logic:** Merge `simple_monitor.py` and `advanced_monitor.py` into a single, more robust `monitor.py` script.
    *   [ ] **Add a UI-Selector Flag:** Implement a command-line argument (e.g., `--ui simple` or `--ui advanced`) to allow the user to choose the monitoring interface.
    *   [ ] **Centralize Configuration:** Remove all hardcoded IP addresses and model names. The `monitor.py` script must load all agent connection details directly from the `config/agents.yaml` file.
    *   [ ] **Extract Shared Code:** Move the common classes (`AgentPerformanceMetrics`, `SystemResourceMonitor`) into a separate utility module (`src/agents/monitoring_utils.py`) to reduce code duplication.

*   **Generalize the Quality Control System (`src/quality/quality_control.py`)**
    *   [ ] **Make Language Checks Pluggable:** The current static analysis has hardcoded checks for C++ and Python. Refactor this to be more modular, allowing new language-specific checks to be added easily without modifying the core logic.
    *   [ ] **Improve Language Detection:** Enhance the `_detect_language` function to be more robust.

### 2. Documentation Changes

The documentation needs to be completely overhauled to reflect the new, general-purpose identity of the project.

*   **Rewrite the `README.md`**
    *   [ ] **Update Title and Description:** Change the title and opening paragraph to describe a "General-Purpose Distributed AI Agent Coordination System."
    *   [ ] **Revise Feature List:** Update the features to emphasize flexibility, portability, and extensibility. Remove all mentions of ROCm, CUDA, and GPU-specific tasks.
    *   [ ] **Provide a Generic "Quick Start" Example:** Replace the current ROCm-focused example with a more common use case, like asking an agent to write a Python script or a simple web page.
    *   [ ] **Update `agents.yaml` Example:** The example configuration should showcase a variety of specializations, such as `python_expert`, `frontend_developer`, and `technical_writer`.
    *   [ ] **Redraw the Architecture Diagram:** The diagram should illustrate a generic coordinator communicating with a configurable set of specialized agents.

*   **Clean Up and Create New Docs**
    *   [ ] **Delete Obsolete Files:** Remove the now-irrelevant `agent_113_rdna4_workload.py` file.
    *   [ ] **Create `docs/architecture.md`:** Add a new document that clearly explains the system's architecture, data flow, and the interaction between the coordinator and the agents.
    *   [ ] **Create `docs/configuration.md`:** Add a detailed guide that explains every setting in the `config/agents.yaml` file, empowering users to customize the system for their needs.

### 3. Installation and User Experience

The project needs a clear, simple, and reliable process for new users to get started.

*   **Create a Configuration Template**
    *   [ ] **Rename `config/agents.yaml` to `config/agents.yaml.template`:** This prevents users from accidentally overwriting their configuration when pulling updates.
    *   [ ] **Populate the Template:** Fill the template with commented-out examples of different agent configurations to guide the user.

*   **Improve the Installation Process**
    *   [ ] **Create an `install.sh` Script:** This script should automate the setup process:
        1.  Check for a valid Python version.
        2.  Create a Python virtual environment.
        3.  Install all dependencies from `requirements.txt`.
        4.  Check if `config/agents.yaml` exists. If not, copy it from the template and instruct the user to edit it.
    *   [ ] **Sanitize `requirements.txt`:** Review and clean up the `requirements.txt` file to ensure it contains only the necessary dependencies for the project to run. Remove any dependencies that were specific to the old ROCm-related work.
