#!/usr/bin/env python3
"""
Shared Configuration Module for Distributed AI Development System
Centralizes environment variable handling and provides consistent configuration access
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path)

class DistributedAIConfig:
    """
    Centralized configuration class using environment variables
    Supports NAS/NFS shared directories for multi-agent coordination
    """
    
    # Base installation directory (can be NAS/NFS shared path)
    DISTRIBUTED_AI_BASE = os.getenv('DISTRIBUTED_AI_BASE', '/home/tony/AI/ROCm/distributed-ai-dev')
    
    # Key directories for agent coordination
    ORCHESTRATION_DIR = os.getenv('ORCHESTRATION_DIR', f'{DISTRIBUTED_AI_BASE}/orchestration')
    REPORTS_DIR = os.getenv('REPORTS_DIR', f'{DISTRIBUTED_AI_BASE}/reports')
    TESTS_DIR = os.getenv('TESTS_DIR', f'{DISTRIBUTED_AI_BASE}/tests')
    RESULTS_DIR = os.getenv('RESULTS_DIR', f'{DISTRIBUTED_AI_BASE}/orchestration/results')
    CONFIG_DIR = os.getenv('CONFIG_DIR', f'{DISTRIBUTED_AI_BASE}/config')
    DOCS_DIR = os.getenv('DOCS_DIR', f'{DISTRIBUTED_AI_BASE}/docs')
    EXAMPLES_DIR = os.getenv('EXAMPLES_DIR', f'{DISTRIBUTED_AI_BASE}/examples')
    
    # Source code directories
    SRC_DIR = os.getenv('SRC_DIR', f'{DISTRIBUTED_AI_BASE}/src')
    AGENTS_DIR = os.getenv('AGENTS_DIR', f'{SRC_DIR}/agents')
    INTERFACES_DIR = os.getenv('INTERFACES_DIR', f'{SRC_DIR}/interfaces')
    CORE_DIR = os.getenv('CORE_DIR', f'{SRC_DIR}/core')
    KERNELS_DIR = os.getenv('KERNELS_DIR', f'{SRC_DIR}/kernels')
    PIPELINE_DIR = os.getenv('PIPELINE_DIR', f'{SRC_DIR}/pipeline')
    PYTORCH_INTEGRATION_DIR = os.getenv('PYTORCH_INTEGRATION_DIR', f'{SRC_DIR}/pytorch_integration')
    
    # ROCm repositories (NAS symlink)
    ROCM_REPOS_BASE = os.getenv('ROCM_REPOS_BASE', '/rust/containers/rocm-dev')
    ROCM_REPOS_LOCAL = os.getenv('ROCM_REPOS_LOCAL', '/home/tony/AI/ROCm/repositories')
    
    # Agent configuration
    AGENT_113_URL = os.getenv('AGENT_113_URL', 'http://192.168.1.113:11434')
    AGENT_113_MODEL = os.getenv('AGENT_113_MODEL', 'devstral:23.6b')
    DEFAULT_TIMEOUT = int(os.getenv('DEFAULT_TIMEOUT', '120000'))
    
    # Additional agent endpoints (for future expansion)
    KERNEL_EXPERT_URL = os.getenv('KERNEL_EXPERT_URL', 'http://machine1:11434')
    KERNEL_EXPERT_MODEL = os.getenv('KERNEL_EXPERT_MODEL', 'codellama:34b')
    PYTORCH_SPECIALIST_URL = os.getenv('PYTORCH_SPECIALIST_URL', 'http://machine2:11434')
    PYTORCH_SPECIALIST_MODEL = os.getenv('PYTORCH_SPECIALIST_MODEL', 'deepseek-coder:33b')
    PERFORMANCE_ANALYZER_URL = os.getenv('PERFORMANCE_ANALYZER_URL', 'http://machine3:11434')
    PERFORMANCE_ANALYZER_MODEL = os.getenv('PERFORMANCE_ANALYZER_MODEL', 'qwen2.5-coder:32b')
    
    # Shared workspace for multi-agent coordination
    SHARED_WORKSPACE = os.getenv('SHARED_WORKSPACE', f'{DISTRIBUTED_AI_BASE}/shared')
    SHARED_CONTEXT = os.getenv('SHARED_CONTEXT', f'{SHARED_WORKSPACE}/context')
    SHARED_RESULTS = os.getenv('SHARED_RESULTS', f'{SHARED_WORKSPACE}/results')
    
    # Development settings
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
    MAX_CONCURRENT_AGENTS = int(os.getenv('MAX_CONCURRENT_AGENTS', '4'))
    
    # NAS/NFS specific settings (for future expansion)
    NAS_MOUNT_POINT = os.getenv('NAS_MOUNT_POINT', '/mnt/nas/distributed-ai-dev')
    NFS_EXPORT_PATH = os.getenv('NFS_EXPORT_PATH', '/exports/distributed-ai-dev')
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist"""
        directories = [
            cls.ORCHESTRATION_DIR,
            cls.REPORTS_DIR,
            cls.TESTS_DIR,
            cls.RESULTS_DIR,
            cls.CONFIG_DIR,
            cls.DOCS_DIR,
            cls.EXAMPLES_DIR,
            cls.SRC_DIR,
            cls.AGENTS_DIR,
            cls.INTERFACES_DIR,
            cls.CORE_DIR,
            cls.KERNELS_DIR,
            cls.PIPELINE_DIR,
            cls.PYTORCH_INTEGRATION_DIR,
            cls.SHARED_WORKSPACE,
            cls.SHARED_CONTEXT,
            cls.SHARED_RESULTS
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
        if cls.DEBUG_MODE:
            print(f"[DEBUG] Ensured {len(directories)} directories exist")
    
    @classmethod
    def get_agent_config(cls, agent_id: str = None):
        """
        Get agent configuration by ID from agents.yaml
        
        DEPRECATED: Use load_agents_from_yaml() instead
        This method is kept for backward compatibility only
        """
        import warnings
        warnings.warn(
            "get_agent_config() is deprecated. Use load_agents_from_yaml() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        try:
            agents = cls.load_agents_from_yaml()
            if agent_id:
                return next((agent for agent in agents if agent['id'] == agent_id), None)
            return agents
        except Exception:
            # Fallback to legacy environment-based config
            legacy_config = {
                'id': 'agent_113',
                'endpoint': cls.AGENT_113_URL,
                'model': cls.AGENT_113_MODEL,
                'specialty': 'kernel_dev',
                'max_concurrent': 1
            }
            return [legacy_config] if not agent_id else legacy_config
    
    @classmethod
    def load_agents_from_yaml(cls):
        """Load agent configuration from agents.yaml (recommended approach)"""
        import yaml
        
        config_path = Path(cls.CONFIG_DIR) / 'agents.yaml'
        
        if not config_path.exists():
            raise FileNotFoundError(f"Agent configuration not found: {config_path}")
        
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        agents = []
        for agent_id, agent_data in yaml_config.get('agents', {}).items():
            if agent_data.get('status') == 'active':
                agent = {
                    'id': agent_id,
                    'name': agent_data.get('name', f'Agent {agent_id}'),
                    'endpoint': agent_data['endpoint'],
                    'model': agent_data['model'],
                    'specialization': agent_data.get('specialization', 'General'),
                    'capabilities': agent_data.get('capabilities', []),
                    'hardware': agent_data.get('hardware', {}),
                    'performance_targets': agent_data.get('performance_targets', {}),
                    'priority': agent_data.get('priority', 3),
                    'status': agent_data.get('status', 'unknown')
                }
                agents.append(agent)
        
        return agents
    
    @classmethod
    def is_nas_available(cls):
        """Check if NAS/NFS mount is available"""
        return Path(cls.NAS_MOUNT_POINT).exists()
    
    @classmethod
    def get_shared_path(cls, relative_path: str = ""):
        """Get shared path, preferring NAS if available"""
        if cls.is_nas_available():
            return Path(cls.NAS_MOUNT_POINT) / relative_path
        return Path(cls.SHARED_WORKSPACE) / relative_path
    
    @classmethod
    def export_env_template(cls, filepath: str = None):
        """Export a template .env file for easy setup"""
        if not filepath:
            filepath = Path(cls.DISTRIBUTED_AI_BASE) / '.env.template'
        
        template = f"""# Distributed AI Development System Environment Configuration Template
# Copy this to .env and customize for your setup

# Base installation directory (can be NAS/NFS shared path)
DISTRIBUTED_AI_BASE={cls.DISTRIBUTED_AI_BASE}

# Key directories for agent coordination
ORCHESTRATION_DIR=${{DISTRIBUTED_AI_BASE}}/orchestration
REPORTS_DIR=${{DISTRIBUTED_AI_BASE}}/reports
TESTS_DIR=${{DISTRIBUTED_AI_BASE}}/tests
RESULTS_DIR=${{DISTRIBUTED_AI_BASE}}/orchestration/results
CONFIG_DIR=${{DISTRIBUTED_AI_BASE}}/config
DOCS_DIR=${{DISTRIBUTED_AI_BASE}}/docs
EXAMPLES_DIR=${{DISTRIBUTED_AI_BASE}}/examples

# Source code directories
SRC_DIR=${{DISTRIBUTED_AI_BASE}}/src
AGENTS_DIR=${{SRC_DIR}}/agents
INTERFACES_DIR=${{SRC_DIR}}/interfaces
CORE_DIR=${{SRC_DIR}}/core
KERNELS_DIR=${{SRC_DIR}}/kernels
PIPELINE_DIR=${{SRC_DIR}}/pipeline
PYTORCH_INTEGRATION_DIR=${{SRC_DIR}}/pytorch_integration

# ROCm repositories (NAS symlink)
ROCM_REPOS_BASE=/rust/containers/rocm-dev
ROCM_REPOS_LOCAL=/home/tony/AI/ROCm/repositories

# Agent configuration - UPDATE THESE FOR YOUR NETWORK
AGENT_113_URL=http://192.168.1.113:11434
AGENT_113_MODEL=devstral:23.6b
DEFAULT_TIMEOUT=120000

# Additional agent endpoints (customize for your setup)
KERNEL_EXPERT_URL=http://machine1:11434
KERNEL_EXPERT_MODEL=codellama:34b
PYTORCH_SPECIALIST_URL=http://machine2:11434
PYTORCH_SPECIALIST_MODEL=deepseek-coder:33b
PERFORMANCE_ANALYZER_URL=http://machine3:11434
PERFORMANCE_ANALYZER_MODEL=qwen2.5-coder:32b

# Shared workspace for multi-agent coordination
SHARED_WORKSPACE=${{DISTRIBUTED_AI_BASE}}/shared
SHARED_CONTEXT=${{SHARED_WORKSPACE}}/context
SHARED_RESULTS=${{SHARED_WORKSPACE}}/results

# Development settings
LOG_LEVEL=INFO
DEBUG_MODE=false
MAX_CONCURRENT_AGENTS=4

# NAS/NFS specific settings (for future expansion)
# NAS_MOUNT_POINT=/mnt/nas/distributed-ai-dev
# NFS_EXPORT_PATH=/exports/distributed-ai-dev
"""
        
        with open(filepath, 'w') as f:
            f.write(template)
        
        return filepath

# Create global config instance
config = DistributedAIConfig()

if __name__ == "__main__":
    print("Distributed AI Development System Configuration")
    print(f"Base directory: {config.DISTRIBUTED_AI_BASE}")
    print(f"Shared workspace: {config.SHARED_WORKSPACE}")
    print(f"Agent 113: {config.AGENT_113_URL}")
    print(f"Debug mode: {config.DEBUG_MODE}")
    
    # Ensure directories exist
    config.ensure_directories()
    print("✅ All directories verified/created")
    
    # Export template
    template_path = config.export_env_template()
    print(f"📝 Environment template exported: {template_path}")