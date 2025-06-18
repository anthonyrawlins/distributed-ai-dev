#!/usr/bin/env python3
"""
Claude Interface for AI Development Coordination
Provides easy integration for Claude to manage the distributed development system
"""

import asyncio
import json
import os
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from ai_dev_coordinator import AIDevCoordinator, Agent, AgentType, Task

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

# Environment variable configuration
class Config:
    """Configuration class using environment variables"""
    
    # Base directories
    DISTRIBUTED_AI_BASE = os.getenv('DISTRIBUTED_AI_BASE', '/home/tony/AI/ROCm/distributed-ai-dev')
    ORCHESTRATION_DIR = os.getenv('ORCHESTRATION_DIR', f'{DISTRIBUTED_AI_BASE}/orchestration')
    REPORTS_DIR = os.getenv('REPORTS_DIR', f'{DISTRIBUTED_AI_BASE}/reports')
    TESTS_DIR = os.getenv('TESTS_DIR', f'{DISTRIBUTED_AI_BASE}/tests')
    RESULTS_DIR = os.getenv('RESULTS_DIR', f'{DISTRIBUTED_AI_BASE}/orchestration/results')
    CONFIG_DIR = os.getenv('CONFIG_DIR', f'{DISTRIBUTED_AI_BASE}/config')
    
    # Source directories
    SRC_DIR = os.getenv('SRC_DIR', f'{DISTRIBUTED_AI_BASE}/src')
    AGENTS_DIR = os.getenv('AGENTS_DIR', f'{SRC_DIR}/agents')
    INTERFACES_DIR = os.getenv('INTERFACES_DIR', f'{SRC_DIR}/interfaces')
    CORE_DIR = os.getenv('CORE_DIR', f'{SRC_DIR}/core')
    KERNELS_DIR = os.getenv('KERNELS_DIR', f'{SRC_DIR}/kernels')
    
    # Shared workspace for multi-agent coordination
    SHARED_WORKSPACE = os.getenv('SHARED_WORKSPACE', f'{DISTRIBUTED_AI_BASE}/shared')
    SHARED_CONTEXT = os.getenv('SHARED_CONTEXT', f'{SHARED_WORKSPACE}/context')
    SHARED_RESULTS = os.getenv('SHARED_RESULTS', f'{SHARED_WORKSPACE}/results')
    
    # Agent configuration
    AGENT_113_URL = os.getenv('AGENT_113_URL', 'http://192.168.1.113:11434')
    AGENT_113_MODEL = os.getenv('AGENT_113_MODEL', 'devstral:23.6b')
    DEFAULT_TIMEOUT = int(os.getenv('DEFAULT_TIMEOUT', '120000'))
    
    # Development settings
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
    MAX_CONCURRENT_AGENTS = int(os.getenv('MAX_CONCURRENT_AGENTS', '4'))
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist"""
        directories = [
            cls.SHARED_WORKSPACE,
            cls.SHARED_CONTEXT,
            cls.SHARED_RESULTS,
            cls.RESULTS_DIR
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

class ClaudeInterface:
    def __init__(self):
        # Ensure required directories exist
        Config.ensure_directories()
        
        self.coordinator = AIDevCoordinator()
        self.session_log = []
        self.config = Config
        
        # Initialize session context file for multi-agent sharing
        self.session_context_file = Path(Config.SHARED_CONTEXT) / f"session_{asyncio.get_event_loop().time()}.json"
        self._init_session_context()
    
    def _init_session_context(self):
        """Initialize shared session context for multi-agent coordination"""
        session_data = {
            "session_id": str(asyncio.get_event_loop().time()),
            "config": {
                "distributed_ai_base": Config.DISTRIBUTED_AI_BASE,
                "shared_workspace": Config.SHARED_WORKSPACE,
                "results_dir": Config.RESULTS_DIR,
                "agent_113_url": Config.AGENT_113_URL
            },
            "active_tasks": {},
            "shared_files": [],
            "coordination_log": []
        }
        
        with open(self.session_context_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        self.log(f"Session context initialized: {self.session_context_file}")
    
    def setup_agents(self, agent_configs: List[Dict[str, str]]):
        """Setup agents from configuration"""
        for config in agent_configs:
            agent = Agent(
                id=config['id'],
                endpoint=config['endpoint'], 
                model=config['model'],
                specialty=AgentType(config['specialty']),
                max_concurrent=config.get('max_concurrent', 2)
            )
            self.coordinator.add_agent(agent)
            self.log(f"Agent {agent.id} registered")
    
    def log(self, message: str):
        """Log session activities to both local and shared storage"""
        timestamp = asyncio.get_event_loop().time()
        log_entry = f"{timestamp}: {message}"
        
        # Local session log
        self.session_log.append(log_entry)
        print(f"[COORDINATOR] {message}")
        
        # Update shared session context
        try:
            with open(self.session_context_file, 'r') as f:
                session_data = json.load(f)
            
            session_data['coordination_log'].append({
                "timestamp": timestamp,
                "message": message,
                "source": "claude_interface"
            })
            
            with open(self.session_context_file, 'w') as f:
                json.dump(session_data, f, indent=2)
        except Exception as e:
            print(f"[WARNING] Failed to update shared context: {e}")
    
    def share_file(self, file_path: str, description: str = "") -> str:
        """Share a file in the distributed workspace for agent access"""
        try:
            source_path = Path(file_path)
            if not source_path.exists():
                raise FileNotFoundError(f"Source file not found: {file_path}")
            
            # Copy file to shared workspace
            shared_file_path = Path(Config.SHARED_WORKSPACE) / source_path.name
            import shutil
            shutil.copy2(source_path, shared_file_path)
            
            # Update session context
            with open(self.session_context_file, 'r') as f:
                session_data = json.load(f)
            
            session_data['shared_files'].append({
                "original_path": str(source_path),
                "shared_path": str(shared_file_path),
                "description": description,
                "timestamp": asyncio.get_event_loop().time()
            })
            
            with open(self.session_context_file, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            self.log(f"File shared: {source_path.name} -> {shared_file_path}")
            return str(shared_file_path)
            
        except Exception as e:
            error_msg = f"Failed to share file {file_path}: {e}"
            self.log(error_msg)
            raise RuntimeError(error_msg)
    
    async def delegate_rocm_optimization(self, 
                                       task_description: str, 
                                       files_involved: List[str] = None,
                                       priority: int = 3) -> str:
        """
        High-level function for Claude to delegate ROCm optimization work
        Returns task IDs for tracking
        """
        files_involved = files_involved or []
        
        # Analyze the task and break it down
        task_breakdown = self._analyze_optimization_task(task_description, files_involved)
        
        created_tasks = []
        for subtask in task_breakdown:
            task = self.coordinator.create_task(
                AgentType(subtask['type']),
                subtask['context'], 
                priority
            )
            created_tasks.append(task.id)
        
        self.log(f"Created {len(created_tasks)} subtasks for: {task_description}")
        
        # Process tasks asynchronously
        await self.coordinator.process_queue()
        
        return f"Tasks created: {', '.join(created_tasks)}"
    
    def _analyze_optimization_task(self, description: str, files: List[str]) -> List[Dict]:
        """Analyze a high-level task and break it into agent-specific subtasks"""
        tasks = []
        
        # Simple heuristics for task breakdown (Claude can enhance this logic)
        if any(keyword in description.lower() for keyword in ['kernel', 'hip', 'cuda', 'attention']):
            tasks.append({
                'type': 'kernel_dev',
                'context': {
                    'objective': description,
                    'files': files,
                    'focus': 'kernel optimization',
                    'architecture': 'RDNA3/CDNA3'
                }
            })
        
        if any(keyword in description.lower() for keyword in ['pytorch', 'python', 'integration']):
            tasks.append({
                'type': 'pytorch_dev', 
                'context': {
                    'objective': description,
                    'files': files,
                    'focus': 'pytorch integration',
                    'requirements': ['autograd compatibility', 'API consistency']
                }
            })
        
        if any(keyword in description.lower() for keyword in ['performance', 'benchmark', 'profile']):
            tasks.append({
                'type': 'profiler',
                'context': {
                    'objective': description,
                    'files': files,
                    'focus': 'performance analysis',
                    'tools': ['rocprof', 'rocm-smi']
                }
            })
        
        if any(keyword in description.lower() for keyword in ['test', 'validate', 'verify']):
            tasks.append({
                'type': 'tester',
                'context': {
                    'objective': description,
                    'files': files,
                    'focus': 'test development',
                    'coverage': ['unit tests', 'integration tests', 'performance tests']
                }
            })
        
        # Default to kernel development if no specific type identified
        if not tasks:
            tasks.append({
                'type': 'kernel_dev',
                'context': {
                    'objective': description,
                    'files': files,
                    'focus': 'general optimization'
                }
            })
        
        return tasks
    
    async def get_task_results(self, task_ids: List[str] = None) -> Dict[str, Any]:
        """Get results from completed tasks"""
        if task_ids:
            tasks = [self.coordinator.get_task_status(tid) for tid in task_ids]
            tasks = [t for t in tasks if t is not None]
        else:
            tasks = self.coordinator.get_completed_tasks()
        
        results = {}
        for task in tasks:
            if task.status.value == 'completed' and task.result:
                results[task.id] = {
                    'type': task.type.value,
                    'agent': task.assigned_agent,
                    'result': task.result,
                    'duration': task.completed_at - task.created_at if task.completed_at else None
                }
        
        return results
    
    def get_progress_summary(self) -> str:
        """Get a human-readable progress summary for Claude"""
        report = self.coordinator.generate_progress_report()
        
        summary = f"""
Development Progress Summary:
============================
Total Tasks: {report['total_tasks']}
Completed: {report['completed']} ({report['completion_rate']:.1%})
In Progress: {report['in_progress']}
Failed: {report['failed']}
Pending: {report['pending']}

Active Agents:
"""
        for agent_id, current_tasks in report['agents'].items():
            summary += f"  {agent_id}: {current_tasks} active tasks\n"
        
        return summary
    
    async def emergency_stop(self):
        """Stop all current tasks (for debugging)"""
        self.log("Emergency stop requested")
        # Implementation would stop all running tasks
        pass
    
    def export_session_log(self) -> str:
        """Export session activities for review"""
        return "\n".join(self.session_log)

# Convenience functions for Claude to use directly
claude_interface = ClaudeInterface()

async def setup_development_network(use_yaml_config: bool = True, custom_endpoints: List[Dict[str, str]] = None):
    """
    Setup function for Claude to configure the agent network
    
    Args:
        use_yaml_config: If True, load from agents.yaml (recommended)
        custom_endpoints: Optional custom agent configuration (overrides YAML)
    
    Example usage:
    # Use YAML configuration (recommended)
    await setup_development_network()
    
    # Use custom configuration
    await setup_development_network(False, [
        {
            'id': 'agent_113',
            'endpoint': 'http://192.168.1.113:11434',
            'model': 'qwen2.5-coder:latest',
            'specialty': 'kernel_dev'
        }
    ])
    """
    if use_yaml_config and not custom_endpoints:
        agents = ACTIVE_AGENT_CONFIG
        claude_interface.setup_agents(agents)
        return f"Development network configured with {len(agents)} agents from YAML config"
    elif custom_endpoints:
        claude_interface.setup_agents(custom_endpoints)
        return f"Development network configured with {len(custom_endpoints)} custom agents"
    else:
        # Fallback
        agents = ACTIVE_AGENT_CONFIG
        claude_interface.setup_agents(agents)
        return f"Development network configured with {len(agents)} fallback agents"

async def delegate_work(task_description: str, files: List[str] = None, priority: int = 3) -> str:
    """
    Main function for Claude to delegate work to the agent network
    
    Example:
    result = await delegate_work(
        "Optimize FlashAttention kernel for RDNA3 architecture",
        files=["/path/to/attention.cpp", "/path/to/kernel.h"],
        priority=5
    )
    """
    return await claude_interface.delegate_rocm_optimization(task_description, files, priority)

async def check_progress() -> str:
    """Quick progress check for Claude"""
    return claude_interface.get_progress_summary()

async def collect_results(task_ids: List[str] = None) -> Dict[str, Any]:
    """Collect completed work results"""
    return await claude_interface.get_task_results(task_ids)

# Agent configuration loader from YAML
def load_agent_config_from_yaml():
    """Load agent configuration from agents.yaml as single source of truth"""
    import yaml
    
    config_path = Path(Config.CONFIG_DIR) / 'agents.yaml'
    
    if not config_path.exists():
        raise FileNotFoundError(f"Agent configuration not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        agent_configs = []
        
        for agent_id, agent_data in yaml_config.get('agents', {}).items():
            if agent_data.get('status') == 'active':
                # Convert YAML format to claude_interface format
                config = {
                    'id': agent_id,
                    'name': agent_data.get('name', f'Agent {agent_id}'),
                    'endpoint': agent_data['endpoint'],
                    'model': agent_data['model'],
                    'specialty': map_specialization(agent_data.get('specialization', 'general')),
                    'max_concurrent': agent_data.get('performance_targets', {}).get('max_concurrent_tasks', 2),
                    'capabilities': agent_data.get('capabilities', []),
                    'hardware': agent_data.get('hardware', {}),
                    'performance_targets': agent_data.get('performance_targets', {})
                }
                agent_configs.append(config)
        
        return agent_configs
        
    except yaml.YAMLError as e:
        raise RuntimeError(f"Error parsing agents.yaml: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading agent configuration: {e}")

def map_specialization(yaml_specialization):
    """Map YAML specialization strings to claude_interface specialty types"""
    mapping = {
        'Senior Kernel Development & ROCm Optimization': 'kernel_dev',
        'Code Generation & Development Support': 'pytorch_dev',
        'Testing & Quality Assurance': 'tester',
        'Documentation & Technical Writing': 'docs_writer',
        'Performance Analysis': 'profiler'
    }
    
    # Check for keywords in specialization string
    spec_lower = yaml_specialization.lower()
    if 'kernel' in spec_lower or 'rocm' in spec_lower:
        return 'kernel_dev'
    elif 'code' in spec_lower or 'development' in spec_lower:
        return 'pytorch_dev'
    elif 'test' in spec_lower or 'quality' in spec_lower:
        return 'tester'
    elif 'doc' in spec_lower or 'writing' in spec_lower:
        return 'docs_writer'
    elif 'performance' in spec_lower or 'profil' in spec_lower:
        return 'profiler'
    else:
        return 'kernel_dev'  # Default to kernel development

def get_active_agents():
    """Get currently active agents from YAML configuration"""
    try:
        return load_agent_config_from_yaml()
    except Exception as e:
        print(f"Warning: Could not load agents.yaml ({e}), using fallback configuration")
        
        # Fallback to environment-based config for Agent 113 only
        return [
            {
                'id': 'agent_113',
                'name': 'Agent 113 Fallback',
                'endpoint': Config.AGENT_113_URL,
                'model': Config.AGENT_113_MODEL,
                'specialty': 'kernel_dev',
                'max_concurrent': 1,
                'capabilities': ['kernel_development', 'rocm_optimization'],
                'hardware': {'note': 'Fallback configuration'},
                'performance_targets': {'min_tokens_per_second': 10.0}
            }
        ]

# Load current agent configuration from YAML
ACTIVE_AGENT_CONFIG = get_active_agents()

if __name__ == "__main__":
    print("Claude Interface for Distributed AI Development")
    print("Ready to coordinate your local agent network!")
    print()
    
    # Show current agent configuration
    try:
        agents = ACTIVE_AGENT_CONFIG
        print(f"✅ Loaded {len(agents)} active agents from configuration:")
        for agent in agents:
            print(f"  - {agent['id']}: {agent['endpoint']} ({agent['model']})")
            print(f"    Specialty: {agent['specialty']}")
            print(f"    Capabilities: {', '.join(agent.get('capabilities', []))}")
        print()
        
        print("🚀 Example usage:")
        print("  # Setup network from YAML config")
        print("  await setup_development_network()")
        print()
        print("  # Delegate work to agents")
        print('  result = await delegate_work("Optimize FlashAttention kernel for RDNA3")')
        print()
        print("  # Check progress")
        print("  progress = await check_progress()")
        print()
        print("  # Collect results")
        print("  results = await collect_results()")
        
    except Exception as e:
        print(f"❌ Error loading agent configuration: {e}")
        print("Please check config/agents.yaml exists and is properly formatted")