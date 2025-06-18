#!/usr/bin/env python3
"""
Claude Interface for AI Development Coordination
Provides easy integration for Claude to manage the distributed development system
"""

import asyncio
import json
from typing import List, Dict, Any
from ai_dev_coordinator import AIDevCoordinator, Agent, AgentType, Task

class ClaudeInterface:
    def __init__(self):
        self.coordinator = AIDevCoordinator()
        self.session_log = []
    
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
        """Log session activities"""
        self.session_log.append(f"{asyncio.get_event_loop().time()}: {message}")
        print(f"[COORDINATOR] {message}")
    
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

async def setup_development_network(endpoints: List[Dict[str, str]]):
    """
    Quick setup function for Claude to configure the agent network
    
    Example usage:
    await setup_development_network([
        {
            'id': 'kernel_dev_1',
            'endpoint': 'http://192.168.1.100:11434',
            'model': 'codellama:34b',
            'specialty': 'kernel_dev'
        },
        {
            'id': 'pytorch_dev_1', 
            'endpoint': 'http://192.168.1.101:11434',
            'model': 'deepseek-coder:33b',
            'specialty': 'pytorch_dev'
        }
    ])
    """
    claude_interface.setup_agents(endpoints)
    return f"Development network configured with {len(endpoints)} agents"

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

# Example configuration for your network
EXAMPLE_AGENT_CONFIG = [
    {
        'id': 'kernel_expert',
        'endpoint': 'http://machine1:11434',
        'model': 'codellama:34b',
        'specialty': 'kernel_dev',
        'max_concurrent': 2
    },
    {
        'id': 'pytorch_specialist', 
        'endpoint': 'http://machine2:11434',
        'model': 'deepseek-coder:33b',
        'specialty': 'pytorch_dev',
        'max_concurrent': 2
    },
    {
        'id': 'performance_analyzer',
        'endpoint': 'http://machine3:11434', 
        'model': 'qwen2.5-coder:32b',
        'specialty': 'profiler',
        'max_concurrent': 1
    },
    {
        'id': 'devstral_architect',
        'endpoint': 'http://192.168.1.113:11434',
        'model': 'devstral:latest',
        'specialty': 'kernel_dev',
        'max_concurrent': 1
    }
]

if __name__ == "__main__":
    print("Claude Interface for Distributed AI Development")
    print("Ready to coordinate your local agent network!")
    
    # Example usage:
    # asyncio.run(setup_development_network(EXAMPLE_AGENT_CONFIG))
    # result = asyncio.run(delegate_work("Optimize VAE decoder for stable diffusion"))
    # print(result)