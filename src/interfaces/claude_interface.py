#!/usr/bin/env python3
"""
Claude Interface for AI Development Coordination
Provides easy integration for Claude to manage the distributed development system
"""

import asyncio
import json
from typing import List, Dict, Any
from ai_dev_coordinator import AIDevCoordinator, Agent, Task

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
                specialty=config['specialty'],
                max_concurrent=config.get('max_concurrent', 2)
            )
            self.coordinator.add_agent(agent)
            self.log(f"Agent {agent.id} registered")

    def log(self, message: str):
        """Log session activities"""
        self.session_log.append(f"{asyncio.get_event_loop().time()}: {message}")
        print(f"[COORDINATOR] {message}")

    async def delegate_task(self,
                              specialization: str,
                              task_description: str,
                              files: Dict[str, str] = None,
                              priority: int = 3) -> str:
        """
        High-level function for Claude to delegate work to the agent network.
        Returns the task ID for tracking.
        """
        files = files or {}
        context = {
            "objective": task_description,
            "files": files,
        }

        task = self.coordinator.create_task(
            specialization,
            context,
            priority
        )

        self.log(f"Created task {task.id} for: {task_description}")

        # Process tasks asynchronously
        await self.coordinator.process_queue()

        return f"Task created: {task.id}"

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
                    'type': task.type,
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
            'id': 'react_developer_1',
            'endpoint': 'http://192.168.1.100:11434',
            'model': 'starcoder2:15b',
            'specialty': 'react_developer'
        },
        {
            'id': 'python_backend_1',
            'endpoint': 'http://192.168.1.101:11434',
            'model': 'deepseek-coder:33b',
            'specialty': 'python_backend'
        }
    ])
    """
    claude_interface.setup_agents(endpoints)
    return f"Development network configured with {len(endpoints)} agents"

async def delegate_work(specialization: str, task_description: str, files: Dict[str, str] = None, priority: int = 3) -> str:
    """
    Main function for Claude to delegate work to the agent network
    
    Example:
    result = await delegate_work(
        "react_developer",
        "Create a login form component.",
        files={"LoginForm.css": ".form { color: blue; }"},
        priority=5
    )
    """
    return await claude_interface.delegate_task(specialization, task_description, files, priority)

async def check_progress() -> str:
    """Quick progress check for Claude"""
    return claude_interface.get_progress_summary()

async def collect_results(task_ids: List[str] = None) -> Dict[str, Any]:
    """Collect completed work results"""
    return await claude_interface.get_task_results(task_ids)

# Example configuration for your network
EXAMPLE_AGENT_CONFIG = [
    {
        'id': 'frontend_expert',
        'endpoint': 'http://machine1:11434',
        'model': 'starcoder2:15b',
        'specialty': 'frontend_developer',
        'max_concurrent': 2
    },
    {
        'id': 'backend_specialist',
        'endpoint': 'http://machine2:11434',
        'model': 'deepseek-coder:33b',
        'specialty': 'backend_developer',
        'max_concurrent': 2
    },
    {
        'id': 'documentation_writer',
        'endpoint': 'http://machine3:11434',
        'model': 'qwen2.5-coder:32b',
        'specialty': 'technical_writer',
        'max_concurrent': 1
    }
]

if __name__ == "__main__":
    print("Claude Interface for Distributed AI Development")
    print("Ready to coordinate your local agent network!")

    # Example usage:
    # asyncio.run(setup_development_network(EXAMPLE_AGENT_CONFIG))
    # result = asyncio.run(delegate_work("frontend_developer", "Create a button component."))
    # print(result)
