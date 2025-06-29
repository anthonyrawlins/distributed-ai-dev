#!/usr/bin/env python3
"""
AI Development Coordinator
Orchestrates multiple Ollama agents for distributed software development
"""

import asyncio
import aiohttp
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Agent:
    id: str
    endpoint: str
    model: str
    specialty: str
    max_concurrent: int = 2
    current_tasks: int = 0

@dataclass
class Task:
    id: str
    type: str
    priority: int  # 1-5, 5 being highest
    context: Dict[str, Any]
    expected_output: str
    max_tokens: int = 4000
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: Optional[str] = None
    result: Optional[Dict] = None
    created_at: float = None
    completed_at: Optional[float] = None

class AIDevCoordinator:
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.tasks: Dict[str, Task] = {}
        self.task_queue: List[Task] = []

    def add_agent(self, agent: Agent):
        """Register a new agent"""
        self.agents[agent.id] = agent
        print(f"Registered agent {agent.id} ({agent.specialty}) at {agent.endpoint}")

    def create_task(self, task_type: str, context: Dict, priority: int = 3) -> Task:
        """Create a new development task"""
        task_id = f"{task_type}_{int(time.time())}"
        task = Task(
            id=task_id,
            type=task_type,
            priority=priority,
            context=context,
            expected_output="structured_json_response",
            created_at=time.time()
        )
        self.tasks[task_id] = task
        self.task_queue.append(task)
        self.task_queue.sort(key=lambda t: t.priority, reverse=True)
        print(f"Created task {task_id} with priority {priority}")
        return task

    def get_available_agent(self, task_type: str) -> Optional[Agent]:
        """Find an available agent for the task type"""
        available_agents = [
            agent for agent in self.agents.values()
            if agent.specialty == task_type and agent.current_tasks < agent.max_concurrent
        ]
        return available_agents[0] if available_agents else None

    async def execute_task(self, task: Task, agent: Agent) -> Dict:
        """Execute a task on a specific agent"""
        agent.current_tasks += 1
        task.status = TaskStatus.IN_PROGRESS
        task.assigned_agent = agent.id

        # Construct a generic prompt
        prompt = f"""You are an AI assistant with the specialty: {agent.specialty}.
Your task is to complete the objective described in the TASK CONTEXT.
You must respond in a structured JSON format."""

        # Construct the full prompt with context
        full_prompt = f"""{prompt}

TASK CONTEXT:
{json.dumps(task.context, indent=2)}

Please complete this task and respond in the specified JSON format."""

        payload = {
            "model": agent.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": task.max_tokens
            }
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{agent.endpoint}/api/generate", json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        task.result = result
                        task.status = TaskStatus.COMPLETED
                        task.completed_at = time.time()
                        print(f"Task {task.id} completed by {agent.id}")
                        return result
                    else:
                        raise Exception(f"HTTP {response.status}: {await response.text()}")

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.result = {"error": str(e)}
            print(f"Task {task.id} failed: {e}")
            return {"error": str(e)}

        finally:
            agent.current_tasks -= 1

    async def process_queue(self):
        """Process the task queue with available agents"""
        while self.task_queue:
            pending_tasks = [t for t in self.task_queue if t.status == TaskStatus.PENDING]
            if not pending_tasks:
                break

            active_tasks = []

            for task in pending_tasks[:]:  # Copy to avoid modification during iteration
                agent = self.get_available_agent(task.type)
                if agent:
                    self.task_queue.remove(task)
                    active_tasks.append(self.execute_task(task, agent))

            if active_tasks:
                await asyncio.gather(*active_tasks, return_exceptions=True)
            else:
                # No available agents, wait a bit
                await asyncio.sleep(1)

    def get_task_status(self, task_id: str) -> Optional[Task]:
        """Get status of a specific task"""
        return self.tasks.get(task_id)

    def get_completed_tasks(self) -> List[Task]:
        """Get all completed tasks"""
        return [task for task in self.tasks.values() if task.status == TaskStatus.COMPLETED]

    def generate_progress_report(self) -> Dict:
        """Generate a progress report"""
        total_tasks = len(self.tasks)
        completed = len([t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED])
        failed = len([t for t in self.tasks.values() if t.status == TaskStatus.FAILED])
        in_progress = len([t for t in self.tasks.values() if t.status == TaskStatus.IN_PROGRESS])

        return {
            "total_tasks": total_tasks,
            "completed": completed,
            "failed": failed,
            "in_progress": in_progress,
            "pending": total_tasks - completed - failed - in_progress,
            "completion_rate": completed / total_tasks if total_tasks > 0 else 0,
            "agents": {agent.id: agent.current_tasks for agent in self.agents.values()}
        }

# Example usage and testing functions
async def demo_coordination():
    """Demonstrate the coordination system"""
    coordinator = AIDevCoordinator()

    # Add example agents (you'll replace with your actual endpoints)
    coordinator.add_agent(Agent(
        id="react_developer_1",
        endpoint="http://machine1:11434",
        model="starcoder2:15b",
        specialty="react_developer"
    ))

    coordinator.add_agent(Agent(
        id="python_backend_1",
        endpoint="http://machine2:11434",
        model="deepseek-coder:33b",
        specialty="python_backend"
    ))

    # Create example tasks
    react_task = coordinator.create_task(
        "react_developer",
        {
            "objective": "Create a React component for a login form.",
            "requirements": ["Use functional components and hooks.", "Include email and password fields.", "Add a submit button."],
            "files": {
                "LoginForm.tsx": ""
            }
        },
        priority=5
    )

    python_task = coordinator.create_task(
        "python_backend",
        {
            "objective": "Create a FastAPI endpoint for user login.",
            "requirements": ["Endpoint should be at /login.", "Accept POST requests.", "Validate email and password."],
            "files": {
                "main.py": "from fastapi import FastAPI\n\napp = FastAPI()\n"
            }
        },
        priority=4
    )

    # Process the queue
    await coordinator.process_queue()

    # Generate report
    report = coordinator.generate_progress_report()
    print("\nProgress Report:")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    print("AI Development Coordinator v1.0")
    print("Ready to orchestrate distributed development")

    # Run demo
    # asyncio.run(demo_coordination())
