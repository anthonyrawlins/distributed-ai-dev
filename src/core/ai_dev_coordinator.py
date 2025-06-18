#!/usr/bin/env python3
"""
AI Development Coordinator
Orchestrates multiple Ollama agents for distributed ROCm development
"""

import asyncio
import aiohttp
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum

class AgentType(Enum):
    KERNEL_DEV = "kernel_dev"
    PYTORCH_DEV = "pytorch_dev" 
    PROFILER = "profiler"
    DOCS_WRITER = "docs_writer"
    TESTER = "tester"

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
    specialty: AgentType
    max_concurrent: int = 2
    current_tasks: int = 0

@dataclass
class Task:
    id: str
    type: AgentType
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
        
        # Agent prompts for specialization
        self.agent_prompts = {
            AgentType.KERNEL_DEV: """You are an expert GPU kernel developer specializing in AMD ROCm/HIP. 
Your focus is on high-performance computing with deep knowledge of:
- C++, HIP, and CUDA kernel development
- RDNA3/CDNA3 GPU architectures
- Memory coalescing and occupancy optimization
- Composable Kernel (CK) framework
- Performance analysis with rocprof

Always provide:
1. Optimized, production-ready code
2. Performance analysis of changes
3. Memory access pattern explanations
4. Compatibility notes for different GPU generations

Format your response as JSON with 'code', 'explanation', and 'performance_notes' fields.""",

            AgentType.PYTORCH_DEV: """You are a PyTorch expert specializing in ROCm backend integration.
Your expertise includes:
- Python and PyTorch internals
- ROCm backend development
- Autograd compatibility
- TunableOp configurations
- HuggingFace integration

Always ensure:
1. API compatibility with upstream PyTorch
2. Proper error handling and validation
3. Documentation strings
4. Test cases for new functionality

Format your response as JSON with 'code', 'tests', 'documentation', and 'integration_notes' fields.""",

            AgentType.PROFILER: """You are a performance analysis expert for GPU computing.
Your specialties include:
- rocprof and rocm-smi analysis
- Memory bandwidth optimization
- Kernel occupancy analysis  
- Benchmark development
- Performance regression detection

Always provide:
1. Detailed performance metrics
2. Bottleneck identification
3. Optimization recommendations
4. Comparative analysis vs baselines

Format your response as JSON with 'analysis', 'metrics', 'bottlenecks', and 'recommendations' fields.""",

            AgentType.DOCS_WRITER: """You are a technical documentation specialist for ML/GPU computing.
Your focus areas:
- Clear, accurate API documentation
- Tutorial and example creation
- Installation and setup guides
- Performance optimization guides

Always include:
1. Clear explanations with examples
2. Code snippets that compile/run
3. Common pitfalls and solutions
4. Cross-references to related docs

Format your response as JSON with 'documentation', 'examples', 'installation_notes', and 'troubleshooting' fields.""",

            AgentType.TESTER: """You are a software testing expert for GPU/ML applications.
Your specialties:
- Unit and integration test development
- Performance regression testing
- Cross-platform compatibility testing
- CI/CD pipeline development

Always provide:
1. Comprehensive test coverage
2. Performance benchmarks
3. Edge case handling
4. Automated test scripts

Format your response as JSON with 'tests', 'benchmarks', 'edge_cases', and 'ci_config' fields."""
        }
    
    def add_agent(self, agent: Agent):
        """Register a new agent"""
        self.agents[agent.id] = agent
        print(f"Registered agent {agent.id} ({agent.specialty.value}) at {agent.endpoint}")
    
    def create_task(self, task_type: AgentType, context: Dict, priority: int = 3) -> Task:
        """Create a new development task"""
        task_id = f"{task_type.value}_{int(time.time())}"
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
    
    def get_available_agent(self, task_type: AgentType) -> Optional[Agent]:
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
        
        prompt = self.agent_prompts[task.type]
        
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
        id="kernel_dev_1",
        endpoint="http://machine1:11434",
        model="codellama:34b",
        specialty=AgentType.KERNEL_DEV
    ))
    
    coordinator.add_agent(Agent(
        id="pytorch_dev_1", 
        endpoint="http://machine2:11434",
        model="deepseek-coder:33b",
        specialty=AgentType.PYTORCH_DEV
    ))
    
    # Create example tasks
    kernel_task = coordinator.create_task(
        AgentType.KERNEL_DEV,
        {
            "objective": "Optimize FlashAttention kernel for RDNA3",
            "input_file": "/path/to/attention.cpp",
            "constraints": ["Maintain backward compatibility", "Target 256 head dimensions"],
            "reference": "https://arxiv.org/abs/2307.08691"
        },
        priority=5
    )
    
    pytorch_task = coordinator.create_task(
        AgentType.PYTORCH_DEV,
        {
            "objective": "Integrate optimized attention into PyTorch",
            "base_code": "torch.nn.functional.scaled_dot_product_attention",
            "requirements": ["ROCm backend support", "Autograd compatibility"]
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
    print("Ready to orchestrate distributed ROCm development")
    
    # Run demo
    # asyncio.run(demo_coordination())