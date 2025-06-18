#!/usr/bin/env python3
"""
Dual Agent Coordinator
Orchestrates complementary ROCm development tasks between Agent 113 and Agent 27
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime
from typing import Dict, Any, List

class DualAgentCoordinator:
    """
    Coordinates work between Agent 113 (DevStral) and Agent 27 (CodeLlama)
    for complementary ROCm development tasks.
    """
    
    def __init__(self):
        self.agents = {
            'agent_113': {
                'name': 'DevStral Senior Architect',
                'endpoint': 'http://192.168.1.113:11434',
                'model': 'devstral:latest',
                'specialization': 'Complex kernel development and architecture',
                'timeout': 60
            },
            'agent_27': {
                'name': 'CodeLlama Development Assistant',
                'endpoint': 'http://192.168.1.27:11434', 
                'model': 'codellama:latest',
                'specialization': 'Code completion and testing support',
                'timeout': 45
            }
        }
        self.task_results = {}
        
    async def submit_task(self, agent_id: str, task_description: str, 
                         context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Submit a task to a specific agent.
        
        Args:
            agent_id: 'agent_113' or 'agent_27'
            task_description: The task to assign
            context: Additional context for the task
            
        Returns:
            Task result dictionary
        """
        agent = self.agents[agent_id]
        context = context or {}
        
        # Customize prompt based on agent specialization
        if agent_id == 'agent_113':
            # DevStral: Complex architecture and kernel design
            prompt = f"""You are a senior ROCm kernel architect working on advanced GPU optimizations.

Task: {task_description}

Focus Areas:
- Advanced kernel design and architecture decisions
- RDNA3/CDNA3 optimization strategies  
- Memory bandwidth optimization
- Performance-critical implementations
- Complex algorithmic approaches

Requirements:
- Provide architectural rationale for design decisions
- Consider GPU hardware characteristics (wavefront size, cache hierarchy)
- Focus on performance and scalability
- Include optimization insights and trade-offs

{self._format_context(context)}

Please provide a comprehensive solution with detailed explanations."""
        
        else:
            # CodeLlama: Code completion and implementation support
            prompt = f"""You are a ROCm development assistant focused on code implementation and testing.

Task: {task_description}

Focus Areas:
- Clean, well-structured code implementation
- Proper error handling and validation
- Testing and debugging support
- Code documentation and comments
- Integration with existing systems

Requirements:
- Write production-ready code with proper error handling
- Include comprehensive comments explaining the implementation
- Provide usage examples and test cases
- Focus on code clarity and maintainability

{self._format_context(context)}

Please provide a complete implementation with examples."""

        payload = {
            "model": agent['model'],
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 1500 if agent_id == 'agent_113' else 1000,
                "top_p": 0.9
            }
        }
        
        start_time = time.time()
        task_id = f"{agent_id}_{int(start_time)}"
        
        print(f"🚀 Assigning task to {agent['name']}")
        print(f"📋 Task: {task_description}")
        print(f"⏳ Agent {agent_id} is working...")
        
        try:
            timeout = aiohttp.ClientTimeout(total=agent['timeout'])
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(f"{agent['endpoint']}/api/generate", json=payload) as response:
                    if response.status != 200:
                        return {
                            'task_id': task_id,
                            'agent_id': agent_id,
                            'status': 'error',
                            'error': f'HTTP {response.status}',
                            'duration': time.time() - start_time
                        }
                    
                    result = await response.json()
                    duration = time.time() - start_time
                    
                    # Extract performance metrics
                    performance = self._extract_performance_metrics(result)
                    
                    task_result = {
                        'task_id': task_id,
                        'agent_id': agent_id,
                        'agent_name': agent['name'],
                        'task': task_description,
                        'status': 'completed',
                        'response': result.get('response', ''),
                        'duration': duration,
                        'performance': performance,
                        'timestamp': datetime.now().isoformat(),
                        'context': context
                    }
                    
                    self.task_results[task_id] = task_result
                    
                    print(f"✅ {agent['name']} completed task in {duration:.1f}s")
                    print(f"⚡ Performance: {performance.get('tokens_per_second', 0):.1f} TPS")
                    
                    return task_result
                    
        except Exception as e:
            error_result = {
                'task_id': task_id,
                'agent_id': agent_id,
                'status': 'error',
                'error': str(e),
                'duration': time.time() - start_time
            }
            self.task_results[task_id] = error_result
            print(f"❌ {agent['name']} task failed: {e}")
            return error_result
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context information for the prompt"""
        if not context:
            return ""
            
        context_str = "\nContext:\n"
        for key, value in context.items():
            context_str += f"- {key}: {value}\n"
        return context_str
    
    def _extract_performance_metrics(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract performance metrics from Ollama response"""
        metrics = {}
        
        if 'eval_count' in result and 'eval_duration' in result:
            eval_count = result['eval_count']
            eval_duration_ns = result['eval_duration']
            if eval_duration_ns > 0:
                metrics['tokens_per_second'] = eval_count / (eval_duration_ns / 1000000000)
        
        for key in ['total_duration', 'load_duration', 'prompt_eval_duration', 'eval_duration']:
            if key in result:
                metrics[f'{key}_ms'] = result[key] / 1000000
        
        if 'prompt_eval_count' in result:
            metrics['prompt_eval_count'] = result['prompt_eval_count']
        if 'eval_count' in result:
            metrics['eval_count'] = result['eval_count']
            
        return metrics
    
    async def coordinate_matrix_multiplication_project(self) -> Dict[str, Any]:
        """
        Coordinate a matrix multiplication optimization project between both agents.
        Agent 113 handles architecture, Agent 27 handles implementation and testing.
        """
        print("🎯 COORDINATING MATRIX MULTIPLICATION OPTIMIZATION PROJECT")
        print("="*80)
        
        # Task 1: Agent 113 - Architecture and algorithm design
        agent_113_task = """Design an optimized matrix multiplication kernel for ROCm that targets RDNA3 architecture.

Specifications:
- Handle matrices up to 4096x4096 elements (float32)
- Optimize for memory bandwidth and compute utilization
- Consider wavefront size (64 threads) and LDS usage
- Design tiling strategy for different matrix sizes
- Address memory coalescing patterns

Deliverables:
- High-level algorithm design and tiling strategy
- Memory access patterns and optimization rationale
- Performance expectations and bottleneck analysis
- Kernel launch configuration recommendations"""

        context_113 = {
            "target_architecture": "RDNA3 (gfx1100)",
            "wavefront_size": 64,
            "lds_size": "65KB per workgroup",
            "memory_bandwidth": "~900GB/s theoretical",
            "compute_capability": "~83 TFLOPS FP32"
        }
        
        # Task 2: Agent 27 - Implementation and testing
        agent_27_task = """Implement a basic matrix multiplication kernel in HIP based on architectural guidance.

Requirements:
- Create host and device code for matrix multiplication
- Include proper memory allocation and data transfer
- Add error checking and validation
- Implement basic tiling (e.g., 16x16 tiles)
- Provide benchmarking and correctness testing

Deliverables:
- Complete HIP kernel implementation
- Host code with memory management
- Test functions to verify correctness
- Basic performance measurement code
- Usage examples and documentation"""

        context_27 = {
            "matrix_sizes": "Start with 512x512, 1024x1024 for testing",
            "tile_size": "16x16 or 32x32 tiles",
            "data_type": "float32",
            "validation": "Compare against CPU reference implementation"
        }
        
        # Submit tasks in parallel
        print("📤 Submitting tasks to both agents...")
        task_113, task_27 = await asyncio.gather(
            self.submit_task('agent_113', agent_113_task, context_113),
            self.submit_task('agent_27', agent_27_task, context_27),
            return_exceptions=True
        )
        
        # Analyze results
        project_results = {
            'project': 'Matrix Multiplication Optimization',
            'agent_113_result': task_113 if not isinstance(task_113, Exception) else {'error': str(task_113)},
            'agent_27_result': task_27 if not isinstance(task_27, Exception) else {'error': str(task_27)},
            'coordination_timestamp': datetime.now().isoformat()
        }
        
        # Save coordination results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"/home/tony/AI/ROCm/dual_agent_coordination_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(project_results, f, indent=2)
        
        print(f"\n💾 Coordination results saved to: {results_file}")
        
        return project_results
    
    def print_coordination_summary(self, results: Dict[str, Any]):
        """Print a summary of the coordination results"""
        print("\n" + "="*80)
        print("📊 DUAL AGENT COORDINATION SUMMARY")
        print("="*80)
        
        agent_113_result = results.get('agent_113_result', {})
        agent_27_result = results.get('agent_27_result', {})
        
        print(f"🏗️  AGENT 113 (DevStral Architecture):")
        if agent_113_result.get('status') == 'completed':
            perf_113 = agent_113_result.get('performance', {})
            print(f"   Status: ✅ COMPLETED")
            print(f"   Duration: {agent_113_result.get('duration', 0):.1f}s")
            print(f"   Performance: {perf_113.get('tokens_per_second', 0):.1f} TPS")
            print(f"   Response Length: {len(agent_113_result.get('response', ''))}")
        else:
            print(f"   Status: ❌ FAILED - {agent_113_result.get('error', 'Unknown error')}")
        
        print(f"\n🔧 AGENT 27 (CodeLlama Implementation):")
        if agent_27_result.get('status') == 'completed':
            perf_27 = agent_27_result.get('performance', {})
            print(f"   Status: ✅ COMPLETED")
            print(f"   Duration: {agent_27_result.get('duration', 0):.1f}s")
            print(f"   Performance: {perf_27.get('tokens_per_second', 0):.1f} TPS")
            print(f"   Response Length: {len(agent_27_result.get('response', ''))}")
        else:
            print(f"   Status: ❌ FAILED - {agent_27_result.get('error', 'Unknown error')}")
        
        # Overall assessment
        both_completed = (agent_113_result.get('status') == 'completed' and 
                         agent_27_result.get('status') == 'completed')
        
        print(f"\n🎯 PROJECT STATUS:")
        if both_completed:
            print("   ✅ SUCCESS - Both agents completed their tasks")
            print("   📈 Ready for integration and further development")
        else:
            print("   ⚠️  PARTIAL - Some tasks failed, review required")
        
        print(f"\n📁 Next Steps:")
        if both_completed:
            print("   1. Review architectural design from Agent 113")
            print("   2. Test implementation from Agent 27")
            print("   3. Integrate design insights with implementation")
            print("   4. Optimize based on both agents' contributions")
        else:
            print("   1. Review failed tasks and retry if needed")
            print("   2. Adjust task complexity or timeout settings")
            print("   3. Consider breaking down complex tasks further")

async def main():
    """Main coordination function"""
    coordinator = DualAgentCoordinator()
    
    print("🤖 DUAL AGENT ROCm DEVELOPMENT COORDINATION")
    print("Orchestrating Agent 113 (DevStral) + Agent 27 (CodeLlama)")
    print("="*80)
    
    # Execute coordinated matrix multiplication project
    results = await coordinator.coordinate_matrix_multiplication_project()
    
    # Print summary
    coordinator.print_coordination_summary(results)
    
    print(f"\n🚀 Coordination complete! Check saved results for detailed analysis.")

if __name__ == "__main__":
    asyncio.run(main())