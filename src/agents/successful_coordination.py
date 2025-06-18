#!/usr/bin/env python3
"""
Successful Agent Coordination
Assigns appropriately scoped ROCm tasks to both agents with proven success
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime

async def coordinate_rocm_development():
    """Coordinate focused ROCm development between both agents"""
    
    print("🎯 SUCCESSFUL AGENT COORDINATION FOR ROCm DEVELOPMENT")
    print("="*70)
    print("Assigning complementary tasks based on each agent's strengths")
    print()
    
    # Agent 113 (DevStral) - Strategic/Architectural task
    task_113 = """Provide 3 key optimization strategies for ROCm matrix multiplication:

1. Memory access pattern optimization
2. Workgroup/wavefront configuration  
3. Performance bottleneck identification

Keep each point concise (2-3 sentences)."""

    # Agent 27 (CodeLlama) - Implementation task
    task_27 = """Write a simple HIP function signature and one line of implementation:

__global__ void matrixMul(float* A, float* B, float* C, int N)
{
    // Add the basic thread indexing line here
}

Just provide the function with the indexing line."""

    print("📋 TASK ASSIGNMENTS:")
    print("🏗️  Agent 113 (DevStral): Strategic optimization insights")
    print("🔧 Agent 27 (CodeLlama): Basic implementation structure")
    print()
    
    # Execute tasks sequentially for better reliability
    print("🚀 Starting coordinated execution...")
    
    # Agent 113 first (architecture/strategy)
    result_113 = await execute_agent_task(
        agent_id="113",
        name="DevStral Senior Architect",
        endpoint="http://192.168.1.113:11434",
        model="devstral:latest",
        task=task_113,
        timeout=25
    )
    
    # Agent 27 second (implementation)
    result_27 = await execute_agent_task(
        agent_id="27", 
        name="CodeLlama Development Assistant",
        endpoint="http://192.168.1.27:11434",
        model="codellama:latest",
        task=task_27,
        timeout=20
    )
    
    # Coordination analysis
    coordination_results = analyze_coordination(result_113, result_27)
    
    # Save comprehensive results
    save_coordination_results(result_113, result_27, coordination_results)
    
    # Print final summary
    print_coordination_summary(result_113, result_27, coordination_results)

async def execute_agent_task(agent_id: str, name: str, endpoint: str, model: str, task: str, timeout: int) -> dict:
    """Execute a task on a specific agent"""
    
    payload = {
        "model": model,
        "prompt": task,
        "stream": False,
        "options": {
            "num_predict": 200,
            "temperature": 0.1,
            "top_p": 0.9
        }
    }
    
    start_time = time.time()
    
    try:
        print(f"⏳ Agent {agent_id} ({name}) working...")
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            async with session.post(f"{endpoint}/api/generate", json=payload) as response:
                if response.status != 200:
                    return {
                        "agent_id": agent_id,
                        "name": name,
                        "status": "error",
                        "error": f"HTTP {response.status}",
                        "duration": time.time() - start_time
                    }
                
                result = await response.json()
                duration = time.time() - start_time
                
                # Extract performance metrics
                eval_count = result.get('eval_count', 0)
                eval_duration = result.get('eval_duration', 0)
                tps = 0
                if eval_duration > 0:
                    tps = eval_count / (eval_duration / 1000000000)
                
                response_text = result.get('response', '')
                
                print(f"✅ Agent {agent_id} completed in {duration:.1f}s ({tps:.1f} TPS)")
                print(f"📝 Response preview: {response_text[:100]}...")
                
                return {
                    "agent_id": agent_id,
                    "name": name,
                    "status": "completed",
                    "task": task,
                    "response": response_text,
                    "duration": duration,
                    "tokens_per_second": tps,
                    "eval_count": eval_count,
                    "timestamp": datetime.now().isoformat()
                }
                
    except Exception as e:
        print(f"❌ Agent {agent_id} failed: {e}")
        return {
            "agent_id": agent_id,
            "name": name,
            "status": "error",
            "error": str(e),
            "duration": time.time() - start_time
        }

def analyze_coordination(result_113: dict, result_27: dict) -> dict:
    """Analyze the coordination between both agents"""
    
    analysis = {
        "coordination_success": False,
        "both_completed": False,
        "complementary_output": False,
        "performance_summary": {},
        "quality_assessment": {}
    }
    
    # Check if both completed
    both_completed = (result_113.get('status') == 'completed' and 
                     result_27.get('status') == 'completed')
    analysis["both_completed"] = both_completed
    
    if both_completed:
        # Performance analysis
        analysis["performance_summary"] = {
            "agent_113_tps": result_113.get('tokens_per_second', 0),
            "agent_27_tps": result_27.get('tokens_per_second', 0),
            "total_duration": result_113.get('duration', 0) + result_27.get('duration', 0),
            "avg_tps": (result_113.get('tokens_per_second', 0) + result_27.get('tokens_per_second', 0)) / 2
        }
        
        # Quality assessment
        response_113 = result_113.get('response', '')
        response_27 = result_27.get('response', '')
        
        # Agent 113 should provide strategic insights
        has_strategy = any(word in response_113.lower() for word in 
                          ['optimization', 'strategy', 'memory', 'performance', 'workgroup'])
        
        # Agent 27 should provide implementation
        has_implementation = any(word in response_27.lower() for word in 
                               ['__global__', 'thread', 'index', 'blockidx', 'threadidx'])
        
        analysis["quality_assessment"] = {
            "agent_113_strategic_content": has_strategy,
            "agent_27_implementation_content": has_implementation,
            "response_lengths": {
                "agent_113": len(response_113),
                "agent_27": len(response_27)
            }
        }
        
        analysis["complementary_output"] = has_strategy and has_implementation
        analysis["coordination_success"] = both_completed and analysis["complementary_output"]
    
    return analysis

def save_coordination_results(result_113: dict, result_27: dict, analysis: dict):
    """Save coordination results to file"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    comprehensive_results = {
        "coordination_project": "ROCm Matrix Multiplication Development",
        "timestamp": timestamp,
        "agent_113_result": result_113,
        "agent_27_result": result_27,
        "coordination_analysis": analysis
    }
    
    filename = f"/home/tony/AI/ROCm/successful_coordination_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    print(f"\n💾 Coordination results saved to: {filename}")

def print_coordination_summary(result_113: dict, result_27: dict, analysis: dict):
    """Print comprehensive coordination summary"""
    
    print("\n" + "="*70)
    print("📊 COORDINATION RESULTS SUMMARY")
    print("="*70)
    
    # Agent 113 Results
    print("🏗️  AGENT 113 (DevStral Strategy):")
    if result_113.get('status') == 'completed':
        print(f"   ✅ Status: COMPLETED")
        print(f"   ⏱️  Duration: {result_113.get('duration', 0):.1f}s")
        print(f"   ⚡ Performance: {result_113.get('tokens_per_second', 0):.1f} TPS")
        print(f"   📝 Strategic Content: {'✅' if analysis['quality_assessment'].get('agent_113_strategic_content') else '❌'}")
    else:
        print(f"   ❌ Status: FAILED - {result_113.get('error', 'Unknown')}")
    
    # Agent 27 Results
    print("\n🔧 AGENT 27 (CodeLlama Implementation):")
    if result_27.get('status') == 'completed':
        print(f"   ✅ Status: COMPLETED")
        print(f"   ⏱️  Duration: {result_27.get('duration', 0):.1f}s")
        print(f"   ⚡ Performance: {result_27.get('tokens_per_second', 0):.1f} TPS")
        print(f"   🔧 Implementation Content: {'✅' if analysis['quality_assessment'].get('agent_27_implementation_content') else '❌'}")
    else:
        print(f"   ❌ Status: FAILED - {result_27.get('error', 'Unknown')}")
    
    # Overall Coordination Assessment
    print(f"\n🎯 COORDINATION ASSESSMENT:")
    print(f"   Both Completed: {'✅' if analysis['both_completed'] else '❌'}")
    print(f"   Complementary Output: {'✅' if analysis['complementary_output'] else '❌'}")
    print(f"   Overall Success: {'✅' if analysis['coordination_success'] else '❌'}")
    
    if analysis['coordination_success']:
        perf = analysis['performance_summary']
        print(f"\n🚀 PERFORMANCE METRICS:")
        print(f"   Combined TPS: {perf['avg_tps']:.1f} average")
        print(f"   Total Duration: {perf['total_duration']:.1f}s")
        print(f"   Agent 113 TPS: {perf['agent_113_tps']:.1f}")
        print(f"   Agent 27 TPS: {perf['agent_27_tps']:.1f}")
        
        print(f"\n✅ SUCCESS: Distributed AI development coordination achieved!")
        print(f"   📈 Both agents contributed complementary ROCm development work")
        print(f"   🎯 Ready for: Scaled multi-agent ROCm optimization projects")
    else:
        print(f"\n⚠️  Coordination needs improvement - review task design and agent capabilities")

if __name__ == "__main__":
    asyncio.run(coordinate_rocm_development())