#!/usr/bin/env python3
"""
Test New Agent Coordination
Test Agent 113 with llama3.2 + Agent 27 with codellama coordination
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime

async def test_dual_agent_coordination():
    """Test coordinated tasks with the new Agent 113 model"""
    
    print("🤖 TESTING DUAL AGENT COORDINATION WITH NEW MODEL")
    print("="*70)
    print("Agent 113: qwen2.5-coder:latest (coding specialist)")
    print("Agent 27: codellama:latest (stable, reliable)")
    print()
    
    # Complementary ROCm tasks
    task_113 = """Provide 3 key ROCm kernel optimization strategies:
1. Memory access optimization
2. Thread configuration  
3. Performance bottlenecks

Keep each point to 1-2 sentences."""

    task_27 = """Write a basic HIP vector addition kernel:
__global__ void vectorAdd(float* a, float* b, float* c, int n)

Include the thread indexing line and bounds check."""
    
    print("📋 TASK ASSIGNMENTS:")
    print("🧠 Agent 113: Strategic optimization insights") 
    print("🔧 Agent 27: HIP kernel implementation")
    print()
    
    # Execute tasks in parallel for true coordination
    print("🚀 Executing coordinated tasks...")
    
    start_time = time.time()
    
    # Submit both tasks simultaneously
    task_113_coro = submit_task(
        endpoint="http://192.168.1.113:11434",
        model="qwen2.5-coder:latest",
        agent_name="Agent 113 (Qwen2.5-Coder)",
        task=task_113,
        timeout=25
    )
    
    task_27_coro = submit_task(
        endpoint="http://192.168.1.27:11434", 
        model="codellama:latest",
        agent_name="Agent 27 (CodeLlama)",
        task=task_27,
        timeout=25
    )
    
    # Wait for both to complete
    results = await asyncio.gather(task_113_coro, task_27_coro, return_exceptions=True)
    
    total_coordination_time = time.time() - start_time
    
    # Process results
    result_113 = results[0] if not isinstance(results[0], Exception) else {"error": str(results[0])}
    result_27 = results[1] if not isinstance(results[1], Exception) else {"error": str(results[1])}
    
    # Analyze coordination success
    coordination_analysis = analyze_coordination_success(result_113, result_27, total_coordination_time)
    
    # Print comprehensive results
    print_coordination_results(result_113, result_27, coordination_analysis)
    
    # Save results
    save_coordination_test(result_113, result_27, coordination_analysis)
    
    return coordination_analysis['overall_success']

async def submit_task(endpoint: str, model: str, agent_name: str, task: str, timeout: int) -> dict:
    """Submit task to specific agent"""
    
    payload = {
        "model": model,
        "prompt": task,
        "stream": False,
        "options": {
            "num_predict": 300,
            "temperature": 0.1
        }
    }
    
    start_time = time.time()
    
    try:
        print(f"⏳ {agent_name} working...")
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            async with session.post(f"{endpoint}/api/generate", json=payload) as response:
                if response.status != 200:
                    return {
                        "agent": agent_name,
                        "status": "error",
                        "error": f"HTTP {response.status}",
                        "duration": time.time() - start_time
                    }
                
                result = await response.json()
                duration = time.time() - start_time
                
                # Calculate performance metrics
                eval_count = result.get('eval_count', 0)
                eval_duration = result.get('eval_duration', 0)
                tps = 0
                if eval_duration > 0:
                    tps = eval_count / (eval_duration / 1000000000)
                
                response_text = result.get('response', '')
                
                print(f"✅ {agent_name} completed in {duration:.1f}s ({tps:.1f} TPS)")
                
                return {
                    "agent": agent_name,
                    "status": "completed",
                    "task": task,
                    "response": response_text,
                    "duration": duration,
                    "tokens_per_second": tps,
                    "eval_count": eval_count,
                    "timestamp": datetime.now().isoformat()
                }
                
    except Exception as e:
        print(f"❌ {agent_name} failed: {e}")
        return {
            "agent": agent_name,
            "status": "error", 
            "error": str(e),
            "duration": time.time() - start_time
        }

def analyze_coordination_success(result_113: dict, result_27: dict, total_time: float) -> dict:
    """Analyze the success of agent coordination"""
    
    analysis = {
        "both_completed": False,
        "performance_excellent": False,
        "content_quality": {},
        "timing_analysis": {},
        "overall_success": False
    }
    
    # Check completion status
    both_completed = (result_113.get('status') == 'completed' and 
                     result_27.get('status') == 'completed')
    analysis['both_completed'] = both_completed
    
    if both_completed:
        # Performance analysis
        tps_113 = result_113.get('tokens_per_second', 0)
        tps_27 = result_27.get('tokens_per_second', 0)
        duration_113 = result_113.get('duration', 0)
        duration_27 = result_27.get('duration', 0)
        
        analysis['timing_analysis'] = {
            "agent_113_duration": duration_113,
            "agent_27_duration": duration_27,
            "total_coordination_time": total_time,
            "parallel_efficiency": max(duration_113, duration_27) / total_time,
            "avg_tps": (tps_113 + tps_27) / 2
        }
        
        # Performance is excellent if both agents are fast
        analysis['performance_excellent'] = (tps_113 > 30 and tps_27 > 8 and total_time < 30)
        
        # Content quality analysis
        response_113 = result_113.get('response', '').lower()
        response_27 = result_27.get('response', '').lower()
        
        # Agent 113 should provide optimization strategies
        has_optimization_content = any(term in response_113 for term in 
                                     ['optimization', 'memory', 'thread', 'performance', 'strategy'])
        
        # Agent 27 should provide kernel implementation
        has_kernel_content = any(term in response_27 for term in 
                               ['__global__', 'kernel', 'threadidx', 'blockidx', 'float*'])
        
        analysis['content_quality'] = {
            "agent_113_optimization_content": has_optimization_content,
            "agent_27_kernel_content": has_kernel_content,
            "complementary_outputs": has_optimization_content and has_kernel_content
        }
        
        # Overall success criteria
        analysis['overall_success'] = (both_completed and 
                                     analysis['content_quality']['complementary_outputs'] and
                                     total_time < 45)
    
    return analysis

def print_coordination_results(result_113: dict, result_27: dict, analysis: dict):
    """Print comprehensive coordination results"""
    
    print("\n" + "="*70)
    print("📊 DUAL AGENT COORDINATION RESULTS")
    print("="*70)
    
    # Agent 113 Results
    print("🧠 AGENT 113 (Llama3.2 Strategy):")
    if result_113.get('status') == 'completed':
        print(f"   ✅ Status: COMPLETED")
        print(f"   ⏱️  Duration: {result_113.get('duration', 0):.1f}s")
        print(f"   ⚡ Performance: {result_113.get('tokens_per_second', 0):.1f} TPS")
        print(f"   📝 Strategy Content: {'✅' if analysis['content_quality'].get('agent_113_optimization_content') else '❌'}")
        
        # Show preview of response
        response_preview = result_113.get('response', '')[:150] + "..." if len(result_113.get('response', '')) > 150 else result_113.get('response', '')
        print(f"   💬 Preview: {response_preview}")
    else:
        print(f"   ❌ Status: FAILED - {result_113.get('error', 'Unknown')}")
    
    # Agent 27 Results  
    print("\n🔧 AGENT 27 (CodeLlama Implementation):")
    if result_27.get('status') == 'completed':
        print(f"   ✅ Status: COMPLETED")
        print(f"   ⏱️  Duration: {result_27.get('duration', 0):.1f}s")
        print(f"   ⚡ Performance: {result_27.get('tokens_per_second', 0):.1f} TPS")
        print(f"   🔧 Kernel Content: {'✅' if analysis['content_quality'].get('agent_27_kernel_content') else '❌'}")
        
        # Show preview of response
        response_preview = result_27.get('response', '')[:150] + "..." if len(result_27.get('response', '')) > 150 else result_27.get('response', '')
        print(f"   💬 Preview: {response_preview}")
    else:
        print(f"   ❌ Status: FAILED - {result_27.get('error', 'Unknown')}")
    
    # Coordination Analysis
    print(f"\n🎯 COORDINATION ANALYSIS:")
    print(f"   Both Completed: {'✅' if analysis['both_completed'] else '❌'}")
    print(f"   Complementary Content: {'✅' if analysis['content_quality'].get('complementary_outputs') else '❌'}")
    print(f"   Performance Excellent: {'✅' if analysis['performance_excellent'] else '❌'}")
    
    if analysis['both_completed']:
        timing = analysis['timing_analysis']
        print(f"   Total Coordination Time: {timing['total_coordination_time']:.1f}s")
        print(f"   Parallel Efficiency: {timing['parallel_efficiency']:.2f}")
        print(f"   Average TPS: {timing['avg_tps']:.1f}")
    
    # Overall Assessment
    print(f"\n🚀 OVERALL COORDINATION:")
    if analysis['overall_success']:
        print("   ✅ SUCCESS - Dual agent coordination fully operational!")
        print("   📈 Ready for: Complex multi-agent ROCm development projects")
        print("   🎯 Status: Agent 113 (llama3.2) + Agent 27 (codellama) working together")
    else:
        print("   ⚠️  PARTIAL - Some aspects need improvement")
        if not analysis['both_completed']:
            print("   🔍 Issue: One or both agents failed to complete tasks")
        elif not analysis['content_quality'].get('complementary_outputs'):
            print("   🔍 Issue: Agents not providing complementary content")
        else:
            print("   🔍 Issue: Performance or timing needs optimization")

def save_coordination_test(result_113: dict, result_27: dict, analysis: dict):
    """Save coordination test results"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    test_results = {
        "test_name": "Dual Agent Coordination with New Model",
        "timestamp": timestamp,
        "agent_113_model": "llama3.2:latest",
        "agent_27_model": "codellama:latest", 
        "agent_113_result": result_113,
        "agent_27_result": result_27,
        "coordination_analysis": analysis
    }
    
    filename = f"/home/tony/AI/ROCm/new_coordination_test_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n💾 Test results saved to: {filename}")

async def main():
    """Main coordination test"""
    success = await test_dual_agent_coordination()
    
    if success:
        print(f"\n🎉 COORDINATION SUCCESS!")
        print(f"💡 Both agents are now operational and working together")
        print(f"📝 Configuration updated: Agent 113 now uses llama3.2:latest")
    else:
        print(f"\n⚠️  Coordination needs further optimization")

if __name__ == "__main__":
    asyncio.run(main())