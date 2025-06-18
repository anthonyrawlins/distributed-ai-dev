#!/usr/bin/env python3
"""
Simple Dual Agent Task Assignment
Assigns focused, manageable tasks to both Agent 113 and Agent 27
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime

async def assign_focused_tasks():
    """Assign focused, manageable ROCm tasks to both agents"""
    
    print("🤖 SIMPLE DUAL AGENT TASK ASSIGNMENT")
    print("="*60)
    
    # Task for Agent 113 (DevStral) - Architecture focus
    task_113 = """Design a simple vector addition kernel strategy for ROCm.

Requirements:
- Target RDNA3 architecture (64-thread wavefronts)
- Optimize memory access patterns
- Consider coalescing for float32 arrays
- Recommend workgroup size and layout

Keep response concise and focused on key optimization points."""

    # Task for Agent 27 (CodeLlama) - Implementation focus  
    task_27 = """Write a basic HIP vector addition kernel.

Requirements:
- __global__ kernel function: vectorAdd(float* a, float* b, float* c, int n)
- Each thread processes one element: c[i] = a[i] + b[i]
- Include bounds checking
- Add simple host launcher function

Provide working code with basic error checking."""

    # Submit both tasks in parallel
    print("📤 Submitting focused tasks to both agents...")
    
    agent_113_task = submit_task_to_agent(
        "http://192.168.1.113:11434", 
        "devstral:latest",
        "Agent 113 (DevStral)",
        task_113,
        timeout=30
    )
    
    agent_27_task = submit_task_to_agent(
        "http://192.168.1.27:11434",
        "codellama:latest", 
        "Agent 27 (CodeLlama)",
        task_27,
        timeout=30
    )
    
    # Wait for both to complete
    results = await asyncio.gather(agent_113_task, agent_27_task, return_exceptions=True)
    
    # Process results
    agent_113_result = results[0] if not isinstance(results[0], Exception) else {"error": str(results[0])}
    agent_27_result = results[1] if not isinstance(results[1], Exception) else {"error": str(results[1])}
    
    # Print results
    print_task_results("Agent 113 (DevStral Architecture)", agent_113_result)
    print_task_results("Agent 27 (CodeLlama Implementation)", agent_27_result)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_data = {
        "agent_113": agent_113_result,
        "agent_27": agent_27_result,
        "timestamp": timestamp
    }
    
    filename = f"/home/tony/AI/ROCm/simple_dual_task_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n💾 Results saved to: {filename}")
    
    # Overall assessment
    success_113 = "error" not in agent_113_result
    success_27 = "error" not in agent_27_result
    
    print(f"\n🎯 TASK COMPLETION SUMMARY:")
    print(f"   Agent 113: {'✅ SUCCESS' if success_113 else '❌ FAILED'}")
    print(f"   Agent 27: {'✅ SUCCESS' if success_27 else '❌ FAILED'}")
    
    if success_113 and success_27:
        print(f"   Status: ✅ BOTH AGENTS COMPLETED THEIR TASKS")
        print(f"   Ready: Integration of architecture design + implementation")
    else:
        print(f"   Status: ⚠️  PARTIAL SUCCESS - Review failed tasks")

async def submit_task_to_agent(endpoint: str, model: str, agent_name: str, task: str, timeout: int = 30):
    """Submit a task to a specific agent"""
    
    payload = {
        "model": model,
        "prompt": task,
        "stream": False,
        "options": {
            "num_predict": 400,  # Shorter responses
            "temperature": 0.1
        }
    }
    
    start_time = time.time()
    
    try:
        print(f"🚀 {agent_name} working on task...")
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            async with session.post(f"{endpoint}/api/generate", json=payload) as response:
                if response.status != 200:
                    return {"error": f"HTTP {response.status}"}
                
                result = await response.json()
                duration = time.time() - start_time
                
                # Calculate TPS
                eval_count = result.get('eval_count', 0)
                eval_duration = result.get('eval_duration', 0)
                tps = 0
                if eval_duration > 0:
                    tps = eval_count / (eval_duration / 1000000000)
                
                print(f"✅ {agent_name} completed in {duration:.1f}s ({tps:.1f} TPS)")
                
                return {
                    "agent": agent_name,
                    "response": result.get('response', ''),
                    "duration": duration,
                    "tokens_per_second": tps,
                    "eval_count": eval_count,
                    "status": "completed"
                }
                
    except Exception as e:
        print(f"❌ {agent_name} failed: {e}")
        return {"error": str(e), "agent": agent_name}

def print_task_results(agent_name: str, result: dict):
    """Print formatted task results"""
    print(f"\n📋 {agent_name.upper()} RESULTS:")
    print("-" * 50)
    
    if "error" in result:
        print(f"❌ Status: FAILED")
        print(f"   Error: {result['error']}")
        return
    
    print(f"✅ Status: COMPLETED")
    print(f"⏱️  Duration: {result.get('duration', 0):.1f}s")
    print(f"⚡ Performance: {result.get('tokens_per_second', 0):.1f} TPS")
    print(f"📝 Tokens: {result.get('eval_count', 0)}")
    
    response = result.get('response', '')
    if len(response) > 200:
        print(f"💬 Response: {response[:200]}...")
    else:
        print(f"💬 Response: {response}")

if __name__ == "__main__":
    asyncio.run(assign_focused_tasks())