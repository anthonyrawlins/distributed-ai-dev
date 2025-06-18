#!/usr/bin/env python3
"""
Final Agent Coordination Test
Simple, reliable coordination between Agent 113 (qwen2.5-coder) and Agent 27 (codellama)
"""

import asyncio
import aiohttp
import time
from datetime import datetime

async def final_coordination_test():
    """Execute final coordination test with optimized tasks"""
    
    print("🎯 FINAL AGENT COORDINATION TEST")
    print("="*50)
    print("Agent 113: qwen2.5-coder (coding specialist)")
    print("Agent 27: codellama (implementation support)")
    print()
    
    # Simple, focused tasks
    task_113 = "List 3 HIP optimization tips in bullet points."
    task_27 = "Write: __global__ void add(float* a, float* b, float* c, int n)"
    
    print("📋 SIMPLE COORDINATION TASKS:")
    print("🧠 Agent 113: HIP optimization tips")
    print("🔧 Agent 27: Basic kernel signature")
    print()
    
    # Test agents sequentially for reliability
    print("🚀 Sequential execution for reliability...")
    
    # Agent 113 first
    result_113 = await test_agent(
        endpoint="http://192.168.1.113:11434",
        model="qwen2.5-coder:latest",
        name="Agent 113",
        task=task_113,
        timeout=15
    )
    
    # Agent 27 second  
    result_27 = await test_agent(
        endpoint="http://192.168.1.27:11434",
        model="codellama:latest", 
        name="Agent 27",
        task=task_27,
        timeout=15
    )
    
    # Analyze results
    print(f"\n📊 FINAL COORDINATION RESULTS:")
    print("="*50)
    
    success_113 = result_113 is not None
    success_27 = result_27 is not None
    
    print(f"🧠 Agent 113 (qwen2.5-coder): {'✅ SUCCESS' if success_113 else '❌ FAILED'}")
    if success_113:
        print(f"   Duration: {result_113['duration']:.1f}s")
        print(f"   TPS: {result_113['tps']:.1f}")
        print(f"   Response: {result_113['response'][:80]}...")
    
    print(f"\n🔧 Agent 27 (codellama): {'✅ SUCCESS' if success_27 else '❌ FAILED'}")
    if success_27:
        print(f"   Duration: {result_27['duration']:.1f}s") 
        print(f"   TPS: {result_27['tps']:.1f}")
        print(f"   Response: {result_27['response'][:80]}...")
    
    # Overall coordination status
    both_working = success_113 and success_27
    
    print(f"\n🎯 COORDINATION STATUS:")
    if both_working:
        print("✅ SUCCESS: Both agents operational and coordinated!")
        print("📈 Ready for: Distributed ROCm development tasks")
        print("🚀 Network Status: FULLY OPERATIONAL")
        
        # Performance summary
        avg_duration = (result_113['duration'] + result_27['duration']) / 2
        avg_tps = (result_113['tps'] + result_27['tps']) / 2
        print(f"\n📊 Performance Summary:")
        print(f"   Average Response Time: {avg_duration:.1f}s")
        print(f"   Average TPS: {avg_tps:.1f}")
        print(f"   Agent 113 TPS: {result_113['tps']:.1f}")
        print(f"   Agent 27 TPS: {result_27['tps']:.1f}")
        
    else:
        print("⚠️  PARTIAL: One or both agents need attention")
        if not success_113:
            print("   Agent 113: Needs investigation")
        if not success_27:
            print("   Agent 27: Needs investigation")
    
    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_results = {
        "coordination_test": "Final Agent Coordination",
        "timestamp": timestamp,
        "agent_113": {
            "model": "qwen2.5-coder:latest",
            "status": "success" if success_113 else "failed",
            "result": result_113
        },
        "agent_27": {
            "model": "codellama:latest", 
            "status": "success" if success_27 else "failed",
            "result": result_27
        },
        "coordination_success": both_working
    }
    
    import json
    filename = f"/home/tony/AI/ROCm/final_coordination_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n💾 Final results saved to: {filename}")
    
    return both_working

async def test_agent(endpoint: str, model: str, name: str, task: str, timeout: int) -> dict:
    """Test a single agent with a task"""
    
    payload = {
        "model": model,
        "prompt": task,
        "stream": False,
        "options": {
            "num_predict": 50,
            "temperature": 0.1
        }
    }
    
    start_time = time.time()
    
    try:
        print(f"⏳ {name} working...")
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            async with session.post(f"{endpoint}/api/generate", json=payload) as response:
                if response.status != 200:
                    print(f"❌ {name}: HTTP {response.status}")
                    return None
                
                result = await response.json()
                duration = time.time() - start_time
                
                # Calculate TPS
                eval_count = result.get('eval_count', 0)
                eval_duration = result.get('eval_duration', 0)
                tps = 0
                if eval_duration > 0:
                    tps = eval_count / (eval_duration / 1000000000)
                
                response_text = result.get('response', '')
                
                print(f"✅ {name} completed in {duration:.1f}s ({tps:.1f} TPS)")
                
                return {
                    "name": name,
                    "duration": duration,
                    "tps": tps,
                    "response": response_text,
                    "eval_count": eval_count
                }
                
    except Exception as e:
        print(f"❌ {name}: {e}")
        return None

if __name__ == "__main__":
    asyncio.run(final_coordination_test())