#!/usr/bin/env python3
"""
Simple Agent 27 Test - Quick Connectivity and Basic Task
"""

import asyncio
import aiohttp
import json
import time

async def quick_test_agent_27():
    """Quick test of Agent 27 with minimal task"""
    endpoint = "http://192.168.1.27:11434"
    
    # Simple prompt
    payload = {
        "model": "codellama:latest",
        "prompt": "Write a simple 'Hello ROCm' comment in C++. Just one line.",
        "stream": False,
        "options": {
            "num_predict": 50,
            "temperature": 0.1
        }
    }
    
    print("🚀 Testing Agent 27 with simple task...")
    start_time = time.time()
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            async with session.post(f"{endpoint}/api/generate", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    duration = time.time() - start_time
                    
                    print(f"✅ Agent 27 responded in {duration:.2f}s")
                    print(f"📝 Response: {result.get('response', 'No response')}")
                    
                    # Basic performance metrics
                    if 'eval_count' in result and 'eval_duration' in result:
                        tokens = result['eval_count']
                        eval_time_ms = result['eval_duration'] / 1000000
                        tps = tokens / (eval_time_ms / 1000) if eval_time_ms > 0 else 0
                        print(f"⚡ Performance: {tps:.1f} TPS, {tokens} tokens in {eval_time_ms:.0f}ms")
                    
                    return True
                else:
                    print(f"❌ HTTP {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

async def test_rocm_knowledge():
    """Test Agent 27's ROCm knowledge with a simple question"""
    endpoint = "http://192.168.1.27:11434"
    
    payload = {
        "model": "codellama:latest", 
        "prompt": "What is the HIP equivalent of cudaMalloc? Answer in one sentence.",
        "stream": False,
        "options": {
            "num_predict": 30,
            "temperature": 0.1
        }
    }
    
    print("\n🧠 Testing ROCm knowledge...")
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as session:
            async with session.post(f"{endpoint}/api/generate", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    response_text = result.get('response', '')
                    
                    print(f"💬 Agent 27: {response_text}")
                    
                    # Check if response shows ROCm knowledge
                    has_hip_knowledge = any(term in response_text.lower() for term in ['hipmalloc', 'hip', 'rocm'])
                    print(f"🎯 ROCm Knowledge: {'✅ Good' if has_hip_knowledge else '❌ Limited'}")
                    
                    return has_hip_knowledge
                else:
                    print(f"❌ HTTP {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

async def main():
    print("🧪 QUICK AGENT 27 TEST")
    print("="*40)
    
    # Test basic connectivity
    basic_test = await quick_test_agent_27()
    
    if basic_test:
        # Test ROCm knowledge
        rocm_test = await test_rocm_knowledge()
        
        print(f"\n📊 QUICK ASSESSMENT:")
        print(f"   Connectivity: {'✅' if basic_test else '❌'}")
        print(f"   ROCm Knowledge: {'✅' if rocm_test else '❌'}")
        
        if basic_test and rocm_test:
            print("   Status: ✅ Agent 27 is ready for ROCm development tasks")
        elif basic_test:
            print("   Status: ⚠️  Agent 27 is responsive but may need ROCm context")
        else:
            print("   Status: ❌ Agent 27 needs investigation")
    else:
        print("\n❌ Agent 27 is not responsive. Check endpoint and model availability.")

if __name__ == "__main__":
    asyncio.run(main())