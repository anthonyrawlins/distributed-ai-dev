#!/usr/bin/env python3
"""
Basic Agent Assignment Test
Very simple tasks to verify both agents can handle work
"""

import asyncio
import aiohttp
import time

async def test_basic_assignment():
    """Test basic task assignment to both agents"""
    
    print("🧪 BASIC AGENT ASSIGNMENT TEST")
    print("="*50)
    
    # Very simple tasks
    task_113 = "Write one sentence about ROCm kernel optimization."
    task_27 = "Write: hipMalloc example in one line."
    
    print("📤 Testing basic assignments...")
    
    # Test Agent 113
    result_113 = await test_agent(
        "http://192.168.1.113:11434",
        "devstral:latest", 
        "Agent 113",
        task_113
    )
    
    # Test Agent 27
    result_27 = await test_agent(
        "http://192.168.1.27:11434",
        "codellama:latest",
        "Agent 27", 
        task_27
    )
    
    # Summary
    print(f"\n📊 ASSIGNMENT TEST RESULTS:")
    print(f"   Agent 113: {'✅ SUCCESS' if result_113 else '❌ FAILED'}")
    print(f"   Agent 27: {'✅ SUCCESS' if result_27 else '❌ FAILED'}")
    
    if result_113 and result_27:
        print(f"\n🎯 BOTH AGENTS CAN HANDLE WORK ASSIGNMENTS!")
        print(f"   Ready for: Coordinated ROCm development tasks")
    else:
        print(f"\n⚠️  AGENTS NEED INVESTIGATION")
        print(f"   Check: Model loading, memory, GPU availability")

async def test_agent(endpoint: str, model: str, name: str, task: str) -> bool:
    """Test a single agent with a simple task"""
    
    payload = {
        "model": model,
        "prompt": task,
        "stream": False,
        "options": {
            "num_predict": 25,
            "temperature": 0.1
        }
    }
    
    try:
        print(f"🚀 Testing {name}...")
        start_time = time.time()
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as session:
            async with session.post(f"{endpoint}/api/generate", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    duration = time.time() - start_time
                    response_text = result.get('response', '')
                    
                    print(f"✅ {name}: {response_text[:100]} ({duration:.1f}s)")
                    return True
                else:
                    print(f"❌ {name}: HTTP {response.status}")
                    return False
                    
    except Exception as e:
        print(f"❌ {name}: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_basic_assignment())