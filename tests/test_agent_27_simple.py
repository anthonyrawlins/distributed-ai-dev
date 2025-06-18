#!/usr/bin/env python3
"""Simple test for Agent 27 with minimal task"""

import asyncio
import aiohttp
import time

async def test_agent_27():
    payload = {
        'model': 'codellama:latest',
        'prompt': 'Write a bash command to check ROCm version.',
        'stream': False,
        'options': {'num_predict': 30, 'temperature': 0.1}
    }
    
    start_time = time.time()
    
    try:
        print("⏳ Testing Agent 27 with simple task...")
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
            async with session.post('http://192.168.1.27:11434/api/generate', json=payload) as response:
                if response.status != 200:
                    print(f'❌ HTTP {response.status}')
                    return
                
                result = await response.json()
                duration = time.time() - start_time
                
                eval_count = result.get('eval_count', 0)
                eval_duration = result.get('eval_duration', 0)
                tps = eval_count / (eval_duration / 1000000000) if eval_duration > 0 else 0
                
                print(f'✅ Agent 27 responded in {duration:.1f}s ({tps:.1f} TPS)')
                print(f'Response: {result.get("response", "")}')
                
    except Exception as e:
        print(f'❌ Agent 27 failed: {e}')

if __name__ == "__main__":
    asyncio.run(test_agent_27())