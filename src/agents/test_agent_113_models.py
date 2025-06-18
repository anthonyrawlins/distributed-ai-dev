#!/usr/bin/env python3
"""
Test Agent 113 with Different Models
Find a working model for Agent 113 to replace devstral timeouts
"""

import asyncio
import aiohttp
import json
import time

async def test_model_performance(model_name: str, test_prompt: str = "Write: hipMalloc example") -> dict:
    """Test a specific model on Agent 113"""
    
    endpoint = "http://192.168.1.113:11434"
    
    payload = {
        "model": model_name,
        "prompt": test_prompt,
        "stream": False,
        "options": {
            "num_predict": 30,
            "temperature": 0.1
        }
    }
    
    start_time = time.time()
    
    try:
        print(f"🧪 Testing {model_name}...")
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as session:
            async with session.post(f"{endpoint}/api/generate", json=payload) as response:
                if response.status != 200:
                    return {
                        "model": model_name,
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
                
                print(f"✅ {model_name}: {duration:.1f}s, {tps:.1f} TPS")
                print(f"   Response: {response_text[:80]}...")
                
                return {
                    "model": model_name,
                    "status": "success",
                    "response": response_text,
                    "duration": duration,
                    "tokens_per_second": tps,
                    "eval_count": eval_count,
                    "response_length": len(response_text)
                }
                
    except Exception as e:
        print(f"❌ {model_name}: {e}")
        return {
            "model": model_name,
            "status": "error",
            "error": str(e),
            "duration": time.time() - start_time
        }

async def test_multiple_models():
    """Test multiple models to find the best performing one"""
    
    print("🔍 TESTING MULTIPLE MODELS ON AGENT 113")
    print("="*60)
    
    # Models to test (in order of preference - smaller/faster first)
    models_to_test = [
        "llama3.2:latest",      # 3.2B - smallest, fastest
        "llama3.1:8b",          # 8B - good balance
        "qwen2.5-coder:latest", # 7.6B - good for coding
        "deepseek-r1:7b",       # 7.6B - reasoning model
        "codellama:latest",     # 7B - code focused
        "mistral:7b-instruct",  # 7.2B - fast instruct
        "phi4:latest",          # 14.7B - larger but efficient
        "qwen2:latest"          # 7.6B - alternative
    ]
    
    test_prompt = "Provide 3 ROCm optimization tips in one sentence each."
    
    results = []
    
    for model in models_to_test:
        result = await test_model_performance(model, test_prompt)
        results.append(result)
        
        # If we find a fast working model, test it more thoroughly
        if (result['status'] == 'success' and 
            result['duration'] < 15 and 
            result['tokens_per_second'] > 3.0):
            
            print(f"\n🎯 {model} looks promising! Testing with ROCm task...")
            
            rocm_task = "Write a simple __device__ function that adds two numbers."
            rocm_result = await test_model_performance(model, rocm_task)
            
            if rocm_result['status'] == 'success':
                print(f"✅ {model} successfully handles ROCm tasks!")
                
                # Save this as the recommended model
                recommendation = {
                    "recommended_model": model,
                    "reason": "Fast, reliable, handles ROCm tasks",
                    "performance": {
                        "avg_duration": (result['duration'] + rocm_result['duration']) / 2,
                        "avg_tps": (result['tokens_per_second'] + rocm_result['tokens_per_second']) / 2
                    },
                    "test_results": [result, rocm_result]
                }
                
                # Save recommendation
                with open("/home/tony/AI/ROCm/agent_113_model_recommendation.json", 'w') as f:
                    json.dump(recommendation, f, indent=2)
                
                print(f"\n💾 Recommendation saved for model: {model}")
                break
        
        # Add small delay between tests
        await asyncio.sleep(1)
    
    # Print summary
    print(f"\n📊 MODEL TEST SUMMARY:")
    print("-" * 50)
    
    successful_models = [r for r in results if r['status'] == 'success']
    failed_models = [r for r in results if r['status'] == 'error']
    
    if successful_models:
        print("✅ WORKING MODELS:")
        for result in successful_models:
            print(f"   {result['model']}: {result['duration']:.1f}s, {result['tokens_per_second']:.1f} TPS")
        
        # Find best performing model
        best_model = min(successful_models, key=lambda x: x['duration'])
        print(f"\n🏆 FASTEST MODEL: {best_model['model']}")
        print(f"   Performance: {best_model['duration']:.1f}s, {best_model['tokens_per_second']:.1f} TPS")
        
        return best_model['model']
    
    else:
        print("❌ NO WORKING MODELS FOUND")
        print("   Issue: Agent 113 may have system problems")
        
        if failed_models:
            print("Failed models:")
            for result in failed_models:
                print(f"   {result['model']}: {result['error']}")
        
        return None

async def test_coordination_with_new_model(model_name: str):
    """Test coordination between agents with new model for Agent 113"""
    
    print(f"\n🤝 TESTING COORDINATION WITH {model_name}")
    print("="*50)
    
    # Simple coordination test
    task_113 = "List 3 ROCm performance tips."
    task_27 = "Write: __global__ void example() { }"
    
    print("📤 Testing coordination between agents...")
    
    # Test Agent 113 with new model
    result_113 = await test_agent_task(
        "http://192.168.1.113:11434",
        model_name,
        "Agent 113",
        task_113
    )
    
    # Test Agent 27 (unchanged)
    result_27 = await test_agent_task(
        "http://192.168.1.27:11434", 
        "codellama:latest",
        "Agent 27",
        task_27
    )
    
    # Analyze coordination
    both_success = (result_113 and result_27)
    
    print(f"\n🎯 COORDINATION TEST RESULTS:")
    print(f"   Agent 113 ({model_name}): {'✅' if result_113 else '❌'}")
    print(f"   Agent 27 (codellama): {'✅' if result_27 else '❌'}")
    print(f"   Coordination Status: {'✅ SUCCESS' if both_success else '❌ PARTIAL'}")
    
    if both_success:
        print(f"\n🚀 DUAL AGENT COORDINATION RESTORED!")
        print(f"   Recommended: Update agents.yaml with model '{model_name}' for Agent 113")
    
    return both_success

async def test_agent_task(endpoint: str, model: str, agent_name: str, task: str) -> bool:
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
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as session:
            async with session.post(f"{endpoint}/api/generate", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    response_text = result.get('response', '')
                    print(f"✅ {agent_name}: {response_text[:60]}...")
                    return True
                else:
                    print(f"❌ {agent_name}: HTTP {response.status}")
                    return False
    except Exception as e:
        print(f"❌ {agent_name}: {e}")
        return False

async def main():
    """Main testing function"""
    
    # Test multiple models
    best_model = await test_multiple_models()
    
    if best_model:
        # Test coordination with the best model
        coordination_success = await test_coordination_with_new_model(best_model)
        
        if coordination_success:
            print(f"\n🎉 SUCCESS! Agent 113 working with model: {best_model}")
            print(f"📝 Next step: Update config/agents.yaml with this model")
        else:
            print(f"\n⚠️  Model {best_model} works individually but coordination needs work")
    else:
        print(f"\n❌ Could not find working model for Agent 113")
        print(f"🔍 Recommend checking Agent 113 system status")

if __name__ == "__main__":
    asyncio.run(main())