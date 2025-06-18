#!/usr/bin/env python3
"""
Agent 27 ROCm Development Task
Comprehensive test of CodeLlama's ROCm development capabilities
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime

async def assign_vector_add_task():
    """Assign a vector addition kernel task to Agent 27"""
    endpoint = "http://192.168.1.27:11434"
    
    # Comprehensive but focused task
    prompt = """You are working on ROCm development. Create a HIP vector addition kernel.

Requirements:
1. Write a __global__ kernel function called vectorAdd
2. Add corresponding host code to launch the kernel  
3. Include proper error checking with hipGetLastError()
4. Use hipMalloc and hipMemcpy for memory management
5. Add comments explaining the key parts

Keep it concise but complete. Focus on correctness."""

    payload = {
        "model": "codellama:latest",
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 800,  # Reasonable length for complete code
            "temperature": 0.1,
            "top_p": 0.9
        }
    }
    
    print("🚀 ASSIGNING VECTOR ADDITION TASK TO AGENT 27")
    print("="*60)
    print("📋 Task: Create HIP vector addition kernel with host code")
    
    start_time = time.time()
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=45)) as session:
            print("⏳ Agent 27 is working...")
            
            async with session.post(f"{endpoint}/api/generate", json=payload) as response:
                if response.status != 200:
                    print(f"❌ HTTP Error {response.status}")
                    return None
                
                result = await response.json()
                duration = time.time() - start_time
                
                # Extract metrics
                response_text = result.get('response', '')
                eval_count = result.get('eval_count', 0)
                eval_duration = result.get('eval_duration', 0)
                total_duration = result.get('total_duration', 0)
                
                tps = 0
                if eval_duration > 0:
                    tps = eval_count / (eval_duration / 1000000000)  # Convert from nanoseconds
                
                print(f"✅ Task completed in {duration:.1f}s")
                print(f"⚡ Performance: {tps:.1f} TPS, {eval_count} tokens")
                
                return {
                    'response': response_text,
                    'duration': duration,
                    'tokens_per_second': tps,
                    'eval_count': eval_count,
                    'timestamp': datetime.now().isoformat()
                }
                
    except Exception as e:
        print(f"❌ Task failed: {e}")
        return None

def analyze_rocm_code(code_response: str) -> dict:
    """Analyze the quality of generated ROCm code"""
    analysis = {
        'has_kernel': False,
        'has_host_code': False,
        'has_error_checking': False,
        'has_memory_management': False,
        'has_comments': False,
        'line_count': len(code_response.split('\n')),
        'rocm_apis': [],
        'quality_score': 0
    }
    
    code_lower = code_response.lower()
    
    # Check for kernel function
    if '__global__' in code_response or 'vectoradd' in code_lower:
        analysis['has_kernel'] = True
    
    # Check for host code
    if 'main' in code_response or 'host' in code_lower:
        analysis['has_host_code'] = True
    
    # Check for error checking
    if any(term in code_response for term in ['hipGetLastError', 'hipError_t', 'HIP_CHECK']):
        analysis['has_error_checking'] = True
    
    # Check for memory management
    if any(term in code_response for term in ['hipMalloc', 'hipMemcpy', 'hipFree']):
        analysis['has_memory_management'] = True
    
    # Check for comments
    if '//' in code_response or '/*' in code_response:
        analysis['has_comments'] = True
    
    # Identify ROCm APIs used
    rocm_apis = ['hipMalloc', 'hipMemcpy', 'hipFree', 'hipLaunchKernelGGL', 'hipGetLastError', '__global__', '__device__']
    analysis['rocm_apis'] = [api for api in rocm_apis if api in code_response]
    
    # Calculate quality score
    score = 0
    if analysis['has_kernel']: score += 30
    if analysis['has_host_code']: score += 25
    if analysis['has_error_checking']: score += 20
    if analysis['has_memory_management']: score += 15
    if analysis['has_comments']: score += 10
    
    analysis['quality_score'] = score
    
    return analysis

def print_code_review(response_data: dict):
    """Print a comprehensive code review"""
    if not response_data:
        print("❌ No code to review")
        return
    
    code = response_data['response']
    analysis = analyze_rocm_code(code)
    
    print("\n" + "="*80)
    print("📋 AGENT 27 CODE REVIEW")
    print("="*80)
    
    print(f"⏱️  Completion Time: {response_data['duration']:.1f}s")
    print(f"⚡ Generation Speed: {response_data['tokens_per_second']:.1f} TPS")
    print(f"📝 Code Length: {analysis['line_count']} lines")
    
    print(f"\n🎯 CODE QUALITY ANALYSIS:")
    print(f"   Overall Score: {analysis['quality_score']}/100")
    print(f"   Has Kernel Function: {'✅' if analysis['has_kernel'] else '❌'}")
    print(f"   Has Host Code: {'✅' if analysis['has_host_code'] else '❌'}")
    print(f"   Error Checking: {'✅' if analysis['has_error_checking'] else '❌'}")
    print(f"   Memory Management: {'✅' if analysis['has_memory_management'] else '❌'}")
    print(f"   Code Comments: {'✅' if analysis['has_comments'] else '❌'}")
    
    if analysis['rocm_apis']:
        print(f"   ROCm APIs Used: {', '.join(analysis['rocm_apis'])}")
    
    # Overall assessment
    if analysis['quality_score'] >= 80:
        print(f"\n🏆 ASSESSMENT: EXCELLENT - Production-ready code quality")
    elif analysis['quality_score'] >= 60:
        print(f"\n✅ ASSESSMENT: GOOD - Suitable for development with minor improvements")
    elif analysis['quality_score'] >= 40:
        print(f"\n⚠️  ASSESSMENT: FAIR - Needs significant improvements")
    else:
        print(f"\n❌ ASSESSMENT: POOR - Major issues, requires complete rework")
    
    print(f"\n💻 GENERATED CODE:")
    print("-" * 80)
    print(code)
    print("-" * 80)

async def main():
    """Main function to test Agent 27 with ROCm development task"""
    
    # Assign and execute the task
    result = await assign_vector_add_task()
    
    if result:
        # Review the code
        print_code_review(result)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"/home/tony/AI/ROCm/agent_27_rocm_task_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n💾 Full results saved to: {filename}")
        
        # Update agent task status in config
        analysis = analyze_rocm_code(result['response'])
        print(f"\n📈 AGENT 27 PERFORMANCE SUMMARY:")
        print(f"   Task: Vector Addition Kernel")
        print(f"   Status: {'✅ COMPLETED' if analysis['quality_score'] >= 50 else '⚠️  NEEDS WORK'}")
        print(f"   Speed: {result['tokens_per_second']:.1f} TPS")
        print(f"   Quality: {analysis['quality_score']}/100")
        
        if analysis['quality_score'] >= 70:
            print("   Recommendation: ✅ Agent 27 ready for ROCm development tasks")
        else:
            print("   Recommendation: ⚠️  Agent 27 needs task refinement or context improvement")
    
    else:
        print("❌ Task assignment failed. Agent 27 may be unavailable or overloaded.")

if __name__ == "__main__":
    asyncio.run(main())