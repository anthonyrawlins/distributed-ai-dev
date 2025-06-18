#!/usr/bin/env python3
"""
Test Agent 27 with Simple ROCm Development Task
Direct communication to test CodeLlama agent capabilities
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime

class Agent27Tester:
    """
    Direct communication interface for testing Agent 27.
    
    Tests Agent 27's ability to handle ROCm development tasks
    and provides feedback on code quality and accuracy.
    """
    
    def __init__(self):
        self.endpoint = "http://192.168.1.27:11434"
        self.model = "codellama:latest"
        self.agent_name = "CodeLlama Development Assistant"
        
    async def test_agent_connectivity(self) -> bool:
        """
        Test if Agent 27 is responsive and available.
        
        Returns:
            bool: True if agent is reachable, False otherwise
        """
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(f"{self.endpoint}/api/tags") as response:
                    if response.status == 200:
                        tags = await response.json()
                        print(f"✅ Agent 27 is online with models: {[m['name'] for m in tags.get('models', [])]}")
                        return True
                    else:
                        print(f"❌ Agent 27 responded with HTTP {response.status}")
                        return False
        except Exception as e:
            print(f"❌ Failed to connect to Agent 27: {e}")
            return False
    
    async def submit_rocm_task(self, task_description: str, expected_output: str = "code") -> dict:
        """
        Submit a ROCm development task to Agent 27.
        
        Args:
            task_description: The development task to assign
            expected_output: Type of output expected (code, explanation, etc.)
            
        Returns:
            Dictionary containing task results and metadata
        """
        
        # Craft the prompt for CodeLlama
        prompt = f"""You are a ROCm development assistant working on AMD GPU optimization projects.

Task: {task_description}

Requirements:
- Write clean, well-commented code
- Use proper HIP/ROCm APIs where applicable
- Consider RDNA3/CDNA3 architecture optimizations
- Provide brief explanations for key optimizations
- Focus on performance and correctness

Please provide a complete implementation with explanations."""

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # More deterministic for code generation
                "num_predict": 2000,  # Allow longer responses for code
                "top_p": 0.9
            }
        }
        
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
                print(f"🚀 Submitting task to Agent 27...")
                print(f"📝 Task: {task_description}")
                
                async with session.post(f"{self.endpoint}/api/generate", json=payload) as response:
                    if response.status != 200:
                        return {
                            'status': 'error',
                            'error': f'HTTP {response.status}',
                            'duration': time.time() - start_time
                        }
                    
                    result = await response.json()
                    duration = time.time() - start_time
                    
                    # Extract performance metrics
                    metrics = {
                        'total_duration_ms': result.get('total_duration', 0) / 1000000,
                        'load_duration_ms': result.get('load_duration', 0) / 1000000,
                        'prompt_eval_count': result.get('prompt_eval_count', 0),
                        'prompt_eval_duration_ms': result.get('prompt_eval_duration', 0) / 1000000,
                        'eval_count': result.get('eval_count', 0),
                        'eval_duration_ms': result.get('eval_duration', 0) / 1000000,
                        'tokens_per_second': 0
                    }
                    
                    if metrics['eval_duration_ms'] > 0:
                        metrics['tokens_per_second'] = (metrics['eval_count'] / metrics['eval_duration_ms']) * 1000
                    
                    return {
                        'status': 'completed',
                        'task': task_description,
                        'response': result.get('response', ''),
                        'duration': duration,
                        'performance': metrics,
                        'timestamp': datetime.now().isoformat()
                    }
                    
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'duration': time.time() - start_time
            }
    
    def analyze_code_quality(self, response: str) -> dict:
        """
        Analyze the quality of the generated code.
        
        Args:
            response: The code response from Agent 27
            
        Returns:
            Dictionary containing quality analysis
        """
        analysis = {
            'has_code': False,
            'has_comments': False,
            'has_hip_apis': False,
            'has_error_checking': False,
            'line_count': 0,
            'quality_score': 0,
            'issues': []
        }
        
        lines = response.split('\n')
        analysis['line_count'] = len(lines)
        
        # Check for code blocks
        if '```' in response or any(line.strip().startswith('#include') for line in lines):
            analysis['has_code'] = True
        
        # Check for comments
        comment_lines = [line for line in lines if '//' in line or '/*' in line or '*/' in line]
        if comment_lines:
            analysis['has_comments'] = True
        
        # Check for HIP/ROCm APIs
        rocm_keywords = ['hip', 'rocm', '__global__', '__device__', 'hipMalloc', 'hipMemcpy', 'hipLaunchKernel']
        if any(keyword.lower() in response.lower() for keyword in rocm_keywords):
            analysis['has_hip_apis'] = True
        
        # Check for error handling
        error_keywords = ['hipGetLastError', 'hipError_t', 'cudaGetLastError', 'check', 'assert']
        if any(keyword in response for keyword in error_keywords):
            analysis['has_error_checking'] = True
        
        # Calculate quality score
        score = 0
        if analysis['has_code']: score += 25
        if analysis['has_comments']: score += 25
        if analysis['has_hip_apis']: score += 30
        if analysis['has_error_checking']: score += 20
        
        analysis['quality_score'] = score
        
        # Identify issues
        if not analysis['has_code']:
            analysis['issues'].append("No code blocks detected")
        if not analysis['has_comments']:
            analysis['issues'].append("Lacks code comments")
        if not analysis['has_hip_apis']:
            analysis['issues'].append("No HIP/ROCm APIs detected")
        if analysis['line_count'] < 10:
            analysis['issues'].append("Response seems too short for a complete implementation")
        
        return analysis
    
    def print_results(self, result: dict):
        """
        Print formatted results from agent interaction.
        
        Args:
            result: Result dictionary from submit_rocm_task
        """
        print("\n" + "="*80)
        print(f"🤖 AGENT 27 TASK RESULTS - {result.get('timestamp', 'Unknown')}")
        print("="*80)
        
        if result['status'] == 'error':
            print(f"❌ Task failed: {result['error']}")
            print(f"⏱️  Duration: {result['duration']:.2f}s")
            return
        
        print(f"📋 Task: {result['task']}")
        print(f"⏱️  Duration: {result['duration']:.2f}s")
        
        # Performance metrics
        perf = result.get('performance', {})
        if perf:
            print(f"🚀 Performance:")
            print(f"   Tokens/Second: {perf.get('tokens_per_second', 0):.1f}")
            print(f"   Total Time: {perf.get('total_duration_ms', 0):.0f}ms")
            print(f"   Eval Time: {perf.get('eval_duration_ms', 0):.0f}ms")
            print(f"   Tokens Generated: {perf.get('eval_count', 0)}")
        
        # Code quality analysis
        analysis = self.analyze_code_quality(result['response'])
        print(f"📊 Code Quality Analysis:")
        print(f"   Quality Score: {analysis['quality_score']}/100")
        print(f"   Has Code: {'✅' if analysis['has_code'] else '❌'}")
        print(f"   Has Comments: {'✅' if analysis['has_comments'] else '❌'}")
        print(f"   Uses HIP/ROCm APIs: {'✅' if analysis['has_hip_apis'] else '❌'}")
        print(f"   Error Checking: {'✅' if analysis['has_error_checking'] else '❌'}")
        print(f"   Response Length: {analysis['line_count']} lines")
        
        if analysis['issues']:
            print(f"⚠️  Issues:")
            for issue in analysis['issues']:
                print(f"   - {issue}")
        
        print(f"\n💬 Agent Response:")
        print("-" * 40)
        print(result['response'])
        print("-" * 40)

async def main():
    """
    Main testing function for Agent 27.
    """
    print("🧪 TESTING AGENT 27 (CodeLlama Development Assistant)")
    print("="*60)
    
    tester = Agent27Tester()
    
    # Test connectivity
    if not await tester.test_agent_connectivity():
        print("Cannot proceed - Agent 27 is not accessible")
        return
    
    # Define a simple ROCm development task
    task = """Create a simple HIP kernel that adds two vectors (vector addition).
The kernel should:
1. Take three parameters: input vector A, input vector B, and output vector C
2. Each thread should compute one element: C[i] = A[i] + B[i]
3. Include proper bounds checking
4. Provide a host function to launch the kernel
5. Include basic error checking"""
    
    # Submit the task
    result = await tester.test_agent_connectivity()
    if result:
        task_result = await tester.submit_rocm_task(task)
        tester.print_results(task_result)
        
        # Save results to file for review
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"/home/tony/AI/ROCm/agent_27_test_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(task_result, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        # Performance assessment
        if task_result['status'] == 'completed':
            perf = task_result.get('performance', {})
            analysis = tester.analyze_code_quality(task_result['response'])
            
            print(f"\n📈 AGENT 27 ASSESSMENT:")
            print(f"   Speed: {perf.get('tokens_per_second', 0):.1f} TPS")
            print(f"   Quality: {analysis['quality_score']}/100")
            
            if analysis['quality_score'] >= 75:
                print("   Status: ✅ EXCELLENT - Ready for ROCm development tasks")
            elif analysis['quality_score'] >= 50:
                print("   Status: ⚠️  GOOD - Suitable for basic development tasks")
            else:
                print("   Status: ❌ NEEDS IMPROVEMENT - Requires task refinement")

if __name__ == "__main__":
    asyncio.run(main())