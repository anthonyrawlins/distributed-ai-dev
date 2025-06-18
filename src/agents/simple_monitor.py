#!/usr/bin/env python3
"""
Simple Agent Monitor - Terminal-friendly version

A lightweight, terminal-compatible monitoring system for distributed AI agents.
Provides real-time performance metrics and system resource monitoring without
requiring curses or special terminal capabilities.

Features:
- Real-time Ollama API performance metrics (TPS, latency, token counts)
- System resource monitoring (CPU, memory, GPU utilization)
- Text-based progress bars and status indicators
- Historical performance tracking with rolling averages
- Cross-platform compatibility (works in any terminal)
- Auto-detection of AMD ROCm and NVIDIA CUDA GPUs
- Clean, readable output format

Usage:
    python3 simple_monitor.py
    
Controls:
    Ctrl+C - Exit monitoring
    
Requires:
    - Python 3.7+
    - psutil for system metrics
    - aiohttp for async HTTP requests
    - rocm-smi or nvidia-smi for GPU metrics (optional)
    
Author: Distributed AI Development System
Compatible with: Claude Code Agent Network
"""

import asyncio
import aiohttp
import json
import time
import psutil
import subprocess
import os
import sys
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, List, Optional, Any

class AgentPerformanceMetrics:
    """
    Lightweight performance metrics tracker for AI agents.
    
    Maintains rolling history of performance data and provides statistical
    analysis capabilities for monitoring agent efficiency over time.
    
    Attributes:
        max_history (int): Maximum number of metric samples to retain
        metrics_history (deque): Rolling buffer of performance metrics
        task_history (deque): Rolling buffer of completed task information
    """
    
    def __init__(self, max_history: int = 100):
        """
        Initialize performance metrics tracker.
        
        Args:
            max_history: Maximum number of metric samples to store
        """
        self.max_history = max_history
        self.metrics_history = deque(maxlen=max_history)
        self.task_history = deque(maxlen=50)
        
    def add_metrics(self, metrics: Dict[str, Any]):
        """
        Add new performance metrics sample.
        
        Args:
            metrics: Dictionary containing performance data including
                    tokens_per_second, total_duration_ms, eval_count, etc.
        """
        metrics['timestamp'] = time.time()
        self.metrics_history.append(metrics)
        
    def get_avg_tps(self, window_minutes: int = 5) -> float:
        """Get average tokens per second over window"""
        cutoff = time.time() - (window_minutes * 60)
        recent = [m for m in self.metrics_history if m['timestamp'] > cutoff and 'tokens_per_second' in m]
        if not recent:
            return 0.0
        return sum(m['tokens_per_second'] for m in recent) / len(recent)
        
    def get_avg_response_time(self, window_minutes: int = 5) -> float:
        """Get average response time over window"""
        cutoff = time.time() - (window_minutes * 60)
        recent = [m for m in self.metrics_history if m['timestamp'] > cutoff and 'total_duration_ms' in m]
        if not recent:
            return 0.0
        return sum(m['total_duration_ms'] for m in recent) / len(recent)

class AgentMonitor:
    """
    Comprehensive monitoring system for individual AI agents.
    
    Coordinates performance monitoring, system resource tracking, and status
    management for a single agent endpoint. Provides both Ollama API metrics
    and system-level resource utilization data.
    
    Attributes:
        agent_id (str): Unique identifier for the agent
        endpoint (str): HTTP endpoint for agent communication
        model (str): Model name running on the agent
        metrics (AgentPerformanceMetrics): Performance data tracker
        last_status (str): Current agent status
    """
    
    def __init__(self, agent_id: str, endpoint: str, model: str):
        """
        Initialize agent monitor.
        
        Args:
            agent_id: Unique identifier for the agent
            endpoint: HTTP endpoint for Ollama API communication
            model: Model name to use for test queries
        """
        self.agent_id = agent_id
        self.endpoint = endpoint
        self.model = model
        self.metrics = AgentPerformanceMetrics()
        self.last_status = "Unknown"
        
    async def collect_detailed_metrics(self, test_prompt: str = "Hello, provide a brief status update.") -> Dict[str, Any]:
        """
        Collect comprehensive performance metrics from Ollama API.
        
        Submits a test prompt to measure real-world performance including
        token generation speed, response latency, and processing efficiency.
        
        Args:
            test_prompt: Test prompt to send to agent for metric collection
            
        Returns:
            Dictionary containing performance metrics or error information
        """
        payload = {
            "model": self.model,
            "prompt": test_prompt,
            "stream": False,
            "options": {
                "num_predict": 50,
                "temperature": 0.1
            }
        }
        
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.post(f"{self.endpoint}/api/generate", json=payload) as response:
                    if response.status != 200:
                        return {'error': f'HTTP {response.status}', 'timestamp': time.time()}
                        
                    result = await response.json()
                    
            total_duration = time.time() - start_time
            
            # Extract Ollama metrics
            metrics = {
                'status': 'active',
                'total_duration_ms': result.get('total_duration', 0) / 1000000,  # Convert from nanoseconds
                'load_duration_ms': result.get('load_duration', 0) / 1000000,
                'prompt_eval_count': result.get('prompt_eval_count', 0),
                'prompt_eval_duration_ms': result.get('prompt_eval_duration', 0) / 1000000,
                'eval_count': result.get('eval_count', 0),
                'eval_duration_ms': result.get('eval_duration', 0) / 1000000,
                'measured_total_duration': total_duration * 1000,
                'response_length': len(result.get('response', ''))
            }
            
            # Calculate derived metrics
            if metrics['eval_duration_ms'] > 0:
                metrics['tokens_per_second'] = (metrics['eval_count'] / metrics['eval_duration_ms']) * 1000
            else:
                metrics['tokens_per_second'] = 0
                
            if metrics['prompt_eval_duration_ms'] > 0:
                metrics['prompt_eval_rate'] = (metrics['prompt_eval_count'] / metrics['prompt_eval_duration_ms']) * 1000
            else:
                metrics['prompt_eval_rate'] = 0
                
            self.metrics.add_metrics(metrics)
            self.last_status = "Active"
            
            return metrics
            
        except Exception as e:
            error_metrics = {
                'error': str(e),
                'status': 'error',
                'timestamp': time.time()
            }
            self.last_status = f"Error: {str(e)[:50]}"
            return error_metrics

    async def get_system_metrics(self) -> Dict[str, Any]:
        """
        Collect comprehensive system resource metrics.
        
        Gathers CPU utilization, memory usage, disk usage, network I/O,
        and GPU metrics (if available) for system health monitoring.
        
        Returns:
            Dictionary containing system resource utilization data
        """
        metrics = {
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory': psutil.virtual_memory()._asdict(),
            'disk': psutil.disk_usage('/')._asdict(),
            'network': psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {},
            'gpu': await self.get_gpu_metrics()
        }
        return metrics
        
    async def get_gpu_metrics(self) -> Dict[str, Any]:
        """
        Detect and collect GPU utilization metrics.
        
        Attempts to detect AMD ROCm or NVIDIA CUDA GPUs and collect
        utilization, memory usage, and temperature data using vendor tools.
        
        Returns:
            Dictionary containing GPU metrics or availability status
        """
        gpu_info = {'available': False, 'type': 'unknown'}
        
        try:
            # Try rocm-smi for AMD GPUs
            result = subprocess.run(['rocm-smi', '--showuse', '--showmemuse', '--csv'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                gpu_info = self.parse_rocm_smi(result.stdout)
                gpu_info['type'] = 'AMD ROCm'
                gpu_info['available'] = True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            try:
                # Try nvidia-smi for NVIDIA GPUs
                result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                                       '--format=csv,nounits,noheader'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    gpu_info = self.parse_nvidia_smi(result.stdout)
                    gpu_info['type'] = 'NVIDIA CUDA'
                    gpu_info['available'] = True
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
                
        return gpu_info
        
    def parse_rocm_smi(self, output: str) -> Dict[str, Any]:
        """Parse rocm-smi output"""
        lines = output.strip().split('\n')
        if len(lines) < 2:
            return {'available': False}
            
        # Parse CSV header and data
        header = lines[0].split(',')
        data = lines[1].split(',')
        
        gpu_data = dict(zip(header, data))
        
        return {
            'utilization': float(gpu_data.get('GPU use (%)', 0)),
            'memory_used': float(gpu_data.get('Memory use (%)', 0)),
            'temperature': float(gpu_data.get('Temperature (°C)', 0)) if 'Temperature (°C)' in gpu_data else 0
        }
        
    def parse_nvidia_smi(self, output: str) -> Dict[str, Any]:
        """Parse nvidia-smi output"""
        line = output.strip()
        if not line:
            return {'available': False}
            
        parts = line.split(', ')
        if len(parts) != 3:
            return {'available': False}
            
        return {
            'utilization': float(parts[0]),
            'memory_used_mb': float(parts[1]),
            'memory_total_mb': float(parts[2]),
            'memory_used_percent': (float(parts[1]) / float(parts[2])) * 100
        }

def clear_screen():
    """
    Clear the terminal screen in a cross-platform manner.
    
    Uses appropriate clear command for Unix/Linux or Windows systems.
    """
    os.system('clear' if os.name == 'posix' else 'cls')

def create_bar(percentage: float, width: int = 20) -> str:
    """
    Create a text-based progress bar with Unicode block characters.
    
    Args:
        percentage: Value from 0-100 to display
        width: Character width of the progress bar
        
    Returns:
        Formatted progress bar string with percentage
    """
    filled = int(percentage / 100 * width)
    bar = '█' * filled + '░' * (width - filled)
    return f"{bar} {percentage:5.1f}%"

def format_status_indicator(status: str) -> str:
    """
    Format agent status with colored emoji indicators.
    
    Args:
        status: Current agent status string
        
    Returns:
        Formatted status with appropriate emoji indicator
    """
    if status == "Active":
        return "🟢 ACTIVE"
    elif "Error" in status:
        return "🔴 ERROR"
    else:
        return "🟡 UNKNOWN"

async def print_monitoring_dashboard(agent: AgentMonitor, system_metrics: Dict[str, Any]):
    """
    Display a comprehensive monitoring dashboard in the terminal.
    
    Renders agent performance metrics, system resource utilization,
    and status information in a clean, readable format similar to
    system monitoring tools like btop or nvtop.
    
    Args:
        agent: AgentMonitor instance with current metrics
        system_metrics: System resource utilization data
    """
    clear_screen()
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    print("🚀 DISTRIBUTED AI DEVELOPMENT - AGENT MONITORING")
    print("=" * 60)
    print(f"📅 {timestamp}")
    print()
    
    # Agent Information
    print(f"📡 {agent.agent_id.upper()} ({agent.model})")
    print(f"   Status: {format_status_indicator(agent.last_status)}")
    print(f"   Endpoint: {agent.endpoint}")
    print()
    
    # Performance Metrics
    avg_tps = agent.metrics.get_avg_tps()
    avg_response = agent.metrics.get_avg_response_time()
    sample_count = len(agent.metrics.metrics_history)
    
    perf_status = "EXCELLENT" if avg_tps > 10.0 else "GOOD" if avg_tps > 5.0 else "FAIR" if avg_tps > 2.0 else "POOR"
    
    print("⚡ PERFORMANCE METRICS")
    print(f"   Tokens/Second: {avg_tps:6.1f} TPS ({perf_status})")
    print(f"   Avg Response:  {avg_response:6.0f} ms")
    print(f"   Sample Count:  {sample_count:6d} measurements")
    
    # Latest metrics if available
    if agent.metrics.metrics_history:
        latest = agent.metrics.metrics_history[-1]
        if 'error' not in latest:
            print()
            print("📊 LATEST MEASUREMENT")
            print(f"   Tokens/Second: {latest.get('tokens_per_second', 0):6.1f} TPS")
            print(f"   Total Time:    {latest.get('total_duration_ms', 0):6.0f} ms")
            print(f"   Eval Time:     {latest.get('eval_duration_ms', 0):6.0f} ms")
            print(f"   Load Time:     {latest.get('load_duration_ms', 0):6.0f} ms")
            print(f"   Token Count:   {latest.get('eval_count', 0):6d} tokens")
    
    print()
    
    # System Resources
    print("💻 SYSTEM RESOURCES")
    
    cpu_percent = system_metrics.get('cpu_percent', 0)
    memory = system_metrics.get('memory', {})
    mem_percent = memory.get('percent', 0)
    
    print(f"   CPU: {create_bar(cpu_percent)}")
    print(f"   RAM: {create_bar(mem_percent)}")
    
    # GPU if available
    gpu = system_metrics.get('gpu', {})
    if gpu.get('available'):
        gpu_util = gpu.get('utilization', 0)
        gpu_mem = gpu.get('memory_used', 0) if 'memory_used' in gpu else gpu.get('memory_used_percent', 0)
        
        print(f"   GPU: {create_bar(gpu_util)} ({gpu.get('type', 'Unknown')})")
        print(f"  VRAM: {create_bar(gpu_mem)}")
        
        if 'temperature' in gpu:
            print(f"  Temp: {gpu['temperature']:5.1f}°C")
    
    print()
    print("=" * 60)
    print("Press Ctrl+C to exit | Refresh every 5 seconds")

async def run_simple_monitor():
    """
    Execute the main monitoring loop for the simple interface.
    
    Initializes the agent monitor, runs continuous metric collection,
    and displays real-time performance data until interrupted by user.
    
    Raises:
        KeyboardInterrupt: On user termination request (Ctrl+C)
    """
    agent = AgentMonitor(
        agent_id='agent_113_devstral',
        endpoint='http://192.168.1.113:11434',
        model='devstral:latest'
    )
    
    print("🚀 Starting Agent Monitor...")
    print("Collecting initial metrics...")
    
    try:
        while True:
            # Collect metrics
            metrics_task = agent.collect_detailed_metrics()
            system_task = agent.get_system_metrics()
            
            # Run in parallel
            agent_metrics, system_metrics = await asyncio.gather(
                metrics_task, system_task, return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(system_metrics, Exception):
                system_metrics = {'error': str(system_metrics)}
            
            # Display dashboard
            await print_monitoring_dashboard(agent, system_metrics)
            
            # Wait for next refresh
            await asyncio.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")
        
    except Exception as e:
        print(f"\nError running monitor: {e}")

if __name__ == "__main__":
    """
    Entry point for the simple monitoring system.
    
    Starts the monitoring loop and handles graceful shutdown on
    keyboard interrupt or unexpected errors.
    """
    asyncio.run(run_simple_monitor())