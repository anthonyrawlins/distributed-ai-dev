#!/usr/bin/env python3
"""
Advanced Agent Monitoring System

A comprehensive real-time monitoring system for distributed AI agents with
a btop/nvtop-style curses interface. Provides detailed performance metrics,
system resource monitoring, and visual feedback for agent networks.

Features:
- Real-time Ollama API performance metrics (TPS, latency, token counts)
- System resource monitoring (CPU, memory, GPU utilization)
- Historical performance tracking with rolling averages
- Color-coded status indicators and progress bars
- Support for multiple agent monitoring
- Auto-detection of AMD ROCm and NVIDIA CUDA GPUs

Usage:
    python3 advanced_monitor.py
    
Controls:
    q - Quit
    r - Force refresh
    
Requires:
    - curses-compatible terminal
    - psutil for system metrics
    - aiohttp for async HTTP requests
    - rocm-smi or nvidia-smi for GPU metrics
    
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
import curses
from typing import Dict, List, Optional, Any

class AgentPerformanceMetrics:
    """
    Tracks and analyzes performance metrics for AI agents.
    
    Maintains rolling history of performance data including tokens per second,
    response times, and task completion metrics. Provides statistical analysis
    with configurable time windows.
    
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
        
    def add_task(self, task_info: Dict[str, Any]):
        """Add task completion info"""
        task_info['timestamp'] = time.time()
        self.task_history.append(task_info)
        
    def get_avg_tps(self, window_minutes: int = 5) -> float:
        """
        Calculate average tokens per second over time window.
        
        Args:
            window_minutes: Time window in minutes for averaging
            
        Returns:
            Average tokens per second, or 0.0 if no data available
        """
        cutoff = time.time() - (window_minutes * 60)
        recent = [m for m in self.metrics_history if m['timestamp'] > cutoff]
        if not recent:
            return 0.0
        return sum(m.get('tokens_per_second', 0) for m in recent) / len(recent)
        
    def get_avg_response_time(self, window_minutes: int = 5) -> float:
        """Get average response time over window"""
        cutoff = time.time() - (window_minutes * 60)
        recent = [m for m in self.metrics_history if m['timestamp'] > cutoff]
        if not recent:
            return 0.0
        return sum(m.get('total_duration_ms', 0) for m in recent) / len(recent)

class SystemResourceMonitor:
    """
    Monitors system resources including CPU, memory, and GPU utilization.
    
    Provides cross-platform system monitoring with automatic GPU detection
    for both AMD ROCm and NVIDIA CUDA environments. Collects metrics via
    psutil and vendor-specific GPU tools.
    
    Attributes:
        agent_host (str): Target host for monitoring (future remote support)
    """
    
    def __init__(self, agent_host: str = "192.168.1.113"):
        """
        Initialize system resource monitor.
        
        Args:
            agent_host: Target agent host IP for future remote monitoring
        """
        self.agent_host = agent_host
        
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Collect system resource metrics"""
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
        utilization, memory usage, and temperature data.
        
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

class AgentMonitor:
    """
    Monitors individual AI agent performance and system resources.
    
    Coordinates performance monitoring for a single agent, collecting both
    Ollama API metrics and system resource utilization. Maintains performance
    history and provides real-time status tracking.
    
    Attributes:
        agent_id (str): Unique identifier for the agent
        endpoint (str): HTTP endpoint for agent communication
        model (str): Model name running on the agent
        metrics (AgentPerformanceMetrics): Performance data tracker
        system_monitor (SystemResourceMonitor): System resource tracker
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
        self.system_monitor = SystemResourceMonitor()
        self.last_status = "Unknown"
        
    async def collect_detailed_metrics(self, test_prompt: str = "Hello, provide a brief status update.") -> Dict[str, Any]:
        """
        Collect comprehensive performance metrics from Ollama API.
        
        Submits a test prompt to the agent and collects detailed timing metrics
        including token generation speed, response latency, and model loading time.
        
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

class ConsoleInterface:
    """
    Manages curses-based console interface for real-time monitoring.
    
    Provides a btop/nvtop-style terminal interface with color coding,
    progress bars, and real-time updates. Handles terminal initialization,
    drawing operations, and user input.
    
    Attributes:
        stdscr: Curses screen object
        height (int): Terminal height in characters
        width (int): Terminal width in characters
    """
    
    def __init__(self):
        """
        Initialize console interface.
        """
        self.stdscr = None
        self.height = 0
        self.width = 0
        
    def init_curses(self):
        """Initialize curses interface"""
        try:
            self.stdscr = curses.initscr()
            curses.noecho()
            curses.cbreak()
            self.stdscr.keypad(True)
            self.stdscr.nodelay(True)
            
            # Check if terminal supports color
            if curses.has_colors():
                curses.start_color()
            else:
                # Fallback for terminals without color support
                pass
        except curses.error as e:
            # Fallback for non-interactive terminals
            raise RuntimeError(f"Terminal does not support curses interface: {e}")
        
        # Define color pairs only if colors are supported
        if curses.has_colors():
            curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)    # Good/Active
            curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)   # Warning
            curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)      # Error/Critical
            curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)     # Info/Headers
            curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK)  # Special
        
        self.height, self.width = self.stdscr.getmaxyx()
        
    def cleanup_curses(self):
        """Cleanup curses interface"""
        if self.stdscr:
            curses.nocbreak()
            self.stdscr.keypad(False)
            curses.echo()
            curses.endwin()
            
    def draw_header(self, line: int) -> int:
        """Draw the header section"""
        header = "🚀 DISTRIBUTED AI DEVELOPMENT - AGENT MONITORING DASHBOARD"
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        self.stdscr.addstr(line, 0, header[:self.width-1], curses.color_pair(4) | curses.A_BOLD)
        self.stdscr.addstr(line, self.width - len(timestamp) - 1, timestamp, curses.color_pair(4))
        line += 1
        
        separator = "═" * (self.width - 1)
        self.stdscr.addstr(line, 0, separator[:self.width-1], curses.color_pair(4))
        line += 2
        
        return line
        
    def draw_agent_section(self, line: int, agent: AgentMonitor, system_metrics: Dict[str, Any]) -> int:
        """Draw agent performance section"""
        # Agent header
        agent_header = f"📡 {agent.agent_id.upper()} ({agent.model})"
        self.stdscr.addstr(line, 0, agent_header[:self.width-1], curses.color_pair(5) | curses.A_BOLD)
        line += 1
        
        # Status and endpoint
        status_color = curses.color_pair(1) if agent.last_status == "Active" else curses.color_pair(3)
        status_line = f"Status: {agent.last_status} | Endpoint: {agent.endpoint}"
        self.stdscr.addstr(line, 2, status_line[:self.width-3], status_color)
        line += 2
        
        # Performance metrics
        avg_tps = agent.metrics.get_avg_tps()
        avg_response = agent.metrics.get_avg_response_time()
        
        perf_color = curses.color_pair(1) if avg_tps > 5.0 else curses.color_pair(2) if avg_tps > 2.0 else curses.color_pair(3)
        
        perf_line = f"⚡ Performance: {avg_tps:.1f} TPS | {avg_response:.0f}ms avg response | {len(agent.metrics.metrics_history)} samples"
        self.stdscr.addstr(line, 2, perf_line[:self.width-3], perf_color)
        line += 1
        
        # Latest metrics if available
        if agent.metrics.metrics_history:
            latest = agent.metrics.metrics_history[-1]
            if 'error' not in latest:
                metrics_line = f"📊 Latest: {latest.get('tokens_per_second', 0):.1f} TPS | "
                metrics_line += f"{latest.get('total_duration_ms', 0):.0f}ms | "
                metrics_line += f"{latest.get('eval_count', 0)} tokens"
                self.stdscr.addstr(line, 2, metrics_line[:self.width-3])
                line += 1
                
        line += 1
        return line
        
    def draw_system_section(self, line: int, system_metrics: Dict[str, Any]) -> int:
        """Draw system resources section"""
        self.stdscr.addstr(line, 0, "💻 SYSTEM RESOURCES", curses.color_pair(4) | curses.A_BOLD)
        line += 1
        
        # CPU and Memory
        cpu_percent = system_metrics.get('cpu_percent', 0)
        memory = system_metrics.get('memory', {})
        mem_percent = memory.get('percent', 0)
        
        cpu_color = curses.color_pair(1) if cpu_percent < 70 else curses.color_pair(2) if cpu_percent < 90 else curses.color_pair(3)
        mem_color = curses.color_pair(1) if mem_percent < 70 else curses.color_pair(2) if mem_percent < 90 else curses.color_pair(3)
        
        cpu_line = f"CPU: {cpu_percent:5.1f}% {'█' * int(cpu_percent/5):<20}"
        mem_line = f"RAM: {mem_percent:5.1f}% {'█' * int(mem_percent/5):<20}"
        
        self.stdscr.addstr(line, 2, cpu_line[:self.width-3], cpu_color)
        line += 1
        self.stdscr.addstr(line, 2, mem_line[:self.width-3], mem_color)
        line += 1
        
        # GPU if available
        gpu = system_metrics.get('gpu', {})
        if gpu.get('available'):
            gpu_util = gpu.get('utilization', 0)
            gpu_mem = gpu.get('memory_used', 0) if 'memory_used' in gpu else gpu.get('memory_used_percent', 0)
            
            gpu_color = curses.color_pair(1) if gpu_util < 70 else curses.color_pair(2) if gpu_util < 90 else curses.color_pair(3)
            
            gpu_line = f"GPU: {gpu_util:5.1f}% {'█' * int(gpu_util/5):<20} ({gpu.get('type', 'Unknown')})"
            self.stdscr.addstr(line, 2, gpu_line[:self.width-3], gpu_color)
            line += 1
            
            mem_line = f"VRAM:{gpu_mem:5.1f}% {'█' * int(gpu_mem/5):<20}"
            self.stdscr.addstr(line, 2, mem_line[:self.width-3], gpu_color)
            line += 1
            
        line += 1
        return line
        
    def draw_footer(self, line: int):
        """Draw footer with controls"""
        if line < self.height - 2:
            footer = "Press 'q' to quit | 'r' to refresh | Updates every 5 seconds"
            self.stdscr.addstr(self.height - 2, 0, footer[:self.width-1], curses.color_pair(4))
            
            separator = "═" * (self.width - 1)
            self.stdscr.addstr(self.height - 1, 0, separator[:self.width-1], curses.color_pair(4))

class AdvancedAgentMonitor:
    """
    Main monitoring system coordinator for distributed AI agents.
    
    Orchestrates monitoring of multiple AI agents with a unified interface.
    Manages the curses UI, coordinates data collection, and handles user
    interaction for the monitoring dashboard.
    
    Attributes:
        agents (Dict[str, AgentMonitor]): Registry of monitored agents
        ui (ConsoleInterface): Terminal user interface manager
        running (bool): Monitoring loop control flag
        refresh_interval (int): Update frequency in seconds
    """
    
    def __init__(self):
        """
        Initialize the advanced monitoring system.
        
        Sets up default Agent 113 configuration and initializes the
        console interface for real-time monitoring.
        """
        self.agents = {
            'agent_113': AgentMonitor(
                agent_id='agent_113_devstral',
                endpoint='http://192.168.1.113:11434',
                model='devstral:latest'
            )
        }
        self.ui = ConsoleInterface()
        self.running = False
        self.refresh_interval = 5  # seconds
        
    async def run(self):
        """
        Execute the main monitoring loop.
        
        Initializes the curses interface and runs the continuous monitoring
        loop, collecting metrics from all agents and updating the display.
        Handles user input for control and graceful shutdown.
        
        Raises:
            KeyboardInterrupt: On user termination request
        """
        self.ui.init_curses()
        self.running = True
        
        try:
            while self.running:
                # Clear screen
                self.ui.stdscr.clear()
                
                # Collect metrics from all agents
                agent_tasks = []
                system_tasks = []
                
                for agent in self.agents.values():
                    agent_tasks.append(agent.collect_detailed_metrics())
                    system_tasks.append(agent.system_monitor.get_system_metrics())
                
                # Run collection in parallel
                agent_results = await asyncio.gather(*agent_tasks, return_exceptions=True)
                system_results = await asyncio.gather(*system_tasks, return_exceptions=True)
                
                # Draw interface
                line = 0
                line = self.ui.draw_header(line)
                
                for i, (agent_id, agent) in enumerate(self.agents.items()):
                    system_metrics = system_results[i] if i < len(system_results) and not isinstance(system_results[i], Exception) else {}
                    line = self.ui.draw_agent_section(line, agent, system_metrics)
                
                # Draw system section for the first agent's host
                if system_results and not isinstance(system_results[0], Exception):
                    line = self.ui.draw_system_section(line, system_results[0])
                
                self.ui.draw_footer(line)
                
                # Refresh screen
                self.ui.stdscr.refresh()
                
                # Check for user input
                for _ in range(self.refresh_interval * 10):  # Check 10 times per second
                    key = self.ui.stdscr.getch()
                    if key == ord('q') or key == 27:  # 'q' or ESC
                        self.running = False
                        break
                    elif key == ord('r'):  # 'r' for immediate refresh
                        break
                    await asyncio.sleep(0.1)
                    
        except KeyboardInterrupt:
            pass
        finally:
            self.ui.cleanup_curses()
            
    def add_agent(self, agent_id: str, endpoint: str, model: str):
        """
        Add a new agent to the monitoring system.
        
        Args:
            agent_id: Unique identifier for the new agent
            endpoint: HTTP endpoint for agent communication
            model: Model name running on the agent
        """
        self.agents[agent_id] = AgentMonitor(agent_id, endpoint, model)

async def main():
    """
    Main entry point for the advanced monitoring system.
    
    Creates and configures the monitoring system, optionally adds additional
    agents, and starts the monitoring loop.
    
    Example:
        # Add additional agents as needed:
        # monitor.add_agent('agent_114', 'http://192.168.1.114:11434', 'llama3:latest')
        # monitor.add_agent('agent_115', 'http://192.168.1.115:11434', 'codestral:latest')
    """
    monitor = AdvancedAgentMonitor()
    
    # You can add more agents here:
    # monitor.add_agent('agent_114', 'http://192.168.1.114:11434', 'llama3:latest')
    
    await monitor.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    except Exception as e:
        print(f"Error running monitor: {e}")