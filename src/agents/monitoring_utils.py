#!/usr/bin/env python3
"""
Monitoring Utilities for Distributed AI Agents

This module contains shared classes and functions for the agent
monitoring system, including performance tracking and system resource
collection.
"""

import asyncio
import aiohttp
import json
import time
import psutil
import subprocess
from collections import deque
from typing import Dict, Any

class AgentPerformanceMetrics:
    """
    Tracks and analyzes performance metrics for AI agents.
    """
    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.metrics_history = deque(maxlen=max_history)
        self.task_history = deque(maxlen=50)

    def add_metrics(self, metrics: Dict[str, Any]):
        metrics['timestamp'] = time.time()
        self.metrics_history.append(metrics)

    def add_task(self, task_info: Dict[str, Any]):
        task_info['timestamp'] = time.time()
        self.task_history.append(task_info)

    def get_avg_tps(self, window_minutes: int = 5) -> float:
        cutoff = time.time() - (window_minutes * 60)
        recent = [m for m in self.metrics_history if m['timestamp'] > cutoff]
        if not recent:
            return 0.0
        return sum(m.get('tokens_per_second', 0) for m in recent) / len(recent)

    def get_avg_response_time(self, window_minutes: int = 5) -> float:
        cutoff = time.time() - (window_minutes * 60)
        recent = [m for m in self.metrics_history if m['timestamp'] > cutoff]
        if not recent:
            return 0.0
        return sum(m.get('total_duration_ms', 0) for m in recent) / len(recent)

class SystemResourceMonitor:
    """
    Monitors system resources including CPU, memory, and GPU utilization.
    """
    def __init__(self, agent_host: str = "localhost"):
        self.agent_host = agent_host

    async def get_system_metrics(self) -> Dict[str, Any]:
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
        gpu_info = {'available': False, 'type': 'unknown'}

        try:
            result = subprocess.run(['rocm-smi', '--showuse', '--showmemuse', '--csv'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                gpu_info = self.parse_rocm_smi(result.stdout)
                gpu_info['type'] = 'AMD ROCm'
                gpu_info['available'] = True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            try:
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
        lines = output.strip().split('\n')
        if len(lines) < 2:
            return {'available': False}

        header = lines[0].split(',')
        data = lines[1].split(',')
        gpu_data = dict(zip(header, data))

        return {
            'utilization': float(gpu_data.get('GPU use (%)', 0)),
            'memory_used': float(gpu_data.get('Memory use (%)', 0)),
            'temperature': float(gpu_data.get('Temperature (°C)', 0)) if 'Temperature (°C)' in gpu_data else 0
        }

    def parse_nvidia_smi(self, output: str) -> Dict[str, Any]:
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
