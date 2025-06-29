#!/usr/bin/env python3
"""
Unified Agent Monitoring System

A comprehensive, configuration-driven monitoring system for the
Distributed AI Development System. Supports multiple UI modes and
loads all configurations from a central YAML file.
"""

import asyncio
import aiohttp
import yaml
import argparse
import os
from pathlib import Path
from typing import Dict, Any

from monitoring_utils import AgentPerformanceMetrics, SystemResourceMonitor

# Conditional imports for different UI modes
try:
    import curses
    CURSES_AVAILABLE = True
except ImportError:
    CURSES_AVAILABLE = False

class AgentMonitor:
    """
    Monitors a single AI agent's performance and system resources.
    """
    def __init__(self, agent_id: str, endpoint: str, model: str):
        self.agent_id = agent_id
        self.endpoint = endpoint
        self.model = model
        self.metrics = AgentPerformanceMetrics()
        self.system_monitor = SystemResourceMonitor()
        self.last_status = "Unknown"

    async def collect_detailed_metrics(self, test_prompt: str = "Hello, provide a brief status update.") -> Dict[str, Any]:
        # ... (omitted for brevity - will be the same as the original implementation)
        pass

class ConsoleInterface:
    """
    Manages the console output, supporting both simple and advanced UI modes.
    """
    def __init__(self, ui_mode: str):
        self.ui_mode = ui_mode
        self.stdscr = None

    def setup(self):
        if self.ui_mode == 'advanced' and CURSES_AVAILABLE:
            self.stdscr = curses.initscr()
            curses.noecho()
            curses.cbreak()
            self.stdscr.keypad(True)
            self.stdscr.nodelay(True)
            if curses.has_colors():
                curses.start_color()
                curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
                curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
                curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
                curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
                curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK)

    def teardown(self):
        if self.ui_mode == 'advanced' and self.stdscr:
            curses.nocbreak()
            self.stdscr.keypad(False)
            curses.echo()
            curses.endwin()

    def draw(self, agents: Dict[str, AgentMonitor]):
        if self.ui_mode == 'advanced' and self.stdscr:
            self.stdscr.clear()
            # ... (omitted for brevity - will be the same as the original advanced_monitor)
            self.stdscr.refresh()
        else:
            os.system('clear' if os.name == 'posix' else 'cls')
            # ... (omitted for brevity - will be the same as the original simple_monitor)

class UnifiedMonitor:
    """
    Main monitoring system coordinator.
    """
    def __init__(self, config_path: str, ui_mode: str):
        self.config = self.load_config(config_path)
        self.agents = self.initialize_agents()
        self.ui = ConsoleInterface(ui_mode)
        self.running = False

    def load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def initialize_agents(self) -> Dict[str, AgentMonitor]:
        agents = {}
        for agent_id, agent_config in self.config.get('agents', {}).items():
            if agent_config.get('status', 'disabled') == 'active':
                agents[agent_id] = AgentMonitor(
                    agent_id=agent_id,
                    endpoint=agent_config['endpoint'],
                    model=agent_config['model']
                )
        return agents

    async def run(self):
        self.ui.setup()
        self.running = True
        try:
            while self.running:
                # ... (omitted for brevity - will be the same as the original main loop)
                self.ui.draw(self.agents)
                await asyncio.sleep(self.config.get('monitoring', {}).get('refresh_interval_seconds', 5))
        finally:
            self.ui.teardown()

def main():
    parser = argparse.ArgumentParser(description="Unified Agent Monitoring System")
    parser.add_argument("--config", default="config/agents.yaml", help="Path to the agent configuration file.")
    parser.add_argument("--ui", default="simple", choices=["simple", "advanced"], help="The user interface mode.")
    args = parser.parse_args()

    if args.ui == 'advanced' and not CURSES_AVAILABLE:
        print("Advanced UI requires the curses library, which is not available on this system. Falling back to simple UI.")
        args.ui = 'simple'

    monitor = UnifiedMonitor(args.config, args.ui)
    asyncio.run(monitor.run())

if __name__ == "__main__":
    main()
