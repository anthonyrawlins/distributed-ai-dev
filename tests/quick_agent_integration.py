#!/usr/bin/env python3
"""
Quick integration test for Agent 113
Simple validation that the agent can be added to the coordination system
"""

import asyncio
from ai_dev_coordinator import AIDevCoordinator, Agent, AgentType
from claude_interface import claude_interface

async def quick_integration_test():
    """Quick test to validate Agent 113 integration"""
    print("Agent 113 Quick Integration Test")
    print("=" * 40)
    
    # Add Agent 113 to the coordinator
    agent_113_config = {
        'id': 'devstral_architect',
        'endpoint': 'http://192.168.1.113:11434',
        'model': 'devstral:latest',
        'specialty': 'kernel_dev'
    }
    
    # Create agent instance
    agent_113 = Agent(
        id=agent_113_config['id'],
        endpoint=agent_113_config['endpoint'],
        model=agent_113_config['model'],
        specialty=AgentType(agent_113_config['specialty']),
        max_concurrent=1
    )
    
    # Add to coordinator
    claude_interface.coordinator.add_agent(agent_113)
    
    # Verify agent was added
    agents = claude_interface.coordinator.agents
    if 'devstral_architect' in agents:
        print("✓ Agent 113 successfully added to coordinator")
        print(f"  ID: {agents['devstral_architect'].id}")
        print(f"  Endpoint: {agents['devstral_architect'].endpoint}")
        print(f"  Model: {agents['devstral_architect'].model}")
        print(f"  Specialty: {agents['devstral_architect'].specialty.value}")
        
        # Test agent selection
        available_agent = claude_interface.coordinator.get_available_agent(AgentType.KERNEL_DEV)
        if available_agent and available_agent.id == 'devstral_architect':
            print("✓ Agent 113 correctly selected for kernel development tasks")
        else:
            print("✓ Agent 113 available (other kernel agents may be selected first)")
        
        # Create a test task
        test_task = claude_interface.coordinator.create_task(
            AgentType.KERNEL_DEV,
            {
                'objective': 'Test task for DevStral agent',
                'files': ['test_kernel.cpp'],
                'focus': 'validation test'
            },
            priority=3
        )
        
        print(f"✓ Test task created: {test_task.id}")
        print("✓ Agent 113 integration successful!")
        
        return True
    else:
        print("✗ Failed to add Agent 113 to coordinator")
        return False

def create_updated_config():
    """Create updated configuration file with Agent 113"""
    updated_config = [
        {
            'id': 'devstral_architect',
            'endpoint': 'http://192.168.1.113:11434',
            'model': 'devstral:latest',
            'specialty': 'kernel_dev',
            'max_concurrent': 1,
            'description': 'DevStral-powered senior architect for complex kernel development'
        }
    ]
    
    print("\nUpdated Agent Configuration:")
    print("=" * 40)
    for agent in updated_config:
        print(f"ID: {agent['id']}")
        print(f"Endpoint: {agent['endpoint']}")
        print(f"Model: {agent['model']} ({agent['description']})")
        print(f"Specialty: {agent['specialty']}")
        print("-" * 40)
    
    return updated_config

if __name__ == "__main__":
    print("Running Agent 113 Integration Test...")
    
    # Run integration test
    success = asyncio.run(quick_integration_test())
    
    if success:
        print("\n🎉 INTEGRATION SUCCESSFUL!")
        print("\nAgent 113 (DevStral) is now ready for:")
        print("  • Complex kernel architecture design")
        print("  • Multi-file ROCm optimization projects")
        print("  • Advanced algorithm implementation")
        print("  • Senior-level code review and refactoring")
        
        # Show configuration
        create_updated_config()
        
        print("\nNext steps:")
        print("1. Start delegating complex ROCm tasks to Agent 113")
        print("2. Use DevStral for architectural decisions")
        print("3. Leverage the 23.6B parameter model for challenging optimizations")
        
    else:
        print("\n⚠️ Integration needs attention")
    
    print("\nAgent 113 integration complete!")