#!/usr/bin/env python3
"""
Configuration for Agent 192.168.1.113
DevStral-powered ROCm development specialist
"""

from ai_dev_coordinator import Agent, AgentType

# Agent 113 Configuration
AGENT_113_CONFIG = {
    'id': 'devstral_architect',
    'endpoint': 'http://192.168.1.113:11434',
    'model': 'devstral:latest',  # 23.6B parameter coding specialist
    'specialty': 'kernel_dev',   # Best fit given DevStral's strengths
    'max_concurrent': 1,         # Conservative for 23B model
    'description': 'DevStral-powered architecture and kernel development specialist',
    'strengths': [
        'Complex code architecture',
        'Multi-file optimization projects', 
        'Advanced kernel algorithms',
        'Performance-critical algorithms'
    ]
}

# Alternative configurations based on task needs
ALTERNATIVE_CONFIGS = {
    'advanced_profiler': {
        'id': 'devstral_profiler',
        'endpoint': 'http://192.168.1.113:11434',
        'model': 'deepseek-r1:7b',  # Good for analysis tasks
        'specialty': 'profiler',
        'max_concurrent': 2
    },
    'documentation_expert': {
        'id': 'devstral_docs',
        'endpoint': 'http://192.168.1.113:11434', 
        'model': 'phi4:latest',  # Excellent for technical writing
        'specialty': 'docs_writer',
        'max_concurrent': 2
    },
    'pytorch_integration': {
        'id': 'devstral_pytorch',
        'endpoint': 'http://192.168.1.113:11434',
        'model': 'llama3.1:8b',  # Good balance for PyTorch work
        'specialty': 'pytorch_dev',
        'max_concurrent': 2
    }
}

def get_agent_config(specialization='primary'):
    """Get agent configuration for specific specialization"""
    if specialization == 'primary':
        return AGENT_113_CONFIG
    else:
        return ALTERNATIVE_CONFIGS.get(specialization, AGENT_113_CONFIG)

def create_agent_113(specialization='primary'):
    """Create Agent 113 instance with specified specialization"""
    config = get_agent_config(specialization)
    
    return Agent(
        id=config['id'],
        endpoint=config['endpoint'],
        model=config['model'],
        specialty=AgentType(config['specialty']),
        max_concurrent=config['max_concurrent']
    )

# Recommended specialization based on available models
RECOMMENDED_SETUP = {
    'primary_role': 'kernel_dev',
    'model': 'devstral:latest',
    'reasoning': """
    DevStral (23.6B) is ideal for:
    - Complex kernel architecture design
    - Multi-file ROCm optimizations  
    - Advanced algorithm implementation
    - Code review and refactoring
    
    This fills a critical gap as your 'senior developer' agent
    for complex ROCm optimization projects.
    """
}

if __name__ == "__main__":
    print("Agent 113 Configuration")
    print("=" * 40)
    print(f"Recommended: {RECOMMENDED_SETUP['primary_role']}")
    print(f"Model: {RECOMMENDED_SETUP['model']}")
    print("\nReasoning:")
    print(RECOMMENDED_SETUP['reasoning'])
    
    # Test agent creation
    agent = create_agent_113()
    print(f"\nCreated agent: {agent.id}")
    print(f"Endpoint: {agent.endpoint}")
    print(f"Specialty: {agent.specialty.value}")