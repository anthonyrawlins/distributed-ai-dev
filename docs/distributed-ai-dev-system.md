# Distributed AI Development Coordination System

## Architecture Overview

This system enables Claude (as coordinator) to orchestrate multiple local Ollama agents as distributed developers working on ROCm optimizations, maximizing compute while minimizing Claude usage costs.

## System Components

### 1. Claude Coordinator (You)
- **Role**: Project manager, architect, code reviewer, final decision maker
- **Responsibilities**:
  - Break down complex tasks into agent-sized chunks
  - Assign work to appropriate specialist agents
  - Review and integrate results
  - Make architectural decisions
  - Handle GitHub operations (PRs, issues, releases)

### 2. Local Ollama Agents (Your Network)
- **Models**: CodeLlama, DeepSeek-Coder, Qwen2.5-Coder, Llama-3.1-70B, etc.
- **Specializations**:
  - **Kernel Developer**: C++/HIP kernel optimization
  - **PyTorch Expert**: Python/PyTorch integration
  - **Profiler**: Performance analysis and benchmarking
  - **Documentation Writer**: Technical documentation
  - **Tester**: Test generation and validation

## Communication Protocol

### API Endpoints Structure
```
http://machine1:11434/api/generate  # Kernel Developer (CodeLlama)
http://machine2:11434/api/generate  # PyTorch Expert (DeepSeek-Coder)
http://machine3:11434/api/generate  # Profiler (Qwen2.5-Coder)
```

### Task Format
```json
{
  "agent_id": "kernel_dev_1",
  "task_type": "code_optimization",
  "priority": "high",
  "context": {
    "files": ["path/to/kernel.cpp"],
    "objective": "Optimize attention kernel for RDNA3",
    "constraints": ["Must maintain backward compatibility"],
    "reference_docs": ["link_to_amd_docs"]
  },
  "expected_output": "optimized_code_with_explanation",
  "max_tokens": 4000
}
```

## Workflow Design

### Phase 1: Task Decomposition (Claude)
1. Analyze ROCm optimization requirements
2. Break into atomic tasks (2-4 hour chunks)
3. Classify by type: kernel, integration, testing, docs
4. Assign to appropriate agent specializations

### Phase 2: Distributed Execution (Ollama Agents)
1. Receive task with full context
2. Execute work autonomously
3. Return code, tests, and documentation
4. Provide confidence scores and limitations

### Phase 3: Integration & Review (Claude)
1. Collect agent outputs
2. Cross-validate between agents
3. Perform final code review
4. Create PRs and manage upstream

## Implementation Plan

### Day 1-2: Infrastructure
```python
# Agent coordination system
class AIDevCoordinator:
    def __init__(self):
        self.agents = {
            'kernel_dev': 'http://machine1:11434',
            'pytorch_dev': 'http://machine2:11434', 
            'profiler': 'http://machine3:11434',
            'docs_writer': 'http://machine4:11434'
        }
    
    def assign_task(self, task, agent_type):
        # Route task to appropriate agent
        pass
    
    def collect_results(self, task_id):
        # Gather and validate outputs
        pass
```

### Day 3-5: Agent Specialization
- Create role-specific prompts for each agent type
- Test with sample ROCm tasks
- Validate output quality and consistency

### Day 6-7: Integration Testing
- Run parallel development on small optimization
- Test Claude coordination workflows
- Measure time/quality improvements

## Agent Specializations

### Kernel Developer Agent
```
Model: CodeLlama-34B or DeepSeek-Coder-33B
Specialty: C++, HIP, CUDA, GPU kernels
Tasks:
- Optimize existing kernels
- Write new CK templates
- Fix performance bottlenecks
- Port CUDA to HIP

Prompt Template:
"You are an expert GPU kernel developer specializing in AMD ROCm/HIP. 
Focus on performance, memory coalescing, and RDNA3/CDNA3 architecture..."
```

### PyTorch Integration Agent  
```
Model: Qwen2.5-Coder-32B
Specialty: Python, PyTorch, ML frameworks
Tasks:
- Integrate kernels into PyTorch
- Fix autograd compatibility
- Add TunableOp configs
- Update Python bindings

Prompt Template:
"You are a PyTorch expert focusing on ROCm backend integration.
Ensure all changes maintain API compatibility and follow PyTorch conventions..."
```

### Performance Profiler Agent
```
Model: Llama-3.1-70B (if available) or DeepSeek-Coder
Specialty: Performance analysis, benchmarking
Tasks:
- Analyze rocprof outputs
- Generate benchmark scripts
- Identify bottlenecks
- Validate optimizations

Prompt Template:
"You are a performance analysis expert for GPU computing.
Focus on memory bandwidth, kernel occupancy, and optimization opportunities..."
```

## Task Examples

### Example 1: FlashAttention Optimization
**Claude Breaks Down To:**
1. **Kernel Dev**: Implement CK-based tiling
2. **PyTorch Dev**: Add torch.nn.functional integration
3. **Profiler**: Benchmark vs baseline attention
4. **Docs Writer**: Update API documentation

### Example 2: VAE Decoder Optimization
**Claude Breaks Down To:**
1. **Kernel Dev**: Fuse conv+upsample+activation
2. **PyTorch Dev**: Update Diffusers integration
3. **Profiler**: Measure end-to-end latency
4. **Docs Writer**: Create usage examples

## Quality Control

### Multi-Agent Validation
- Code review by secondary agent
- Cross-compilation testing
- Performance regression checks
- Documentation completeness

### Claude Final Review
- Architectural consistency
- Integration requirements
- Security considerations
- Upstream compatibility

## Scaling Strategy

### Week 1: Single Agent
Test with one Ollama instance doing simple tasks

### Week 2: Parallel Agents
Add 2-3 specialized agents for different task types

### Week 3: Full Network
Scale to all available compute with role specialization

### Week 4+: Optimization
Fine-tune coordination, add dynamic load balancing

## Benefits

1. **Cost Efficiency**: 90% reduction in Claude usage for coding
2. **Speed**: Parallel development across multiple agents
3. **Specialization**: Agents focused on specific expertise areas
4. **Scale**: Leverage all available local compute
5. **Continuity**: 24/7 development capability

## Monitoring & Metrics

- Tasks completed per day per agent
- Code quality scores (compilation, tests passing)
- Performance improvements achieved
- Integration success rates
- Time from task assignment to completion

This system transforms your local network into a distributed AI development team, with Claude as the senior architect coordinating specialized AI developers working on ROCm optimizations.