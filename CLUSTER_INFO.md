# deepblackcloud Development Cluster

## Infrastructure Overview

This document provides detailed information about the deepblackcloud development cluster that powers the Distributed AI Development System.

### Network Configuration

**Base Network**: 192.168.1.0/24  
**Domain**: home.deepblack.cloud  
**Reverse Proxy**: Traefik  
**DNS**: Local DNS with wildcard *.home.deepblack.cloud  

### Hardware Specifications

#### ACACIA (192.168.1.72) - NAS & Services Host
- **Hostname**: acacia
- **OS**: Pop!_OS 22.04 LTS (Ubuntu-based)
- **CPU**: 2x Intel Xeon E5-2680 v4 @ 2.40GHz
  - 28 cores per socket, 56 threads total
  - L3 Cache: 70 MiB (35 MiB per socket)
- **Memory**: 128GB RAM (125Gi usable)
- **GPU**: NVIDIA GeForce GTX 1070 (8GB VRAM)
- **Storage**: 
  - 6x Hard drives (2x 2.7TB, 4x 3.6TB) - RAID array
  - 2x 240GB NVMe SSDs
  - 1x 1TB NVMe SSD (system drive)
- **Network Role**: NAS server, Docker Swarm worker
- **Services**: SSH, Cockpit, NFS server, Ollama, AnythingLLM

#### WALNUT (192.168.1.27) - Development Platform
- **Hostname**: walnut
- **OS**: Pop!_OS 22.04 LTS (Ubuntu-based)
- **Hardware**: Gigabyte X570 AORUS ELITE WIFI
- **CPU**: AMD Ryzen 7 5800X3D 8-Core @ 3.4-4.55GHz
  - 8 cores, 16 threads
  - 96MB L3 cache (3D V-Cache)
- **Memory**: 64GB RAM (62Gi usable)
- **GPU**: AMD RX 9060 XT (RDNA 4 architecture, 16GB VRAM)
- **Storage**: 2x 1TB NVMe SSDs
- **Network Role**: Docker Swarm manager
- **Services**: SSH, Cockpit, Docker Swarm manager, Ollama (28 models)

#### IRONWOOD (192.168.1.113) - Workstation
- **Hostname**: ironwood
- **OS**: Pop!_OS 22.04 LTS (Ubuntu-based)
- **Hardware**: MSI MS-7B09 (TRX40 platform)
- **CPU**: AMD Ryzen Threadripper 2920X 12-Core @ 3.5GHz
  - 12 cores, 24 threads
  - 32MB L3 cache
- **Memory**: 128GB RAM (125Gi usable)
- **GPU**: NVIDIA GeForce RTX 3070 (8GB VRAM)
- **Storage**: High-performance storage array
- **Network Role**: Docker Swarm worker, development workstation
- **Services**: SSH, Cockpit, Ollama (12 models)

### AI/ML Infrastructure

#### Ollama Model Distribution

**WALNUT (28 models)**: Comprehensive collection
- **Advanced Models**: starcoder2:15b, devstral, deepseek-coder-v2, qwen3
- **General Purpose**: phi4, llama3.1:8b, gemma3:12b, mistral:7b-instruct
- **Specialized**: llava (vision), qwq (reasoning), qwen2.5-coder
- **Embeddings**: mxbai-embed-large, nomic-embed-text
- **Legacy**: llama3.2, qwen2, gemma2, deepseek-r1 variants

**IRONWOOD (12 models)**: Development focused
- **Primary**: deepseek-coder-v2, devstral, phi4
- **Support**: llama3.1:8b, llava, qwen2.5-coder, mistral:7b-instruct

**ACACIA (6 models)**: Infrastructure focused
- **Primary**: deepseek-r1:7b, codellama, qwen2.5
- **Support**: llava, llama3.2

#### AnythingLLM RAG System
- **Host**: ACACIA (http://192.168.1.72:3051)
- **Features**: Persistent conversations, document integration, specialized workspaces
- **Integration**: Obsidian vault with real-time auto-sync
- **Workspaces**: 
  - Code Development Hub (starcoder2:15b)
  - DevOps & Infrastructure (devstral)
  - AI & Research Assistant (qwen3)
  - My Workspace (devstral)

#### n8n Workflow Automation
- **URL**: https://n8n.home.deepblack.cloud
- **Capabilities**: Intelligent model routing, code review, documentation generation
- **Integration**: Multi-machine load balancing across all Ollama instances

### Docker Swarm Configuration

#### Swarm Roles
- **Manager**: WALNUT (192.168.1.27)
- **Workers**: ACACIA (192.168.1.72), IRONWOOD (192.168.1.113)

#### Service Distribution
- **Traefik**: Load balancer and reverse proxy
- **Portainer**: Container management UI
- **n8n**: Workflow automation
- **AnythingLLM**: Knowledge management (ACACIA)
- **Cockpit**: System monitoring (all nodes)

### Storage Architecture

#### NFS Shares (ACACIA)
- **Mount Point**: /rust/containers/
- **Usage**: Docker Compose files, shared configurations
- **Access**: All cluster nodes have NFS access

#### Local Storage
- **WALNUT**: 2TB NVMe (development workspaces, Docker volumes)
- **IRONWOOD**: High-performance array (build caches, temporary files)
- **ACACIA**: RAID array (persistent data, backups, media storage)

### Network Services

#### External Access (via Traefik)
- **n8n**: https://n8n.home.deepblack.cloud
- **Portainer**: https://*.home.deepblack.cloud/portainer
- **Swarmpit**: https://*.home.deepblack.cloud/swarmpit
- **Filebrowser**: https://*.home.deepblack.cloud/filebrowser

#### Direct Access
- **AnythingLLM**: http://192.168.1.72:3051
- **Cockpit**: https://192.168.1.{72,27,113}:9090
- **Ollama APIs**: http://192.168.1.{72,27,113}:11434

### Authentication & Security

#### Access Control
- **SSH Access**: Key-based authentication preferred
- **Web Services**: Traefik handles SSL termination
- **Internal Network**: All AI services on private network
- **Firewall**: UFW configured on all nodes

#### Credentials Management
- **SSH Password**: Stored in `tony-pass` file
- **Service Passwords**: Environment variables and secrets
- **API Keys**: Centralized in configuration files

### Performance Characteristics

#### AI Inference Performance
- **WALNUT**: 8+ TPS (starcoder2:15b), 20s response time
- **IRONWOOD**: 6+ TPS (deepseek-coder-v2), 25s response time  
- **ACACIA**: 3+ TPS (deepseek-r1:7b), 30s response time

#### Resource Utilization
- **CPU**: 60-80% average across cluster during active development
- **Memory**: 70-90% utilization with large models loaded
- **GPU**: On-demand loading based on model requirements
- **Network**: Gigabit Ethernet, low latency internal communication

### Monitoring & Observability

#### System Monitoring
- **Cockpit**: Web-based system monitoring per node
- **CLI Tools**: htop, iotop, nvidia-smi for real-time monitoring
- **Distributed Monitoring**: Custom Python scripts for agent performance

#### Log Aggregation
- **Local Logs**: /var/log/ on each node
- **Application Logs**: Docker container logs via Portainer
- **Agent Logs**: Centralized in distributed-ai-dev system

### Backup & Recovery

#### Backup Strategy
- **Configuration**: Git repositories for all configurations
- **Data**: NFS shares backed up to external storage
- **Snapshots**: Docker volume snapshots for stateful services
- **Documentation**: All system knowledge in Obsidian vault

#### Recovery Procedures
- **Node Failure**: Docker Swarm handles service migration
- **Data Loss**: Restore from NFS backups
- **Configuration**: Git-based configuration recovery
- **Service Recovery**: Docker Compose recreation from shared storage

### Development Workflow Integration

#### Version Control
- **Primary**: GitHub repositories
- **Local**: Git repositories on shared storage
- **Backup**: Multiple repository mirrors

#### CI/CD Integration
- **GitHub Actions**: Triggered by repository events
- **Local Runners**: Self-hosted runners on cluster nodes
- **Docker Registry**: Local registry for custom images
- **Deployment**: Automated deployment to staging/production

### Scaling Considerations

#### Horizontal Scaling
- **Additional Nodes**: Easy addition via Docker Swarm join
- **Load Balancing**: Traefik handles automatic load distribution
- **Service Scaling**: Docker Swarm manages service replicas

#### Vertical Scaling
- **Memory**: All nodes support memory expansion
- **Storage**: RAID arrays allow capacity expansion
- **GPU**: PCIe slots available for additional GPUs
- **CPU**: Platform-specific upgrade paths available

### Cost Analysis

#### Hardware Investment
- **Total Investment**: ~$15,000 for complete cluster
- **Operational Cost**: <$200/month (electricity, internet)
- **Maintenance**: Minimal due to redundant design

#### ROI Comparison
- **Cloud Alternative**: $2000-5000/month for equivalent AI compute
- **Payback Period**: 6-12 months
- **Control Benefits**: Full data sovereignty, no usage limits

### Future Roadmap

#### Planned Upgrades
- **GPU Expansion**: Additional RTX 4090 or MI300X cards
- **Memory Scaling**: Upgrade to 256GB on compute nodes
- **Storage**: NVMe expansion for faster AI model loading
- **Networking**: 10GbE upgrade for faster inter-node communication

#### Software Enhancements
- **Model Management**: Automated model updates and optimization
- **Workload Scheduling**: Advanced job scheduling and resource allocation
- **Monitoring**: Enhanced observability and alerting
- **Security**: Zero-trust networking and enhanced authentication

This cluster provides a robust foundation for distributed AI development with room for significant growth and enhancement as requirements evolve.