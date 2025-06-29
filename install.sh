#!/bin/bash

# This script automates the setup process for the
# Distributed AI Development System.

# Check for Python
if ! command -v python3 &> /dev/null
then
    echo "Python 3 could not be found. Please install it."
    exit 1
fi

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment and install dependencies
source .venv/bin/activate

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Check for config file
if [ ! -f "config/agents.yaml" ]; then
    echo "Configuration file not found."
    echo "Copying template to config/agents.yaml..."
    cp config/agents.yaml.template config/agents.yaml
    echo "Please edit config/agents.yaml to define your agent network."
fi

echo "Setup complete!"
