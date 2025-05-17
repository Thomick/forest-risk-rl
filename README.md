# Tree-level Model of Forest Growth with Interactions Between Trees

A reinforcement learning environment (via Gym) for modeling forest growth dynamics that accounts for tree-to-tree interactions.

*Developed by Thomas Michel*

## Overview

This project provides a reinforcement learning framework for simulating forest management and growth. The environments implement the Gym interface, allowing RL agents to interact with the forest simulation through states, actions, rewards, and transitions. The implementation models tree-level growth patterns and inter-tree interactions, enabling research on optimal forest management strategies.

## Project Status

The codebase is functional as a Gym-compatible RL environment library. The experimental components require direct code modification to adjust parameters here as examples of how to use the code.

## Repository Structure

The main code is organized in the `forest_risk_rl` folder:

- **`envs/`** and **`learners/`** - Contains code adapted from the [Average Reward Reinforcement Learning repository](https://gitlab.inria.fr/omaillar/average-reward-reinforcement-learning)
- **`basic_model/`** - Preliminary implementation of the basic discrete model
- **`envs/discrete_env.py`** and **`envs/linear_dynamic_env.py`** - Core implementations of the forest RL environments
- **`experiments/`** - Experiment scripts and visualization code for testing RL algorithms

## Installation and Usage

### Option 1: Using a virtual environment (recommended)

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run an experiment
python forest_risk_rl/experiments/[experiment_name].py
```

### Option 2: Direct execution (if dependencies are already installed)

```bash
python -m forest_risk_rl.experiments.[experiment_name]
```
