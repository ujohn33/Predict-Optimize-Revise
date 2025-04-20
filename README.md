# Predict-Optimize-Revise

Repository for the paper: **"Balancing Forecast Accuracy and Switching Costs in Online Optimization of Energy Management Systems"**

## Overview

This repository contains the implementation and experimental results for studying the trade-off between forecast update frequency and control action recalculation in energy management systems. The code explores how different commitment levels affect system performance, particularly focusing on the balance between forecast accuracy and the costs associated with frequently changing control decisions.

## Repository Structure

### Core Components

- **`agents/`**: Implementation of agents that combine forecasting and optimization
  - `general_agent.py`: Main agent implementation used in experiments
  - `orderenforcingwrapper.py`: Wrapper for compatibility with CityLearn environment

- **`ems/`**: Energy Management System optimization models
  - `gurobi_mpc.py`: Model Predictive Control implementation using Gurobi solver

- **`forecast/`**: Forecast generation and scenario modeling
  - `scenarios_lean.py`: Various scenario generator implementations
  - `file.py`: File-based scenario handlers with sliding window implementations

- **`utils/`**: Utility functions for data processing and visualization
- **`rewards/`**: Reward function definitions for the energy management system

### Experimental Scripts

- **`local_evaluation.py`**: Script for local evaluation of agents
- **`hpc_run.py`**: Script for running experiments on HPC clusters
- **`run_hpc.sbatch`**: SLURM script for executing experiments in parallel on HPC systems

### Key Experiments

#### Optimization with Different Commitment Levels

The experiments studying the effect of different commitment strategies are run in:
- `notebooks/optimization/deterministic_opt_scores.ipynb`: Runs optimization with varying commitment levels

#### Theoretical Analysis Illustrations

The theoretical analysis from the paper is illustrated in:
- `notebooks/analysis/demo_convergence.ipynb`: Visualizations supporting the convergence analysis

## Usage

### Local Evaluation

To run the local evaluation:

```bash
python local_evaluation.py
```

### Running on HPC

For large-scale parameter studies, use the SLURM script:

```bash
sbatch run_hpc.sbatch
```

## Results

The `results/` directory contains experimental results from the paper. The logs of the stochastic runs are available on request.

## Notes

- The `debug_logs/` and `archive/` folders should be ignored as they contain temporary and debug information
- When running experiments, you can configure different parameters like `steps_skip` (control update frequency) and `steps_skip_forecast` (forecast update frequency)
- The code supports multiple dataset phases (phase 1 and phase 3 are most commonly used)

## References

If you use this code in your research, please cite:
```
@article{your-citation-info,
  title={Balancing Forecast Accuracy and Switching Costs in Online Optimization of Energy Management Systems},
  author={Author Names},
  year={2023}
}
```