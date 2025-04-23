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

## Optimization Process

### Deterministic Optimization

The deterministic optimization process follows these steps:

1. Setup forecast data source:
   ```python
   file_name = f"data/together_forecast/phase_{phase_num}_forecast_sampled_1h.csv"
   scenario_gen = ScenarioFile_sliding(file_name, n_scenarios=n_scen, steps_ahead=24, steps_skip=steps_skip_forecast)
   ```

2. Initialize the optimization manager with specified update frequency:
   ```python
   log_exten = f"debug_logs/gurobi_phase_{phase_num}_step_leap_{steps_skip}_forecast_step_{steps_skip_forecast}.csv"
   manager = GurobiMPC(0, steps_skip=steps_skip, file_name=log_exten)
   ```

3. Create an agent and evaluate performance:
   ```python
   agent_used = GeneralAgent(scenario_gen, manager)
   tc_temp, _, _, _, _ = evaluate(agent_used, total_steps=total_steps, phase_num=phase_num, grid_include=True)
   ```

### Stochastic Optimization

The stochastic optimization uses a similar process but with different scenario generation:

1. Setup stochastic forecast data source:
   ```python
   scenario_gen = Scenario_Generator(
           forec_file=file_name,
           type="norm_noise",
           n_scenarios=n_scen,
           steps_ahead=24,
           revision_forec_freq=steps_skip_forecast,
           n_buildings=n_buildings
   )
   ```

2. Initialize the optimization manager:
   ```python
   manager = GurobiMPC(0, steps_skip=steps_skip, grid_include=grid_cost_bool, file_name=log_exten)
   ```

3. Create an agent and evaluate performance:
   ```python
   agent_used = GeneralAgent(scenario_gen, manager)
   tc, apc, aec, agc, agent_time_elapsed = evaluate(agent_used, total_steps=total_steps, phase_num=phase_num, grid_include=grid_cost_bool)
   ```

## Key Parameters

- **`steps_skip`**: Control frequency - how often optimization is recalculated (larger values mean less frequent recalculation)
- **`steps_skip_forecast`**: Forecast update frequency - how often the forecast is updated (larger values mean less frequent updates)
- **`n_scenarios`**: Number of scenarios used in stochastic optimization
- **`steps_ahead`**: Prediction horizon (24 hours in most experiments)
- **`phase_num`**: Dataset phase number (1 or 3)
- **`total_steps`**: Total number of simulation steps (typically 9000)
- **`grid_include`**: Whether to include grid costs in the optimization

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
  journal={Journal Name},
  year={2023}
}
```