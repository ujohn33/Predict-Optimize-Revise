#!/bin/bash
####  SBATCH --output=/dev/null
#SBATCH --job-name=citylearn_scenarios
#SBATCH --time=1-00:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --partition=skylake,skylake_mpi
#SBATCH --mem-per-cpu=1G
#SBATCH --array=0-11
#SBATCH --mail-type=ALL


# 145 was the previous number
# File to run on HPC
# First activate your python environment (you might have to create your own) with:
# virtualenv --system-site-packages $VSC_DATA/scenarioenv
# And then intall the requirements with
# pip install -r requirements.txt

module load Python/3.10.4-GCCcore-11.3.0
module load matplotlib/3.5.2-foss-2022a
module load SciPy-bundle/2022.05-foss-2022a
module load legacy-software
module load Gurobi/9.1.2

source /scratch/brussel/105/vsc10528/distiboost/bin/activate

cd /scratch/brussel/vo/000/bvo00037/vsc10528/citylearn_scenarios

python hpc_run.py $SLURM_ARRAY_TASK_ID
