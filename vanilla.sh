#!/bin/bash
#SBATCH --job-name=test
#SBATCH --time=00:10:00
#SBATCH --partition=gpu_h100
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --tasks-per-node=4
#SBATCH --output=./multi_node/%x_%A_%a.out

module load 2023
module load foss/2023a
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.4.0

source ./venv/bin/activate

export MASTER_PORT=12340
export WORLD_SIZE=$SLURM_NTASKS
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

srun python ./vanilla.py --entity=bjmaat --project_name=multi_slurm_test --run_name=test_auto --strategy=auto