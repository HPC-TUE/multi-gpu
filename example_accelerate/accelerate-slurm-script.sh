#!/bin/bash
#SBATCH --job-name=hf_accelerate_train
#SBATCH --time=0:05:00
#SBATCH --partition=gpu_h100
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --tasks-per-node=4
#SBATCH --output=./results/accelerate_train_%x_%A_%a.out

echo "Starting multi-node job on $(hostname)..."
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Nodes allocated: $SLURM_JOB_NODELIST"

# Load necessary modules
module load 2023
module load foss/2023a
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.4.0

. /etc/bashrc
. ~/.bashrc

# Activate virtual environment
source ./venv/bin/activate

export TRITON_CACHE_DIR='./.tritone' # Change the Triton dir for Deepspeed
export NCCL_SOCKET_IFNAME="eno2np0" # For H100
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

echo "Process ID (PID): $$"

# Optional: Specify checkpoint to resume from (comment out if starting fresh)
# CHECKPOINT_PATH="./checkpoints/checkpoint_X"

# Conditional checkpoint resuming
if [ -n "$CHECKPOINT_PATH" ] && [ -d "$CHECKPOINT_PATH" ]; then
    CHECKPOINT_ARG="--resume_from_checkpoint=$CHECKPOINT_PATH"
else
    CHECKPOINT_ARG=""
fi

# Check if MASTER_PORT is not set, and provide a default if it's empty
if [ -z "$MASTER_PORT" ]; then
    MASTER_PORT=29500  # Common default port for distributed training
fi
