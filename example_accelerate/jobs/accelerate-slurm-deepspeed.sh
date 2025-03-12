#!/bin/bash
#SBATCH --job-name=hf_accelerate_train_deepspeed
#SBATCH --time=0:10:00
#SBATCH --partition=gpu_h100
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --tasks-per-node=4 # crucial - only 1 task per dist per node!
#SBATCH --output=./results/%x_%A_%a.out

echo "Starting multi-node job on host $(hostname)..."
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

##### For debugging 
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_TIMEOUT=60
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2 # enables direct memory access between GPUs across different nodes 
######

if [[ "$SLURM_JOB_PARTITION" == *h100* ]]; then
    echo "H100: NCCL_SOCKET_IFNAME = eno2np0"
    export NCCL_SOCKET_IFNAME="eno2np0"
else
    echo "Not H100: NCCL_SOCKET_IFNAME = eno1np0"
    export NCCL_SOCKET_IFNAME="eno1np0"
fi 
export TRITON_CACHE_DIR='./.tritone' # Change the Triton dir for Deepspeed
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

echo "Node: $(hostname)"
echo "Process ID (PID): $$"
echo "Machine rank: $SLURM_PROCID"

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
    MASTER_PORT=29500  # Default port for distributed training
fi

echo "START TIME: $(date)"

# Run the training script with Accelerate
# Note: it is important to escape `$SLURM_PROCID` since we want the srun on each node to evaluate this variable
accelerate launch --config_file ./example_accelerate/configs/deepspeed_config.yaml \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $SLURM_PROCID \
    --num_processes=$(($SLURM_NNODES * $SLURM_GPUS_PER_NODE)) \
    --num_machines=$SLURM_NNODES \
    ./example_accelerate/accelerate-training-script.py\
    --project_name="multi_slurm_test_accelerate" \
    --entity="AI_team_SCC_TUe" \
    --run_name="test_accelerate_multinode_deepspeed_example" \
    $CHECKPOINT_ARG "$@"

echo "END TIME: $(date)"
