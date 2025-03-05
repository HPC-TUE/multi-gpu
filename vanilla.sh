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

export TRITON_CACHE_DIR='./.tritone' # Change the Triton dir for Deepspeed
export NCCL_DEBUG=WARN
if [[ "$SLURM_JOB_PARTITION" == *h100* ]]; then
    export NCCL_SOCKET_IFNAME="eno2np0"
else
    export NCCL_SOCKET_IFNAME="eno1np0"
fi 

export MASTER_PORT=12340
export WORLD_SIZE=$SLURM_NTASKS
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

srun python ./vanilla.py --nodes=2 "$@"