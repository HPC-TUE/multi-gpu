#!/bin/bash
#SBATCH --job-name=test
#SBATCH --time=1:00:00
#SBATCH --partition=gpu_a100
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --tasks-per-node=4
#SBATCH --output=./multi_node/%x_%A_%a.out
#SBATCH --exclusive

module load 2023
module load foss/2023a
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.4.0

source ./venv/bin/activate

export TRITON_CACHE_DIR='./.tritone'
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME="eno1np0" # Tell Snellius to use traditional networking, related to infiniband issues: https://servicedesk.surf.nl/wiki/display/WIKI/Snellius+known+issues#Snelliusknownissues-UsingNCCLforGPU%3C=%3EGPUcommunication
export NCCL_IGNORE_DUPLICATE_IDS=1

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip

echo $$

# 
srun torchrun \
  --nnodes 2 \
  --nproc_per_node 4 \
  --rdzv_id $RANDOM \
  --rdzv_backend c10d \
  --rdzv_endpoint $head_node_ip:29500 \
  ./vanilla.py "$@" --entity=bjmaat --project_name=multi_slurm_test --run_name=test_auto --strategy=auto