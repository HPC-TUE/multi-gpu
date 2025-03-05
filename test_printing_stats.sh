#!/bin/bash
#SBATCH --job-name=test
#SBATCH --time=00:05:00
#SBATCH --partition=gpu_h100
#SBATCH -N 2
#SBATCH --gpus-per-node=4
#SBATCH --tasks-per-node=4
#SBATCH --output=./results/%x_%A_%a.out

echo "Starting multi-node job on $(hostname)..."
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Nodes allocated: $SLURM_JOB_NODELIST"

module load 2023
module load foss/2023a
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.4.0

. /etc/bashrc
. ~/.bashrc

source ./venv/bin/activate

# export NCCL_SOCKET_IFNAME="eno1np0" # For A100. Tell Snellius to use traditional networking, related to infiniband issues: https://servicedesk.surf.nl/wiki/display/WIKI/Snellius+known+issues#Snelliusknownissues-UsingNCCLforGPU%3C=%3EGPUcommunication
export NCCL_SOCKET_IFNAME="eno2np0" # For H100

echo "Process ID (PID): $$"
echo "-----------------------------"

srun python -c "
import os
import torch
import socket

local_rank = int(os.environ.get('SLURM_LOCALID', 0))

if local_rank == 0:
    print(f'Local rank: {local_rank}')
    node_name = socket.gethostname()
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count()

    print(f'Node: {node_name}')
    print(f'CUDA_VISIBLE_DEVICES: {cuda_visible_devices}')
    print(f'CUDA available: {cuda_available}')
    print(f'CUDA device count: {cuda_device_count}')

    if cuda_available and cuda_device_count > 0:
        gpu_info = []
        for i in range(cuda_device_count):
            device = torch.cuda.get_device_properties(i)
            gpu_info.append(f'GPU {i}: {device.name}, Total memory: {device.total_memory / 1024**3:.2f} GB')
        print('\n'.join(gpu_info))

    print('Multi-node setup is working correctly!')
    print('-------------------------------------')
" "$@"

echo "Job completed successfully!"