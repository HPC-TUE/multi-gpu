#!/bin/bash
#SBATCH --job-name=test
#SBATCH --time=1:00:00
#SBATCH --partition=gpu_a100
#SBATCH -N 2
#SBATCH --gpus-per-node=4
#SBATCH --tasks-per-node=4
#SBATCH --output=./multi_node/%x_%A_%a.out

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

module load CUDA/12.4.0

. /etc/bashrc
. ~/.bashrc

source ./venv/bin/activate

export NCCL_SOCKET_IFNAME="eno1np0" # Tell Snellius to use traditional networking, related to infiniband issues: https://servicedesk.surf.nl/wiki/display/WIKI/Snellius+known+issues#Snelliusknownissues-UsingNCCLforGPU%3C=%3EGPUcommunication

echo $$

srun python -c "import os, torch; print(os.environ.get(\"CUDA_VISIBLE_DEVICES\"), torch.cuda.is_available(), torch.cuda.device_count())"