#!/bin/bash
#SBATCH --job-name=test
#SBATCH --time=1:00:00
#SBATCH --partition=gpu_h100 # Run on the h100 (see a100.sh for other way of running)
#SBATCH --nodes=2 # Use 2 nodes
#SBATCH --gpus-per-node=4 # Tell the scheduler to use 4 gpus per node, gres parameter is depricated on Snellius for this
#SBATCH --tasks-per-node=4 # Tell the scheduler that there will be 4 gpus and thus 4 tasks per node
#SBATCH --output=./h100_LIT/%x_%A_%a.out

# Load modules, in the future you may replace these with more up-to-date versions of CUDA
module load 2023
module load foss/2023a
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.4.0

export TRITON_CACHE_DIR='./.tritone' # Change the Triton dir for Deepspeed
export NCCL_SOCKET_IFNAME="eno2np0" # Tell Snellius to use traditional networking, related to infiniband issues: https://servicedesk.surf.nl/wiki/display/WIKI/Snellius+known+issues#Snelliusknownissues-UsingNCCLforGPU%3C=%3EGPUcommunication

# Activate your environment
source ./venv/bin/activate

# Run your code, using srun fixes the multinode running
srun python ./lightning.py --nodes=2 "$@"
