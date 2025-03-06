#!/bin/bash
#SBATCH --job-name=test
#SBATCH --time=00:05:00
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
pip install --no-cache-dir -r requirements.txt

# Run your code, using srun fixes the multinode running
srun python ./example_lighting/lightning_example.py --nodes=2 "$@" --entity=AI_team_SCC_TUe --project_name=multi_slurm_test_lighting --run_name=test_slurm_auto --strategy=auto
