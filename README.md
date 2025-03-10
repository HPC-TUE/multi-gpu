
## Introduction
Welcome to the example repository on how to run AI code in a distributed multi-node setting, specifically for the super computer [Snellius](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660184/Snellius). This super computer employs SLURM as its scheduling system and is widely used across The Netherlands for executing different types of jobs. This repository aims to make it more insightful on how to use the job scheduler with multiple nodes in an AI setting. 

#### SLURM scheduling
Snellius uses a [SLURM](https://slurm.schedmd.com/documentation.html) scheduler like many HPC, for basic setup information on Snellius click here.  
Running in multi-node on Snellius especially for this code as of 2025 we recommend loading modules similar to:  
```
module load 2023
module load foss/2023a
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.4.0
```

These modules tell your environment to use Python 3.11 and CUDA 12.4.0. foss/2023a is required to compile certain pacakge dependencies, in your case newer packages may be available and should be tested before use. In case this is the case please create a pull request with updated code.  

The implementation for the schedulers can be found in the respective .sh files of the different versions.

Currently Snellius requires you to set the specific network device to be able to use NCCL, for the h100 use eno2np0 and for the other cards use eno1np0:
```
if [[ "$SLURM_JOB_PARTITION" == *h100* ]]; then
    export NCCL_SOCKET_IFNAME="eno2np0"
else
    export NCCL_SOCKET_IFNAME="eno1np0"
fi 
```

#### PyTorch Code
All the implementations of the PyTorch code can be found under vanilla.py with different strategies as well as the boilerplate examples of using torch distributions, this version of code can be cumbersum to implement, but it works.

#### PyTorch Lightning
In extension to this we also allow show the use of PyTorch Lightning, this library requires some time to properly migrate to but we notice that it smoothens the use of multiple nodes in the long run. Mainly the amount of bugs and errors thrown by improper implementation of code in the vanilla PyTorch version is reduced by using Lightning.

## Features
The strategies that are currently implemented are:
- PyTorch Lightning:
    - [x] DDP - NCCL
    - [x] FSDP - NCCL
    - [x] Deepspeed - NCCL
    - [ ] FairScale - ?
    - [ ] Horovod - ?
- PyTorch:
    - [x] DDP - NCCL
    - [x] FSDP - NCCL
    - [x] Deepspeed - NCCL
    - [x] FairScale - NCCL
    - [ ] Horovod - ?

## Installation
Installation of these libraries are as straighforward as running the following steps:
```
module load 2023
module load foss/2023a
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.4.0
```

Then define a venv and activate it:
```
python -m venv venv
source ./venv/bin/activate
```

Afterwards install the required packages:
```
pip install -r requirements.txt
```

## Usage
Simply submit your bash jobs using sbatch. If you want you can debug the code using salloc by spawning a two node run such as:
```
salloc --partition=gpu_a100 --gres=gpu:4 --nodes=2 --gpus-per-node=4 --time=1:00:00 --tasks-per-node=4
```

This command gives you 2 nodes with 4 A100 GPUs each. After being allocated the space on the server you can run your nodes by executing the rest of the bash scripts once (the exports etc.). Then run the python scripts using srun to let python know you have access to multiple nodes.

### The bash scripts
Under lightning.sh you can find the code to run the PyTorch Lightning version and under vanilla.sh you can find the PyTorch version. Lets analyze the different bash scripts:

#### The headers
Both scripts use the same header:
```
#!/bin/bash
#SBATCH --job-name=test
#SBATCH --time=1:00:00
#SBATCH --partition=gpu_a100
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --tasks-per-node=4
#SBATCH --output=./a100_LIT/%x_%A_%a.out
```
Most notable differences to make multiple nodes work under Snellius are the use of the  ```--gpus-per-node``` and the ```--tasks-per-node``` flags. Here you tell the scheduler to use 4 GPUs per node and each node will have 4 tasks, one per GPU. This is important otherwise MPI can not tell Python how to distribute the across the GPUs and nodes.

#### The Modules
Furthermore we load the modules in both instances in the same way:
```
module load 2023
module load foss/2023a
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.4.0
```

We require at least Python 3.11 and the latest possible CUDA. Furthermore we load the foss package, this package defines the buildscripts and dependencies for GCC some of the installable packages require this.

#### NCCL config
To make sure the code properly executes on Snellius we need to tell NCCL how to behave. We do this according to:
```
export NCCL_DEBUG=WARN
if [[ "$SLURM_JOB_PARTITION" == *h100* ]]; then
    export NCCL_SOCKET_IFNAME="eno2np0"
else
    export NCCL_SOCKET_IFNAME="eno1np0"
fi 
```
We tell NCCL to warn us in case there are any huge issues and we tell it which Networking device to use when communicating between the nodes. [Infiniband is broken on Snellius](https://servicedesk.surf.nl/wiki/display/WIKI/Snellius+known+issues#Snelliusknownissues-UsingNCCLforGPU%3C=%3EGPUcommunication) hence we have this if else statement.

Additionally we tell tritone where to cache its files, thisis important due to writing limitations in the default cache dir.
```
export TRITON_CACHE_DIR='./.tritone'
```
#### PyTorch Multi-Node
Additionally there are some tricks to running multi-node PyTorch. First of the requirement to tell Torch about the multi-node setup:  
```
export MASTER_PORT=12340
export WORLD_SIZE=$SLURM_NTASKS
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
```

The MASTER_PORT is the port to run the nodes on, the WORLD_SIZE is the total number of available GPUs and the MASTER_ADDR is the ip adress of the headnode.


#### CLI arguments
The following CLI arguments are available:
- ```--batch_size: int = 32```
- ```--lr: float = 1e-4```
- ```--strategy: str = 'auto'```
- ```--nodes: int = 2```
- ```--project_name: str = [project_name]```
- ```--entity: str = [username | team_name]```
- ```--run_name: str = [run_name]```
- ```--resume: bool = False```
- ```--dev: bool = False```
- ```--seed: int = 42```



#### Executing the script example:
```
sbatch ./lightning.sh --entity=[username | team_name] --project_name=[project_name] --run_name=[run_name] --strategy=auto
```

### Dependencies
#### Fire
For argument parsing we use the fire package, this makes life a little bit easier. All the arguments of the main command are also cli commands for the python file.

#### WandB
For logging the scripts use [wandb](https://www.wandb.ai), if you dont want to use this library feel free to remove it. The use of WandB is industry standard, but requires you to create an account. Do add these arguments when using sbatch to login properly:
```
--entity=[username | team_name] --project_name=[project_name] --run_name=[run_name]
```

## Issues
- [Infiniband is broken on Snellius](https://servicedesk.surf.nl/wiki/display/WIKI/Snellius+known+issues#Snelliusknownissues-UsingNCCLforGPU%3C=%3EGPUcommunication)
- srun with single node-multi gpu runs doesnt work, run the code without srun!

## Acknowledgements
I could not have figured proper implementations out without the [gist of TengdaHan](https://gist.github.com/TengdaHan/1dd10d335c7ca6f13810fff41e809904
).