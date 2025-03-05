Couldnt have figured this out without the gist of TengdaHan:  
https://gist.github.com/TengdaHan/1dd10d335c7ca6f13810fff41e809904

Strategies currently implemented
- Pytorch Lightning:
    - [x] DDP - NCCL
    - [x] FSDP - NCCL
    - [x] Deepspeed - NCCL
    - [ ] FairScale - ?
    - [ ] Horovod - ?
- Torch:
    - [x] DDP - NCCL
    - [x] FSDP - NCCL
    - [x] Deepspeed - NCCL
    - [x] FairScale - NCCL
    - [ ] Horovod - GLOO

Further impl look at:  
https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/index.html