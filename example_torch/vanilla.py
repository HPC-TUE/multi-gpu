import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, random_split

import fire, random, wandb, numpy, os, natsort, glob, tqdm
from collections import defaultdict

class RndDataset(Dataset):
    # Simple random dataset, replace with your own DS
    def __init__(self):
        super().__init__()

    def __getitem__(self, index):
        return torch.randn((10000,)), torch.randn((1,))
    
    def __len__(self):
        return 10000

class Model(nn.Module):
    def __init__(self, features = 10000):
        super().__init__()
        self.linear = nn.Linear(features,5120)
        self.linear2 = nn.Linear(5120,2560)
        self.linear3 = nn.Linear(2560,1)

    def forward(self, x):
        x = self.linear(x)
        x = nn.functional.sigmoid(x)
        x = self.linear2(x)
        x = nn.functional.sigmoid(x)
        x = self.linear3(x)
        x = nn.functional.sigmoid(x)
        return x

class Trainer():
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        max_epochs: int,
        accumulate_gradient_batches: int | None = None
    ):
        self.max_epochs = max_epochs
        self.accumulate_gradient_batches = accumulate_gradient_batches
        
        self.optimizer = optimizer
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model = model.to(self.local_rank)
        self.model = DDP(self.model, device_ids=[self.local_rank])

    def train_step(self, batch):
        x, y = batch
        y_ = self.model(x)
        loss = nn.functional.mse_loss(y_, y)
        return loss
    
    @torch.no_grad
    def val_step(self, batch):
        x, y = batch
        y_ = self.model(x)
        loss = nn.functional.mse_loss(y_, y)
        return loss
    
    @torch.no_grad
    def test_step(self, batch):
        x, y = batch
        y_ = self.model(x)
        loss = nn.functional.mse_loss(y_, y)
        return loss

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader | None = None):
        for epoch in range(self.max_epochs):
            self.epoch = epoch

            # Train loop
            if self.local_rank == 0:
                pbar = tqdm.tqdm(total=len(train_dataloader))
            self.optimizer.zero_grad()
            for batch_idx, batch in enumerate(train_dataloader):
                batch = (x.to(self.local_rank) for x in batch)
                loss = self.train_step(batch)
                if self.accumulate_gradient_batches is not None:
                    loss = loss / self.accumulate_gradient_batches
                loss.backward()

                if self.accumulate_gradient_batches is not None:
                    if (batch_idx + 1) % self.accumulate_gradient_batches == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                else:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if self.local_rank == 0:
                    wandb.log({
                        'loss': loss.item()
                    })
                    pbar.set_postfix({
                        "loss": loss.item()
                    })
                    pbar.update(1)
                    
            if self.local_rank == 0:
                pbar.close()

            # Validation loop
            if isinstance(val_dataloader, DataLoader):
                if self.local_rank == 0:
                    pbar = tqdm.tqdm(total=len(val_dataloader))
                for batch in val_dataloader:
                    batch = (x.to(self.local_rank) for x in batch)
                    loss = self.val_step(batch)

                    if self.local_rank == 0:
                        wandb.log({
                            "loss": loss.item()
                        })
                        pbar.set_postfix({
                            "loss": loss.item()
                        })
                        pbar.update(1)

                if self.local_rank == 0:
                    pbar.close()

    def test(self, test_dataloader: DataLoader):
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(test_dataloader))
        for batch in test_dataloader:
            batch = (x.to(self.local_rank) for x in batch)
            loss = self.test_step(batch)

            if self.local_rank == 0:
                wandb.log({
                    "loss": loss.item()
                })
                pbar.set_postfix({
                    "loss": loss.item()
                })
                pbar.update(1)

        if self.local_rank == 0:
            pbar.close()

def main(
    batch_size = 32,
    lr = 1e-4,

    strategy = 'auto', # use DDP

    project_name = None,
    entity = None,
    run_name = None,

    resume = None,
    dev = False, # If dev = True it will not connect to WandB (best for development)

    seed = 42
):
    assert project_name is not None, "Error project name not provided, give one and log into WandB using 'wandb login' in the CLI"
    assert entity is not None, "Error entity not provided use your WandB username, give one and log into WandB using 'wandb login' in the CLI"
    assert strategy in ['FSDP', 'fsdp', 'deepspeed', 'auto'], "Error, incorrect strategy, please use FSDP, fsdp, auto or deepspeed"
    assert run_name is not None and not resume, "Error, run_name cannot be none when resuming"

    print(project_name, entity, strategy)

    # Set the seed
    torch.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)

    # We do this because:
    # You are using a CUDA device ('NVIDIA A100-SXM4-40GB') that has Tensor Cores. To properly utilize them, you should set torch.set_float32_matmul_precision('medium' | 'high') which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    torch.set_float32_matmul_precision('high') # Do medium for lower precision (not perse nescessary to do this step)

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend='nccl')

    # init WandB
    wandb.init(
        project=project_name, 
        entity=entity, 
        name=run_name, 
        config=locals(), 
        mode='disabled' if dev else 'online', 
        resume = 'allow' if isinstance(resume, str) else False,
        id = resume
    )

    # Fix run name endings, in case resuming
    if resume:
        run_name += f'_{resume}'
    elif run_name is not None:
        run_name += f'_{wandb.run.id}' if not dev and isinstance(wandb.run.id, str) else ''
    else:
        run_name = wandb.run.name

    # Create ckpt folder
    os.makedirs(f'./checkpoints', exist_ok=True)

    dataset = RndDataset()
    train_size = int(0.7 * len(dataset))  # 70% for training
    test_size = int(0.2 * len(dataset))   # 20% for testing
    val_size = len(dataset) - train_size - test_size  # Remaining 10% for validation

    # Perform random split
    train_dataset, test_dataset, val_dataset = random_split(dataset, [train_size, test_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = Model(features = 10000)
    wandb.watch(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    trainer = Trainer(
        model,
        optimizer,
        max_epochs=250
    )

    trainer.train(
        train_loader,
        val_loader
    )

    # Important cleanup step
    dist.destroy_process_group()


if __name__ == "__main__":
    fire.Fire(main) # Fire automates argparse for CLI arguments (I like this package a lot).