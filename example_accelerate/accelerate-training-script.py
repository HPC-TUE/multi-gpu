"""
Training strategies:
Gradient accumulation is activated, change the steps with gradient_accumulation_steps
Enabled DeepSpeed
Enabled Zero 1
Enabled Data Parallelism
refer to https://huggingface.co/docs/accelerate/usage_guides/fsdp
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, ProjectConfiguration, GradientAccumulationPlugin


import wandb
import fire
import os
from datetime import datetime

import deepspeed

# Setup logging
logger = get_logger(__name__)

class RandomDataset(Dataset):
    def __init__(self, num_samples=10000, feature_dim=10000):
        super().__init__()
        self.data = torch.randn((num_samples, feature_dim))
        self.labels = torch.randn((num_samples, 1))

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
    def __len__(self):
        return len(self.data)

class MultiLayerNetwork(nn.Module):
    def __init__(self, input_dim=10000, hidden_dims=[5120, 2560], output_dim=1):
        super().__init__()
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.SiLU()) 
            layers.append(nn.Dropout(0.1))  # Add regularization
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def main(
    batch_size=32,
    lr=1e-4,
    num_epochs=5,
    project_name=None,
    entity=None,
    run_name=None,
    seed=42,
    mixed_precision='bf16', # low precision e.g. FP8 refer to https://huggingface.co/docs/accelerate/usage_guides/low_precision_training
    gradient_accumulation_steps=2, # define the number of steps to perform before each call to step()
    dev=False,
    lr_scheduler='cosine',  # Learning rate scheduler type
    resume_from_checkpoint=None,  # Option to resume from a checkpoint
):
    # Validate inputs
    assert project_name is not None, "Error project name not provided, give one and log into WandB using 'wandb login' in the CLI"
    assert entity is not None, "Error entity not provided use your WandB username, give one and log into WandB using 'wandb login' in the CLI"

    # Generate unique run name if not provided
    if run_name is None:
        run_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Set seed for reproducibility
    set_seed(seed)

    # Configure Accelerator with explicit checkpointing
    project_config = ProjectConfiguration(
        project_dir="./checkpoints",
        automatic_checkpoint_naming=True,
        total_limit=5  # Keep last 5 checkpoints
    )
    
    # Optional Config gradient_accumulation_plugin
    gradient_accumulation_plugin = GradientAccumulationPlugin(num_steps=gradient_accumulation_steps, sync_each_batch = True) #  Whether to synchronize setting the gradients at each data batch. Seting to True may reduce memory requirements when using gradient accumulation with distributed training, at expense of speed.

    # Configure Accelerator
    accelerator = Accelerator(
        gradient_accumulation_plugin=gradient_accumulation_plugin,
        mixed_precision=mixed_precision,
        log_with="wandb" if not dev else None,
        project_config=project_config,
        split_batches=True  # Ensure each process gets the same batch size
    )

    # Initialize dataset and splits
    dataset = RandomDataset()
    train_size = int(0.7 * len(dataset))
    test_size = int(0.2 * len(dataset))
    val_size = len(dataset) - train_size - test_size

    train_dataset, test_dataset, val_dataset = random_split(dataset, [train_size, test_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model and optimizer
    model = MultiLayerNetwork()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    # Learning rate scheduler configuration
    if lr_scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif lr_scheduler == 'onecycle':
        scheduler = OneCycleLR(
            optimizer, 
            max_lr=lr, 
            total_steps=num_epochs * len(train_loader)
        )
    else:
        scheduler = None
    
    # Prepare with Accelerator
    model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader
    )
    # Register scheduler for checkpointing if used
    if scheduler:
        accelerator.register_for_checkpointing(scheduler)
    
    # Attempt to load checkpoint if specified
    if resume_from_checkpoint:
        try:
            accelerator.load_state(resume_from_checkpoint)
            logger.info(f"Loaded checkpoint from {resume_from_checkpoint}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
    
    # Enable Deepspeed with data parallelism and Zero-1
    ds_config = {
    "train_batch_size": 4 * 8,  # DP * total processes
    "zero_optimization": {
        "stage": 1,  # ZeRO Stage 1
        "allgather_partitions": True,
        "allgather_bucket_size": 5e8,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        }
    }
    
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    config=ds_config
    )

    # Initialize WandB (if not in dev mode)
    if not dev:
        accelerator.init_trackers(
            project_name,
            config={
                "learning_rate": lr,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "seed": seed,
                "mixed_precision": mixed_precision,
                "lr_scheduler": lr_scheduler
            },
            init_kwargs={"wandb": {"entity": entity, "name": run_name}}
        )

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            with accelerator.accumulate(model):
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                # Step learning rate scheduler if needed
                if lr_scheduler == 'onecycle':
                    scheduler.step()
                
                total_train_loss += loss.detach().float()

                if accelerator.is_main_process and batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")
        
        # Step cosine annealing scheduler after epoch
        if lr_scheduler == 'cosine':
            scheduler.step()
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                val_loss = loss_fn(outputs, targets)
                total_val_loss += val_loss.detach().float()

        # Log metrics
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        # Save checkpoint if best validation loss
        if avg_val_loss < best_val_loss and accelerator.is_main_process:
            best_val_loss = avg_val_loss
            accelerator.save_state()  # Save checkpoint
        
        if not dev:
            accelerator.log({
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "learning_rate": optimizer.param_groups[0]['lr']
            }, step=epoch)
            
    # Final testing
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            test_loss = loss_fn(outputs, targets)
            total_test_loss += test_loss.detach().float()

    avg_test_loss = total_test_loss / len(test_loader)

    if not dev and accelerator.is_main_process:
        logger.info(f"Final Test Loss: {avg_test_loss}")
        accelerator.log({"test_loss": avg_test_loss})

    # Cleanup
    if not dev:
        accelerator.end_training()

if __name__ == "__main__":
    fire.Fire(main)
