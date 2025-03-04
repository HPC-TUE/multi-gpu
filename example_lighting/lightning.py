import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, random_split

import example_lighting.lightning as L
from lightning.pytorch.loggers import WandbLogger
from example_lighting.lightning import Callback
from lightning.pytorch.callbacks import ModelCheckpoint, GradientAccumulationScheduler, LearningRateMonitor
from lightning.pytorch.strategies import DeepSpeedStrategy, FSDPStrategy
from lightning.pytorch.plugins.precision import DeepSpeedPrecision
from deepspeed.ops.adam import DeepSpeedCPUAdam

import fire, random, wandb, numpy, os, natsort, glob
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

class LitModel(L.LightningModule):
    def __init__(self, lr, strategy = None, model = None, model_args = None):
        super().__init__()
        assert isinstance(model, Model) or model is Model, "Error, not given an instanced model or a reference."
        assert (model_args is not None and (model is Model and not isinstance(model, Model))) or (model_args is None and isinstance(model, Model)), "Error, no model_args given"

        self.lr = lr
        self.strategy = strategy
        self.model = model # Store the model or store none and define
        self.model_args = model_args
        self.val_log_dict = defaultdict(float)
        self.test_log_dict = defaultdict(float)
        self.save_hyperparameters(ignore=['model']) # Lightning thing, no need to also save the self.model bc it is saved in the ckpt

    def forawrd(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_ = self.model(x)

        loss = nn.functional.mse_loss(y_, y)
        
        self.log_dict(
            {
                'losses/loss': loss.item(),
            }, 
            sync_dist = True # important without multiple logging can occur
        )

        return loss
    
    def validation_step(self, batch, batch_idx):
        # We will sum over the validation number.
        x, y = batch
        y_ = self.model(x)

        loss = nn.functional.mse_loss(y_, y)

        self.val_log_dict['loss'] += loss.item()
        self.val_log_dict['steps'] += 1
    
    def on_validation_epoch_end(self):
        # Then calculate the mean validation loss of the set, this creates nicer and more informative graphs (seeing val loss per epoch).
        self.log_dict(
            {
                'losses/val_loss': self.val_log_dict['loss'] / self.val_log_dict['steps'],
            }, 
            sync_dist = True # important without multiple logging can occur
        )

        self.val_log_dict = defaultdict(float)

    # Do the same for the test data.
    def test_step(self, batch, batch_idx):
        # We will sum over the test number.
        x, y = batch
        y_ = self.model(x)

        loss = nn.functional.mse_loss(y_, y)

        self.test_log_dict['loss'] += loss.item()
        self.test_log_dict['steps'] += 1
    
    def on_test_epoch_end(self):
        # Then calculate the mean test loss of the set, this creates nicer and more informative graphs (seeing val loss per epoch).
        self.log_dict(
            {
                'losses/test_loss': self.test_log_dict['loss'] / self.test_log_dict['steps'],
            }, 
            sync_dist = True # important without multiple logging can occur
        )

        self.test_log_dict = defaultdict(float) 

    def configure_model(self):
        if self.model is Model and not isinstance(self.model, Model):
            self.model = self.model(**self.model_args)

    def configure_optimizers(self):
        if self.strategy == 'deepspeed':
            optimizer = DeepSpeedCPUAdam(lr=self.lr, model_params=self.parameters())
        else:
            optimizer = torch.optim.Adam(lr=self.lr, params=self.parameters(), foreach=False if self.strategy in ['fsdp', 'FSDP'] else None) # In FSDP DO NOT use foreach, FSDP shards the model across GPUs and foreach can cause problems because it clusters the weights too much.
        
        # May define your scheduler here
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: min(x, 1.0, 1.0)))

        # May define weight initialization here as well. The code below is not standard code
        # for module in self.model.modules(): 
        #     if isinstance(module, torch.nn.Linear):  # Check if it's a Linear layer
        #         truncated_normal_(module.weight, std=0.2 / (2 * self.model_params[0]) ** 0.5) * 0.5

        return optimizer
    
        # or if you have defined a scheduler:
        # return (
        #     [optimizer], 
        #     [{
        #         'scheduler': scheduler,
        #         'interval': 'step',
        #         'frequency': 1,
        #     }]
        # )

def main(
    batch_size = 32,
    lr = 1e-4,

    strategy = 'auto', # use DDP
    nodes = 1, # Number of SLURM nodes
    device_count = 0,

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

    # Save the name we overwrite strategy later
    strategy_str = strategy
    
    # init WandB
    wandb_logger = WandbLogger(
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
        run_name += f'_{wandb_logger.experiment.id}' if not dev and isinstance(wandb_logger.experiment.id, str) else ''
    else:
        run_name = wandb_logger.experiment.name

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

    callbacks = []

    # You can do some more fun stuff with the config if needed, zero-stage-2 is the best for training stage-3 for inferencing
    if strategy_str == 'deepspeed':
        strategy = DeepSpeedStrategy(
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            offload_optimizer=True, # Faster than recollating to a single GPU with Deepspeed =
            offload_optimizer_device='cpu', # Hence we declared the custom Adam Deepspeed above
            # precision_plugin=DeepSpeedPrecision('bf16-mixed') # In case you want to use AMP

        )  
    # The way to customize the FSDP strat, I like FULL_SHARD the most (slowest but saves the most mem).
    elif strategy_str in ['FSDP', 'fsdp']: 
        strategy = FSDPStrategy(
            accelerator='cuda' if torch.cuda.is_available() else 'cpu',
            sharding_strategy='FULL_SHARD', # There are different strategies but FULL_SHARD is nice and works well
            cpu_offload=False, # Offload weight to the CPU for optimizer step
            auto_wrap_policy=None, # Other trick for efficiency, I have not noticed many benefits
            state_dict_type="full", # The way of saving, keep it full if you want to keep it in a single dict (otherwise it shards).
            # For some networks you can benefit from this, skipping recalculation of backwards gradients during activations (Like Transformers).
            # activation_checkpointing_policy= {
            #     TransformerBlock # Just import the torch module above in the code and mention it here as an uninstanced class
            # }
        )

    if not dev:
        callbacks.append(
            ModelCheckpoint(
                monitor='losses/val_loss', 
                dirpath=f'./checkpoints/{run_name}/',
                save_last=True,
                filename='epoch{epoch:02d}-step{step:04d}',
                save_top_k=1,
                every_n_epochs=1
            )
        )

    print(torch.cuda.device_count())

    trainer = L.Trainer(
        default_root_dir=f'./checkpoints/{run_name}/'  if not dev else None,
        strategy=strategy,
        accelerator='auto' if strategy_str in ['deepspeed', 'fsdp', 'FSDP'] else 'cuda' if torch.cuda.is_available() else 'cpu',
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        num_nodes=nodes,
        callbacks=callbacks,
        enable_checkpointing=(not dev),
        logger=[wandb_logger] if not dev else None,
        log_every_n_steps=1,
        max_epochs=2
        # accumulate_grad_batches=1, # Useful if you cannot fit a full batch in memory! Will pass multiple smaller batches and sum them after n steps. In case you want a batch size of 64 and only 32 fits we set this to 2, aka 2*32 = 64
    )

    if strategy_str == 'auto':
        with trainer.init_module():
            model = Model(features = 10000)
            model = LitModel(lr, strategy = strategy_str, model = model)
    else:
        # We init the model inside the configure_optimizers because: https://lightning.ai/docs/pytorch/stable/advanced/model_init.html#model-parallel-training-fsdp-and-deepspeed
        model = Model
        model_args = {'features': 10000}
        model = LitModel(lr, strategy = strategy_str, model = model, model_args = model_args)
        model.configure_model()
    model.configure_optimizers()

    wandb_logger.watch(model)

    trainer.fit(
        model=model, 
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=natsort.natsorted(glob.glob(f'./checkpoints/{run_name}/*'))[-1] if resume is not None else None # Continue from ckpt
    )

    trainer.test(
        model=model, 
        dataloaders=test_loader,
        ckpt_path=natsort.natsorted(glob.glob(f'./checkpoints/{run_name}/*'))[-1] # Take the latest ckpt
    )

if __name__ == "__main__":
    fire.Fire(main) # Fire automates argparse for CLI arguments (I like this package a lot).