import sys
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader 

# Project Imports
import compatibility as compat
from config import Config
from dataset_load import MyDataset
from model_augment import GeometryAugment
from model_ddim import DDIMDiffuser, TransformerDenoiser
from model_sampler import save_sample
from model_trainers import LSAParallel, LSASerial, NoisePredictor, SamplePredictor
from utils import print_config

# Suppress nested tensor warnings
import warnings
warnings.filterwarnings("ignore", message="enable_nested_tensor is True")


def train_fn(rank, config):
    """
    Main training loop.
    Args:
        index: Process index (0 for master).
        config: Instance of io.Config containing parsed settings.
    """
    device = compat.get_device()
    compat.master_print(f"Process {rank} initialized on {device}", rank)

    #--------------------------------------------
    # Setup & Config
    #--------------------------------------------
    if rank == 0:
        compat.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        compat.SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Checkpoints: {compat.CHECKPOINTS_DIR}")
        print(f"Samples:     {compat.SAMPLES_DIR}")

    #--------------------------------------------
    # Load Data
    #--------------------------------------------
    compat.master_print(f"Loading data from {config.data_path}...", rank)
    dataset = MyDataset(Path(config.data_path))  # CPU
    sampler = compat.get_sampler(dataset)        # Split data for TPU cores
    
    loader_args = {
        'batch_size': config.train['batch_size'],
        'sampler': sampler,
        'shuffle': (sampler is None),              # Shuffle only if NOT using a sampler
        'num_workers': 0 if compat.IS_TPU else 4,  # Keep 0 for safety on TPU VMs
        'drop_last': True
    }
    raw_loader = DataLoader(dataset, **loader_args)
    train_loader = compat.get_loader(raw_loader, device)  # Pre-fetch to device

    if rank == 0:
        print(dataset)
        print(f"Batches/Core:  {len(raw_loader)}")

    #--------------------------------------------
    # Model Initialization
    #--------------------------------------------
    denoiser = TransformerDenoiser(**config.denoiser).to(device)
    diffuser = DDIMDiffuser(num_timesteps=1000).to(device)
    augmenter = GeometryAugment().to(device)
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=config.train['lr'])

    # Load weights if a checkpoint was provided in args
    config.load_model_state(denoiser, optimizer)

    # Select Trainer
    Trainers = {
        'Noise': NoisePredictor,
        'Sample': SamplePredictor,
        'LSAS': LSASerial,
        'LSAP': LSAParallel,
    }
    Trainer = Trainers[config.train['trainer']]
    trainer = Trainer(denoiser, diffuser, optimizer, device)
    compat.master_print(f"Trainer: {Trainer.__name__}", rank)
    
    #--------------------------------------------
    # Training Loop
    #--------------------------------------------
    start_epoch = config.resume_epoch
    total_epochs = start_epoch + config.train['num_epochs']
    iterator = range(start_epoch, total_epochs)
    
    compat.master_print(f"Starting training for {len(iterator)} epochs...", rank)

    for epoch in iterator:
        epoch_loss = 0
        count = 0
        
        # Enable progress bar only on master process
        progressbar = tqdm(train_loader, disable=(rank != 0))
        
        for batch in progressbar:
            xya, colors, labels = batch
            xysc = augmenter(xya)
            loss = trainer(xysc, colors, labels)
            epoch_loss += loss
            count += 1
            
            progressbar.set_description(f"Epoch {epoch} | Loss: {loss:.4f}")
        
        # Log Average
        avg_loss = epoch_loss / count if count > 0 else 0
        compat.master_print(f"Epoch {epoch} Done. Avg Loss: {avg_loss:.4f}", rank)

        if rank == 0:
            # Save Checkpoint
            config.save_checkpoint(epoch, denoiser, optimizer, avg_loss, dataset)
            
            # Save Sample
            label_idx = int(config.timestamp[:2])
            name = dataset.class_lookup[label_idx]
            save_name = f"sample_{config.timestamp}_e{epoch:03d}_c{label_idx}_{name}.svg"
            save_path = compat.SAMPLES_DIR / save_name

            # Call the reusable function from model_sampler.py
            save_sample(denoiser, diffuser, device, save_path, 
                        dataset.num_tiles, dataset.symmetry, dataset.side, 
                        label_idx)
#------
# Main
#------
config = Config()
print(config)
compat.launch(train_fn, (config,))