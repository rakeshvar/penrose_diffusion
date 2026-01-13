import sys
import yaml
import torch
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Standard Torch Imports
from torch.utils.data import DataLoader 

# Project Imports
import compatibility as compat
from dataset_load import MyDataset
from model_augment import GeometryAugment
from model_ddim import DDIMDiffuser, TransformerDenoiser
from model_trainers import LSAParallel, LSASerial, NoisePredictor, SamplePredictor
from utils import print_config, xysc_to_xyac

# Suppress nested tensor warnings
import warnings
warnings.filterwarnings("ignore", message="enable_nested_tensor is True")


def load_full_config(config_name):
    CONFIG_FILE = 'configs.yaml'
    with open(CONFIG_FILE, 'r') as f:
        all_configs = yaml.safe_load(f)
    if config_name not in all_configs:
        raise ValueError(f"Config '{config_name}' not found.")
    return all_configs[config_name]


def train_fn(index, config_name, data_path_str, loading_from_checkpoint, checkpoint_path):
    device = compat.get_device()
    compat.master_print(f"Process {index} initialized on {device}", index)

    #--------------------------------------------
    # Setup & Config
    #--------------------------------------------
    if index == 0:
        compat.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        compat.SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    if loading_from_checkpoint:
        # Load to CPU to avoid VRAM collisions
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        config = ckpt['config']
        # If data path wasn't provided in args, try to use the one from checkpoint
        if not data_path_str and 'data_path' in ckpt:
            data_path_str = ckpt['data_path']
        compat.master_print("Resumed config from checkpoint.", index)
    else:
        config = load_full_config(config_name)
        ckpt = {}

    denoiser_config = config['denoiser']
    train_config = config['train']

    if index == 0:
        print(f"Checkpoints: {compat.CHECKPOINTS_DIR}")
        print(f"Samples:     {compat.SAMPLES_DIR}")
        print_config(config)

    #--------------------------------------------
    # Load Data
    #--------------------------------------------
    compat.master_print(f"Loading data from {data_path_str}...", index)
    dataset = MyDataset(Path(data_path_str))  # CPU
    sampler = compat.get_sampler(dataset)     # Split data for TPU cores
    
    loader_args = {
        'batch_size': train_config['batch_size'],
        'sampler': sampler,
        'shuffle': (sampler is None),              # Shuffle only if NOT using a sampler
        'num_workers': 0 if compat.IS_TPU else 4,  # Keep 0 for safety on TPU VMs
        'drop_last': True
    }
    raw_loader = DataLoader(dataset, **loader_args)
    train_loader = compat.get_loader(raw_loader, device)  # Pre-fetch to device

    if index == 0:
        print(dataset)
        print(f"Batches/Core:  {len(raw_loader)}")

    #--------------------------------------------
    # Model Initialization
    #--------------------------------------------
    denoiser = TransformerDenoiser(**denoiser_config).to(device)
    diffuser = DDIMDiffuser(num_timesteps=1000).to(device)
    augmenter = GeometryAugment().to(device)
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=train_config['lr'])

    start_epoch = 0
    if loading_from_checkpoint:
        denoiser.load_state_dict(ckpt['denoiser_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        del ckpt 

    Trainers = {
        'Noise': NoisePredictor,
        'Sample': SamplePredictor,
        'LSAS': LSASerial,
        'LSAP': LSAParallel,
    }
    trainer_name = train_config['trainer']
    Trainer = Trainers[trainer_name]
    trainer = Trainer(denoiser, diffuser, optimizer, device)
    
    #--------------------------------------------
    # Training Loop
    #--------------------------------------------
    total_epochs = start_epoch + train_config['num_epochs']
    
    iterator = range(start_epoch, total_epochs)
    compat.master_print(f"Starting training for {len(iterator)} epochs...", index)

    for epoch in iterator:
        epoch_loss = 0
        count = 0
        
        progressbar = tqdm(train_loader, disable=(index != 0)) # Enable tqdm only on master
        
        for batch in progressbar:
            xya, colors, labels = batch
            xysc = augmenter(xya)
            loss = trainer(xysc, colors, labels)
            epoch_loss += loss
            count += 1
            
            progressbar.set_description(f"Epoch {epoch} | Loss: {loss:.4f}")
        
        # Log Average
        avg_loss = epoch_loss / count if count > 0 else 0
        compat.master_print(f"Epoch {epoch} Done. Avg Loss: {avg_loss:.4f}", index)

        # 8. Save Checkpoint (Master Only)
        if index == 0:
            save_name = f"{config_name}_e{epoch:03d}.pt"
            save_path = compat.CHECKPOINTS_DIR / save_name
            
            checkpoint_data = {
                'epoch': epoch,
                'denoiser_state_dict': denoiser.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'loss': avg_loss,
                'data_path': data_path_str,
                'symmetry': dataset.symmetry,
                'num_tiles': dataset.num_tiles,
                'side': dataset.side,
                'num_classes': dataset.num_classes,
                'class_lookup': dataset.class_lookup
            }
            compat.save_checkpoint(checkpoint_data, save_path)
            print(f"Saved checkpoint: {save_path}")

#------
# Main
#------
if len(sys.argv) < 2:
    print(f"Usage: python {sys.argv[0]} <config_name_or_checkpoint> [data_path]")
    sys.exit()

arg1 = sys.argv[1]
loading_from_checkpoint = arg1.endswith('.pt')

data_path_str = ""
checkpoint_path = ""
config_name = "default"

if loading_from_checkpoint:
    checkpoint_path = arg1
    if len(sys.argv) > 2:
        data_path_str = sys.argv[2]
else:
    config_name = arg1
    if len(sys.argv) > 2:
        data_path_str = sys.argv[2]
    else:
        print("Error: Please provide data path for new training.")
        sys.exit()

compat.launch(train_fn, (
    config_name, 
    data_path_str, 
    loading_from_checkpoint, 
    checkpoint_path
    ))