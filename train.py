import random
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

import code.compatibility as compat
from code.compatibility import master_print as mprint
from code.config import Config
from code.data.load import MyDataset
from code.model.augment import GeometryAugment
from code.model.ddim import DDIMDiffuser, TransformerDenoiser
from code.model.sampler import save_sample
from code.model.losses import NoisePredictionLoss, SampleAssistedLoss, SamplePredictionLoss, LSALossSerial, LSALossParallel, get_loss

# Suppress nested tensor warnings
import warnings
warnings.filterwarnings("ignore", message="enable_nested_tensor is True")


def train_fn(rank, config):
    """
    Main training loop.
    Args:
        rank: Process rank/index (0 for master).
        config: Instance of config.Config containing parsed settings.
    """
    device = compat.get_device()
    mprint(f"Process {rank} initialized on {device}", rank)
    compat.print_env(rank)
    is_master = compat.is_master()

    #--------------------------------------------
    # Load Data
    #--------------------------------------------
    mprint(f"Loading data from {config.data_path}...", rank)
    dataset = MyDataset(Path(config.data_path))         # CPU
    distributed_sampler = compat.get_maybe_distributed_sampler(dataset)   # Split data for TPU cores

    loader_args = {
        'batch_size': config.train['batch_size'],
        'sampler': distributed_sampler,
        'shuffle': distributed_sampler is None,           # distributed_sampler handles shuffling
        'num_workers': 0 if distributed_sampler else 4,   # distributed_sampler handles multi-threading
        'drop_last': True
    }
    raw_loader = DataLoader(dataset, **loader_args)
    train_loader = compat.get_loader(raw_loader, device)  # Pre-fetch to device

    mprint(dataset, rank) # type: ignore
    mprint(f"Batches/Core:  {len(raw_loader)}", rank)

    #--------------------------------------------
    # Model Initialization
    #--------------------------------------------
    augmenter = GeometryAugment().to(device)
    diffuser = DDIMDiffuser(num_timesteps=1000).to(device)

    denoiser = TransformerDenoiser(**config.denoiser).to(device)
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=config.train['lr'])
    config.load_model_state(denoiser, optimizer) # if checkpoint

    # Move to device
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
            elif not isinstance(v, (int, float, bool, type(None))):
                warnings.warn(f"Unexpected optimizer state type: '{type(v)}' for key '{k}'")

    # Select Loss Function
    lossfunctor = get_loss(config.train['loss'], 
                      denoiser, diffuser, optimizer, device)
    mprint(f"Loss function: {lossfunctor} ({lossfunctor.abbr})", rank)

    #--------------------------------------------
    # Training Loop
    #--------------------------------------------
    start_epoch = config.resume_epoch
    total_epochs = start_epoch + config.train['num_epochs']
    iterator = range(start_epoch, total_epochs)

    sample_label = random.randint(0, dataset.num_classes - 1)
    sample_name = dataset.class_lookup[sample_label]

    mprint(f"Starting training for {len(iterator)} epochs...", rank)

    for epoch in iterator:
        epoch_loss = 0
        count = 0

        # Enable progress bar only on master process
        progressbar = tqdm(train_loader, disable=(rank != 0))

        for batch in progressbar:
            xya, colors, labels = batch
            xysc = augmenter(xya)
            loss = lossfunctor(xysc, colors, labels)
            epoch_loss += loss
            count += 1

            progressbar.set_description(f"Epoch {epoch} | Loss: {loss:.4f}")

        # Log Average
        avg_loss = epoch_loss / count if count > 0 else 0
        mprint(f"Epoch {epoch} Done. Avg Loss: {avg_loss:.4f}", rank)

        if is_master:
            # Save Checkpoint
            config.save_checkpoint(epoch, denoiser, optimizer, avg_loss, dataset, lossfunctor.abbr)

            # Save Sample
            save_name = f"sample_{config.timestamp}_e{epoch:03d}_c{sample_label:02d}_{sample_name}.svg"
            svg_path = config.samples_dir / save_name

            # Call the reusable function from model_sampler.py
            save_sample(denoiser, diffuser, device, svg_path,
                        dataset.num_tiles, dataset.symmetry, dataset.side,
                        sample_label)
            print()
            
#------
# Main
#------
if __name__ == "__main__":
    config = Config()
    print(config)
    compat.launch(train_fn, (config,))