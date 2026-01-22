import random
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

import code.compatibility as compat
from code.compatibility import master_print as mprint
from code.config import Config
from code.data.load import MyDataset
from code.model.augment import GeometryAugment
from code.model.ddim import DDIMDiffuser, TransformerDenoiser
from code.model.loss_helpers import circle_loss, lattice_loss
from code.model.sampler import save_sample
from code.model.losses import get_loss
from code.utils import safe_path
from code.wandblog import WandBLog

# Suppress nested tensor warnings
import warnings
warnings.filterwarnings("ignore", message="enable_nested_tensor is True")


def train_fn(rank:int, config:Config):
    """
    Main training loop.
    Args:
        rank: Process rank/index (0 for master).
        config: Instance of config.Config containing parsed settings.
    """
    device = compat.get_device()
    print(f"Process {rank} initialized on {device}", rank)
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

    # 2. Init Scheduler (Create it NOW, before loading state)
    warmup_epochs = min(10, int(config.train['num_epochs'] * 0.05)) 
    decay_epochs = config.train['num_epochs'] - warmup_epochs

    scheduler1 = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    scheduler2 = CosineAnnealingLR(optimizer, T_max=decay_epochs)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_epochs])

    # 3. Load State (Pass scheduler so it can load its internal counter/state)
    # You will need to update config.load_model_state to accept this argument
    config.load_model_state(denoiser, optimizer, scheduler) 

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

    # We are ready to set an identifier based on timestamp, loss and num_tiles    
    config.set_identifier(lossfunctor.abbr, dataset.num_tiles)
    mprint(f"Identifier for this run: {config.identifier}", rank)

    #--------------------------------------------
    # Initialize WandB Logger
    #--------------------------------------------
    mprint("Initializing WandB...", rank)
    if not config.wandb['run_name']:
        config.wandb['run_name'] = config.identifier

    wandblog = WandBLog(rank, config.wandb)
    wandblog.info(config, compat, dataset, denoiser, lossfunctor)
    mprint("WandB initialized.", rank)

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
        total_loss = 0
        min_loss, max_loss = 1e9, -1e9
        count = 0

        # Enable progress bar only on master process
        progressbar = tqdm(train_loader, disable=(rank != 0))

        for batch_idx, batch in enumerate(progressbar):
            xya, colors, labels = batch
            xysc_hat = augmenter(xya)
            loss = lossfunctor(xysc_hat, colors, labels)
            total_loss += loss
            min_loss = min(min_loss, loss)
            max_loss = max(max_loss, loss)
            count += 1

            progressbar.set_description(f"Epoch {epoch} | Loss: {loss:.4f}")
        scheduler.step()
        avg_loss = total_loss / count if count > 0 else 0
        mprint(f"Epoch {epoch} Done. Avg Loss: {avg_loss:.4f}", rank)

        wandblog.lsepoch_metrics(epoch, {
            'loss_avg': avg_loss,
            'loss_min': min_loss,
            'loss_max': max_loss,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        wandblog.lsgradient_norm(epoch, denoiser)

        if is_master:
            config.save_checkpoint(epoch, denoiser, optimizer, scheduler, avg_loss, dataset, wandblog)

            if config.train['save_samples']:
                # Sample via the diffuser (using the denoiser)
                svg_name = f"sample_{config.timestamp}_e{epoch:03d}_c{sample_label:02d}_{sample_name}.svg"
                svg_path = safe_path(config.samples_dir, svg_name)
                svg, xysc_hat = save_sample(denoiser, diffuser, device, svg_path,
                                    dataset.num_tiles, dataset.symmetry, dataset.side,
                                    sample_label)
                compat.write(svg, svg_path)
                wandblog.lsvg(epoch, svg, sample_label, sample_name)

                # Save some special losses
                latticeloss = lattice_loss(xysc_hat, dataset.side, dataset.symmetry).item()
                circleloss = circle_loss(xysc_hat).item()
                
                wandblog.log_step({'lattice_loss': latticeloss, 'circle_loss': circleloss}, step=epoch)

            print()

    wandblog.finish()
            
#------
# Main
#------
if __name__ == "__main__":
    config = Config()
    print(config)
    compat.launch(train_fn, (config,))
