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
from code.filesystem import CheckPointer, load_model_state
from code.model.augment import GeometryAugment
from code.model.ddim import DDIMDiffuser, TransformerDenoiser
from code.model.loss_helpers import circle_loss, lattice_loss, equal_angle_loss_var, equal_angle_loss_circular
from code.model.sampler import save_sample
from code.model.losses import get_loss
from code.utils import pairwise_compare
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
    print(f"Process {rank} initialized on {device}")
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

    load_model_state(config.checkpoint_path, denoiser, optimizer, scheduler, rank)

    # Move to device
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
            elif not isinstance(v, (int, float, bool, type(None))):
                warnings.warn(f"Unexpected optimizer state type: '{type(v)}' for key '{k}'")

    # Select Loss Function
    lossfunctor = get_loss(config.train['loss'], denoiser, diffuser, optimizer, device)
    mprint(f"Loss function: {lossfunctor} ({lossfunctor.abbr})", rank)

    #--------------------------------------------
    # Set an identifier for the run
    #--------------------------------------------
    identifier = f"{config.timestamp}_d{config.denoiser['d_model']}_{lossfunctor.abbr}_t{dataset.num_tiles:03d}"
    mprint(f"Identifier for this run: {identifier}", rank)

    #--------------------------------------------
    # Initialize WandB Logger
    #--------------------------------------------
    mprint("Initializing WandB (Maybe)...", rank)
    if not config.wandb['run_name']:
        config.wandb['run_name'] = identifier

    wandblog = WandBLog(rank, config.wandb)
    wandblog.info(config, compat, dataset, denoiser, lossfunctor)

    #--------------------------------------------
    # Initialize Checkpointer
    #--------------------------------------------
    if is_master:
        ckptr = CheckPointer(config.output_base_dir, identifier)
        ckptr.add_fixed_ckpt_data(dataset, config.config, config.data_path_orig, wandblog.get_run_id())

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
        count = 0

        # Enable progress bar only on master process
        progressbar = tqdm(train_loader, disable=(rank != 0))

        for batch in progressbar:
            xya, colors, labels = batch
            xysc_hat = augmenter(xya)
            loss = lossfunctor(xysc_hat, colors, labels)
            total_loss += loss
            count += 1

            progressbar.set_description(f"Epoch {epoch} | Loss: {loss:.4f}")
        
        scheduler.step()
        avg_loss = total_loss / count if count > 0 else 0

        to_log = {
            'loss': avg_loss,
            'grad_norm': lsgradient_norm(denoiser),
            'learning_rate': optimizer.param_groups[0]['lr']
        }

        if is_master:
            ckptr.save_checkpoint(epoch, denoiser, optimizer, scheduler, avg_loss) # type: ignore

            if config.train['save_samples']:
                # Sample via the diffuser (using the denoiser)
                svg, xysc_hat = save_sample(denoiser, diffuser, device, None,
                    dataset.num_tiles, dataset.symmetry, dataset.side, sample_label)
                
                svg_fname = f"sv{config.timestamp}_e{epoch:03d}_{sample_name}.svg"
                ckptr.save_svg(svg, svg_fname)                                  # type: ignore
                wandblog.lsvg(epoch, svg, sample_label, sample_name)

                # Save some special losses
                latticeloss = lattice_loss(xysc_hat, dataset.side, dataset.symmetry).item()
                circleloss = circle_loss(xysc_hat).item()
                ealoss_var = equal_angle_loss_var(xysc_hat).item()
                ealoss_cir = equal_angle_loss_circular(xysc_hat).item()
                to_log.update({'lattice_loss': latticeloss, 'circle_loss': circleloss, 
                               'equal_angle_loss_var': ealoss_var, 'equal_angle_loss_circular': ealoss_cir})
                pairwise_compare([latticeloss, circleloss, ealoss_var, ealoss_cir], ["lattice", "circle", "eal_var", "eal_cir"], f"Losses")

        wandblog.log_step(to_log, step=epoch)
        mprint(f"Epoch {epoch} done. Average Loss: {avg_loss:.4f}\n", rank)

    wandblog.finish()
    mprint("\n======\nDone!\n======", rank)


def lsgradient_norm(denoiser):
    total_sq_norm = 0.0
    for p in denoiser.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_sq_norm += param_norm.item() ** 2
    return total_sq_norm ** 0.5

#------
# Main
#------
if __name__ == "__main__":
    config = Config()
    print(config)
    compat.launch(train_fn, (config,))
