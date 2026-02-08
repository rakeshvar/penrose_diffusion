import random
import torch
from pathlib import Path
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import torch.nn.utils as nn_utils

import code.compatibility as compat
from code.config import Config
from code.utils.advanced import get_random_colors, xyac_to_svgs
from code.utils.lossy import lattice_loss
from code.wandblog import WandBLog
from code.data.load import MyDataset
from code.filesystem import CheckPointer, load_checkpoint
from code.compatibility import master_print as mprint

from code.models import get_model_class

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
    config.model['num_tiles'] = dataset.num_tiles
    config.model['num_classes'] = dataset.num_classes
    config.model['side'] = dataset.side
    config.model['symmetry'] = dataset.symmetry
    #--------------------------------------------
    # Model Initialization
    #--------------------------------------------
    Model = get_model_class(config.model['model'])
    model = Model(config.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train['lr'])

    # Scheduler (Create it NOW, before loading state)
    warmup_epochs = min(10, int(config.train['num_epochs'] * 0.05))
    decay_epochs = config.train['num_epochs'] - warmup_epochs
    scheduler1 = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    scheduler2 = CosineAnnealingLR(optimizer, T_max=decay_epochs)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_epochs])

    load_checkpoint(config.checkpoint_path, model, optimizer, scheduler, rank)

    # Move to device
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
            elif not isinstance(v, (int, float, bool, type(None))):
                warnings.warn(f"Unexpected optimizer state type: '{type(v)}' for key '{k}'")


    #--------------------------------------------
    # Set an identifier for the run
    #--------------------------------------------
    identifier = f"{config.timestamp}_{model.descriptor}_t{dataset.num_tiles:03d}"
    mprint(f"Identifier for this run: {identifier}", rank)

    #--------------------------------------------
    # Initialize WandB Logger
    #--------------------------------------------
    mprint("Initializing WandB (Maybe)...", rank)
    if not config.wandb['run_name']:
        config.wandb['run_name'] = identifier

    wandblog = WandBLog(rank, config.wandb)
    wandblog.info(config, compat, dataset, model)

    #--------------------------------------------
    # Initialize Checkpointer
    #--------------------------------------------
    if is_master:
        ckptr = CheckPointer(config.output_base_dir, identifier)
        ckptr.add_fixed_ckpt_data(dataset, config.config, config.data_path_orig, wandblog.get_run_id())

    #--------------------------------------------
    # Sample label and colors (to generate tiles)
    #--------------------------------------------
    sample_label = random.randint(0, dataset.num_classes - 1)
    sample_label_tr = torch.tensor([sample_label], dtype=torch.long, device=device)
    sample_name = dataset.class_lookup[sample_label]
    sample_colors = get_random_colors(dataset.symmetry, 1, dataset.num_tiles, device)

    #--------------------------------------------
    # Training Loop
    #--------------------------------------------
    start_epoch = config.resume_epoch
    total_epochs = start_epoch + config.train['num_epochs']
    iterator = range(start_epoch, total_epochs)
    mprint(f"Starting training for {len(iterator)} epochs...", rank)
    num_aux_losses = len(model.aux_loss_names)

    for epoch in iterator:
        total_loss = 0
        aux_loss_sums = torch.zeros(num_aux_losses, device=device)
        count = 0
        num_nans = 0

        # Enable progress bar only on master process
        progressbar = tqdm(train_loader, disable=(rank != 0))

        for batch in progressbar:
            xya, colors, labels = batch
            loss, aux_losses = model.train_step(xya, colors, labels)

            if torch.isnan(loss):
                num_nans += 1
                return

            # Backpropagate
            optimizer.zero_grad()
            loss.backward()
            compat.optimizer_step(optimizer)

            total_loss += loss.item()
            aux_loss_sums += aux_losses.detach()
            count += 1

            progressbar.set_description(f"Epoch {epoch} | Loss: {loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / count if count > 0 else 0

        to_log = {
            'loss/avg_loss': avg_loss,
            'grad_norm': nn_utils.clip_grad_norm_(model.parameters(), float('inf')),
            'learning_rate': optimizer.param_groups[0]['lr'],
            'num_nans': num_nans/(count + num_nans)
        }
        # Add averaged aux losses
        aux_loss_avgs = (aux_loss_sums / count).cpu().numpy()
        for name, value in zip(model.aux_loss_names, aux_loss_avgs):
            to_log["loss/"+name] = float(value)

        if is_master:
          ckptr.save_checkpoint(epoch, model, optimizer, scheduler, avg_loss) # type: ignore

          if config.train['save_samples']:
            samples = model.sample(sample_colors, sample_label_tr, 50)
            svg = xyac_to_svgs(samples, dataset.symmetry, dataset.side)[0]
            svg_fname = f"sv{config.timestamp}_e{epoch:03d}_{sample_name}.svg"
            ckptr.save_svg(svg, svg_fname)                                    # type: ignore
            wandblog.lsvg(epoch, svg, sample_label, sample_name)
            loss_lattice = lattice_loss(dataset.symmetry, samples, dataset.side)
            to_log['loss/lattice_sample'] = loss_lattice
            mprint(f"Lattice loss: {loss_lattice:.4f}", rank)

        wandblog.log_step(to_log, step=epoch)
        mprint(f"Epoch {epoch} done. Average Loss: {avg_loss:.4f}\n", rank)

    wandblog.finish()
    mprint("\n======\nDone!\n======", rank)


#------
# Main
#------
if __name__ == "__main__":
    config = Config()
    print(config)
    compat.launch(train_fn, (config,))
