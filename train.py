import random
import torch
from pathlib import Path
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    LinearLR,
    SequentialLR,
)
import torch.nn.utils as nn_utils

import code.compatibility as compat
from code.config import Config
from code.utils.advanced import get_random_colors, xyac_to_svgs
from code.utils.lossy import lattice_loss
from code.wandblog import WandBLog
from code.data.load import MyDataset
from code.filesystem import CheckPointer, load_checkpoint

from code.models import get_model_class

# Suppress nested tensor warnings
import warnings
warnings.filterwarnings("ignore", message="enable_nested_tensor is True")
warnings.filterwarnings("ignore", category=FutureWarning)


def build_lr_scheduler(optimizer, peak_lr, num_epochs, *, min_lr_factor=0.1):
    warmup_epochs = min(10, int(num_epochs * 0.05))
    decay_epochs = num_epochs - warmup_epochs
    scheduler1 = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    scheduler2 = CosineAnnealingLR(
        optimizer,
        T_max=decay_epochs,
        eta_min=min_lr_factor * peak_lr,
    )
    scheduler3 = LambdaLR(optimizer, lr_lambda=lambda _: min_lr_factor)
    return SequentialLR(
        optimizer,
        schedulers=[scheduler1, scheduler2, scheduler3],
        milestones=[warmup_epochs, num_epochs],
    )


def train_fn(rank:int, config:Config):
    """
    Main training loop.
    Args:
        rank: Process rank/index
        config: Instance of config.Config containing parsed settings.
    """
    device = compat.get_device()
    print(f"Process {rank} initialized on {device}.")

    is_master = compat.is_master()
    if is_master:
        print(f"{rank} is master.")
        compat.print_env()
        mprint = print
    else:
        mprint = lambda *args, **kwargs: None

    #--------------------------------------------
    # Load Data
    #--------------------------------------------
    mprint(f"Loading data from {config.data_path}...")
    dataset = MyDataset(Path(config.data_path))         # CPU
    distributed_sampler = compat.get_maybe_distributed_sampler(dataset)   # Split data for TPU cores

    loader_args = {
        'batch_size': config.train['batch_size'],
        'sampler': distributed_sampler,
        'shuffle': distributed_sampler is None,           # distributed_sampler handles shuffling
        'num_workers': 0 if distributed_sampler else 4,   # distributed_sampler handles multi-threading
        'drop_last': True,
        'pin_memory': config.model.get('diffuser') == 'otfm' and device.type == 'cuda',
    }
    data_loader = DataLoader(dataset, **loader_args)
    device_data_loader = compat.get_loader(data_loader, device)  # Pre-fetch to device

    mprint(dataset) # type: ignore
    mprint(f"Batches/Core:  {len(data_loader)}")

    #--------------------------------------------
    # Model Initialization
    #--------------------------------------------
    Model = get_model_class(config.model['model'])
    model = Model(config.model, dataset).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train['lr'])
    model.runtime_setup(optimizer)

    # Scheduler (Create it NOW, before loading state)
    scheduler = build_lr_scheduler(
        optimizer,
        config.train['lr'],
        config.train['num_epochs'],
    )

    load_checkpoint(config.checkpoint_path, model, optimizer, scheduler, mprint)

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
    mprint(f"Identifier for this run: {identifier}")

    #--------------------------------------------
    # Initialize WandB Logger
    #--------------------------------------------
    mprint("Initializing WandB (Maybe)...")
    if not config.wandb['run_name']:
        config.wandb['run_name'] = identifier

    wandblog = WandBLog(is_master, config.wandb)
    wandblog.info(config, compat, dataset, model)

    #--------------------------------------------
    # Initialize Checkpointer
    #--------------------------------------------
    if is_master:
        ckptr = CheckPointer(
            config.output_base_dir,
            identifier,
            keep_best_n=config.train['keep_best_n'],
        )
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
    mprint(f"Starting training for {len(iterator)} epochs...")
    num_aux_losses = len(model.aux_loss_names)

    for epoch in iterator:
        total_loss = 0
        aux_loss_sums = torch.zeros(num_aux_losses, device=device)
        count = 0

        # OTFM owns device prefetch so batch k+1 can match while batch k trains.
        uses_prepared_batches = getattr(model, 'uses_prepared_batches', False)
        if uses_prepared_batches:
            raw_loader = (
                data_loader
                if device.type == 'cuda' and config.model.get('ot_async_prefetch', True)
                else device_data_loader
            )
            training_batches = model.iter_training_batches(raw_loader)
            progressbar = tqdm(
                training_batches,
                total=len(data_loader),
                disable=not is_master,
            )
        else:
            progressbar = tqdm(device_data_loader, disable=not is_master)

        for batch in progressbar:
            try:
                if uses_prepared_batches:
                    loss, aux_losses = model.train_prepared_step(batch)
                else:
                    xya, colors, labels = batch
                    loss, aux_losses = model.train_step(xya, colors, labels)
            except RuntimeError as e:
                print(f"RuntimeError in Epoch {epoch} at Batch {count}. Total Loss: {total_loss:.4f}")
                raise e

            # Backpropagate
            optimizer.zero_grad()
            loss.backward()
            nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.)
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
        }
        if uses_prepared_batches:
            to_log['performance/ot_wait_ms'] = model.ot_mean_wait_ms
            mprint(f"OT prefetch exposed wait: {model.ot_mean_wait_ms:.3f} ms/batch")
        # Add averaged aux losses
        aux_loss_avgs = (aux_loss_sums / count).cpu().numpy()
        for name, value in zip(model.aux_loss_names, aux_loss_avgs):
            to_log["loss/"+name] = float(value)
            mprint(f"{name} = {value:.4f}")

        is_final_epoch = epoch == total_epochs - 1
        if is_master and (epoch % config.train['save_interval'] == 0 or is_final_epoch):
          ckptr.save_checkpoint(
              epoch,
              model,
              optimizer,
              scheduler,
              avg_loss,
              is_final=is_final_epoch,
          ) # type: ignore

          if config.train['save_samples']:
            samples = model.sample(sample_colors, sample_label_tr, 50)
            svg = xyac_to_svgs(samples, dataset.symmetry, dataset.side)[0]
            svg_fname = f"sv{config.timestamp}_e{epoch:03d}_{sample_name}.svg"
            ckptr.save_svg(
                svg,
                svg_fname,
                avg_loss,
                is_final=is_final_epoch,
            ) # type: ignore
            wandblog.lsvg(epoch, svg, sample_label, sample_name)
            try:
                loss_lattice = lattice_loss(dataset.symmetry, samples, dataset.side)
            except NotImplementedError:
                mprint(f"Lattice loss is not implemented for symmetry {dataset.symmetry}; skipping.")
            else:
                to_log['loss/lattice_sample'] = loss_lattice
                mprint(f"Lattice loss: {loss_lattice:.4f}")

        wandblog.log_step(to_log, step=epoch)
        mprint(f"Epoch {epoch} done. Average Loss: {avg_loss:.4f}\n")

    model.runtime_teardown()
    wandblog.finish()
    mprint("\n======\nDone!\n======")


#------
# Main
#------
if __name__ == "__main__":
    config = Config()
    print(config)
    compat.launch(train_fn, (config,))
