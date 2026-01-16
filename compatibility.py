# pyright: reportPossiblyUnboundVariable=false
# pyright: reportMissingImports=false
import os
import sys
from pathlib import Path

try:
    import google.colab
    IS_COLAB = True
except ImportError:
    IS_COLAB = False


import torch
from torch.utils.data.distributed import DistributedSampler

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl
    IS_TPU = True
except ImportError:
    IS_TPU = False

IS_GPU = torch.cuda.is_available()
IS_CPU = not (IS_TPU or IS_GPU)

print("######################")
print("IS_CPU:", IS_CPU)
print("IS_TPU:", IS_TPU)
print("IS_GPU:", IS_GPU)
print("IS_COLAB:", IS_COLAB)
print("######################")


def setup_paths():
    """
    Determines checkpoint and sample directories based on the environment.
    Auto-mounts Google Drive if in Colab.
    """
    if IS_COLAB:
        if not Path('/content/drive').exists():
            print("-" * 60)
            print("ERROR: Google Drive is not mounted.")
            print("Please run the following in a separate code cell BEFORE running this script:")
            print("    from google.colab import drive")
            print("    drive.mount('/content/drive')")
            sys.exit(1)

        base_dir = Path('/content/drive/MyDrive/penrose_diffusion')
    else:
        # Local or standard TPU VM
        base_dir = Path('.')

    c_dir = base_dir / 'checkpoints'
    s_dir = base_dir / 'samples'
    return c_dir, s_dir

CHECKPOINTS_DIR, SAMPLES_DIR = setup_paths()


def get_device():
    if IS_TPU:
        return xm.xla_device()
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def master_print(msg, rank):
    """Helper to print only from the master process."""
    if rank == 0:
        print(msg)


def get_maybe_sampler(dataset):
    """
    Returns a DistributedSampler if on TPU to split data across cores.
    Returns None for standard GPU/CPU training.
    """
    if IS_TPU:
        return DistributedSampler(
            dataset,
            num_replicas=get_world_size(),
            rank=get_ordinal(),
            shuffle=True
        )
    return None


class GpuDeviceLoader:
    """Mimics MpDeviceLoader for GPU/CPU: moves batch to device during iteration."""
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            yield [
                x.to(self.device) if isinstance(x, torch.Tensor) else x
                for x in batch
            ]


def get_loader(dataloader, device):
    """
    Returns an iterator that yields batches pre-moved to the device.
    """
    if IS_TPU:
        # MpDeviceLoader pre-fetches to TPU efficiently
        return pl.MpDeviceLoader(dataloader, device)
    else:
        # Wrapper for GPU/CPU
        return GpuDeviceLoader(dataloader, device)


def optimizer_step(optimizer):
    """Triggers XLA graph execution on TPU or standard step on GPU."""
    if IS_TPU:
        xm.optimizer_step(optimizer)
        xm.mark_step()
    else:
        optimizer.step()


def save_checkpoint(data, path):
    """Ensures only the master process saves to disk."""
    if IS_TPU:
        xm.save(data, path)
    else:
        torch.save(data, path)

def get_world_size():
    if IS_TPU:
        # PJRT (new)
        if hasattr(xm, "world_size"):
            return xm.world_size()
        # XRT (old fallback)
        if hasattr(xm, "xrt_world_size"):
            return xm.xrt_world_size()
        # Single process fallback
    return 1


def get_ordinal():
    if IS_TPU:
        if hasattr(xm, "get_ordinal"):
            return xm.get_ordinal()
        if hasattr(xm, "xrt_global_ordinal"):
            return xm.xrt_global_ordinal()
    return 0

def launch(train_fn, args=()):
    """
    Universal launcher:
    - TPU: Spawns processes (auto-detects count).
    - GPU/CPU: Runs the function directly.
    """
    if IS_TPU:
        nprocs = get_world_size()
        print(f"TPU Detected. Initializing Multi-VM Spawn with {nprocs} processes.")
        xmp.spawn(train_fn, args=args, nprocs=nprocs, start_method='spawn')
    else:
        print(f"Running single process. {'Colab ' if IS_COLAB else ''}")
        train_fn(0, *args)