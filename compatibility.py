# pyright: reportPossiblyUnboundVariable=false
# pyright: reportMissingImports=false
from pathlib import Path

import torch
from torch.utils.data.distributed import DistributedSampler

try:
    from google.colab import drive
    IS_COLAB = True
except ImportError:
    IS_COLAB = False

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl
    IS_TPU = True
except ImportError:
    IS_TPU = False


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
            exit(1)

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
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def master_print(msg, index):
    """Helper to print only from the master process."""
    if index == 0:
        print(msg)


def get_sampler(dataset):
    """
    Returns a DistributedSampler if on TPU to split data across cores.
    Returns None for standard GPU/CPU training.
    """
    if IS_TPU:
        return DistributedSampler(
            dataset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
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
    else:
        optimizer.step()


def save_checkpoint(data, path):
    """Ensures only the master process saves to disk."""
    if IS_TPU:
        xm.save(data, path)
    else:
        torch.save(data, path)


def launch(train_fn, args=()):
    """
    Universal launcher:
    - TPU: Spawns 8 processes.
    - GPU/CPU: Runs the function directly.
    """
    if IS_TPU:
        print("TPU Detected. Spawning 8 processes...")
        xmp.spawn(train_fn, args=args, nprocs=8, start_method='fork')
    else:
        print(f"Running single process. {'Colab ' if IS_COLAB else ''}")
        train_fn(0, *args)