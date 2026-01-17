# pyright: reportPossiblyUnboundVariable=false
# pyright: reportMissingImports=false

import os
import sys
from pathlib import Path

# -------------------------------------------------
# Environment detection
# -------------------------------------------------

try:
    import google.colab
    IS_COLAB = True
except ImportError:
    IS_COLAB = False

import torch

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.runtime as xr
    IS_TPU = True
except ImportError:
    IS_TPU = False

IS_GPU = torch.cuda.is_available()
IS_CPU = not (IS_TPU or IS_GPU)

# -------------------------------------------------
# Hard PJRT guard
# -------------------------------------------------

if IS_TPU:
    pjrt = os.environ.get("PJRT_DEVICE")
    assert pjrt == "TPU", (
        f"TPU detected but PJRT_DEVICE={pjrt!r}. "
        "This setup requires PJRT."
    )
    IS_PJRT = True

print("######################")
print("IS_CPU:  ", IS_CPU)
print("IS_GPU:  ", IS_GPU)
print("IS_TPU:  ", IS_TPU)
print("IS_PJRT: ", IS_PJRT)
print("IS_COLAB:", IS_COLAB)
print("######################")

# -------------------------------------------------
# Paths
# -------------------------------------------------

def setup_paths():
    """
    Determines checkpoint and sample directories.
    Auto-mounts Google Drive if in Colab.
    """
    if IS_COLAB:
        if not Path("/content/drive").exists():
            print("ERROR: Google Drive not mounted.")
            print("Run:")
            print("  from google.colab import drive")
            print("  drive.mount('/content/drive')")
            sys.exit(1)

        base_dir = Path("/content/drive/MyDrive/penrose_diffusion")
    else:
        base_dir = Path(".")

    return base_dir / "checkpoints", base_dir / "samples"


CHECKPOINTS_DIR, SAMPLES_DIR = setup_paths()

# -------------------------------------------------
# Device helpers
# -------------------------------------------------

def get_device():
    if IS_TPU:
        return xm.xla_device()
    if IS_GPU:
        return torch.device("cuda")
    return torch.device("cpu")


def master_print(msg: str, rank: int):
    if rank == 0:
        print(msg)

# -------------------------------------------------
# Data loading
# -------------------------------------------------

from torch.utils.data.distributed import DistributedSampler

def get_maybe_sampler(dataset):
    if not IS_TPU:
        return None

    return DistributedSampler(
        dataset,
        num_replicas=xr.world_size(),
        rank=xr.global_ordinal(),
        shuffle=True,
    )


class GpuDeviceLoader:
    """GPU / CPU equivalent of MpDeviceLoader."""
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
    if IS_TPU:
        return pl.MpDeviceLoader(dataloader, device)
    return GpuDeviceLoader(dataloader, device)

# -------------------------------------------------
# Optimizer / checkpoint helpers
# -------------------------------------------------

def optimizer_step(optimizer):
    if IS_TPU:
        xm.optimizer_step(optimizer)
        xm.mark_step()
    else:
        optimizer.step()


def save_checkpoint(data, path):
    if IS_TPU:
        xm.save(data, path)
    else:
        torch.save(data, path)

# -------------------------------------------------
# Launch
# -------------------------------------------------

def launch(train_fn, args=()):
    """
    Universal launcher:
    - TPU: PJRT spawn (process count decided by XLA)
    - GPU/CPU: single process
    """
    if IS_TPU:
        print("TPU detected. Launching via PJRT.")
        xmp.spawn(train_fn, args=args)
    else:
        print("Running single-process.")
        train_fn(0, *args)
