# pyright: reportPossiblyUnboundVariable=false
# pyright: reportMissingImports=false
import os
import sys
import socket

from  contextlib import contextmanager
from pathlib import Path

import torch

# -------------------------------------------------
# Environment detection
# -------------------------------------------------
# Colab
#-------
try:
    import google.colab
    IS_COLAB = True
except ImportError:
    IS_COLAB = False

# GCP
#------
try:
    socket.gethostbyname("metadata.google.internal")
    IS_GCP = True
except socket.gaierror:
    IS_GCP = False

# TPU
#------
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as xpl
    import torch_xla.runtime as xrt
    IS_TPU = True
except ImportError:
    IS_TPU = False

# GPU
#------
IS_GPU = torch.cuda.is_available()

# CPU
#------
IS_CPU = not (IS_TPU or IS_GPU)

# -------------------------------------------------
# PJRT check
# -------------------------------------------------

if IS_TPU:
    pjrt = os.environ.get("PJRT_DEVICE")
    assert pjrt == "TPU", (
        f"TPU detected but PJRT_DEVICE={pjrt!r}. "
        "This setup requires PJRT."
    )

# -------------------------------------------------
# Default Base Directory for Checkpoints and Samples
# -------------------------------------------------

if IS_COLAB:
    if not Path("/content/drive").exists():
        print("ERROR: Google Drive not mounted.")
        print("Run:")
        print("  from google.colab import drive")
        print("  drive.mount('/content/drive')")
        sys.exit(1)
    OUTPUT_BASE_DIR = "/content/drive/MyDrive/penrose_diffusion"

elif IS_GCP:
    OUTPUT_BASE_DIR = "gs://penrose_diffusion/"

else:
    OUTPUT_BASE_DIR = "."

# -------------------------------------------------
# Device helpers
# -------------------------------------------------

def print_env():
    print("-------------------")
    print("IS_CPU:  ", IS_CPU)
    print("IS_GPU:  ", IS_GPU)
    print("IS_TPU:  ", IS_TPU)
    print("IS_GCP:  ", IS_GCP)
    print("IS_COLAB:", IS_COLAB)
    print("-------------------")

def get_device():
    if IS_TPU:
        return xm.xla_device()
    if IS_GPU:
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def is_master():
    return (not IS_TPU) or xm.is_master_ordinal()

# -------------------------------------------------
# Data loading
# -------------------------------------------------

from torch.utils.data.distributed import DistributedSampler

def get_maybe_distributed_sampler(dataset):
    if IS_TPU:
        return DistributedSampler(
            dataset,
            num_replicas=xrt.world_size(),
            rank=xrt.global_ordinal(),
            shuffle=True,
        )
    else:
        return None


class MyDeviceLoader:
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
        return xpl.MpDeviceLoader(dataloader, device)
    else:
        return MyDeviceLoader(dataloader, device)

# -------------------------------------------------
# Optimizer / mark step
# -------------------------------------------------

def optimizer_step(optimizer):
    if IS_TPU:
        xm.optimizer_step(optimizer)
        xm.mark_step()
    else:
        optimizer.step()

def maybe_mark_step():
    if IS_TPU:
        xm.mark_step()

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

# -------------------------------------------------
# Load
# -------------------------------------------------

def local_save(data, local_path):
    if IS_TPU:       xm.save(data, local_path)
    else:            torch.save(data, local_path)

#--------------------------------------------------
# Float32
#--------------------------------------------------
@contextmanager
def fp32():
    if IS_TPU:
        with torch.autocast(enabled=False): # type: ignore
            yield
    else:
        yield