# pyright: reportPossiblyUnboundVariable=false
# pyright: reportMissingImports=false

import os
import sys
import socket
from pathlib import Path

import torch

import fsspec
import tempfile
import shutil
import subprocess


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
    OUTPUT_BASE_DIR = "gs://penrose-diffusion/"

else:
    OUTPUT_BASE_DIR = "."

# -------------------------------------------------
# Device helpers
# -------------------------------------------------

def print_env(rank):
    if rank == 0:
        print("######################")
        print("IS_CPU:  ", IS_CPU)
        print("IS_GPU:  ", IS_GPU)
        print("IS_TPU:  ", IS_TPU)
        print("IS_GCP:  ", IS_GCP)
        print("IS_COLAB:", IS_COLAB)
        print("######################")

def get_device():
    if IS_TPU:
        return xm.xla_device()
    if IS_GPU:
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def master_print(msg, rank):
    if rank == 0:
        print(msg)

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


def load(path, map_location="cpu"):
    p = str(path)
    if p.startswith("gs://"):
        with fsspec.open(p, "rb") as f:
            ckpt = torch.load(f, map_location=map_location) # type: ignore  

    else:
        ckpt = torch.load(p, map_location=map_location)

    return ckpt


def download_to_local(path, suffix=""):
    p = str(path)
    if not p.startswith("gs://"):
        return p

    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tf.close()
    tmp_path = tf.name

    with fsspec.open(p, "rb") as src, open(tmp_path, "wb") as dst:
        shutil.copyfileobj(src, dst) # type: ignore

    return tmp_path

# -------------------------------------------------
# Save
# -------------------------------------------------
def _cpuify(obj):
    """Recursively move torch/xla tensors to CPU so torch.save works on them."""
    if isinstance(obj, torch.Tensor):
        if IS_TPU:
            return xm.to_cpu(obj)
        else:
            return obj.cpu()
    elif isinstance(obj, dict):
        return {k: _cpuify(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_cpuify(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_cpuify(v) for v in obj)
    else:
        return obj

def save(data, path):
    """
    Save checkpoint 'data' to path.
    Supports local paths and gs:// URIs via fsspec/gcsfs.
    Fallback: save to local tmp file and run gsutil cp if fsspec isn't available.
    """
    path_str = str(path)
    data_cpu = _cpuify(data)

    # Remote (GCS) using fsspec (gcsfs)
    if path_str.startswith("gs://"):
        try:
            # fsspec.open will use gcsfs when 'gs://' is detected (gcsfs must be installed)
            with fsspec.open(path_str, "wb") as f:
                # torch.save accepts a file-like object opened in binary write mode
                torch.save(data_cpu, f) # type: ignore
            return
        except Exception as e:
            # helpful debug message (do not crash silently)
            print(f"[compat.save] fsspec/gcsfs write failed: {e}. Falling back to local tmp + gsutil.")

        # Fallback: write locally then gsutil cp
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
            tmp_path = tmp.name
        torch.save(data_cpu, tmp_path)
        try:
            subprocess.check_call(["gsutil", "cp", tmp_path, path_str])
        except Exception as e:
            print(f"[compat.save] fallback gsutil upload failed: {e}")
            raise
        finally:
            try:
                import os
                os.remove(tmp_path)
            except Exception:
                pass

    else:
        # Local save
        if IS_TPU:
            # xm.save tries to behave like torch.save, but it may not accept file-like objects so use path string
            xm.save(data_cpu, path_str)
        else:
            torch.save(data_cpu, path_str)


def write(text: str, path):
    path_str = str(path)    
    with fsspec.open(path_str, "wt", encoding="utf-8") as f:
        f.write(text) # type: ignore