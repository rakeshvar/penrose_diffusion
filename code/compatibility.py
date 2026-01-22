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
    OUTPUT_BASE_DIR = "gs://penrose_diffusion/"

else:
    OUTPUT_BASE_DIR = "."

# -------------------------------------------------
# Device helpers
# -------------------------------------------------

def print_env(rank):
    if rank == 0:
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
def write(text: str, path):
    """
    Write text (e.g., SVG string) to path.
    - If GCS (gs://): Write locally to temp file first, then gsutil cp.
    - If local: Direct write to file.
    No fsspec/gcsfs used at all.
    Heavy debugging prints included.
    Multi-process safe: only master ordinal writes, with rendezvous sync on TPU.
    """
    path_str = str(path)
    print(f"[compat.write] === START WRITE === Path: {path_str}")

    # TPU multi-process sync: all cores wait here before proceeding
    if IS_TPU:
        print("[compat.write] TPU detected. Performing rendezvous sync before write.")
        xm.rendezvous('write_sync_start')

    # Only master ordinal performs the actual write
    if IS_TPU and not xm.is_master_ordinal():
        print(f"[compat.write] Non-master ordinal. Skipping write.")
        return

    print("[compat.write] This process is writing (master ordinal or single-process).")

    if path_str.startswith("gs://"):
        print("[compat.write] GCS path detected. Using local temp + gsutil cp strategy.")

        # Create local temp file
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".svg", mode="w", encoding="utf-8") as tmp:
                local_path = tmp.name
                tmp.write(text)  # Write text directly to temp file
            print(f"[compat.write] Created and wrote to local temp file: {local_path}")
        except Exception as e:
            print(f"[compat.write] ERROR: Failed to create/write temp file: {e}")
            raise

        try:
            # Upload with gsutil
            print(f"[compat.write] Starting gsutil cp: {local_path} -> {path_str}")
            subprocess.check_call(["gsutil", "cp", local_path, path_str])
            print("[compat.write] gsutil cp SUCCESSFUL")

            # Verification
            print("[compat.write] Verifying upload with gsutil ls -l...")
            ls_result = subprocess.run(["gsutil", "ls", "-l", path_str], capture_output=True, text=True)
            if ls_result.returncode == 0:
                print(f"[compat.write] VERIFIED in GCS:\n{ls_result.stdout.strip()}")
            else:
                print(f"[compat.write] WARNING: Verification failed (stderr): {ls_result.stderr.strip()}")

        except subprocess.CalledProcessError as e:
            print(f"[compat.write] ERROR: gsutil command failed: {e}")
            raise
        except Exception as e:
            print(f"[compat.write] ERROR during upload/verification: {e}")
            raise
        finally:
            # Cleanup local temp
            try:
                os.remove(local_path)
                print(f"[compat.write] Cleaned up local temp file: {local_path}")
            except Exception as e:
                print(f"[compat.write] WARNING: Failed to delete temp file {local_path}: {e}")

    else:
        print("[compat.write] Local filesystem path detected. Writing directly.")
        
        try:
            with open(path_str, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"[compat.write] Direct local write SUCCESSFUL to {path_str}")
        except Exception as e:
            print(f"[compat.write] ERROR during direct local write: {e}")
            raise

    # Final sync for TPU
    if IS_TPU:
        print("[compat.write] Master finished. Rendezvous sync end.")
        xm.rendezvous('write_sync_end')

    print(f"[compat.write] === WRITE COMPLETED === Path: {path_str}")
    
    
def save(data, path):
    """
    Save checkpoint 'data' to path.
    - If GCS (gs://): Save locally first (xm.save on TPU for efficiency), then gsutil cp.
    - If local: Direct save (xm.save on TPU, torch.save otherwise).
    No fsspec/gcsfs used at all.
    Heavy debugging prints included.
    Multi-process safe: only master ordinal saves, with rendezvous sync on TPU.
    """
    path_str = str(path)
    print(f"[compat.save] === START SAVE === Path: {path_str}")

    # TPU multi-process sync: all cores wait here before proceeding
    if IS_TPU:
        print("[compat.save] TPU detected. Performing rendezvous sync before save.")
        xm.rendezvous('save_sync_start')

    # Only master ordinal performs the actual save (safe even if called from non-master)
    if IS_TPU and not xm.is_master_ordinal():
        print(f"[compat.save] Non-master ordinal (world_size={xrt.world_size()}). Skipping save.")
        return

    print("[compat.save] This process is saving (master ordinal or single-process).")

    if path_str.startswith("gs://"):
        print("[compat.save] GCS path detected. Using local temp + gsutil cp strategy.")

        # Create local temp file
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
                local_path = tmp.name
            print(f"[compat.save] Created local temp file: {local_path}")
        except Exception as e:
            print(f"[compat.save] ERROR: Failed to create temp file: {e}")
            raise

        # === Local save 
        try:
            print("[compat.save] Starting local save step...")
            if IS_TPU:
                print("[compat.save] TPU: Saving locally with xm.save (efficient, no manual transfer).")
                xm.save(data, local_path)
            else:
                print("[compat.save] Non-TPU: Saving locally with torch.save (NO _cpuify for now).")
                torch.save(data, local_path)  # Direct save without _cpuify as requested
            
            print(f"[compat.save] Local save SUCCESSFUL to {local_path}")
        except Exception as e:
            print(f"[compat.save] ERROR during local save: {e}")
            raise

        try:
            # Upload with gsutil
            print(f"[compat.save] Starting gsutil cp: {local_path} -> {path_str}")
            subprocess.check_call(["gsutil", "cp", local_path, path_str])
            print("[compat.save] gsutil cp SUCCESSFUL")

            # Verification
            print("[compat.save] Verifying upload with gsutil ls -l...")
            ls_result = subprocess.run(["gsutil", "ls", "-l", path_str], capture_output=True, text=True)
            if ls_result.returncode == 0:
                print(f"[compat.save] VERIFIED in GCS:\n{ls_result.stdout.strip()}")
            else:
                print(f"[compat.save] WARNING: Verification failed (stderr): {ls_result.stderr.strip()}")

        except subprocess.CalledProcessError as e:
            print(f"[compat.save] ERROR: gsutil command failed: {e}")
            raise
        except Exception as e:
            print(f"[compat.save] ERROR during upload/verification: {e}")
            raise
        finally:
            # Cleanup local temp
            try:
                os.remove(local_path)
                print(f"[compat.save] Cleaned up local temp file: {local_path}")
            except Exception as e:
                print(f"[compat.save] WARNING: Failed to delete temp file {local_path}: {e}")

    else:
        print("[compat.save] Local filesystem path detected. Saving directly.")
        
        try:
            if IS_TPU:
                print("[compat.save] TPU: Direct xm.save to local path.")
                xm.save(data, path_str)
            else:
                print("[compat.save] Non-TPU: Direct torch.save (NO _cpuify for now).")
                torch.save(data, path_str)  # Direct save without _cpuify
            
            print(f"[compat.save] Direct local save SUCCESSFUL to {path_str}")
        except Exception as e:
            print(f"[compat.save] ERROR during direct local save: {e}")
            raise

    # Final sync for TPU (all cores wait until master finished)
    if IS_TPU:
        print("[compat.save] Master finished. Rendezvous sync end.")
        xm.rendezvous('save_sync_end')

    print(f"[compat.save] === SAVE COMPLETED === Path: {path_str}")