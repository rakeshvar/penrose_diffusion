import fsspec
import shutil
import tempfile
from pathlib import Path

import torch

import code.compatibility as compat

#--------------------------------------------
# Helpers to load/download
#--------------------------------------------
def safe_torch_load(path, map_location="cpu"):
    p = str(path)
    if p.startswith("gs://"):
        with fsspec.open(p, "rb") as f:
            ckpt = torch.load(f, weights_only=False, map_location=map_location) # type: ignore

    else:
        ckpt = torch.load(p, weights_only=False, map_location=map_location)

    return ckpt


def maybe_download(path:str, suffix=""):
    if not path.startswith("gs://"):
        return path

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name

    print(f"Downloading {path} to {tmp_path}")
    with fsspec.open(path, "rb") as src, open(tmp_path, "wb") as dst:
        shutil.copyfileobj(src, dst)

    return tmp_path


#--------------------------------------------
# Loader
#--------------------------------------------
def load_checkpoint(checkpoint_path, model, optimizer, scheduler, mprint):
    if not checkpoint_path:
        return

    mprint(f"Loading weights from {checkpoint_path}...")
    ckpt = safe_torch_load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'], strict=True)

    if optimizer and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    if scheduler and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])

    mprint(f"Resumed weights from Epoch {ckpt['epoch']}.")


#--------------------------------------------
# Checkpoint and SVG Output Manager
#--------------------------------------------
class CheckPointer:
    def __init__(self, folder:str, identifier:str, keep_last_n:int=2):
        self.folder = folder.rstrip("/")
        self.name_format = identifier
        self.keep_last_n = keep_last_n

        self.is_gcs = self.folder.startswith("gs://")
        self.is_local = not self.is_gcs
        self.fixed_ckpt_data = {}
        self.saved_checkpoints = []
        self.saved_svgs = []

        self.ckpt_folder = self.folder + "/checkpoints"
        self.svg_folder = self.folder + "/svg"

        if self.is_gcs:
            from google.cloud import storage
            self.storage_client = storage.Client()

        if self.is_local:
            Path(self.ckpt_folder).mkdir(parents=True, exist_ok=True)
            Path(self.svg_folder).mkdir(parents=True, exist_ok=True)


    def add_fixed_ckpt_data_generic(self, **kwargs):
        self.fixed_ckpt_data.update(kwargs)

    def add_fixed_ckpt_data(self, dataset, config, data_path, wandb_run_id):
        self.fixed_ckpt_data['side']         = dataset.side
        self.fixed_ckpt_data['symmetry']     = dataset.symmetry
        self.fixed_ckpt_data['num_tiles']    = dataset.num_tiles
        self.fixed_ckpt_data['num_classes']  = dataset.num_classes
        self.fixed_ckpt_data['class_lookup'] = dataset.class_lookup

        self.fixed_ckpt_data['config']       = config
        self.fixed_ckpt_data['data_path']    = data_path
        self.fixed_ckpt_data['wandb_run_id'] = wandb_run_id


    def save_checkpoint(self, epoch, model, optimizer, scheduler, loss):
        data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
        }
        data.update(self.fixed_ckpt_data)
        path = self.ckpt_folder + f"/cp{self.name_format}_e{epoch:03d}.pt"

        if self.is_local:
            compat.local_save(data, path)
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
                local_tmp_path = tmp.name
            compat.local_save(data, local_tmp_path)
            self._upload_to_gcs(local_tmp_path, path)

        print(f"Saved Checkpoint: {path}"
              f"   +(Loss: {loss:.4f})")
        self._keep_only_last_n(self.saved_checkpoints, path, loss)


    def save_svg(self, svg, file_name: str):
        path = self.svg_folder + f"/{file_name}"

        if self.is_local:
            Path(path).write_text(svg)

        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".svg", mode="wt", encoding="utf-8") as tmp:
                local_tmp_path = tmp.name
                tmp.write(svg)
            self._upload_to_gcs(local_tmp_path, path)

        print(f"Saved SVG       : {path}")
        self._keep_only_last_n(self.saved_svgs, path)


    def _upload_to_gcs(self, local_tmp_path: str, path: str):
        bucket_name, blob_name = _gcs_bucket_blob(path)
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_tmp_path)
        Path(local_tmp_path).unlink()


    def _keep_only_last_n(self, saved_paths, path: str, loss=None):
        saved_paths.append((path, loss))

        while len(saved_paths) > self.keep_last_n:
            to_remove, removed_loss = saved_paths.pop(0)
            self._delete_resource(to_remove)
            loss_suffix = f"   -(Loss: {removed_loss:.4f})" if removed_loss is not None else ""
            print(f" - Deleted      : {to_remove}{loss_suffix}")

    def _delete_resource(self, path: str):
        if self.is_local:
            Path(path).unlink()
            return

        try:
            bucket_name, blob_name = _gcs_bucket_blob(path)
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            if blob.exists():
                blob.delete()
            else:
                print(f"Warning: {path} does not exist on GCS")

        except Exception as e:
            print(f"Warning: Failed to delete {path}: {e}")


def _gcs_bucket_blob(path: str):
    """Splits 'gs://bucket/path/to/blob' into ('bucket', 'path/to/blob')."""
    assert path.startswith("gs://"), f"Not a GCS path: {path}"

    path_no_prefix = path[5:]
    if "/" not in path_no_prefix:
        bucket_name, blob_name = path_no_prefix, ""   # Root of bucket
    else:
        bucket_name, blob_name = path_no_prefix.split("/", 1)

    return bucket_name, blob_name

