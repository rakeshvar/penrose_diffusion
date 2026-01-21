import sys
import argparse
from textwrap import dedent
import yaml
from datetime import datetime
from pathlib import Path

import torch
import code.compatibility as compat
from code.utils import safe_path

def recursive_update(base, update):
    """Recursively updates the base dictionary with values from the update dictionary."""
    for k, v in update.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            recursive_update(base[k], v)
        else:
            base[k] = v
    return base

class Config:
    def __init__(self):
        #-----------------------
        # Setup Argparse
        #-----------------------
        parser = argparse.ArgumentParser(description="Training Argument Parser")

        # Catch-all for your flexible positional args (ckpt, npz, config keys, .conf files)
        parser.add_argument('args', nargs='*', help="Sequence of .pt, .npz, .conf files, or config keys")

        # Flags for overrides
        parser.add_argument('-t', '--train', action='append', help="Override train config (key=value)")
        parser.add_argument('-d', '--denoiser', action='append', help="Override denoiser config (key=value)")
        parser.add_argument('-w', '--wandb', action='append', help="Override wandb config (key=value)")
        parser.add_argument('-o', '--output_base_dir', help="Base directory for checkpoints, samples, etc.", 
                            default=compat.OUTPUT_BASE_DIR)

        # Parse
        self.parsed = parser.parse_args()

        #-----------------------
        # Setup Config
        #-----------------------
        self.library = {}
        self.checkpoint_path = None
        self.resume_epoch = 0
        self.conf_dict = {}

        #-----------------------
        # Identify Checkpoint
        # We need this first to establish the base config
        #-----------------------
        for arg in self.parsed.args:
            if arg.endswith('.pt'):
                self.checkpoint_path = arg
                break

        #-----------------------
        # Load Base Config (from checkpoint or defaults)
        #-----------------------
        with open('configs.yaml', 'r') as f:
            self.library = yaml.safe_load(f)
        self.conf_dict = self.library['default']

        if self.checkpoint_path:
            ckpt = compat.load(self.checkpoint_path, map_location='cpu')
            recursive_update(self.conf_dict, ckpt['config']) 
            
            if 'data_path' in ckpt:
                self.data_path_orig = ckpt['data_path']
            self.resume_epoch = ckpt.get('epoch', -1) + 1
            
            # Preserve wandb run_id for resuming
            if 'wandb_run_id' in ckpt:
                if ckpt['wandb_run_id']:
                    self.conf_dict.setdefault('wandb', {})['run_id'] = ckpt['wandb_run_id']
            
            del ckpt

        # Set defaults so we can safely update them later
        self.train = self.conf_dict.setdefault('train', {})
        self.denoiser = self.conf_dict.setdefault('denoiser', {})
        self.wandb = self.conf_dict.setdefault('wandb', {})

        #-----------------------
        # Process Positional Arguments (Files & Config Groups)
        #-----------------------
        for arg in self.parsed.args:
            # Checkpoint already handled
            if arg == self.checkpoint_path:
                continue
            
            # Data file
            if arg.endswith('.npz'):
                self.data_path_orig = arg

            # Specific config file
            elif arg.endswith(('.conf', '.yaml', '.yml')):
                with open(arg, 'r') as f:
                    recursive_update(self.conf_dict, yaml.safe_load(f))

            # Config Group from configs.yaml (e.g., 'small', 'toy')
            elif '.' not in arg and '=' not in arg:
                if arg in self.library:
                    recursive_update(self.conf_dict, self.library[arg])
                else:
                    print(f"Warning: Config group '{arg}' not found in configs.yaml")

        #-----------------------
        # Apply Flag Overrides (-t, -d, -w)
        #-----------------------
        if self.parsed.train:
            print("Applying Train Overrides:")
            for kv in self.parsed.train:
                self._update_from_kv(kv, self.train)

        if self.parsed.denoiser:
            print("Applying Denoiser Overrides:")
            for kv in self.parsed.denoiser:
                self._update_from_kv(kv, self.denoiser)

        if self.parsed.wandb:
            print("Applying WandB Overrides:")
            for kv in self.parsed.wandb:
                self._update_from_kv(kv, self.wandb)
                
        # Setup Naming Template
        self.timestamp = datetime.now().strftime("%m%d_%H%M")

        # Track saved checkpoints to prevent clutter (Keep Max 2)
        self.saved_checkpoints = []

        #----------------------------
        # Setup Directories
        #----------------------------
        out = self.parsed.output_base_dir
        if isinstance(out, Path):
            out = str(out)
        self.output_base_dir = out  # keep string

        # If remote GCS, create remote paths as strings and don't try to mkdir using pathlib
        if self.output_base_dir.startswith("gs://"):
            self.checkpoints_dir = f"{self.output_base_dir.rstrip('/')}/checkpoints"
            self.samples_dir = f"{self.output_base_dir.rstrip('/')}/samples"
            self.logs_dir = f"{self.output_base_dir.rstrip('/')}/logs"
            print(f"Using remote GCS output base: {self.output_base_dir}")
        else:
            # local filesystem: continue to use Path and mkdir
            self.output_base_dir = Path(self.output_base_dir)
            self.output_base_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoints_dir = self.output_base_dir / 'checkpoints'
            self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
            self.samples_dir = self.output_base_dir / 'samples'
            self.samples_dir.mkdir(parents=True, exist_ok=True)
            self.logs_dir = self.output_base_dir / 'logs'
            self.logs_dir.mkdir(parents=True, exist_ok=True)

        if not self.data_path_orig:
            raise ValueError("Must specify either a checkpoint or data path.\n"
                             f" (e.g., python {sys.argv[0]} checkpoint.pt data.npz)")
        
        self.data_path = compat.download_to_local(self.data_path_orig, suffix='.npz')

    def _update_from_kv(self, kv_str, target_dict):
        """Parses key=value and updates target_dict with type inference."""
        if '=' not in kv_str:
            print(f"Warning: Invalid key-value pair '{kv_str}'. format must be key=value")
            return

        key, val_str = kv_str.split('=', 1)

        # Type Inference
        val = val_str
        if val_str.lower() == 'true':
            val = True
        elif val_str.lower() == 'false':
            val = False
        else:
            try:
                val = int(val_str)
            except ValueError:
                try:
                    val = float(val_str)
                except ValueError:
                    pass # keep as string

        print(f"  Override: {key} = {val} ({type(val).__name__})")
        target_dict[key] = val

    def load_model_state(self, denoiser, optimizer):
        if self.checkpoint_path:
            print(f"Loading weights from {self.checkpoint_path}...")
            ckpt = compat.load(self.checkpoint_path, map_location='cpu')
            denoiser.load_state_dict(ckpt['denoiser_state_dict'], strict=True)
    
            if optimizer and 'optimizer_state_dict' in ckpt:
                # Sometimes we want to discard momentum, etc.
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    
            print(f"Resumed weights from Epoch {ckpt.get('epoch', 0)}.")


    def set_identifier(self, lossname, num_tiles):
        self.identifier = f"{self.timestamp}_{lossname}_t{num_tiles:03d}"
    
    def save_checkpoint(self, epoch, denoiser, optimizer, loss, dataset, wandblog=None):
        ckpt_fname = "cp" + self.identifier + f"_e{epoch:03d}.pt"
        ckpt_fpath = safe_path(self.checkpoints_dir, ckpt_fname)

        checkpoint_data = {
            'epoch': int(epoch),
            'denoiser_state_dict': denoiser.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': self.conf_dict,
            'loss': loss,
            'data_path': str(self.data_path_orig),
            # Metadata for inference/sampling
            'symmetry': dataset.symmetry,
            'num_tiles': dataset.num_tiles,
            'side': dataset.side,
            'num_classes': dataset.num_classes,
            'class_lookup': dataset.class_lookup,
        }
        
        # Save WandB run ID for resuming
        if wandblog:
            run_id = wandblog.get_run_id()
            if run_id:
                checkpoint_data['wandb_run_id'] = run_id

        compat.save(checkpoint_data, ckpt_fpath)
        print(f"Saved checkpoint: {ckpt_fpath}")
        self.saved_checkpoints.append(ckpt_fpath)

        while len(self.saved_checkpoints) > 2:
            to_remove = self.saved_checkpoints.pop(0)
            try:
                if to_remove.exists():
                    to_remove.unlink()
                    print(f"Deleted old checkpoint: {to_remove.name}")
            except (OSError, AttributeError) as e:
                print(f"Error deleting checkpoint {to_remove}: {e}")


    def __str__(self):
        return dedent(f"""
        ================================
                Configuration         
        ================================
        Timestamp:       {self.timestamp}
        Checkpoint Path: {self.checkpoint_path}
        Data Path:       {self.data_path_orig} -> {self.data_path}
        Output Base Dir: {self.output_base_dir}
        Resume Epoch:    {self.resume_epoch}
        ---------------------------------
        {yaml.dump(self.conf_dict, default_flow_style=False, sort_keys=False)}
        =================================
        """)