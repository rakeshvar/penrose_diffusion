import sys
import argparse
import yaml
from datetime import datetime
from pathlib import Path

import torch
import code.compatibility as compat

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
        # Setup Argparse
        parser = argparse.ArgumentParser(description="Training Argument Parser")

        # Catch-all for your flexible positional args (ckpt, npz, config keys, .conf files)
        parser.add_argument('args', nargs='*', help="Sequence of .pt, .npz, .conf files, or config keys")

        # Flags for overrides
        parser.add_argument('-t', '--train', action='append', help="Override train config (key=value)")
        parser.add_argument('-d', '--denoiser', action='append', help="Override denoiser config (key=value)")
        parser.add_argument('-o', '--output_base_dir', help="Base directory for checkpoints, samples, etc.", 
                            default=compat.OUTPUT_BASE_DIR)

        # Parse
        self.parsed = parser.parse_args()

        self.data_path = None
        self.checkpoint_path = None
        self.resume_epoch = 0
        self.conf = {}

        # Identify Checkpoint (First .pt file found)
        # We need this first to establish the base config
        for arg in self.parsed.args:
            if arg.endswith('.pt'):
                self.checkpoint_path = arg
                break

        # Load Base Config (from checkpoint or defaults)
        with open('configs.yaml', 'r') as f:
            self.library = yaml.safe_load(f)

        if self.checkpoint_path:
            # Load to CPU to avoid VRAM collisions
            ckpt = torch.load(self.checkpoint_path, map_location='cpu')
            self.conf = ckpt['config']
            if 'data_path' in ckpt:
                self.data_path = ckpt['data_path']

            self.resume_epoch = ckpt.get('epoch', -1) + 1
            del ckpt            # users must call load_state separately
        else:
            self.conf = self.library['default']

        # Set defaults so we can safely update them later
        self.train = self.conf.setdefault('train', {})
        self.denoiser = self.conf.setdefault('denoiser', {})

        # Process Positional Arguments (Files & Config Groups)
        for arg in self.parsed.args:
            # Checkpoint already handled
            if arg == self.checkpoint_path:
                continue
            
            # Data file
            if arg.endswith('.npz'):
                self.data_path = arg

            # Specific config file
            elif arg.endswith(('.conf', '.yaml', '.yml')):
                with open(arg, 'r') as f:
                    new_conf = yaml.safe_load(f)
                recursive_update(self.conf, new_conf)

            # Config Group from configs.yaml (e.g., 'small', 'toy')
            elif '.' not in arg and '=' not in arg:
                if arg in self.library:
                    recursive_update(self.conf, self.library[arg])
                else:
                    print(f"Warning: Config group '{arg}' not found in configs.yaml")

        # Apply Flag Overrides (-t and -d)
        if self.parsed.train:
            print("Applying Train Overrides:")
            for kv in self.parsed.train:
                self._update_from_kv(kv, self.train)

        if self.parsed.denoiser:
            print("Applying Denoiser Overrides:")
            for kv in self.parsed.denoiser:
                self._update_from_kv(kv, self.denoiser)

        # Setup Naming Template
        self.timestamp = datetime.now().strftime("%m%d_%H%M")

        # Track saved checkpoints to prevent clutter (Keep Max 2)
        self.saved_checkpoints = []

        # Setup Directories
        self.output_base_dir = Path(self.parsed.output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoints_dir = self.output_base_dir / 'checkpoints'
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving checkpoints to {self.checkpoints_dir}")

        self.samples_dir = self.output_base_dir / 'samples'
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving samples to {self.samples_dir}")

        self.logs_dir = self.output_base_dir / 'logs'
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving logs to {self.logs_dir}")

        if not self.data_path:
            raise ValueError("Must specify either a checkpoint or data path.\n"
                             f" (e.g., python {sys.argv[0]} checkpoint.pt data.npz)")

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

    def load_model_state(self, model, optimizer=None):
        """
        Helper to load state dicts into model/optimizer if a checkpoint was provided.
        """
        if not self.checkpoint_path:
            return

        print(f"Loading weights from {self.checkpoint_path}...")
        ckpt = torch.load(self.checkpoint_path, map_location='cpu')

        model.load_state_dict(ckpt['denoiser_state_dict'])
        if optimizer and 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        print(f"Resumed weights from Epoch {ckpt.get('epoch', 0)}.")

    def save_checkpoint(self, epoch, denoiser, optimizer, loss, dataset, lossname):
        ckpt_fname = f"cp{self.timestamp}_{lossname}_t{dataset.num_tiles:03d}_e{epoch:03d}.pt"
        ckpt_fpath = self.checkpoints_dir / ckpt_fname

        checkpoint_data = {
            'epoch': epoch,
            'denoiser_state_dict': denoiser.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': self.conf,
            'loss': loss,
            'data_path': str(self.data_path),
            # Metadata for inference/sampling
            'symmetry': getattr(dataset, 'symmetry', 0),
            'num_tiles': getattr(dataset, 'num_tiles', 0),
            'side': getattr(dataset, 'side', 1.0),
            'num_classes': getattr(dataset, 'num_classes', 0),
            'class_lookup': getattr(dataset, 'class_lookup', {}),
        }

        compat.save(checkpoint_data, ckpt_fpath)
        print(f"Saved checkpoint: {ckpt_fpath}")
        self.saved_checkpoints.append(ckpt_fpath)

        while len(self.saved_checkpoints) > 2:
            to_remove = self.saved_checkpoints.pop(0)
            try:
                if to_remove.exists():
                    to_remove.unlink()
                    print(f"Deleted old checkpoint: {to_remove.name}")
            except OSError as e:
                print(f"Error deleting checkpoint {to_remove}: {e}")


    def __str__(self):
        s = []
        s.append("\n" + "="*40)
        s.append("        Configuration         ")
        s.append("="*40)
        s.append(f"Timestamp:       {self.timestamp}")
        s.append(f"Checkpoint Path: {self.checkpoint_path}")
        s.append(f"Data Path:       {self.data_path}")
        s.append(f"Resume Epoch:    {self.resume_epoch}")
        s.append("-" * 30)
        s.append(yaml.dump(self.conf, default_flow_style=False, sort_keys=False))
        s.append("="*40 + "\n")
        return "\n".join(s)