import argparse
import sys
import yaml

from datetime import datetime
from textwrap import dedent
from pathlib import Path

import code.compatibility as compat
from .filesystem import safe_torch_load, maybe_download
from .utils.basic import infer_type

SUBCONFIGS = ['train', 'model', 'wandb']

def update_config(target, source):
    for subconfig in source:
        if subconfig not in SUBCONFIGS:
            raise ValueError(f"Unknown subconfig: {subconfig}")
        
        for k, v in source[subconfig].items():
            target[subconfig][k] = v


class Config:
    def __init__(self):
        #-----------------------
        # Setup Argparse
        #-----------------------
        parser = argparse.ArgumentParser(description="Training Argument Parser")

        # Catch-all for your flexible positional args (ckpt, npz, config keys, .conf files)
        parser.add_argument('args', nargs='*', help="Sequence of .pt, .npz, and config keys")

        # Flags for overrides
        parser.add_argument('-t', '--train', action='append', help="Override train config (key=value)")
        parser.add_argument('-m', '--model', action='append', help="Override model config (key=value)")
        parser.add_argument('-w', '--wandb', action='append', help="Override wandb config (key=value)")
        parser.add_argument('-o', '--output_base_dir', help="Base directory for checkpoints, samples, etc.",
                            default=compat.OUTPUT_BASE_DIR)

        # Parse
        print("\n\n" + "#" * 80)
        print("Parsing command line arguments...")
        self.parsed = parser.parse_args()

        #-----------------------
        # Init Config
        #-----------------------
        self.checkpoint_path = None
        self.resume_epoch = 0
        self.config = {}

        # Initialize and link to sub-configs
        self.train = self.config.setdefault('train', {})
        self.model = self.config.setdefault('model', {})
        self.wandb = self.config.setdefault('wandb', {})

        #-----------------------
        # Load All Configs
        #-----------------------
        contents = []
        for f in Path('configs').glob('*.yaml'):
            contents.append(f.read_text())
        self.allconfigs = yaml.safe_load('\n'.join(contents))

        #-----------------------
        # Update from Checkpoint
        #-----------------------
        # Identify checkpoint
        for arg in self.parsed.args:
            if arg.endswith('.pt'):
                self.checkpoint_path = arg
                break

        # Update config and data_path
        if self.checkpoint_path:
            print(f"Loading config from checkpoint: {self.checkpoint_path}")
            ckpt = safe_torch_load(self.checkpoint_path, map_location='cpu')
            update_config(self.config, ckpt['config'])

            if 'data_path' in ckpt:
                self.data_path_orig = ckpt['data_path']
            self.resume_epoch = ckpt.get('epoch', -1) + 1

            # Preserve wandb run_id for resuming
            if 'wandb_run_id' in ckpt:
                if ckpt['wandb_run_id']:
                    self.wandb['run_id'] = ckpt['wandb_run_id']

            del ckpt

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
                    update_config(self.config, yaml.safe_load(f))

            # Config Group from configs.yaml (e.g., 'small', 'toy')
            elif '.' not in arg and '=' not in arg:
                if arg in self.allconfigs:
                    update_config(self.config, self.allconfigs[arg])
                else:
                    avl = sorted(self.allconfigs.keys())
                    avl = [a for a in avl if not a.startswith('_')]
                    raise ValueError(f"Config group '{arg}' not found in configs/*.yaml. "
                                     f"Available groups: {avl}")

        #-----------------------
        # Apply Flag Overrides (-t, -m, -w)
        #-----------------------
        if self.parsed.train:
            for kv in self.parsed.train:
                self._update_from_kv(kv, "train")

        if self.parsed.model:
            for kv in self.parsed.model:
                self._update_from_kv(kv, "model")

        if self.parsed.wandb:
            for kv in self.parsed.wandb:
                self._update_from_kv(kv, "wandb")

        #-----------------------
        # Finalize
        #-----------------------
        self.timestamp = datetime.now().strftime("%m%d_%H%M")
        self.output_base_dir = self.parsed.output_base_dir

        if 'data_path_orig' not in self.__dict__:
            raise ValueError("Must specify either a checkpoint or data path.\n"
                             f" (e.g., python {sys.argv[0]} checkpoint.pt data.npz)")

        self.data_path = maybe_download(self.data_path_orig, suffix='.npz')


    def _update_from_kv(self, kv_str, subconfig):
        """
        Parses key=value and updates target_dict with type inference.
        """

        if '=' not in kv_str:
            raise ValueError(f"Invalid key-value pair '{kv_str} for subconfig '{subconfig}'."
                             "Format must be key=value")

        subconfig_dict = self.config[subconfig]
        key, val = kv_str.split('=', 1)

        if key not in subconfig_dict:
            raise ValueError(f"Key '{key}' not found in config {subconfig}: {subconfig_dict}")

        val = infer_type(val)
        print(f"Overriding config.{subconfig}.{key} = {val} (type = {type(val).__name__})"
              f"\toverrides {subconfig_dict[key]}")
        subconfig_dict[key] = val


    def __str__(self):
        return dedent(f"""
        ================================
                Configuration
        ================================
        Timestamp        : {self.timestamp}
        Checkpoint Path  : {self.checkpoint_path}
        Data Path        : {self.data_path_orig}
                ⤷        : {self.data_path}
        Output Base Dir  : {self.output_base_dir}
        Resume Epoch     : {self.resume_epoch}
        ---------------------------------
        """) + \
        yaml.dump(self.config, default_flow_style=False, sort_keys=False) + \
        "================================="