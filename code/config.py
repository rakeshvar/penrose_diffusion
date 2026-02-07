import sys
import argparse
from textwrap import dedent
import yaml
from datetime import datetime

import code.compatibility as compat
from code.filesystem import safe_torch_load, maybe_download
from code.utils.basic import infer_type


def deep_merge_dict(target, source):
    """Helper: Merges update into base recursively."""
    for k, v in source.items():
        if isinstance(v, dict) and k in target and isinstance(target[k], dict):
            deep_merge_dict(target[k], v)
        else:
            if k not in target:
                print(f"WARNING: Adding new/unrecognized config key: {k}")
            target[k] = v


VALID_SUBCONFIGS = ['train', 'model', 'wandb']

def update_config(target, source):
    for subconfig in source:
        if subconfig not in VALID_SUBCONFIGS:
            print(f"WARNING: Unknown config section: {subconfig}. Ignoring.")
            continue

        assert isinstance( source[subconfig], dict), f"Sub-Config {subconfig} is not a dictionary. Got {source[subconfig]}."

        if subconfig not in target:
            target[subconfig] = {}

        deep_merge_dict(target[subconfig], source[subconfig])


class Config:
    def __init__(self):
        #-----------------------
        # Setup Argparse
        #-----------------------
        parser = argparse.ArgumentParser(description="Training Argument Parser")
        parser.add_argument('base_config', help="First arguments must be the base config from configs.yaml")

        # Catch-all for your flexible positional args (ckpt, npz, config keys, .conf files)
        parser.add_argument('args', nargs='*', help="Sequence of .pt, .npz, .conf files, or config keys")

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
        # Setup Config
        #-----------------------
        self.library = {}
        self.checkpoint_path = None
        self.resume_epoch = 0
        self.config = {}

        #-----------------------
        # Load Base Config
        #-----------------------
        with open('configs.yaml', 'r') as f:
            self.library = yaml.safe_load(f)
        self.config = self.library[self.parsed.base_config]

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
                    self.config.setdefault('wandb', {})['run_id'] = ckpt['wandb_run_id']

            del ckpt

        #-----------------------
        # Link to sub configs
        # - initialize if they don't exist
        #-----------------------
        self.train = self.config.setdefault('train', {})
        self.model = self.config.setdefault('model', {})
        self.wandb = self.config.setdefault('wandb', {})

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
                if arg in self.library:
                    update_config(self.config, self.library[arg])
                else:
                    raise ValueError(f"Config group '{arg}' not found in configs.yaml. "
                                     f"\nAvailable groups: {sorted(self.library.keys())}")

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