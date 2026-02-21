import sys
import random
from collections import namedtuple
import torch
from pathlib import Path

from code.models import get_model_class
from code.utils.advanced import get_random_colors, xyac_to_svgs
from code.utils.basic import print_config
from code.utils.lossy import lattice_loss

Dataset = namedtuple('Dataset', ['side', 'symmetry', 'num_tiles', 'num_classes', 'class_lookup', 'vocab_size', 'canvas_xyac'])


def pretty_print(val, name="", indent=0):
    pad = "  " * indent
    label = f"{pad}{name}: " if name else pad

    if isinstance(val, torch.Tensor):
        print(f"{label}Tensor | shape={list(val.shape)} dtype={val.dtype} "
              f"min={val.float().min().item():.4f} max={val.float().max().item():.4f} mean={val.float().mean().item():.4f}")

    elif isinstance(val, dict):
        keys = list(val.keys())
        print(f"{label}dict | len={len(keys)} | keys: {keys}")
        for k in keys:
            pretty_print(val[k], name=str(k), indent=indent + 1)

    elif isinstance(val, (list, tuple)):
        type_name = type(val).__name__
        print(f"{label}{type_name} | len={len(val)}")
        for i, item in enumerate(val):
            pretty_print(item, name=f"[{i}]", indent=indent + 1)

    else:
        print(f"{label}{type(val).__name__} = {val}")

#--------------------------------------------
# Interactive CLI
#--------------------------------------------
def get_user_class(num_classes, class_lookup, inp=None):
    """Prompts user for a class index or name."""
    rand_label = random.randint(0, num_classes - 1)
    prompt = f"q to quit. \nGenerate Class (default {rand_label} aka '{class_lookup[rand_label]}'): "
    if inp is None:
        inp = input(prompt)

    if inp.lower() == 'q':
        sys.exit()

    if inp == '':
        return rand_label, class_lookup[rand_label]

    # Try parsing as int index
    try:
        label = int(inp)
        return label, class_lookup[label]
    except ValueError:
        pass

    # Try parsing as string name
    cname = inp.lower()
    name_to_idx = {v.lower(): k for k, v in class_lookup.items()}

    if cname in name_to_idx:
        return name_to_idx[cname], class_lookup[name_to_idx[cname]]

    print(f"Could not find class '{cname}'. Using random.")
    return rand_label, class_lookup[rand_label]


if __name__ == "__main__":
    # 1. Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} checkpoint.pt [num_steps=50]")
        sys.exit()

    cp_path = Path(sys.argv[1])
    assert cp_path.exists(), f"Checkpoint {cp_path} not found."

    if len(sys.argv) > 2:
        num_samples = int(sys.argv[2])
    else:
        num_samples = 1400
    num_steps = 1000

    # 2. Load Checkpoint
    print("#" * 20, cp_path.name, "#" * 20)
    cp = torch.load(cp_path, map_location=device)

    config = cp['config']
    model_config = config['model']
    # print_config(config)
    dataset = Dataset(
        side         = cp['side'],
        symmetry     = cp['symmetry'],
        num_tiles    = cp['num_tiles'],
        num_classes  = model_config.pop('num_classes') if 'num_classes' in model_config else cp['num_classes'],
        class_lookup = cp['class_lookup'],
        vocab_size   = model_config['vocab_size'] if 'vocab_size' in model_config else None,
        canvas_xyac  = model_config['canvas_xyac'] if 'canvas_xyac' in model_config else None,
    )
    print(f"Dataset: side={dataset.side} symmetry={dataset.symmetry} num_tiles={dataset.num_tiles} "
          f" num_classes={dataset.num_classes} class_lookup={len(dataset.class_lookup)}"
          f" vocab_size={dataset.vocab_size} canvas_xyac={dataset.canvas_xyac.shape if dataset.canvas_xyac is not None else None}")
    print(model_config)

    # 3. Initialize Models
    Model = get_model_class(config['model']['model'])
    model = Model(config['model'], dataset)
    model.load_state_dict(cp['model_state_dict'])
    model.eval()

    # 4. Extract Metadata
    num_tiles = cp['num_tiles']
    symmetry = cp['symmetry']
    side = cp['side']
    num_classes = cp['num_classes']
    class_lookup = cp['class_lookup']

    # 5. Interactive Loop
    i = 0

    print("\n--- Interactive Sampler ---")
    print("Press Enter to use random class, type a number/name to select, or 'q' to quit.")

    while i < num_samples:
        sample_label, sample_name = get_user_class(num_classes, class_lookup, '' if num_samples == 1 else None)
        sample_label_tr = torch.tensor([sample_label], dtype=torch.long, device=device)
        sample_colors = get_random_colors(symmetry, 1, num_tiles, device)

        fname = f"library/samples/{cp_path.stem}_i{i:02d}_{sample_name}.svg"

        samples = model.sample(sample_colors, sample_label_tr, num_steps)
        svg = xyac_to_svgs(samples, symmetry, side)[0]
        with open(fname, 'w') as fp:
            fp.write(svg)
            print(f"Saved to {fname}")

        loss_lattice = lattice_loss(symmetry, samples, side)
        print(f"Lattice loss: {loss_lattice:.4f}")


        i += 1