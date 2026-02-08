import sys
import random
import torch
from pathlib import Path

from code.models import get_model_class
from code.utils.advanced import get_random_colors, xyac_to_svgs
from code.utils.basic import print_config
from code.utils.lossy import lattice_loss


#--------------------------------------------
# Interactive CLI
#--------------------------------------------
def get_user_class(num_classes, class_lookup):
    """Prompts user for a class index or name."""
    rand_label = random.randint(0, num_classes - 1)
    prompt = f"q to quit. \nGenerate Class (default {rand_label} aka '{class_lookup[rand_label]}'): "
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
        num_steps = int(sys.argv[2])
    else:
        num_steps = 1000

    # 2. Load Checkpoint
    print(f"Loading {cp_path}...")
    cp = torch.load(cp_path, map_location=device)
    config = cp['config']
    print_config(config)

    # 3. Initialize Models
    Model = get_model_class(config['model']['model'])
    model = Model(config['model'])
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

    while True:
        sample_label, sample_name = get_user_class(num_classes, class_lookup)
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