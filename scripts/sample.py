import sys
import random
import torch
from pathlib import Path

from code.model.ddim import DDIMDiffuser, TransformerDenoiser
from code.model.sampler import save_sample

#--------------------------------------------
# Interactive CLI
#--------------------------------------------
def get_user_class(num_classes, class_lookup):
    """Prompts user for a class index or name."""
    rand_label = random.randint(0, num_classes - 1)
    prompt = f"Generate Class (default {rand_label} aka '{class_lookup.get(rand_label, '?')}'): "
    inp = input(prompt)

    if inp.lower() == 'q':
        sys.exit()

    if inp == '':
        return rand_label, class_lookup.get(rand_label, str(rand_label))

    # Try parsing as int index
    try:
        label = int(inp)
        return label, class_lookup.get(label, str(label))
    except ValueError:
        pass

    # Try parsing as string name
    cname = inp.lower()
    # Invert lookup: name -> index
    # Assuming class_lookup is {index: name}
    name_to_idx = {v.lower(): k for k, v in class_lookup.items()}

    if cname in name_to_idx:
        return name_to_idx[cname], class_lookup[name_to_idx[cname]]

    print(f"Could not find class '{cname}'. Using random.")
    return rand_label, class_lookup.get(rand_label, str(rand_label))


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
        num_steps = 50

    # 2. Load Checkpoint
    print(f"Loading {cp_path}...")
    cp = torch.load(cp_path, map_location=device)
    config = cp['config']

    print("Config:")
    for k, v in config.items():
        if isinstance(v, dict):
            print(k)
            for kk, vv in v.items():
                print(f"\t{kk}: {vv}")
        else:
            print(f"{k}: {v}")

    # 3. Initialize Models
    denoiser_config = config['denoiser']
    denoiser = TransformerDenoiser(**denoiser_config).to(device)
    denoiser.load_state_dict(cp['denoiser_state_dict'])
    denoiser.eval()

    diffuser = DDIMDiffuser(num_timesteps=1000).to(device)

    # 4. Extract Metadata
    num_tiles = cp.get('num_tiles')
    symmetry = cp.get('symmetry')
    side = cp.get('side', 1.0)
    num_classes = cp.get('num_classes')
    class_lookup = cp.get('class_lookup', {})

    if num_tiles is None:
        print("Error: Checkpoint missing metadata (num_tiles).")
        sys.exit(1)

    # 5. Interactive Loop
    i = 0

    print("\n--- Interactive Sampler ---")
    print("Press Enter to use random class, type a number/name to select, or 'q' to quit.")

    while True:
        label, cname = get_user_class(num_classes, class_lookup)

        fname = f"library/samples/{cp_path.stem}_i{i:02d}_{cname}.svg"

        save_sample(
            denoiser=denoiser,
            diffuser=diffuser,
            device=device,
            save_path=fname,
            num_tiles=num_tiles,
            symmetry=symmetry,
            side=side,
            label=label,
            num_steps=num_steps
        )
        i += 1