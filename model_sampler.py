import sys
import random
import torch
from pathlib import Path

from model_ddim import DDIMDiffuser, TransformerDenoiser
from hex_base import HexGrid
from pen_base import PenGrid
from hex_svg import save_svg as hex_save_svg
from pen_svg import save_svg as pen_save_svg
from utils import xysc_to_xyac

#--------------------------------------------
# Reusable Sampling Function
#--------------------------------------------
def save_sample(denoiser, diffuser, device, save_path, 
                num_tiles, symmetry, side, label, num_steps=50):
    """
    Generates a sample for a specific class label and saves it as an SVG.
    Can be called from train.py or standalone.
    """
    # Ensure directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        class_labels = torch.tensor([label], device=device)
        
        # Sample from the model
        # Returns sine/cosine coords (xysc) and colors
        xysc, colors = diffuser.sample(
            denoiser, batch_size=1, num_tiles=num_tiles,
            class_labels=class_labels, symmetry=symmetry, num_steps=num_steps
        )

        # Convert to standard coordinates (xyac)
        samples = xysc_to_xyac(xysc, colors)
        sample_np = samples[0] # Take first from batch

        # Create Grid Object and Save SVG
        if symmetry == 6:
            grid = HexGrid(sample_np, side=side)
            hex_save_svg(grid, str(save_path))
        else:
            grid = PenGrid(sample_np, from_np=True, side=side)
            pen_save_svg(grid, str(save_path))

    print(f"Saved sample -> {save_path}")


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
        print(f"Usage: python {sys.argv[0]} checkpoint.pt")
        sys.exit()

    cp_path = Path(sys.argv[1])
    assert cp_path.exists(), f"Checkpoint {cp_path} not found."

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
    # These keys should exist in checkpoints saved by your new io.py
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
    samples_dir = Path("samples")
    samples_dir.mkdir(exist_ok=True)
    
    print("\n--- Interactive Sampler ---")
    print("Press Enter to use random class, type a number/name to select, or 'q' to quit.")
    
    while True:
        label, cname = get_user_class(num_classes, class_lookup)
        
        fname = samples_dir / f"{cp_path.stem}_i{i:02d}_{cname}.svg"
        
        save_sample(
            denoiser=denoiser,
            diffuser=diffuser,
            device=device,
            save_path=fname,
            num_tiles=num_tiles,
            symmetry=symmetry,
            side=side,
            label=label
        )
        i += 1