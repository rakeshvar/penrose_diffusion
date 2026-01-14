import sys
import torch
import numpy as np
from pathlib import Path

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

# Project imports
from dataset_load import MyDataset
from model_augment import GeometryAugment
from model_ddim import DDIMDiffuser
from utils import xysc_to_xyac
from hex_base import HexGrid
from pen_base import PenGrid
from hex_svg import save_svg as hex_save_svg
from pen_svg import save_svg as pen_save_svg

def save_grid_svg(xyac, symmetry, side, filename):
    """Helper to save the appropriate grid type based on symmetry."""
    # xyac is expected to be numpy array: [N, 4] -> (x, y, angle, color)
    if symmetry == 6:
        grid = HexGrid(xyac, side=side)
        hex_save_svg(grid, filename)
    else:
        # Assumes Penrose (symmetry 5)
        grid = PenGrid(xyac, from_np=True, side=side)
        pen_save_svg(grid, filename)
    print(f"Saved {filename}", end=" ")

    # --- Added Stats Printing ---
    m = xyac.mean(axis=0)
    s = xyac.std(axis=0)
    # print mean(std) with two decimals in one line
    # like 1.23(.51) 4.56(.78) 7.89(.12) .33(.48)
    print(
        f"{m[0]:6.2f}({s[0]:4.2f}) {m[1]:6.2f}({s[1]:4.2f}) {m[2]:6.2f}({s[2]:4.2f})")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <data.npz>")
        sys.exit(1)

    data_path = Path(sys.argv[1])
    if not data_path.exists():
        print(f"Error: File {data_path} not found.")
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Dataset
    print(f"Loading data from {data_path}...")
    dataset = MyDataset(data_path)
    print(dataset)
    print(dataset.class_lookup)

    # 2. Setup Models
    augmenter = GeometryAugment().to(device)
    diffuser = DDIMDiffuser(num_timesteps=1000).to(device)

    # Output directory
    out_dir = Path("test_output")
    out_dir.mkdir(exist_ok=True)
    print(f"Saving output SVGs to: {out_dir.absolute()}")

    # 3. Loop
    while True:
        idx = np.random.randint(0, len(dataset))
        print(f"\nProcessing Sample Index: {idx}")
        
        # Get Sample (Add Batch Dimension)
        # dataset[i] returns (xya, colors, labels)
        xya_0, colors, label = dataset[idx]
        label = dataset.class_lookup[label.item()]
        
        xya_0 = xya_0.unsqueeze(0).to(device)   # (1, N, 3)
        colors = colors.unsqueeze(0).to(device) # (1, N)
        
        # --- A. Original ---
        # Construct xyac for original (x, y, angle from xya, color from colors)
        # xya_0 is (1, N, 3) -> [x, y, angle]
        xya_np = xya_0.cpu().numpy()[0]
        colors_np = colors.cpu().numpy()[0]
        # Stack to (N, 4) -> x, y, angle, color
        xyac_orig = np.column_stack([xya_np, colors_np])
        
        save_grid_svg(xyac_orig, dataset.symmetry, dataset.side, 
                      out_dir / f"{idx:2d}_{label}_00_originals.svg")

        # --- B. Augment ---
        # GeometryAugment expects (B, N, 3) -> Returns (B, N, 4) [x, y, sin, cos]
        xysc_aug = augmenter(xya_0) 
        
        # Convert back to xyac for visualization
        xyac_aug = xysc_to_xyac(xysc_aug, colors[..., np.newaxis])[0] # Take 0th batch
        
        save_grid_svg(xyac_aug, dataset.symmetry, dataset.side, 
                      out_dir / f"{idx:2d}_{label}_01_augmented.svg")

        # --- C. Diffuse Loop ---
        print("Diffusing...")
        # We iterate t from 0 to 1000 with step 20
        # Note: In DDIM/DDPM, t=0 is data (clean), t=999 is noise.
        # q_sample adds noise corresponding to timestep t.
        
        steps = np.linspace(0, 999, 20).astype(int)
        
        for t_val in steps:
            t_tensor = torch.tensor([t_val], device=device).long()
            
            # q_sample returns (x_t, noise)
            xysc_t, _ = diffuser.q_sample(xysc_aug, t_tensor)
            
            # Convert to xyac for visualization
            xyac_t = xysc_to_xyac(xysc_t, colors[..., np.newaxis])[0]
            
            filename = out_dir / f"{idx:2d}_{label}_02_diff_t{t_val:03d}.svg"
            save_grid_svg(xyac_t, dataset.symmetry, dataset.side, filename)

        print("\nSequence complete.")
        user_input = input("Press Enter to process another random sample (or 'q' to quit): ")
        if user_input.lower() == 'q':
            break

if __name__ == "__main__":
    main()