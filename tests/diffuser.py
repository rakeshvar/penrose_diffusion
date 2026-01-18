import sys
import torch
import numpy as np
from pathlib import Path

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

# Project imports
from code.data.load import MyDataset
from code.model.augment import GeometryAugment
from code.model.ddim import DDIMDiffuser
from code.utils import xysc_to_xyac
from code.hex.base import HexGrid
from code.pen.base import PenGrid
from code.hex.svg import save_svg as hex_save_svg
from code.pen.svg import save_svg as pen_save_svg

def save_grid_svg(xyac, symmetry, side, filename):
    if symmetry == 6:
        grid = HexGrid(xyac, side=side)
        hex_save_svg(grid, filename)

    else:
        grid = PenGrid(xyac, from_np=True, side=side)
        pen_save_svg(grid, filename)

    print(f"Saved {filename}", end=" ")

    m = xyac.mean(axis=0)
    s = xyac.std(axis=0)
    print(f"{m[0]:6.2f}({s[0]:4.2f}) {m[1]:6.2f}({s[1]:4.2f}) {m[2]:6.2f}({s[2]:4.2f})")


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

    # Load Dataset
    print(f"Loading data from {data_path}...")
    dataset = MyDataset(data_path)
    print(dataset)

    # Setup Models
    augmenter = GeometryAugment().to(device)
    diffuser = DDIMDiffuser(num_timesteps=1000).to(device)

    # Output directory
    out_dir = Path("library/diffuser_output")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving output SVGs to: {out_dir.absolute()}")
    print("Printing Statistics... x:mean(std) y:mean(std) angle:mean(std)")

    # Samples
    while True:
        idx = np.random.randint(0, len(dataset))
        print(f"\nProcessing Sample Index: {idx}")

        # Original
        xya_0, colors, label = dataset[idx]
        label = dataset.class_lookup[label.item()]

        xya_0 = xya_0.unsqueeze(0).to(device)   # (1, N, 3)
        colors = colors.unsqueeze(0).to(device) # (1, N)

        xya_np = xya_0.cpu().numpy()[0]
        colors_np = colors.cpu().numpy()[0]
        xyac_orig = np.column_stack([xya_np, colors_np])

        save_grid_svg(xyac_orig, dataset.symmetry, dataset.side,
                      out_dir / f"{idx:2d}_{label}_00_originals.svg")

        # Augmented
        xysc_aug = augmenter(xya_0)
        xyac_aug = xysc_to_xyac(xysc_aug, colors[..., np.newaxis])[0] 

        save_grid_svg(xyac_aug, dataset.symmetry, dataset.side,
                      out_dir / f"{idx:2d}_{label}_01_augmented.svg")

        # Diffusion
        print("Diffusing...")

        steps = np.linspace(0, 999, 20).astype(int)
        # t=0 is data (clean), t=999 is noise.

        for t_val in steps:
            t_tensor = torch.tensor([t_val], device=device).long()
            xysc_t, _ = diffuser.q_sample(xysc_aug, t_tensor)

            xyac_t = xysc_to_xyac(xysc_t, colors[..., np.newaxis])[0]
            filename = out_dir / f"{idx:2d}_{label}_02_diff_t{t_val:03d}.svg"
            save_grid_svg(xyac_t, dataset.symmetry, dataset.side, filename)

        user_input = input("Press Enter to process another random sample (or 'q' to quit): ")
        if user_input.lower() == 'q':
            break

if __name__ == "__main__":
    main()