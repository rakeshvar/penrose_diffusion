import sys
import torch
import numpy as np
from pathlib import Path

from code.utils.lossy import lattice_loss

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

# Project imports
from code.data.load import MyDataset
from code.augment import GeometryAugment
from code.models.diffuser import Diffuser
from code.polygons.hex.xya import HexGrid
from code.polygons.pen.xya import PenGrid
from code.polygons.hex.svg import save_svg as hex_save_svg
from code.polygons.pen.svg import save_svg as pen_save_svg
from code.utils.basic import TablePrinter

# Table Printer
tp = TablePrinter(8, 7, 4)

def save_grid_svg(xyac, symmetry, side, filename, t):
    if symmetry == 6:
        grid = HexGrid(xyac, side=side)
        hex_save_svg(grid, filename, print_ok=False)

    else:
        grid = PenGrid(xyac, from_np=True, side=side)
        pen_save_svg(grid, filename, print_ok=False)


    m = xyac.mean(axis=0)
    s = xyac.std(axis=0)
    loss_lattice = lattice_loss(symmetry, torch.from_numpy(xyac).unsqueeze(0), side).item() # add batch dim
    tp.line(t, m[0], s[0], m[1], s[1], m[2], s[2], loss_lattice)


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
    diffuser = Diffuser(2, num_timesteps=1000).to(device)

    # Output directory
    out_dir = Path("library/diffuser_output")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving output SVGs to: {out_dir.absolute()}")
    print("Printing Statistics... x:mean(std) y:mean(std) angle:mean(std)")


    # Samples
    while True:
        idx = np.random.randint(0, len(dataset))
        print(f"\nProcessing Sample Index: {idx}")
        tp.top_line()
        tp.line("t", "x", "(std)", "y", "(std)", "angle", "(std)", "lattice")
        tp.mid_line()

        # Original
        xya_0, colors, label = dataset[idx]
        label = dataset.class_lookup[label.item()]

        xya_0 = xya_0.unsqueeze(0).to(device)   # (1, N, 3)
        colors = colors.unsqueeze(0).to(device) # (1, N)

        xya_np = xya_0.cpu().numpy()[0]
        colors_np = colors.cpu().numpy()[0]
        xyac_orig = np.column_stack([xya_np, colors_np])

        orig_file_name = out_dir / f"{idx:2d}_{label}_00_originals.svg"
        save_grid_svg(xyac_orig, dataset.symmetry, dataset.side,
                      orig_file_name, "orig")

        # Augmented
        xya_aug = augmenter(xya_0)
        xyac_aug = np.column_stack([xya_aug.cpu().numpy()[0], colors_np])

        save_grid_svg(xyac_aug, dataset.symmetry, dataset.side,
                      out_dir / f"{idx:2d}_{label}_01_augmented.svg", "augm")

        # Diffusion
        tp.mid_line()

        steps = np.arange(0, 1001, 50)
        steps[-1] -= 1
        # t=0 is data (clean), t=999 is noise.

        for t_val in steps:
            t_tensor = torch.tensor([t_val], device=device).long()
            xya_t, _ = diffuser.q_sample(xya_aug, t_tensor)

            xyac_t = np.column_stack([xya_t.cpu().numpy()[0], colors_np])
            filename = out_dir / f"{idx:2d}_{label}_02_diff_t{t_val:03d}.svg"
            save_grid_svg(xyac_t, dataset.symmetry, dataset.side, filename, t_val)

        tp.bot_line()
        print("Saved files as ", orig_file_name, end="\n\n")

        user_input = input("Press Enter to process another random sample (or 'q' to quit): ")
        if user_input.lower() == 'q':
            break

if __name__ == "__main__":
    main()