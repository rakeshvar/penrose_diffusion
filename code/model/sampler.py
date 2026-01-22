import torch
from pathlib import Path

from code.hex.base import HexGrid
from code.pen.base import PenGrid
from code.hex.svg import save_svg as hex_save_svg
from code.pen.svg import save_svg as pen_save_svg
from code.utils import xysc_to_xyac

#--------------------------------------------
# Reusable Sampling Function
#--------------------------------------------
def save_sample(denoiser, diffuser, device, save_path,
                num_tiles, symmetry, side, label, num_steps=50):
    """
    Generates a sample for a specific class label and saves it as an SVG.
    Can be called from train.py or standalone.
    """

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
            svg = hex_save_svg(grid, save_path)
        else:
            grid = PenGrid(sample_np, from_np=True, side=side)
            svg = pen_save_svg(grid, save_path)
    
    return svg, xysc
