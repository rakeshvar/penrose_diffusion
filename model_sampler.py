import sys
from pathlib import Path

import torch
from model_ddim import DDIMDiffusion, TransformerDenoiser

from hex_base import HexGrid
from pen_base import PenGrid
from hex_svg import save_svg as hex_save_svg
from pen_svg import save_svg as pen_save_svg

#--------------------------------------------
# Argument Parsing
#--------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if len(sys.argv) < 2:
    print(f"Usage: python {sys.argv[0]} checkpoint")
    sys.exit()

checkpoint_path = Path(sys.argv[1])
assert checkpoint_path.exists(), f"Checkpoint {checkpoint_path} not found."

checkpoint = torch.load(checkpoint_path, map_location=device)
config = checkpoint['config']

print("Config:")
for k, v in config.items():
    print(k)
    for kk, vv in v.items():
        print(f"\t{kk}: {vv}")

denoiser_config = config['denoiser']
train_config = config['train']

#--------------------------------------------
# Initialization
#--------------------------------------------
denoiser = TransformerDenoiser(**denoiser_config)
denoiser.to(device)
denoiser.load_state_dict(checkpoint['denoiser_state_dict'])

diffuser = DDIMDiffusion(num_timesteps=1000)

#--------------------------------------------
# Sample
#--------------------------------------------
def sample(label, xyac, sample_fpath):
    print("Generating sample...")
    with torch.no_grad():
        class_labels = torch.tensor([label], device=device)
        samples = diffuser.sample(
            denoiser, batch_size=1, num_tiles=checkpoint['num_tiles'],
            class_labels=class_labels, symmetry=checkpoint['symmetry'], num_steps=50
        )

        samples = samples.cpu().numpy()

        if checkpoint['symmetry'] == 6:
            grid = HexGrid(samples[0], side=checkpoint['side'])
            hex_save_svg(grid, sample_fpath)
        else:
            grid = PenGrid(samples[0], from_np=True, side=checkpoint['side'])
            pen_save_svg(grid, sample_fpath)

    print(f"Saved sample -> {sample_fpath}")

#--------------------------------------------
# Main
#--------------------------------------------
i = 0
while True:
    sample(47, None, f"out/{checkpoint_path.stem}_ep{checkpoint['epoch']}_i{i:02d}.svg")
    input("Press Enter to continue. Ctrl+C to exit...")
    i += 1
