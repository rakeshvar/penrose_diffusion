import random
import sys
from pathlib import Path

import torch
from model_ddim import DDIMDiffuser, TransformerDenoiser

from hex_base import HexGrid
from pen_base import PenGrid
from hex_svg import save_svg as hex_save_svg
from pen_svg import save_svg as pen_save_svg
from utils import xysc_to_xyac

#--------------------------------------------
# Argument Parsing
#--------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if len(sys.argv) < 2:
    print(f"Usage: python {sys.argv[0]} checkpoint")
    sys.exit()

cp_path = Path(sys.argv[1])
assert cp_path.exists(), f"Checkpoint {cp_path} not found."

cp = torch.load(cp_path, map_location=device)
config = cp['config']

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
denoiser.load_state_dict(cp['denoiser_state_dict'])
denoiser.eval()

diffuser = DDIMDiffuser(num_timesteps=1000)
diffuser.to(device)

#--------------------------------------------
# Sample
#--------------------------------------------
def sample(label, sample_fpath):
    print("Generating sample...")
    with torch.no_grad():
        class_labels = torch.tensor([label], device=device)
        xysc, colors = diffuser.sample(
            denoiser, batch_size=1, num_tiles=cp['num_tiles'],
            class_labels=class_labels, symmetry=cp['symmetry'], num_steps=50
        )

        samples = xysc_to_xyac(xysc, colors)

        if cp['symmetry'] == 6:
            grid = HexGrid(samples[0], side=cp['side'])
            hex_save_svg(grid, sample_fpath)
        else:
            grid = PenGrid(samples[0], from_np=True, side=cp['side'])
            pen_save_svg(grid, sample_fpath)

    print(f"Saved sample -> {sample_fpath}")

#--------------------------------------------
# Main
#--------------------------------------------
def get_random_class():
    label = random.randint(0, cp['num_classes'] - 1)
    inp = input("Generate Class: " + str(cp['class_lookup'][label]))

    if inp == 'q': 
        sys.exit()

    if inp == '':
        return label, cp['class_lookup'][label]

    try:
        label = int(inp)
        return label, cp['class_lookup'][label]
    
    except ValueError or KeyError:
        cname = inp.lower()
        try:
            label = cp['class_lookup'][cname]
        except KeyError:
            print(f"Could not find class {cname}")
            return get_random_class()

i = 0
while True:
    label, cname = get_random_class()    
    svg_fname = f"samples/{cp_path.stem}_i{i:02d}_{cname}.svg"
    sample(label, svg_fname)
    i += 1
