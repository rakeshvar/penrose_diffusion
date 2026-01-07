import sys
import yaml
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from ddim import DDIMDiffusion, TransformerDenoiser
from dataset_load import GPUTensorLoader

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
    print(f"Usage: python {sys.argv[0]} <config/checkpoint> [data]")
    sys.exit()

loading_from_checkpoint = sys.argv[1].endswith('.pt')

if not loading_from_checkpoint:
    CONFIG_FILE = 'configs.yaml'
    config_name = sys.argv[1]
    print(f"Loading config group '{config_name}' from {CONFIG_FILE}...")

    with open(CONFIG_FILE, 'r') as f:
        all_configs = yaml.safe_load(f)

    if config_name not in all_configs:
        print("Available configs:")
        for configk, configv in all_configs.items():
            print(configk)
            for k, v in configv.items():
                print(f"\t{k}: {v}")

        raise ValueError(f"Config '{config_name}' not found.")
    else:
        config = all_configs[config_name]

    assert len(sys.argv) > 2, f"Usage: python {sys.argv[0]} {config_name} data.npz"
    data_path_str = sys.argv[2]

    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    checkpoint_path = Path(f"checkpoints/cp{timestamp}.pt")
    print(f"No checkpoint path provided. Will save to: {checkpoint_path}")
    checkpoint = {}

else:
    checkpoint_path = Path(sys.argv[1])
    assert checkpoint_path.exists(), f"Checkpoint {checkpoint_path} not found."

    print(f"Found checkpoint {checkpoint_path}. Resuming...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    if len(sys.argv) > 2:
        data_path_str = sys.argv[2]
    else:
        data_path_str = checkpoint['data_path']

print("Config:")
for k, v in config.items():
    print(k)
    for kk, vv in v.items():
        print(f"\t{kk}: {vv}")

denoiser_config = config['denoiser']
train_config = config['train']

#--------------------------------------------
# Load Data
#--------------------------------------------
print(f"Loading data from {data_path_str}...")
dataloader = GPUTensorLoader(
    Path(data_path_str),
    device,
    batch_size=train_config['batch_size'],
    shuffle=True,
)
print(dataloader)

#--------------------------------------------
# Initialization
#--------------------------------------------
denoiser = TransformerDenoiser(**denoiser_config)
denoiser.to(device)

diffuser = DDIMDiffusion(num_timesteps=1000)
diffuser.to(device)

optimizer = torch.optim.AdamW(denoiser.parameters(), lr=train_config['lr'])

#--------------------------------------------
# Checkpoint Loading
#--------------------------------------------
start_epoch = 0

if loading_from_checkpoint:
    denoiser.load_state_dict(checkpoint['denoiser_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1

#--------------------------------------------
# Training Step
#--------------------------------------------
def train_step(xyac, labels):
    denoiser.train()
    B = xyac.shape[0]

    # Forward pass — Add Noise
    t = torch.randint(0, diffuser.num_timesteps, (B,), device=device).long()
    noise = torch.randn_like(xyac[..., :3])
    xyac_noisy, noise_target = diffuser.q_sample(xyac, t, noise)

    # Predict noise
    noise_pred = denoiser(xyac_noisy, t.float() / diffuser.num_timesteps, labels)
    loss = F.mse_loss(noise_pred, noise_target)

    # Backpropagate
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

#--------------------------------------------
# Training Loop
#--------------------------------------------

total_epochs = start_epoch + train_config['num_epochs']

for epoch in range(start_epoch, total_epochs):
    print(f"\nEpoch {epoch}/{total_epochs}")
    epoch_loss = 0

    for i, (xyac, labels) in enumerate(tqdm(dataloader)):
        epoch_loss += train_step(xyac, labels)

    avg_loss = epoch_loss / len(dataloader)
    print(f"Average Loss: {avg_loss:.4f}")

    #------------
    # CHECKPOINT
    #------------
    checkpoint = {
        'epoch': epoch,
        'denoiser_state_dict': denoiser.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'loss': avg_loss,
        'data_path': data_path_str,
        'symmetry': dataloader.symmetry,
        'num_tiles': dataloader.num_tiles,
        'side': dataloader.side,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint -> {checkpoint_path}")

    #--------
    # Sample
    #--------
    print("Generating sample...")
    with torch.no_grad():
        class_labels = torch.tensor([47], device=device)
        samples = diffuser.sample(
            denoiser, batch_size=1, num_tiles=dataloader.num_tiles,
            class_labels=class_labels, symmetry=dataloader.symmetry, num_steps=50
        )

        samples = samples.cpu().numpy()
        sample_fpath = f"out/{checkpoint_path.stem}_ep{epoch}.svg"
        if dataloader.symmetry == 6:
            grid = HexGrid(samples[0], side=dataloader.side)
            hex_save_svg(grid, sample_fpath)
        else:
            grid = PenGrid(samples[0], from_np=True, side=dataloader.side)
            pen_save_svg(grid, sample_fpath)
