import argparse
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
parser = argparse.ArgumentParser(description="Train DDIM Transformer")
parser.add_argument('-g', '--config', type=str, help='Config to use from configs.yaml')
parser.add_argument('-d', '--data_path', type=str, default=None, help='Path to the .npz data file')
parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                    help='Path to save/load checkpoint. If None, creates a new timestamped file.')
args = parser.parse_args()

#--------------------------------------------
# Configure
#--------------------------------------------
CONFIG_FILE = 'configs.yaml'
print(f"Loading config group '{args.config}' from {CONFIG_FILE}...")
with open(CONFIG_FILE, 'r') as f:
    all_configs = yaml.safe_load(f)

if args.config not in all_configs:
    print("Available configs:")
    for configk, configv in all_configs.items():
        print(configk)
        for k, v in configv.items():
            print(f"\t{k}: {v}")

    available = list(all_configs.keys())
    # Filter out anchors if they show up as keys (depends on yaml loader version)
    available = [k for k in available if not k.startswith('default')]
    raise ValueError(f"Config '{args.config}' not found. Available configs: {available}")

config = all_configs[args.config]
for k, v in config.items():
    print(k)
    for kk, vv in v.items():
        print(f"\t{kk}: {vv}")

model_config = config['model']
train_config = config['train']
data_config = config['data']

if args.data_path:
    data_path_str = args.data_path
else:
    data_path_str = data_config['path']

if args.checkpoint_path:
    ckpt_path = Path(args.checkpoint_path)
else:
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    ckpt_path = Path(f"ckpt_{timestamp}.pt")
    print(f"No checkpoint path provided. Will save to: {ckpt_path}")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

#--------------------------------------------
# Load Data
#--------------------------------------------
print(f"Loading data from {data_path_str}...")
dataloader = GPUTensorLoader(
    Path(data_path_str),
    device,
    batch_size=train_config['batch_size'],
    shuffle=True,
    use_only=data_config['use_only']
)
print(dataloader)

#--------------------------------------------
# Model Initialization
#--------------------------------------------
model = TransformerDenoiser(**model_config)
model.to(device)

diffuser = DDIMDiffusion(num_timesteps=1000) #config['num_timesteps']
optimizer = torch.optim.AdamW(model.parameters(), lr=train_config['lr'])

#--------------------------------------------
# Checkpoint Loading
#--------------------------------------------
start_epoch = 0

if ckpt_path.exists():
    print(f"Found checkpoint {ckpt_path}. Resuming...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
else:
    print(f"Starting fresh training. Checkpoints will be saved to {ckpt_path}")

#--------------------------------------------
# Training Step
#--------------------------------------------
def train_step(xyac, labels):
    model.train()
    B = xyac.shape[0]

    # Forward pass — Add Noise
    t = torch.randint(0, diffuser.num_timesteps, (B,), device=device).long()
    noise = torch.randn_like(xyac[..., :3])
    xyac_noisy, noise_target = diffuser.q_sample(xyac, t, noise)

    # Predict noise
    noise_pred = model(xyac_noisy, t.float() / diffuser.num_timesteps, labels)
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

    for i, (xyac, labels) in tqdm(enumerate(dataloader)):
        epoch_loss += train_step(xyac, labels)

    avg_loss = epoch_loss / len(dataloader)
    print(f"Average Loss: {avg_loss:.4f}")

    #------------
    # CHECKPOINT
    #------------
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': model_config,
        'loss': avg_loss
    }
    torch.save(checkpoint, ckpt_path)
    print(f"Saved checkpoint -> {ckpt_path}")

    #--------
    # Sample
    #--------
    print("Generating sample...")
    with torch.no_grad():
        class_labels = torch.tensor([47], device=device)
        samples = diffuser.sample(
            model, batch_size=1, num_tokens=dataloader.num_tokens,
            class_labels=class_labels, symmetry=dataloader.symmetry, num_steps=50
        )

        samples = samples.cpu().numpy()
        sample_fpath = f"out/{ckpt_path.stem}_ep{epoch}.svg"
        if dataloader.symmetry == 6:
            grid = HexGrid(samples[0], side=dataloader.side)
            hex_save_svg(grid, sample_fpath)
        else:
            grid = PenGrid(samples[0], from_np=True, side=dataloader.side)
            pen_save_svg(grid, sample_fpath)
