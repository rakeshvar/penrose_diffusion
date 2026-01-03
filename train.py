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
parser.add_argument('-g', '--config', type=str, default='base', 
                    help='Config to use from configs.yaml (e.g., toy, base, large)')
parser.add_argument('-d', '--data_path', type=str, default='data/Data_hex_s768_c110.npz', 
                    help='Path to the .npz data file')
parser.add_argument('-c', '--checkpoint_path', type=str, default=None, 
                    help='Path to save/load checkpoint. If None, creates a new timestamped file.')
args = parser.parse_args()

data_path = Path(args.data_path)
if args.checkpoint_path:
    ckpt_path = Path(args.checkpoint_path)
else:
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    ckpt_path = Path(f"ckpt_{timestamp}.pt")
    print(f"No checkpoint path provided. Will save to: {ckpt_path}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if 'hex' in str(data_path):
    symmetry = 6
elif 'pen' in str(data_path):
    symmetry = 5
else:
    raise ValueError(f"Could not figure out hexagons or pentagons from {data_path}")

#--------------------------------------------
# Load Configuration
#--------------------------------------------
config_file = 'configs.yaml'
print(f"Loading config group '{args.config}' from {config_file}...")
with open(config_file, 'r') as f:
    all_configs = yaml.safe_load(f)

if args.config not in all_configs:
    available = list(all_configs.keys())
    # Filter out anchors if they show up as keys (depends on yaml loader version)
    available = [k for k in available if not k.startswith('default')] 
    raise ValueError(f"Group '{args.config}' not found. Available groups: {available}")

config = all_configs[args.config]

model_config = config['model']
train_config = config['train']
data_path_str = config['data_path']

#--------------------------------------------
# Setup
#--------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if args.checkpoint_path:
    ckpt_path = Path(args.checkpoint_path)
else:
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    ckpt_path = Path(f"ckpt_{args.group}_{timestamp}.pt")
    print(f"Checkpoint will save to: {ckpt_path}")

#--------------------------------------------
# Load Data
#--------------------------------------------
print(f"Loading data from {data_path_str}...")
dataloader = GPUTensorLoader(
    Path(data_path_str), 
    device, 
    batch_size=train_config['batch_size'], 
    shuffle=True
)

#--------------------------------------------
# 4. Model Initialization
#--------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = TransformerDenoiser(**model_config)
model.to(device)

diffuser = DDIMDiffusion(num_timesteps=train_config['num_timesteps'])
optimizer = torch.optim.AdamW(model.parameters(), lr=train_config['lr'])

#--------------------------------------------
# 5. Resume Logic
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
def train_step(x, y):
    """
    Training step for diffusion model
    """
    model.train()
    B = x.shape[0]
    
    # Forward pass — Add Noise
    t = torch.randint(0, diffuser.num_timesteps, (B,), device=device).long()
    noise = torch.randn_like(x[..., :3])
    x_noisy, noise_target = diffuser.q_sample(x, t, noise)
    
    # Predict noise
    noise_pred = model(x_noisy, t.float() / diffuser.num_timesteps, y)
    loss = F.mse_loss(noise_pred, noise_target)
    
    # Backward pass
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
    
    for i, (x, y) in tqdm(enumerate(dataloader)):
        epoch_loss += train_step(x, y)
    
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
    SAMPLE_BATCH_SIZE = 2
    print("Generating samples...")
    with torch.no_grad():
        class_labels = torch.randint(0, 70, (SAMPLE_BATCH_SIZE,), device=device)
        samples = diffuser.sample(
            model, batch_size=SAMPLE_BATCH_SIZE, num_polygons=dataloader.num_points,
            class_labels=class_labels, num_steps=50
        )

        samples = samples.cpu().numpy()
        for i in range(SAMPLE_BATCH_SIZE):
            if symmetry == 6:
                grid = HexGrid(samples[i], side=.01) # TODO: Read from Data
                hex_save_svg(grid, f"out/{ckpt_path.stem}_ep{epoch}_{i}.svg")
            else:
                grid = PenGrid(samples[i], from_np=True, side=.05)
                pen_save_svg(grid, f"out/{ckpt_path.stem}_ep{epoch}_{i}.svg")
