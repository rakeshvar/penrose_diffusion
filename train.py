import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

# Custom modules
from ddim import DDIMDiffusion, TransformerDenoiser
from dataset_load import GPUTensorLoader

#--------------------------------------------
# Argument Parsing
#--------------------------------------------
parser = argparse.ArgumentParser(description="Train DDIM Transformer")
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

#--------------------------------------------
# Load Data
#--------------------------------------------
BATCH_SZ = 32
dataloader = GPUTensorLoader(data_path, device, batch_size=BATCH_SZ, shuffle=True)

#--------------------------------------------
# 3. Hyperparameters & Config
#--------------------------------------------
config = {
    'point_dim': dataloader.point_dim,  # Auto-filled
    'noise_dim': 3,
    'num_classes': 70,
    'class_embed_dim': 256,
    'time_embed_dim': 256,
    'd_model': 512,
    'num_heads': 8,
    'num_layers': 6,
    'dropout': 0.1,
    'max_points': dataloader.num_points, # Auto-filled
    'lr': 1e-4,
    'batch_size': BATCH_SZ,
    'num_timesteps': 1000
}

#--------------------------------------------
# 4. Model Initialization
#--------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = TransformerDenoiser(**{k: v for k, v in config.items() 
                                if k in ['point_dim', 'noise_dim', 'num_classes', 
                                        'class_embed_dim', 'time_embed_dim', 'd_model',
                                        'num_heads', 'num_layers', 'dropout', 'max_points']})
model.to(device)

diffusion = DDIMDiffusion(num_timesteps=config['num_timesteps'])
optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])

#--------------------------------------------
# 5. Resume Logic
#--------------------------------------------
start_epoch = 0

if ckpt_path.exists():
    print(f"Found checkpoint at {ckpt_path}. Loading...")
    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        
        # Validation
        if 'config' in checkpoint:
            saved_pts = checkpoint['config'].get('max_points')
            if saved_pts != config['max_points']:
                raise ValueError(f"Checkpoint max_points ({saved_pts}) != Data max_points ({config['max_points']})")
        
        print(f"Resuming training from epoch {start_epoch}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Starting fresh training instead.")
else:
    print(f"Starting fresh training. Checkpoints will be saved to {ckpt_path}")

#--------------------------------------------
# 6. Training Loop
#--------------------------------------------
def train_step(x, y):
    """
    Training step for diffusion model
    """
    model.train()
    x = x.to(device)
    class_labels = y.to(device)
    B = x.shape[0]
    
    t = torch.randint(0, diffusion.num_timesteps, (B,), device=device).long()
    noise = torch.randn_like(x[..., :3])
    x_noisy, noise_target = diffusion.q_sample(x, t, noise)
    
    noise_pred = model(x_noisy, t.float() / diffusion.num_timesteps, class_labels)
    loss = F.mse_loss(noise_pred, noise_target)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

batches_per_epoch = 10

for epoch in range(start_epoch, 100):
    print(f"\nEpoch {epoch}/100")
    epoch_loss = 0
    steps = 0
    
    for i, (x, y) in tqdm(enumerate(dataloader)):
        if i >= batches_per_epoch:
            break
        loss = train_step(x, y)
        epoch_loss += loss
        steps += 1
    
    avg_loss = epoch_loss / steps if steps > 0 else 0
    print(f"Average Loss: {avg_loss:.4f}")
    
    # --- SAVE CHECKPOINT ---
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'loss': avg_loss
    }
    torch.save(checkpoint, ckpt_path)
    print(f"Saved checkpoint -> {ckpt_path}")
    
    # --- GENERATE SAMPLES ---
    if epoch % 10 == 0:
        print("Generating samples...")
        with torch.no_grad():
            # Use auto-detected config for sample shape
            class_labels = torch.randint(0, 70, (4,), device=device)
            samples = diffusion.sample(
                model, 
                shape=(4, config['max_points'], config['point_dim']),
                class_labels=class_labels,
                num_steps=50
            )
