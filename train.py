import torch
from tqdm import tqdm
from ddim import DDIMDiffusion, TransformerDenoiser
import torch.nn.functional as F

#--------------------------------------------
# Hyperparameters
#--------------------------------------------
batches_per_epoch = 10
BATCH_SZ = 32

config = {
    'point_dim': 4,
    'noise_dim': 3,
    'num_classes': 70,
    'class_embed_dim': 256,
    'time_embed_dim': 256,
    'd_model': 512,
    'num_heads': 8,
    'num_layers': 6,
    'dropout': 0.1,
    'max_points': 2*256,
    'lr': 1e-4,
    'batch_size': BATCH_SZ,
    'num_timesteps': 1000
}

#--------------------------------------------
# Model
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
# Training loop example
#--------------------------------------------
def train_step(x, y):
    """
    Training step for diffusion model
    """
    model.train()
    
    x = x.to(device)  # (B, N, 4)
    class_labels = y.to(device)  # (B,)
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

from dataset_load import get_dataloader
dataloader = get_dataloader('data/Data_hex_s768_c110.npz', batch_size=config['batch_size'], shuffle=False, device=device)

for epoch in range(100):
    print(f"Epoch {epoch}")
    for i, (x, y) in tqdm(enumerate(dataloader)):
        if i > batches_per_epoch:
            break
        loss = train_step(x, y)
    
    # Generate samples
    if epoch % 10 == 0:
        with torch.no_grad():
            # Sample random class labels
            class_labels = torch.randint(0, 70, (4,), device=device)
            samples = diffusion.sample(
                model, 
                shape=(4, config['max_points'], config['point_dim']),
                class_labels=class_labels,
                num_steps=50
            )