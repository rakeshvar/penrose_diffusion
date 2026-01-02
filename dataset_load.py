import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

class GPUDataset(Dataset):
    """
    Loads the entire dataset into GPU memory at initialization.
    Requires ~3.6GB VRAM for 308k samples @ float32.
    """
    def __init__(self, npz_path, device='cuda'):
        print(f"Loading {npz_path} into CPU RAM...")
        with np.load(npz_path) as data:
            cpu_x = torch.from_numpy(data['x'])
            cpu_y = torch.from_numpy(data['y'])

        print(f"Casting to float32 and moving to {device}...")
        # X: (N, 768, 4) float32 | Y: (N,) Long
        self.x = cpu_x.to(device).to(torch.float32)
        self.y = cpu_y.to(device).long()
        
        self.n_samples = self.x.shape[0]
        print(f"Dataset ready on {device}. Memory footprint: ~{self.x.element_size() * self.x.nelement() / 1e9:.2f} GB")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def get_dataloader(npz_path, batch_size=32, shuffle=True, device='cuda'):
    full_dataset = GPUDataset(npz_path, device=device)
    return DataLoader(full_dataset, batch_size=64, shuffle=True)
