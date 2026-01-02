import numpy as np
import torch
from pathlib import Path

class GPUTensorLoader:
    """
    Loads an .npz dataset entirely into GPU memory and yields batches directly.
    Acts as both a Dataset and a DataLoader.
    """
    def __init__(self, data_path, device, batch_size, shuffle):
        path = Path(data_path)
        self.device = device
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        print(f"Loading {path.name} into CPU RAM...")
        with np.load(path) as data:
            cpu_x = torch.from_numpy(data['x'])         # (N, Points, Features)
            cpu_y = torch.from_numpy(data['y'])

        print(f"Moving {cpu_x.shape[0]} samples to {device}...")
        
        # Move raw data to GPU once
        self.x = cpu_x.to(device).to(torch.float32)
        self.y = cpu_y.to(device).long()
        
        self.n_samples = self.x.shape[0]
        self.num_points = self.x.shape[1]
        self.point_dim = self.x.shape[2]
        
        mem_mb = (self.x.element_size() * self.x.nelement()) / 1e6
        print(f"Dataset ready on {device}. Shape: {self.x.shape} | Mem: ~{mem_mb:.2f} MB")

    def __iter__(self):
        """Yields batches of (x, y)"""
        if self.shuffle:
            indices = torch.randperm(self.n_samples, device=self.device)
        else:
            indices = torch.arange(self.n_samples, device=self.device)
            
        for start_idx in range(0, self.n_samples, self.batch_size):
            # We don't drop the last batch here, but you can if desired
            idx = indices[start_idx : start_idx + self.batch_size]
            
            # Slicing pre-loaded GPU tensors is extremely fast
            yield self.x[idx], self.y[idx]

    def __len__(self):
        """Returns the number of batches per epoch (for tqdm)"""
        return (self.n_samples + self.batch_size - 1) // self.batch_size