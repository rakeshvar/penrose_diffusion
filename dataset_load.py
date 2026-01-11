import numpy as np
import torch
from pathlib import Path

class DataLoader:
    """
    Loads an .npz dataset entirely into GPU memory and yields batches directly.
    Acts as both a Dataset and a DataLoader.

    Features:
    - 'drop_last': strictly drops incomplete batches.
    - 'progress_bar': optional internal tqdm wrapper.
    """
    def __init__(self, data_path, device, batch_size, shuffle):
        path = Path(data_path)
        self.device = device
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_path_name = path.name

        print(f"Loading {path.name} into CPU RAM...")
        with np.load(path) as data:
            xya = torch.from_numpy(data['xya'])         # (N, num_tiles, xya)
            colors = torch.from_numpy(data['colors'])   # (N, num_tiles, 1)
            labels = torch.from_numpy(data['labels'])   # (N,)
            self.symmetry = data['symmetry'].item()
            self.side = data['side'].item()

        print(f"Moving {xya.shape[0]} samples to {device}...")

        # Move raw data to GPU once if available and convert from float16 to float32
        self.xya = xya.to(device).to(torch.float32)
        self.colors = colors.to(device).long()
        self.labels = labels.to(device).long()

        self.n_samples = self.xya.shape[0]
        self.num_tiles = self.xya.shape[1]

        # Calculate memory usage for __str__
        self.mem_mb = (self.xya.element_size() * self.xya.nelement() +
                       self.colors.element_size() * self.colors.nelement() +
                       self.labels.element_size() * self.labels.nelement()) / 1e6

    def __iter__(self):
        """Yields batches of (xyac, labels). Drops the last batch if incomplete."""
        if self.shuffle:
            indices = torch.randperm(self.n_samples, device=self.device)
        else:
            indices = torch.arange(self.n_samples, device=self.device)

        num_samples = (self.n_samples // self.batch_size) * self.batch_size

        for start_idx in range(0, num_samples, self.batch_size):
            idx = indices[start_idx : start_idx + self.batch_size]

            yield self.xya[idx], self.colors[idx], self.labels[idx]

    def __len__(self):
        """Returns the number of FULL batches per epoch"""
        return self.n_samples // self.batch_size

    def __str__(self):
        """Pretty-print dataloader status."""
        return (
            f"=== GPUTensorLoader Info ===\n"
            f"  • Source:     {self.data_path_name}\n"
            f"  • Symmetry:   {self.symmetry}\n"
            f"  • Side:       {self.side}\n"
            f"  • Samples:    {self.n_samples}\n"
            f"  • Tiles:      {self.num_tiles}\n"
            f"  • Memory:     {self.mem_mb:.2f} MB\n"
            f"  • Device:     {self.device}\n"
            f"  • Batch Size: {self.batch_size}\n"
            f"  • Batches:    {len(self)} (Full batches)\n"
            f"  • Shuffle:    {self.shuffle}\n"
            f"============================"
        )
