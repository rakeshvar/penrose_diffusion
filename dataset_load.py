import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

class GPUTensorLoader:
    """
    Loads an .npz dataset entirely into GPU memory and yields batches directly.
    Acts as both a Dataset and a DataLoader.
    
    Features:
    - 'use_only': load a subset of data.
    - 'drop_last': strictly drops incomplete batches.
    - 'progress_bar': optional internal tqdm wrapper.
    """
    def __init__(self, data_path, device, batch_size, shuffle, use_only, progress_bar=True):
        path = Path(data_path)
        self.device = device
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.progress_bar = progress_bar
        self.data_path_name = path.name
        
        print(f"Loading {path.name} into CPU RAM...")
        with np.load(path) as data:
            xyac = torch.from_numpy(data['xyac'])         # (N, Points, Features)
            labels = torch.from_numpy(data['labels'])
            self.symmetry = data['symmetry'].item()
            self.side = data['side']

        # Apply use_only filter if specified
        print(f"Use Only: {use_only}")
        if use_only is not None:
            print(f"Subsetting data: using only first {use_only} samples.")
            xyac = xyac[:use_only]
            labels = labels[:use_only]

        print(f"Moving {xyac.shape[0]} samples to {device}...")
        
        # Move raw data to GPU once
        self.xyac = xyac.to(device).to(torch.float32)
        self.labels = labels.to(device).long()
        
        self.n_samples = self.xyac.shape[0]
        self.num_tokens = self.xyac.shape[1]
        self.point_dim = self.xyac.shape[2]
        
        # Calculate memory usage for __str__
        self.mem_mb = (self.xyac.element_size() * self.xyac.nelement() + 
                       self.labels.element_size() * self.labels.nelement()) / 1e6
        
        print(f"Dataset ready on {device}. Shape: {self.xyac.shape}")

    def __iter__(self):
        """Yields batches of (xyac, labels). Drops the last batch if incomplete."""
        if self.shuffle:
            indices = torch.randperm(self.n_samples, device=self.device)
        else:
            indices = torch.arange(self.n_samples, device=self.device)
            
        # Calculate stopping point to drop last batch
        n_full_batches = self.n_samples // self.batch_size
        limit = n_full_batches * self.batch_size
        
        # Define the range of start indices
        batch_starts = range(0, limit, self.batch_size)
        
        # Wrap in tqdm if requested
        if self.progress_bar:
            batch_starts = tqdm(batch_starts, desc=f"Epoch ({self.data_path_name})", total=n_full_batches, unit="batch")
            
        for start_idx in batch_starts:
            idx = indices[start_idx : start_idx + self.batch_size]
            
            # Slicing pre-loaded GPU tensors is extremely fast
            yield self.xyac[idx], self.labels[idx]

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
            f"  • Shape:      (N={self.n_samples}, P={self.num_tokens}, D={self.point_dim})\n"
            f"  • Memory:     ~{self.mem_mb:.2f} MB\n"
            f"  • Device:     {self.device}\n"
            f"  • Batch Size: {self.batch_size}\n"
            f"  • Batches:    {len(self)} (Full batches only)\n"
            f"  • Shuffle:    {self.shuffle}\n"
            f"  • P-Bar:      {self.progress_bar}\n"
            f"============================"
        )
