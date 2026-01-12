import numpy as np
import torch

from utils import npz_stats

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
            xysc = torch.from_numpy(data['xysc'])       # (N, num_tiles, xysc)
            colors = torch.from_numpy(data['colors'])   # (N, num_tiles, 1)
            labels = torch.from_numpy(data['labels'])   # (N,)
            self.symmetry = data['symmetry'].item()
            self.side = data['side'].item()

        print(f"Moving {xysc.shape[0]} samples to {device}...")

        # Move raw data to GPU once if available and convert from float16 to float32
        self.xysc = xysc.to(device).to(torch.float32)
        self.colors = colors.to(device).long()
        self.labels = labels.to(device).long()

        self.n_samples = self.xysc.shape[0]
        self.num_tiles = self.xysc.shape[1]

        # Calculate memory usage for __str__
        self.mem_mb = (self.xysc.element_size() * self.xysc.nelement() +
                       self.colors.element_size() * self.colors.nelement() +
                       self.labels.element_size() * self.labels.nelement()) / 1e6

    def __iter__(self):
        """Yields batches of (xysc, labels). Drops the last batch if incomplete."""
        if self.shuffle:
            indices = torch.randperm(self.n_samples, device=self.device)
        else:
            indices = torch.arange(self.n_samples, device=self.device)

        num_samples = (self.n_samples // self.batch_size) * self.batch_size

        for start_idx in range(0, num_samples, self.batch_size):
            idx = indices[start_idx : start_idx + self.batch_size]

            yield self.xysc[idx], self.colors[idx], self.labels[idx]

    def __len__(self):
        """Returns the number of FULL batches per epoch"""
        return self.n_samples // self.batch_size

    def __str__(self):
        """Pretty-print dataloader status."""
        return (
            f"=== DataLoader ===\n"
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
        )

if __name__ == "__main__":
    from pathlib import Path

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for path in Path(".").rglob("*.npz"):
        print("#"* 50, "\nOpening: ", path)
        try:
            print(DataLoader(path, device, 32, True))
            npz_stats(path)
        except KeyError as e:
            print("Invalid data file. ", e)

