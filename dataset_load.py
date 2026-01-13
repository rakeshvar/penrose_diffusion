import numpy as np
import torch

from utils import npz_stats
from pathlib import Path

class DataLoader:
    def __init__(self, data_path, device, batch_size, shuffle):
        path = Path(data_path)
        self.device = device
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_path_name = path.name

        print(f"Loading {path.name} into CPU RAM...")
        with np.load(path) as data:
            xya = torch.from_numpy(data['xya'])           # (M, N, xya)
            colors = torch.from_numpy(data['colors'])     # (M, N)
            labels = torch.from_numpy(data['labels'])     # (M,)
            self.symmetry = data['symmetry'].item()
            self.side = data['side'].item()
            self.num_classes = data['num_classes'].item()
            self.class_lookup = data['class_lookup'].item()

        xy_means = xya[..., :2].mean(dim=1, keepdim=True) # (M, 1, 2)
        xya[..., :2] -= xy_means

        print(f"Moving {xya.shape[0]} samples to {device}...")
        self.xya = xya.to(device).float()
        self.colors = colors.to(device).long()
        self.labels = labels.to(device).long()

        self.n_samples = self.xya.shape[0]
        self.num_tiles = self.xya.shape[1]
        self.mem_mb = (self.xya.element_size() * self.xya.nelement() +
                       self.colors.element_size() * self.colors.nelement() +
                       self.labels.element_size() * self.labels.nelement()) / 2**20

    def __iter__(self):
        """Yields batches of (xysc, labels). Drops the last batch if incomplete."""
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for path in Path(".").rglob("*.npz"):
        print("#"* 50, "\nOpening: ", path)
        try:
            print(DataLoader(path, device, 32, True))
            npz_stats(path)
        except KeyError as e:
            print("Invalid data file. ", e)

