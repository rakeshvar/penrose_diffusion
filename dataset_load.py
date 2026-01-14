from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import npz_stats

class MyDataset(Dataset):
    def __init__(self, data_path):
        path = Path(data_path)
        self.data_path_name = path.name

        print(f"Loading {path.name} into CPU RAM...")
        with np.load(path, allow_pickle=True) as data:
            xya = torch.from_numpy(data['xya'])           # (M, N, xya)
            colors = torch.from_numpy(data['colors'])     # (M, N)
            labels = torch.from_numpy(data['labels'])     # (M,)
            self.symmetry = data['symmetry'].item()
            self.side = data['side'].item()
            self.num_classes = data['num_classes'].item()
            self.class_lookup = data['class_lookup'].item()

        xy_means = xya[..., :2].mean(dim=1, keepdim=True) # (M, 1, 2)
        xya[..., :2] -= xy_means

        # Typecast
        self.xya = xya.float()
        self.colors = colors.long()
        self.labels = labels.long()

        self.n_samples = self.xya.shape[0]
        self.num_tiles = self.xya.shape[1]
        self.mem_mb = (self.xya.element_size() * self.xya.nelement() +
                       self.colors.element_size() * self.colors.nelement() +
                       self.labels.element_size() * self.labels.nelement()) / 2**20

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.xya[idx], self.colors[idx], self.labels[idx]

    def __str__(self):
        """Pretty-print dataloader status."""
        return (
            f"=== Dataset ===\n"
            f"  • Source:     {self.data_path_name}\n"
            f"  • Symmetry:   {self.symmetry}\n"
            f"  • Side:       {self.side}\n"
            f"  • Samples:    {self.n_samples}\n"
            f"  • Tiles:      {self.num_tiles}\n"
            f"  • Memory:     {self.mem_mb:.2f} MB\n"
            f"  • Batches:    {len(self)} (Full batches)\n"
        )

if __name__ == "__main__":
    for path in Path(".").rglob("*.npz"):
        print("#"* 50, "\nOpening: ", path)
        try:
            dataset = MyDataset(path)
            print(dataset)
            npz_stats(path)
        except Exception as e:
            print(f"Error opening file {path}:\n\t", e)
