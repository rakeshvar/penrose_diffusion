from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data_path):
        path = Path(data_path)
        self.data_path_name = path.name

        with np.load(path, allow_pickle=True) as data:
            xya = torch.from_numpy(data['xya'])       if 'xya' in data else None           # (M, N, xya)
            colors = torch.from_numpy(data['colors']) if 'colors' in data else None  # (M, N)
            qr = torch.from_numpy(data['qr'])         if 'qr' in data else None              # (M, N, 2)
            labels = torch.from_numpy(data['labels'])                                # (M,)
            self.symmetry = data['symmetry'].item()
            self.side = data['side'].item()
            self.num_classes = data['num_classes'].item()
            self.class_lookup = data['class_lookup'].item()

        if colors is not None:
            self.colors = colors.long()
        else:
            self.colors = None

        if xya is not None:
            xy_means = xya[..., :2].mean(dim=1, keepdim=True) # (M, 1, 2)
            xya[..., :2] -= xy_means
            self.data = xya.float()

        else:
            assert  qr is not None, "Either xya or qr must be provided."
            self.data = qr.long()

        self.labels = labels.long()
        self.n_samples = len(self.labels)
        self.num_tiles = self.data.shape[1]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if self.colors is not None:
            return self.data[idx], self.colors[idx], self.labels[idx]
        else:
            return self.data[idx], torch.tensor([]), self.labels[idx]

    def __str__(self):
        """Pretty-print dataloader status."""
        return (
            f"=== Dataset ===\n"
            f"  • Source:     {self.data_path_name}\n"
            f"  • Symmetry:   {self.symmetry}\n"
            f"  • Side:       {self.side}\n"
            f"  • Tiles:      {self.num_tiles}\n"
            f"  • Samples:    {len(self)} (20 copies * {self.n_samples//(self.num_classes*20)} per class * {self.num_classes} classes)\n"
        )
