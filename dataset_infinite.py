from data_generator import Generator6
from data_imageset import ImageSet
import numpy as np
import random
import torch
from torch.utils.data import IterableDataset, DataLoader

class GeneratorIterator(IterableDataset):
    def __init__(self, generator_instance):
        self.generator = generator_instance

    def __iter__(self):
        while True:
            sample = self.generator.get_sample()
            x = torch.from_numpy(sample['x']).float()
            y = torch.tensor(sample['y'], dtype=torch.long)
            yield x, y

def worker_init_fn(worker_id):
    """
    Ensures each worker process has a different seed for numpy and random.
    Without this, every worker would generate the exact same sequence of data.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_data_loader(num_tiles, batch_size):
    folder = "MPEG7/gifs"
    imageset = ImageSet(folder)
    generator_instance = Generator6(imageset, num_tiles=num_tiles, target_halfside=5., unit_side=.05)
    dataloader = DataLoader(
            GeneratorIterator(generator_instance),
            batch_size=batch_size,
            num_workers=4,
            worker_init_fn=worker_init_fn,
            prefetch_factor=2,
            pin_memory=True
        )
    return dataloader