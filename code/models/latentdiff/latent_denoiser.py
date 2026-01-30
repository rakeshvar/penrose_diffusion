import torch.nn as nn

class LatentDenoiser(nn.Module):
    def __init__(self, D, num_classes, T=1000):
        super().__init__()
        self.time_embed = nn.Embedding(T, D)
        self.class_embed = nn.Embedding(num_classes + 1, D) # CFG

        self.net = nn.Sequential(
            nn.Linear(D, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, D),
        )

    def forward(self, z_t, t, cls):
        h = z_t + self.time_embed(t) + self.class_embed(cls)
        return self.net(h)
    
    @property
    def device(self):
        return next(self.parameters()).device

