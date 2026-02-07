import math
import torch
import torch.nn as nn
import numpy as np
π = math.pi

class GeometryAugment(nn.Module):
    """
    Applies random rigid-body transformations (Rotation + Translation) to a batch of tiles.
    Takes (x, y, angle) -> Returns (x, y, sin, cos).
    """
    def __init__(self,
                 rot_range=np.pi/3,
                 translate_range=0.125
                 ):
        super().__init__()
        self.rot_range = rot_range
        self.translate_range = translate_range

    @torch.no_grad()
    def forward(self, xya):
        B, N, THREE = xya.shape
        device = xya.device
        x = xya[..., 0]
        y = xya[..., 1]
        a = xya[..., 2]

        # Random Rotation
        θ = (torch.rand(B, 1, device=device) * 2 - 1) * self.rot_range
        cos_θ = torch.cos(θ)
        sin_θ = torch.sin(θ)
        x_rot = x * cos_θ - y * sin_θ
        y_rot = x * sin_θ + y * cos_θ
        a_rot = a + θ
        a_rot = ((a_rot + π) % (2*π)) - π

        # Random Translation
        dx = (torch.rand(B, 1, device=device) * 2 - 1) * self.translate_range
        dy = (torch.rand(B, 1, device=device) * 2 - 1) * self.translate_range
        x_rot = x_rot + dx
        y_rot = y_rot + dy

        return torch.stack([x_rot, y_rot, a_rot], dim=-1)