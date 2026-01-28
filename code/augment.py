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
                 translate_range=0.125,
                 change_to_sincos=True):
        super().__init__()
        self.rot_range = rot_range
        self.translate_range = translate_range
        self.change_to_sincos = change_to_sincos

    @torch.no_grad()
    def forward(self, xya):
        """
        Args:
            xya: (B, N, 3) Tensor containing [x, y, angle] (can be float16)
        Returns:
            xysc: (B, N, 4) Transformed tensor [x, y, sin, cos]
        """
        B = xya.shape[0]
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

        # Random Translation
        dx = (torch.rand(B, 1, device=device) * 2 - 1) * self.translate_range
        dy = (torch.rand(B, 1, device=device) * 2 - 1) * self.translate_range
        x_rot = x_rot + dx
        y_rot = y_rot + dy

        if self.change_to_sincos:
            s_rot = torch.sin(a_rot)
            c_rot = torch.cos(a_rot)
            xysc = torch.stack([x_rot, y_rot, s_rot, c_rot], dim=-1)
            return xysc
        else:
            a_rot = ((a_rot + π) % (2*π)) - π
            return torch.stack([x_rot, y_rot, a_rot], dim=-1)