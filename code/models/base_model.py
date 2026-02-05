from abc import ABC, abstractmethod
import torch.nn as nn

class AbstractModel(nn.Module, ABC):
    """Base class for all diffusion models"""

    @property
    @abstractmethod
    def descriptor(self):
        raise NotImplementedError
    
    @abstractmethod
    def train_step(self, *args, **kwargs):
        """Single training step"""
        raise NotImplementedError
    
    @abstractmethod
    def passthrough(self, *args, **kwargs):
        """Passthrough"""
        raise NotImplementedError
    
    @abstractmethod
    def sample(self, *args, **kwargs):
        """Generate samples"""
        raise NotImplementedError