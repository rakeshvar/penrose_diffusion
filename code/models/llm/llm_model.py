
from code.models.base_model import AbstractModel


class LLModel(AbstractModel):
    def __init__(self, config):
        super().__init__()
        raise NotImplementedError

    def sample(self, *args, **kwargs):
        raise NotImplementedError

    def train_step(self, *args, **kwargs):
        raise NotImplementedError

    def passthrough(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def descriptor(self):
        raise NotImplementedError