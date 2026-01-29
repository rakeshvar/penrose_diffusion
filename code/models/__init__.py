
from code.models.directdiff.model import DirectDiffusionModel
from code.models.latentdiff.model import LatentDiffusionModel


_MODEL_REGISTRY = {
    'direct': DirectDiffusionModel,
    'latent': LatentDiffusionModel,
}

def get_model_class(model_name):
    try:
        return _MODEL_REGISTRY[model_name]
    except KeyError:
        raise ValueError(f"Unknown model '{model_name}'. Available: ", _MODEL_REGISTRY.keys())