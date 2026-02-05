from .directdiff.direct_model import DirectDiffusionModel
from .latentdiff.latent_model import LatentDiffusionModel
from .llm.llm_model import LLModel

_MODEL_REGISTRY = {
    'direct': DirectDiffusionModel,
    'isab': DirectDiffusionModel,
    'latent': LatentDiffusionModel,
    'llm': LLModel
}

def get_model_class(model_name):
    try:
        return _MODEL_REGISTRY[model_name]
    except KeyError:
        raise ValueError(f"Unknown model '{model_name}'. Available: ", _MODEL_REGISTRY.keys())