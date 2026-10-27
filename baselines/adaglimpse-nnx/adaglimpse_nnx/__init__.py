"""AdaGlimpse's architecture in JAX/Flax NNX, as the CanViT paper's architecture control trains it."""

from adaglimpse_nnx.config import ARCHITECTURE_CONTROL, AdaGlimpseConfig
from adaglimpse_nnx.model import AdaGlimpseForPretraining, AdaGlimpseOutput, GlimpseSet

__all__ = ["ARCHITECTURE_CONTROL", "AdaGlimpseConfig", "AdaGlimpseForPretraining", "AdaGlimpseOutput", "GlimpseSet"]
