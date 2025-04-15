"""Base Deep Learning Model Configuration Class."""

from dataclasses import field
from enum import Enum

import jax.numpy as jnp
from dataserious import BaseConfig
from flax import linen as nn


class FloatPrecision(str, Enum):
    """Floating Point Precision."""

    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    BFLOAT16 = "bfloat16"

    @property
    def flax_dtype(self) -> jnp.dtype:
        """Get the Flax Data Type."""
        return getattr(jnp, self.value)


class Activation(str, Enum):
    """Activation Function."""

    SIGMOID = "sigmoid"
    RELU = "relu"
    GELU = "gelu"
    TANH = "tanh"
    SOFTMAX = "softmax"
    LEAKY_RELU = "leaky_relu"
    IDENTITY = "identity"

    @property
    def flax_activation(self):
        """Get the Flax Activation Function."""
        if self.value == "identity":
            return lambda x: x
        return getattr(nn, self.value)


class ModelConfig(BaseConfig):
    """Base Model representing Deep Learning Model Configuration."""

    model: str = field(
        metadata={
            "description": (
                "The LiteralString must match actual class name"
                "implementing the model!"
            ),
            "searchable": True,
        },
    )

    @classmethod
    def get_name_mapping(cls):
        """Get the mapping of model names to the model classes."""
        return {c.model: c for c in cls.get_all_subclasses()}
