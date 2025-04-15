"""Basic fully connected neural networks."""

import flax.linen as nn
import jax.numpy as jnp

import src.sai.config.models.tabular as cfg
from src.sai.architectures.components import (
    FullyConnected,
)


class FCN(nn.Module):
    """Fully connected neural network."""

    config: cfg.FCNConfig

    def setup(self):
        """Initialize the fully connected neural network."""
        self.fcn = FullyConnected(
            hidden_sizes=self.config.hidden_structure,
            activation=self.config.activation.flax_activation,
            use_bias=self.config.use_bias,
            last_layer_activation=None,
            blockid=None,
        )

    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """Forward pass."""
        return self.fcn(x)